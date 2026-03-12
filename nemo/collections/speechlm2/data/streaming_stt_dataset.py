# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.utils.data
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio
from omegaconf import DictConfig

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.parts.alignments import WordAlignment, get_word_alignments_for_batch
from nemo.collections.speechlm2.parts.utils import to_dataclass

AUDIO_TOKEN_IDX = -200
IGNORE_INDEX = -100


@dataclass
class StreamingSTTBatch:
    """
    A batch of data for StreamingSTTModel.

    Attributes:
        audios: (B, T) audio signals.
        audio_lens: (B,) lengths of the audio signals in samples.
        input_tokens: (B, L) input token IDs for the LLM. Audio positions are marked with AUDIO_TOKEN_IDX.
        input_token_lens: (B,) lengths of the input token sequences.
        target_tokens: (B, L) target token IDs for the LLM. Non-trainable positions are IGNORE_INDEX.
        target_token_lens: (B,) lengths of the target token sequences.
        text: list of ground-truth transcription strings.
    """

    audios: torch.Tensor
    audio_lens: torch.Tensor
    input_tokens: torch.Tensor
    input_token_lens: torch.Tensor
    target_tokens: Optional[torch.Tensor] = None
    target_token_lens: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None


@dataclass
class StreamingSTTDataConfig:
    sample_rate: int
    frame_length_in_secs: float
    chunk_size: int
    num_delay_frames: int = 0
    audio_tag: str = "<audio>"
    blank_token: str = "<blank>"
    system_role: str = "system"
    system_prompt: str = "Transcribe the audio into text."
    prompt_field: str = "system_prompt"


def get_llm_messages_for_sample(
    system_role: str,
    system_prompt: str,
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: int,
    audio_duration_secs: float,
    frame_length_in_secs: float,
    alignments: Optional[List[WordAlignment]] = None,
) -> List[dict]:
    """
    Get the LLM messages for a sample, using the alignments to determine the turns for the audio and text.

    The conversation is structured as alternating user (audio chunks) and assistant (transcription or blank) turns.
    A word becomes "ready" at the chunk whose end frame >= word_end_frame + num_delay_frames.

    For example, if the alignments are:
    [
        WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
        WordAlignment(text="World", start_time=0.60, end_time=0.80),
    ]
    And the audio duration is 1s, audio_tag is "<audio>", chunk_size is 2, frame_length_in_secs is 0.08s,
    num_delay_frames is 0, then the messages will be:
    [
        {"role": "system", "content": "Transcribe the audio into text."},
        {"role": "user", "content": "<audio><audio>"},  # frames 0-1, 0~0.16s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 2-3, 0.16~0.32s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 4-5, 0.32~0.48s
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "<audio><audio>"},  # frames 6-7, 0.48~0.64s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 8-9, 0.64~0.80s
        {"role": "assistant", "content": "World"},
        {"role": "user", "content": "<audio><audio>"},  # frames 10-11, 0.80~0.96s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 12-13, 0.96~1.12s
        {"role": "assistant", "content": "<blank>"},
    ]

    Note: the last chunk may extend beyond audio_duration_secs since num_frames is
    ceiled to a multiple of chunk_size. The model must pad the audio accordingly.

    Args:
        system_role: The role of the system.
        system_prompt: The prompt for the system.
        audio_tag: The tag for the audio placeholder.
        blank_token: The token for blank/no-emission.
        chunk_size: The number of frames per chunk.
        num_delay_frames: Number of frames to delay word emission after word end.
        audio_duration_secs: The duration of the audio in seconds.
        frame_length_in_secs: The length of a single frame in seconds.
        alignments: List of WordAlignment objects for the sample.
    """
    num_frames = math.ceil(audio_duration_secs / frame_length_in_secs)
    num_chunks = math.ceil(num_frames / chunk_size) if num_frames > 0 else 0

    messages = [{"role": system_role, "content": system_prompt}]

    if alignments is None:
        alignments = []

    word_idx = 0
    for chunk_i in range(num_chunks):
        chunk_end_frame = (chunk_i + 1) * chunk_size

        # User turn: one audio tag per frame in the chunk
        messages.append({"role": "user", "content": audio_tag * chunk_size})

        # Collect words whose end_time (in frames) + delay <= chunk_end_frame
        ready_words = []
        while word_idx < len(alignments):
            word = alignments[word_idx]
            word_end_frame = round(word.end_time / frame_length_in_secs)
            ready_frame = word_end_frame + num_delay_frames
            if ready_frame <= chunk_end_frame:
                ready_words.append(word.text)
                word_idx += 1
            else:
                break

        # Assistant turn: transcribed words or blank
        if ready_words:
            messages.append({"role": "assistant", "content": " ".join(ready_words)})
        else:
            messages.append({"role": "assistant", "content": blank_token})

    return messages


def get_llm_messages_for_batch(
    system_role: str,
    system_prompt: List[str],
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: int,
    audio_durations_secs: List[float],
    frame_length_in_secs: float,
    alignments: Optional[List[List[WordAlignment]]] = None,
) -> List[List[dict]]:
    """
    Get the LLM messages for a batch of samples.

    Args:
        system_role: The role of the system.
        system_prompt: The list of prompts for each sample in the batch.
        audio_tag: The tag for the audio placeholder.
        blank_token: The token for blank/no-emission.
        chunk_size: The number of frames per chunk.
        num_delay_frames: Number of frames to delay word emission after word end.
        audio_durations_secs: List of audio durations in seconds, one per sample.
        frame_length_in_secs: The length of a single frame in seconds.
        alignments: List of lists of WordAlignment objects for the batch.
    """
    batch_messages = []
    for sample_alignments, duration_secs, prompt in zip(alignments, audio_durations_secs, system_prompt):
        batch_messages.append(
            get_llm_messages_for_sample(
                system_role=system_role,
                system_prompt=prompt,
                audio_tag=audio_tag,
                blank_token=blank_token,
                chunk_size=chunk_size,
                num_delay_frames=num_delay_frames,
                audio_duration_secs=duration_secs,
                frame_length_in_secs=frame_length_in_secs,
                alignments=sample_alignments,
            )
        )
    return batch_messages


def _tokenize_with_assistant_mask(
    messages: List[dict],
    tokenizer,
) -> tuple[list[int], list[int]]:
    """
    Tokenize chat messages and return (input_ids, assistant_mask).

    First tries HF's ``return_assistant_tokens_mask`` (requires ``{% generation %}``
    in the chat template).  If that returns an all-zero mask, falls back to a
    sequential-search strategy: tokenize each assistant turn's content separately
    and locate it in the full token sequence.

    Args:
        messages: list of ``{"role": ..., "content": ...}`` dicts.
        tokenizer: NeMo AutoTokenizer (``tokenizer.tokenizer`` is the HF tokenizer).

    Returns:
        (input_ids, assistant_mask) — both plain Python lists of ints.
    """
    hf_tok = tokenizer.tokenizer

    # --- primary path: use HF's built-in mask ---
    result = hf_tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
    )
    input_ids = list(result["input_ids"])
    assistant_mask = list(result["assistant_masks"])

    if any(assistant_mask):
        return input_ids, assistant_mask

    # --- fallback: locate assistant content tokens by sequential search ---
    # Tokenize each assistant turn's content and find it in the full sequence.
    # Works for ChatML-style templates where special tokens prevent BPE merging
    # across turn boundaries.
    assistant_mask = [0] * len(input_ids)

    assistant_content_ids = []
    for msg in messages:
        if msg["role"] == "assistant":
            content_ids = hf_tok.encode(msg["content"], add_special_tokens=False)
            assistant_content_ids.append(content_ids)

    pos = 0
    for content_ids in assistant_content_ids:
        if not content_ids:
            continue
        clen = len(content_ids)
        for i in range(pos, len(input_ids) - clen + 1):
            if input_ids[i : i + clen] == content_ids:
                assistant_mask[i : i + clen] = [1] * clen
                pos = i + clen
                break

    return input_ids, assistant_mask


def _replace_audio_chunks(
    token_ids: list[int],
    chunk_ids: list[int],
    chunk_size: int,
    mask: list | None = None,
) -> list[int] | tuple[list[int], list]:
    """Replace each occurrence of *chunk_ids* with *chunk_size* copies of ``AUDIO_TOKEN_IDX``.

    This handles multi-token audio tags where BPE merges tokens across adjacent
    tags (e.g., ``<audio><audio>`` tokenizes differently from ``encode("<audio>") * 2``).
    By matching the full chunk at once, we avoid the BPE boundary problem.

    When *mask* is provided it is adjusted in sync: each matched span is replaced
    with *chunk_size* copies of the first element of that span (typically 0 for
    user-turn content).

    Returns:
        new_token_ids            when mask is None
        (new_token_ids, new_mask) when mask is provided
    """
    chunk_len = len(chunk_ids)
    new_ids: list[int] = []
    new_mask: list | None = [] if mask is not None else None
    i = 0
    n = len(token_ids)
    while i < n:
        if token_ids[i : i + chunk_len] == chunk_ids:
            new_ids.extend([AUDIO_TOKEN_IDX] * chunk_size)
            if new_mask is not None:
                new_mask.extend([mask[i]] * chunk_size)
            i += chunk_len
        else:
            new_ids.append(token_ids[i])
            if new_mask is not None:
                new_mask.append(mask[i])
            i += 1
    return (new_ids, new_mask) if mask is not None else new_ids


class StreamingSTTDataset(torch.utils.data.Dataset):
    """
    Dataset for StreamingSTTModel.
    Operates directly on Lhotse Cuts (no NeMoMultimodalConversation wrapper).
    """

    def __init__(self, cfg: DictConfig | dict, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.cfg: StreamingSTTDataConfig = to_dataclass(StreamingSTTDataConfig, cfg)

        # Tokenize the full audio chunk string (audio_tag * chunk_size) to get
        # its token ID sequence.  We must encode the full chunk as a single string
        # because BPE may merge tokens across adjacent audio tags (e.g.,
        # "<audio><audio>" tokenizes differently from encode("<audio>") * 2).
        audio_chunk_str = self.cfg.audio_tag * self.cfg.chunk_size
        self.audio_chunk_ids = self.tokenizer.tokenizer.encode(audio_chunk_str, add_special_tokens=False)

        # blank_token is part of the LLM output vocabulary — it must be a single
        # special token, otherwise loss is dominated by multi-token blanks and
        # generation becomes unreliable.  The model's __init__ should have called
        # tokenizer.add_special_tokens() before passing the tokenizer here.
        blank_ids = self.tokenizer.tokenizer.encode(self.cfg.blank_token, add_special_tokens=False)
        if len(blank_ids) != 1:
            raise ValueError(
                f"blank_token '{self.cfg.blank_token}' tokenizes into {len(blank_ids)} tokens {blank_ids}. "
                f"It must be a single special token. Make sure the model adds it via "
                f"tokenizer.add_special_tokens() before constructing the dataset."
            )
        self.blank_id = blank_ids[0]

    def __getitem__(self, cuts: CutSet) -> StreamingSTTBatch | None:
        try:
            audios, audio_lens, cuts = collate_audio(cuts, fault_tolerant=True)
        except Exception as e:
            logging.warning(f"Error collating audio from cuts: {e}")
            return None
        if len(cuts) == 0:
            return None

        alignments = get_word_alignments_for_batch(cuts)
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        system_prompts = [cut.custom.get(self.cfg.prompt_field, self.cfg.system_prompt) for cut in cuts]

        batch_messages = get_llm_messages_for_batch(
            system_role=self.cfg.system_role,
            system_prompt=system_prompts,
            audio_tag=self.cfg.audio_tag,
            blank_token=self.cfg.blank_token,
            chunk_size=self.cfg.chunk_size,
            num_delay_frames=self.cfg.num_delay_frames,
            audio_durations_secs=audio_durations_secs,
            frame_length_in_secs=self.cfg.frame_length_in_secs,
            alignments=alignments,
        )

        all_input_ids = []
        all_target_ids = []

        for messages in batch_messages:
            # Tokenize and compute assistant content mask.
            input_ids, assistant_mask = _tokenize_with_assistant_mask(messages, self.tokenizer)

            # Replace each audio chunk token sequence with chunk_size AUDIO_TOKEN_IDX markers.
            # We match the full chunk (audio_tag * chunk_size) as a unit because BPE
            # may merge tokens across adjacent audio tags.
            input_ids, assistant_mask = _replace_audio_chunks(
                input_ids, self.audio_chunk_ids, self.cfg.chunk_size, mask=assistant_mask
            )

            # Build targets: next-token prediction with loss only on assistant content.
            # target[i] corresponds to input[i] and holds the token at position i+1.
            # Loss is applied only where assistant_mask[i+1] is True.
            target_ids = input_ids[1:] + [IGNORE_INDEX]
            target_mask = assistant_mask[1:] + [0]
            target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_target_ids.append(torch.tensor(target_ids, dtype=torch.long))

        # Left-pad to uniform length within the batch
        input_tokens = left_collate_vectors(all_input_ids, padding_value=self.tokenizer.pad_id)
        target_tokens = left_collate_vectors(all_target_ids, padding_value=IGNORE_INDEX)
        input_token_lens = torch.tensor([len(ids) for ids in all_input_ids], dtype=torch.long)
        target_token_lens = input_token_lens.clone()

        texts = [" ".join(s.text for s in cut.supervisions) for cut in cuts]

        return StreamingSTTBatch(
            audios=audios,
            audio_lens=audio_lens,
            input_tokens=input_tokens,
            input_token_lens=input_token_lens,
            target_tokens=target_tokens,
            target_token_lens=target_token_lens,
            text=texts,
        )
