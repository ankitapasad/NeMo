# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import random
import re

import torch
import torch.utils.data
import torchaudio
from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.data.lhotse.text_adapters import Formattable
from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.force_align import ForceAligner
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.utils import logging

import inflect
import re

_inflect = inflect.engine()

_COMMA_RE    = re.compile(r"([0-9][0-9,]+[0-9])")
_DECIMAL_RE  = re.compile(r"\b([0-9]+)\.([0-9]+)\b")
_DOLLARS_RE  = re.compile(r"\$([0-9,]+(?:\.[0-9]+)?)")
_ORDINAL_RE  = re.compile(r"\b([0-9]+)(st|nd|rd|th)\b", re.IGNORECASE)
_NUMBER_RE   = re.compile(r"\b[0-9]+\b")

# Roman numerals: only real standalone uppercase numerals
_ROMAN_RE = re.compile(r"(?<![A-Z])[IVXLCDM]{2,}(?![A-Z])")

_ROMAN = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}


def _roman_to_int(s):
    total = 0
    prev = 0
    for c in reversed(s):
        val = _ROMAN[c]
        total += -val if val < prev else val
        prev = val
    return total

def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal(m):
    return f"{_expand_number(int(m.group(1)))} point {_expand_digits(m.group(2))}"


def _expand_digits(s):
    return " ".join(_inflect.number_to_words(int(c)) for c in s)

def _expand_dollars(m):
    raw = m.group(1).replace(",", "")
    parts = raw.split(".")

    dollars = int(parts[0])
    cents = int(parts[1]) if len(parts) > 1 else 0

    out = []
    if dollars:
        out.append(f"{_expand_number(dollars)} {'dollar' if dollars == 1 else 'dollars'}")
    if cents:
        out.append(f"{_expand_number(cents)} {'cent' if cents == 1 else 'cents'}")
    return " ".join(out) if out else "zero dollars"

def _expand_ordinal(m):
    n = int(m.group(1))
    return _inflect.ordinal(_inflect.number_to_words(n))

def _expand_roman(m):
    return _inflect.number_to_words(_roman_to_int(m.group()))

def _expand_number(num):
    return _inflect.number_to_words(num, andword="")

def normalize_numbers(text):
    try:
        text = re.sub(_COMMA_RE, _remove_commas, text)
        text = re.sub(_ROMAN_RE, _expand_roman, text)
        text = re.sub(_DOLLARS_RE, _expand_dollars, text)
        text = re.sub(_DECIMAL_RE, _expand_decimal, text)
        text = re.sub(_ORDINAL_RE, _expand_ordinal, text)
        text = re.sub(_NUMBER_RE, lambda m: _expand_number(int(m.group())), text)
        return text

    except Exception:
        return text   # fallback: return input unchanged

class DuplexS2SDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

        aug_by_swap_role (bool, optional):
            Whether to augment data by swapping user/agent roles. Defaults to False.

        include_turn_metadata (bool, optional):
            Whether to include detailed turn metadata in the output. Defaults to False.

        cfg (dict, optional):
            Configuration dictionary containing dataset-specific settings (e.g., word_align_position).

        model_cfg (dict, optional):
            Model configuration dictionary containing settings like predict_user_text, force_align_user_text,
            and force_align_device.

    Returns:
        A dictionary with the following top-level keys:
            - audio_data: Dictionary containing audio and token data (None if no audio cuts present):
                - sample_id: List of cut IDs [B]
                - source_audio: Tensor of source waveform samples [B, T]
                - source_audio_lens: Tensor of source audio lengths [B]
                - target_audio: Tensor of target waveform samples [B, T]
                - target_audio_lens: Tensor of target audio lengths [B]
                - target_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                    at positions aligned with audio frames
                - target_token_lens: Tensor of target token sequence lengths [B]
                - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                    at positions aligned with audio frames
                - source_token_lens: Tensor of source token sequence lengths [B]
                - source_texts: List of source texts joined from input_roles supervisions [B]
                - target_texts: List of target texts joined from output_roles supervisions [B]
                - all_texts: List of all texts joined from all supervisions [B]
                - target_first_turn_audio: Tensor of first turn target audio [B, T]
                - target_first_turn_audio_lens: Tensor of first turn audio lengths [B]
                - formatter: List of formatter names for each cut [B]
                - aug_by_noise: List of boolean flags for noise augmentation [B]
                - prompt_tokens: (Optional, if system prompts exist) Tensor of prompt text tokens [B, T]
                - prompt_token_lens: (Optional, if system prompts exist) Tensor of prompt token sequence lengths [B]
                - target_turn_texts: (Optional, if include_turn_metadata=True) List of lists of turn dictionaries [B]
                    Each turn dict contains: start_time, duration, role, text
                - source_turn_texts: (Optional, if include_turn_metadata=True) List of lists of turn dictionaries [B]
                    Each turn dict contains: start_time, duration, role, text
                - system_prompt: (Optional, if include_turn_metadata=True) List of system prompts [B]
            - text_data: Dictionary containing text-only data (None if no text cuts present):
                - text_tokens: Tensor of text tokens [B, T]
                - text_token_lens: Tensor of text token sequence lengths [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - For target tokens: BOS tokens mark the beginning of each speech segment.
        - For source tokens: special BOS markers ('^' and regular BOS) are used to distinguish user and agent turns.
        - EOS tokens mark the end of each speech segment (or interruption points).
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is segment-level alignment by default.
        - When force_align_user_text is enabled, user audio segments are
          force-aligned using wav2vec2 to generate word-level timestamps, which are then
          converted to frame-level token positions for more precise alignment.
        - Role swapping augmentation (when enabled) creates additional training examples by swapping
          user and agent roles while adjusting audio channels accordingly.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        aug_by_swap_role: bool = False,
        include_turn_metadata: bool = False,
        cfg: dict = None,
        model_cfg: dict = None,
        force_align_user_text: bool = None,
        early_interruption_prob: float = None,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.aug_by_swap_role = aug_by_swap_role
        self.include_turn_metadata = include_turn_metadata

        self.word_align_position = cfg.get("word_align_position", "left") if cfg is not None else "left"
        self.predict_user_text = model_cfg.get("predict_user_text", False) if model_cfg is not None else False
        # Force alignment settings: use explicit parameter if provided, otherwise fall back to config
        if force_align_user_text is not None:
            self.force_align_user_text = force_align_user_text
        else:
            self.force_align_user_text = (
                model_cfg.get("force_align_user_text", False) if model_cfg is not None else False
            )
        # Default to CPU for force alignment to avoid OOM during training/validation when main model is on GPU
        self.force_align_device = model_cfg.get("force_align_device", "cpu") if model_cfg is not None else "cpu"

        self.fix_eos_placements = model_cfg.get("fix_eos_placements", True) if model_cfg is not None else True

        if early_interruption_prob is not None:
            self.early_interruption_prob = early_interruption_prob
        else:
            self.early_interruption_prob = cfg.get("early_interruption_prob", 0.0) if cfg is not None else 0.0

        self.cfg = cfg
        self.model_cfg = model_cfg

        self.use_numbers_norm = model_cfg.get("use_numbers_norm", True) if model_cfg is not None else True

        # Initialize force aligner only when needed during training
        # This avoids loading the wav2vec2 model during validation
        self.force_aligner = None
        self._force_aligner_initialized = False

        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

        user_bos_token = '^'
        user_eos_token = '$'

        self.user_bos_id = self.tokenizer.text_to_ids(user_bos_token)[0]
        self.user_eos_id = self.tokenizer.text_to_ids(user_eos_token)[0]
        self.agent_bos_id = self.tokenizer.bos
        self.agent_eos_id = self.tokenizer.eos
        self.pad_id = get_pad_id(self.tokenizer)

    def _create_minimal_batch(self) -> dict:
        """Create a minimal valid batch when all cuts are filtered out."""
        # Create minimal tensors with batch size 1
        device = torch.device('cpu')  # Default device

        return {
            "sample_id": ["empty_batch"],
            "source_audio": torch.zeros((1, 1000), dtype=torch.float32),  # 1 second of silence at 16kHz
            "source_audio_lens": torch.tensor([1000], dtype=torch.long),
            "agent_bos_vad": None,
            "target_audio": torch.zeros((1, 22050), dtype=torch.float32),  # 1 second of silence at 22.05kHz
            "target_audio_lens": torch.tensor([22050], dtype=torch.long),
            "target_tokens": torch.full((1, 50), self.pad_id, dtype=torch.long),
            "target_token_lens": torch.tensor([1], dtype=torch.long),
            "source_tokens": torch.full((1, 50), self.pad_id, dtype=torch.long),
            "source_token_lens": torch.tensor([1], dtype=torch.long),
            "source_texts": [""],
            "target_texts": [""],
            "all_texts": [""],
            "target_first_turn_audio": torch.zeros((1, 22050), dtype=torch.float32),
            "target_first_turn_audio_lens": torch.tensor([22050], dtype=torch.long),
            "formatter": ["s2s_duplex"],
        }

    def _has_valid_input_and_target(self, cut: Cut) -> bool:
        has_input_text = any(s.text.strip() for s in cut.supervisions if s.speaker in self.input_roles)
        has_target_audio = hasattr(cut, "target_audio") and cut.target_audio is not None
        return has_input_text and has_target_audio

    def _save_audacity_labels(self, target_tokens, batch_idx, suffix, debug_dir, frame_length, bos_id, eos_id, pad_id):
        """Save tokens as Audacity label track (.txt format).

        To use in Audacity:
        1. Open the corresponding .wav file
        2. File -> Import -> Labels
        3. Select the _labels.txt file
        """
        import os

        os.makedirs(debug_dir, exist_ok=True)

        tokens = target_tokens[batch_idx].cpu().tolist()
        label_path = f"{debug_dir}/batch{batch_idx}_{suffix}_labels.txt"

        with open(label_path, 'w') as f:
            for i, tok in enumerate(tokens):
                if tok == pad_id:
                    continue
                if tok not in [bos_id, eos_id]:
                    continue
                start_time = i * frame_length
                end_time = (i + 1) * frame_length

                if tok == bos_id:
                    label = "BOS"
                elif tok == eos_id:
                    label = "EOS"
                else:
                    label = f"{tok}"

                f.write(f"{start_time:.4f}\t{end_time:.4f}\t{label}\n")

        print(f"Saved Audacity labels: {label_path}")

    def _apply_early_interruption_augmentation(
        self,
        target_tokens: torch.Tensor,
        target_audio: torch.Tensor,
        target_token_lens: torch.Tensor,
        target_audio_lens: torch.Tensor,
        source_tokens: torch.Tensor,
        source_audio: torch.Tensor,
        source_token_lens: torch.Tensor,
        source_audio_lens: torch.Tensor,
        batch_idx: int,
        identifier: str,
    ) -> bool:
        """Simulate early interruption by randomly truncating an agent turn with overlap.

        Creates a realistic interruption scenario where:
        1. User starts interrupting at cutoff_pos (user BOS inserted in source_tokens)
        2. Agent continues speaking for overlap_tokens (640 milliseconds)
        3. Agent stops at cutoff_pos + overlap_tokens (agent EOS placed here)

        This creates an overlap period where both speakers are talking simultaneously.

        Returns:
            bool: True if augmentation was successfully applied, False otherwise.
        """
        bos_id = self.agent_bos_id
        eos_id = self.agent_eos_id
        pad_id = self.pad_id

        # Save 2-channel audio before modification
        if self.model_cfg is not None and self.model_cfg.get("debug", False):
            import os

            import torchaudio

            debug_dir = (
                self.model_cfg.get("debug_dir", "/tmp/ei_debug") if self.model_cfg is not None else "/tmp/ei_debug"
            )
            os.makedirs(debug_dir, exist_ok=True)
            if identifier is not None:
                audio_identity = identifier
            else:
                audio_identity = random.randint(1, 1000)

            # Get actual lengths
            src_len = source_audio_lens[batch_idx].item()
            tgt_len = target_audio_lens[batch_idx].item()
            max_len = max(src_len, tgt_len)

            # Create 2-channel audio: [source, target]
            src_audio_before = source_audio[batch_idx, :max_len].clone().cpu()
            tgt_audio_before = target_audio[batch_idx, :max_len].clone().cpu()
            stereo_before = torch.stack([src_audio_before, tgt_audio_before], dim=0)  # (2, T)

            # Normalize to prevent clipping
            if stereo_before.abs().max() > 0:
                stereo_before = stereo_before / stereo_before.abs().max() * 0.9

            sample_rate = int(self.source_sample_rate)  # Assuming same for both
            torchaudio.save(f"{debug_dir}/batch{batch_idx}_BEFORE_{audio_identity}.wav", stereo_before, sample_rate)
            self._save_audacity_labels(
                target_tokens,
                batch_idx,
                f"BEFORE_target_{audio_identity}",
                debug_dir,
                self.frame_length,
                bos_id,
                eos_id,
                pad_id,
            )
            self._save_audacity_labels(
                source_tokens,
                batch_idx,
                f"BEFORE_source_{audio_identity}",
                debug_dir,
                self.frame_length,
                self.user_bos_id,
                self.user_eos_id,
                pad_id,
            )

        target_seq = target_tokens[batch_idx]

        # Overlap period: 640 milliseconds = 8 tokens (80ms per token)
        overlap_tokens = self.cfg.get("early_interruption_overlap_tokens", 8) if self.cfg is not None else 8

        bos_positions = (target_seq == bos_id).nonzero(as_tuple=True)[0]
        eos_positions = (target_seq == eos_id).nonzero(as_tuple=True)[0]

        if len(bos_positions) == 0 or len(eos_positions) == 0:
            return False

        # Find all complete turns
        turns = []
        for bos_pos in bos_positions:
            matching_eos = eos_positions[eos_positions > bos_pos]
            if len(matching_eos) > 0:
                eos_pos = matching_eos[0]
                turn_tokens = target_seq[bos_pos + 1 : eos_pos]
                non_pad_mask = turn_tokens != pad_id
                all_non_pad_positions = (bos_pos + 1 + non_pad_mask.nonzero(as_tuple=True)[0]).tolist()

                # Filter out positions in the last overlap_tokens before eos to ensure overlap
                # Only keep positions where there's at least overlap_tokens of content remaining
                non_pad_positions = [pos for pos in all_non_pad_positions if (eos_pos - pos) > overlap_tokens]

                if len(non_pad_positions) > 0:
                    turns.append(
                        {'bos_pos': bos_pos.item(), 'eos_pos': eos_pos.item(), 'non_pad_positions': non_pad_positions}
                    )

        if len(turns) == 0:
            return False

        # Randomly select one turn and cutoff position
        selected_turn = random.choice(turns)
        cutoff_pos = random.choice(selected_turn['non_pad_positions'])
        original_eos_pos = selected_turn['eos_pos']

        if self.model_cfg is not None and self.model_cfg.get("debug", False):
            print(f"batch_idx {batch_idx}, selected_turn: {selected_turn}")
            print(f"original_eos_pos: {original_eos_pos}")
            print(f"cutoff_pos: {cutoff_pos}")
            print(
                f"tokens from cutoff_pos to original_eos_pos: {target_tokens[batch_idx][cutoff_pos:original_eos_pos]}"
            )

        # Agent stops at cutoff_pos + overlap_tokens to create overlap period
        new_eos_pos = min(cutoff_pos + overlap_tokens, original_eos_pos)
        frames_to_remove = original_eos_pos - cutoff_pos
        if frames_to_remove <= 0:
            return False

        # Update target_tokens: place eos at new_eos_pos, shift tail, pad at end
        target_tokens[batch_idx, new_eos_pos] = eos_id
        seq_len = target_tokens.shape[1]
        cont_start_pos = original_eos_pos + overlap_tokens
        tail_length = seq_len - (cont_start_pos + 1)
        if tail_length > 0:
            target_tokens[batch_idx, new_eos_pos + 1 : new_eos_pos + 1 + tail_length] = target_tokens[
                batch_idx, cont_start_pos + 1 : cont_start_pos + 1 + tail_length
            ].clone()
        target_tokens[batch_idx, -frames_to_remove:] = pad_id
        target_token_lens[batch_idx] -= frames_to_remove

        # Update source_tokens: shift tail (from cutoff_pos)
        source_seq_len = source_tokens.shape[1]
        source_tail_length = source_seq_len - (original_eos_pos + 1)
        if source_tail_length > 0:
            source_tokens[batch_idx, cutoff_pos + 1 : cutoff_pos + 1 + source_tail_length] = source_tokens[
                batch_idx, original_eos_pos + 1 : original_eos_pos + 1 + source_tail_length
            ].clone()
        source_tokens[batch_idx, -frames_to_remove:] = pad_id
        source_token_lens[batch_idx] -= frames_to_remove

        # Update audio: shift and pad with silence
        old_target_len = target_audio_lens[batch_idx].item()
        old_source_len = source_audio_lens[batch_idx].item()
        if old_target_len != old_source_len:
            logging.warning(f"old_target_len != old_source_len: {old_target_len} != {old_source_len}")
        assert (
            self.target_sample_rate == self.source_sample_rate
        ), "This function assumes target and source sample rates are the same"
        old_conv_audio_len = min(old_target_len, old_source_len)

        new_eos_sample = min(int((new_eos_pos + 1) * self.frame_length * self.target_sample_rate), old_conv_audio_len)
        original_eos_sample = min(
            int((original_eos_pos + 1) * self.frame_length * self.target_sample_rate), old_conv_audio_len
        )
        cont_start_sample = min(
            int((cont_start_pos + 1) * self.frame_length * self.target_sample_rate), old_conv_audio_len
        )
        cutoff_sample = min(int((cutoff_pos + 1) * self.frame_length * self.source_sample_rate), old_conv_audio_len)

        samples_to_remove = original_eos_sample - cutoff_sample

        # Update target audio: shift and pad with silence
        tail_audio_length = old_conv_audio_len - cont_start_sample
        if tail_audio_length > 0:
            target_audio[batch_idx, new_eos_sample : new_eos_sample + tail_audio_length] = target_audio[
                batch_idx, cont_start_sample:old_conv_audio_len
            ].clone()

        if new_eos_sample + tail_audio_length < target_audio.shape[1]:
            target_audio[
                batch_idx, new_eos_sample + tail_audio_length : new_eos_sample + tail_audio_length + samples_to_remove
            ] = 0
        target_audio_lens[batch_idx] = old_conv_audio_len - samples_to_remove

        # Update source_audio: shift and pad with silence
        source_tail_audio_length = old_conv_audio_len - original_eos_sample
        if source_tail_audio_length > 0:
            source_audio[batch_idx, cutoff_sample : cutoff_sample + source_tail_audio_length] = source_audio[
                batch_idx, original_eos_sample:old_conv_audio_len
            ].clone()

        if cutoff_sample + source_tail_audio_length < source_audio.shape[1]:
            source_audio[
                batch_idx,
                cutoff_sample
                + source_tail_audio_length : cutoff_sample
                + source_tail_audio_length
                + samples_to_remove,
            ] = 0
        source_audio_lens[batch_idx] = old_conv_audio_len - samples_to_remove

        # Save 2-channel audio and target_tokens after modification
        if self.model_cfg is not None and self.model_cfg.get("debug", False):
            src_audio_after = source_audio[batch_idx, : source_audio_lens[batch_idx].item()].clone().cpu()
            tgt_audio_after = target_audio[batch_idx, : target_audio_lens[batch_idx].item()].clone().cpu()
            stereo_after = torch.stack([src_audio_after, tgt_audio_after], dim=0)  # (2, T)

            if stereo_after.abs().max() > 0:
                stereo_after = stereo_after / stereo_after.abs().max() * 0.9

            torchaudio.save(f"{debug_dir}/batch{batch_idx}_AFTER_{audio_identity}.wav", stereo_after, sample_rate)
            print(f"Saved debug audio to {debug_dir}/batch{batch_idx}_*.wav")
            self._save_audacity_labels(
                target_tokens[: target_token_lens[batch_idx].item()],
                batch_idx,
                f"AFTER_target_{audio_identity}",
                debug_dir,
                self.frame_length,
                bos_id,
                eos_id,
                pad_id,
            )
        return True

    def __getitem__(self, all_cuts: CutSet) -> dict:
        # audio mini-batch
        cuts = all_cuts.filter(lambda c: isinstance(c, Cut))
        audio_data = None
        early_interruption_stats = None

        if cuts and hasattr(cuts[0], 'formatter') and cuts[0].formatter == 'nemo_tarred_to_duplex':
            filtered_cuts = []
            skipped_cuts = []
            for cut in cuts:
                if self._has_valid_input_and_target(cut):
                    filtered_cuts.append(cut)
                else:
                    skipped_cuts.append(cut.id)
            if skipped_cuts:
                logging.info(
                    f"Skipped {len(skipped_cuts)} cuts with empty input text. Skipped cut ids: {', '.join(skipped_cuts)}"
                )
            if not filtered_cuts:
                logging.warning(
                    f"All cuts were filtered out! Original batch size: {len(cuts)}. Returning minimal valid batch to continue training."
                )
                return self._create_minimal_batch()
            cuts = CutSet.from_cuts(filtered_cuts)

        if cuts:
            swapped_cuts = []

            if self.aug_by_swap_role:
                for cut in cuts:
                    total_turns = cut.custom.get('total_turns', len(cut.supervisions))

                    if total_turns > 4 and total_turns % 2 == 0:
                        swapped_cut = self._create_role_swapped_cut(cut)
                        if swapped_cut:
                            swapped_cuts.append(swapped_cut)

            if swapped_cuts:
                all_cuts_combined = CutSet.from_cuts(list(cuts) + swapped_cuts)
            else:
                all_cuts_combined = cuts

            prompt_tokens, prompt_token_lens = collate_system_prompt(all_cuts_combined, self.tokenizer, self.pad_id)
            source_audio, source_audio_lens = collate_audio(all_cuts_combined.resample(self.source_sample_rate))
            target_audio, target_audio_lens = collate_audio(
                all_cuts_combined.resample(self.target_sample_rate), recording_field="target_audio"
            )

            target_tokens, target_token_lens, ei_flags = collate_token_channel(
                all_cuts_combined,
                self.tokenizer,
                self.pad_id,
                self.frame_length,
                roles=self.output_roles,
                bos_id=self.agent_bos_id,
                eos_id=self.agent_eos_id,
                remove_timestamps=True,
                skip_eos=self.fix_eos_placements,
                early_interruption_flag_from_cfg=self.early_interruption_prob > 0,
                use_numbers_norm=self.use_numbers_norm,
            )

            # Run force alignment if enabled
            # NOTE: For validation, a separate dataset instance is created with force_align_user_text=False
            if self.force_align_user_text:
                # Only create ForceAligner when first needed
                if not self._force_aligner_initialized:
                    logging.info(f"Initializing ForceAligner on device {self.force_align_device}")
                    self.force_aligner = ForceAligner(device=self.force_align_device, frame_length=self.frame_length)
                    self._force_aligner_initialized = True
                logging.info(
                    f"Force aligning user text for {len(all_cuts_combined)} cuts on device {self.force_align_device}"
                )
                all_cuts_combined = self.force_aligner.batch_force_align_user_audio(
                    all_cuts_combined, source_sample_rate=self.source_sample_rate
                )

                # Check if we have any cuts left after filtering
                if len(all_cuts_combined) == 0:
                    logging.warning(
                        "All cuts filtered out due to force alignment failures, returning minimal valid batch to continue training."
                    )
                    return self._create_minimal_batch()

            source_tokens, source_token_lens, _ = collate_token_channel(
                all_cuts_combined,
                self.tokenizer,
                self.pad_id,
                self.frame_length,
                roles=self.input_roles,
                bos_id=self.user_bos_id,
                eos_id=self.user_eos_id,
                word_align_position=self.word_align_position,
                remove_timestamps=not self.predict_user_text,
                agent_token_channel=target_tokens if self.fix_eos_placements else None,
                agent_token_channel_lengths=target_token_lens if self.fix_eos_placements else None,
                agent_eos_id=self.agent_eos_id if self.fix_eos_placements else None,
            )

            # Early interruption augmentation
            batch_early_interruption_total = 0
            batch_early_interruption_attempted = 0
            batch_early_interruption_successful = 0
            if self.early_interruption_prob > 0 and torch.is_grad_enabled():
                for batch_idx in range(target_tokens.shape[0]):
                    batch_early_interruption_total += 1
                    if ei_flags[batch_idx]:
                        if random.random() < self.early_interruption_prob:
                            batch_early_interruption_attempted += 1
                            if self.model_cfg is not None and self.model_cfg.get("debug", False):
                                identifier_dirname = getattr(all_cuts_combined[batch_idx], 'shard_origin', None)
                                identifier_cutname = all_cuts_combined[batch_idx].id
                                identifier = f"{identifier_dirname}_{identifier_cutname}"
                            else:
                                identifier = None
                            success = self._apply_early_interruption_augmentation(
                                target_tokens,
                                target_audio,
                                target_token_lens,
                                target_audio_lens,
                                source_tokens,
                                source_audio,
                                source_token_lens,
                                source_audio_lens,
                                batch_idx,
                                identifier,
                            )
                            if success:
                                batch_early_interruption_successful += 1

            try:
                target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
                    all_cuts_combined.resample(self.target_sample_rate),
                    roles=self.output_roles,
                    recording_field="target_audio",
                )
            except Exception as e:
                target_first_turn_audio = None
                target_first_turn_audio_lens = None

            audio_data = {
                "sample_id": [str(cut.id) for cut in all_cuts_combined],
                "source_audio": source_audio,
                "source_audio_lens": source_audio_lens,
                "target_audio": target_audio,
                "target_audio_lens": target_audio_lens,
                "target_tokens": target_tokens,
                "target_token_lens": target_token_lens,
                "source_tokens": source_tokens,
                "source_token_lens": source_token_lens,
                "source_texts": [
                    " ".join(_strip_timestamps(s.text) for s in cut.supervisions if s.speaker in self.input_roles)
                    for cut in all_cuts_combined
                ],
                "target_texts": [
                    " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles)
                    for cut in all_cuts_combined
                ],
                "all_texts": [
                    " ".join(_strip_timestamps(s.text) for s in cut.supervisions) for cut in all_cuts_combined
                ],
                "target_first_turn_audio": target_first_turn_audio,
                "target_first_turn_audio_lens": target_first_turn_audio_lens,
                "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in all_cuts_combined],
                "aug_by_noise": [getattr(cut, "aug_by_noise", True) for cut in all_cuts_combined],
            }

            # Per-batch early interruption stats for logging (model accumulates for cumulative)
            early_interruption_stats = {
                "batch_total": batch_early_interruption_total,
                "batch_attempted": batch_early_interruption_attempted,
                "batch_successful": batch_early_interruption_successful,
            }

            if torch.sum(prompt_token_lens) > 0:
                audio_data['prompt_tokens'] = prompt_tokens
                audio_data['prompt_token_lens'] = prompt_token_lens

            # Optionally include detailed turn metadata for analysis
            if self.include_turn_metadata:
                audio_data["target_turn_texts"] = [
                    [
                        {
                            "start_time": s.start,
                            "duration": s.duration,
                            "role": s.speaker,
                            "text": s.text,
                        }
                        for s in cut.supervisions
                        if s.speaker in self.output_roles
                    ]
                    for cut in all_cuts_combined
                ]
                audio_data["source_turn_texts"] = [
                    [
                        {
                            "start_time": s.start,
                            "duration": s.duration,
                            "role": s.speaker,
                            "text": s.text,
                        }
                        for s in cut.supervisions
                        if s.speaker in self.input_roles
                    ]
                    for cut in all_cuts_combined
                ]
                audio_data["system_prompt"] = [cut.custom.get('system_prompt', '') for cut in all_cuts_combined]

        text_cuts = all_cuts.filter(lambda c: isinstance(c, Formattable))
        text_data = None
        if text_cuts:
            text_tokens = []
            text_token_lens = []
            for c in text_cuts:
                text_ids = c.input_ids
                text_tokens.append(text_ids)
                text_token_lens.append(text_ids.shape[0])

            text_tokens = collate_vectors(text_tokens, padding_value=self.pad_id)
            text_token_lens = torch.tensor(text_token_lens, dtype=torch.long)
            text_data = {
                "text_tokens": text_tokens,
                "text_token_lens": text_token_lens,
            }

        return {
            "audio_data": audio_data,
            "text_data": text_data,
            "early_interruption_stats": early_interruption_stats,
        }

    def _create_role_swapped_cut(self, cut):

        from io import BytesIO

        import numpy as np
        import soundfile as sf
        from lhotse import AudioSource

        swapped_supervisions = []
        for sup in cut.supervisions:
            if sup.speaker == 'User':
                new_speaker = 'Assistant'
            elif sup.speaker == 'Assistant':
                new_speaker = 'User'
            else:
                continue

            swapped_sup = SupervisionSegment(
                id=sup.id + "_swapped",
                recording_id=sup.recording_id,
                start=sup.start,
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=new_speaker,
                gender=sup.gender,
                custom=sup.custom,
                alignment=sup.alignment,
            )
            swapped_supervisions.append(swapped_sup)

        swapped_supervisions = sorted(swapped_supervisions, key=lambda s: s.start)

        first_agent_idx = None
        last_user_idx = None

        for i, sup in enumerate(swapped_supervisions):
            if sup.speaker == 'Assistant' and first_agent_idx is None:
                first_agent_idx = i
            if sup.speaker == 'User':
                last_user_idx = i

        filtered_supervisions = []
        for i, sup in enumerate(swapped_supervisions):
            if i != first_agent_idx and i != last_user_idx:
                filtered_supervisions.append(sup)

        if not filtered_supervisions:
            return None

        first_remaining_start = filtered_supervisions[0].start
        last_remaining_end = max(s.start + s.duration for s in filtered_supervisions)
        new_duration = last_remaining_end - first_remaining_start

        adjusted_supervisions = []
        for sup in filtered_supervisions:
            adjusted_sup = SupervisionSegment(
                id=sup.id,
                recording_id=sup.recording_id,
                start=sup.start - first_remaining_start,  # 减去offset
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=sup.speaker,
                gender=sup.gender,
                custom=sup.custom,
                alignment=sup.alignment,
            )
            adjusted_supervisions.append(adjusted_sup)

        total_duration = max(s.start + s.duration for s in adjusted_supervisions)
        total_samples = int(total_duration * cut.sampling_rate)

        new_source_audio = np.zeros(total_samples, dtype=np.float32)
        new_target_audio = np.zeros(total_samples, dtype=np.float32)

        for sup in adjusted_supervisions:
            start_sample = int(sup.start * cut.sampling_rate)
            end_sample = int((sup.start + sup.duration) * cut.sampling_rate)

            if sup.speaker == 'User':

                original_start = sup.start + first_remaining_start
                agent_audio = (
                    cut.custom['target_audio']
                    .to_cut()
                    .truncate(offset=original_start, duration=sup.duration)
                    .load_audio()
                )
                if len(agent_audio.shape) > 1:
                    agent_audio = agent_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(agent_audio))
                new_source_audio[start_sample:actual_end] = agent_audio[: actual_end - start_sample]

            elif sup.speaker == 'Assistant':
                original_start = sup.start + first_remaining_start
                user_audio = cut.recording.to_cut().truncate(offset=original_start, duration=sup.duration).load_audio()
                if len(user_audio.shape) > 1:
                    user_audio = user_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(user_audio))
                new_target_audio[start_sample:actual_end] = user_audio[: actual_end - start_sample]

        source_buffer = BytesIO()
        sf.write(source_buffer, new_source_audio, cut.sampling_rate, format='wav')
        source_buffer.seek(0)

        new_source_recording = Recording(
            id=f"{cut.id}_swapped_source",
            sampling_rate=cut.sampling_rate,
            num_samples=len(new_source_audio),
            duration=total_duration,
            sources=[AudioSource(type="memory", channels=[0], source=source_buffer.getvalue())],
        )

        target_buffer = BytesIO()
        sf.write(target_buffer, new_target_audio, cut.sampling_rate, format='wav')
        target_buffer.seek(0)

        new_target_recording = Recording(
            id=f"{cut.id}_swapped_target",
            sampling_rate=cut.sampling_rate,
            num_samples=len(new_target_audio),
            duration=total_duration,
            sources=[AudioSource(type="memory", channels=[0], source=target_buffer.getvalue())],
        )

        swapped_cut = MonoCut(
            id=f"{cut.id}_swapped",
            start=0,
            duration=total_duration,
            channel=0,
            supervisions=adjusted_supervisions,
            recording=new_source_recording,
            custom={
                **cut.custom,
                'total_turns': len(adjusted_supervisions),
                'role_swapped': True,
                'target_audio': new_target_recording,
            },
        )

        return swapped_cut


def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        # Find supervisions that match the specified roles
        matching_supervisions = [s for s in cut.supervisions if s.speaker in roles]

        if not matching_supervisions:
            # Log warning and skip this cut if no matching supervisions found
            logging.warning(
                f"No supervisions found with roles {roles} for cut {cut.id}. Available speakers: {[s.speaker for s in cut.supervisions]}"
            )
            continue

        first_supervision = matching_supervisions[0]
        truncated_audio = cut.truncate(
            offset=max(0, first_supervision.start), duration=first_supervision.duration
        ).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    if not first_turn_audios:
        # If no valid audio was found, return empty tensors
        logging.error(f"No valid audio found for any cuts with roles {roles}")
        return torch.empty(0), torch.empty(0)

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    pad_id: int,
    frame_length: Seconds,
    roles: set[str],
    bos_id: int = None,
    eos_id: int = None,
    word_align_position: str = 'left',
    remove_timestamps: bool = False,
    skip_eos: bool = False,
    agent_token_channel: torch.Tensor = None,
    agent_token_channel_lengths: torch.Tensor = None,
    agent_eos_id: int = None,
    early_interruption_flag_from_cfg: bool = None,
    use_numbers_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = [
        build_token_channel(
            c,
            tokenizer=tokenizer,
            frame_length=frame_length,
            roles=roles,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            word_align_position=word_align_position,
            remove_timestamps=remove_timestamps,
            skip_eos=skip_eos,
            cut_agent_token_channel=agent_token_channel[cut_idx] if agent_token_channel is not None else None,
            cut_agent_token_channel_length=(
                agent_token_channel_lengths[cut_idx] if agent_token_channel_lengths is not None else None
            ),
            agent_eos_id=agent_eos_id,
            use_numbers_norm=use_numbers_norm,
        )
        for cut_idx, c in enumerate(cuts)
    ]

    ei_flags = [getattr(c, 'otf_interruption', early_interruption_flag_from_cfg) for c in cuts]

    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens, ei_flags


def collate_system_prompt(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate system prompts from cuts.
    System prompts should be stored in cut.custom['system_prompt'].
    """
    tokens = []
    for c in cuts:
        # Check if system prompt exists in custom field
        if c.custom and c.custom.get("system_prompt", None):
            prompt_text = c.custom["system_prompt"]
            tokens.append(
                torch.as_tensor(
                    [tokenizer.bos] + tokenizer.text_to_ids(prompt_text) + [tokenizer.eos], dtype=torch.long
                )
            )
        else:
            # No system prompt for this cut
            tokens.append(torch.as_tensor([], dtype=torch.long))

    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
    cut: Cut,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    pad_id: int = -1,
    bos_id: int = None,
    eos_id: int = None,
    word_align_position: str = 'left',
    remove_timestamps: bool = False,
    skip_eos: bool = False,
    cut_agent_token_channel: torch.Tensor = None,
    cut_agent_token_channel_length: torch.Tensor = None,
    eos_offset_frames: int = 8,
    agent_eos_id: int = None,
    use_numbers_norm: bool = False,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id

    if cut_agent_token_channel is not None:
        if cut_agent_token_channel_length is None or cut_agent_token_channel_length.item() != total:
            logging.error(
                f"Mismatch between agent token and source token lengths: {cut_agent_token_channel_length.item() if cut_agent_token_channel_length is not None else 'None'} != {total}"
            )
            return None

    for supervision in cut.supervisions:
        if supervision.speaker in roles:

            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos >= len(tokens):  # Changed from > to >= for robustness
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}"
                )
                continue
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            available_frames_for_text = eospos - pos

            text = supervision.text
            if use_numbers_norm:
                text = normalize_numbers(text)

            # Use different bos_id for user and agent
            text_ids = torch.as_tensor(
                [bos_id]
                + _text_to_ids(
                    text,
                    tokenizer,
                    available_frames_for_text=available_frames_for_text,
                    word_align_position=word_align_position,
                    remove_timestamps=remove_timestamps,
                )
            )

            if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                # Truncate text_ids to fit before the eos position.
                text_ids = text_ids[:available_frames_for_text]
            elif available_frames_for_text <= 0:
                # If there's no space for text (e.g., start >= end), use an empty sequence.
                text_ids = torch.tensor([], dtype=torch.long)

            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
                endpos = pos + len(text_ids)

            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            if not skip_eos:
                # Place EOS token - critical for turn-taking behavior
                if cut_agent_token_channel is not None:
                    # Place agent EOS as a function of user start time
                    assert agent_eos_id is not None, "Agent EOS ID is not set"
                    user_bospos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
                    agent_eospos = user_bospos + eos_offset_frames
                    if agent_eospos < cut_agent_token_channel_length.item():
                        cut_agent_token_channel[agent_eospos] = agent_eos_id
                    else:
                        logging.warning(
                            f"Agent EOS position {agent_eospos} is out of bounds for agent token channel {cut_agent_token_channel.shape}"
                        )

                if eospos < len(tokens) and eos_id is not None:
                    # Place user EOS
                    # Normal case: place EOS at the intended position
                    tokens[eospos] = eos_id

    return tokens


def _strip_timestamps(
    text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces


def _text_to_ids(
    text: str,
    tokenizer: TokenizerSpec,
    _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
    available_frames_for_text=None,
    word_align_position='left',
    remove_timestamps=False,
):
    if not remove_timestamps and re.compile(_TIMESTAMP_PATTERN_STR).search(text):
        text_ids = _text_with_timestamps_to_ids(
            text, tokenizer, _TIMESTAMP_PATTERN_STR, available_frames_for_text, word_align_position
        )
    else:
        _TIMESTAMP_PATTERN = re.compile(_TIMESTAMP_PATTERN_STR)
        text = _TIMESTAMP_PATTERN.sub("", text)
        # Remove extra spaces between words
        text = " ".join(text.strip().split())
        text_ids = tokenizer.text_to_ids(text)
    return text_ids


def _text_with_timestamps_to_ids(
    text: str,
    tokenizer: TokenizerSpec,
    _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
    available_frames_for_text=None,
    word_align_position='left',
) -> list[int]:
    text_ids = []
    text_ids, start_times, end_times, word_lens = _extract_text_and_time_tokens(
        text, tokenizer, _TIMESTAMP_PATTERN_STR
    )
    text_ids_with_timestamps = _expand_text_with_timestamps_and_word_lengths(
        text_ids,
        word_lens,
        start_times,
        end_times,
        available_frames_for_text,
        frame_rate=0.08,
        pad_id=get_pad_id(tokenizer),
        word_align_position=word_align_position,
    )
    return text_ids_with_timestamps


def _extract_text_and_time_tokens(text, tokenizer: TokenizerSpec, _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>"):
    # Find all time tokens
    time_tokens = re.findall(_TIMESTAMP_PATTERN_STR, text)
    start_time = [int(time_tokens[i]) for i in range(0, len(time_tokens), 2)]
    end_time = [int(time_tokens[i]) for i in range(1, len(time_tokens), 2)]
    # Remove all time tokens to isolate words
    words = re.sub(_TIMESTAMP_PATTERN_STR, '', text).split()
    # Process each word, tokenize it, and calculate token lengths
    text_ids = []
    word_lens = []
    for i, word in enumerate(words):
        word_with_space = word if i == 0 else ' ' + word
        word_ids = tokenizer.text_to_ids(word_with_space)
        word_len = len(word_ids)
        text_ids.extend(word_ids)
        word_lens.append(word_len)
    return text_ids, start_time, end_time, word_lens


def _expand_text_with_timestamps_and_word_lengths(
    text_ids,
    word_lens,
    start_time,
    end_time,
    available_frames_for_text,
    frame_rate=0.08,
    pad_id=None,
    word_align_position='left',
):
    """
    Expand word tokens according to start time tokens and word lengths for a batch of sequences.

    Args:
    - word_tokens: List of text ids w/o timestamps
    - word_lens: List of word lengths
    - start_time: List of start times
    - end_time: List of end times
    - available_frames_for_text: Maximum number of frames for text
    - frame_rate: Frame rate resolution
    - pad_id: Padding ID to use for empty positions in the tensor

    Returns:
    - text ids with word-level timestamps
    """

    def discretize_time(start_token, speech_frame_rate=0.08, timestamp_frame_rate=0.08):
        return int(start_token * timestamp_frame_rate / speech_frame_rate)

    if pad_id is None:
        raise ValueError("pad_id must be provided.")

    max_length = available_frames_for_text

    # Create the empty tensor with pad_id as the default value
    text_ids_with_timestamps = [pad_id] * max_length

    # Populate ids of each word starting at start_idx and ending at end_idx
    cur_word_idx = 0  # Start frame index of current word
    for word_idx, word_len in enumerate(word_lens):
        start_idx = discretize_time(start_time[word_idx], speech_frame_rate=frame_rate)
        end_idx = discretize_time(end_time[word_idx], speech_frame_rate=frame_rate)
        if word_align_position == 'left':
            end_idx = min(start_idx + word_len, end_idx)
        elif word_align_position == 'right':
            start_idx = max(start_idx, end_idx - word_len)
        else:
            raise ValueError(f"Unknown word_align_position: {word_align_position}")

        # Get ids of a single word
        word_ids = text_ids[cur_word_idx : cur_word_idx + word_len]

        # Populate a single word
        for i in range(start_idx, end_idx + 1):  # End inclusive at word level
            if i - start_idx < len(word_ids) and i < max_length:
                token_id = word_ids[i - start_idx]
                text_ids_with_timestamps[i] = token_id

        # Move to the next word in the concatenated word tokens
        cur_word_idx += word_len

    return text_ids_with_timestamps
