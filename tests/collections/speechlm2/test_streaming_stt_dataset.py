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

"""
Tests for StreamingSTTDataset message generation, token replacement, and
input/target construction.

The primary reference is the docstring example in get_llm_messages_for_sample:

    alignments = [
        WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
        WordAlignment(text="World", start_time=0.60, end_time=0.80),
    ]
    audio_duration = 1s, chunk_size = 2, frame_length = 0.08s, delay = 0

    → 13 frames, 7 chunks, assistant responses:
      [<blank>, <blank>, Hello, <blank>, World, <blank>, <blank>]
"""

import math

import pytest

from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    AUDIO_TOKEN_IDX,
    IGNORE_INDEX,
    _replace_audio_chunks,
    _tokenize_with_assistant_mask,
    compute_word_spans,
    get_llm_messages_for_batch,
    get_llm_messages_for_sample,
)
from nemo.collections.speechlm2.parts.alignments import WordAlignment

# ---------------------------------------------------------------------------
# Shared constants & helpers matching the docstring example
# ---------------------------------------------------------------------------
AUDIO_TAG = "<audio>"
BLANK_TOKEN = "<blank>"
SYSTEM_ROLE = "system"
SYSTEM_PROMPT = "Transcribe the audio into text."
CHUNK_SIZE = 2
FRAME_LEN = 0.08  # seconds
DOCSTRING_ALIGNMENTS = [
    WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
    WordAlignment(text="World", start_time=0.60, end_time=0.80),
]


def _make_messages(**overrides):
    """Convenience wrapper around get_llm_messages_for_sample with docstring defaults."""
    kw = dict(
        system_role=SYSTEM_ROLE,
        system_prompt=SYSTEM_PROMPT,
        audio_tag=AUDIO_TAG,
        blank_token=BLANK_TOKEN,
        chunk_size=CHUNK_SIZE,
        num_delay_frames=0,
        audio_duration_secs=1.0,
        frame_length_in_secs=FRAME_LEN,
        alignments=DOCSTRING_ALIGNMENTS,
    )
    kw.update(overrides)
    return get_llm_messages_for_sample(**kw)


# ---------------------------------------------------------------------------
# Mock tokenizer used by TestTokenPositions
# ---------------------------------------------------------------------------
class _MockHFTokenizer:
    """
    Deterministic HF tokenizer mock.

    Token layout per message:
        [HEADER_START, ROLE_ID, HEADER_END, ...content..., FOOTER, NEWLINE]

    Content encoding:
        system  → [50]
        user    → [AUDIO_TAG_ID] per <audio> tag in content
        assistant → [BLANK_ID] for "<blank>", else [200, 201, ...] per word
    """

    HEADER_START = 1
    ROLE_IDS = {"system": 10, "user": 11, "assistant": 12}
    HEADER_END = 2
    FOOTER = 3
    NEWLINE = 4
    AUDIO_TAG_ID = 100
    BLANK_ID = 101
    SYSTEM_CONTENT_ID = 50

    HEADER = [HEADER_START, None, HEADER_END]  # None → filled per role
    N_HEADER = 3
    N_FOOTER = 2

    def __init__(self, audio_tag=AUDIO_TAG, blank_token=BLANK_TOKEN):
        self.audio_tag = audio_tag
        self.blank_token = blank_token
        self.unk_token_id = 0
        self.eos_token_id = _MockHFTokenizer.FOOTER  # EOS = first footer token (like Qwen3)
        self._next_word_id = 200
        # Cache for content → token IDs mapping so encode() and apply_chat_template() agree.
        self._content_cache: dict[str, list[int]] = {}

    def _content_to_ids(self, content: str, role: str) -> list[int]:
        """Deterministic content → token IDs, consistent between encode() and apply_chat_template()."""
        if role == "user":
            return [self.AUDIO_TAG_ID] * content.count(self.audio_tag)
        if role == "assistant":
            if content == self.blank_token:
                return [self.BLANK_ID]
            # Assign stable IDs per unique content string
            if content not in self._content_cache:
                ids = []
                for _ in content.split():
                    ids.append(self._next_word_id)
                    self._next_word_id += 1
                self._content_cache[content] = ids
            return list(self._content_cache[content])
        # system
        return [self.SYSTEM_CONTENT_ID]

    def encode(self, text, add_special_tokens=False):
        if text == self.audio_tag:
            return [self.AUDIO_TAG_ID]
        if text == self.blank_token:
            return [self.BLANK_ID]
        # Footer text from the ChatML-like text template
        if text == "<|im_end|>\n":
            return [self.FOOTER, self.NEWLINE]
        # Handle repeated audio tags (chunk encoding)
        if self.audio_tag in text and text == self.audio_tag * text.count(self.audio_tag):
            return [self.AUDIO_TAG_ID] * text.count(self.audio_tag)
        # For assistant word content, use the cache
        if text in self._content_cache:
            return list(self._content_cache[text])
        # Unknown text — assign stable IDs
        ids = []
        for _ in text.split():
            ids.append(self._next_word_id)
            self._next_word_id += 1
        self._content_cache[text] = ids
        return list(ids)

    def apply_chat_template(self, messages, **kwargs):
        tokenize = kwargs.get("tokenize", True)

        if not tokenize:
            # Return ChatML-like text form for sentinel-based footer discovery.
            text = ""
            for msg in messages:
                text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            return text

        input_ids = []
        assistant_masks = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            header = [self.HEADER_START, self.ROLE_IDS[role], self.HEADER_END]
            input_ids.extend(header)
            assistant_masks.extend([0] * len(header))

            ids = self._content_to_ids(content, role)

            input_ids.extend(ids)
            assistant_masks.extend([1 if role == "assistant" else 0] * len(ids))

            footer = [self.FOOTER, self.NEWLINE]
            input_ids.extend(footer)
            assistant_masks.extend([0] * len(footer))

        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


class _MockHFTokenizerMultiToken(_MockHFTokenizer):
    """Mock where <audio> tokenizes into 3 tokens: [60, 61, 62].

    Simulates BPE merging across adjacent tags: ``<audio><audio>`` tokenizes as
    [60, 61, 70, 61, 62] (5 tokens) instead of [60, 61, 62, 60, 61, 62] (6 tokens),
    because ``62`` (``>``) and ``60`` (``<``) merge into ``70`` (``><``).
    """

    MULTI_AUDIO_TAG_IDS = [60, 61, 62]
    MERGED_BOUNDARY = 70  # simulates BPE merge of > + <

    def encode(self, text, add_special_tokens=False):
        if text == self.audio_tag:
            return list(self.MULTI_AUDIO_TAG_IDS)
        # Simulate BPE merging across adjacent audio tags
        n = text.count(self.audio_tag)
        if n > 0 and text == self.audio_tag * n:
            # First tag: [60, 61, 62], subsequent: [70, 61, 62] (merged boundary)
            ids = list(self.MULTI_AUDIO_TAG_IDS)
            for _ in range(n - 1):
                ids.append(self.MERGED_BOUNDARY)
                ids.extend(self.MULTI_AUDIO_TAG_IDS[1:])  # skip first token, use merged
            return ids
        return super().encode(text, add_special_tokens=add_special_tokens)

    def _content_to_ids(self, content: str, role: str) -> list[int]:
        if role == "user":
            return self.encode(content, add_special_tokens=False)
        return super()._content_to_ids(content, role)


class _MockHFTokenizerNoGeneration(_MockHFTokenizer):
    """Mock that simulates a tokenizer without {% generation %} — returns all-zero masks."""

    def apply_chat_template(self, messages, **kwargs):
        result = super().apply_chat_template(messages, **kwargs)
        # Zero out the masks to simulate missing {% generation %} support.
        # When tokenize=False, result is a string — pass through unchanged.
        if isinstance(result, dict):
            result["assistant_masks"] = [0] * len(result["assistant_masks"])
        return result


class _MockHFTokenizerNoEOS(_MockHFTokenizerNoGeneration):
    """Mock without eos_token_id — footer trimming should fall back to full footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eos_token_id = None


class _MockNemoTokenizer:
    """Wraps a mock HF tokenizer to mimic NeMo AutoTokenizer interface."""

    def __init__(self, hf_tok):
        self.tokenizer = hf_tok


def _run_pipeline(messages, mock_hf_tok, chunk_size=CHUNK_SIZE):
    """Simulate the __getitem__ tokenization pipeline: tokenize → replace → build targets."""
    audio_chunk_ids = mock_hf_tok.encode(AUDIO_TAG * chunk_size, add_special_tokens=False)
    nemo_tok = _MockNemoTokenizer(mock_hf_tok)

    input_ids, assistant_mask = _tokenize_with_assistant_mask(messages, nemo_tok)

    input_ids, assistant_mask = _replace_audio_chunks(
        input_ids,
        audio_chunk_ids,
        chunk_size,
        mask=assistant_mask,
    )

    target_ids = input_ids[1:] + [IGNORE_INDEX]
    target_mask = assistant_mask[1:] + [0]
    target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

    return input_ids, target_ids, assistant_mask


# ===========================================================================
# Tests: get_llm_messages_for_sample
# ===========================================================================
class TestGetLlmMessagesForSample:

    def test_docstring_example_structure(self):
        """Total messages: 1 system + 7*(user + assistant) = 15."""
        msgs = _make_messages()
        assert len(msgs) == 15
        assert msgs[0] == {"role": SYSTEM_ROLE, "content": SYSTEM_PROMPT}

    def test_docstring_example_roles_alternate(self):
        msgs = _make_messages()
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system"
        for i in range(1, len(roles), 2):
            assert roles[i] == "user"
            assert roles[i + 1] == "assistant"

    def test_docstring_example_user_turns(self):
        msgs = _make_messages()
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert len(user_msgs) == 7
        assert all(m["content"] == AUDIO_TAG * CHUNK_SIZE for m in user_msgs)

    def test_docstring_example_assistant_responses(self):
        msgs = _make_messages()
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst == [BLANK_TOKEN, BLANK_TOKEN, "Hello", BLANK_TOKEN, "World", BLANK_TOKEN, BLANK_TOKEN]

    def test_num_chunks(self):
        """ceil(13 frames / 2) = 7 chunks."""
        msgs = _make_messages()
        assert sum(1 for m in msgs if m["role"] == "user") == 7

    def test_total_audio_tags_equals_chunks_times_chunk_size(self):
        msgs = _make_messages()
        user_msgs = [m for m in msgs if m["role"] == "user"]
        total = sum(m["content"].count(AUDIO_TAG) for m in user_msgs)
        assert total == len(user_msgs) * CHUNK_SIZE

    def test_delay_shifts_emission(self):
        """With delay=2, Hello (end_frame=6) → ready_frame=8 → chunk 3 (end=8)."""
        msgs = _make_messages(
            num_delay_frames=2,
            alignments=[WordAlignment(text="Hello", start_time=0.16, end_time=0.48)],
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[:3] == [BLANK_TOKEN, BLANK_TOKEN, BLANK_TOKEN]
        assert asst[3] == "Hello"

    def test_empty_alignments_all_blank(self):
        msgs = _make_messages(alignments=[])
        for m in msgs:
            if m["role"] == "assistant":
                assert m["content"] == BLANK_TOKEN

    def test_none_alignments_all_blank(self):
        msgs = _make_messages(alignments=None)
        for m in msgs:
            if m["role"] == "assistant":
                assert m["content"] == BLANK_TOKEN

    def test_multiple_words_in_same_chunk(self):
        alignments = [
            WordAlignment(text="A", start_time=0.0, end_time=0.04),
            WordAlignment(text="B", start_time=0.05, end_time=0.08),
        ]
        msgs = _make_messages(alignments=alignments, audio_duration_secs=0.16)
        # A: end_frame=round(0.04/0.08)=0, B: end_frame=1. Both ≤ chunk 0 end=2.
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "A B"

    def test_zero_duration_only_system(self):
        msgs = _make_messages(audio_duration_secs=0.0, alignments=[])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"

    def test_chunk_size_1(self):
        msgs = _make_messages(chunk_size=1, alignments=[])
        # 13 frames → 13 chunks
        assert sum(1 for m in msgs if m["role"] == "user") == 13
        user_msgs = [m for m in msgs if m["role"] == "user"]
        assert all(m["content"] == AUDIO_TAG for m in user_msgs)

    def test_residual_words_appended_to_last_turn(self):
        """Words whose ready_frame exceeds the last chunk should still appear."""
        # Audio is 0.16s → 2 frames → 1 chunk (end_frame=2).
        # Word ends at 0.20s → end_frame=ceil(0.20/0.08)=3. With delay=0, ready_frame=3 > 2.
        # The word would be dropped without the residual fix.
        alignments = [WordAlignment(text="Late", start_time=0.10, end_time=0.20)]
        msgs = _make_messages(
            audio_duration_secs=0.16,
            alignments=alignments,
            num_delay_frames=0,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert "Late" in asst[-1], f"Residual word 'Late' not in last turn: {asst}"

    def test_residual_words_with_delay(self):
        """Delay pushes a word past the last chunk — it should still be emitted."""
        # Audio is 1.0s. Word ends at 0.96s → end_frame=12. With delay=2, ready_frame=14.
        # Last chunk end_frame = ceil(13/2)*2 = 14. So ready_frame=14 <= 14, it fits.
        # But if word ends at 1.0s → end_frame=13, ready_frame=15 > 14. Residual.
        alignments = [
            WordAlignment(text="Hello", start_time=0.0, end_time=0.48),
            WordAlignment(text="World", start_time=0.80, end_time=1.0),
        ]
        msgs = _make_messages(
            audio_duration_secs=1.0,
            alignments=alignments,
            num_delay_frames=2,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        all_text = " ".join(a for a in asst if a != BLANK_TOKEN)
        assert "Hello" in all_text, f"'Hello' missing: {asst}"
        assert "World" in all_text, f"'World' missing: {asst}"

    def test_residual_replaces_blank_last_turn(self):
        """If last turn was blank and there are residual words, blank is replaced."""
        # Short audio, word ends after it
        alignments = [WordAlignment(text="Overflow", start_time=0.0, end_time=0.20)]
        msgs = _make_messages(
            audio_duration_secs=0.08,  # 1 frame → 1 chunk (end_frame=2 with chunk_size=2? No, ceil(1/2)=1 chunk, end_frame=2)
            alignments=alignments,
            num_delay_frames=0,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # The word should appear, not blank
        assert asst[-1] == "Overflow", f"Expected 'Overflow' but got: {asst}"


# ===========================================================================
# Tests: compute_word_spans
# ===========================================================================
class TestComputeWordSpans:

    def test_simple(self):
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello world")
        assert spans == [(0, 5), (6, 11)]

    def test_trailing_punctuation_included(self):
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello, world!")
        assert spans == [(0, 6), (7, 13)]  # "hello," and "world!"

    def test_quotes_included(self):
        alignments = [
            WordAlignment(text="good", start_time=0.0, end_time=0.2),
            WordAlignment(text="night", start_time=0.3, end_time=0.5),
        ]
        spans = compute_word_spans(alignments, "'good night'")
        # "good" found at idx 1, trailing: nothing (space follows)
        # "night" found at idx 6, trailing: "'"
        assert spans == [(1, 5), (6, 12)]

    def test_case_insensitive_match(self):
        alignments = [WordAlignment(text="Hello", start_time=0.0, end_time=0.3)]
        spans = compute_word_spans(alignments, "HELLO world")
        assert spans == [(0, 5)]

    def test_word_not_found(self):
        alignments = [WordAlignment(text="missing", start_time=0.0, end_time=0.3)]
        spans = compute_word_spans(alignments, "hello world")
        assert spans == [None]

    def test_sequential_search(self):
        """Repeated words match sequentially, not all to the first occurrence."""
        alignments = [
            WordAlignment(text="the", start_time=0.0, end_time=0.1),
            WordAlignment(text="the", start_time=0.5, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "the cat and the dog")
        assert spans == [(0, 3), (12, 15)]

    def test_empty_alignments(self):
        assert compute_word_spans([], "hello world") == []

    # --- preserve_whitespace ---

    def test_preserve_whitespace_extends_to_next_word(self):
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello world", preserve_whitespace=True)
        # "hello " (includes trailing space), "world" (no trailing space at end)
        assert spans == [(0, 6), (6, 11)]

    def test_preserve_whitespace_with_punctuation(self):
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.3),
            WordAlignment(text="world", start_time=0.4, end_time=0.6),
        ]
        spans = compute_word_spans(alignments, "hello, world!", preserve_whitespace=True)
        # "hello, " (punct + space), "world!" (punct, no trailing space)
        assert spans == [(0, 7), (7, 13)]

    def test_preserve_whitespace_multi_space(self):
        alignments = [
            WordAlignment(text="a", start_time=0.0, end_time=0.1),
            WordAlignment(text="b", start_time=0.2, end_time=0.3),
        ]
        spans = compute_word_spans(alignments, "a   b", preserve_whitespace=True)
        # "a   " (3 spaces consumed), "b"
        assert spans == [(0, 4), (4, 5)]

    def test_preserve_whitespace_last_word_no_trailing(self):
        """Last word's span should not extend beyond the transcript."""
        alignments = [WordAlignment(text="end", start_time=0.0, end_time=0.3)]
        spans = compute_word_spans(alignments, "the end", preserve_whitespace=True)
        assert spans == [(4, 7)]  # no trailing space to consume


# ===========================================================================
# Tests: get_llm_messages_for_sample with transcript
# ===========================================================================
class TestTranscriptPreservation:

    def test_punctuation_preserved(self):
        """Trailing punctuation from transcript should be in assistant content."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.08),
            WordAlignment(text="world", start_time=0.10, end_time=0.16),
        ]
        msgs = get_llm_messages_for_sample(
            system_role=SYSTEM_ROLE,
            system_prompt=SYSTEM_PROMPT,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=2,
            num_delay_frames=0,
            audio_duration_secs=0.16,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
            transcript="Hello, World!",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "Hello, World!"

    def test_multi_word_chunk_preserves_spacing(self):
        """Multiple words in one chunk use the transcript's inter-word text."""
        alignments = [
            WordAlignment(text="said", start_time=0.0, end_time=0.06),
            WordAlignment(text="good", start_time=0.07, end_time=0.10),
        ]
        msgs = get_llm_messages_for_sample(
            system_role=SYSTEM_ROLE,
            system_prompt=SYSTEM_PROMPT,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=2,
            num_delay_frames=0,
            audio_duration_secs=0.16,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
            transcript="she said good night",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # Trailing space is included because preserve_whitespace extends through
        # whitespace after "good" (up to "night"), ensuring correct concatenation
        # when turns are joined.
        assert asst[0] == "said good "

    def test_without_transcript_falls_back(self):
        """Without transcript, words are joined with plain space."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.08),
            WordAlignment(text="world", start_time=0.10, end_time=0.16),
        ]
        msgs = get_llm_messages_for_sample(
            system_role=SYSTEM_ROLE,
            system_prompt=SYSTEM_PROMPT,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=2,
            num_delay_frames=0,
            audio_duration_secs=0.16,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
            transcript=None,
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "hello world"

    def test_single_word_with_comma(self):
        """A single word followed by comma should include the comma."""
        alignments = [
            WordAlignment(text="yes", start_time=0.0, end_time=0.08),
            WordAlignment(text="indeed", start_time=0.20, end_time=0.30),
        ]
        msgs = get_llm_messages_for_sample(
            system_role=SYSTEM_ROLE,
            system_prompt=SYSTEM_PROMPT,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=2,
            num_delay_frames=0,
            audio_duration_secs=0.32,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
            transcript="yes, indeed",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # "yes" is ready at chunk 0 (end_frame=1 <= 2), alone in its chunk.
        # Trailing space included via preserve_whitespace (space before "indeed").
        assert asst[0] == "yes, "

    def test_blanks_unchanged_with_transcript(self):
        """Blank chunks still produce <blank> even when transcript is provided."""
        msgs = _make_messages(transcript="Hello World")
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # First two chunks are blank
        assert asst[0] == BLANK_TOKEN
        assert asst[1] == BLANK_TOKEN


# ===========================================================================
# Tests: get_llm_messages_for_batch
# ===========================================================================
class TestGetLlmMessagesForBatch:

    def test_per_sample_duration(self):
        """Each sample gets messages based on its own duration, not a shared max."""
        alignments = [[], []]
        durations = [0.16, 0.32]  # 2 frames → 1 chunk, 4 frames → 2 chunks
        batch = get_llm_messages_for_batch(
            system_role=SYSTEM_ROLE,
            system_prompt=[SYSTEM_PROMPT, SYSTEM_PROMPT],
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=CHUNK_SIZE,
            num_delay_frames=0,
            audio_durations_secs=durations,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
        )
        assert len(batch) == 2
        chunks_0 = sum(1 for m in batch[0] if m["role"] == "user")
        chunks_1 = sum(1 for m in batch[1] if m["role"] == "user")
        assert chunks_0 == 1
        assert chunks_1 == 2

    def test_per_sample_system_prompt(self):
        """Each sample gets its own system prompt from the list."""
        prompts = ["Transcribe in English.", "Transcribe in French."]
        alignments = [[], []]
        durations = [0.16, 0.16]
        batch = get_llm_messages_for_batch(
            system_role=SYSTEM_ROLE,
            system_prompt=prompts,
            audio_tag=AUDIO_TAG,
            blank_token=BLANK_TOKEN,
            chunk_size=CHUNK_SIZE,
            num_delay_frames=0,
            audio_durations_secs=durations,
            frame_length_in_secs=FRAME_LEN,
            alignments=alignments,
        )
        assert batch[0][0] == {"role": SYSTEM_ROLE, "content": "Transcribe in English."}
        assert batch[1][0] == {"role": SYSTEM_ROLE, "content": "Transcribe in French."}


# ===========================================================================
# Tests: _replace_audio_chunks
# ===========================================================================
class TestReplaceAudioChunks:

    AUD = AUDIO_TOKEN_IDX

    def test_single_token_chunk(self):
        """Single-token audio tag, chunk_size=2 → 2 AUDIO_TOKEN_IDX per chunk."""
        ids = [1, 100, 100, 2]
        result = _replace_audio_chunks(ids, [100, 100], chunk_size=2)
        assert result == [1, self.AUD, self.AUD, 2]

    def test_multi_token_chunk_with_bpe_merge(self):
        """Simulates BPE merge: <audio><audio> → [60, 61, 70, 61, 62] (5 tokens, not 6)."""
        chunk_ids = [60, 61, 70, 61, 62]
        ids = [1, 2, 3] + chunk_ids + [4, 5]
        result = _replace_audio_chunks(ids, chunk_ids, chunk_size=2)
        assert result == [1, 2, 3, self.AUD, self.AUD, 4, 5]

    def test_multiple_chunks(self):
        chunk_ids = [60, 61, 70, 61, 62]
        ids = chunk_ids + [99] + chunk_ids + [88]
        result = _replace_audio_chunks(ids, chunk_ids, chunk_size=2)
        assert result == [self.AUD, self.AUD, 99, self.AUD, self.AUD, 88]
        assert result.count(self.AUD) == 4  # 2 chunks × 2

    def test_chunk_size_1(self):
        """chunk_size=1: each chunk token sequence replaced with 1 AUDIO_TOKEN_IDX."""
        chunk_ids = [60, 61, 62]  # single <audio> as 3 BPE tokens
        ids = [1] + chunk_ids + [2] + chunk_ids + [3]
        result = _replace_audio_chunks(ids, chunk_ids, chunk_size=1)
        assert result == [1, self.AUD, 2, self.AUD, 3]

    def test_chunk_size_4(self):
        """chunk_size=4: each chunk replaced with 4 AUDIO_TOKEN_IDX."""
        chunk_ids = [10, 11, 12, 13]
        ids = [1] + chunk_ids + [2]
        result = _replace_audio_chunks(ids, chunk_ids, chunk_size=4)
        assert result == [1, self.AUD, self.AUD, self.AUD, self.AUD, 2]

    def test_mask_sync(self):
        chunk_ids = [60, 61, 70, 61, 62]
        ids = [1] + chunk_ids + [2]
        mask = [0] + [0, 0, 0, 0, 0] + [1]
        new_ids, new_mask = _replace_audio_chunks(ids, chunk_ids, chunk_size=2, mask=mask)
        assert new_ids == [1, self.AUD, self.AUD, 2]
        assert new_mask == [0, 0, 0, 1]
        assert len(new_ids) == len(new_mask)

    def test_mask_length_with_different_chunk_size(self):
        """Mask length must match ids length after chunk replacement."""
        chunk_ids = [10, 11, 12, 13, 14]  # 5 BPE tokens
        ids = [1] + chunk_ids + [2] + chunk_ids + [3]
        mask = [0] + [0] * 5 + [1] + [0] * 5 + [1]
        new_ids, new_mask = _replace_audio_chunks(ids, chunk_ids, chunk_size=3, mask=mask)
        # 5 tokens → 3 AUDIO_TOKEN_IDX per chunk, 2 chunks
        assert new_ids.count(self.AUD) == 6
        assert len(new_ids) == len(new_mask)

    def test_no_match(self):
        result = _replace_audio_chunks([1, 2, 3], [100, 100], chunk_size=2)
        assert result == [1, 2, 3]


# ===========================================================================
# Tests: token positions (full pipeline through mock tokenizer)
# ===========================================================================
class TestTokenPositions:
    """
    Verify audio/text token counts and positions in input_ids and target_ids
    using the docstring example.
    """

    def test_audio_token_count_single_token_tag(self):
        """AUDIO_TOKEN_IDX count == num_chunks * chunk_size (single-token tag)."""
        msgs = _make_messages()
        (
            input_ids,
            _,
            _,
        ) = _run_pipeline(msgs, _MockHFTokenizer())
        num_chunks = 7
        assert input_ids.count(AUDIO_TOKEN_IDX) == num_chunks * CHUNK_SIZE

    def test_audio_token_count_multi_token_tag(self):
        """Same count even when the audio tag tokenizes into 3 tokens."""
        msgs = _make_messages()
        input_ids, _, _ = _run_pipeline(msgs, _MockHFTokenizerMultiToken())
        num_chunks = 7
        assert input_ids.count(AUDIO_TOKEN_IDX) == num_chunks * CHUNK_SIZE

    def test_no_audio_token_at_assistant_position(self):
        msgs = _make_messages()
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        for i, (tid, m) in enumerate(zip(input_ids, assistant_mask)):
            if m:
                assert tid != AUDIO_TOKEN_IDX, f"Audio token at assistant position {i}"

    def test_no_audio_token_at_assistant_position_multi(self):
        msgs = _make_messages()
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizerMultiToken())
        for i, (tid, m) in enumerate(zip(input_ids, assistant_mask)):
            if m:
                assert tid != AUDIO_TOKEN_IDX, f"Audio token at assistant position {i}"

    def test_target_ignore_at_non_assistant(self):
        """Every non-assistant position in target must be IGNORE_INDEX."""
        msgs = _make_messages()
        input_ids, target_ids, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        shifted_mask = assistant_mask[1:] + [0]
        for i, (tid, m) in enumerate(zip(target_ids, shifted_mask)):
            if not m:
                assert tid == IGNORE_INDEX, f"target[{i}]={tid} should be IGNORE_INDEX"

    def test_target_real_at_assistant(self):
        """Every assistant position in target must hold a real token ID."""
        msgs = _make_messages()
        input_ids, target_ids, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        shifted_mask = assistant_mask[1:] + [0]
        for i, (tid, m) in enumerate(zip(target_ids, shifted_mask)):
            if m:
                assert tid != IGNORE_INDEX, f"target[{i}] should be a real token"

    def test_target_equals_next_input_at_assistant(self):
        """target[i] must equal input[i+1] at trainable positions (next-token prediction)."""
        msgs = _make_messages()
        input_ids, target_ids, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        shifted = input_ids[1:] + [IGNORE_INDEX]
        shifted_mask = assistant_mask[1:] + [0]
        for i, m in enumerate(shifted_mask):
            if m:
                assert target_ids[i] == shifted[i], f"target[{i}]={target_ids[i]} != next input {shifted[i]}"

    def test_input_target_same_length(self):
        msgs = _make_messages()
        input_ids, target_ids, _ = _run_pipeline(msgs, _MockHFTokenizer())
        assert len(input_ids) == len(target_ids)

    def test_input_target_same_length_multi_token(self):
        msgs = _make_messages()
        input_ids, target_ids, _ = _run_pipeline(msgs, _MockHFTokenizerMultiToken())
        assert len(input_ids) == len(target_ids)

    def test_mask_length_matches_input_after_replace(self):
        """After multi-token collapse, mask and input_ids must have the same length."""
        msgs = _make_messages()
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizerMultiToken())
        assert len(input_ids) == len(assistant_mask)

    def test_all_blank_targets_with_no_alignments(self):
        """With no alignments, every assistant content token in input should be BLANK_ID."""
        msgs = _make_messages(alignments=[])
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        blank_id = _MockHFTokenizer.BLANK_ID
        for i, (tid, m) in enumerate(zip(input_ids, assistant_mask)):
            if m:
                assert tid == blank_id, f"Expected blank at position {i}, got {tid}"

    def test_hello_appears_at_chunk_2(self):
        """'Hello' (end_time=0.48s, end_frame=6) is emitted at chunk 2 (end_frame=6)."""
        msgs = _make_messages(
            alignments=[WordAlignment(text="Hello", start_time=0.16, end_time=0.48)],
        )
        input_ids, _, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        blank_id = _MockHFTokenizer.BLANK_ID

        # Collect assistant content token values in order
        asst_tokens = [tid for tid, m in zip(input_ids, assistant_mask) if m]
        # Chunks: 0=blank, 1=blank, 2=Hello (non-blank), 3..6=blank
        assert asst_tokens[0] == blank_id
        assert asst_tokens[1] == blank_id
        assert asst_tokens[2] != blank_id  # Hello word token
        assert all(t == blank_id for t in asst_tokens[3:])

    def test_trainable_token_count(self):
        """Number of trainable positions in target == number of assistant content tokens."""
        msgs = _make_messages()
        _, target_ids, assistant_mask = _run_pipeline(msgs, _MockHFTokenizer())
        n_trainable = sum(1 for t in target_ids if t != IGNORE_INDEX)
        n_assistant = sum(assistant_mask)
        # The shifted mask loses the first assistant token's prediction target
        # when it's preceded by a non-assistant token, but gains/loses nothing
        # else.  The exact count: sum(shifted_mask).
        shifted_mask = assistant_mask[1:] + [0]
        assert n_trainable == sum(shifted_mask)


# ===========================================================================
# Tests: _tokenize_with_assistant_mask fallback
# ===========================================================================
class TestTokenizeWithAssistantMaskFallback:
    """
    Verify the sequential-search fallback produces the same mask as the
    primary path when the tokenizer doesn't support {% generation %}.
    """

    def test_fallback_includes_primary_content_plus_footer(self):
        """Fallback mask should include all primary-masked (content) positions plus footer tokens."""
        msgs = _make_messages()
        primary_tok = _MockHFTokenizer()
        fallback_tok = _MockHFTokenizerNoGeneration()

        nemo_primary = _MockNemoTokenizer(primary_tok)
        nemo_fallback = _MockNemoTokenizer(fallback_tok)

        ids_p, mask_p = _tokenize_with_assistant_mask(msgs, nemo_primary)
        ids_f, mask_f = _tokenize_with_assistant_mask(msgs, nemo_fallback)

        assert ids_p == ids_f, "Token IDs should be identical"
        # Fallback mask includes all primary-masked (content) positions
        for i, (mp, mf) in enumerate(zip(mask_p, mask_f)):
            if mp:
                assert mf, f"Position {i}: primary has mask=1 but fallback has mask=0"
        # Fallback has additional masked positions (footer tokens)
        assert sum(mask_f) > sum(mask_p), "Fallback should have additional footer positions"

    def test_fallback_has_nonzero_mask(self):
        """Fallback should produce assistant-masked tokens, not all zeros."""
        msgs = _make_messages()
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        _, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)
        assert any(mask), "Fallback mask should have at least one assistant token"

    def test_fallback_mask_count_includes_eos(self):
        """Number of masked tokens should equal assistant content + 1 EOS token per turn."""
        msgs = _make_messages()
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        _, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)
        # 7 assistant turns: 7 content tokens + 7 * 1 EOS token (FOOTER only, not NEWLINE) = 14
        assert sum(mask) == 7 + 7 * 1

    def test_fallback_eos_in_mask_but_not_rest_of_footer(self):
        """Fallback should mask the EOS token but not post-EOS footer tokens."""
        msgs = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "<blank>"},
        ]
        tok = _MockHFTokenizerNoGeneration()
        nemo_tok = _MockNemoTokenizer(tok)

        input_ids, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)

        # Find the blank token position
        blank_id = _MockHFTokenizer.BLANK_ID
        blank_pos = input_ids.index(blank_id)

        # FOOTER (= eos_token_id) should be masked, NEWLINE should NOT
        assert input_ids[blank_pos + 1] == _MockHFTokenizer.FOOTER
        assert input_ids[blank_pos + 2] == _MockHFTokenizer.NEWLINE
        assert mask[blank_pos] == 1, "Content token should be masked"
        assert mask[blank_pos + 1] == 1, "EOS (FOOTER) should be masked"
        assert mask[blank_pos + 2] == 0, "Post-EOS (NEWLINE) should NOT be masked"

    def test_fallback_full_footer_when_no_eos(self):
        """When eos_token_id is None, the full footer should be included in the mask."""
        msgs = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "<blank>"},
        ]
        tok = _MockHFTokenizerNoEOS()
        nemo_tok = _MockNemoTokenizer(tok)

        input_ids, mask = _tokenize_with_assistant_mask(msgs, nemo_tok)

        blank_id = _MockHFTokenizer.BLANK_ID
        blank_pos = input_ids.index(blank_id)

        # Without EOS, the full footer (FOOTER + NEWLINE) should be masked
        assert mask[blank_pos] == 1, "Content should be masked"
        assert mask[blank_pos + 1] == 1, "FOOTER should be masked"
        assert mask[blank_pos + 2] == 1, "NEWLINE should be masked (no EOS trimming)"

    def test_fallback_pipeline_produces_trainable_targets(self):
        """Full pipeline with fallback tokenizer should have non-zero trainable targets."""
        msgs = _make_messages()
        tok = _MockHFTokenizerNoGeneration()
        input_ids, target_ids, assistant_mask = _run_pipeline(msgs, tok)

        n_trainable = sum(1 for t in target_ids if t != IGNORE_INDEX)
        assert n_trainable > 0, "Should have trainable targets with fallback mask"

    def test_fallback_eot_in_target(self):
        """After shift, the model should be trained to predict the end-of-turn token."""
        msgs = [
            {"role": "system", "content": "test"},
            {"role": "user", "content": "<audio><audio>"},
            {"role": "assistant", "content": "<blank>"},
        ]
        tok = _MockHFTokenizerNoGeneration()
        input_ids, target_ids, _ = _run_pipeline(msgs, tok)

        # The FOOTER token (end-of-turn) should appear as a trainable target
        footer_id = _MockHFTokenizer.FOOTER
        assert footer_id in target_ids, "FOOTER should appear as a trainable target"
        # Verify it's not masked out
        footer_target_pos = target_ids.index(footer_id)
        assert target_ids[footer_target_pos] != IGNORE_INDEX


# ===========================================================================
# Tests: chunk_size=-1 (offline / single-chunk mode)
# ===========================================================================
class TestOfflineSingleChunk:
    """Verify chunk_size=-1 treats the whole audio as one chunk."""

    def test_single_chunk_structure(self):
        """chunk_size=-1 should produce exactly 1 user turn + 1 assistant turn."""
        msgs = _make_messages(chunk_size=-1)
        user_msgs = [m for m in msgs if m["role"] == "user"]
        asst_msgs = [m for m in msgs if m["role"] == "assistant"]
        assert len(user_msgs) == 1
        assert len(asst_msgs) == 1

    def test_audio_tags_equal_num_frames(self):
        """The single user turn should have num_frames audio tags."""
        msgs = _make_messages(chunk_size=-1, audio_duration_secs=1.0)
        user_msg = [m for m in msgs if m["role"] == "user"][0]
        num_frames = math.ceil(1.0 / FRAME_LEN)  # 13
        assert user_msg["content"] == AUDIO_TAG * num_frames

    def test_all_words_in_single_turn(self):
        """All words should appear in the single assistant turn."""
        msgs = _make_messages(chunk_size=-1)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert len(asst) == 1
        assert "Hello" in asst[0]
        assert "World" in asst[0]

    def test_no_blanks(self):
        """With all audio in one chunk, all words are ready — no blanks."""
        msgs = _make_messages(chunk_size=-1)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert BLANK_TOKEN not in asst

    def test_transcript_preserved(self):
        """Punctuation from transcript should be preserved in single-chunk mode."""
        alignments = [
            WordAlignment(text="hello", start_time=0.0, end_time=0.08),
            WordAlignment(text="world", start_time=0.10, end_time=0.16),
        ]
        msgs = _make_messages(
            chunk_size=-1,
            audio_duration_secs=0.16,
            alignments=alignments,
            transcript="Hello, World!",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst[0] == "Hello, World!"

    def test_empty_alignments_no_transcript_produces_blank(self):
        """No alignments and no transcript → single blank turn."""
        msgs = _make_messages(chunk_size=-1, alignments=[], transcript=None)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst == [BLANK_TOKEN]

    def test_empty_alignments_with_transcript_uses_transcript(self):
        """No alignments but transcript provided → uses raw transcript."""
        msgs = _make_messages(
            chunk_size=-1,
            alignments=[],
            audio_duration_secs=1.0,
            transcript="Hello, World!",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst == ["Hello, World!"]

    def test_none_alignments_with_transcript_uses_transcript(self):
        """None alignments with transcript → uses raw transcript."""
        msgs = _make_messages(
            chunk_size=-1,
            alignments=None,
            audio_duration_secs=1.0,
            transcript="some text here",
        )
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert asst == ["some text here"]

    def test_zero_duration(self):
        """Zero-duration audio with no alignments → early return with empty user turn and blank."""
        msgs = _make_messages(chunk_size=-1, audio_duration_secs=0.0, alignments=[])
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1] == {"role": "user", "content": ""}
        assert msgs[2] == {"role": "assistant", "content": BLANK_TOKEN}

    def test_delay_ignored_single_chunk(self):
        """With one chunk spanning all frames, delay shouldn't matter
        (all words fit within the single chunk's end frame)."""
        msgs = _make_messages(chunk_size=-1, num_delay_frames=2)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        # num_frames=13, chunk_end_frame=13.
        # Hello: end_frame=6, ready=8 <= 13 ✓
        # World: end_frame=10, ready=12 <= 13 ✓
        assert len(asst) == 1
        assert "Hello" in asst[0]
        assert "World" in asst[0]

    def test_delay_causes_residual_in_single_chunk(self):
        """Large delay can push a word past the single chunk's end frame → residual."""
        # Audio 1.0s → 13 frames → chunk_end_frame=13
        # World: end_frame=ceil(0.80/0.08)=10, with delay=5 → ready_frame=15 > 13
        msgs = _make_messages(chunk_size=-1, num_delay_frames=5)
        asst = [m["content"] for m in msgs if m["role"] == "assistant"]
        assert len(asst) == 1
        # World should still appear via the residual logic
        assert "Hello" in asst[0]
        assert "World" in asst[0]

    def test_matches_large_chunk_size(self):
        """chunk_size=-1 should produce the same result as chunk_size=num_frames."""
        num_frames = math.ceil(1.0 / FRAME_LEN)
        msgs_offline = _make_messages(chunk_size=-1)
        msgs_large = _make_messages(chunk_size=num_frames)
        # Both should have 1 system + 1 user + 1 assistant = 3 messages
        assert len(msgs_offline) == len(msgs_large)
        # Same assistant content
        asst_offline = [m["content"] for m in msgs_offline if m["role"] == "assistant"]
        asst_large = [m["content"] for m in msgs_large if m["role"] == "assistant"]
        assert asst_offline == asst_large
        # Same user content (same number of audio tags)
        user_offline = [m["content"] for m in msgs_offline if m["role"] == "user"]
        user_large = [m["content"] for m in msgs_large if m["role"] == "user"]
        assert user_offline == user_large

    def test_chunk_size_zero_raises(self):
        """chunk_size=0 should raise an assertion error."""
        with pytest.raises(AssertionError):
            _make_messages(chunk_size=0)
