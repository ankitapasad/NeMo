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
Tests for StreamingSTTModel inference helpers: ``_find_sublist`` and
``_ensure_inference_cache``.

Uses the real Qwen3 tokenizer to ensure the template is correct for the
actual model used in training.
"""

from types import SimpleNamespace

import pytest
from transformers import AutoTokenizer

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX
from nemo.collections.speechlm2.models.streaming_stt_model import StreamingSTTModel, _find_sublist

PRETRAINED_LLM = "Qwen/Qwen3-1.7B"
BLANK_TOKEN = "<blank>"
CHUNK_SIZE = 2

# Qwen3 special token IDs
IM_START = 151644
IM_END = 151645
NEWLINE = 198
THINK_START = 151667
THINK_END = 151668


@pytest.fixture(scope="module")
def hf_tok():
    return AutoTokenizer.from_pretrained(PRETRAINED_LLM)


def _make_mock_self(hf_tok, chunk_size=CHUNK_SIZE, blank_token=BLANK_TOKEN):
    """Build a minimal namespace that satisfies ``_ensure_inference_cache``."""
    return SimpleNamespace(
        tokenizer=SimpleNamespace(tokenizer=hf_tok),
        core_cfg=SimpleNamespace(chunk_size=chunk_size, audio_tag="<audio>"),
        blank_token=blank_token,
    )


def _run_ensure_cache(hf_tok, chunk_size=CHUNK_SIZE):
    """Call ``_ensure_inference_cache`` on a mock self and return it."""
    mock = _make_mock_self(hf_tok, chunk_size=chunk_size)
    StreamingSTTModel._ensure_inference_cache(mock)
    return mock


# ===========================================================================
# Tests: _find_sublist
# ===========================================================================
class TestFindSublist:

    def test_found_middle(self):
        assert _find_sublist([1, 2, 3, 4, 5], [3, 4]) == 2

    def test_found_start(self):
        assert _find_sublist([1, 2, 3], [1, 2]) == 0

    def test_found_end(self):
        assert _find_sublist([1, 2, 3], [2, 3]) == 1

    def test_exact_match(self):
        assert _find_sublist([1, 2], [1, 2]) == 0

    def test_not_found(self):
        assert _find_sublist([1, 2, 3], [4, 5]) is None

    def test_single_element(self):
        assert _find_sublist([1, 2, 3], [2]) == 1

    def test_empty_haystack(self):
        assert _find_sublist([], [1]) is None

    def test_returns_first_occurrence(self):
        assert _find_sublist([1, 2, 1, 2], [1, 2]) == 0


# ===========================================================================
# Tests: _ensure_inference_cache  (real Qwen3 tokenizer)
# ===========================================================================
class TestEnsureInferenceCache:

    def test_audio_slot_count(self, hf_tok):
        """Turn template must contain exactly chunk_size AUDIO_TOKEN_IDX markers."""
        mock = _run_ensure_cache(hf_tok, chunk_size=2)
        assert mock._turn_template_ids.count(AUDIO_TOKEN_IDX) == 2

    def test_audio_slot_count_different_chunk_sizes(self, hf_tok):
        for cs in (1, 2, 4, 8):
            mock = _run_ensure_cache(hf_tok, chunk_size=cs)
            assert mock._turn_template_ids.count(AUDIO_TOKEN_IDX) == cs

    def test_no_think_tokens_in_template(self, hf_tok):
        """The turn template must NOT contain Qwen3 <think>/<\/think> tokens."""
        mock = _run_ensure_cache(hf_tok)
        assert THINK_START not in mock._turn_template_ids
        assert THINK_END not in mock._turn_template_ids

    def test_template_starts_with_user_header(self, hf_tok):
        """Template must start with <|im_start|> user \\n."""
        mock = _run_ensure_cache(hf_tok)
        ids = mock._turn_template_ids
        assert ids[0] == IM_START
        # "user" token followed by newline
        user_id = hf_tok.encode("user", add_special_tokens=False)[0]
        assert ids[1] == user_id
        assert ids[2] == NEWLINE

    def test_template_ends_with_assistant_header(self, hf_tok):
        """Template must end with <|im_start|> assistant \\n."""
        mock = _run_ensure_cache(hf_tok)
        ids = mock._turn_template_ids
        asst_id = hf_tok.encode("assistant", add_special_tokens=False)[0]
        assert ids[-3:] == [IM_START, asst_id, NEWLINE]

    def test_audio_block_is_contiguous(self, hf_tok):
        """AUDIO_TOKEN_IDX markers should form a contiguous block."""
        mock = _run_ensure_cache(hf_tok, chunk_size=4)
        ids = mock._turn_template_ids
        first = ids.index(AUDIO_TOKEN_IDX)
        block = ids[first : first + 4]
        assert block == [AUDIO_TOKEN_IDX] * 4
        # No stray markers outside the block
        assert ids[:first].count(AUDIO_TOKEN_IDX) == 0
        assert ids[first + 4 :].count(AUDIO_TOKEN_IDX) == 0

    def test_template_matches_training_format(self, hf_tok):
        """The turn template should exactly match what the dataset produces
        for a single non-last user+assistant turn (no think tags)."""
        mock = _run_ensure_cache(hf_tok, chunk_size=2)
        ids = mock._turn_template_ids

        # Manually construct the expected template
        user_header = [IM_START] + hf_tok.encode("user", add_special_tokens=False) + [NEWLINE]
        audio = [AUDIO_TOKEN_IDX] * 2
        user_footer = [IM_END, NEWLINE]
        asst_header = [IM_START] + hf_tok.encode("assistant", add_special_tokens=False) + [NEWLINE]
        expected = user_header + audio + user_footer + asst_header
        assert ids == expected

    def test_assistant_footer_ids(self, hf_tok):
        """Assistant footer should be [<|im_end|>, \\n]."""
        mock = _run_ensure_cache(hf_tok)
        assert mock._asst_footer_ids == [IM_END, NEWLINE]

    def test_eos_id(self, hf_tok):
        """EOS ID must be set from the tokenizer."""
        mock = _run_ensure_cache(hf_tok)
        assert mock._eos_id == hf_tok.eos_token_id

    def test_eos_in_footer(self, hf_tok):
        """For Qwen3, eos_token_id (<|im_end|>) is the first token of the footer."""
        mock = _run_ensure_cache(hf_tok)
        assert mock._eos_in_footer is True
        assert mock._eos_id == mock._asst_footer_ids[0]

    def test_blank_id(self, hf_tok):
        """Blank token ID should be resolved (not UNK)."""
        mock = _make_mock_self(hf_tok)
        # Add <blank> as a special token (as the model __init__ does)
        hf_tok_copy = AutoTokenizer.from_pretrained(PRETRAINED_LLM)
        hf_tok_copy.add_special_tokens({"additional_special_tokens": [BLANK_TOKEN]})
        mock.tokenizer.tokenizer = hf_tok_copy
        StreamingSTTModel._ensure_inference_cache(mock)
        blank_id = hf_tok_copy.convert_tokens_to_ids(BLANK_TOKEN)
        assert mock._blank_id == blank_id
        assert blank_id != hf_tok_copy.unk_token_id

    def test_idempotent(self, hf_tok):
        """Calling _ensure_inference_cache twice should not change results."""
        mock = _make_mock_self(hf_tok)
        StreamingSTTModel._ensure_inference_cache(mock)
        first_template = list(mock._turn_template_ids)
        first_footer = list(mock._asst_footer_ids)
        first_eos = mock._eos_id
        first_eos_in_footer = mock._eos_in_footer

        StreamingSTTModel._ensure_inference_cache(mock)
        assert mock._turn_template_ids == first_template
        assert mock._asst_footer_ids == first_footer
        assert mock._eos_id == first_eos
        assert mock._eos_in_footer == first_eos_in_footer
