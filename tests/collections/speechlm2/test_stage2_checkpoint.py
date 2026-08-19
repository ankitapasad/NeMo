# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import pytest
import torch
from torch import nn

from nemo.collections.speechlm2.parts.stage2_checkpoint import prepare_asr_stage2_state_dict


class _Tokenizer:
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def encode(self, token, add_special_tokens=False):
        assert add_special_tokens is False
        return [self.token_ids[token]]


class _TokenizerWrapper:
    def __init__(self, token_ids):
        self.tokenizer = _Tokenizer(token_ids)


class _TinyTiedModel(nn.Module):
    def __init__(self, rows=6, hidden=3):
        super().__init__()
        self.embed_tokens = nn.Embedding(rows, hidden)
        self.lm_head = nn.Linear(hidden, rows, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.tokenizer = _TokenizerWrapper(
            {
                "<sou>": rows - 4,
                "<eou>": rows - 3,
                "<|text_start|>": rows - 2,
                "<|text_end|>": rows - 1,
            }
        )


def _source_state(model, source_rows=4):
    return {
        "embed_tokens.weight": torch.arange(source_rows * 3, dtype=torch.float32).reshape(source_rows, 3),
        "lm_head.weight": torch.arange(source_rows * 3, dtype=torch.float32).reshape(source_rows, 3),
    }


def test_expands_vocab_rows_preserves_boundaries_and_remaps_compact_text():
    model = _TinyTiedModel()
    boundary_rows = model.embed_tokens.weight.detach()[2:4].clone()
    source = _source_state(model)

    expanded = prepare_asr_stage2_state_dict(model, source, expected_source_vocab_size=4)
    model.load_state_dict(expanded, strict=True)

    torch.testing.assert_close(model.embed_tokens.weight[:2], source["embed_tokens.weight"][:2])
    torch.testing.assert_close(model.embed_tokens.weight[2:4], boundary_rows)
    torch.testing.assert_close(model.embed_tokens.weight[4:], source["embed_tokens.weight"][2:])
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_rejects_any_non_vocab_shape_mismatch():
    model = _TinyTiedModel()
    model.extra = nn.Parameter(torch.ones(2))
    source = _source_state(model)
    source["extra"] = torch.ones(1)

    with pytest.raises(RuntimeError, match="Only the embedding and LM-head"):
        prepare_asr_stage2_state_dict(model, source, expected_source_vocab_size=4)


def test_rejects_non_appended_boundary_token_ids():
    model = _TinyTiedModel()
    model.tokenizer = _TokenizerWrapper({"<sou>": 1, "<eou>": 3, "<|text_start|>": 4, "<|text_end|>": 5})

    with pytest.raises(RuntimeError, match="must occupy the source compact-text rows"):
        prepare_asr_stage2_state_dict(model, _source_state(model), expected_source_vocab_size=4)


def test_rejects_non_appended_compact_text_token_ids():
    model = _TinyTiedModel()
    model.tokenizer = _TokenizerWrapper({"<sou>": 2, "<eou>": 3, "<|text_start|>": 1, "<|text_end|>": 5})

    with pytest.raises(RuntimeError, match="must move to appended rows"):
        prepare_asr_stage2_state_dict(model, _source_state(model), expected_source_vocab_size=4)
