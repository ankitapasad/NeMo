# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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
    def __init__(self, boundary_tokens=("<sou>", "<eou>"), source_rows=4, hidden=3):
        super().__init__()
        rows = source_rows + len(boundary_tokens)
        self.embed_tokens = nn.Embedding(rows, hidden)
        self.lm_head = nn.Linear(hidden, rows, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        first_inserted_id = source_rows - 2
        token_ids = {
            token: first_inserted_id + index for index, token in enumerate(boundary_tokens)
        }
        token_ids.update(
            {
                "<|text_start|>": first_inserted_id + len(boundary_tokens),
                "<|text_end|>": first_inserted_id + len(boundary_tokens) + 1,
            }
        )
        self.tokenizer = _TokenizerWrapper(token_ids)


def _source_state(source_rows=4, hidden=3):
    source = torch.arange(source_rows * hidden, dtype=torch.float32).reshape(source_rows, hidden)
    return {"embed_tokens.weight": source.clone(), "lm_head.weight": source.clone()}


@pytest.mark.parametrize(
    "boundary_tokens",
    [
        ("<sou>", "<eou>"),
        ("<sou>", "<eou>", "<sob>"),
        ("<sou>", "<eou>", "<sob>", "<eob>"),
    ],
)
def test_strict_vocab_expansion_preserves_inserted_rows_and_remaps_compact_text(
    boundary_tokens,
):
    model = _TinyTiedModel(boundary_tokens=boundary_tokens)
    source = _source_state()
    first_inserted_id = 2
    inserted_rows = model.embed_tokens.weight.detach()[
        first_inserted_id : first_inserted_id + len(boundary_tokens)
    ].clone()

    expanded = prepare_asr_stage2_state_dict(
        model,
        source,
        boundary_tokens=boundary_tokens,
        expected_source_vocab_size=4,
    )
    model.load_state_dict(expanded, strict=True)

    torch.testing.assert_close(model.embed_tokens.weight[:2], source["embed_tokens.weight"][:2])
    torch.testing.assert_close(
        model.embed_tokens.weight[2 : 2 + len(boundary_tokens)], inserted_rows
    )
    torch.testing.assert_close(
        model.embed_tokens.weight[2 + len(boundary_tokens) :],
        source["embed_tokens.weight"][2:],
    )
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()


def test_rejects_any_non_vocab_shape_mismatch():
    model = _TinyTiedModel()
    model.extra = nn.Parameter(torch.ones(2))
    source = _source_state()
    source["extra"] = torch.ones(1)

    with pytest.raises(RuntimeError, match="Only the embedding and LM-head"):
        prepare_asr_stage2_state_dict(model, source, expected_source_vocab_size=4)


def test_rejects_non_appended_boundary_token_ids():
    model = _TinyTiedModel()
    model.tokenizer = _TokenizerWrapper(
        {"<sou>": 1, "<eou>": 3, "<|text_start|>": 4, "<|text_end|>": 5}
    )

    with pytest.raises(RuntimeError, match="must occupy the source compact-text rows"):
        prepare_asr_stage2_state_dict(model, _source_state(), expected_source_vocab_size=4)


def test_rejects_non_appended_compact_text_token_ids():
    model = _TinyTiedModel()
    model.tokenizer = _TokenizerWrapper(
        {"<sou>": 2, "<eou>": 3, "<|text_start|>": 1, "<|text_end|>": 5}
    )

    with pytest.raises(RuntimeError, match="must move to appended rows"):
        prepare_asr_stage2_state_dict(model, _source_state(), expected_source_vocab_size=4)
