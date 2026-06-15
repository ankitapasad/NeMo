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

from types import SimpleNamespace

import torch
import torch.nn as nn

import nemo.collections.speechlm2.models.streaming_stt_model as streaming_stt_model


class _FakeTokenizer:
    def __init__(self):
        self.tokenizer = self
        self.vocab = {"<unk>": 0}
        self.unk_id = 0
        self.unk_token_id = 0
        self.pad_id = 0
        self.added_batches = []

    def __len__(self):
        return len(self.vocab)

    def text_to_tokens(self, token):
        if token in self.vocab:
            return [token]
        return list(token)

    def text_to_ids(self, token):
        return [self.vocab[token]]

    def add_special_tokens(self, special_tokens_dict):
        tokens = list(special_tokens_dict["additional_special_tokens"])
        self.added_batches.append(tokens)
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)


class _FakeLLM(nn.Module):
    def __init__(self, vocab_size=1, hidden_size=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.resize_sizes = []

    def resize_token_embeddings(self, vocab_size):
        self.resize_sizes.append(vocab_size)
        self.model.embed_tokens = nn.Embedding(vocab_size, self.config.hidden_size)
        self.lm_head = nn.Linear(self.config.hidden_size, vocab_size)


def _fake_perception():
    return SimpleNamespace(
        encoder=nn.Linear(1, 1),
        modality_adapter=nn.Linear(1, 1),
        proj=nn.Linear(1, 1),
    )


def _minimal_cfg():
    return {
        "pretrained_llm": "fake-llm",
        "pretrained_asr": "fake-asr",
        "load_llm_weights": False,
        "load_asr_weights": False,
        "blank_token": "<blank>",
        "freeze_speech_encoder": False,
        "freeze_modality_adapter": False,
        "freeze_modality_proj": False,
        "freeze_llm_model": False,
        "freeze_llm_head": False,
        "freeze_embed_tokens": False,
        "chunk_size": 2,
        "sample_rate": 16000,
        "frame_length_in_secs": 0.08,
        "compact_template": True,
        "use_text_tokens": True,
        "text_start_token": "<|text_start|>",
        "text_end_token": "<|text_end|>",
        "add_utterance_boundary_tokens": True,
        "utterance_start_token": "<sou>",
        "utterance_end_token": "<eou>",
    }


def test_model_init_adds_boundary_and_text_tokens(monkeypatch):
    fake_tokenizer = _FakeTokenizer()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(streaming_stt_model, "AutoTokenizer", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(streaming_stt_model, "load_pretrained_hf", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(streaming_stt_model, "setup_perception", lambda *args, **kwargs: _fake_perception())
    monkeypatch.setattr(streaming_stt_model, "ModelSummary", lambda *args, **kwargs: "summary")

    model = streaming_stt_model.StreamingSTTModel(_minimal_cfg())

    assert fake_tokenizer.added_batches == [
        ["<blank>"],
        ["<sou>", "<eou>"],
        ["<|text_start|>", "<|text_end|>"],
    ]
    assert fake_llm.resize_sizes == [2, 4, 6]
    assert model._compact_write_token == "<|text_start|>"
    assert model._compact_end_token == "<|text_end|>"


def test_model_init_maps_legacy_te_config(monkeypatch):
    fake_tokenizer = _FakeTokenizer()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(streaming_stt_model, "AutoTokenizer", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(streaming_stt_model, "load_pretrained_hf", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(streaming_stt_model, "setup_perception", lambda *args, **kwargs: _fake_perception())
    monkeypatch.setattr(streaming_stt_model, "ModelSummary", lambda *args, **kwargs: "summary")

    cfg = _minimal_cfg()
    cfg.pop("use_text_tokens")
    cfg.pop("text_start_token")
    cfg.pop("text_end_token")
    cfg["use_te_tokens"] = True
    cfg["te_start_token"] = "<|te_start|>"
    cfg["te_end_token"] = "<|te_end|>"

    model = streaming_stt_model.StreamingSTTModel(cfg)

    assert fake_tokenizer.added_batches == [
        ["<blank>"],
        ["<sou>", "<eou>"],
        ["<|te_start|>", "<|te_end|>"],
    ]
    assert model._compact_write_token == "<|te_start|>"
    assert model._compact_end_token == "<|te_end|>"


def test_model_init_end_only_no_blank_adds_only_text_end(monkeypatch):
    fake_tokenizer = _FakeTokenizer()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(streaming_stt_model, "AutoTokenizer", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(streaming_stt_model, "load_pretrained_hf", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(streaming_stt_model, "setup_perception", lambda *args, **kwargs: _fake_perception())
    monkeypatch.setattr(streaming_stt_model, "ModelSummary", lambda *args, **kwargs: "summary")

    cfg = _minimal_cfg()
    cfg["compact_text_end_only_no_blank"] = True
    model = streaming_stt_model.StreamingSTTModel(cfg)

    assert fake_tokenizer.added_batches == [
        ["<sou>", "<eou>"],
        ["<|text_end|>"],
    ]
    assert fake_llm.resize_sizes == [3, 4]
    assert model.blank_token == ""
    assert model._compact_write_token is None
    assert model._compact_end_token == "<|text_end|>"


class _ValidationLogger:
    def __init__(self):
        self.logged = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = value

    def log_dict(self, metrics, **kwargs):
        self.logged.update(metrics)


def _append_boundary_totals(container, name, **values):
    defaults = {
        "num_samples": 1,
        "sou_target_count": 1,
        "eou_target_count": 1,
        "sou_pred_count": 1,
        "eou_pred_count": 1,
        "sou_collar_hit": 1,
        "eou_collar_hit": 1,
    }
    defaults.update(values)
    for metric, value in defaults.items():
        container._partial_boundary_metrics[name][metric].append(torch.tensor(value))


def test_validation_epoch_end_logs_boundary_metrics_by_loader_and_overall():
    model = _ValidationLogger()
    streaming_stt_model.StreamingSTTModel.on_validation_epoch_start(model)
    _append_boundary_totals(
        model,
        "dev",
        num_samples=2,
        sou_target_count=2,
        eou_target_count=2,
        sou_pred_count=1,
        eou_pred_count=2,
        sou_collar_hit=2,
        eou_collar_hit=2,
    )
    _append_boundary_totals(
        model,
        "test",
        num_samples=1,
        sou_target_count=1,
        eou_target_count=1,
        sou_pred_count=1,
        eou_pred_count=0,
        sou_collar_hit=1,
        eou_collar_hit=0,
    )

    streaming_stt_model.StreamingSTTModel.on_validation_epoch_end(model)

    assert torch.isclose(model.logged["val_sou_pred_per_sample_dev"], torch.tensor(0.5))
    assert torch.isclose(model.logged["val_eou_pred_per_sample_dev"], torch.tensor(1.0))
    assert torch.isclose(model.logged["val_sou_collar_acc_dev"], torch.tensor(1.0))
    assert torch.isclose(model.logged["val_eou_collar_acc_dev"], torch.tensor(1.0))
    assert torch.isclose(model.logged["val_sou_pred_per_sample"], torch.tensor(2 / 3))
    assert torch.isclose(model.logged["val_eou_pred_per_sample"], torch.tensor(2 / 3))
    assert torch.isclose(model.logged["val_sou_collar_acc"], torch.tensor(1.0))
    assert torch.isclose(model.logged["val_eou_collar_acc"], torch.tensor(2 / 3))
    assert all(not metric.startswith("val_boundary") for metric in model.logged)
    assert "val_sou_target_per_sample" not in model.logged
    assert "val_eou_target_per_sample" not in model.logged
