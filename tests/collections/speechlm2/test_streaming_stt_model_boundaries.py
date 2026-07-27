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

import pytest
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


def test_model_rejects_per_sample_boundaries_without_boundary_token_support():
    cfg = _minimal_cfg()
    cfg["add_utterance_boundary_tokens"] = False
    with pytest.raises(
        ValueError,
        match="use_per_sample_utterance_boundary_tokens=True requires model.add_utterance_boundary_tokens=True",
    ):
        streaming_stt_model.StreamingSTTModel(
            cfg,
            data_cfg={"use_per_sample_utterance_boundary_tokens": True},
        )


def test_model_rejects_per_sample_timestamps_without_per_sample_boundaries():
    with pytest.raises(
        ValueError,
        match=(
            "use_per_sample_utterance_boundary_timestamps=True requires "
            "use_per_sample_utterance_boundary_tokens=True"
        ),
    ):
        streaming_stt_model.StreamingSTTModel(
            _minimal_cfg(),
            data_cfg={
                "use_per_sample_utterance_boundary_tokens": False,
                "use_per_sample_utterance_boundary_timestamps": True,
            },
        )


def test_model_rejects_mismatched_agent_backchannel_flags():
    cfg = _minimal_cfg()
    cfg["enable_agent_backchannels"] = True
    with pytest.raises(
        ValueError,
        match="model.enable_agent_backchannels and data.dataset.enable_agent_backchannels must match",
    ):
        streaming_stt_model.StreamingSTTModel(
            cfg,
            data_cfg={"enable_agent_backchannels": False},
        )


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


def test_model_init_adds_agent_backchannel_tokens_only_when_enabled(monkeypatch):
    fake_tokenizer = _FakeTokenizer()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(streaming_stt_model, "AutoTokenizer", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(streaming_stt_model, "load_pretrained_hf", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(streaming_stt_model, "setup_perception", lambda *args, **kwargs: _fake_perception())
    monkeypatch.setattr(streaming_stt_model, "ModelSummary", lambda *args, **kwargs: "summary")

    cfg = _minimal_cfg()
    cfg["enable_agent_backchannels"] = True
    model = streaming_stt_model.StreamingSTTModel(
        cfg,
        data_cfg={
            "enable_agent_backchannels": True,
            "agent_backchannel_start_token": "<soab>",
            "agent_backchannel_end_token": "<eoab>",
        },
    )

    assert fake_tokenizer.added_batches == [
        ["<blank>"],
        ["<sou>", "<eou>"],
        ["<soab>", "<eoab>"],
        ["<|text_start|>", "<|text_end|>"],
    ]
    assert fake_llm.resize_sizes == [2, 4, 6, 8]
    assert model.core_cfg.agent_backchannel_start_loss_weight == 1.0
    assert model.core_cfg.agent_backchannel_end_loss_weight == 1.0


def test_model_init_ignores_inactive_agent_backchannel_config_values(monkeypatch):
    fake_tokenizer = _FakeTokenizer()
    fake_llm = _FakeLLM()

    monkeypatch.setattr(streaming_stt_model, "AutoTokenizer", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(streaming_stt_model, "load_pretrained_hf", lambda *args, **kwargs: fake_llm)
    monkeypatch.setattr(streaming_stt_model, "setup_perception", lambda *args, **kwargs: _fake_perception())
    monkeypatch.setattr(streaming_stt_model, "ModelSummary", lambda *args, **kwargs: "summary")

    cfg = _minimal_cfg()
    cfg.update(
        enable_agent_backchannels=False,
        agent_backchannel_start_token="",
        agent_backchannel_end_token="",
    )
    streaming_stt_model.StreamingSTTModel(
        cfg,
        data_cfg={
            "enable_agent_backchannels": False,
            "agent_backchannel_start_token": "",
            "agent_backchannel_end_token": "",
        },
    )

    assert fake_tokenizer.added_batches == [
        ["<blank>"],
        ["<sou>", "<eou>"],
        ["<|text_start|>", "<|text_end|>"],
    ]
    assert fake_llm.resize_sizes == [2, 4, 6]


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


def test_weighted_lm_loss_applies_separate_boundary_weights():
    per_token_loss = torch.tensor([1.0, 2.0, 3.0, 4.0, 99.0])
    targets = torch.tensor([10, 11, 12, 13, streaming_stt_model.IGNORE_INDEX])

    loss, metrics = streaming_stt_model._compute_weighted_lm_loss(
        per_token_loss=per_token_loss,
        flat_targets=targets,
        blank_id=12,
        has_blank=True,
        blank_loss_weight=0.5,
        sou_id=10,
        eou_id=11,
        utterance_start_loss_weight=2.0,
        utterance_end_loss_weight=5.0,
    )

    expected = torch.tensor((1.0 * 2.0 + 2.0 * 5.0 + 3.0 * 0.5 + 4.0) / (2.0 + 5.0 + 0.5 + 1.0))
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["loss_sou"], torch.tensor(1.0))
    assert torch.allclose(metrics["loss_eou"], torch.tensor(2.0))
    assert torch.allclose(metrics["loss_blank"], torch.tensor(3.0))
    assert torch.allclose(metrics["sou_ratio"], torch.tensor(0.25))
    assert torch.allclose(metrics["eou_ratio"], torch.tensor(0.25))


def test_weighted_lm_loss_defaults_to_unweighted_boundary_tokens_without_backchannel_metrics():
    per_token_loss = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([10, 11, 13])

    loss, metrics = streaming_stt_model._compute_weighted_lm_loss(
        per_token_loss=per_token_loss,
        flat_targets=targets,
        blank_id=12,
        has_blank=True,
        blank_loss_weight=1.0,
        sou_id=10,
        eou_id=11,
    )

    assert torch.allclose(loss, per_token_loss.mean())
    assert {"loss_soab", "loss_eoab", "soab_ratio", "eoab_ratio"}.isdisjoint(metrics)


def test_weighted_lm_loss_applies_agent_backchannel_marker_weights_only_to_markers():
    per_token_loss = torch.tensor([1.0, 2.0, 3.0, 4.0])
    targets = torch.tensor([20, 21, 22, 23])

    loss, metrics = streaming_stt_model._compute_weighted_lm_loss(
        per_token_loss=per_token_loss,
        flat_targets=targets,
        blank_id=23,
        has_blank=True,
        blank_loss_weight=1.0,
        soab_id=20,
        eoab_id=22,
        agent_backchannel_start_loss_weight=2.0,
        agent_backchannel_end_loss_weight=5.0,
    )

    expected = torch.tensor((1.0 * 2.0 + 2.0 + 3.0 * 5.0 + 4.0) / (2.0 + 1.0 + 5.0 + 1.0))
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["loss_soab"], torch.tensor(1.0))
    assert torch.allclose(metrics["loss_eoab"], torch.tensor(3.0))
    assert torch.allclose(metrics["soab_ratio"], torch.tensor(0.25))
    assert torch.allclose(metrics["eoab_ratio"], torch.tensor(0.25))


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
    assert not any("backchannel" in metric or "soab" in metric or "eoab" in metric for metric in model.logged)


def test_validation_epoch_end_logs_only_pooled_output_backchannel_metrics():
    model = _ValidationLogger()
    model.core_cfg = SimpleNamespace(
        enable_validation_checkpoint_score=False,
        enable_agent_backchannels=True,
    )
    streaming_stt_model.StreamingSTTModel.on_validation_epoch_start(model)
    first = {
        "num_samples": 2,
        "paired_backchannel_count": 3,
        "unpaired_soab_count": 1,
        "unpaired_eoab_count": 0,
        "hardcoded_backchannel_count": 2,
    }
    second = {
        "num_samples": 1,
        "paired_backchannel_count": 1,
        "unpaired_soab_count": 0,
        "unpaired_eoab_count": 2,
        "hardcoded_backchannel_count": 1,
    }
    for values in (first, second):
        for metric, value in values.items():
            model._partial_agent_backchannel_metrics[metric].append(torch.tensor(value))

    streaming_stt_model.StreamingSTTModel.on_validation_epoch_end(model)

    assert model.logged["val_paired_backchannel_tokens_per_utterance"].item() == pytest.approx(4 / 3)
    assert model.logged["val_unpaired_soab_tokens_per_utterance"].item() == pytest.approx(1 / 3)
    assert model.logged["val_unpaired_eoab_tokens_per_utterance"].item() == pytest.approx(2 / 3)
    assert model.logged["val_hardcoded_backchannel_rate"].item() == pytest.approx(3 / 4)
    assert not any("d7_complete" in metric or "d7_pause" in metric for metric in model.logged)


def test_validation_epoch_end_checkpoint_score_equal_weights_three_cohorts():
    model = _ValidationLogger()
    model.core_cfg = SimpleNamespace(enable_validation_checkpoint_score=True)
    streaming_stt_model.StreamingSTTModel.on_validation_epoch_start(model)

    # Token accuracy is collected for every loader, but boundary-aware loaders
    # contribute their SOU/EOU macro F1 to the checkpoint score instead.
    model._partial_accuracies["d7_complete"].append(torch.tensor(0.99))
    model._partial_accuracies["d7_pause"].append(torch.tensor(0.98))
    model._partial_accuracies["mcv"].append(torch.tensor(0.75))
    _append_boundary_totals(
        model,
        "d7_complete",
        sou_target_count=10,
        eou_target_count=10,
        sou_pred_count=10,
        eou_pred_count=10,
        sou_collar_hit=8,
        eou_collar_hit=6,
    )
    _append_boundary_totals(
        model,
        "d7_pause",
        sou_target_count=10,
        eou_target_count=10,
        sou_pred_count=20,
        eou_pred_count=10,
        sou_collar_hit=10,
        eou_collar_hit=10,
    )

    streaming_stt_model.StreamingSTTModel.on_validation_epoch_end(model)

    complete_macro_f1 = (0.8 + 0.6) / 2
    pause_macro_f1 = ((2 * 0.5 * 1.0 / (0.5 + 1.0)) + 1.0) / 2
    expected = (complete_macro_f1 + pause_macro_f1 + 0.75) / 3
    assert torch.isclose(model.logged["val_checkpoint_score"], torch.tensor(expected))
