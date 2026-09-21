# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from unittest.mock import Mock

import pytest
import torch

from nemo.collections.speechlm2.parts.augmentation import AudioAugmenter


def _balanced_cfg(**overrides):
    cfg = {
        "use_balanced_augmentation_controller": True,
        "use_noise_aug": True,
        "noise_prob": 0.5,
        "noise_aug_path": "/noise",
        "noise_min_snr": 20,
        "noise_max_snr": 50,
        "noise_prob_scale_user": 0.3,
        "noise_prob_scale_user_min_snr": -15,
        "noise_prob_scale_user_max_snr": 24,
        "noise_resample": True,
        "noise_prob_low_pass": 0.05,
        "use_room_ir_aug": True,
        "roomir_prob": 0.5,
        "roomir_aug_path": "/roomir",
        "roomir_use_loudness_norm": True,
        "use_mic_ir_aug": True,
        "micir_prob": 0.3,
        "micir_aug_path": "/micir",
        "micir_use_loudness_norm": True,
        "use_codec_aug": True,
        "codec_prob": 0.1,
        "codec_settings": {"test": []},
    }
    cfg.update(overrides)
    return cfg


def _recording_augmenter():
    augmenter = AudioAugmenter(sample_rate=16000)
    calls = []

    def record(name):
        def apply(audio, *_args, **_kwargs):
            calls.append(name)
            return audio

        return apply

    augmenter.add_noise_to_batch = Mock(side_effect=record("noise"))
    augmenter.add_room_ir_to_batch = Mock(side_effect=record("room"))
    augmenter.add_mic_ir_to_batch = Mock(side_effect=record("mic"))
    augmenter.add_codec_to_batch = Mock(side_effect=record("codec"))
    return augmenter, calls


def _audio():
    return torch.zeros(2, 32), torch.tensor([32, 24])


def test_balanced_controller_returns_clean_batch_above_top_level_gate(monkeypatch):
    augmenter, calls = _recording_augmenter()
    audio, lengths = _audio()
    original = audio.clone()
    monkeypatch.setattr("random.random", lambda: 0.75)

    result = augmenter.augment_batch(_balanced_cfg(), audio, lengths)

    assert calls == []
    assert torch.equal(result, original)
    assert result.data_ptr() == audio.data_ptr()


@pytest.mark.parametrize(
    ("extra_draw", "expected_calls"),
    [
        (0.00, ["noise", "room"]),
        (0.49, ["noise", "room"]),
        (0.50, ["noise", "mic"]),
        (0.79, ["noise", "mic"]),
        (0.80, ["noise", "codec"]),
        (0.89, ["noise", "codec"]),
        (0.90, ["noise"]),
        (0.99, ["noise"]),
    ],
)
def test_balanced_controller_adds_noise_and_at_most_one_extra(monkeypatch, extra_draw, expected_calls):
    augmenter, calls = _recording_augmenter()
    audio, lengths = _audio()
    draws = iter([0.25, extra_draw])
    monkeypatch.setattr("random.random", lambda: next(draws))
    monkeypatch.setattr("random.randint", lambda _low, _high: 35)

    augmenter.augment_batch(_balanced_cfg(), audio, lengths)

    assert calls == expected_calls
    assert sum(name in calls for name in ("room", "mic", "codec")) <= 1
    augmenter.add_noise_to_batch.assert_called_once()
    noise_kwargs = augmenter.add_noise_to_batch.call_args.kwargs
    assert noise_kwargs["snr_db"] == 35
    assert noise_kwargs["noise_prob_scale_user_min_snr"] == -15
    assert noise_kwargs["noise_prob_scale_user_max_snr"] == 24
    assert noise_kwargs["noise_prob_low_pass"] == pytest.approx(0.05)


def test_balanced_controller_rejects_overfull_extra_distribution(monkeypatch):
    augmenter, _calls = _recording_augmenter()
    audio, lengths = _audio()
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.randint", lambda _low, _high: 35)

    with pytest.raises(ValueError, match="must be <= 1"):
        augmenter.augment_batch(
            _balanced_cfg(roomir_prob=0.6, micir_prob=0.3, codec_prob=0.2), audio, lengths
        )


def test_independent_controller_remains_backward_compatible(monkeypatch):
    augmenter, calls = _recording_augmenter()
    audio, lengths = _audio()
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.randint", lambda _low, _high: 35)
    cfg = _balanced_cfg(
        use_balanced_augmentation_controller=False,
        noise_prob=1.0,
        roomir_prob=1.0,
        micir_prob=1.0,
        codec_prob=1.0,
    )

    augmenter.augment_batch(cfg, audio, lengths)

    assert calls == ["noise", "room", "mic", "codec"]
