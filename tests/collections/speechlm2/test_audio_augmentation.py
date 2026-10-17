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


@pytest.mark.parametrize("effect", ["lowpass", "room", "mic", "codec"])
def test_bfloat16_numpy_boundaries(tmp_path, effect):
    """Actual CPU DSP/I/O accepts BF16 and leaves shape/device/dtype usable."""
    import numpy as np
    import soundfile as sf

    sr = 16000
    audio = torch.sin(torch.arange(sr) * (2 * torch.pi * 440 / sr)).reshape(1, -1).to(torch.bfloat16)
    lengths = torch.tensor([sr])
    aug = AudioAugmenter(sr)
    ir = np.zeros(128, dtype=np.float32)
    ir[0], ir[24] = 1.0, 0.2
    sf.write(tmp_path / "ir.wav", ir, sr)
    if effect == "lowpass":
        result = aug._apply_lowpass_filter(audio[0], 1000).reshape_as(audio)
        assert result.dtype == torch.float32
    elif effect == "room":
        result = aug.add_room_ir_to_batch(audio, lengths, str(tmp_path), use_loudness_norm=False)
    elif effect == "mic":
        result = aug.add_mic_ir_to_batch(audio, lengths, str(tmp_path), use_loudness_norm=False)
    else:
        import shutil
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg is required for the codec integration test")
        # Call codec conversion directly too: add_codec_to_batch deliberately
        # catches codec failures, so output finiteness alone cannot prove I/O.
        decoded = aug._apply_ffmpeg_codec(audio[0].float().numpy(), ["-c:a", "pcm_mulaw", "-f", "wav"])
        assert decoded.size > 0 and np.isfinite(decoded).all()
        result = aug.add_codec_to_batch(audio, lengths, {"mulaw": ["-c:a", "pcm_mulaw", "-f", "wav"]})
    assert result.shape == audio.shape
    assert result.device == audio.device
    assert torch.isfinite(result).all()
    if effect != "lowpass":
        assert result.dtype == audio.dtype


@pytest.mark.parametrize('attenuation', [-15, 0, 24])
def test_speech_attenuation_sign_and_legacy_alias(tmp_path, attenuation):
    import numpy as np
    import soundfile as sf
    sf.write(tmp_path / 'zero.wav', np.zeros(100, dtype=np.float32), 100)
    augmenter = AudioAugmenter(sample_rate=100)
    kwargs = dict(noise_folder=str(tmp_path), noise_prob_scale_user=1.0, noise_prob_low_pass=0)
    audio = torch.full((1, 100), 0.01)
    new = augmenter.add_noise_to_batch(audio.clone(), **kwargs,
        speech_attenuation_min_db=attenuation, speech_attenuation_max_db=attenuation)
    old = augmenter.add_noise_to_batch(audio.clone(), **kwargs,
        noise_prob_scale_user_min_snr=attenuation, noise_prob_scale_user_max_snr=attenuation)
    assert torch.allclose(new, audio * 10 ** (-attenuation / 20), atol=1e-7)
    assert torch.equal(new, old)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_speech_span_snr_ignores_context_later_turn_and_padding(tmp_path, dtype):
    import numpy as np
    import soundfile as sf
    sf.write(tmp_path / 'noise.wav', np.full(1000, 0.1, dtype=np.float32), 100)
    augmenter = AudioAugmenter(sample_rate=100)
    audio = torch.zeros(2, 1000, dtype=dtype)
    audio[:, 100:200] = 0.25  # first turn
    audio[1, 400:800] = 0.9  # different later turn/context
    out = augmenter.add_noise_to_batch(audio.clone(), str(tmp_path), snr_db=20,
        noise_prob_scale_user=0, noise_prob_low_pass=0, speech_power_spans=[(100, 200)] * 2)
    noise = out.float() - audio.float()
    assert torch.equal(noise[0, 100:200], noise[1, 100:200])
    actual = 10 * torch.log10(audio[0, 100:200].float().square().mean() / noise[0, 100:200].square().mean())
    assert actual.item() == pytest.approx(20, abs=0.35 if dtype == torch.bfloat16 else 0.01)
    short = augmenter.add_noise_to_batch(audio[:1, :250].clone(), str(tmp_path), snr_db=20,
        noise_prob_scale_user=0, noise_prob_low_pass=0, speech_power_spans=[(100, 200)])
    assert torch.equal(short[0, 100:200], out[0, 100:200])


@pytest.mark.parametrize('span', [(10, 10), (-1, 20), (0, 101)])
def test_invalid_power_span_fails_before_loading_noise(span):
    with pytest.raises(ValueError, match='Speech power span'):
        AudioAugmenter().add_noise_to_batch(torch.zeros(1, 100), '/unused', speech_power_spans=[span])


@pytest.mark.parametrize('balanced', [False, True])
def test_noise_path_forwards_measurement_span_and_named_attenuation(monkeypatch, balanced):
    augmenter, _ = _recording_augmenter()
    monkeypatch.setattr('random.random', lambda: 0.0)
    cfg = _balanced_cfg(use_balanced_augmentation_controller=balanced, noise_prob=1.0,
        speech_attenuation_min_db=-6, speech_attenuation_max_db=12)
    audio, lengths = _audio()
    spans = [(3, 10), (4, 12)]
    augmenter.augment_batch(cfg, audio, lengths, speech_power_spans=spans)
    kwargs = augmenter.add_noise_to_batch.call_args.kwargs
    assert kwargs['speech_power_spans'] == spans
    assert kwargs['speech_attenuation_min_db'] == -6
    assert kwargs['speech_attenuation_max_db'] == 12


@pytest.mark.parametrize('mode,expected', [('independent_stack', ['noise', 'room', 'mic', 'codec']),
                                         ('noise_plus_at_most_one', ['noise', 'room'])])
def test_named_mode_overrides_legacy_flag(monkeypatch, mode, expected):
    augmenter, calls = _recording_augmenter()
    monkeypatch.setattr('random.random', lambda: 0.0)
    cfg = _balanced_cfg(augmentation_mode=mode,
        use_balanced_augmentation_controller=(mode == 'independent_stack'))
    augmenter.augment_batch(cfg, *_audio())
    assert calls == expected


def test_invalid_named_mode_rejected():
    with pytest.raises(ValueError, match='augmentation_mode'):
        AudioAugmenter().augment_batch({'augmentation_mode': 'typo'}, *_audio())
