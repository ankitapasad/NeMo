# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Production-format tarred-audio gate for streaming STT validation."""

from __future__ import annotations

import io
import json
import tarfile
import wave

import numpy as np

from nemo.collections.common.data.lhotse.nemo_adapters import LazyNeMoTarredIterator


def _pcm16_wav(samples: np.ndarray, sampling_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sampling_rate)
        writer.writeframes(samples.astype("<i2").tobytes())
    return output.getvalue()


def test_production_nemo_tarred_iterator_loads_exact_offset_slice(tmp_path, monkeypatch):
    """Use the sequential NeMo tar path and the same offset semantics as training."""
    monkeypatch.delenv("USE_AIS_GET_BATCH", raising=False)
    sampling_rate = 8000
    source = np.arange(3 * sampling_rate, dtype=np.int16) - 12000
    audio_bytes = _pcm16_wav(source, sampling_rate)

    tar_path = tmp_path / "audio_0.tar"
    with tarfile.open(tar_path, "w") as archive:
        member = tarfile.TarInfo("shared.wav")
        member.size = len(audio_bytes)
        archive.addfile(member, io.BytesIO(audio_bytes))

    manifest_path = tmp_path / "manifest_0.json"
    manifest_path.write_text(
        json.dumps(
            {
                "audio_filepath": "shared-sub000000.wav",
                "duration": 1.0,
                "offset": 0.0,
                "sampling_rate": sampling_rate,
                "text": "policy-excluded slice",
                "clip_id": "clip-skipme",
                "shard_id": 0,
                "_skipme": True,
                "skip_reason": "policy-excluded",
            }
        )
        + "\n"
        + json.dumps(
            {
                # Production manifests expose logical offset slices using the
                # ``-subNNNNNN`` alias while the tar contains one shared WAV.
                "audio_filepath": "shared-sub000001.wav",
                "duration": 1.0,
                "offset": 1.0,
                "sampling_rate": sampling_rate,
                "text": "offset slice",
                "clip_id": "clip-offset-1",
                "shard_id": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cuts = list(
        LazyNeMoTarredIterator(
            manifest_path=manifest_path,
            tar_paths=tar_path,
            shuffle_shards=False,
            shard_seed=0,
        )
    )

    assert len(cuts) == 1
    cut = cuts[0]
    assert cut.custom["clip_id"] == "clip-offset-1"
    assert cut.duration == 1.0
    assert cut.tar_origin == str(tar_path)
    loaded = cut.load_audio()[0]
    expected = source[sampling_rate : 2 * sampling_rate].astype(np.float32) / 32768.0
    np.testing.assert_allclose(loaded, expected, rtol=0.0, atol=1.0 / 32768.0)
