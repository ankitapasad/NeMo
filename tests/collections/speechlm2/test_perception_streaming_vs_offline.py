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
Diagnostic test — offline vs cache-aware streaming perception embeddings.

Compares per-chunk embeddings between offline (full-audio) and streaming
(cache-aware, BatchedCacheFeatureBufferer) perception to pinpoint the
source of WER degradation in streaming mode.

Run with:
    cd /tmp && python -m pytest /path/to/test_perception_streaming_vs_offline.py -v -s -p no:conftest
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.speechlm2.parts.pretrained import setup_perception

# --- Constants matching the streaming STT eval config ---
PRETRAINED_ASR = "nvidia/nemotron-speech-streaming-en-0.6b"
ATT_CONTEXT_SIZE = [70, 1]
CHUNK_SIZE = 2  # SpeechLM chunk_size (number of audio embedding frames per chunk)
FRAME_LENGTH_IN_SECS = 0.08  # seconds per SpeechLM frame
SAMPLE_RATE = 16000
AUDIO_PAD_TO = 16
OUTPUT_DIM = 1536  # Qwen3-1.7B hidden_size
AUDIO_DURATION_SECS = 3.0


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def perception(device):
    """Load the pretrained perception module with streaming config."""
    cfg = OmegaConf.create(
        {
            "audio_pad_to": AUDIO_PAD_TO,
            "perception": {
                "modality_adapter": {
                    "_target_": "nemo.collections.speechlm2.modules.perception.IdentityConnector",
                },
            },
        }
    )
    p = setup_perception(
        cfg=cfg,
        output_dim=OUTPUT_DIM,
        pretrained_asr=PRETRAINED_ASR,
        pretrained_weights=True,
        audio_pad_to=AUDIO_PAD_TO,
        att_context_size=ATT_CONTEXT_SIZE,
    )
    p.encoder.setup_streaming_params()
    p.eval()
    p.to(device)
    return p


@pytest.fixture(scope="module")
def test_audio(device):
    """Generate deterministic ~3s sinusoidal test audio at 16kHz."""
    n_samples = int(AUDIO_DURATION_SECS * SAMPLE_RATE)
    t = torch.arange(n_samples, dtype=torch.float32, device=device)
    audio = 0.5 * torch.sin(2 * math.pi * 440 * t / SAMPLE_RATE) + 0.3 * torch.sin(2 * math.pi * 880 * t / SAMPLE_RATE)
    return audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_buffer_params(perception):
    """Derive feature buffer params exactly as get_audio_feature_buffer does."""
    streaming_cfg = perception.encoder.streaming_cfg
    preprocessor_cfg = perception.cfg.preprocessor
    window_stride = preprocessor_cfg.window_stride

    pre_encode_cache_size = streaming_cfg.pre_encode_cache_size
    if isinstance(pre_encode_cache_size, list):
        pre_encode_cache_size = pre_encode_cache_size[1]

    pre_encode_cache_secs = pre_encode_cache_size * window_stride
    chunk_secs = CHUNK_SIZE * FRAME_LENGTH_IN_SECS
    buffer_secs = pre_encode_cache_secs + chunk_secs
    chunk_samples = int(chunk_secs * SAMPLE_RATE)

    return preprocessor_cfg, buffer_secs, chunk_secs, chunk_samples


def _make_feature_buffer(preprocessor_cfg, buffer_secs, chunk_secs, device):
    return BatchedCacheFeatureBufferer(
        num_slots=1,
        sample_rate=SAMPLE_RATE,
        buffer_size_in_secs=buffer_secs,
        chunk_size_in_secs=chunk_secs,
        preprocessor_cfg=preprocessor_cfg,
        device=device,
    )


# ---------------------------------------------------------------------------
# Test 1 — streaming config alignment (hypothesis #1)
# ---------------------------------------------------------------------------


class TestStreamingConfigAlignment:

    def test_streaming_config_alignment(self, perception):
        streaming_cfg = perception.encoder.streaming_cfg

        print("\n=== Streaming Config ===")
        print(f"  chunk_size              = {streaming_cfg.chunk_size}")
        print(f"  shift_size              = {streaming_cfg.shift_size}")
        print(f"  valid_out_len           = {streaming_cfg.valid_out_len}")
        print(f"  pre_encode_cache_size   = {streaming_cfg.pre_encode_cache_size}")
        print(f"  drop_extra_pre_encoded  = {streaming_cfg.drop_extra_pre_encoded}")
        print(f"  last_channel_cache_size = {streaming_cfg.last_channel_cache_size}")
        print(f"  cache_drop_size         = {streaming_cfg.cache_drop_size}")
        print(f"=== SpeechLM Config ===")
        print(f"  chunk_size              = {CHUNK_SIZE}")
        print(f"  frame_length_in_secs    = {FRAME_LENGTH_IN_SECS}")

        # Hypothesis #1: valid_out_len must equal SpeechLM chunk_size
        assert streaming_cfg.valid_out_len == CHUNK_SIZE, (
            f"valid_out_len ({streaming_cfg.valid_out_len}) != " f"SpeechLM chunk_size ({CHUNK_SIZE})"
        )


# ---------------------------------------------------------------------------
# Test 2 — mel features streaming vs offline (hypothesis #4)
# ---------------------------------------------------------------------------


class TestMelFeaturesStreamingVsOffline:

    def test_mel_features_streaming_vs_offline(self, perception, device, test_audio):
        preprocessor_cfg, buffer_secs, chunk_secs, chunk_samples = _get_buffer_params(perception)
        audio_1d = test_audio
        audio_len = audio_1d.shape[0]

        print(f"\n=== Mel Feature Test ===")
        print(f"  buffer_size_in_secs  = {buffer_secs}")
        print(f"  chunk_size_in_secs   = {chunk_secs}")
        print(f"  chunk_samples        = {chunk_samples}")

        # --- Offline: full-audio mel ---
        with torch.no_grad():
            offline_mel, offline_mel_len = perception.preprocessor(
                input_signal=audio_1d.unsqueeze(0),
                length=torch.tensor([audio_len], device=device),
            )
        print(f"  offline mel shape    = {offline_mel.shape}, mel_len = {offline_mel_len}")

        # --- Streaming: per-chunk mel via BatchedCacheFeatureBufferer ---
        feature_buffer = _make_feature_buffer(preprocessor_cfg, buffer_secs, chunk_secs, device)
        print(f"  feature_buffer_len   = {feature_buffer.feature_buffer_len}")
        print(f"  feature_chunk_len    = {feature_buffer.feature_chunk_len}")

        num_chunks = math.ceil(audio_len / chunk_samples)
        streaming_mel_chunks = []

        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, audio_len)
            chunk_wav = audio_1d[start:end]

            frame = Frame(samples=chunk_wav, length=end - start, stream_id=0)
            features, right_paddings = feature_buffer.update([frame])

            fb = features[0]  # (D, feature_buffer_len)
            chunk_mel = fb[:, -feature_buffer.feature_chunk_len :]
            streaming_mel_chunks.append(chunk_mel)

            print(
                f"  Chunk {i} [{start}:{end}]: fb_shape={fb.shape}, "
                f"chunk_mel_shape={chunk_mel.shape}, right_padding={right_paddings[0]}"
            )

        # --- Compare: slice offline mel into chunks ---
        fcl = feature_buffer.feature_chunk_len
        offline_frames = offline_mel.shape[-1]
        print(f"\n  Per-chunk mel comparison:")

        for i, s_chunk in enumerate(streaming_mel_chunks):
            off_start = i * fcl
            off_end = min(off_start + fcl, offline_frames)
            if off_start >= offline_frames:
                print(f"  Chunk {i}: offline frames exhausted at {offline_frames}")
                break
            o_chunk = offline_mel[0, :, off_start:off_end]

            min_len = min(s_chunk.shape[-1], o_chunk.shape[-1])
            s, o = s_chunk[:, :min_len], o_chunk[:, :min_len]

            mse = torch.mean((s - o) ** 2).item()
            max_diff = torch.max(torch.abs(s - o)).item()
            print(
                f"  Chunk {i}: MSE={mse:.6f}, MaxDiff={max_diff:.6f}, "
                f"streaming={s_chunk.shape}, offline={o_chunk.shape}"
            )


# ---------------------------------------------------------------------------
# Test 3 — full perception embeddings streaming vs offline (core test)
# ---------------------------------------------------------------------------


class TestPerceptionEmbeddingsStreamingVsOffline:

    def test_perception_embeddings_streaming_vs_offline(self, perception, device, test_audio):
        preprocessor_cfg, buffer_secs, chunk_secs, chunk_samples = _get_buffer_params(perception)
        audio_1d = test_audio
        audio_len = audio_1d.shape[0]
        num_chunks = math.ceil(audio_len / chunk_samples)

        print(f"\n=== Perception Embeddings Test ===")
        print(f"  chunk_samples = {chunk_samples}, num_chunks = {num_chunks}, audio_len = {audio_len}")

        # --- Method A: Offline ---
        with torch.no_grad():
            offline_embs, offline_emb_lens = perception(
                input_signal=audio_1d.unsqueeze(0),
                input_signal_length=torch.tensor([audio_len], device=device),
            )
        print(f"  Offline embs: shape={offline_embs.shape}, lens={offline_emb_lens}")

        # --- Method B: Streaming with cache ---
        cache_last_channel, cache_last_time, cache_last_channel_len = perception.get_initial_cache_state(
            batch_size=1, device=device
        )

        feature_buffer = _make_feature_buffer(preprocessor_cfg, buffer_secs, chunk_secs, device)

        streaming_chunks = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, audio_len)
            chunk_wav = audio_1d[start:end]

            frame = Frame(samples=chunk_wav, length=end - start, stream_id=0)
            features, right_paddings = feature_buffer.update([frame])

            processed_signal = features[0].unsqueeze(0)
            processed_signal_length = torch.tensor(
                [processed_signal.shape[-1] - int(right_paddings[0])], device=device
            ).long()

            with torch.no_grad():
                outputs = perception(
                    processed_signal=processed_signal,
                    processed_signal_length=processed_signal_length,
                    cache_last_channel=cache_last_channel,
                    cache_last_time=cache_last_time,
                    cache_last_channel_len=cache_last_channel_len,
                    streaming=True,
                )
            embs, emb_lens, new_cache = outputs

            print(
                f"  Chunk {i} [{start}:{end}]: "
                f"mel_shape={processed_signal.shape}, proc_sig_len={processed_signal_length.item()}, "
                f"emb_shape={embs.shape}, emb_lens={emb_lens}"
            )

            # Update cache
            if new_cache is not None:
                cache_last_channel = new_cache["cache_last_channel"]
                cache_last_time = new_cache["cache_last_time"]
                cache_last_channel_len = new_cache["cache_last_channel_len"]

            streaming_chunks.append(embs.squeeze(0))  # (F, H)

        # --- Concatenate and compare ---
        streaming_all = torch.cat(streaming_chunks, dim=0)  # (T_streaming, H)
        offline_all = offline_embs.squeeze(0)  # (T_offline, H)

        print(f"\n  Total frames: offline={offline_all.shape[0]}, streaming={streaming_all.shape[0]}")

        compare_len = min(streaming_all.shape[0], offline_all.shape[0])
        chunk_frames = CHUNK_SIZE

        print(f"  Per-chunk comparison:")
        n_compare_chunks = compare_len // chunk_frames
        for i in range(n_compare_chunks):
            cs = i * chunk_frames
            ce = cs + chunk_frames
            s_chunk = streaming_all[cs:ce]
            o_chunk = offline_all[cs:ce]

            mse = torch.mean((s_chunk - o_chunk) ** 2).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                s_chunk.flatten().unsqueeze(0),
                o_chunk.flatten().unsqueeze(0),
            ).item()
            max_diff = torch.max(torch.abs(s_chunk - o_chunk)).item()

            print(f"  Chunk {i} [{cs}:{ce}]: MSE={mse:.6f}, CosSim={cos_sim:.6f}, MaxDiff={max_diff:.6f}")

        # Overall metrics
        s_all = streaming_all[:compare_len]
        o_all = offline_all[:compare_len]
        overall_mse = torch.mean((s_all - o_all) ** 2).item()
        overall_cos = torch.nn.functional.cosine_similarity(
            s_all.flatten().unsqueeze(0),
            o_all.flatten().unsqueeze(0),
        ).item()
        overall_max = torch.max(torch.abs(s_all - o_all)).item()

        print(
            f"\n  Overall [{compare_len} frames]: "
            f"MSE={overall_mse:.6f}, CosSim={overall_cos:.6f}, MaxDiff={overall_max:.6f}"
        )

        # --- Frame alignment sweep ---
        # Check if streaming frames are shifted relative to offline.
        # Compare streaming_all against offline_all[offset:] for offsets -2..+2.
        print(f"\n  Frame alignment sweep (per-frame avg CosSim at different offsets):")
        for offset in range(-2, 3):
            if offset >= 0:
                s = streaming_all[: min(streaming_all.shape[0], offline_all.shape[0] - offset)]
                o = offline_all[offset : offset + s.shape[0]]
            else:
                o = offline_all[: min(offline_all.shape[0], streaming_all.shape[0] + offset)]
                s = streaming_all[-offset : -offset + o.shape[0]]
            n = min(s.shape[0], o.shape[0])
            if n == 0:
                continue
            s, o = s[:n], o[:n]
            per_frame_cos = torch.nn.functional.cosine_similarity(s, o, dim=1)  # (n,)
            print(
                f"    offset={offset:+d}: mean_CosSim={per_frame_cos.mean().item():.6f}, "
                f"min={per_frame_cos.min().item():.6f}, frames={n}"
            )
