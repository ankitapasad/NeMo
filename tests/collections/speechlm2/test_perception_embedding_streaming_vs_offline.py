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
Diagnostic test — perception embedding equivalence: streaming vs offline.

Measures how the mel-feature differences from BatchedCacheFeatureBufferer propagate
through the full perception module (preprocessor → ConformerEncoder → proj) and
affect the final embeddings that the LLM receives.

Three comparisons isolate separate error sources:

  A) "streaming mel → streaming encoder" vs "offline (full-audio)"
     = total streaming-offline gap (what the LLM actually sees)

  B) "offline mel slices → streaming encoder" vs "offline (full-audio)"
     = encoder caching error alone (mel is perfect, only caching introduces error)

  C) Comparison of (A) vs (B) per-frame
     = the additional degradation caused by mel-feature differences on top of
       encoder caching

Run with:
    cd /tmp && python -m pytest /path/to/test_perception_embedding_streaming_vs_offline.py -v -s -p no:conftest
"""

import math

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.constants import LOG_MEL_ZERO
from nemo.collections.speechlm2.parts.pretrained import setup_perception

# --- Constants matching the streaming STT eval config ---
PRETRAINED_ASR = "nvidia/nemotron-speech-streaming-en-0.6b"
ATT_CONTEXT_SIZE = [70, 1]
CHUNK_SIZE = 2  # SpeechLM chunk_size (number of audio embedding frames per chunk)
FRAME_LENGTH_IN_SECS = 0.08  # seconds per SpeechLM frame
SAMPLE_RATE = 16000
AUDIO_PAD_TO = 16
OUTPUT_DIM = 1536  # Qwen3-1.7B hidden_size


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


def _generate_test_audio(duration_secs, device):
    """Generate deterministic sinusoidal test audio at 16kHz."""
    n_samples = int(duration_secs * SAMPLE_RATE)
    t = torch.arange(n_samples, dtype=torch.float32, device=device)
    audio = 0.5 * torch.sin(2 * math.pi * 440 * t / SAMPLE_RATE) + 0.3 * torch.sin(2 * math.pi * 880 * t / SAMPLE_RATE)
    return audio


def _run_streaming_perception(perception, feature_buffers, right_paddings_list, device):
    """Run streaming perception on pre-computed feature buffers.

    Args:
        feature_buffers: list of (D, fbl) tensors — the full mel buffer per chunk
        right_paddings_list: list of int — right padding per chunk
    Returns:
        list of (valid_out_len, H) embedding tensors, one per chunk
    """
    cache_last_channel, cache_last_time, cache_last_channel_len = perception.get_initial_cache_state(
        batch_size=1, device=device
    )
    streaming_chunks = []
    for fb, rp in zip(feature_buffers, right_paddings_list):
        processed_signal = fb.unsqueeze(0)  # (1, D, fbl)
        processed_signal_length = torch.tensor([processed_signal.shape[-1] - rp], device=device).long()

        with torch.no_grad():
            embs, emb_lens, new_cache = perception(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=cache_last_channel,
                cache_last_time=cache_last_time,
                cache_last_channel_len=cache_last_channel_len,
                streaming=True,
            )
        if new_cache is not None:
            cache_last_channel = new_cache["cache_last_channel"]
            cache_last_time = new_cache["cache_last_time"]
            cache_last_channel_len = new_cache["cache_last_channel_len"]
        streaming_chunks.append(embs.squeeze(0))  # (valid_out_len, H)

    return streaming_chunks


def _per_frame_metrics(streaming_all, offline_all, compare_len):
    """Compute per-frame MSE, cosine similarity, L2 distance, and relative L2.

    Relative L2 = ||streaming - offline||_2 / ||offline||_2
    This measures the L2 distance as a fraction of the offline embedding's magnitude,
    giving a scale-independent error metric. Unlike cosine similarity, it captures
    both directional and magnitude differences. A relative L2 of 0.25 means the
    error vector is 25% the size of the reference embedding.
    """
    mses, cos_sims, max_diffs, l2_dists, rel_l2s = [], [], [], [], []
    for i in range(compare_len):
        s = streaming_all[i]
        o = offline_all[i]
        diff = s - o
        mse = torch.mean(diff**2).item()
        cos = F.cosine_similarity(s.unsqueeze(0), o.unsqueeze(0)).item()
        mad = torch.max(torch.abs(diff)).item()
        l2 = torch.norm(diff).item()
        o_norm = torch.norm(o).item()
        rl2 = l2 / o_norm if o_norm > 0 else float('inf')
        mses.append(mse)
        cos_sims.append(cos)
        max_diffs.append(mad)
        l2_dists.append(l2)
        rel_l2s.append(rl2)
    return mses, cos_sims, max_diffs, l2_dists, rel_l2s


def _print_per_chunk_table(label, mses, cos_sims, max_diffs, l2_dists, rel_l2s, frames_per_chunk, compare_len):
    """Print a per-chunk summary table."""
    num_chunks = math.ceil(compare_len / frames_per_chunk)
    max_print = 8  # first N + last 2
    print(f"\n  {label} — per-chunk summary (frames_per_chunk={frames_per_chunk}):")
    print(
        f"  {'Chunk':>6} {'Frames':>12} {'mean_CosSim':>12} {'min_CosSim':>12} "
        f"{'mean_RelL2':>11} {'max_RelL2':>10} {'mean_L2':>10} {'max_L2':>10}"
    )
    for i in range(num_chunks):
        cs = i * frames_per_chunk
        ce = min(cs + frames_per_chunk, compare_len)
        if cs >= compare_len:
            break
        chunk_cos = cos_sims[cs:ce]
        chunk_rl2 = rel_l2s[cs:ce]
        chunk_l2 = l2_dists[cs:ce]

        should_print = (i < max_print) or (i >= num_chunks - 2)
        if i == max_print and num_chunks > max_print + 2:
            print(f"  {'...':>6}")
        if should_print:
            print(
                f"  {i:>6} [{cs:>4}:{ce:<4}] "
                f"{sum(chunk_cos)/len(chunk_cos):>12.6f} {min(chunk_cos):>12.6f} "
                f"{sum(chunk_rl2)/len(chunk_rl2):>11.4f} {max(chunk_rl2):>10.4f} "
                f"{sum(chunk_l2)/len(chunk_l2):>10.4f} {max(chunk_l2):>10.4f}"
            )


def _print_overall(label, mses, cos_sims, max_diffs, l2_dists, rel_l2s, compare_len):
    """Print overall summary stats."""
    mse_t = torch.tensor(mses)
    cos_t = torch.tensor(cos_sims)
    mad_t = torch.tensor(max_diffs)
    l2_t = torch.tensor(l2_dists)
    rl2_t = torch.tensor(rel_l2s)
    print(f"\n  {label} — overall ({compare_len} frames):")
    print(
        f"    CosSim:  mean={cos_t.mean().item():.4f}, median={cos_t.median().item():.4f}, "
        f"min={cos_t.min().item():.4f}"
    )
    print(
        f"    RelL2:   mean={rl2_t.mean().item():.4f}, median={rl2_t.median().item():.4f}, "
        f"max={rl2_t.max().item():.4f}"
    )
    print(
        f"    L2:      mean={l2_t.mean().item():.4f}, median={l2_t.median().item():.4f}, "
        f"max={l2_t.max().item():.4f}"
    )
    print(
        f"    MSE:     mean={mse_t.mean().item():.6f}, median={mse_t.median().item():.6f}, "
        f"max={mse_t.max().item():.6f}"
    )
    return mse_t, cos_t, mad_t, l2_t, rl2_t


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestPerceptionEmbeddingStreamingVsOffline:

    @pytest.mark.parametrize("audio_duration", [3.0, 10.0])
    def test_perception_embedding_streaming_vs_offline(self, perception, device, audio_duration):
        """Compare perception embeddings: streaming vs offline, and isolate error sources."""
        preprocessor_cfg, buffer_secs, chunk_secs, chunk_samples = _get_buffer_params(perception)
        audio_1d = _generate_test_audio(audio_duration, device)
        audio_len = audio_1d.shape[0]
        num_chunks = math.ceil(audio_len / chunk_samples)

        fbl = int(buffer_secs / preprocessor_cfg.window_stride)
        fcl = int(chunk_secs / preprocessor_cfg.window_stride)
        n_cache = fbl - fcl

        print(f"\n{'=' * 70}")
        print(
            f"Audio: {audio_len} samples, {audio_len / SAMPLE_RATE:.2f}s, "
            f"{num_chunks} chunks of {chunk_samples} samples"
        )
        print(f"Feature buffer: fbl={fbl}, fcl={fcl}, cache={n_cache}")
        print(f"Encoder valid_out_len={perception.encoder.streaming_cfg.valid_out_len} " f"(frames per chunk)")

        # ==================================================================
        # Step 1: Offline — full-audio perception
        # ==================================================================
        with torch.no_grad():
            offline_embs, offline_emb_lens = perception(
                input_signal=audio_1d.unsqueeze(0),
                input_signal_length=torch.tensor([audio_len], device=device),
            )
        offline_all = offline_embs.squeeze(0)  # (T_offline, H)
        print(
            f"\nOffline perception: embs shape={offline_embs.shape}, "
            f"lens={offline_emb_lens.item()}, H={offline_all.shape[-1]}"
        )

        # Also get the offline mel for experiment B
        with torch.no_grad():
            offline_mel, offline_mel_len = perception.preprocessor(
                input_signal=audio_1d.unsqueeze(0),
                length=torch.tensor([audio_len], device=device),
            )
        valid_mel_len = offline_mel_len[0].item()
        print(f"Offline mel: shape={offline_mel.shape}, valid_len={valid_mel_len}")

        # ==================================================================
        # Step 2A: Streaming with real feature buffer (streaming mel)
        # ==================================================================
        feature_buffer = _make_feature_buffer(preprocessor_cfg, buffer_secs, chunk_secs, device)

        streaming_mel_buffers = []
        streaming_right_paddings = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, audio_len)
            chunk_wav = audio_1d[start:end]
            frame = Frame(samples=chunk_wav, length=end - start, stream_id=0)
            features, right_paddings = feature_buffer.update([frame])
            streaming_mel_buffers.append(features[0].clone())  # (D, fbl)
            streaming_right_paddings.append(int(right_paddings[0]))

        streaming_chunks_A = _run_streaming_perception(
            perception, streaming_mel_buffers, streaming_right_paddings, device
        )
        streaming_A = torch.cat(streaming_chunks_A, dim=0)  # (T_stream, H)
        print(f"\n[A] Streaming (real mel): {streaming_A.shape[0]} frames")

        # ==================================================================
        # Step 2B: Streaming with offline mel slices (perfect mel)
        # ==================================================================
        # Build feature buffers from offline mel, mimicking the cache structure:
        # buffer_i = [offline_mel[:, max(0, i*fcl-n_cache) : (i+1)*fcl]]
        # left-padded with LOG_MEL_ZERO if i*fcl < n_cache
        offline_mel_buffers = []
        offline_mel_right_paddings = []
        for i in range(num_chunks):
            off_start = max(0, i * fcl - n_cache)
            off_end = min((i + 1) * fcl, valid_mel_len)
            if off_start >= valid_mel_len:
                # past the end — fill with zeros
                buf = torch.full(
                    (offline_mel.shape[1], fbl),
                    LOG_MEL_ZERO,
                    device=device,
                    dtype=offline_mel.dtype,
                )
            else:
                buf_slice = offline_mel[0, :, off_start:off_end]
                pad_left = fbl - buf_slice.shape[-1]
                if pad_left > 0:
                    buf = torch.cat(
                        [
                            torch.full(
                                (buf_slice.shape[0], pad_left), LOG_MEL_ZERO, device=device, dtype=buf_slice.dtype
                            ),
                            buf_slice,
                        ],
                        dim=-1,
                    )
                else:
                    buf = buf_slice
            offline_mel_buffers.append(buf)
            # For non-last chunks, right_padding=0; for last chunk, compute based on valid mel
            rp = max(0, (i + 1) * fcl - valid_mel_len)
            offline_mel_right_paddings.append(rp)

        streaming_chunks_B = _run_streaming_perception(
            perception, offline_mel_buffers, offline_mel_right_paddings, device
        )
        streaming_B = torch.cat(streaming_chunks_B, dim=0)  # (T_stream, H)
        print(f"[B] Streaming (offline mel slices): {streaming_B.shape[0]} frames")

        # ==================================================================
        # Step 3: Per-frame comparison
        # ==================================================================
        compare_len = min(streaming_A.shape[0], streaming_B.shape[0], offline_all.shape[0])
        # Exclude last chunk from the main comparison (partial audio)
        last_chunk_start_frame = (num_chunks - 1) * CHUNK_SIZE
        compare_interior = min(last_chunk_start_frame, compare_len)
        frames_per_chunk = CHUNK_SIZE

        print(f"\nComparing {compare_len} frames total, {compare_interior} excluding last chunk")

        # --- Comparison A: streaming mel + streaming encoder vs offline ---
        mses_A, cos_A, mad_A, l2_A, rl2_A = _per_frame_metrics(streaming_A, offline_all, compare_len)
        _print_per_chunk_table(
            "[A] streaming_mel + streaming_enc vs offline",
            mses_A,
            cos_A,
            mad_A,
            l2_A,
            rl2_A,
            frames_per_chunk,
            compare_len,
        )
        mse_t_A, cos_t_A, mad_t_A, l2_t_A, rl2_t_A = _print_overall(
            "[A] streaming_mel + streaming_enc vs offline", mses_A, cos_A, mad_A, l2_A, rl2_A, compare_len
        )

        # --- Comparison B: offline mel + streaming encoder vs offline ---
        mses_B, cos_B, mad_B, l2_B, rl2_B = _per_frame_metrics(streaming_B, offline_all, compare_len)
        _print_per_chunk_table(
            "[B] offline_mel + streaming_enc vs offline",
            mses_B,
            cos_B,
            mad_B,
            l2_B,
            rl2_B,
            frames_per_chunk,
            compare_len,
        )
        mse_t_B, cos_t_B, mad_t_B, l2_t_B, rl2_t_B = _print_overall(
            "[B] offline_mel + streaming_enc vs offline", mses_B, cos_B, mad_B, l2_B, rl2_B, compare_len
        )

        # --- Comparison C: streaming mel vs offline mel through streaming encoder ---
        mses_C, cos_C, mad_C, l2_C, rl2_C = _per_frame_metrics(streaming_A, streaming_B, compare_len)
        _print_per_chunk_table(
            "[C] streaming_mel vs offline_mel (both through streaming enc)",
            mses_C,
            cos_C,
            mad_C,
            l2_C,
            rl2_C,
            frames_per_chunk,
            compare_len,
        )
        mse_t_C, cos_t_C, mad_t_C, l2_t_C, rl2_t_C = _print_overall(
            "[C] streaming_mel vs offline_mel (both through streaming enc)",
            mses_C,
            cos_C,
            mad_C,
            l2_C,
            rl2_C,
            compare_len,
        )

        # ==================================================================
        # Step 4: Error attribution summary
        # ==================================================================
        print(f"\n{'=' * 70}")
        print(f"ERROR ATTRIBUTION SUMMARY (excluding last chunk, {compare_interior} frames)")

        if compare_interior > 0:
            # Recompute on interior only
            mses_Ai = mses_A[:compare_interior]
            mses_Bi = mses_B[:compare_interior]
            mses_Ci = mses_C[:compare_interior]
            cos_Ai = cos_A[:compare_interior]
            cos_Bi = cos_B[:compare_interior]
            cos_Ci = cos_C[:compare_interior]
            rl2_Ai = rl2_A[:compare_interior]
            rl2_Bi = rl2_B[:compare_interior]
            rl2_Ci = rl2_C[:compare_interior]

            mean_mse_A = sum(mses_Ai) / len(mses_Ai)
            mean_mse_B = sum(mses_Bi) / len(mses_Bi)
            mean_mse_C = sum(mses_Ci) / len(mses_Ci)
            mean_cos_A = sum(cos_Ai) / len(cos_Ai)
            mean_cos_B = sum(cos_Bi) / len(cos_Bi)
            mean_cos_C = sum(cos_Ci) / len(cos_Ci)
            mean_rl2_A = sum(rl2_Ai) / len(rl2_Ai)
            mean_rl2_B = sum(rl2_Bi) / len(rl2_Bi)
            mean_rl2_C = sum(rl2_Ci) / len(rl2_Ci)
            rl2_t_Ai = torch.tensor(rl2_Ai)
            rl2_t_Bi = torch.tensor(rl2_Bi)
            rl2_t_Ci = torch.tensor(rl2_Ci)

            print(f"\n  [A] Total streaming-offline gap:")
            print(
                f"      mean CosSim={mean_cos_A:.4f}, mean RelL2={mean_rl2_A:.4f}, "
                f"median RelL2={rl2_t_Ai.median().item():.4f}, max RelL2={rl2_t_Ai.max().item():.4f}"
            )
            print(f"  [B] Encoder caching alone (perfect mel):")
            print(
                f"      mean CosSim={mean_cos_B:.4f}, mean RelL2={mean_rl2_B:.4f}, "
                f"median RelL2={rl2_t_Bi.median().item():.4f}, max RelL2={rl2_t_Bi.max().item():.4f}"
            )
            print(f"  [C] Mel difference impact (streaming vs offline mel, same encoder):")
            print(
                f"      mean CosSim={mean_cos_C:.4f}, mean RelL2={mean_rl2_C:.4f}, "
                f"median RelL2={rl2_t_Ci.median().item():.4f}, max RelL2={rl2_t_Ci.max().item():.4f}"
            )

        # ==================================================================
        # Step 5: Frame alignment sweep
        # ==================================================================
        print(f"\n  Frame alignment sweep (streaming_A vs offline, per-frame avg CosSim):")
        for offset in range(-2, 3):
            if offset >= 0:
                s = streaming_A[: min(streaming_A.shape[0], offline_all.shape[0] - offset)]
                o = offline_all[offset : offset + s.shape[0]]
            else:
                o = offline_all[: min(offline_all.shape[0], streaming_A.shape[0] + offset)]
                s = streaming_A[-offset : -offset + o.shape[0]]
            n = min(s.shape[0], o.shape[0])
            if n == 0:
                continue
            s, o = s[:n], o[:n]
            per_frame_cos = F.cosine_similarity(s, o, dim=1)
            print(
                f"    offset={offset:+d}: mean_CosSim={per_frame_cos.mean().item():.6f}, "
                f"min={per_frame_cos.min().item():.6f}, frames={n}"
            )

        # ==================================================================
        # Step 6: Embedding magnitude analysis
        # ==================================================================
        print(f"\n  Embedding magnitude analysis:")
        offline_norms = torch.norm(offline_all[:compare_interior], dim=-1)
        stream_A_norms = torch.norm(streaming_A[:compare_interior], dim=-1)
        stream_B_norms = torch.norm(streaming_B[:compare_interior], dim=-1)
        print(
            f"    Offline    L2 norm: mean={offline_norms.mean().item():.4f}, " f"std={offline_norms.std().item():.4f}"
        )
        print(
            f"    Stream [A] L2 norm: mean={stream_A_norms.mean().item():.4f}, "
            f"std={stream_A_norms.std().item():.4f}"
        )
        print(
            f"    Stream [B] L2 norm: mean={stream_B_norms.mean().item():.4f}, "
            f"std={stream_B_norms.std().item():.4f}"
        )
        norm_ratio_A = stream_A_norms / offline_norms
        norm_ratio_B = stream_B_norms / offline_norms
        print(
            f"    Norm ratio [A]/offline: mean={norm_ratio_A.mean().item():.4f}, "
            f"std={norm_ratio_A.std().item():.4f}"
        )
        print(
            f"    Norm ratio [B]/offline: mean={norm_ratio_B.mean().item():.4f}, "
            f"std={norm_ratio_B.std().item():.4f}"
        )

        print(f"\n{'=' * 70}")
