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
Standalone diagnostic test — cache-aware streaming encoder vs offline encoder.

Isolates whether the cache-aware streaming encoder corrupts perception output,
independent of mel-feature differences. Both paths receive the **same offline mel**
features; the only variable is whether the encoder runs in one pass (offline) or
chunk-by-chunk with cache (streaming).

This corresponds to experiment [B] from test_perception_embedding_streaming_vs_offline.py,
extracted as a focused standalone test.

Run with:
    cd /tmp && python -m pytest /path/to/test_encoder_streaming_vs_offline.py -v -s -p no:conftest
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

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


def _generate_test_audio(duration_secs, device):
    """Generate deterministic sinusoidal test audio at 16kHz."""
    n_samples = int(duration_secs * SAMPLE_RATE)
    t = torch.arange(n_samples, dtype=torch.float32, device=device)
    audio = 0.5 * torch.sin(2 * math.pi * 440 * t / SAMPLE_RATE) + 0.3 * torch.sin(2 * math.pi * 880 * t / SAMPLE_RATE)
    return audio


def _run_streaming_perception(perception, feature_buffers, right_paddings_list, device, dtype=None):
    """Run streaming perception on pre-computed feature buffers.

    Args:
        feature_buffers: list of (D, fbl) tensors -- the full mel buffer per chunk
        right_paddings_list: list of int -- right padding per chunk
        dtype: optional dtype to cast cache state to (e.g. torch.bfloat16)
    Returns:
        list of (valid_out_len, H) embedding tensors, one per chunk
    """
    cache_last_channel, cache_last_time, cache_last_channel_len = perception.get_initial_cache_state(
        batch_size=1, device=device
    )
    if dtype is not None:
        cache_last_channel = cache_last_channel.to(dtype)
        cache_last_time = cache_last_time.to(dtype)
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
    """Compute per-frame cosine similarity, L2 distance, and relative L2."""
    cos_sims, l2_dists, rel_l2s, offline_norms = [], [], [], []
    for i in range(compare_len):
        s = streaming_all[i]
        o = offline_all[i]
        diff = s - o
        cos = F.cosine_similarity(s.unsqueeze(0), o.unsqueeze(0)).item()
        l2 = torch.norm(diff).item()
        o_norm = torch.norm(o).item()
        rl2 = l2 / o_norm if o_norm > 0 else float('inf')
        cos_sims.append(cos)
        l2_dists.append(l2)
        rel_l2s.append(rl2)
        offline_norms.append(o_norm)
    return cos_sims, l2_dists, rel_l2s, offline_norms


def _format_table(chunks_data, valid_out_len):
    """Format a per-chunk table with per-frame metrics.

    Args:
        chunks_data: list of dicts, one per chunk, each containing lists of
                     per-frame metrics for that chunk
        valid_out_len: frames per chunk
    Returns:
        list of formatted lines
    """
    lines = []
    # Build header
    frame_headers = []
    for f in range(valid_out_len):
        frame_headers.append(f"Frame{f}")
    header = f"{'Chunk':>5}  "
    sub_header = f"{'':>5}  "
    for fh in frame_headers:
        header += f"{'':>4}{fh:^38}"
        sub_header += f"  {'CosSim':>8} {'RelL2':>8} {'L2':>8} {'||o||':>8}"
    lines.append(header)
    lines.append(sub_header)
    lines.append("-" * len(sub_header))

    for cd in chunks_data:
        row = f"{cd['chunk_idx']:>5}  "
        for f in range(valid_out_len):
            if f < len(cd['cos_sims']):
                row += (
                    f"  {cd['cos_sims'][f]:>8.6f}"
                    f" {cd['rel_l2s'][f]:>8.4f}"
                    f" {cd['l2_dists'][f]:>8.4f}"
                    f" {cd['offline_norms'][f]:>8.4f}"
                )
            else:
                row += f"  {'n/a':>8} {'n/a':>8} {'n/a':>8} {'n/a':>8}"
        lines.append(row)

    return lines


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestEncoderStreamingVsOffline:

    @pytest.mark.parametrize("audio_duration", [3.0, 10.0])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
    def test_encoder_streaming_vs_offline(self, perception, device, audio_duration, dtype):
        """Compare encoder output: chunk-by-chunk with cache vs one-pass, same offline mel."""
        if dtype == torch.bfloat16 and not (device.type == "cuda" and torch.cuda.is_bf16_supported()):
            pytest.skip("bfloat16 requires CUDA with bf16 support")

        # Cast perception to the requested dtype (and back after the test)
        orig_dtype = next(perception.parameters()).dtype
        perception.to(dtype)
        try:
            self._run_test(perception, device, audio_duration, dtype)
        finally:
            perception.to(orig_dtype)

    def _run_test(self, perception, device, audio_duration, dtype):
        preprocessor_cfg, buffer_secs, chunk_secs, chunk_samples = _get_buffer_params(perception)
        audio_1d = _generate_test_audio(audio_duration, device).to(dtype)
        audio_len = audio_1d.shape[0]
        num_chunks = math.ceil(audio_len / chunk_samples)

        fbl = int(buffer_secs / preprocessor_cfg.window_stride)
        fcl = int(chunk_secs / preprocessor_cfg.window_stride)
        n_cache = fbl - fcl

        streaming_cfg = perception.encoder.streaming_cfg
        valid_out_len = streaming_cfg.valid_out_len

        # --- Config diagnostics ---
        print(f"\n{'=' * 80}")
        print(f"ENCODER STREAMING VS OFFLINE TEST ({dtype})")
        print(f"{'=' * 80}")
        print(f"Config:")
        print(f"  dtype: {dtype}")
        print(f"  Encoder type: {type(perception.encoder).__name__}")
        print(f"  valid_out_len: {valid_out_len} (frames per chunk)")
        print(f"  Subsampling factor: {getattr(perception.encoder, 'subsampling_factor', 'N/A')}")
        print(f"  Feature buffer: fbl={fbl}, fcl={fcl}, cache={n_cache}")
        print(f"  Cache sizes: pre_encode={streaming_cfg.pre_encode_cache_size}")
        print(f"Audio:")
        print(f"  Samples: {audio_len}, Duration: {audio_len / SAMPLE_RATE:.2f}s")
        print(f"  Chunks: {num_chunks} (chunk_samples={chunk_samples})")

        # ==================================================================
        # Step 1: Compute offline mel features
        # ==================================================================
        with torch.no_grad():
            offline_mel, offline_mel_len = perception.preprocessor(
                input_signal=audio_1d.unsqueeze(0),
                length=torch.tensor([audio_len], device=device),
            )
        valid_mel_len = offline_mel_len[0].item()
        print(f"\nOffline mel: shape={offline_mel.shape}, valid_len={valid_mel_len}")

        # ==================================================================
        # Step 2: Offline perception (one-pass, no cache)
        # ==================================================================
        with torch.no_grad():
            offline_embs, offline_emb_lens = perception(
                processed_signal=offline_mel,
                processed_signal_length=offline_mel_len,
            )
        offline_all = offline_embs.squeeze(0)  # (T_offline, H)
        print(
            f"Offline perception: embs shape={offline_embs.shape}, "
            f"lens={offline_emb_lens.item()}, H={offline_all.shape[-1]}"
        )

        # ==================================================================
        # Step 3: Build mel buffers from offline mel for streaming
        # ==================================================================
        offline_mel_buffers = []
        offline_mel_right_paddings = []
        for i in range(num_chunks):
            off_start = max(0, i * fcl - n_cache)
            off_end = min((i + 1) * fcl, valid_mel_len)
            if off_start >= valid_mel_len:
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
            rp = max(0, (i + 1) * fcl - valid_mel_len)
            offline_mel_right_paddings.append(rp)

        print(
            f"\nStreaming mel buffers: {len(offline_mel_buffers)} chunks, "
            f"each shape=({offline_mel.shape[1]}, {fbl})"
        )

        # ==================================================================
        # Step 4: Streaming perception (chunk-by-chunk with cache)
        # ==================================================================
        streaming_chunks = _run_streaming_perception(
            perception,
            offline_mel_buffers,
            offline_mel_right_paddings,
            device,
            dtype=dtype,
        )
        streaming_all = torch.cat(streaming_chunks, dim=0)  # (T_stream, H)
        print(
            f"Streaming perception: {streaming_all.shape[0]} frames total "
            f"({len(streaming_chunks)} chunks x {valid_out_len} frames)"
        )

        # ==================================================================
        # Step 5: Per-frame comparison
        # ==================================================================
        compare_len = min(streaming_all.shape[0], offline_all.shape[0])
        print(
            f"\nComparing {compare_len} frames "
            f"(streaming={streaming_all.shape[0]}, offline={offline_all.shape[0]})"
        )

        # Cast to float32 for metric computation to avoid bf16 precision loss
        cos_sims, l2_dists, rel_l2s, offline_norms = _per_frame_metrics(
            streaming_all.float(), offline_all.float(), compare_len
        )

        # --- Per-chunk table ---
        chunks_table_data = []
        for ci in range(len(streaming_chunks)):
            frame_start = ci * valid_out_len
            frame_end = min(frame_start + valid_out_len, compare_len)
            if frame_start >= compare_len:
                break
            n_frames = frame_end - frame_start
            chunks_table_data.append(
                {
                    'chunk_idx': ci,
                    'cos_sims': cos_sims[frame_start:frame_end],
                    'rel_l2s': rel_l2s[frame_start:frame_end],
                    'l2_dists': l2_dists[frame_start:frame_end],
                    'offline_norms': offline_norms[frame_start:frame_end],
                }
            )

        print(f"\nPer-chunk table:")
        table_lines = _format_table(chunks_table_data, valid_out_len)
        for line in table_lines:
            print(f"  {line}")

        # --- Overall stats ---
        cos_t = torch.tensor(cos_sims)
        l2_t = torch.tensor(l2_dists)
        rl2_t = torch.tensor(rel_l2s)

        print(f"\nOverall ({compare_len} frames):")
        print(
            f"  CosSim:  min={cos_t.min().item():.6f}, median={cos_t.median().item():.6f}, "
            f"mean={cos_t.mean().item():.6f}, max={cos_t.max().item():.6f}"
        )
        print(
            f"  RelL2:   min={rl2_t.min().item():.6f}, median={rl2_t.median().item():.6f}, "
            f"mean={rl2_t.mean().item():.6f}, max={rl2_t.max().item():.6f}"
        )
        print(
            f"  L2:      min={l2_t.min().item():.6f}, median={l2_t.median().item():.6f}, "
            f"mean={l2_t.mean().item():.6f}, max={l2_t.max().item():.6f}"
        )

        # --- Per-chunk streaming shapes (diagnostic) ---
        print(f"\n  Per-chunk streaming emb shapes:")
        for ci, chunk_embs in enumerate(streaming_chunks):
            print(f"    Chunk {ci}: {chunk_embs.shape}")

        print(f"\n{'=' * 80}")

        # --- Save report ---
        self._save_report(
            audio_duration,
            audio_len,
            num_chunks,
            chunk_samples,
            fbl,
            fcl,
            n_cache,
            valid_out_len,
            perception,
            offline_embs,
            offline_emb_lens,
            streaming_chunks,
            compare_len,
            chunks_table_data,
            cos_t,
            l2_t,
            rl2_t,
            dtype,
        )

    @staticmethod
    def _save_report(
        audio_duration,
        audio_len,
        num_chunks,
        chunk_samples,
        fbl,
        fcl,
        n_cache,
        valid_out_len,
        perception,
        offline_embs,
        offline_emb_lens,
        streaming_chunks,
        compare_len,
        chunks_table_data,
        cos_t,
        l2_t,
        rl2_t,
        dtype,
    ):
        """Write a markdown report to nemo_experiments/debug_logs/."""
        # Find the repo root (where nemo_experiments should be)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        report_dir = os.path.join(repo_root, "nemo_experiments", "debug_logs")
        os.makedirs(report_dir, exist_ok=True)
        dtype_tag = "bf16" if dtype == torch.bfloat16 else "fp32"
        report_path = os.path.join(report_dir, f"encoder_streaming_vs_offline_report_{dtype_tag}.md")

        lines = []
        lines.append("# Encoder Streaming vs Offline Report")
        lines.append("")
        lines.append("## Question")
        lines.append("")
        lines.append("If we give the encoder **perfect offline mel** features, does running it")
        lines.append("chunk-by-chunk with cache produce the same embeddings as running it in one pass?")
        lines.append("")
        lines.append(f"## Config")
        lines.append("")
        lines.append(f"- dtype: `{dtype}`")
        lines.append(f"- Encoder type: `{type(perception.encoder).__name__}`")
        lines.append(f"- valid_out_len: {valid_out_len}")
        lines.append(f"- Subsampling factor: {getattr(perception.encoder, 'subsampling_factor', 'N/A')}")
        lines.append(f"- Feature buffer length (fbl): {fbl}")
        lines.append(f"- Feature chunk length (fcl): {fcl}")
        lines.append(f"- Cache frames (n_cache): {n_cache}")
        lines.append("")
        lines.append(f"## Audio: {audio_duration}s")
        lines.append("")
        lines.append(f"- Samples: {audio_len}")
        lines.append(f"- Duration: {audio_len / SAMPLE_RATE:.2f}s")
        lines.append(f"- Chunks: {num_chunks} (chunk_samples={chunk_samples})")
        lines.append("")
        lines.append(f"## Shapes")
        lines.append("")
        lines.append(f"- Offline embs: {list(offline_embs.shape)}, lens={offline_emb_lens.item()}")
        lines.append(
            f"- Streaming: {len(streaming_chunks)} chunks, " f"shapes: {[list(c.shape) for c in streaming_chunks]}"
        )
        lines.append(f"- Compare length: {compare_len} frames")
        lines.append("")
        lines.append("## Per-chunk table")
        lines.append("")
        # Table header
        frame_cols = []
        for f in range(valid_out_len):
            frame_cols.extend([f"F{f}_CosSim", f"F{f}_RelL2", f"F{f}_L2", f"F{f}_||o||"])
        lines.append("| Chunk | " + " | ".join(frame_cols) + " |")
        lines.append("|-------|" + "|".join(["--------"] * len(frame_cols)) + "|")
        for cd in chunks_table_data:
            row = f"| {cd['chunk_idx']:>5} |"
            for f in range(valid_out_len):
                if f < len(cd['cos_sims']):
                    row += (
                        f" {cd['cos_sims'][f]:.6f} |"
                        f" {cd['rel_l2s'][f]:.4f} |"
                        f" {cd['l2_dists'][f]:.4f} |"
                        f" {cd['offline_norms'][f]:.4f} |"
                    )
                else:
                    row += " n/a | n/a | n/a | n/a |"
            lines.append(row)
        lines.append("")
        lines.append("## Overall")
        lines.append("")
        lines.append(f"| Metric | Min | Median | Mean | Max |")
        lines.append(f"|--------|-----|--------|------|-----|")
        lines.append(
            f"| CosSim | {cos_t.min().item():.6f} | {cos_t.median().item():.6f} "
            f"| {cos_t.mean().item():.6f} | {cos_t.max().item():.6f} |"
        )
        lines.append(
            f"| RelL2  | {rl2_t.min().item():.6f} | {rl2_t.median().item():.6f} "
            f"| {rl2_t.mean().item():.6f} | {rl2_t.max().item():.6f} |"
        )
        lines.append(
            f"| L2     | {l2_t.min().item():.6f} | {l2_t.median().item():.6f} "
            f"| {l2_t.mean().item():.6f} | {l2_t.max().item():.6f} |"
        )
        lines.append("")

        with open(report_path, "w") as f:
            f.write("\n".join(lines))
        print(f"\nReport saved to: {report_path}")
