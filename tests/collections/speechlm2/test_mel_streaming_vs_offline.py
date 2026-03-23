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
Diagnostic test — mel feature equivalence: streaming BatchedCacheFeatureBufferer vs offline preprocessor.

Verifies that per-frame mel features from chunk-by-chunk streaming extraction
are numerically identical to the corresponding frames from offline (full-audio)
preprocessing.

Theoretical expectation:
  For chunk i, the feature buffer processes extended_chunk_size = 2720 audio samples
  (160 lookback + 2560 new). The preprocessor (STFT with center=True) produces 17+ mel
  frames. The buffer takes the last 16 (discarding the lookback frame). These 16 frames
  correspond to STFT windows at the same absolute audio positions as
  offline_mel[:, :, i*16 : (i+1)*16], so they should be numerically identical for
  interior chunks.

Run with:
    cd /tmp && python -m pytest /path/to/test_mel_streaming_vs_offline.py -v -s -p no:conftest
"""

import math

import pytest
import torch
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


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestMelStreamingVsOffline:

    @pytest.mark.parametrize("audio_duration", [3.0, 10.0])
    def test_mel_streaming_vs_offline(self, perception, device, audio_duration):
        """Compare per-frame mel features from BatchedCacheFeatureBufferer vs offline preprocessor."""
        preprocessor_cfg, buffer_secs, chunk_secs, chunk_samples = _get_buffer_params(perception)
        audio_1d = _generate_test_audio(audio_duration, device)
        audio_len = audio_1d.shape[0]

        # --- Print preprocessor config ---
        print(f"\n{'=' * 70}")
        print(f"Preprocessor config:")
        print(f"  window_size   = {preprocessor_cfg.window_size}")
        print(f"  window_stride = {preprocessor_cfg.window_stride}")
        print(f"  n_fft         = {preprocessor_cfg.get('n_fft', 'auto')}")
        print(f"  pad_to        = {preprocessor_cfg.get('pad_to', 0)}")
        print(f"  features      = {preprocessor_cfg.features}")
        print(f"  exact_pad     = {preprocessor_cfg.get('exact_pad', False)}")
        print(f"  normalize     = {preprocessor_cfg.get('normalize', 'None')}")
        print(f"  preemph       = {preprocessor_cfg.get('preemph', 'None')}")
        print(f"Audio: {audio_len} samples, {audio_len / SAMPLE_RATE:.2f}s")

        # Compute derived STFT params for diagnostics
        window_samples = int(preprocessor_cfg.window_size * SAMPLE_RATE)
        hop_samples = int(preprocessor_cfg.window_stride * SAMPLE_RATE)
        n_fft = preprocessor_cfg.get('n_fft', None) or (2 ** math.ceil(math.log2(window_samples)))
        exact_pad = preprocessor_cfg.get('exact_pad', False)
        if exact_pad:
            stft_pad = (n_fft - hop_samples) // 2
        else:
            stft_pad = n_fft // 2  # center=True default
        print(
            f"  Derived: window_samples={window_samples}, hop_samples={hop_samples}, "
            f"n_fft={n_fft}, stft_pad={stft_pad}"
        )

        # --- Offline mel ---
        with torch.no_grad():
            offline_mel, offline_mel_len = perception.preprocessor(
                input_signal=audio_1d.unsqueeze(0),
                length=torch.tensor([audio_len], device=device),
            )
        valid_offline_len = offline_mel_len[0].item()
        print(f"Offline mel: shape={offline_mel.shape}, valid_len={valid_offline_len}")

        # --- Streaming mel via BatchedCacheFeatureBufferer ---
        feature_buffer = _make_feature_buffer(preprocessor_cfg, buffer_secs, chunk_secs, device)
        fbl = feature_buffer.feature_buffer_len
        fcl = feature_buffer.feature_chunk_len
        n_look_back = fbl - fcl
        ecs = feature_buffer.extended_chunk_size

        print(f"Feature buffer: feature_buffer_len={fbl}, feature_chunk_len={fcl}")
        print(f"  n_chunk_look_back={feature_buffer.n_chunk_look_back} samples, " f"extended_chunk_size={ecs} samples")
        print(f"  chunk_samples={chunk_samples}")
        print(f"  STFT context needed (stft_pad)={stft_pad} vs lookback={feature_buffer.n_chunk_look_back}")
        if stft_pad > feature_buffer.n_chunk_look_back:
            print(
                f"  WARNING: stft_pad ({stft_pad}) > n_chunk_look_back "
                f"({feature_buffer.n_chunk_look_back}): boundary frames will see "
                f"zeros instead of real audio for {stft_pad - feature_buffer.n_chunk_look_back} samples"
            )

        # Compute expected valid frames from extended chunk
        expected_stft_frames = (ecs + 2 * stft_pad - n_fft) // hop_samples + 1
        print(f"  Expected STFT frames from extended chunk: {expected_stft_frames}")

        num_chunks = math.ceil(audio_len / chunk_samples)

        streaming_new_mels = []  # collect the fcl NEW mel frames per chunk
        streaming_full_buffers = []  # collect the full fbl-frame buffers
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, audio_len)
            chunk_wav = audio_1d[start:end]

            frame = Frame(samples=chunk_wav, length=end - start, stream_id=0)
            features, right_paddings = feature_buffer.update([frame])

            fb = features[0]  # (D, feature_buffer_len)
            new_mel = fb[:, -fcl:]  # last fcl frames
            streaming_new_mels.append(new_mel)
            streaming_full_buffers.append(fb.clone())

        # --- Concatenate streaming mel ---
        streaming_mel_all = torch.cat(streaming_new_mels, dim=-1)  # (D, N*fcl)

        print(f"Streaming mel concatenated: shape={streaming_mel_all.shape}")
        print(f"Offline valid length: {valid_offline_len}")

        # --- Per-frame comparison (new mel only) ---
        compare_len = min(streaming_mel_all.shape[-1], valid_offline_len)
        print(f"\n--- Per-frame comparison (new mel only, {compare_len} frames) ---")

        frame_mses = []
        frame_max_diffs = []
        for frame_idx in range(compare_len):
            s_frame = streaming_mel_all[:, frame_idx]
            o_frame = offline_mel[0, :, frame_idx]
            mse = torch.mean((s_frame - o_frame) ** 2).item()
            mad = torch.max(torch.abs(s_frame - o_frame)).item()
            frame_mses.append(mse)
            frame_max_diffs.append(mad)

        # Classify frames as boundary or interior
        # Boundary frames: first and last frame within each chunk's new mel block
        boundary_indices = set()
        for i in range(num_chunks):
            first_frame = i * fcl
            last_frame = i * fcl + fcl - 1
            if first_frame < compare_len:
                boundary_indices.add(first_frame)
            if last_frame < compare_len:
                boundary_indices.add(last_frame)

        # Print per-chunk summary (limit verbose output for long audio)
        max_verbose_chunks = 6  # print details for first N and last 2 chunks
        for i in range(num_chunks):
            chunk_start = i * fcl
            chunk_end = min(chunk_start + fcl, compare_len)
            if chunk_start >= compare_len:
                break
            chunk_mses = frame_mses[chunk_start:chunk_end]
            chunk_mads = frame_max_diffs[chunk_start:chunk_end]
            mean_mse = sum(chunk_mses) / len(chunk_mses)
            max_mse = max(chunk_mses)
            max_mad = max(chunk_mads)

            # Separate interior vs boundary stats for this chunk
            interior_mses_chunk = [frame_mses[f] for f in range(chunk_start, chunk_end) if f not in boundary_indices]
            boundary_mses_chunk = [frame_mses[f] for f in range(chunk_start, chunk_end) if f in boundary_indices]

            # Only print details for first N and last 2 chunks
            should_print = (i < max_verbose_chunks) or (i >= num_chunks - 2)
            if i == max_verbose_chunks and num_chunks > max_verbose_chunks + 2:
                print(f"  ... (skipping chunks {max_verbose_chunks}-{num_chunks-3} — same pattern) ...")

            if should_print:
                print(f"  Chunk {i}: frames [{chunk_start}:{chunk_end}]")
                print(
                    f"    overall:  mean_MSE={mean_mse:.8f}, max_MSE={max_mse:.8f}, " f"max_MaxAbsDiff={max_mad:.8f}"
                )
                if interior_mses_chunk:
                    print(
                        f"    interior: mean_MSE={sum(interior_mses_chunk)/len(interior_mses_chunk):.8f}, "
                        f"max_MSE={max(interior_mses_chunk):.8f}"
                    )
                if boundary_mses_chunk:
                    print(
                        f"    boundary: mean_MSE={sum(boundary_mses_chunk)/len(boundary_mses_chunk):.8f}, "
                        f"max_MSE={max(boundary_mses_chunk):.8f}"
                    )
                # Print individual frames with significant error
                for fidx in range(chunk_start, chunk_end):
                    if frame_mses[fidx] > 0.01:
                        tag = "BOUNDARY" if fidx in boundary_indices else "interior"
                        print(
                            f"    frame {fidx} [{tag}]: MSE={frame_mses[fidx]:.6f}, "
                            f"MaxAbsDiff={frame_max_diffs[fidx]:.6f}"
                        )

        # --- Full buffer comparison (what encoder sees) ---
        print(f"\n--- Full buffer comparison (what encoder sees) ---")
        buffer_mses = []
        for i, fb in enumerate(streaming_full_buffers):
            off_start = max(0, i * fcl - n_look_back)
            off_end = min((i + 1) * fcl, valid_offline_len)
            if off_start >= valid_offline_len:
                break
            o_equiv = offline_mel[0, :, off_start:off_end]
            pad_left = fbl - o_equiv.shape[-1]
            if pad_left > 0:
                o_equiv = torch.cat(
                    [torch.full((o_equiv.shape[0], pad_left), LOG_MEL_ZERO, device=device), o_equiv],
                    dim=-1,
                )

            min_len = min(fb.shape[-1], o_equiv.shape[-1])
            mse = torch.mean((fb[:, :min_len] - o_equiv[:, :min_len]) ** 2).item()
            mad = torch.max(torch.abs(fb[:, :min_len] - o_equiv[:, :min_len])).item()
            buffer_mses.append(mse)

            should_print = (i < max_verbose_chunks) or (i >= num_chunks - 2)
            if i == max_verbose_chunks and num_chunks > max_verbose_chunks + 2:
                print(f"  ... (skipping chunks {max_verbose_chunks}-{num_chunks-3}) ...")
            if should_print:
                print(f"  Chunk {i}: buffer shape={fb.shape}, offline_equiv shape={o_equiv.shape}")
                print(f"    MSE={mse:.8f}, MaxAbsDiff={mad:.8f}")
        if buffer_mses:
            print(
                f"  Buffer MSE summary: mean={sum(buffer_mses)/len(buffer_mses):.6f}, "
                f"max={max(buffer_mses):.6f}, min={min(buffer_mses):.6f}"
            )

        # --- Overall summary ---
        mse_tensor = torch.tensor(frame_mses)
        mad_tensor = torch.tensor(frame_max_diffs)

        # Separate boundary vs interior
        boundary_mask = torch.zeros(compare_len, dtype=torch.bool)
        for idx in boundary_indices:
            if idx < compare_len:
                boundary_mask[idx] = True
        interior_mask = ~boundary_mask

        interior_mses = mse_tensor[interior_mask]
        boundary_mses = mse_tensor[boundary_mask]

        print(f"\n--- Overall ---")
        print(f"Total frames compared: {compare_len}")
        print(f"  Boundary frames: {boundary_mask.sum().item()}")
        print(f"  Interior frames: {interior_mask.sum().item()}")

        print(f"\nBoundary frames:")
        print(f"  mean MSE:       {boundary_mses.mean().item():.6f}")
        print(f"  max MSE:        {boundary_mses.max().item():.6f}")
        print(f"  max MaxAbsDiff: {mad_tensor[boundary_mask].max().item():.6f}")

        if interior_mses.numel() > 0:
            print(f"\nInterior frames:")
            print(f"  mean MSE:       {interior_mses.mean().item():.8f}")
            print(f"  max MSE:        {interior_mses.max().item():.8f}")
            print(f"  max MaxAbsDiff: {mad_tensor[interior_mask].max().item():.8f}")

        frames_above_1e6 = (mse_tensor > 1e-6).sum().item()
        frames_above_1e3 = (mse_tensor > 1e-3).sum().item()
        frames_above_1 = (mse_tensor > 1.0).sum().item()
        print(f"\nFrames with MSE > 1e-6: {frames_above_1e6} / {compare_len}")
        print(f"Frames with MSE > 1e-3: {frames_above_1e3} / {compare_len}")
        print(f"Frames with MSE > 1.0:  {frames_above_1} / {compare_len}", end="")
        if frames_above_1 > 0:
            indices = (mse_tensor > 1.0).nonzero(as_tuple=True)[0].tolist()
            all_boundary = all(idx in boundary_indices for idx in indices)
            non_boundary_high = [idx for idx in indices if idx not in boundary_indices]
            print(f"  (all boundary: {all_boundary})")
            if non_boundary_high:
                print(f"  Non-boundary frames with MSE > 1.0: {non_boundary_high}")
                for idx in non_boundary_high[:10]:
                    chunk_idx = idx // fcl
                    within_chunk = idx % fcl
                    in_last_chunk = chunk_idx == num_chunks - 1
                    print(
                        f"    frame {idx} (chunk {chunk_idx}, pos {within_chunk}): "
                        f"MSE={frame_mses[idx]:.6f}"
                        f"{' [LAST CHUNK - partial audio causes frame misalignment]' if in_last_chunk else ''}"
                    )
        else:
            print()

        # --- Assertion: interior frames should be near-identical ---
        # Exclude the very last chunk which may have right-padding differences
        last_chunk_start = (num_chunks - 1) * fcl
        interior_end = min(last_chunk_start, compare_len)
        if interior_end > 0:
            # Only check truly interior frames (not boundary)
            check_mask = interior_mask[:interior_end]
            if check_mask.any():
                check_mses = mse_tensor[:interior_end][check_mask]
                max_interior_mse = check_mses.max().item()
                mean_interior_mse = check_mses.mean().item()
                print(f"\nInterior frames [0:{interior_end}] (excl. boundary):")
                print(f"  max MSE:  {max_interior_mse:.8f}")
                print(f"  mean MSE: {mean_interior_mse:.8f}")
                assert max_interior_mse < 1e-4, (
                    f"Interior frame mel MSE too high: max={max_interior_mse:.8f}, "
                    f"mean={mean_interior_mse:.8f}. "
                    f"Streaming and offline preprocessors produce different mel features "
                    f"even for non-boundary frames."
                )
