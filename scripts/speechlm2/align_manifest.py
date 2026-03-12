#!/usr/bin/env python3
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
Add word-level alignments to NeMo manifests using QwenForcedAligner.

Usage:
    # Single manifest
    python scripts/speechlm2/align_manifest.py \
        --input /path/to/manifest.json \
        --batch-size 8

    # Multiple manifests (comma-separated)
    python scripts/speechlm2/align_manifest.py \
        --input /path/to/train.json,/path/to/dev.json,/path/to/test.json \
        --batch-size 8

Reads each line of the input manifest (JSON-lines with ``audio_filepath``,
``text``, ``duration``), runs forced alignment in batches, and writes a new
manifest with an ``-aligned`` suffix containing an additional ``alignments``
field per utterance:

    {"audio_filepath": "...", "text": "...", "duration": ...,
     "alignments": [{"text": "hello", "start_time": 0.12, "end_time": 0.36}, ...]}
"""

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.collections.speechlm2.modules.qwen_forced_aligner import QwenForcedAligner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def read_manifest(path: str) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_audio(audio_path: str) -> np.ndarray:
    """Load audio and resample to 16 kHz mono float32."""
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def get_output_path(input_path: str) -> str:
    """Derive output path by adding '-aligned' suffix before the extension."""
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}-aligned{p.suffix}"))


def align_manifest(
    input_path: str,
    output_path: str,
    aligner: QwenForcedAligner,
    batch_size: int,
    manifest_label: str = "",
):
    """Align a single manifest file and write the output."""
    log.info("%sProcessing: %s -> %s", manifest_label, input_path, output_path)
    entries = read_manifest(input_path)

    n_aligned = 0
    n_failed = 0

    pbar = tqdm(
        range(0, len(entries), batch_size),
        desc=f"{manifest_label}{Path(input_path).name}",
        unit="batch",
        total=(len(entries) + batch_size - 1) // batch_size,
    )

    with open(output_path, "w") as out_f:
        for batch_start in pbar:
            batch = entries[batch_start : batch_start + batch_size]

            audio_arrays = []
            texts = []
            valid_indices = []

            for i, entry in enumerate(batch):
                audio_path = get_full_path(entry["audio_filepath"], manifest_file=input_path)
                text = entry.get("text", "")
                if not text:
                    log.warning("Skipping entry %d: empty text", batch_start + i)
                    continue
                try:
                    audio = load_audio(audio_path)
                except Exception as e:
                    log.warning("Skipping entry %d: failed to load audio %s: %s", batch_start + i, audio_path, e)
                    continue

                audio_arrays.append(audio)
                texts.append(text)
                valid_indices.append(i)

            alignment_map = {}
            if audio_arrays:
                try:
                    batch_alignments = aligner.align_numpy(audio_arrays, texts)
                except Exception as e:
                    log.warning(
                        "Alignment failed for batch starting at %d: %s. Writing entries without alignments.",
                        batch_start,
                        e,
                    )
                    batch_alignments = [[] for _ in valid_indices]
                    n_failed += len(valid_indices)

                for idx, aligns in zip(valid_indices, batch_alignments):
                    alignment_map[idx] = [asdict(a) for a in aligns]
                    n_aligned += 1

            for i, entry in enumerate(batch):
                out_entry = dict(entry)
                out_entry["alignments"] = alignment_map.get(i, [])
                out_f.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

            pbar.set_postfix(aligned=n_aligned, failed=n_failed)

    log.info("%sDone. Aligned: %d, Failed: %d, Total: %d", manifest_label, n_aligned, n_failed, len(entries))


def main():
    parser = argparse.ArgumentParser(description="Add word-level alignments to NeMo manifests.")
    parser.add_argument(
        "--input",
        required=True,
        help="Comma-separated paths to input NeMo manifests (JSON-lines).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Comma-separated output paths (one per input). Defaults to <input-stem>-aligned.json.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ForcedAligner-0.6B", help="Pretrained aligner model.")
    parser.add_argument("--language", default="English", help="Language for alignment.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for alignment.")
    parser.add_argument("--device", default="cuda", help="Device for the aligner model.")
    args = parser.parse_args()

    input_paths = [p.strip() for p in args.input.split(",")]
    if args.output is not None:
        output_paths = [p.strip() for p in args.output.split(",")]
        if len(output_paths) != len(input_paths):
            parser.error(
                f"Number of --output paths ({len(output_paths)}) must match "
                f"number of --input paths ({len(input_paths)})."
            )
    else:
        output_paths = [get_output_path(p) for p in input_paths]

    log.info("Loading aligner: %s", args.model)
    aligner = QwenForcedAligner(
        pretrained_model=args.model,
        language=args.language,
        device=args.device,
    )

    n_manifests = len(input_paths)
    for mi, (input_path, output_path) in enumerate(zip(input_paths, output_paths), 1):
        label = f"[{mi}/{n_manifests}] "
        align_manifest(input_path, output_path, aligner, args.batch_size, manifest_label=label)

    log.info("All %d manifest(s) processed.", n_manifests)


if __name__ == "__main__":
    main()
