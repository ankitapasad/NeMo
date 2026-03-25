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
Offline evaluation script for StreamingSTTModel.

Usage::

    python streaming_stt_generate.py \
        pretrained_name=nvidia/streaming-stt-v1 \
        inputs=/data/test.jsonl \
        batch_size=32

    # Simulate streaming (chunk-by-chunk with blanks):
    python streaming_stt_generate.py \
        pretrained_name=nvidia/streaming-stt-v1 \
        inputs=/data/test.jsonl \
        simulate_streaming=true

The model's ``generate()`` method returns ``list[str]`` directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from tqdm import tqdm
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2.models import StreamingSTTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


class ToAudio(torch.utils.data.Dataset):
    """Minimal dataset that loads audio from a CutSet."""

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


@dataclass
class StreamingSTTEvalConfig:
    pretrained_name: str = ""
    inputs: str = ""
    batch_size: int = 64
    max_new_tokens: int = 64
    system_prompt: str = "Transcribe the audio into text."
    output_manifest: Optional[str] = "streaming_stt_generations.jsonl"
    verbose: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_normalizer: Optional[str] = "english"  # "english", "basic", or "none"
    use_offline_embs: bool = False
    seed: Optional[int] = None  # Set for deterministic results


@hydra_runner(config_name="StreamingSTTEvalConfig", schema=StreamingSTTEvalConfig)
def main(cfg: StreamingSTTEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.seed is not None:
        import os

        logging.warning(f"Setting random seed to {cfg.seed}, this will slow down the inference")
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
    else:
        logging.warning("Random seed not set, results will not be deterministic")

    model = StreamingSTTModel.from_pretrained(cfg.pretrained_name)
    model = model.eval().to(getattr(torch, cfg.dtype)).to(cfg.device)

    cuts = guess_parse_cutset(cfg.inputs)
    # Resample to model's expected sample rate if needed.
    sample_cut = next(iter(cuts))
    if sample_cut.sampling_rate != model.sampling_rate:
        logging.info(f"Resampling cuts from {sample_cut.sampling_rate} to {model.sampling_rate} Hz")
        cuts = CutSet.from_cuts(c.resample(model.sampling_rate) for c in cuts)
    cuts = cuts.sort_by_duration()
    sampler = lhotse.dataset.DynamicCutSampler(cuts, max_cuts=cfg.batch_size)
    num_batches = math.ceil(len(cuts) / cfg.batch_size)
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=sampler,
        num_workers=1,
        batch_size=None,
    )

    _normalizer_key = cfg.use_normalizer.lower() if isinstance(cfg.use_normalizer, str) else cfg.use_normalizer
    normalizer = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}.get(_normalizer_key, lambda x: x)

    refs = []
    hyps = []
    input_durations = []
    infer_durations = []

    for batch_idx, batch in tqdm(enumerate(dloader), total=num_batches):
        ts = perf_counter()
        logging
        batch_hyps_raw = model.generate(
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            system_prompt=cfg.system_prompt,
            max_new_tokens=cfg.max_new_tokens,
            use_offline_embs=cfg.use_offline_embs,
        )
        batch_infer_duration = perf_counter() - ts

        batch_duration = sum(c.duration for c in batch["cuts"])
        batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
        batch_hyps = [normalizer(h.strip()) for h in batch_hyps_raw]

        if cfg.verbose:
            batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_refs)
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info("--------------------------------")
            logging.info(
                f"Batch {batch_idx}: "
                f"WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] "
                f"RTFx={batch_rtfx:.1f}"
            )
            for ref, hyp in zip(batch_refs, batch_hyps):
                logging.info(f"\n[REF]\t`{ref}`\n[HYP]\t`{hyp}`\n")
            logging.info("--------------------------------")

        refs.extend(batch_refs)
        hyps.extend(batch_hyps)
        input_durations.append(batch_duration)
        infer_durations.append(batch_infer_duration)

    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = sum(input_durations) / sum(infer_durations)
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    logging.info(f"RTFx: {rtfx:.1f}")

    if cfg.output_manifest is not None:
        log_file = Path(cfg.output_manifest).parent / "log.txt"
        with open(log_file, "a") as f:
            f.write(f"======{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}======\n")
            f.write(f"Input: {cfg.inputs}\n")
            f.write(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]\n")
            f.write(f"RTFx: {rtfx:.1f}\n")
            f.write(f"=============================================\n\n")
        with SequentialJsonlWriter(cfg.output_manifest) as writer:
            for cut, ref, hyp in zip(cuts, refs, hyps):
                wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=[hyp], references=[ref], use_cer=False)
                writer.write(
                    {
                        "id": cut.id,
                        "duration": cut.duration,
                        "text": ref,
                        "pred_text": hyp,
                        "wer": wer,
                        "ins": nins,
                        "del": ndel,
                        "sub": nsub,
                    }
                )


if __name__ == "__main__":
    main()
