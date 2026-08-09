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

The model's ``generate()`` method returns ``list[str]`` by default. Detailed
token and boundary records are opt-in through ``log_boundary_events=true``.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Optional

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import GenerationConfig
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.common.data.lhotse.dataloader import pad_extra_duration
from nemo.collections.speechlm2.models import StreamingGenerationRecord, StreamingSTTModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


def _cut_custom(cut) -> dict:
    custom = getattr(cut, "custom", None)
    return custom if isinstance(custom, dict) else {}


def _source_clip_id(cut) -> Optional[str]:
    custom = _cut_custom(cut)
    clip_id = getattr(cut, "clip_id", None) or custom.get("clip_id")
    return str(clip_id) if clip_id is not None else None


def _output_record(
    cut,
    reference_text: str,
    normalized_prediction: str,
    generation_record: Optional[StreamingGenerationRecord] = None,
) -> dict:
    """Build one output row in the dataloader's actual yield order."""
    wer, _, nins, ndel, nsub = word_error_rate_detail(
        hypotheses=[normalized_prediction], references=[reference_text], use_cer=False
    )
    output = {
        "id": cut.id,
        "clip_id": _source_clip_id(cut),
        "duration": cut.duration,
        "text": reference_text,
        "pred_text": normalized_prediction,
        "wer": wer,
        "ins": nins,
        "del": ndel,
        "sub": nsub,
    }
    if generation_record is not None:
        if output["clip_id"] is None:
            raise RuntimeError(f"Detailed generation output requires source clip_id metadata; cut={cut.id!r}")
        if len(generation_record.sampled_token_ids) != len(generation_record.sampled_token_pieces):
            raise RuntimeError(f"Sampled token ID/piece length mismatch for clip_id={output['clip_id']!r}")
        output.update(
            {
                "pred_text_unnormalized": generation_record.pred_text_unnormalized.strip(),
                "sampled_token_ids": list(generation_record.sampled_token_ids),
                "sampled_token_pieces": list(generation_record.sampled_token_pieces),
                "boundary_events": [asdict(event) for event in generation_record.boundary_events],
            }
        )
    return output


class ToAudio(torch.utils.data.Dataset):
    """Minimal dataset that loads audio from a CutSet."""

    def __getitem__(self, cuts: CutSet):
        audios, audio_lens = cuts.load_audio(collate=True)
        return {"cuts": cuts, "audios": audios, "audio_lens": audio_lens}


@dataclass
class StreamingSTTGenerationConfig:
    """
    A proxy class for GenerationConfig so that we can use OmegaConf with hydra overrides.
    All parameters will be passed to GenerationConfig.
    """

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


@dataclass
class StreamingSTTEvalConfig:
    pretrained_name: str = ""
    inputs: str = ""
    batch_size: int = 64
    max_batch_duration: Optional[float] = None
    num_workers: int = 4
    max_new_tokens: int = 64
    system_prompt: str = "Transcribe the audio into text."
    output_manifest: Optional[str] = "streaming_stt_generations.jsonl"
    verbose: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"
    use_normalizer: Optional[str] = "english"  # "english", "basic", or "none"
    use_offline_embs: bool = False
    seed: Optional[int] = None  # Set for deterministic results
    pad_extra_duration: Optional[float] = 0.0
    use_state_machine_inference: bool = (
        False  # recommended turned off for chunk_size > 0, no effect for chunk_size <= 0
    )
    dynamic_min_chunk_size: int = 0  # dynamic chunking: min frames before allowing generation
    dynamic_max_chunk_size: Optional[int] = None  # dynamic chunking: max frames before forcing generation
    # When set, LM-head boundary decision uses a probability threshold:
    # emit when p(user_footer_first_id) ≥ threshold (instead of argmax). Useful
    # to recover boundaries where the LM is moderately confident but loses
    # argmax to <blank>. Has no effect when use_chunk_classifier is True.
    lm_head_emit_threshold: Optional[float] = None
    # When True, dump per-LISTENING-frame diagnostics (LM head top-5, prob of
    # user_footer_first / blank, aux head sigmoid, decision taken) to a
    # sibling JSONL alongside output_manifest. Slows inference; use on
    # small eval sets when debugging boundary-decision behavior.
    debug_log_audio_frames: bool = False
    # Collect stream-relative SOU/EOU event timestamps and include them in
    # output records. This is opt-in because it adds diagnostic bookkeeping.
    log_boundary_events: bool = False
    generation_config: StreamingSTTGenerationConfig = field(default_factory=StreamingSTTGenerationConfig)


@hydra_runner(config_name="StreamingSTTEvalConfig", schema=StreamingSTTEvalConfig)
def main(cfg: StreamingSTTEvalConfig):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.seed is not None:
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
    cuts = cuts.map(partial(pad_extra_duration, extra_duration=cfg.pad_extra_duration))
    sampler = lhotse.dataset.DynamicCutSampler(
        cuts,
        max_cuts=cfg.batch_size,
        max_duration=cfg.max_batch_duration,
    )
    num_batches = None if cfg.max_batch_duration is not None else math.ceil(len(cuts) / cfg.batch_size)
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=sampler,
        num_workers=cfg.num_workers,
        batch_size=None,
    )

    _normalizer_key = cfg.use_normalizer.lower() if isinstance(cfg.use_normalizer, str) else cfg.use_normalizer
    normalizer = {"english": EnglishTextNormalizer(), "basic": BasicTextNormalizer()}.get(_normalizer_key, lambda x: x)

    refs = []
    hyps = []
    input_duration = 0.0
    infer_duration = 0.0

    # Optional per-frame debug log file (one record per LISTENING frame per
    # cut, keyed by cut id). Only opened when debug_log_audio_frames=True.
    debug_log_writer = None
    if cfg.debug_log_audio_frames and cfg.output_manifest is not None:
        manifest_path = Path(cfg.output_manifest)
        debug_log_path = manifest_path.with_name(
            manifest_path.stem.replace("_generations", "") + "_audio_frame_log.jsonl"
        )
        debug_log_writer = SequentialJsonlWriter(str(debug_log_path))
        logging.info(f"Audio frame debug log → {debug_log_path}")

    output_writer = SequentialJsonlWriter(cfg.output_manifest) if cfg.output_manifest is not None else None
    try:
        for batch_idx, batch in tqdm(enumerate(dloader), total=num_batches):
            ts = perf_counter()
            cfg.generation_config.max_new_tokens = cfg.max_new_tokens
            generation_config = GenerationConfig(**OmegaConf.to_container(cfg.generation_config))
            batch_debug_logs: Optional[list] = [] if cfg.debug_log_audio_frames else None
            batch_results = model.generate(
                audios=batch["audios"].to(model.device, non_blocking=True),
                audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
                system_prompt=cfg.system_prompt,
                max_new_tokens=cfg.max_new_tokens,
                generation_config=generation_config,
                use_offline_embs=cfg.use_offline_embs,
                use_state_machine_inference=cfg.use_state_machine_inference,
                dynamic_min_chunk_size=cfg.dynamic_min_chunk_size,
                dynamic_max_chunk_size=cfg.dynamic_max_chunk_size,
                lm_head_emit_threshold=cfg.lm_head_emit_threshold,
                debug_logs=batch_debug_logs,
                return_generation_records=cfg.log_boundary_events,
            )
            batch_infer_duration = perf_counter() - ts
            batch_cuts = list(batch["cuts"])
            if len(batch_results) != len(batch_cuts):
                raise RuntimeError(
                    f"Generation result count ({len(batch_results)}) did not match yielded cut count "
                    f"({len(batch_cuts)}) for batch {batch_idx}"
                )

            if debug_log_writer is not None and batch_debug_logs is not None:
                if len(batch_debug_logs) != len(batch_cuts):
                    raise RuntimeError(
                        f"Debug record count ({len(batch_debug_logs)}) did not match yielded cut count "
                        f"({len(batch_cuts)}) for batch {batch_idx}"
                    )
                for cut, frames in zip(batch_cuts, batch_debug_logs):
                    debug_log_writer.write({"id": cut.id, "duration": cut.duration, "frames": frames})

            generation_records: list[Optional[StreamingGenerationRecord]]
            if cfg.log_boundary_events:
                if not all(isinstance(result, StreamingGenerationRecord) for result in batch_results):
                    raise RuntimeError("log_boundary_events=true requires detailed StreamingGenerationRecord outputs")
                generation_records = list(batch_results)
                batch_unnormalized_predictions = [record.pred_text_unnormalized.strip() for record in batch_results]
            else:
                if not all(isinstance(result, str) for result in batch_results):
                    raise RuntimeError("Text-only generation expected string outputs")
                generation_records = [None] * len(batch_results)
                batch_unnormalized_predictions = [prediction.strip() for prediction in batch_results]

            batch_duration = sum(cut.duration for cut in batch_cuts)
            batch_refs = [normalizer(cut.supervisions[0].text) for cut in batch_cuts]
            batch_hyps = [normalizer(prediction) for prediction in batch_unnormalized_predictions]

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
            input_duration += batch_duration
            infer_duration += batch_infer_duration
            if output_writer is not None:
                for cut, ref, hyp, generation_record in zip(
                    batch_cuts, batch_refs, batch_hyps, generation_records
                ):
                    output_writer.write(_output_record(cut, ref, hyp, generation_record))
    finally:
        if debug_log_writer is not None:
            debug_log_writer.close()
        if output_writer is not None:
            output_writer.close()

    wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    rtfx = input_duration / infer_duration
    logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    logging.info(f"RTFx: {rtfx:.1f}")

    if cfg.output_manifest is not None:
        log_file = Path(cfg.output_manifest).parent / "log.txt"
        with open(log_file, "a") as f:
            f.write(f"======{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}======\n")
            f.write(f"Input: {cfg.inputs}\n")
            f.write(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]\n")
            f.write(f"RTFx: {rtfx:.1f}\n")
            f.write("=============================================\n\n")


if __name__ == "__main__":
    main()
