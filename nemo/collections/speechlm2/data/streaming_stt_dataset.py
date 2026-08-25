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

import logging
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any, Iterable, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.parts.alignments import (
    ForcedAligner,
    WordAlignment,
    add_utterance_boundary_alignments,
    get_word_alignments_for_batch,
)
from nemo.collections.speechlm2.parts.utils import to_dataclass

AUDIO_TOKEN_IDX = -200
IGNORE_INDEX = -100
UTTERANCE_BOUNDARY_TIMESTAMP_SOURCE_ALIGNMENT = "alignment"
UTTERANCE_BOUNDARY_TIMESTAMP_SOURCE_GT_PREFERRED = "gt_preferred"
UTTERANCE_BOUNDARY_TIMESTAMP_SOURCES = {
    UTTERANCE_BOUNDARY_TIMESTAMP_SOURCE_ALIGNMENT,
    UTTERANCE_BOUNDARY_TIMESTAMP_SOURCE_GT_PREFERRED,
}
LEAN_MULTI_TURN_SCHEMA_VERSION = "lean_multi_turn_v2"
LEAN_MULTI_TURN_SOURCE_SAMPLE_TYPES = {"complete_turn", "pause_within_turn"}
USER_BACKCHANNEL_MODE_IGNORE = "ignore"
USER_BACKCHANNEL_MODE_SOB_EOU = "sob_eou"
USER_BACKCHANNEL_MODE_SOB_EOB = "sob_eob"
USER_BACKCHANNEL_MODES = {
    USER_BACKCHANNEL_MODE_IGNORE,
    USER_BACKCHANNEL_MODE_SOB_EOU,
    USER_BACKCHANNEL_MODE_SOB_EOB,
}


@dataclass(frozen=True)
class MultiTurnTargetSegment:
    """A timed target-speaker segment from a lean multi-turn manifest row."""

    text: str
    start_time: float
    end_time: float
    turn_ordinal: int = 0
    turn_id: str | None = None
    source_sample_id: str | None = None
    source_sample_type: str | None = None
    user_backchannel_event_id: int = 0


@dataclass(frozen=True)
class MultiTurnSample:
    """Validated row-local multi-turn training metadata."""

    schema_version: str
    transcript: str
    segments: tuple[MultiTurnTargetSegment, ...]
    num_substantive_turns: int
    num_user_backchannel_events: int


def _require_mapping(value: Any, *, field: str, cut_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Cut {cut_id!r} field {field!r} must be a mapping")
    return value


def _require_list(value: Any, *, field: str, cut_id: str) -> list:
    if not isinstance(value, list):
        raise TypeError(f"Cut {cut_id!r} field {field!r} must be a list")
    return value


def _fragment_text(fragment: Mapping[str, Any], *, field: str, cut_id: str) -> str:
    text = fragment.get("text")
    if not isinstance(text, str) or not text.strip():
        raise TypeError(f"Cut {cut_id!r} field {field!r}.text must be a non-empty string")
    return text.strip()


def parse_lean_multiturn_metadata(
    custom: Mapping[str, Any] | None,
    *,
    audio_duration_secs: float,
    cut_id: str = "<unknown>",
    user_backchannel_mode: str = USER_BACKCHANNEL_MODE_IGNORE,
) -> MultiTurnSample | None:
    """Parse a supported lean multi-turn row without synthesizing conversations.

    Substantive components map one-for-one, in order, to ``utterance_regions``.
    User backchannels are omitted in ``ignore`` mode. Boundary modes
    retain each annotated fragment as one distinct chronological event.
    """
    if user_backchannel_mode not in USER_BACKCHANNEL_MODES:
        raise ValueError(
            f"user_backchannel_mode must be one of {sorted(USER_BACKCHANNEL_MODES)}; "
            f"got {user_backchannel_mode!r}"
        )
    if custom is None:
        return None
    if not isinstance(custom, Mapping):
        raise TypeError(f"Cut {cut_id!r} custom metadata must be a mapping")
    curation = custom.get("curation")
    if curation is None:
        return None
    curation = _require_mapping(curation, field="curation", cut_id=cut_id)
    schema_version = curation.get("schema_version")
    if schema_version != LEAN_MULTI_TURN_SCHEMA_VERSION:
        target = curation.get("target")
        if isinstance(target, Mapping) and "utterance_regions" in target:
            raise ValueError(
                f"Cut {cut_id!r} has multi-turn utterance_regions but unsupported "
                f"curation.schema_version={schema_version!r}"
            )
        return None
    substantive_component_type = "substantive_turn"

    duration = _validate_timestamp(audio_duration_secs, field="audio_duration_secs", cut_id=cut_id)
    if duration <= 0:
        raise ValueError(f"Cut {cut_id!r} audio duration must be positive; got {duration}")

    target = _require_mapping(curation.get("target"), field="curation.target", cut_id=cut_id)
    components = _require_list(target.get("components"), field="curation.target.components", cut_id=cut_id)
    regions = _require_list(target.get("utterance_regions"), field="curation.target.utterance_regions", cut_id=cut_id)
    if not regions:
        raise ValueError(f"Cut {cut_id!r} lean multi-turn row has no utterance regions")

    validated_regions: list[dict[str, Any]] = []
    previous_region_end = -math.inf
    for region_idx, raw_region in enumerate(regions):
        prefix = f"curation.target.utterance_regions[{region_idx}]"
        region = _require_mapping(raw_region, field=prefix, cut_id=cut_id)
        start = _validate_timestamp(region.get("start"), field=f"{prefix}.start", cut_id=cut_id)
        region_duration = _validate_timestamp(region.get("duration"), field=f"{prefix}.duration", cut_id=cut_id)
        if region_duration <= 0:
            raise ValueError(f"Cut {cut_id!r} {prefix}.duration must be positive")
        end = start + region_duration
        if start < -1e-6 or end > duration + 1e-6:
            raise ValueError(
                f"Cut {cut_id!r} {prefix} must be contained in the audio; "
                f"got start={start}, end={end}, duration={duration}"
            )
        if start < previous_region_end - 1e-6:
            raise ValueError(f"Cut {cut_id!r} substantive utterance regions must be chronological and non-overlapping")
        turn_id = region.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            raise TypeError(f"Cut {cut_id!r} {prefix}.turn_id must be a non-empty string")
        source_sample_id = region.get("source_sample_id")
        source_sample_type = region.get("source_sample_type")
        if not isinstance(source_sample_id, str) or not source_sample_id:
            raise TypeError(f"Cut {cut_id!r} {prefix}.source_sample_id must be a non-empty string")
        if source_sample_type not in LEAN_MULTI_TURN_SOURCE_SAMPLE_TYPES:
            raise ValueError(
                f"Cut {cut_id!r} {prefix}.source_sample_type must be one of "
                f"{sorted(LEAN_MULTI_TURN_SOURCE_SAMPLE_TYPES)}; got {source_sample_type!r}"
            )
        validated_regions.append(
            {
                "start": max(0.0, start),
                "end": min(duration, end),
                "turn_id": turn_id,
                "source_sample_id": source_sample_id,
                "source_sample_type": source_sample_type,
            }
        )
        previous_region_end = end

    substantive_components = [
        component
        for component in components
        if isinstance(component, Mapping) and component.get("type") == substantive_component_type
    ]
    if len(substantive_components) != len(validated_regions):
        raise ValueError(
            f"Cut {cut_id!r} has {len(substantive_components)} {substantive_component_type} "
            f"components but "
            f"{len(validated_regions)} utterance regions"
        )

    segments: list[MultiTurnTargetSegment] = []
    transcript_pieces: list[str] = []
    complete_idx = 0
    user_backchannel_event_idx = 0
    previous_component_start = -math.inf
    for component_idx, raw_component in enumerate(components):
        prefix = f"curation.target.components[{component_idx}]"
        component = _require_mapping(raw_component, field=prefix, cut_id=cut_id)
        component_type = component.get("type")
        if component_type not in {substantive_component_type, "backchannel"}:
            raise ValueError(
                f"Cut {cut_id!r} {prefix}.type must be {substantive_component_type!r} "
                f"or 'backchannel'; "
                f"got {component_type!r}"
            )
        fragments = _require_list(component.get("fragments"), field=f"{prefix}.fragments", cut_id=cut_id)
        if not fragments:
            raise ValueError(f"Cut {cut_id!r} {prefix}.fragments must not be empty")

        validated_fragments: list[tuple[float, float, str]] = []
        for fragment_idx, raw_fragment in enumerate(fragments):
            fragment_prefix = f"{prefix}.fragments[{fragment_idx}]"
            fragment = _require_mapping(raw_fragment, field=fragment_prefix, cut_id=cut_id)
            start = _validate_timestamp(fragment.get("start"), field=f"{fragment_prefix}.start", cut_id=cut_id)
            fragment_duration = _validate_timestamp(
                fragment.get("duration"), field=f"{fragment_prefix}.duration", cut_id=cut_id
            )
            if fragment_duration <= 0:
                raise ValueError(f"Cut {cut_id!r} {fragment_prefix}.duration must be positive")
            end = start + fragment_duration
            if start < -1e-6 or end > duration + 1e-6:
                raise ValueError(f"Cut {cut_id!r} {fragment_prefix} must be contained in the audio")
            text = _fragment_text(fragment, field=fragment_prefix, cut_id=cut_id)
            validated_fragments.append((max(0.0, start), min(duration, end), text))

        component_start = min(start for start, _, _ in validated_fragments)
        if component_start < previous_component_start - 1e-6:
            raise ValueError(f"Cut {cut_id!r} target components must be chronological")
        previous_component_start = component_start

        if component_type == substantive_component_type:
            region = validated_regions[complete_idx]
            region_start, region_end = region["start"], region["end"]
            for identity_field in ("turn_id", "source_sample_id", "source_sample_type"):
                if component.get(identity_field) != region[identity_field]:
                    raise ValueError(
                        f"Cut {cut_id!r} {prefix}.{identity_field} must match "
                        f"curation.target.utterance_regions[{complete_idx}].{identity_field}"
                    )
            for fragment_start, fragment_end, _ in validated_fragments:
                if fragment_start < region_start - 1e-6 or fragment_end > region_end + 1e-6:
                    raise ValueError(
                        f"Cut {cut_id!r} {prefix} fragments must be contained in substantive "
                        f"utterance region {complete_idx}"
                    )
            text = " ".join(fragment_text for _, _, fragment_text in validated_fragments)
            complete_idx += 1
            transcript_pieces.append(text)
            segments.append(
                MultiTurnTargetSegment(
                    text=text,
                    start_time=region_start,
                    end_time=region_end,
                    turn_ordinal=complete_idx,
                    turn_id=region["turn_id"],
                    source_sample_id=region["source_sample_id"],
                    source_sample_type=region["source_sample_type"],
                )
            )
        elif user_backchannel_mode in {
            USER_BACKCHANNEL_MODE_SOB_EOU,
            USER_BACKCHANNEL_MODE_SOB_EOB,
        }:
            for start, end, text in validated_fragments:
                user_backchannel_event_idx += 1
                transcript_pieces.append(text)
                segments.append(
                    MultiTurnTargetSegment(
                        text=text,
                        start_time=start,
                        end_time=end,
                        user_backchannel_event_id=user_backchannel_event_idx,
                    )
                )

    segments.sort(key=lambda segment: (segment.start_time, segment.end_time, segment.turn_ordinal == 0))
    return MultiTurnSample(
        schema_version=schema_version,
        transcript=" ".join(transcript_pieces),
        segments=tuple(segments),
        num_substantive_turns=len(validated_regions),
        num_user_backchannel_events=user_backchannel_event_idx,
    )


def _debug_boundary_window_positions(input_ids, boundary_ids, radius=2):
    """Return original sequence positions within ``radius`` of each boundary token."""
    boundary_ids = {token_id for token_id in boundary_ids if token_id is not None}
    positions = set()
    for boundary_pos, token_id in enumerate(input_ids):
        if token_id in boundary_ids:
            positions.update(range(max(0, boundary_pos - radius), min(len(input_ids), boundary_pos + radius + 1)))
    return sorted(positions)


def _debug_dump_sequence(
    messages,
    input_ids,
    target_ids,
    assistant_mask,
    tokenizer,
    blank_id,
    sou_id=None,
    eou_id=None,
    transcript=None,
    do_breakpoint=False,
):
    """Print compact input/target windows around each utterance boundary."""
    hf_tok = tokenizer.tokenizer
    sep = "=" * 100

    print(f"\n{sep}")
    print("DEBUG: SEQUENCE DUMP (first sample)")
    print(sep)

    # print("\n--- MESSAGES (role: content) ---")
    # for i, msg in enumerate(messages):
    #     content = msg["content"]
    #     if len(content) > 80:
    #         content = content[:40] + f"...({len(content)} chars)..." + content[-20:]
    #     print(f"  [{i:3d}] {msg['role']:>9s}: {repr(content)}")

    if transcript is not None:
        print(f"\n--- TRANSCRIPT ---\n{repr(transcript)}")

    boundary_positions = [i for i, token_id in enumerate(input_ids) if token_id in {sou_id, eou_id} - {None}]
    display_positions = _debug_boundary_window_positions(input_ids, (sou_id, eou_id), radius=2)

    print(
        f"\n--- BOUNDARY TOKEN WINDOWS " f"(len={len(input_ids)}, boundaries={len(boundary_positions)}, context=2) ---"
    )
    print(f"  {'pos':>5s}  {'input_id':>9s}  {'target_id':>9s}  {'mask':>4s}  {'input_tok':<20s}  {'target_tok':<20s}")
    print(f"  {'-' * 5}  {'-' * 9}  {'-' * 9}  {'-' * 4}  {'-' * 20}  {'-' * 20}")

    previous_position = None
    for i in display_positions:
        if previous_position is not None and i > previous_position + 1:
            omitted = i - previous_position - 1
            print(f"  {'...':>5s}  ... {omitted} token{'s' if omitted != 1 else ''} omitted ...")
        inp_id = input_ids[i]
        tgt_id = target_ids[i] if i < len(target_ids) else None
        mask = assistant_mask[i] if i < len(assistant_mask) else 0

        if inp_id == AUDIO_TOKEN_IDX:
            inp_tok = "[AUDIO]"
        else:
            inp_tok = repr(hf_tok.decode([inp_id]))

        if tgt_id is None or tgt_id == IGNORE_INDEX:
            tgt_tok = "---"
            tgt_id_str = "IGNORE"
        elif tgt_id == AUDIO_TOKEN_IDX:
            tgt_tok = "[AUDIO]"
            tgt_id_str = str(tgt_id)
        else:
            tgt_tok = repr(hf_tok.decode([tgt_id]))
            tgt_id_str = str(tgt_id)

        mask_str = "*" if mask else "."
        print(f"  {i:5d}  {str(inp_id):>9s}  {tgt_id_str:>9s}  {mask_str:>4s}  {inp_tok:<20s}  {tgt_tok:<20s}")
        previous_position = i

    if not display_positions:
        print("  (no <sou>/<eou> tokens found)")

    n_audio = sum(1 for token_id in input_ids if token_id == AUDIO_TOKEN_IDX)
    n_blank_tgt = sum(1 for token_id in target_ids if token_id == blank_id)
    n_loss = sum(1 for token_id in target_ids if token_id != IGNORE_INDEX)
    n_mask = sum(assistant_mask)
    print("\n--- SUMMARY ---")
    print(f"  Total tokens:       {len(input_ids)}")
    print(f"  Audio frames:       {n_audio}")
    print(f"  Assistant mask sum: {n_mask} ({n_mask / len(input_ids):.3f})")
    print(f"  Loss positions:     {n_loss} ({n_loss / len(input_ids):.3f})")
    print(f"  Blank targets:      {n_blank_tgt}")
    print(f"  Boundary positions: {boundary_positions}")
    print(
        f"  sou_id={sou_id}, eou_id={eou_id}, blank_id={blank_id}, "
        f"AUDIO_TOKEN_IDX={AUDIO_TOKEN_IDX}, IGNORE_INDEX={IGNORE_INDEX}"
    )
    print(sep + "\n")

    if do_breakpoint:
        breakpoint()


def right_collate_vectors(
    tensors: Iterable[Union[torch.Tensor, np.ndarray]],
    padding_value: Union[int, float] = CrossEntropyLoss().ignore_index,
) -> torch.Tensor:
    tensors = [torch.as_tensor(t) for t in tensors]
    assert all(len(t.shape) == 1 for t in tensors), "Expected only 1-D input tensors."
    return pad_sequence(tensors, batch_first=True, padding_value=padding_value, padding_side="right")


@dataclass
class StreamingSTTBatch:
    """
    A batch of data for StreamingSTTModel.

    Attributes:
        audios: (B, T) audio signals.
        audio_lens: (B,) lengths of the audio signals in samples.
        input_tokens: (B, L) input token IDs for the LLM. Audio positions are marked with AUDIO_TOKEN_IDX.
        input_token_lens: (B,) lengths of the input token sequences.
        target_tokens: (B, L) target token IDs for the LLM. Non-trainable positions are IGNORE_INDEX.
        target_token_lens: (B,) lengths of the target token sequences.
        text: list of ground-truth transcription strings.
        is_multiturn: (B,) row-local lean-multi-turn indicator.
        num_substantive_turns: (B,) number of substantive target turns in each row.
        boundary_turn_ordinals: (B, L) substantive turn ordinal at SOU/EOU targets, else zero.
        user_backchannel_event_ids: (B, L) user backchannel event ID at SOB/closing-marker targets, else zero.
        user_backchannel_reference_frames: (B, L) annotated fragment boundary frame at SOB/closing-marker
            targets, else -1.
        num_user_backchannel_events: (B,) number of enabled user-backchannel events in each row.
        cuts: Optional[CutSet] containing the cuts for the batch.
    """

    audios: Optional[torch.Tensor] = None
    audio_lens: Optional[torch.Tensor] = None
    input_tokens: Optional[torch.Tensor] = None
    input_token_lens: Optional[torch.Tensor] = None
    target_tokens: Optional[torch.Tensor] = None
    target_token_lens: Optional[torch.Tensor] = None
    text: Optional[List[str]] = None
    is_multiturn: Optional[torch.Tensor] = None
    num_substantive_turns: Optional[torch.Tensor] = None
    boundary_turn_ordinals: Optional[torch.Tensor] = None
    user_backchannel_event_ids: Optional[torch.Tensor] = None
    user_backchannel_reference_frames: Optional[torch.Tensor] = None
    num_user_backchannel_events: Optional[torch.Tensor] = None
    cuts: Optional[CutSet] = None


@dataclass
class StreamingSTTDataConfig:
    sample_rate: int
    frame_length_in_secs: float
    chunk_size: int
    num_delay_frames: int = 0
    words_per_group: int = 1
    audio_tag: str = "<audio>"
    blank_token: str = "<blank>"
    system_role: str = "system"
    system_prompt: str = "Transcribe the audio into text."
    prompt_field: str = "system_prompt"
    compact_template: bool = False
    write_token: str = "<|im_start|>"
    use_text_tokens: bool = False
    text_start_token: str = "<|text_start|>"
    text_end_token: str = "<|text_end|>"
    compact_text_end_only_no_blank: bool = False
    add_utterance_boundary_tokens: bool = False
    use_per_sample_utterance_boundary_tokens: bool = False
    use_per_sample_utterance_boundary_timestamps: bool = False
    utterance_start_token: str = "<sou>"
    utterance_end_token: str = "<eou>"
    # Absolute delays from first-word onset and last-word end. Boundary
    # alignments do not inherit num_delay_frames.
    utterance_start_boundary_delay_frames: int = 2
    utterance_end_boundary_delay_frames: int = 2
    # Opt-in agent-backchannel targets sourced from D7 curation metadata.
    # This is deliberately independent from utterance boundary training.
    enable_agent_backchannels: bool = False
    agent_backchannel_start_token: str = "<soab>"
    agent_backchannel_end_token: str = "<eoab>"
    agent_backchannel_delay_frames: int = 0
    # Row-local user backchannels from supported lean multi-turn metadata.
    # Boundary modes emit <sob> text followed by shared <eou> or dedicated <eob>.
    user_backchannel_mode: str = USER_BACKCHANNEL_MODE_IGNORE
    user_backchannel_start_token: str = "<sob>"
    user_backchannel_end_token: str = "<eob>"
    user_backchannel_start_delay_frames: int = 4
    user_backchannel_end_delay_frames: int = 2
    # Symmetric acoustic context supplied to the online forced aligner around
    # each lean multi-turn substantive region. This does not move the
    # authoritative manifest SOU/EOU timestamps.
    multiturn_forced_alignment_buffer_s: float = 0.5
    # K — only effective in dynamic chunking (chunk_size == 0). Each audio
    # segment is rounded UP to a multiple of K frames (and total audio is
    # padded to K-multiple). The model implicitly learns to emit only at
    # K-aligned positions; deploy-time K' (any multiple of K_train) is set via
    # dynamic_min_chunk_size / dynamic_max_chunk_size. Default 1 = no-op.
    chunk_step: int = 1

    def __post_init__(self):
        if self.utterance_start_boundary_delay_frames < 0:
            raise ValueError("utterance_start_boundary_delay_frames must be non-negative")
        if self.utterance_end_boundary_delay_frames < 0:
            raise ValueError("utterance_end_boundary_delay_frames must be non-negative")
        if self.enable_agent_backchannels:
            if self.agent_backchannel_delay_frames < 0:
                raise ValueError("agent_backchannel_delay_frames must be non-negative")
            if not self.agent_backchannel_start_token or not self.agent_backchannel_end_token:
                raise ValueError("agent backchannel tokens must be non-empty")
            if self.agent_backchannel_start_token == self.agent_backchannel_end_token:
                raise ValueError("agent backchannel start and end tokens must be different")
        if self.user_backchannel_mode not in USER_BACKCHANNEL_MODES:
            raise ValueError(
                f"user_backchannel_mode must be one of {sorted(USER_BACKCHANNEL_MODES)}; "
                f"got {self.user_backchannel_mode!r}"
            )
        if self.user_backchannel_start_delay_frames < 0:
            raise ValueError("user_backchannel_start_delay_frames must be non-negative")
        if self.user_backchannel_end_delay_frames < 0:
            raise ValueError("user_backchannel_end_delay_frames must be non-negative")
        if self.user_backchannel_mode in {
            USER_BACKCHANNEL_MODE_SOB_EOU,
            USER_BACKCHANNEL_MODE_SOB_EOB,
        }:
            if not self.add_utterance_boundary_tokens:
                raise ValueError(
                    f"user_backchannel_mode={self.user_backchannel_mode!r} requires "
                    "add_utterance_boundary_tokens=True"
                )
            if not self.user_backchannel_start_token:
                raise ValueError("user_backchannel_start_token must be non-empty in boundary modes")
            if self.user_backchannel_start_token in {
                self.utterance_start_token,
                self.utterance_end_token,
            }:
                raise ValueError("user backchannel SOB must differ from SOU and EOU")
        if self.user_backchannel_mode == USER_BACKCHANNEL_MODE_SOB_EOB:
            if not self.user_backchannel_end_token:
                raise ValueError("user_backchannel_end_token must be non-empty in sob_eob mode")
            if self.user_backchannel_end_token in {
                self.utterance_start_token,
                self.utterance_end_token,
                self.user_backchannel_start_token,
            }:
                raise ValueError("user backchannel EOB must differ from SOU, EOU, and SOB")
        if (
            isinstance(self.multiturn_forced_alignment_buffer_s, bool)
            or not isinstance(self.multiturn_forced_alignment_buffer_s, Real)
            or not math.isfinite(self.multiturn_forced_alignment_buffer_s)
            or self.multiturn_forced_alignment_buffer_s < 0
        ):
            raise ValueError("multiturn_forced_alignment_buffer_s must be a non-negative finite number")


def _normalize_legacy_text_token_config(cfg: DictConfig | dict) -> DictConfig | dict:
    """Normalize supported legacy dataset configuration keys."""
    if "use_text_tokens" not in cfg and "use_te_tokens" in cfg:
        cfg["use_text_tokens"] = cfg["use_te_tokens"]
    if "text_start_token" not in cfg and "te_start_token" in cfg:
        cfg["text_start_token"] = cfg["te_start_token"]
    if "text_end_token" not in cfg and "te_end_token" in cfg:
        cfg["text_end_token"] = cfg["te_end_token"]
    for legacy_key in ("use_te_tokens", "te_start_token", "te_end_token"):
        if legacy_key in cfg:
            del cfg[legacy_key]

    if "utterance_boundary_delay_frames" in cfg:
        raise ValueError(
            "utterance_boundary_delay_frames is no longer supported; configure "
            "utterance_start_boundary_delay_frames and utterance_end_boundary_delay_frames explicitly"
        )

    if "utterance_boundary_margin_secs" in cfg:
        raise ValueError(
            "utterance_boundary_margin_secs is deprecated and no longer supported; convert it to "
            "utterance_end_boundary_delay_frames in the launch configuration"
        )
    return cfg


def decode_with_blank(
    ids: list[int],
    blank_token: str,
    tokenizer: AutoTokenizer,
    replace_blank: Optional[str] = None,
    strip_whitespace: bool = False,
    collapse_whitespace: bool = True,
    join_with: Optional[str] = " ",
    write_token: Optional[str] = None,
) -> str:
    """Decode token IDs, treating blank tokens as segment boundaries.

    Splits the token sequence at ``blank_token`` boundaries, decodes each
    segment separately (preserving BPE within each turn), then joins with
    spaces.

    Args:
        ids: Token IDs to decode.
        blank_token: The blank token string (e.g., ``"<blank>"``).
        tokenizer: NeMo AutoTokenizer.
        replace_blank: If provided, blank tokens are replaced with this string
            in the output instead of being skipped.  For example,
            ``replace_blank=""`` keeps the spacing, ``replace_blank="..."``
            inserts an ellipsis.
        strip_whitespace: If True, strip whitespace from the output.
        collapse_whitespace: If True, collapse multiple consecutive whitespace characters into a single space.
        join_with: If provided, join the segments divided by blank tokens with this string, else join with empty string.
    """
    if blank_token == "":
        # No blank token: use EOS (e.g. <|im_end|>) as chunk separator so
        # per-chunk outputs get joined with spaces instead of BPE-merged into one run.
        blank_id = tokenizer.tokenizer.eos_token_id
    else:
        blank_id = tokenizer.tokenizer.convert_tokens_to_ids(blank_token)
    write_id = None
    if write_token is not None:
        write_id = tokenizer.tokenizer.convert_tokens_to_ids(write_token)

    segments = []
    current = []
    for tid in ids:
        if tid == blank_id:
            if current:
                segments.append(tokenizer.ids_to_tokens(current))
                current = []
            if replace_blank is not None:
                segments.append(replace_blank)
        elif tid == write_id:
            continue
        else:
            current.append(tid)
    if current:
        segments.append(tokenizer.ids_to_tokens(current))

    text_segments = []
    for seg in segments:
        if isinstance(seg, str):
            text_segments.append(seg)
        else:
            text_segments.append(tokenizer.tokens_to_text(seg, remove_special_tokens=True))
    text = join_with.join(text_segments) if join_with else "".join(text_segments)

    if strip_whitespace:
        text = text.strip()
    if collapse_whitespace:
        text = re.sub(r'\s+', ' ', text)
    return text


def compute_word_spans(
    alignments: List[WordAlignment],
    transcript: str,
    preserve_trailing_whitespace: bool = False,
    preserve_leading_whitespace: bool = False,
) -> List[tuple[int, int]]:
    """Find (start, end) character positions for each alignment word in the transcript.

    Trailing punctuation (non-alphanumeric, non-whitespace characters) that
    immediately follows a word is always included in the span so that commas,
    periods, quotes, etc. are preserved.

    Args:
        alignments: Word-level alignment results.
        transcript: Original transcription string.
        preserve_trailing_whitespace: When True, each span extends through
            trailing whitespace up to (but not including) the next alphanumeric
            character.  This is useful when extracting multi-word spans so
            that ``transcript[first_span[0]:last_span[1]]`` includes the
            inter-word spaces.
        preserve_leading_whitespace: When True, each span extends backward
            through preceding whitespace (not crossing the previous word's
            span end).  This matches GPT-style BPE tokenization where a
            leading space is part of the word token (e.g. ``" world"`` vs
            ``"world"``).  For ``"hello world"`` this yields
            ``[(0,5), (5,11)]`` = ``"hello"``, ``" world"``.

    Returns a list parallel to *alignments*.  If a word cannot be located, its
    span is ``None``.
    """

    if preserve_trailing_whitespace and preserve_leading_whitespace:
        raise ValueError(
            "preserve_trailing_whitespace and preserve_leading_whitespace cannot be True at the same time"
        )
    spans: List[tuple[int, int] | None] = []
    search_pos = 0
    for word in alignments:
        idx = transcript.lower().find(word.text.lower(), search_pos)
        if idx == -1:
            spans.append(None)
            continue
        start = idx
        # Optionally extend start backward through leading whitespace,
        # clamped at the previous word's span end.
        if preserve_leading_whitespace:
            while start > search_pos and transcript[start - 1].isspace():
                start -= 1
        end = idx + len(word.text)
        # Include trailing punctuation (e.g., comma, period, quotes)
        while end < len(transcript) and not transcript[end].isalnum() and not transcript[end].isspace():
            end += 1
        # Optionally include trailing whitespace up to the next word
        if preserve_trailing_whitespace:
            while end < len(transcript) and transcript[end].isspace():
                end += 1
        spans.append((start, end))
        search_pos = end
    return spans


def _alignment_delay_frames(word: WordAlignment, default_delay_frames: int) -> int:
    return default_delay_frames if word.delay_frames is None else word.delay_frames


def _alignment_ready_frame(word: WordAlignment, frame_length_in_secs: float, default_delay_frames: int) -> int:
    return math.ceil(word.end_time / frame_length_in_secs) + _alignment_delay_frames(word, default_delay_frames)


def _validate_timestamp(value, *, field: str, cut_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"Cut {cut_id!r} custom field {field!r} must be a finite number; got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Cut {cut_id!r} custom field {field!r} must be finite; got {value!r}")
    return value


def build_agent_backchannel_alignments(
    custom: Mapping[str, Any],
    *,
    audio_duration_secs: float,
    start_token: str,
    end_token: str,
    delay_frames: int = 0,
    cut_id: str = "<unknown>",
) -> List[WordAlignment]:
    """Build atomic agent-backchannel alignments from D7 curation metadata.

    Only ``confirmed`` fragments that are fully contained in both the extracted
    clip and the primary-speaker utterance are eligible. The metadata text is
    preserved exactly inside one atomic ``<soab>...<eoab>`` alignment scheduled
    from the fragment onset.

    Samples without D7 curation metadata are a no-op. Once a fragment declares
    itself confirmed, malformed required fields are treated as data errors rather
    than silently dropping a supervised event.
    """
    if not isinstance(custom, Mapping):
        raise TypeError(f"Cut {cut_id!r} custom metadata must be a mapping; got {type(custom).__name__}")
    if delay_frames < 0:
        raise ValueError(f"agent backchannel delay_frames must be non-negative; got {delay_frames}")
    if not start_token or not end_token or start_token == end_token:
        raise ValueError("agent backchannel start/end tokens must be non-empty and different")

    curation = custom.get("curation")
    if curation is None:
        return []
    if not isinstance(curation, Mapping):
        raise TypeError(f"Cut {cut_id!r} custom field 'curation' must be a mapping")
    other_speaker = curation.get("other_speaker")
    if other_speaker is None:
        return []
    if not isinstance(other_speaker, Mapping):
        raise TypeError(f"Cut {cut_id!r} curation.other_speaker must be a mapping")
    fragments = other_speaker.get("fragments", [])
    if not isinstance(fragments, list):
        raise TypeError(f"Cut {cut_id!r} curation.other_speaker.fragments must be a list")

    confirmed_fragments: list[tuple[int, Mapping[str, Any]]] = []
    for fragment_idx, fragment in enumerate(fragments):
        if not isinstance(fragment, Mapping):
            raise TypeError(f"Cut {cut_id!r} curation.other_speaker.fragments[{fragment_idx}] must be a mapping")
        backchannel = fragment.get("backchannel")
        if backchannel is None:
            continue
        if not isinstance(backchannel, Mapping):
            raise TypeError(
                f"Cut {cut_id!r} curation.other_speaker.fragments[{fragment_idx}].backchannel must be a mapping"
            )
        if backchannel.get("status") == "confirmed":
            confirmed_fragments.append((fragment_idx, fragment))

    if not confirmed_fragments:
        return []

    duration = _validate_timestamp(audio_duration_secs, field="audio_duration_secs", cut_id=cut_id)
    eligible_ranges: list[tuple[float, float]]
    if curation.get("schema_version") == LEAN_MULTI_TURN_SCHEMA_VERSION:
        target = _require_mapping(curation.get("target"), field="curation.target", cut_id=cut_id)
        regions = _require_list(
            target.get("utterance_regions"), field="curation.target.utterance_regions", cut_id=cut_id
        )
        eligible_ranges = []
        for region_idx, raw_region in enumerate(regions):
            prefix = f"curation.target.utterance_regions[{region_idx}]"
            region = _require_mapping(raw_region, field=prefix, cut_id=cut_id)
            start = _validate_timestamp(region.get("start"), field=f"{prefix}.start", cut_id=cut_id)
            region_duration = _validate_timestamp(region.get("duration"), field=f"{prefix}.duration", cut_id=cut_id)
            eligible_ranges.append((start, start + region_duration))
    else:
        utterance_start = _validate_timestamp(
            custom.get("utterance_start_time"), field="utterance_start_time", cut_id=cut_id
        )
        utterance_end = _validate_timestamp(
            custom.get("utterance_end_time"), field="utterance_end_time", cut_id=cut_id
        )
        eligible_ranges = [(utterance_start, utterance_end)]

    if duration < 0 or any(start < 0 or end < start or end > duration + 1e-6 for start, end in eligible_ranges):
        raise ValueError(
            f"Cut {cut_id!r} requires every target utterance range to satisfy "
            f"0 <= start <= end <= audio duration; got ranges={eligible_ranges}, duration={duration}"
        )

    alignments: list[WordAlignment] = []
    for fragment_idx, fragment in confirmed_fragments:
        prefix = f"curation.other_speaker.fragments[{fragment_idx}]"
        text = fragment.get("text")
        if not isinstance(text, str) or not text.strip():
            raise TypeError(f"Cut {cut_id!r} confirmed {prefix}.text must be a non-empty string; got {text!r}")
        if start_token in text or end_token in text:
            raise ValueError(f"Cut {cut_id!r} confirmed {prefix}.text contains an agent backchannel marker")
        start = _validate_timestamp(fragment.get("start"), field=f"{prefix}.start", cut_id=cut_id)
        fragment_duration = _validate_timestamp(fragment.get("duration"), field=f"{prefix}.duration", cut_id=cut_id)
        if fragment_duration < 0:
            raise ValueError(
                f"Cut {cut_id!r} confirmed {prefix}.duration must be non-negative; got {fragment_duration}"
            )
        fragment_end = start + fragment_duration

        # Valid-but-ineligible edge fragments are intentionally skipped.
        if start < -1e-6 or fragment_end > duration + 1e-6:
            continue
        if not any(
            start >= utterance_start - 1e-6 and fragment_end <= utterance_end + 1e-6
            for utterance_start, utterance_end in eligible_ranges
        ):
            continue

        alignments.append(
            WordAlignment(
                text=f"{start_token}{text}{end_token}",
                start_time=max(0.0, start),
                end_time=max(0.0, start),
                delay_frames=delay_frames,
            )
        )
    return alignments


def merge_agent_backchannel_alignments(
    alignments: List[WordAlignment],
    agent_backchannels: List[WordAlignment],
    *,
    frame_length_in_secs: float,
    default_delay_frames: int,
    utterance_start_token: str = "<sou>",
    utterance_end_token: str = "<eou>",
) -> List[WordAlignment]:
    """Merge transcript and agent-backchannel emissions by effective ready frame.

    Boundary pairs remain chronological for both single- and multi-turn rows.
    At an exact ready-frame tie, SOU precedes transcript, then agent
    backchannels, then EOU; original order otherwise remains stable.
    """
    if not agent_backchannels:
        return alignments
    if frame_length_in_secs <= 0:
        raise ValueError("frame_length_in_secs must be positive")

    tagged = []
    for idx, item in enumerate(alignments):
        if item.text == utterance_start_token:
            priority = 0
        elif item.text == utterance_end_token:
            priority = 3
        else:
            priority = 1
        tagged.append((item, priority, idx))
    tagged.extend((item, 2, len(alignments) + idx) for idx, item in enumerate(agent_backchannels))
    tagged.sort(
        key=lambda tagged: (
            _alignment_ready_frame(tagged[0], frame_length_in_secs, default_delay_frames),
            tagged[1],
            tagged[2],
        )
    )
    return [item for item, _, _ in tagged]


def add_gt_utterance_boundary_alignments(
    alignments: List[WordAlignment],
    *,
    utterance_start_time: float,
    utterance_end_time: float,
    audio_duration_secs: float,
    start_token: str,
    end_token: str,
    start_delay_frames: int,
    end_delay_frames: int,
    cut_id: str = "<unknown>",
) -> List[WordAlignment]:
    """Clip word timing to authoritative GT boundaries and add ordered SOU/EOU alignments.

    SOU/EOU remain fixed at the manifest region boundaries. Each selected
    word endpoint is independently clamped into that closed interval, so a
    small boundary-crossing overlap is retained rather than moving a boundary.
    """
    sou = _validate_timestamp(utterance_start_time, field="utterance_start_time", cut_id=cut_id)
    eou = _validate_timestamp(utterance_end_time, field="utterance_end_time", cut_id=cut_id)
    duration = _validate_timestamp(audio_duration_secs, field="audio_duration_secs", cut_id=cut_id)
    if duration < 0:
        raise ValueError(f"Cut {cut_id!r} audio duration must be non-negative; got {duration}")
    if sou < 0 or eou < sou or eou > duration + 1e-6:
        raise ValueError(
            f"Cut {cut_id!r} requires 0 <= utterance_start_time <= utterance_end_time <= "
            f"audio duration; got start={sou}, end={eou}, duration={duration}"
        )
    eou = min(eou, duration)

    clipped = []
    for word_idx, word in enumerate(alignments):
        start = _validate_timestamp(word.start_time, field=f"alignments[{word_idx}].start_time", cut_id=cut_id)
        end = _validate_timestamp(word.end_time, field=f"alignments[{word_idx}].end_time", cut_id=cut_id)
        if start < 0 or end < start:
            raise ValueError(
                f"Cut {cut_id!r} alignment {word_idx} requires 0 <= start_time <= end_time; "
                f"got start={start}, end={end}"
            )
        clipped.append(
            WordAlignment(
                text=word.text,
                start_time=min(max(start, sou), eou),
                end_time=min(max(end, sou), eou),
                delay_frames=word.delay_frames,
            )
        )

    return [
        WordAlignment(start_token, sou, sou, delay_frames=start_delay_frames),
        *clipped,
        WordAlignment(end_token, eou, eou, delay_frames=end_delay_frames),
    ]


def build_lean_multiturn_alignments(
    sample: MultiTurnSample,
    alignments: List[WordAlignment],
    *,
    audio_duration_secs: float,
    start_token: str,
    end_token: str,
    start_delay_frames: int,
    end_delay_frames: int,
    user_backchannel_start_token: str | None = None,
    user_backchannel_end_token: str | None = None,
    user_backchannel_start_delay_frames: int = 4,
    user_backchannel_end_delay_frames: int = 2,
    cut_id: str = "<unknown>",
) -> List[WordAlignment]:
    """Select target words and add the configured per-segment boundaries.

    A word belongs to the first substantive region containing its midpoint and
    is consumed at most once. Words whose midpoints lie outside every region
    are omitted. A non-empty turn with no selected word alignment is rejected.
    Substantive turns use SOU/EOU. User-backchannel words inherit the default text
    delay and each annotated fragment stays atomic as SOB, words, then the
    configured EOU or EOB marker.
    """
    if not alignments and sample.transcript:
        raise ValueError(f"Cut {cut_id!r} has a non-empty lean multi-turn transcript but no usable word alignments")

    ordered_alignments = sorted(alignments, key=lambda item: (item.start_time, item.end_time))
    consumed: set[int] = set()
    resolved: list[WordAlignment] = []
    for segment_idx, segment in enumerate(sample.segments):
        segment_alignments = []
        for alignment_idx, alignment in enumerate(ordered_alignments):
            if alignment_idx in consumed:
                continue
            start = _validate_timestamp(
                alignment.start_time, field=f"alignments[{alignment_idx}].start_time", cut_id=cut_id
            )
            end = _validate_timestamp(alignment.end_time, field=f"alignments[{alignment_idx}].end_time", cut_id=cut_id)
            if start < 0 or end < start:
                raise ValueError(f"Cut {cut_id!r} alignment {alignment_idx} requires 0 <= start_time <= end_time")
            midpoint = (start + end) / 2
            if segment.start_time - 1e-6 <= midpoint <= segment.end_time + 1e-6:
                consumed.add(alignment_idx)
                segment_alignments.append(alignment)

        if segment.text and not segment_alignments:
            raise ValueError(
                f"Cut {cut_id!r} target segment {segment_idx} ({segment.text!r}) has no usable word alignments"
            )

        if segment.turn_ordinal:
            segment_alignments = add_gt_utterance_boundary_alignments(
                segment_alignments,
                utterance_start_time=segment.start_time,
                utterance_end_time=segment.end_time,
                audio_duration_secs=audio_duration_secs,
                start_token=start_token,
                end_token=end_token,
                start_delay_frames=start_delay_frames,
                end_delay_frames=end_delay_frames,
                cut_id=cut_id,
            )
        elif user_backchannel_start_token is not None and user_backchannel_end_token is not None:
            segment_alignments = [
                WordAlignment(
                    text=alignment.text,
                    start_time=min(max(alignment.start_time, segment.start_time), segment.end_time),
                    end_time=min(max(alignment.end_time, segment.start_time), segment.end_time),
                    delay_frames=None,
                )
                for alignment in segment_alignments
            ]
            segment_alignments = [
                WordAlignment(
                    user_backchannel_start_token,
                    segment.start_time,
                    segment.start_time,
                    delay_frames=user_backchannel_start_delay_frames,
                ),
                *segment_alignments,
                WordAlignment(
                    user_backchannel_end_token,
                    segment.end_time,
                    segment.end_time,
                    delay_frames=user_backchannel_end_delay_frames,
                ),
            ]
        else:
            segment_alignments = [
                WordAlignment(
                    text=alignment.text,
                    start_time=min(max(alignment.start_time, segment.start_time), segment.end_time),
                    end_time=min(max(alignment.end_time, segment.start_time), segment.end_time),
                    delay_frames=None,
                )
                for alignment in segment_alignments
            ]
        resolved.extend(segment_alignments)

    return resolved


def build_multiturn_marker_metadata(
    target_ids: list[int],
    sample: MultiTurnSample,
    *,
    sou_id: int,
    eou_id: int,
    sob_id: int | None,
    user_backchannel_end_id: int | None,
    frame_length_in_secs: float,
    sample_idx: int = 0,
) -> tuple[list[int], list[int], list[int]]:
    """Map emitted marker targets to substantive ordinals and user-backchannel events.

    User-backchannel reference frames come directly from annotated fragment
    boundaries, not word alignments or delayed target positions.
    """
    expected_markers: list[tuple[int, int, int, int]] = []
    for segment in sample.segments:
        start_reference_frame = math.ceil(segment.start_time / frame_length_in_secs)
        end_reference_frame = math.ceil(segment.end_time / frame_length_in_secs)
        if segment.turn_ordinal:
            expected_markers.extend(
                [
                    (sou_id, segment.turn_ordinal, 0, -1),
                    (eou_id, segment.turn_ordinal, 0, -1),
                ]
            )
        elif sob_id is not None and user_backchannel_end_id is not None:
            expected_markers.extend(
                [
                    (sob_id, 0, segment.user_backchannel_event_id, start_reference_frame),
                    (user_backchannel_end_id, 0, segment.user_backchannel_event_id, end_reference_frame),
                ]
            )

    marker_token_ids = {sou_id, eou_id}
    if sob_id is not None:
        marker_token_ids.add(sob_id)
    if user_backchannel_end_id is not None:
        marker_token_ids.add(user_backchannel_end_id)
    marker_indices = [idx for idx, token_id in enumerate(target_ids) if token_id in marker_token_ids]
    actual_markers = [target_ids[idx] for idx in marker_indices]
    expected_token_ids = [marker[0] for marker in expected_markers]
    if actual_markers != expected_token_ids:
        raise RuntimeError(
            f"Multi-turn sample {sample_idx} produced marker sequence {actual_markers}; "
            f"expected {expected_token_ids}"
        )

    boundary_turn_ordinals = [0] * len(target_ids)
    user_backchannel_event_ids = [0] * len(target_ids)
    user_backchannel_reference_frames = [-1] * len(target_ids)
    for target_idx, (_, turn_ordinal, event_id, reference_frame) in zip(
        marker_indices, expected_markers
    ):
        boundary_turn_ordinals[target_idx] = turn_ordinal
        user_backchannel_event_ids[target_idx] = event_id
        user_backchannel_reference_frames[target_idx] = reference_frame
    return boundary_turn_ordinals, user_backchannel_event_ids, user_backchannel_reference_frames


def _append_alignment_text(
    content: str,
    text: str,
    *,
    agent_backchannel_start_token: Optional[str] = None,
    agent_backchannel_end_token: Optional[str] = None,
) -> str:
    if not content:
        return text
    if agent_backchannel_start_token and text.startswith(agent_backchannel_start_token):
        return content + text
    if agent_backchannel_end_token and content.endswith(agent_backchannel_end_token):
        return content + text
    if text.startswith("<") and text.endswith(">"):
        return content + text
    if text.startswith(" ") or content.endswith(" "):
        return content + text
    return content + " " + text


def _build_alignment_content(
    alignments: List[WordAlignment],
    indices: List[int],
    word_spans: Optional[List[tuple[int, int] | None]],
    transcript: Optional[str],
    agent_backchannel_start_token: Optional[str] = None,
    agent_backchannel_end_token: Optional[str] = None,
) -> str:
    content = ""
    for idx in indices:
        span = word_spans[idx] if word_spans and transcript else None
        if span is not None:
            piece = transcript[span[0] : span[1]]
        else:
            piece = alignments[idx].text
        content = _append_alignment_text(
            content,
            piece,
            agent_backchannel_start_token=agent_backchannel_start_token,
            agent_backchannel_end_token=agent_backchannel_end_token,
        )
    return content


def _buffer_has_forced_alignment(alignments: List[WordAlignment], indices: List[int]) -> bool:
    return any(alignments[i].delay_frames is not None for i in indices)


def get_llm_messages_for_sample(
    system_role: str,
    system_prompt: str,
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: int,
    audio_duration_secs: float,
    frame_length_in_secs: float,
    alignments: Optional[List[WordAlignment]] = None,
    transcript: Optional[str] = None,
    words_per_group: int = 1,
    chunk_step: int = 1,
    agent_backchannel_start_token: Optional[str] = None,
    agent_backchannel_end_token: Optional[str] = None,
) -> List[dict]:
    """
    Get the LLM messages for a sample, using the alignments to determine the turns for the audio and text.

    The conversation is structured as alternating user (audio chunks) and assistant (transcription or blank) turns.
    A word becomes "ready" at the chunk whose end frame >= word_end_frame + num_delay_frames.

    For example, if the alignments are:
    [
        WordAlignment(text="Hello", start_time=0.16, end_time=0.48),
        WordAlignment(text="World", start_time=0.60, end_time=0.80),
    ]
    And the audio duration is 1s, audio_tag is "<audio>", chunk_size is 2, frame_length_in_secs is 0.08s,
    num_delay_frames is 0, then the messages will be:
    [
        {"role": "system", "content": "Transcribe the audio into text."},
        {"role": "user", "content": "<audio><audio>"},  # frames 0-1, 0~0.16s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 2-3, 0.16~0.32s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 4-5, 0.32~0.48s
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "<audio><audio>"},  # frames 6-7, 0.48~0.64s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 8-9, 0.64~0.80s
        {"role": "assistant", "content": "World"},
        {"role": "user", "content": "<audio><audio>"},  # frames 10-11, 0.80~0.96s
        {"role": "assistant", "content": "<blank>"},
        {"role": "user", "content": "<audio><audio>"},  # frames 12-13, 0.96~1.12s
        {"role": "assistant", "content": "<blank>"},
    ]

    Note: the last chunk may extend beyond audio_duration_secs since num_frames is
    ceiled to a multiple of chunk_size. The model must pad the audio accordingly.

    Args:
        system_role: The role of the system.
        system_prompt: The prompt for the system.
        audio_tag: The tag for the audio placeholder.
        blank_token: The token for blank/no-emission.
        chunk_size: The number of frames per chunk. If -1, the whole audio is used as a single chunk.
        num_delay_frames: Number of frames to delay word emission after word end.
        audio_duration_secs: The duration of the audio in seconds.
        frame_length_in_secs: The length of a single frame in seconds.
        alignments: List of WordAlignment objects for the sample.
        agent_backchannel_start_token: Optional marker used to avoid synthesizing
            whitespace immediately before an agent backchannel span.
        agent_backchannel_end_token: Optional marker used to avoid synthesizing
            whitespace immediately after an agent backchannel span.
    """

    messages = [{"role": system_role, "content": system_prompt}]

    num_frames = math.ceil(audio_duration_secs / frame_length_in_secs)

    if chunk_size < 0 or chunk_size is None:
        # Offline mode: use the whole audio as a single chunk
        num_chunks = 1 if num_frames > 0 else 0
        chunk_size = num_frames
        offline_mode = True
        num_delay_frames = 0  # delay is not used in offline mode
    else:
        offline_mode = False

    if alignments is None:
        alignments = []

    if offline_mode and not alignments:
        messages.append({"role": "user", "content": audio_tag * num_frames})
        messages.append({"role": "assistant", "content": transcript if transcript is not None else blank_token})
        return messages

    # Pre-compute word character spans if transcript is provided.
    word_spans = compute_word_spans(alignments, transcript, preserve_leading_whitespace=True) if transcript else None

    if chunk_size == 0:
        # Dynamic chunking: one user turn per word group, sized to word boundary.
        # The model learns to predict when to stop listening via audio-position targets.
        # When chunk_step > 1, each segment's frame count is rounded UP to a
        # multiple of K so the model only ever emits at K-aligned positions.
        K = max(int(chunk_step), 1)
        prev_end_frame = 0
        word_buffer: list[int] = []  # indices of buffered words

        for word_idx, word in enumerate(alignments):
            word_buffer.append(word_idx)

            # Emit when buffer reaches words_per_group or this is the last word
            if (
                len(word_buffer) < words_per_group
                and word_idx < len(alignments) - 1
                and not _buffer_has_forced_alignment(alignments, word_buffer)
            ):
                continue

            # Chunk boundary = end frame of the last word in this group, snapped
            # UP to the next multiple of K. num_frames here is already K-padded
            # (caller guarantees this), so the clamp keeps things K-aligned.
            group_end_frame = max(
                _alignment_ready_frame(alignments[i], frame_length_in_secs, num_delay_frames) for i in word_buffer
            )
            if K > 1:
                group_end_frame = ((group_end_frame + K - 1) // K) * K
            group_end_frame = min(group_end_frame, num_frames)
            n_frames_chunk = group_end_frame - prev_end_frame

            if n_frames_chunk > 0:
                messages.append({"role": "user", "content": audio_tag * n_frames_chunk})

            # Build assistant content from all buffered words.
            content = _build_alignment_content(
                alignments,
                word_buffer,
                word_spans,
                transcript,
                agent_backchannel_start_token=agent_backchannel_start_token,
                agent_backchannel_end_token=agent_backchannel_end_token,
            )

            if n_frames_chunk <= 0 and messages[-1]["role"] == "assistant":
                # Words at same boundary as previous group — append
                messages[-1]["content"] = _append_alignment_text(
                    messages[-1]["content"],
                    content,
                    agent_backchannel_start_token=agent_backchannel_start_token,
                    agent_backchannel_end_token=agent_backchannel_end_token,
                )
            else:
                messages.append({"role": "assistant", "content": content})

            prev_end_frame = group_end_frame
            word_buffer = []

        # Trailing silence frames (after last word) — user turn only, no assistant.
        if prev_end_frame < num_frames:
            messages.append({"role": "user", "content": audio_tag * (num_frames - prev_end_frame)})
    else:
        # Fixed chunking: split the audio into equal-sized chunks.
        num_chunks = math.ceil(num_frames / chunk_size) if num_frames > 0 else 0

        word_idx = 0
        word_buffer: list[int] = []  # indices of words buffered for words_per_group grouping
        for chunk_i in range(num_chunks):
            chunk_end_frame = (chunk_i + 1) * chunk_size

            # User turn: one audio tag per frame in the chunk
            messages.append({"role": "user", "content": audio_tag * chunk_size})

            # Collect indices of words whose end_time (in frames) + delay <= chunk_end_frame
            while word_idx < len(alignments):
                word = alignments[word_idx]
                ready_frame = _alignment_ready_frame(word, frame_length_in_secs, num_delay_frames)
                if ready_frame <= chunk_end_frame:
                    word_buffer.append(word_idx)
                    word_idx += 1
                else:
                    break

            # Emit words when buffer reaches words_per_group, or at the last chunk
            is_last_chunk = chunk_i == num_chunks - 1
            if word_buffer and (
                len(word_buffer) >= words_per_group
                or is_last_chunk
                or _buffer_has_forced_alignment(alignments, word_buffer)
            ):
                content = _build_alignment_content(
                    alignments,
                    word_buffer,
                    word_spans,
                    transcript,
                    agent_backchannel_start_token=agent_backchannel_start_token,
                    agent_backchannel_end_token=agent_backchannel_end_token,
                )
                messages.append({"role": "assistant", "content": content})
                word_buffer = []
            else:
                messages.append({"role": "assistant", "content": blank_token})

        # Append any residual words that weren't emitted (e.g., due to delay pushing
        # them past the last chunk boundary, or alignment end_time > audio_duration).
        if word_idx < len(alignments):
            residual_indices = list(range(word_idx, len(alignments)))
            content = _build_alignment_content(
                alignments,
                residual_indices,
                word_spans,
                transcript,
                agent_backchannel_start_token=agent_backchannel_start_token,
                agent_backchannel_end_token=agent_backchannel_end_token,
            )
            if messages[-1]["role"] == "assistant" and messages[-1]["content"] == blank_token:
                messages[-1]["content"] = content
            elif messages[-1]["role"] == "assistant":
                messages[-1]["content"] = _append_alignment_text(
                    messages[-1]["content"],
                    content,
                    agent_backchannel_start_token=agent_backchannel_start_token,
                    agent_backchannel_end_token=agent_backchannel_end_token,
                )
            else:
                messages.append({"role": "assistant", "content": content})

    return messages


def get_llm_messages_for_batch(
    system_role: str,
    system_prompt: List[str],
    audio_tag: str,
    blank_token: str,
    chunk_size: int,
    num_delay_frames: int,
    audio_durations_secs: List[float],
    frame_length_in_secs: float,
    alignments: Optional[List[List[WordAlignment]]] = None,
    transcripts: Optional[List[str]] = None,
    words_per_group: int = 1,
    chunk_step: int = 1,
    agent_backchannel_start_token: Optional[str] = None,
    agent_backchannel_end_token: Optional[str] = None,
) -> List[List[dict]]:
    """
    Get the LLM messages for a batch of samples.

    Args:
        system_role: The role of the system.
        system_prompt: The list of prompts for each sample in the batch.
        audio_tag: The tag for the audio placeholder.
        blank_token: The token for blank/no-emission.
        chunk_size: The number of frames per chunk.
        num_delay_frames: Number of frames to delay word emission after word end.
        audio_durations_secs: List of audio durations in seconds, one per sample.
        frame_length_in_secs: The length of a single frame in seconds.
        alignments: List of lists of WordAlignment objects for the batch.
        transcripts: Original transcription strings, one per sample.  When provided,
            assistant turn content preserves punctuation and spacing from the transcript.
        words_per_group: Minimum number of words to buffer before emitting an
            assistant turn (default 1 = emit each word immediately).
        agent_backchannel_start_token: Optional agent-backchannel start marker.
        agent_backchannel_end_token: Optional agent-backchannel end marker.
    """
    if transcripts is None:
        transcripts = [None] * len(audio_durations_secs)
    batch_messages = []
    for sample_alignments, duration_secs, prompt, transcript in zip(
        alignments,
        audio_durations_secs,
        system_prompt,
        transcripts,
    ):
        batch_messages.append(
            get_llm_messages_for_sample(
                system_role=system_role,
                system_prompt=prompt,
                audio_tag=audio_tag,
                blank_token=blank_token,
                chunk_size=chunk_size,
                num_delay_frames=num_delay_frames,
                audio_duration_secs=duration_secs,
                frame_length_in_secs=frame_length_in_secs,
                alignments=sample_alignments,
                transcript=transcript,
                words_per_group=words_per_group,
                chunk_step=chunk_step,
                agent_backchannel_start_token=agent_backchannel_start_token,
                agent_backchannel_end_token=agent_backchannel_end_token,
            )
        )
    return batch_messages


def parse_chat_template_ids(hf_tok, last_turn: bool = False) -> tuple[list[int], list[int], list[int]]:
    """Discover turn-structure token IDs from a HuggingFace chat template.

    Extracts the structural token IDs that surround user and assistant content
    in the chat template.  Uses a 2-message sentinel conversation (1 user +
    1 assistant) to get the ``user_header``, ``asst_footer``, and the full
    ``user_footer_and_asst_header`` (which may include Qwen3-style
    ``<think>...</think>`` suppression tags).

    When ``last_turn=False``, a second 4-message sentinel is used to obtain
    the assistant header *without* thinking tags — Qwen3 only injects them on
    the last assistant turn, and in streaming each chunk is a non-final turn.

    When ``last_turn=True``, the 2-message result is returned as-is, since the
    assistant turn IS the last turn and must include thinking suppression tags
    to match training.

    Args:
        hf_tok: A HuggingFace tokenizer (``tokenizer.tokenizer``).
        last_turn: When True, the extracted assistant header corresponds to the
            last turn in the conversation, which may include thinking
            suppression tags (e.g. for single-turn offline inference).

    Returns:
        ``(user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids)``

        - *user_header_ids*: tokens before user content, BOS stripped
          (e.g. ``[<|im_start|>, user, \\n]``).
        - *user_footer_and_asst_header_ids*: tokens between user content and
          assistant content.
        - *asst_footer_ids*: tokens after assistant content
          (e.g. ``[<|im_end|>, \\n]``).
    """
    _SENTINEL = "XSENTINELX"

    # --- 2-message template: correct footer, full assistant header ---
    convo_2msg = hf_tok.apply_chat_template(
        [
            {"role": "user", "content": _SENTINEL},
            {"role": "assistant", "content": _SENTINEL},
        ],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    parts = convo_2msg.split(_SENTINEL)
    assert len(parts) >= 3, f"Expected >=3 parts after splitting on sentinel, got {len(parts)}: {parts}"

    user_header_ids = hf_tok.encode(parts[0], add_special_tokens=False)
    asst_footer_ids = hf_tok.encode(parts[2], add_special_tokens=False) if parts[2].strip() else []

    # Strip leading BOS from user header — it is already in the KV cache
    # from the system prompt during inference.
    bos_id = getattr(hf_tok, "bos_token_id", None)
    if user_header_ids and bos_id is not None and user_header_ids[0] == bos_id:
        user_header_ids = user_header_ids[1:]

    if last_turn:
        # Last turn: use the 2-msg assistant header (includes thinking tags).
        user_footer_and_asst_header_ids = hf_tok.encode(parts[1], add_special_tokens=False)
    else:
        # Non-last turn: use the 4-msg assistant header (no thinking tags).
        # The 4-msg trick places the sentinel on the first assistant turn,
        # which is NOT the last turn → Qwen3 omits thinking tags.
        convo_4msg = hf_tok.apply_chat_template(
            [
                {"role": "user", "content": _SENTINEL},
                {"role": "assistant", "content": _SENTINEL},
                {"role": "user", "content": "x"},
                {"role": "assistant", "content": "x"},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        parts_4msg = convo_4msg.split(_SENTINEL)
        assert len(parts_4msg) >= 3
        user_footer_and_asst_header_ids = hf_tok.encode(parts_4msg[1], add_special_tokens=False)

    return user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids


def build_compact_turn_markers(
    hf_tok,
    write_token: Optional[str],
    end_token: Optional[str] = None,
) -> tuple[list[int], list[int], list[int]]:
    """Return the compact-format analogue of ``parse_chat_template_ids``.

    Compact format drops the user/assistant role delimiters: turns look like
    ``<audio>*N <write_token> TEXT <end_token>`` with no header before audio and
    the ``write_token`` marking the audio→text transition.  The turn-end is
    the tokenizer's native EOS when ``end_token`` is not provided.

    ``write_token`` should be an existing vocab token the LLM saw pretraining
    as a turn-boundary marker (e.g. ``"<|im_start|>"`` for Qwen3,
    ``"<start_of_turn>"`` for Gemma).
    """
    if write_token is None:
        write_ids = []
    else:
        write_ids = hf_tok.encode(write_token, add_special_tokens=False)
        if len(write_ids) != 1:
            raise ValueError(
                f"write_token {write_token!r} must encode to exactly 1 token, got {write_ids}. "
                f"Pick a tokenizer-native turn-boundary token or override via config."
            )
    if end_token is None:
        end_id = getattr(hf_tok, "eos_token_id", None)
        if end_id is None:
            raise ValueError("tokenizer.eos_token_id is required for compact_template=True without end_token")
    else:
        end_ids = hf_tok.encode(end_token, add_special_tokens=False)
        if len(end_ids) != 1:
            raise ValueError(
                f"end_token {end_token!r} must encode to exactly 1 token, got {end_ids}. "
                f"Pick a tokenizer-native turn-boundary token or add it as a special token."
            )
        end_id = end_ids[0]
    return [], write_ids, [end_id]


def _tokenize_compact_with_assistant_mask(
    messages: List[dict],
    tokenizer: AutoTokenizer,
    write_id: Optional[int],
    eos_id: int,
    blank_token: Optional[str] = None,
    suppress_blank: bool = False,
) -> tuple[list[int], list[int]]:
    """Tokenize chat messages in compact format and return (input_ids, assistant_mask).

    Compact per-turn layout (no role wrapping between audio and text):
        [system_wrapped] [user_content, <write>, asst_content, <eos>]*K

    The system prompt IS still wrapped via ``apply_chat_template`` (Qwen3 system
    block), only the per-turn scaffolding is compacted.  Loss is applied on
    ``<write>``, assistant content, and ``<eos>`` — mirroring the HF path where
    the ``<|im_end|>\\n`` footer is trainable.
    """
    hf_tok = tokenizer.tokenizer

    input_ids: list[int] = []
    assistant_mask: list[int] = []

    # --- System section: keep Qwen3-style wrapping ---
    system_msgs = [m for m in messages if m["role"] == "system"]
    if system_msgs:
        system_ids = hf_tok.apply_chat_template(
            system_msgs,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        input_ids.extend(list(system_ids))
        assistant_mask.extend([0] * len(system_ids))

    # --- Per-turn compact encoding ---
    turn_msgs = [m for m in messages if m["role"] != "system"]
    # Pairs: (user, assistant). The final turn may be user-only (trailing silence).
    i = 0
    while i < len(turn_msgs):
        msg = turn_msgs[i]
        if msg["role"] == "user":
            user_ids = hf_tok.encode(msg["content"], add_special_tokens=False) if msg["content"] else []
            input_ids.extend(user_ids)
            assistant_mask.extend([0] * len(user_ids))
            i += 1
            # Pair with following assistant turn if present.
            if i < len(turn_msgs) and turn_msgs[i]["role"] == "assistant":
                asst = turn_msgs[i]
                asst_content = "" if suppress_blank and asst["content"] == blank_token else asst["content"]
                asst_ids = hf_tok.encode(asst_content, add_special_tokens=False) if asst_content else []
                # Optional text-start marker.
                if write_id is not None:
                    input_ids.append(write_id)
                    assistant_mask.append(1)
                # assistant content
                input_ids.extend(asst_ids)
                assistant_mask.extend([1] * len(asst_ids))
                # eos
                input_ids.append(eos_id)
                assistant_mask.append(1)
                i += 1
        else:
            # Orphan assistant (shouldn't normally occur) — treat as standalone asst segment.
            asst_content = "" if suppress_blank and msg["content"] == blank_token else msg["content"]
            asst_ids = hf_tok.encode(asst_content, add_special_tokens=False) if asst_content else []
            if write_id is not None:
                input_ids.append(write_id)
                assistant_mask.append(1)
            input_ids.extend(asst_ids)
            assistant_mask.extend([1] * len(asst_ids))
            input_ids.append(eos_id)
            assistant_mask.append(1)
            i += 1

    return input_ids, assistant_mask


def _tokenize_with_assistant_mask(
    messages: List[dict],
    tokenizer: AutoTokenizer,
) -> tuple[list[int], list[int]]:
    """
    Tokenize chat messages and return (input_ids, assistant_mask).

    First tries HF's ``return_assistant_tokens_mask`` (requires ``{% generation %}``
    in the chat template).  If that returns an all-zero mask, falls back to a
    sequential-search strategy: tokenize each assistant turn's content separately
    and locate it in the full token sequence.

    Args:
        messages: list of ``{"role": ..., "content": ...}`` dicts.
        tokenizer: NeMo AutoTokenizer (``tokenizer.tokenizer`` is the HF tokenizer).

    Returns:
        (input_ids, assistant_mask) — both plain Python lists of ints.
    """
    hf_tok = tokenizer.tokenizer

    # --- primary path: use HF's built-in mask ---
    result = hf_tok.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        enable_thinking=False,
    )
    input_ids = list(result["input_ids"])
    assistant_mask = list(result["assistant_masks"])

    if any(assistant_mask):
        return input_ids, assistant_mask

    # --- fallback: diff-based content detection ---
    # Tokenize the same messages but with all assistant contents replaced by
    # a single-character sentinel.  The two-pointer walk then identifies
    # content tokens (present in full but replaced by the sentinel in the
    # reference).
    #
    # We use a sentinel instead of "" (empty string) to preserve BPE context
    # boundaries.  With "", template tokens adjacent to the content can merge
    # (e.g. "assistant\n" + content "\n" → token "\n\n" vs "assistant\n" + "" →
    # token "\n"), causing the two-pointer to desync.  A sentinel like "X"
    # tokenizes to exactly 1 token and prevents BPE merging with neighbors.
    _SENTINEL_CHAR = "X"
    assistant_mask = [0] * len(input_ids)

    msgs_sentinel = [{**m, "content": _SENTINEL_CHAR} if m["role"] == "assistant" else m for m in messages]
    ids_sentinel_result = hf_tok.apply_chat_template(
        msgs_sentinel,
        tokenize=True,
        enable_thinking=False,
    )
    ids_sentinel = list(
        ids_sentinel_result["input_ids"] if hasattr(ids_sentinel_result, "keys") else ids_sentinel_result
    )

    eos_id = getattr(hf_tok, 'eos_token_id', None)
    i, j = 0, 0  # pointers into input_ids and ids_sentinel
    while i < len(input_ids) and j < len(ids_sentinel):
        if input_ids[i] == ids_sentinel[j]:
            i += 1
            j += 1
        else:
            # Divergence: ids_sentinel has the sentinel (1 token) where
            # input_ids has the actual content (1+ tokens).
            j += 1  # skip the sentinel token
            while i < len(input_ids) and (j >= len(ids_sentinel) or input_ids[i] != ids_sentinel[j]):
                assistant_mask[i] = 1
                i += 1
            # Include EOS token in the footer so the model learns to emit it.
            if eos_id is not None and i < len(input_ids) and input_ids[i] == eos_id:
                assistant_mask[i] = 1

    # Any remaining tokens in input_ids are also content.
    while i < len(input_ids):
        assistant_mask[i] = 1
        i += 1

    return input_ids, assistant_mask


def _replace_audio_chunks(
    token_ids: list[int],
    chunk_ids: list[int],
    chunk_size: int,
    mask: list | None = None,
) -> list[int] | tuple[list[int], list]:
    """Replace each occurrence of *chunk_ids* with *chunk_size* copies of ``AUDIO_TOKEN_IDX``.

    This handles multi-token audio tags where BPE merges tokens across adjacent
    tags (e.g., ``<audio><audio>`` tokenizes differently from ``encode("<audio>") * 2``).
    By matching the full chunk at once, we avoid the BPE boundary problem.

    When *mask* is provided it is adjusted in sync: each matched span is replaced
    with *chunk_size* copies of the first element of that span (typically 0 for
    user-turn content).

    Returns:
        new_token_ids            when mask is None
        (new_token_ids, new_mask) when mask is provided
    """
    chunk_len = len(chunk_ids)
    new_ids: list[int] = []
    new_mask: list | None = [] if mask is not None else None
    i = 0
    n = len(token_ids)
    while i < n:
        if token_ids[i : i + chunk_len] == chunk_ids:
            new_ids.extend([AUDIO_TOKEN_IDX] * chunk_size)
            if new_mask is not None:
                new_mask.extend([mask[i]] * chunk_size)
            i += chunk_len
        else:
            new_ids.append(token_ids[i])
            if new_mask is not None:
                new_mask.append(mask[i])
            i += 1
    return (new_ids, new_mask) if mask is not None else new_ids


class StreamingSTTDataset(torch.utils.data.Dataset):
    """
    Dataset for StreamingSTTModel.
    Operates directly on Lhotse Cuts (no NeMoMultimodalConversation wrapper).
    """

    def __init__(self, cfg: DictConfig | dict, tokenizer: AutoTokenizer, defer_get_batch: bool = False):
        """
        Args:
            cfg: Configuration for the dataset.
            tokenizer: Tokenizer for the dataset.
            defer_get_batch: If True, defer the get_batch_data call to the __getitem__ method and let the model do it.
                This is used in online forced alignment mode.
        """
        self.defer_get_batch = defer_get_batch
        self.tokenizer = tokenizer
        self.cfg: StreamingSTTDataConfig = to_dataclass(
            StreamingSTTDataConfig, _normalize_legacy_text_token_config(cfg)
        )
        if self.cfg.compact_text_end_only_no_blank and not self.cfg.compact_template:
            raise ValueError("compact_text_end_only_no_blank=True requires compact_template=True")
        if (
            self.cfg.use_per_sample_utterance_boundary_timestamps
            and not self.cfg.use_per_sample_utterance_boundary_tokens
        ):
            raise ValueError(
                "use_per_sample_utterance_boundary_timestamps=True requires "
                "use_per_sample_utterance_boundary_tokens=True"
            )
        self._warned_missing_gt_boundary_cut_ids: set[str] = set()
        # Unescape Python escape sequences (e.g. "\\n" → "\n") because Hydra/OmegaConf
        # loads YAML strings literally without interpreting backslash escapes.
        self.cfg.blank_token = self.cfg.blank_token.encode().decode('unicode_escape')
        if self.cfg.compact_text_end_only_no_blank:
            self.cfg.blank_token = ""

        # Tokenize the full audio chunk string (audio_tag * chunk_size) to get
        # its token ID sequence.  We must encode the full chunk as a single string
        # because BPE may merge tokens across adjacent audio tags (e.g.,
        # "<audio><audio>" tokenizes differently from encode("<audio>") * 2).
        # When chunk_size=-1 (offline mode), audio_chunk_ids is computed per sample
        # in get_batch_data because num_frames varies per sample.
        if self.cfg.chunk_size > 0:
            audio_chunk_str = self.cfg.audio_tag * self.cfg.chunk_size
            self.audio_chunk_ids = self.tokenizer.tokenizer.encode(audio_chunk_str, add_special_tokens=False)
        else:
            self.audio_chunk_ids = None

        # blank_token is part of the LLM output vocabulary — it must be a single
        # special token, otherwise loss is dominated by multi-token blanks and
        # generation becomes unreliable.  The model's __init__ should have called
        # tokenizer.add_special_tokens() before passing the tokenizer here.
        # An empty blank_token ("") disables the explicit blank: chunks without
        # words get empty assistant turns, stop signal is <|im_end|> alone.
        if self.cfg.blank_token == "":
            if self.cfg.chunk_size == 0:
                raise ValueError(
                    "blank_token='' is not supported with dynamic chunking (chunk_size=0) — "
                    "dynamic chunking requires a token to predict at non-final audio positions."
                )
            self.blank_id = -1
            logging.info("blank_token is empty: blank token mechanism disabled (fixed chunking only)")
        else:
            blank_ids = self.tokenizer.tokenizer.encode(self.cfg.blank_token, add_special_tokens=False)
            logging.info(f"blank_token: {str(self.cfg.blank_token)}, blank_id: {blank_ids}")
            if len(blank_ids) != 1:
                raise ValueError(
                    f"blank_token '{self.cfg.blank_token}' tokenizes into {len(blank_ids)} tokens {blank_ids}. "
                    f"It must be a single special token. Make sure the model adds it via "
                    f"tokenizer.add_special_tokens() before constructing the dataset."
                )
            self.blank_id = blank_ids[0]

        # Compact template: cache write_id and eos_id. Skip the parse_chat_template_ids
        # call since we derive the markers directly from config.
        if self.cfg.compact_template:
            hf_tok = self.tokenizer.tokenizer
            write_token = (
                None
                if self.cfg.compact_text_end_only_no_blank
                else self.cfg.text_start_token if self.cfg.use_text_tokens else self.cfg.write_token
            )
            end_token = self.cfg.text_end_token if self.cfg.use_text_tokens else None
            _, ufah_ids, af_ids = build_compact_turn_markers(hf_tok, write_token, end_token=end_token)
            self._write_id = ufah_ids[0] if ufah_ids else None
            self._compact_eos_id = af_ids[0]
            logging.info(
                f"compact_template enabled: write_token={write_token!r} "
                f"(id={self._write_id}), end_token={end_token!r}, end_id={self._compact_eos_id}"
            )
        else:
            self._write_id = None
            self._compact_eos_id = None

        # For dynamic chunking (chunk_size=0): cache the first token of the
        # user footer sequence (e.g. <|im_end|>).  This is the target the model
        # predicts at the last audio frame of each chunk to signal "ready to transcribe".
        if self.cfg.chunk_size == 0:
            if self.cfg.compact_template:
                # Compact: boundary target is write_id (<|im_start|> in Qwen3).
                self._user_footer_first_id = self._write_id if self._write_id is not None else self._compact_eos_id
            else:
                hf_tok = self.tokenizer.tokenizer
                _, user_footer_and_asst_header_ids, _ = parse_chat_template_ids(hf_tok)
                self._user_footer_first_id = user_footer_and_asst_header_ids[0]
        else:
            self._user_footer_first_id = None

        self._sou_id = None
        self._eou_id = None
        self._sob_id = None
        self._eob_id = None
        self._user_backchannel_end_id = None
        if self.cfg.add_utterance_boundary_tokens:
            sou_ids = self.tokenizer.tokenizer.encode(self.cfg.utterance_start_token, add_special_tokens=False)
            eou_ids = self.tokenizer.tokenizer.encode(self.cfg.utterance_end_token, add_special_tokens=False)
            if len(sou_ids) != 1 or len(eou_ids) != 1:
                raise ValueError(
                    "utterance boundary markers must each encode to one token; "
                    f"got {self.cfg.utterance_start_token!r}->{sou_ids}, "
                    f"{self.cfg.utterance_end_token!r}->{eou_ids}"
                )
            self._sou_id, self._eou_id = sou_ids[0], eou_ids[0]
        if self.cfg.user_backchannel_mode in {
            USER_BACKCHANNEL_MODE_SOB_EOU,
            USER_BACKCHANNEL_MODE_SOB_EOB,
        }:
            sob_ids = self.tokenizer.tokenizer.encode(
                self.cfg.user_backchannel_start_token, add_special_tokens=False
            )
            if len(sob_ids) != 1:
                raise ValueError(
                    "user backchannel start marker must encode to one token; "
                    f"got {self.cfg.user_backchannel_start_token!r}->{sob_ids}"
                )
            self._sob_id = sob_ids[0]
            if self.cfg.user_backchannel_mode == USER_BACKCHANNEL_MODE_SOB_EOU:
                self._user_backchannel_end_id = self._eou_id
            else:
                eob_ids = self.tokenizer.tokenizer.encode(
                    self.cfg.user_backchannel_end_token, add_special_tokens=False
                )
                if len(eob_ids) != 1:
                    raise ValueError(
                        "user backchannel end marker must encode to one token; "
                        f"got {self.cfg.user_backchannel_end_token!r}->{eob_ids}"
                    )
                self._eob_id = eob_ids[0]
                self._user_backchannel_end_id = self._eob_id

    def _get_multiturn_samples(
        self,
        cuts: CutSet | list,
        audio_durations_secs: List[float],
    ) -> List[MultiTurnSample | None]:
        samples = []
        for cut, duration_secs in zip(cuts, audio_durations_secs):
            cut_id = str(getattr(cut, "id", "<unknown>"))
            samples.append(
                parse_lean_multiturn_metadata(
                    cut.custom or {},
                    audio_duration_secs=duration_secs,
                    cut_id=cut_id,
                    user_backchannel_mode=self.cfg.user_backchannel_mode,
                )
            )
        return samples

    @staticmethod
    def _resolve_multiturn_transcripts(
        text: List[str],
        multiturn_samples: List[MultiTurnSample | None],
    ) -> List[str]:
        return [
            sample.transcript if sample is not None else transcript
            for transcript, sample in zip(text, multiturn_samples)
        ]

    def get_online_alignments(
        self,
        *,
        cuts: CutSet | list,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        text: List[str],
        forced_aligner: ForcedAligner,
    ) -> List[List[WordAlignment]]:
        """Batch legacy utterances and buffered row-local target regions in one aligner call.

        Legacy rows are aligned over their complete clips. Each lean multi-turn
        substantive region is aligned independently with up to
        ``multiturn_forced_alignment_buffer_s`` of real audio on either side,
        clipped to the sample bounds. The buffer supplies acoustic context only;
        manifest region starts and ends remain the authoritative SOU/EOU times.
        User-backchannel fragments use their exact annotated spans so their metadata
        remains an atomic fallback when the forced aligner cannot resolve them.
        Returned turn-local word timestamps are shifted back into coordinates of
        the complete sampled clip.
        """
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()
        multiturn_samples = self._get_multiturn_samples(cuts, audio_durations_secs)
        resolved_text = self._resolve_multiturn_transcripts(text, multiturn_samples)

        alignment_audios: list[torch.Tensor] = []
        alignment_lens: list[int] = []
        alignment_text: list[str] = []
        owners: list[tuple[int, float, str, MultiTurnTargetSegment | None]] = []
        alignment_buffer_samples = round(self.cfg.multiturn_forced_alignment_buffer_s * self.cfg.sample_rate)
        for sample_idx, (cut, sample, transcript) in enumerate(zip(cuts, multiturn_samples, resolved_text)):
            cut_id = str(getattr(cut, "id", "<unknown>"))
            sample_num_samples = int(audio_lens[sample_idx].item())
            if sample is None:
                alignment_audios.append(audios[sample_idx, :sample_num_samples])
                alignment_lens.append(sample_num_samples)
                alignment_text.append(transcript)
                owners.append((sample_idx, 0.0, cut_id, None))
                continue

            for segment in sample.segments:
                region_start_sample = round(segment.start_time * self.cfg.sample_rate)
                region_end_sample = round(segment.end_time * self.cfg.sample_rate)
                if segment.user_backchannel_event_id:
                    start_sample = max(0, region_start_sample)
                    end_sample = min(sample_num_samples, region_end_sample)
                else:
                    start_sample = max(0, region_start_sample - alignment_buffer_samples)
                    end_sample = min(sample_num_samples, region_end_sample + alignment_buffer_samples)
                if end_sample <= start_sample:
                    raise ValueError(f"Cut {cut_id!r} target segment {segment.text!r} has no audio samples")
                alignment_audios.append(audios[sample_idx, start_sample:end_sample])
                alignment_lens.append(end_sample - start_sample)
                alignment_text.append(segment.text)
                owners.append((sample_idx, start_sample / self.cfg.sample_rate, cut_id, segment))

        if not alignment_audios:
            return [[] for _ in text]

        padded_audio = pad_sequence(alignment_audios, batch_first=True)
        padded_lens = torch.tensor(alignment_lens, dtype=audio_lens.dtype, device=audio_lens.device)
        flat_alignments = forced_aligner.align(padded_audio, padded_lens, alignment_text)
        if len(flat_alignments) != len(owners):
            raise RuntimeError(
                f"Forced aligner returned {len(flat_alignments)} results for {len(owners)} target segments"
            )

        batch_alignments: list[list[WordAlignment]] = [[] for _ in text]
        for segment_alignments, (sample_idx, offset_secs, cut_id, segment) in zip(flat_alignments, owners):
            if segment is not None and segment.user_backchannel_event_id:
                # Very short user backchannels (for example, "Mm.") can be
                # unalignable or receive a nonempty forced-alignment result
                # whose timestamps fall outside the fragment. Their confirmed
                # metadata supplies authoritative text and fragment boundaries.
                # Keep substantive turns strict, but preserve an unusable
                # backchannel result as one atomic annotated alignment.
                fragment_duration = segment.end_time - segment.start_time
                usable_user_backchannel_result = bool(segment_alignments)
                for alignment in segment_alignments:
                    start = alignment.start_time
                    end = alignment.end_time
                    usable_user_backchannel_result = usable_user_backchannel_result and (
                        isinstance(start, Real)
                        and not isinstance(start, bool)
                        and math.isfinite(float(start))
                        and isinstance(end, Real)
                        and not isinstance(end, bool)
                        and math.isfinite(float(end))
                        and 0.0 <= float(start) <= float(end)
                        and float(end) <= fragment_duration + 1e-6
                        and (float(start) + float(end)) / 2 <= fragment_duration + 1e-6
                    )
                if not usable_user_backchannel_result:
                    logging.warning(
                        "Cut %r user backchannel event %d (%r) produced no usable forced alignment "
                        "from %d returned entries; using its annotated fragment timing",
                        cut_id,
                        segment.user_backchannel_event_id,
                        segment.text,
                        len(segment_alignments),
                    )
                    segment_alignments = [
                        WordAlignment(
                            text=segment.text,
                            start_time=0.0,
                            end_time=fragment_duration,
                        )
                    ]
            batch_alignments[sample_idx].extend(
                WordAlignment(
                    text=alignment.text,
                    start_time=alignment.start_time + offset_secs,
                    end_time=alignment.end_time + offset_secs,
                    delay_frames=alignment.delay_frames,
                )
                for alignment in segment_alignments
            )
        for sample_alignments in batch_alignments:
            sample_alignments.sort(key=lambda item: (item.start_time, item.end_time))
        return batch_alignments

    def __getitem__(self, cuts: CutSet) -> StreamingSTTBatch | None:
        try:
            audios, audio_lens, cuts = collate_audio(cuts, fault_tolerant=True)
        except Exception as e:
            logging.warning(f"Error collating audio from cuts: {e}")
            return None
        if len(cuts) == 0:
            logging.warning("No cuts found in the batch")
            return None

        text = [cut.supervisions[0].text for cut in cuts]
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()
        multiturn_samples = self._get_multiturn_samples(cuts, audio_durations_secs)
        text = self._resolve_multiturn_transcripts(text, multiturn_samples)
        is_multiturn = torch.tensor([sample is not None for sample in multiturn_samples], dtype=torch.bool)
        num_substantive_turns = torch.tensor(
            [sample.num_substantive_turns if sample is not None else 0 for sample in multiturn_samples],
            dtype=torch.long,
        )
        num_user_backchannel_events = torch.tensor(
            [sample.num_user_backchannel_events if sample is not None else 0 for sample in multiturn_samples],
            dtype=torch.long,
        )

        if self.defer_get_batch:
            return StreamingSTTBatch(
                cuts=cuts,
                audios=audios,
                audio_lens=audio_lens,
                text=text,
                is_multiturn=is_multiturn,
                num_substantive_turns=num_substantive_turns,
                num_user_backchannel_events=num_user_backchannel_events,
            )

        alignments = get_word_alignments_for_batch(cuts)

        return self.get_batch_data(cuts, audios, audio_lens, alignments, text)

    def get_batch_data(
        self,
        cuts: CutSet,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        alignments: List[List[WordAlignment]],
        text: List[str],
    ) -> StreamingSTTBatch:
        audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()
        clip_audio_durations_secs = list(audio_durations_secs)
        multiturn_samples = self._get_multiturn_samples(cuts, clip_audio_durations_secs)
        text = self._resolve_multiturn_transcripts(text, multiturn_samples)
        if any(sample is not None for sample in multiturn_samples) and not self.cfg.add_utterance_boundary_tokens:
            raise ValueError("lean multi-turn rows require data.dataset.add_utterance_boundary_tokens=True")
        is_multiturn = torch.tensor([sample is not None for sample in multiturn_samples], dtype=torch.bool)
        num_substantive_turns = torch.tensor(
            [sample.num_substantive_turns if sample is not None else 0 for sample in multiturn_samples],
            dtype=torch.long,
        )
        num_user_backchannel_events = torch.tensor(
            [sample.num_user_backchannel_events if sample is not None else 0 for sample in multiturn_samples],
            dtype=torch.long,
        )

        # K-step alignment (dynamic chunking only): pad each waveform up to a
        # multiple of K frames so the encoder produces exactly that many
        # embeddings, matching the K-snapped segment lengths the dataset will
        # construct below. K=1 → no-op.
        K = max(int(getattr(self.cfg, "chunk_step", 1)), 1)
        if K > 1 and self.cfg.chunk_size == 0:
            new_lens = []
            for dur in audio_durations_secs:
                num_frames = math.ceil(dur / self.cfg.frame_length_in_secs)
                num_frames_padded = math.ceil(num_frames / K) * K
                samples_padded = math.ceil(num_frames_padded * self.cfg.frame_length_in_secs * self.cfg.sample_rate)
                new_lens.append(samples_padded)
            max_samples = max(new_lens) if new_lens else int(audio_lens.max().item())
            if audios.shape[1] < max_samples:
                audios = F.pad(audios, (0, max_samples - audios.shape[1]))
            audio_lens = torch.tensor(new_lens, dtype=audio_lens.dtype, device=audio_lens.device)
            audio_durations_secs = (audio_lens.float() / self.cfg.sample_rate).tolist()

        if self.cfg.use_per_sample_utterance_boundary_tokens:
            system_prompts = []
            add_boundary_tokens = []
            timestamp_sources = []
            for cut, multiturn_sample in zip(cuts, multiturn_samples):
                custom = cut.custom or {}
                cut_id = getattr(cut, "id", "<unknown>")
                if self.cfg.prompt_field not in custom:
                    raise ValueError(
                        f"Cut {cut_id!r} is missing required custom field {self.cfg.prompt_field!r} "
                        "while use_per_sample_utterance_boundary_tokens=True"
                    )
                prompt = custom[self.cfg.prompt_field]
                if not isinstance(prompt, str) or not prompt:
                    raise TypeError(
                        f"Cut {cut_id!r} custom field {self.cfg.prompt_field!r} must be a non-empty string; "
                        f"got {prompt!r}"
                    )
                if multiturn_sample is not None:
                    add_boundaries = custom.get("add_utterance_boundary_tokens", True)
                    if not isinstance(add_boundaries, bool):
                        raise TypeError(
                            f"Cut {cut_id!r} custom field 'add_utterance_boundary_tokens' must be a boolean; "
                            f"got {add_boundaries!r}"
                        )
                    if not add_boundaries:
                        raise ValueError(
                            f"Cut {cut_id!r} is lean multi-turn and cannot disable utterance boundary tokens"
                        )
                else:
                    if "add_utterance_boundary_tokens" not in custom:
                        raise ValueError(
                            f"Cut {cut_id!r} is missing required custom field 'add_utterance_boundary_tokens' "
                            "while use_per_sample_utterance_boundary_tokens=True"
                        )
                    add_boundaries = custom["add_utterance_boundary_tokens"]
                    if not isinstance(add_boundaries, bool):
                        raise TypeError(
                            f"Cut {cut_id!r} custom field 'add_utterance_boundary_tokens' must be a boolean; "
                            f"got {add_boundaries!r}"
                        )
                system_prompts.append(prompt)
                add_boundary_tokens.append(add_boundaries)
                timestamp_source = None
                if add_boundaries and self.cfg.use_per_sample_utterance_boundary_timestamps:
                    if multiturn_sample is not None:
                        timestamp_source = multiturn_sample.schema_version
                    elif "utterance_boundary_timestamp_source" not in custom:
                        raise ValueError(
                            f"Cut {cut_id!r} is missing required custom field "
                            "'utterance_boundary_timestamp_source' while "
                            "use_per_sample_utterance_boundary_timestamps=True"
                        )
                    else:
                        timestamp_source = custom["utterance_boundary_timestamp_source"]
                        if (
                            not isinstance(timestamp_source, str)
                            or timestamp_source not in UTTERANCE_BOUNDARY_TIMESTAMP_SOURCES
                        ):
                            raise ValueError(
                                f"Cut {cut_id!r} custom field 'utterance_boundary_timestamp_source' must be one of "
                                f"{sorted(UTTERANCE_BOUNDARY_TIMESTAMP_SOURCES)}; got {timestamp_source!r}"
                            )
                timestamp_sources.append(timestamp_source)

            resolved_alignments = []
            for (
                cut,
                sample_alignments,
                transcript,
                duration_secs,
                add_boundaries,
                timestamp_source,
                multiturn_sample,
            ) in zip(
                cuts,
                alignments,
                text,
                audio_durations_secs,
                add_boundary_tokens,
                timestamp_sources,
                multiturn_samples,
            ):
                if not add_boundaries:
                    resolved_alignments.append(sample_alignments)
                    continue

                cut_id = str(getattr(cut, "id", "<unknown>"))
                if multiturn_sample is not None:
                    resolved_alignments.append(
                        build_lean_multiturn_alignments(
                            multiturn_sample,
                            sample_alignments,
                            audio_duration_secs=duration_secs,
                            start_token=self.cfg.utterance_start_token,
                            end_token=self.cfg.utterance_end_token,
                            start_delay_frames=self.cfg.utterance_start_boundary_delay_frames,
                            end_delay_frames=self.cfg.utterance_end_boundary_delay_frames,
                            user_backchannel_start_token=(
                                self.cfg.user_backchannel_start_token
                                if self.cfg.user_backchannel_mode
                                in {USER_BACKCHANNEL_MODE_SOB_EOU, USER_BACKCHANNEL_MODE_SOB_EOB}
                                else None
                            ),
                            user_backchannel_end_token=(
                                self.cfg.utterance_end_token
                                if self.cfg.user_backchannel_mode == USER_BACKCHANNEL_MODE_SOB_EOU
                                else self.cfg.user_backchannel_end_token
                                if self.cfg.user_backchannel_mode == USER_BACKCHANNEL_MODE_SOB_EOB
                                else None
                            ),
                            user_backchannel_start_delay_frames=self.cfg.user_backchannel_start_delay_frames,
                            user_backchannel_end_delay_frames=self.cfg.user_backchannel_end_delay_frames,
                            cut_id=cut_id,
                        )
                    )
                    continue
                if transcript and not sample_alignments:
                    raise ValueError(
                        f"Cut {cut_id!r} has a non-empty transcript but no usable word alignments for "
                        "utterance boundary training"
                    )
                custom = cut.custom or {}
                use_gt = timestamp_source == UTTERANCE_BOUNDARY_TIMESTAMP_SOURCE_GT_PREFERRED
                has_gt_start = "utterance_start_time" in custom
                has_gt_end = "utterance_end_time" in custom
                if use_gt and has_gt_start != has_gt_end:
                    raise ValueError(
                        f"Cut {cut_id!r} must provide both 'utterance_start_time' and 'utterance_end_time', "
                        "or neither, when utterance_boundary_timestamp_source='gt_preferred'"
                    )
                if use_gt and has_gt_start:
                    sample_alignments = add_gt_utterance_boundary_alignments(
                        sample_alignments,
                        utterance_start_time=custom["utterance_start_time"],
                        utterance_end_time=custom["utterance_end_time"],
                        audio_duration_secs=duration_secs,
                        start_token=self.cfg.utterance_start_token,
                        end_token=self.cfg.utterance_end_token,
                        start_delay_frames=self.cfg.utterance_start_boundary_delay_frames,
                        end_delay_frames=self.cfg.utterance_end_boundary_delay_frames,
                        cut_id=cut_id,
                    )
                else:
                    if use_gt and cut_id not in self._warned_missing_gt_boundary_cut_ids:
                        logging.warning(
                            "Cut %r requested gt_preferred utterance boundary timestamps but both GT fields "
                            "are absent; falling back to alignment-derived SOU/EOU timestamps",
                            cut_id,
                        )
                        self._warned_missing_gt_boundary_cut_ids.add(cut_id)
                    sample_alignments = add_utterance_boundary_alignments(
                        sample_alignments,
                        audio_duration_secs=duration_secs,
                        start_token=self.cfg.utterance_start_token,
                        end_token=self.cfg.utterance_end_token,
                        start_delay_frames=self.cfg.utterance_start_boundary_delay_frames,
                        end_delay_frames=self.cfg.utterance_end_boundary_delay_frames,
                    )
                resolved_alignments.append(sample_alignments)
            alignments = resolved_alignments
        else:
            # Preserve legacy global behavior while dispatching actual multi-turn rows locally.
            resolved_alignments = []
            for cut, sample_alignments, duration_secs, multiturn_sample in zip(
                cuts, alignments, audio_durations_secs, multiturn_samples
            ):
                if multiturn_sample is not None:
                    sample_alignments = build_lean_multiturn_alignments(
                        multiturn_sample,
                        sample_alignments,
                        audio_duration_secs=duration_secs,
                        start_token=self.cfg.utterance_start_token,
                        end_token=self.cfg.utterance_end_token,
                        start_delay_frames=self.cfg.utterance_start_boundary_delay_frames,
                        end_delay_frames=self.cfg.utterance_end_boundary_delay_frames,
                        user_backchannel_start_token=(
                            self.cfg.user_backchannel_start_token
                            if self.cfg.user_backchannel_mode
                            in {USER_BACKCHANNEL_MODE_SOB_EOU, USER_BACKCHANNEL_MODE_SOB_EOB}
                            else None
                        ),
                        user_backchannel_end_token=(
                            self.cfg.utterance_end_token
                            if self.cfg.user_backchannel_mode == USER_BACKCHANNEL_MODE_SOB_EOU
                            else self.cfg.user_backchannel_end_token
                            if self.cfg.user_backchannel_mode == USER_BACKCHANNEL_MODE_SOB_EOB
                            else None
                        ),
                        user_backchannel_start_delay_frames=self.cfg.user_backchannel_start_delay_frames,
                        user_backchannel_end_delay_frames=self.cfg.user_backchannel_end_delay_frames,
                        cut_id=str(getattr(cut, "id", "<unknown>")),
                    )
                elif self.cfg.add_utterance_boundary_tokens:
                    sample_alignments = add_utterance_boundary_alignments(
                        sample_alignments,
                        audio_duration_secs=duration_secs,
                        start_token=self.cfg.utterance_start_token,
                        end_token=self.cfg.utterance_end_token,
                        start_delay_frames=self.cfg.utterance_start_boundary_delay_frames,
                        end_delay_frames=self.cfg.utterance_end_boundary_delay_frames,
                    )
                resolved_alignments.append(sample_alignments)
            alignments = resolved_alignments

            system_prompts = [(cut.custom or {}).get(self.cfg.prompt_field, self.cfg.system_prompt) for cut in cuts]

        if self.cfg.enable_agent_backchannels:
            merged_alignments = []
            for cut, sample_alignments, clip_duration_secs in zip(cuts, alignments, clip_audio_durations_secs):
                cut_id = str(getattr(cut, "id", "<unknown>"))
                backchannels = build_agent_backchannel_alignments(
                    cut.custom or {},
                    audio_duration_secs=clip_duration_secs,
                    start_token=self.cfg.agent_backchannel_start_token,
                    end_token=self.cfg.agent_backchannel_end_token,
                    delay_frames=self.cfg.agent_backchannel_delay_frames,
                    cut_id=cut_id,
                )
                merged_alignments.append(
                    merge_agent_backchannel_alignments(
                        sample_alignments,
                        backchannels,
                        frame_length_in_secs=self.cfg.frame_length_in_secs,
                        default_delay_frames=self.cfg.num_delay_frames,
                        utterance_start_token=self.cfg.utterance_start_token,
                        utterance_end_token=self.cfg.utterance_end_token,
                    )
                )
            alignments = merged_alignments

        batch_messages = get_llm_messages_for_batch(
            system_role=self.cfg.system_role,
            system_prompt=system_prompts,
            audio_tag=self.cfg.audio_tag,
            blank_token=self.cfg.blank_token,
            chunk_size=self.cfg.chunk_size,
            num_delay_frames=self.cfg.num_delay_frames,
            audio_durations_secs=audio_durations_secs,
            frame_length_in_secs=self.cfg.frame_length_in_secs,
            alignments=alignments,
            transcripts=text,
            words_per_group=self.cfg.words_per_group,
            chunk_step=K,
            agent_backchannel_start_token=(
                self.cfg.agent_backchannel_start_token if self.cfg.enable_agent_backchannels else None
            ),
            agent_backchannel_end_token=(
                self.cfg.agent_backchannel_end_token if self.cfg.enable_agent_backchannels else None
            ),
        )

        all_input_ids = []
        all_target_ids = []
        all_boundary_turn_ordinals = []
        all_user_backchannel_event_ids = []
        all_user_backchannel_reference_frames = []

        for sample_idx, messages in enumerate(batch_messages):
            # Tokenize and compute assistant content mask.
            if self.cfg.compact_template:
                input_ids, assistant_mask = _tokenize_compact_with_assistant_mask(
                    messages,
                    self.tokenizer,
                    self._write_id,
                    self._compact_eos_id,
                    blank_token=self.cfg.blank_token,
                    suppress_blank=self.cfg.compact_text_end_only_no_blank,
                )
            else:
                input_ids, assistant_mask = _tokenize_with_assistant_mask(messages, self.tokenizer)

            # Replace each audio chunk token sequence with chunk_size AUDIO_TOKEN_IDX markers.
            # We match the full chunk (audio_tag * chunk_size) as a unit because BPE
            # may merge tokens across adjacent audio tags.
            if self.audio_chunk_ids is not None:
                # Fixed chunking: single pre-computed pattern
                input_ids, assistant_mask = _replace_audio_chunks(
                    input_ids, self.audio_chunk_ids, self.cfg.chunk_size, mask=assistant_mask
                )
            else:
                # Offline (chunk_size=-1) or dynamic (chunk_size=0): variable audio tag
                # counts per user turn.  Replace each user turn's audio tags separately.
                hf_tok = self.tokenizer.tokenizer
                for msg in messages:
                    if msg["role"] != "user":
                        continue
                    n_tags = msg["content"].count(self.cfg.audio_tag)
                    if n_tags == 0:
                        continue
                    chunk_ids = hf_tok.encode(self.cfg.audio_tag * n_tags, add_special_tokens=False)
                    input_ids, assistant_mask = _replace_audio_chunks(
                        input_ids, chunk_ids, n_tags, mask=assistant_mask
                    )

            # Build targets: next-token prediction with loss only on assistant content.
            # target[i] corresponds to input[i] and holds the token at position i+1.
            # Loss is applied only where assistant_mask[i+1] is True.
            target_ids = input_ids[1:] + [IGNORE_INDEX]
            target_mask = assistant_mask[1:] + [0]
            target_ids = [tid if m else IGNORE_INDEX for tid, m in zip(target_ids, target_mask)]

            # Optional sequence curation debug:
            #   STREAMING_STT_DEBUG_SEQUENCE=1 dumps the first sample's messages/input/target table.
            #   STREAMING_STT_DEBUG_BREAKPOINT=1 also drops into pdb after the dump.
            debug_sequence = os.environ.get("STREAMING_STT_DEBUG_SEQUENCE", "").lower() in {"1", "true", "yes", "y"}
            debug_breakpoint = os.environ.get("STREAMING_STT_DEBUG_BREAKPOINT", "").lower() in {
                "1",
                "true",
                "yes",
                "y",
            }
            if sample_idx == 0 and (debug_sequence or debug_breakpoint):
                _debug_dump_sequence(
                    messages,
                    input_ids,
                    target_ids,
                    assistant_mask,
                    self.tokenizer,
                    self.blank_id,
                    sou_id=self._sou_id,
                    eou_id=self._eou_id,
                    transcript=text[sample_idx],
                    do_breakpoint=debug_breakpoint,
                )

            # Dynamic chunking: train the model to predict at audio positions.
            # Non-final audio frames → target = blank_id ("need more audio")
            # Final audio frame (before user footer) → target = user_footer first token ("ready")
            if self.cfg.chunk_size == 0:
                user_footer_id = self._user_footer_first_id
                for i in range(len(input_ids)):
                    if input_ids[i] != AUDIO_TOKEN_IDX:
                        continue
                    next_is_audio = i + 1 < len(input_ids) and input_ids[i + 1] == AUDIO_TOKEN_IDX
                    target_ids[i] = self.blank_id if next_is_audio else user_footer_id

            boundary_turn_ordinals = [0] * len(target_ids)
            user_backchannel_event_ids = [0] * len(target_ids)
            user_backchannel_reference_frames = [-1] * len(target_ids)
            multiturn_sample = multiturn_samples[sample_idx]
            if multiturn_sample is not None:
                (
                    boundary_turn_ordinals,
                    user_backchannel_event_ids,
                    user_backchannel_reference_frames,
                ) = build_multiturn_marker_metadata(
                    target_ids,
                    multiturn_sample,
                    sou_id=self._sou_id,
                    eou_id=self._eou_id,
                    sob_id=self._sob_id,
                    user_backchannel_end_id=self._user_backchannel_end_id,
                    frame_length_in_secs=self.cfg.frame_length_in_secs,
                    sample_idx=sample_idx,
                )

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_target_ids.append(torch.tensor(target_ids, dtype=torch.long))
            all_boundary_turn_ordinals.append(torch.tensor(boundary_turn_ordinals, dtype=torch.long))
            all_user_backchannel_event_ids.append(torch.tensor(user_backchannel_event_ids, dtype=torch.long))
            all_user_backchannel_reference_frames.append(
                torch.tensor(user_backchannel_reference_frames, dtype=torch.long)
            )

        if self.cfg.chunk_size >= 0:  # fixed chunking or dynamic chunking: right-pad
            input_tokens = right_collate_vectors(all_input_ids, padding_value=self.tokenizer.pad_id)
            target_tokens = right_collate_vectors(all_target_ids, padding_value=IGNORE_INDEX)
            boundary_turn_ordinals = right_collate_vectors(all_boundary_turn_ordinals, padding_value=0)
            user_backchannel_event_ids = right_collate_vectors(all_user_backchannel_event_ids, padding_value=0)
            user_backchannel_reference_frames = right_collate_vectors(
                all_user_backchannel_reference_frames, padding_value=-1
            )
            input_token_lens = torch.tensor([len(ids) for ids in all_input_ids], dtype=torch.long)
            target_token_lens = torch.tensor([len(ids) for ids in all_target_ids], dtype=torch.long)
        else:  # offline mode: left-pad
            input_tokens = left_collate_vectors(all_input_ids, padding_value=self.tokenizer.pad_id)
            target_tokens = left_collate_vectors(all_target_ids, padding_value=IGNORE_INDEX)
            boundary_turn_ordinals = left_collate_vectors(all_boundary_turn_ordinals, padding_value=0)
            user_backchannel_event_ids = left_collate_vectors(all_user_backchannel_event_ids, padding_value=0)
            user_backchannel_reference_frames = left_collate_vectors(
                all_user_backchannel_reference_frames, padding_value=-1
            )
            # length is the same size as input_tokens.shape[1] since they're left-padded
            input_token_lens = torch.tensor(
                [input_tokens.shape[1] for _ in range(len(all_input_ids))], dtype=torch.long
            )
            target_token_lens = torch.tensor(
                [target_tokens.shape[1] for _ in range(len(all_target_ids))], dtype=torch.long
            )

        return StreamingSTTBatch(
            audios=audios,
            audio_lens=audio_lens,
            input_tokens=input_tokens,
            input_token_lens=input_token_lens,
            target_tokens=target_tokens,
            target_token_lens=target_token_lens,
            text=text,
            is_multiturn=is_multiturn,
            num_substantive_turns=num_substantive_turns,
            boundary_turn_ordinals=boundary_turn_ordinals,
            user_backchannel_event_ids=user_backchannel_event_ids,
            user_backchannel_reference_frames=user_backchannel_reference_frames,
            num_user_backchannel_events=num_user_backchannel_events,
        )
