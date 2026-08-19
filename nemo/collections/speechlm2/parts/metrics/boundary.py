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

import torch
from torch import Tensor


def count_boundary_collar_hits(
    target_mask: Tensor,
    pred_mask: Tensor,
    frame_idx: Tensor,
    before_frames: int,
    after_frames: int,
) -> Tensor:
    """Count deterministic one-to-one target/prediction matches inside a collar."""
    hits = torch.zeros((), dtype=torch.long, device=target_mask.device)
    for sample_idx in range(target_mask.shape[0]):
        target_frames = frame_idx[sample_idx][target_mask[sample_idx]]
        pred_frames = frame_idx[sample_idx][pred_mask[sample_idx]]
        if target_frames.numel() == 0 or pred_frames.numel() == 0:
            continue
        used_predictions = torch.zeros_like(pred_frames, dtype=torch.bool)
        for target_frame in target_frames:
            in_window = (
                (~used_predictions)
                & (pred_frames >= target_frame - before_frames)
                & (pred_frames <= target_frame + after_frames)
            )
            candidate_indices = torch.nonzero(in_window, as_tuple=False).flatten()
            if candidate_indices.numel() == 0:
                continue
            candidate_distances = (pred_frames[candidate_indices] - target_frame).abs()
            best_index = candidate_indices[candidate_distances.argmin()]
            used_predictions[best_index] = True
            hits = hits + 1
    return hits


def boundary_collar_precision_recall_f1(
    collar_hits: Tensor, prediction_count: Tensor, target_count: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute zero-safe collar precision, recall, and F1 from aggregate counts."""
    hits = collar_hits.float()
    precision = hits / prediction_count.clamp(min=1).float()
    recall = hits / target_count.clamp(min=1).float()
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))
    return precision, recall, f1


def compute_boundary_token_metrics(
    pred_ids: Tensor,
    target_ids: Tensor,
    input_tokens: Tensor,
    sou_id: int,
    eou_id: int,
    ignore_index: int,
    audio_token_idx: int,
    sou_before_frames: int = 4,
    sou_after_frames: int = 4,
    eou_before_frames: int = 2,
    eou_after_frames: int = 4,
) -> dict[str, Tensor]:
    valid_mask = target_ids != ignore_index
    sou_targets = valid_mask & (target_ids == sou_id)
    eou_targets = valid_mask & (target_ids == eou_id)
    sou_preds = valid_mask & (pred_ids == sou_id)
    eou_preds = valid_mask & (pred_ids == eou_id)
    frame_idx = (input_tokens == audio_token_idx).long().cumsum(dim=1)

    sou_collar_hit = count_boundary_collar_hits(
        target_mask=sou_targets,
        pred_mask=sou_preds,
        frame_idx=frame_idx,
        before_frames=sou_before_frames,
        after_frames=sou_after_frames,
    )
    eou_collar_hit = count_boundary_collar_hits(
        target_mask=eou_targets,
        pred_mask=eou_preds,
        frame_idx=frame_idx,
        before_frames=eou_before_frames,
        after_frames=eou_after_frames,
    )

    return {
        "num_samples": torch.as_tensor(target_ids.shape[0], dtype=torch.long, device=target_ids.device),
        "sou_target_count": sou_targets.long().sum(),
        "eou_target_count": eou_targets.long().sum(),
        "sou_pred_count": sou_preds.long().sum(),
        "eou_pred_count": eou_preds.long().sum(),
        "sou_collar_hit": sou_collar_hit,
        "eou_collar_hit": eou_collar_hit,
    }


def compute_ordinal_boundary_timing_metrics(
    pred_ids: Tensor,
    target_ids: Tensor,
    input_tokens: Tensor,
    boundary_turn_ordinals: Tensor,
    is_multiturn: Tensor,
    num_substantive_turns: Tensor,
    sou_id: int,
    eou_id: int,
    ignore_index: int,
    audio_token_idx: int,
    max_turns: int = 4,
    detection_tolerance_frames: int = 4,
) -> dict[str, Tensor]:
    """Classify the first boundary prediction inside each reference turn window.

    SOU turn-k window:
        previous reference EOU (or clip start) -> current reference EOU.
        A first prediction before reference SOU is early, from reference SOU
        through ``detection_tolerance_frames`` later is detection, and anything
        later is late.

    EOU turn-k window:
        current reference SOU -> next reference SOU (or clip end).
        A first prediction before ``reference EOU - tolerance`` is early,
        within +/- tolerance is detection, and anything later is late.

    Windows are half-open except at clip end. Every prediction after the first
    in its window is spurious. Predictions outside all same-token reference
    windows are also spurious. Rows without a kth substantive turn are excluded
    from turn-k metrics.
    """
    if pred_ids.shape != target_ids.shape or input_tokens.shape != target_ids.shape:
        raise ValueError("pred_ids, target_ids, and input_tokens must have identical shapes")
    if boundary_turn_ordinals.shape != target_ids.shape:
        raise ValueError("boundary_turn_ordinals must have the same shape as target_ids")
    if is_multiturn.shape != num_substantive_turns.shape or is_multiturn.numel() != target_ids.shape[0]:
        raise ValueError("multi-turn row metadata must have shape (batch_size,)")
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if detection_tolerance_frames < 0:
        raise ValueError("detection_tolerance_frames must be non-negative")

    valid_mask = target_ids != ignore_index
    frame_idx = (input_tokens == audio_token_idx).long().cumsum(dim=1)
    result: dict[str, Tensor] = {}

    for turn_ordinal in range(1, max_turns + 1):
        eligible = is_multiturn.bool() & (num_substantive_turns >= turn_ordinal)
        prefix = f"turn_{turn_ordinal}"
        result[f"{prefix}_num_samples"] = eligible.long().sum()
        for token_name in ("sou", "eou"):
            for outcome in ("early", "detection", "late", "missing", "spurious"):
                result[f"{prefix}_{token_name}_{outcome}_count"] = torch.zeros(
                    (), dtype=torch.long, device=target_ids.device
                )

    for token_name in ("sou", "eou"):
        result[f"{token_name}_spurious_outside_count"] = torch.zeros((), dtype=torch.long, device=target_ids.device)

    for sample_idx in range(target_ids.shape[0]):
        if not bool(is_multiturn[sample_idx]):
            continue
        num_turns = int(num_substantive_turns[sample_idx].item())
        if num_turns <= 0:
            raise ValueError("Multi-turn sample must contain at least one substantive turn")

        reference_frames: dict[str, list[Tensor]] = {"sou": [], "eou": []}
        for token_name, token_id in (("sou", sou_id), ("eou", eou_id)):
            for turn_ordinal in range(1, num_turns + 1):
                target_mask = (
                    valid_mask[sample_idx]
                    & (target_ids[sample_idx] == token_id)
                    & (boundary_turn_ordinals[sample_idx] == turn_ordinal)
                )
                frames = frame_idx[sample_idx][target_mask]
                if frames.numel() != 1:
                    raise ValueError(
                        f"Eligible sample has {frames.numel()} {token_name.upper()} targets "
                        f"for substantive turn {turn_ordinal}; expected exactly one"
                    )
                reference_frames[token_name].append(frames[0])

        clip_end_frame = frame_idx[sample_idx].max()
        for token_name, token_id in (("sou", sou_id), ("eou", eou_id)):
            prediction_frames = frame_idx[sample_idx][valid_mask[sample_idx] & (pred_ids[sample_idx] == token_id)]
            assigned_predictions = torch.zeros_like(prediction_frames, dtype=torch.bool)

            for turn_ordinal in range(1, num_turns + 1):
                if token_name == "sou":
                    window_start = (
                        torch.zeros_like(reference_frames["eou"][0])
                        if turn_ordinal == 1
                        else reference_frames["eou"][turn_ordinal - 2]
                    )
                    window_end = reference_frames["eou"][turn_ordinal - 1]
                    in_window = (prediction_frames >= window_start) & (prediction_frames < window_end)
                else:
                    window_start = reference_frames["sou"][turn_ordinal - 1]
                    if turn_ordinal < num_turns:
                        window_end = reference_frames["sou"][turn_ordinal]
                        in_window = (prediction_frames >= window_start) & (prediction_frames < window_end)
                    else:
                        in_window = (prediction_frames >= window_start) & (prediction_frames <= clip_end_frame)

                assigned_predictions |= in_window
                if turn_ordinal > max_turns:
                    continue

                prefix = f"turn_{turn_ordinal}_{token_name}"
                candidates = prediction_frames[in_window]
                if candidates.numel() == 0:
                    result[f"{prefix}_missing_count"] += 1
                    continue

                first_prediction = candidates[0]
                reference = reference_frames[token_name][turn_ordinal - 1]
                if token_name == "sou":
                    if first_prediction < reference:
                        outcome = "early"
                    elif first_prediction <= reference + detection_tolerance_frames:
                        outcome = "detection"
                    else:
                        outcome = "late"
                else:
                    if first_prediction < reference - detection_tolerance_frames:
                        outcome = "early"
                    elif first_prediction <= reference + detection_tolerance_frames:
                        outcome = "detection"
                    else:
                        outcome = "late"
                result[f"{prefix}_{outcome}_count"] += 1
                result[f"{prefix}_spurious_count"] += candidates.numel() - 1

            result[f"{token_name}_spurious_outside_count"] += (~assigned_predictions).long().sum()

    return result
