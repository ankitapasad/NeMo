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


def compute_user_backchannel_metrics(
    pred_ids: Tensor,
    target_ids: Tensor,
    input_tokens: Tensor,
    user_backchannel_event_ids: Tensor,
    user_backchannel_reference_frames: Tensor,
    num_user_backchannel_events: Tensor,
    boundary_turn_ordinals: Tensor,
    *,
    sob_id: int,
    user_backchannel_end_id: int,
    sou_id: int,
    eou_id: int,
    ignore_index: int,
    audio_token_idx: int,
    collar_frames: int,
) -> dict[str, Tensor]:
    """Match user-backchannel markers against annotated boundaries.

    Start markers use global deterministic nearest one-to-one matching inside a
    symmetric collar. A matched SOU is a confusion. The configured closing
    marker first uses the same in-collar matcher, then accepts the nearest later
    marker until the next backchannel start, substantive SOU, or clip end.
    Early markers outside the collar remain unmatched. In ``sob_eob`` mode an
    EOU in the same window is reported as a closing-marker confusion. The
    returned exclusion mask removes every prediction owned by a backchannel
    window from substantive metrics.
    """
    expected_shape = target_ids.shape
    for name, value in (
        ("pred_ids", pred_ids),
        ("input_tokens", input_tokens),
        ("user_backchannel_event_ids", user_backchannel_event_ids),
        ("user_backchannel_reference_frames", user_backchannel_reference_frames),
        ("boundary_turn_ordinals", boundary_turn_ordinals),
    ):
        if value.shape != expected_shape:
            raise ValueError(f"{name} must have the same shape as target_ids")
    if num_user_backchannel_events.shape != (target_ids.shape[0],):
        raise ValueError("num_user_backchannel_events must have shape (batch_size,)")
    if collar_frames < 0:
        raise ValueError("collar_frames must be non-negative")
    if len({sob_id, sou_id, eou_id}) != 3:
        raise ValueError("SOB, SOU, and EOU token IDs must be distinct")
    if user_backchannel_end_id in {sob_id, sou_id}:
        raise ValueError("The user backchannel end token must differ from SOB and SOU")

    device = target_ids.device
    valid_mask = target_ids != ignore_index
    frame_idx = (input_tokens == audio_token_idx).long().cumsum(dim=1)
    prediction_exclusion_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    counts = {
        "num_samples": 0,
        "event_count": 0,
        "sob_correct_count": 0,
        "sou_confusion_count": 0,
        "no_start_marker_count": 0,
        "sob_spurious_count": 0,
        "end_in_collar_count": 0,
        "end_late_count": 0,
        "end_missing_count": 0,
        "end_spurious_count": 0,
        "eou_end_confusion_count": 0,
        "paired_event_count": 0,
    }
    sob_latencies: list[float] = []
    end_latencies: list[float] = []

    for sample_idx in range(target_ids.shape[0]):
        num_events = int(num_user_backchannel_events[sample_idx].item())
        if num_events == 0:
            continue
        counts["num_samples"] += 1
        counts["event_count"] += num_events
        sample_valid = valid_mask[sample_idx]
        sample_frames = frame_idx[sample_idx]

        events: list[dict[str, int | bool]] = []
        for event_id in range(1, num_events + 1):
            event_mask = sample_valid & (user_backchannel_event_ids[sample_idx] == event_id)
            sob_positions = torch.nonzero(
                event_mask & (target_ids[sample_idx] == sob_id), as_tuple=False
            ).flatten()
            end_positions = torch.nonzero(
                event_mask & (target_ids[sample_idx] == user_backchannel_end_id), as_tuple=False
            ).flatten()
            if sob_positions.numel() != 1 or end_positions.numel() != 1:
                raise ValueError(
                    "User-backchannel event "
                    f"{event_id} must have exactly one SOB and one closing-marker target; "
                    f"got {sob_positions.numel()} and {end_positions.numel()}"
                )
            start_reference = int(
                user_backchannel_reference_frames[sample_idx, sob_positions[0]].item()
            )
            end_reference = int(
                user_backchannel_reference_frames[sample_idx, end_positions[0]].item()
            )
            if start_reference < 0 or end_reference < start_reference:
                raise ValueError(
                    f"User-backchannel event {event_id} has invalid annotated reference frames "
                    f"{start_reference}/{end_reference}"
                )
            events.append(
                {
                    "event_id": event_id,
                    "start": start_reference,
                    "end": end_reference,
                }
            )
        if [int(event["start"]) for event in events] != sorted(
            int(event["start"]) for event in events
        ):
            raise ValueError("User-backchannel event IDs must follow annotated chronological order")

        clip_end = int(sample_frames.max().item())
        substantive_sou_frames = sample_frames[
            sample_valid
            & (target_ids[sample_idx] == sou_id)
            & (boundary_turn_ordinals[sample_idx] > 0)
        ].tolist()
        for event_idx, event in enumerate(events):
            start_reference = int(event["start"])
            stops = [frame for frame in substantive_sou_frames if frame > start_reference]
            stops.extend(
                int(next_event["start"])
                for next_event in events[event_idx + 1 :]
                if int(next_event["start"]) > start_reference
            )
            if stops:
                event["stop"] = min(stops)
                event["stop_is_clip"] = False
            else:
                event["stop"] = clip_end
                event["stop_is_clip"] = True

        start_predictions = []
        for position in torch.nonzero(
            sample_valid
            & ((pred_ids[sample_idx] == sob_id) | (pred_ids[sample_idx] == sou_id)),
            as_tuple=False,
        ).flatten().tolist():
            start_predictions.append(
                {
                    "position": position,
                    "frame": int(sample_frames[position].item()),
                    "token": int(pred_ids[sample_idx, position].item()),
                }
            )

        start_edges = []
        for event_idx, event in enumerate(events):
            reference = int(event["start"])
            for pred_idx, prediction in enumerate(start_predictions):
                distance = abs(int(prediction["frame"]) - reference)
                if distance <= collar_frames:
                    start_edges.append(
                        (
                            distance,
                            event_idx,
                            int(prediction["frame"]),
                            int(prediction["position"]),
                            pred_idx,
                        )
                    )
        matched_start_events: dict[int, int] = {}
        used_start_predictions: set[int] = set()
        for _, event_idx, _, _, pred_idx in sorted(start_edges):
            if event_idx in matched_start_events or pred_idx in used_start_predictions:
                continue
            matched_start_events[event_idx] = pred_idx
            used_start_predictions.add(pred_idx)

        correct_sob_events: set[int] = set()
        matched_sob_predictions: set[int] = set()
        for event_idx, event in enumerate(events):
            pred_idx = matched_start_events.get(event_idx)
            if pred_idx is None:
                counts["no_start_marker_count"] += 1
                continue
            prediction = start_predictions[pred_idx]
            if int(prediction["token"]) == sob_id:
                counts["sob_correct_count"] += 1
                correct_sob_events.add(event_idx)
                matched_sob_predictions.add(pred_idx)
                sob_latencies.append(float(int(prediction["frame"]) - int(event["start"])))
            else:
                counts["sou_confusion_count"] += 1

        for pred_idx, prediction in enumerate(start_predictions):
            position = int(prediction["position"])
            frame = int(prediction["frame"])
            token = int(prediction["token"])
            if token == sob_id:
                prediction_exclusion_mask[sample_idx, position] = True
                if pred_idx not in matched_sob_predictions:
                    counts["sob_spurious_count"] += 1
            elif any(abs(frame - int(event["start"])) <= collar_frames for event in events):
                prediction_exclusion_mask[sample_idx, position] = True

        end_predictions = []
        for position in torch.nonzero(
            sample_valid & (pred_ids[sample_idx] == user_backchannel_end_id), as_tuple=False
        ).flatten().tolist():
            end_predictions.append(
                {
                    "position": position,
                    "frame": int(sample_frames[position].item()),
                }
            )

        def _before_stop(event: dict[str, int | bool], frame: int) -> bool:
            stop = int(event["stop"])
            return frame <= stop if bool(event["stop_is_clip"]) else frame < stop

        end_edges = []
        for event_idx, event in enumerate(events):
            reference = int(event["end"])
            for pred_idx, prediction in enumerate(end_predictions):
                frame = int(prediction["frame"])
                distance = abs(frame - reference)
                if distance <= collar_frames and _before_stop(event, frame):
                    end_edges.append((distance, event_idx, frame, int(prediction["position"]), pred_idx))

        matched_end_events: dict[int, tuple[int, str]] = {}
        used_end_predictions: set[int] = set()
        for _, event_idx, _, _, pred_idx in sorted(end_edges):
            if event_idx in matched_end_events or pred_idx in used_end_predictions:
                continue
            matched_end_events[event_idx] = (pred_idx, "in_collar")
            used_end_predictions.add(pred_idx)

        for event_idx, event in enumerate(events):
            if event_idx in matched_end_events:
                continue
            reference = int(event["end"])
            late_candidates = [
                (int(prediction["frame"]) - reference, int(prediction["position"]), pred_idx)
                for pred_idx, prediction in enumerate(end_predictions)
                if pred_idx not in used_end_predictions
                and int(prediction["frame"]) > reference + collar_frames
                and _before_stop(event, int(prediction["frame"]))
            ]
            if late_candidates:
                _, _, pred_idx = min(late_candidates)
                matched_end_events[event_idx] = (pred_idx, "late")
                used_end_predictions.add(pred_idx)

        for event_idx, event in enumerate(events):
            match = matched_end_events.get(event_idx)
            if match is None:
                counts["end_missing_count"] += 1
                continue
            pred_idx, outcome = match
            if outcome == "in_collar":
                counts["end_in_collar_count"] += 1
            else:
                counts["end_late_count"] += 1
            latency = int(end_predictions[pred_idx]["frame"]) - int(event["end"])
            end_latencies.append(float(latency))
            if event_idx in correct_sob_events:
                counts["paired_event_count"] += 1

        for pred_idx, prediction in enumerate(end_predictions):
            frame = int(prediction["frame"])
            in_backchannel_window = any(
                frame >= int(event["start"]) - collar_frames and _before_stop(event, frame)
                for event in events
            )
            if not in_backchannel_window:
                continue
            prediction_exclusion_mask[sample_idx, int(prediction["position"])] = True
            if pred_idx not in used_end_predictions:
                counts["end_spurious_count"] += 1

        if user_backchannel_end_id != eou_id:
            eou_predictions = []
            for position in torch.nonzero(
                sample_valid & (pred_ids[sample_idx] == eou_id), as_tuple=False
            ).flatten().tolist():
                frame = int(sample_frames[position].item())
                owning_events = [
                    event_idx
                    for event_idx, event in enumerate(events)
                    if frame >= int(event["start"]) - collar_frames and _before_stop(event, frame)
                ]
                if not owning_events:
                    continue
                prediction_exclusion_mask[sample_idx, position] = True
                eou_predictions.append({"position": position, "frame": frame})

            confusion_edges = []
            for event_idx, event in enumerate(events):
                reference = int(event["end"])
                for pred_idx, prediction in enumerate(eou_predictions):
                    frame = int(prediction["frame"])
                    if not _before_stop(event, frame):
                        continue
                    if abs(frame - reference) <= collar_frames:
                        rank = (0, abs(frame - reference))
                    elif frame > reference + collar_frames:
                        rank = (1, frame - reference)
                    else:
                        continue
                    confusion_edges.append(
                        (*rank, event_idx, frame, int(prediction["position"]), pred_idx)
                    )
            used_confusion_events: set[int] = set()
            used_eou_predictions: set[int] = set()
            for _, _, event_idx, _, _, pred_idx in sorted(confusion_edges):
                if event_idx in used_confusion_events or pred_idx in used_eou_predictions:
                    continue
                used_confusion_events.add(event_idx)
                used_eou_predictions.add(pred_idx)
            counts["eou_end_confusion_count"] += len(used_confusion_events)

    metrics = {
        name: torch.as_tensor(value, dtype=torch.long, device=device) for name, value in counts.items()
    }
    metrics["sob_latency_frames"] = torch.as_tensor(sob_latencies, dtype=torch.float32, device=device)
    metrics["end_latency_frames"] = torch.as_tensor(end_latencies, dtype=torch.float32, device=device)
    metrics["prediction_exclusion_mask"] = prediction_exclusion_mask
    return metrics


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
    user_backchannel_event_ids: Tensor | None = None,
    prediction_exclusion_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    if user_backchannel_event_ids is None:
        user_backchannel_event_ids = torch.zeros_like(target_ids)
    if prediction_exclusion_mask is None:
        prediction_exclusion_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    if user_backchannel_event_ids.shape != target_ids.shape:
        raise ValueError("user_backchannel_event_ids must have the same shape as target_ids")
    if prediction_exclusion_mask.shape != target_ids.shape:
        raise ValueError("prediction_exclusion_mask must have the same shape as target_ids")
    valid_mask = target_ids != ignore_index
    substantive_target = user_backchannel_event_ids == 0
    eligible_prediction = valid_mask & ~prediction_exclusion_mask
    sou_targets = valid_mask & substantive_target & (target_ids == sou_id)
    eou_targets = valid_mask & substantive_target & (target_ids == eou_id)
    sou_preds = eligible_prediction & (pred_ids == sou_id)
    eou_preds = eligible_prediction & (pred_ids == eou_id)
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
    prediction_exclusion_mask: Tensor | None = None,
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
    if prediction_exclusion_mask is None:
        prediction_exclusion_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    if prediction_exclusion_mask.shape != target_ids.shape:
        raise ValueError("prediction_exclusion_mask must have the same shape as target_ids")

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
            prediction_frames = frame_idx[sample_idx][
                valid_mask[sample_idx]
                & ~prediction_exclusion_mask[sample_idx]
                & (pred_ids[sample_idx] == token_id)
            ]
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
