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

import pytest
import torch

from nemo.collections.speechlm2.parts.metrics.boundary import (
    boundary_collar_precision_recall_f1,
    compute_ordinal_boundary_timing_metrics,
    count_boundary_collar_hits,
)


@pytest.mark.parametrize(
    ("hits", "predictions", "targets", "precision", "recall", "f1"),
    [
        (8, 10, 8, 0.8, 1.0, 8 / 9),
        (4, 4, 8, 1.0, 0.5, 2 / 3),
        (0, 0, 8, 0.0, 0.0, 0.0),
        (0, 0, 0, 0.0, 0.0, 0.0),
    ],
)
def test_boundary_collar_precision_recall_f1(hits, predictions, targets, precision, recall, f1):
    actual = boundary_collar_precision_recall_f1(torch.tensor(hits), torch.tensor(predictions), torch.tensor(targets))
    assert actual[0].item() == pytest.approx(precision)
    assert actual[1].item() == pytest.approx(recall)
    assert actual[2].item() == pytest.approx(f1)


def test_extra_pause_boundary_predictions_reduce_f1():
    precision, recall, f1 = boundary_collar_precision_recall_f1(
        collar_hits=torch.tensor(5),
        prediction_count=torch.tensor(10),
        target_count=torch.tensor(5),
    )
    assert precision.item() == pytest.approx(0.5)
    assert recall.item() == pytest.approx(1.0)
    assert f1.item() == pytest.approx(2 / 3)


def test_boundary_collar_matching_is_one_to_one():
    target_mask = torch.tensor([[False, False, True, True, False]])
    pred_mask = torch.tensor([[False, False, True, False, False]])
    frame_idx = torch.arange(1, 6).unsqueeze(0)

    hits = count_boundary_collar_hits(
        target_mask=target_mask,
        pred_mask=pred_mask,
        frame_idx=frame_idx,
        before_frames=1,
        after_frames=1,
    )

    assert hits.item() == 1


def test_ordinal_timing_metrics_use_disjoint_reference_regions_and_count_spurious():
    sou_id, eou_id, regular_id, ignore_index, audio_token_idx = 10, 11, 1, -100, -200
    target_ids = torch.full((3, 16), regular_id)
    pred_ids = torch.full((3, 16), regular_id)
    boundary_ordinals = torch.zeros_like(target_ids)

    # Two-turn row: the extra SOU in turn 1 must not shift turn 2's detection.
    for position, token_id, ordinal in ((2, sou_id, 1), (6, eou_id, 1), (9, sou_id, 2), (13, eou_id, 2)):
        target_ids[0, position] = token_id
        boundary_ordinals[0, position] = ordinal
    pred_ids[0, 1] = sou_id  # first turn-1 SOU: early
    pred_ids[0, 3] = sou_id  # additional turn-1 SOU: spurious
    pred_ids[0, 10] = sou_id  # turn-2 SOU: detection
    pred_ids[0, 14] = sou_id  # after final EOU: outside-region spurious
    pred_ids[0, 4] = eou_id  # turn-1 EOU: detection
    pred_ids[0, 5] = eou_id  # additional turn-1 EOU: spurious
    pred_ids[0, 15] = eou_id  # turn-2 EOU: detection

    # One-turn row: SOU is detected at +4 frames; EOU is missing.
    target_ids[1, 3] = sou_id
    target_ids[1, 10] = eou_id
    boundary_ordinals[1, 3] = 1
    boundary_ordinals[1, 10] = 1
    pred_ids[1, 7] = sou_id

    # A legacy row with boundaries is deliberately excluded.
    target_ids[2, 0] = pred_ids[2, 0] = sou_id
    target_ids[2, 15] = pred_ids[2, 15] = eou_id

    metrics = compute_ordinal_boundary_timing_metrics(
        pred_ids=pred_ids,
        target_ids=target_ids,
        input_tokens=torch.full_like(target_ids, audio_token_idx),
        boundary_turn_ordinals=boundary_ordinals,
        is_multiturn=torch.tensor([True, True, False]),
        num_substantive_turns=torch.tensor([2, 1, 0]),
        sou_id=sou_id,
        eou_id=eou_id,
        ignore_index=ignore_index,
        audio_token_idx=audio_token_idx,
        max_turns=4,
    )

    assert metrics["turn_1_num_samples"].item() == 2
    assert metrics["turn_1_sou_early_count"].item() == 1
    assert metrics["turn_1_sou_detection_count"].item() == 1
    assert metrics["turn_1_sou_late_count"].item() == 0
    assert metrics["turn_1_sou_missing_count"].item() == 0
    assert metrics["turn_1_sou_spurious_count"].item() == 1
    assert metrics["turn_1_eou_detection_count"].item() == 1
    assert metrics["turn_1_eou_missing_count"].item() == 1
    assert metrics["turn_1_eou_spurious_count"].item() == 1
    assert metrics["turn_2_num_samples"].item() == 1
    assert metrics["turn_2_sou_detection_count"].item() == 1
    assert metrics["turn_2_eou_detection_count"].item() == 1
    assert metrics["turn_3_num_samples"].item() == 0
    assert metrics["sou_spurious_outside_count"].item() == 1
    assert metrics["eou_spurious_outside_count"].item() == 0


def test_ordinal_timing_metrics_classify_late_sou_and_early_eou():
    sou_id, eou_id, regular_id, ignore_index, audio_token_idx = 10, 11, 1, -100, -200
    target_ids = torch.full((1, 12), regular_id)
    pred_ids = torch.full_like(target_ids, regular_id)
    boundary_ordinals = torch.zeros_like(target_ids)
    target_ids[0, 2] = sou_id
    target_ids[0, 9] = eou_id
    boundary_ordinals[0, 2] = 1
    boundary_ordinals[0, 9] = 1
    pred_ids[0, 7] = sou_id
    pred_ids[0, 3] = eou_id

    metrics = compute_ordinal_boundary_timing_metrics(
        pred_ids=pred_ids,
        target_ids=target_ids,
        input_tokens=torch.full_like(target_ids, audio_token_idx),
        boundary_turn_ordinals=boundary_ordinals,
        is_multiturn=torch.tensor([True]),
        num_substantive_turns=torch.tensor([1]),
        sou_id=sou_id,
        eou_id=eou_id,
        ignore_index=ignore_index,
        audio_token_idx=audio_token_idx,
        max_turns=4,
    )

    assert metrics["turn_1_sou_late_count"].item() == 1
    assert metrics["turn_1_eou_early_count"].item() == 1
