# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from nemo.collections.speechlm2.data.streaming_stt_dataset import AUDIO_TOKEN_IDX, IGNORE_INDEX
from nemo.collections.speechlm2.models import streaming_stt_model
from nemo.collections.speechlm2.parts.metrics.boundary import (
    compute_boundary_token_metrics,
    compute_ordinal_boundary_timing_metrics,
    compute_user_backchannel_metrics,
)


SOB = 10
SOU = 11
EOU = 12
TEXT = 13
EOB = 14


def _metric_case(end_id=EOU):
    length = 80
    input_tokens = torch.full((1, length), AUDIO_TOKEN_IDX, dtype=torch.long)
    target_ids = torch.full((1, length), TEXT, dtype=torch.long)
    pred_ids = torch.full((1, length), TEXT, dtype=torch.long)
    event_ids = torch.zeros((1, length), dtype=torch.long)
    reference_frames = torch.full((1, length), -1, dtype=torch.long)
    ordinals = torch.zeros((1, length), dtype=torch.long)

    for event_id, (sob_pos, end_pos, start, end) in enumerate(
        ((0, 1, 10, 15), (2, 3, 30, 35), (4, 5, 50, 55)), start=1
    ):
        target_ids[0, sob_pos] = SOB
        target_ids[0, end_pos] = end_id
        event_ids[0, sob_pos] = event_id
        event_ids[0, end_pos] = event_id
        reference_frames[0, sob_pos] = start
        reference_frames[0, end_pos] = end

    target_ids[0, 59] = SOU
    target_ids[0, 69] = EOU
    ordinals[0, 59] = 1
    ordinals[0, 69] = 1

    pred_ids[0, 10] = SOB
    pred_ids[0, 28] = SOU
    pred_ids[0, 19] = SOB
    pred_ids[0, 12] = end_id
    pred_ids[0, 17] = end_id
    pred_ids[0, 44] = end_id
    pred_ids[0, 50] = end_id
    pred_ids[0, 59] = SOU
    pred_ids[0, 69] = EOU

    metrics = compute_user_backchannel_metrics(
        pred_ids=pred_ids,
        target_ids=target_ids,
        input_tokens=input_tokens,
        user_backchannel_event_ids=event_ids,
        user_backchannel_reference_frames=reference_frames,
        num_user_backchannel_events=torch.tensor([3]),
        boundary_turn_ordinals=ordinals,
        sob_id=SOB,
        user_backchannel_end_id=end_id,
        sou_id=SOU,
        eou_id=EOU,
        ignore_index=IGNORE_INDEX,
        audio_token_idx=AUDIO_TOKEN_IDX,
        collar_frames=3,
    )
    return metrics, pred_ids, target_ids, input_tokens, event_ids, ordinals


@pytest.mark.parametrize("end_id", [EOU, EOB])
def test_user_backchannel_confusion_end_matching_pairing_and_latencies(end_id):
    metrics, *_ = _metric_case(end_id)
    assert metrics["event_count"].item() == 3
    assert metrics["sob_correct_count"].item() == 1
    assert metrics["sou_confusion_count"].item() == 1
    assert metrics["no_start_marker_count"].item() == 1
    assert metrics["sob_spurious_count"].item() == 1
    assert metrics["end_in_collar_count"].item() == 1
    assert metrics["end_late_count"].item() == 1
    assert metrics["end_missing_count"].item() == 1
    assert metrics["end_spurious_count"].item() == 2
    assert metrics["paired_event_count"].item() == 1
    torch.testing.assert_close(metrics["sob_latency_frames"], torch.tensor([1.0]))
    torch.testing.assert_close(metrics["end_latency_frames"], torch.tensor([-2.0, 10.0]))


def test_sob_eob_counts_eou_as_end_confusion_and_excludes_it_from_substantive_metrics():
    metrics, pred_ids, target_ids, input_tokens, event_ids, ordinals = _metric_case(EOB)
    pred_ids[0, 12] = EOU
    pred_ids[0, 17] = TEXT
    pred_ids[0, 44] = EOU
    metrics = compute_user_backchannel_metrics(
        pred_ids=pred_ids,
        target_ids=target_ids,
        input_tokens=input_tokens,
        user_backchannel_event_ids=event_ids,
        user_backchannel_reference_frames=torch.where(event_ids > 0, torch.tensor(
            [[10, 15, 30, 35, 50, 55] + [-1] * 74]
        ), torch.full_like(event_ids, -1)),
        num_user_backchannel_events=torch.tensor([3]),
        boundary_turn_ordinals=ordinals,
        sob_id=SOB,
        user_backchannel_end_id=EOB,
        sou_id=SOU,
        eou_id=EOU,
        ignore_index=IGNORE_INDEX,
        audio_token_idx=AUDIO_TOKEN_IDX,
        collar_frames=3,
    )
    assert metrics["eou_end_confusion_count"].item() == 2
    assert metrics["prediction_exclusion_mask"][0, 12]
    assert metrics["prediction_exclusion_mask"][0, 44]

    collar = compute_boundary_token_metrics(
        pred_ids,
        target_ids,
        input_tokens,
        sou_id=SOU,
        eou_id=EOU,
        ignore_index=IGNORE_INDEX,
        audio_token_idx=AUDIO_TOKEN_IDX,
        user_backchannel_event_ids=event_ids,
        prediction_exclusion_mask=metrics["prediction_exclusion_mask"],
    )
    assert collar["eou_target_count"].item() == 1
    assert collar["eou_pred_count"].item() == 1


def test_backchannel_window_predictions_are_excluded_from_all_substantive_metrics():
    metrics, pred_ids, target_ids, input_tokens, event_ids, ordinals = _metric_case()
    exclusion = metrics["prediction_exclusion_mask"]
    assert exclusion[0, 28]
    assert exclusion[0, 12] and exclusion[0, 17] and exclusion[0, 44] and exclusion[0, 50]
    assert not exclusion[0, 59] and not exclusion[0, 69]

    collar = compute_boundary_token_metrics(
        pred_ids,
        target_ids,
        input_tokens,
        sou_id=SOU,
        eou_id=EOU,
        ignore_index=IGNORE_INDEX,
        audio_token_idx=AUDIO_TOKEN_IDX,
        sou_before_frames=0,
        sou_after_frames=0,
        eou_before_frames=0,
        eou_after_frames=0,
        user_backchannel_event_ids=event_ids,
        prediction_exclusion_mask=exclusion,
    )
    assert collar["sou_target_count"].item() == 1
    assert collar["eou_target_count"].item() == 1
    assert collar["sou_pred_count"].item() == 1
    assert collar["eou_pred_count"].item() == 1

    ordinal = compute_ordinal_boundary_timing_metrics(
        pred_ids,
        target_ids,
        input_tokens,
        boundary_turn_ordinals=ordinals,
        is_multiturn=torch.tensor([True]),
        num_substantive_turns=torch.tensor([1]),
        sou_id=SOU,
        eou_id=EOU,
        ignore_index=IGNORE_INDEX,
        audio_token_idx=AUDIO_TOKEN_IDX,
        max_turns=1,
        detection_tolerance_frames=0,
        prediction_exclusion_mask=exclusion,
    )
    assert ordinal["turn_1_sou_detection_count"].item() == 1
    assert ordinal["turn_1_eou_detection_count"].item() == 1
    assert ordinal["turn_1_sou_spurious_count"].item() == 0
    assert ordinal["turn_1_eou_spurious_count"].item() == 0


class _Logger:
    def __init__(self, mode="sob_eou"):
        self.logged = {}
        self.user_backchannel_mode = mode
        self.core_cfg = SimpleNamespace(
            enable_validation_checkpoint_score=False,
            enable_agent_backchannels=False,
            frame_length_in_secs=0.08,
        )

    def log(self, name, value, **kwargs):
        self.logged[name] = value

    def log_dict(self, metrics, **kwargs):
        self.logged.update(metrics)


def _append_epoch_metrics(model, metrics):
    for name, value in metrics.items():
        if name != "prediction_exclusion_mask":
            model._partial_user_backchannel_metrics[name].append(value)


@pytest.mark.parametrize(("mode", "end_id", "end_name"), [("sob_eou", EOU, "eou"), ("sob_eob", EOB, "eob")])
def test_pooled_latency_quantiles_and_mode_specific_names(mode, end_id, end_name):
    model = _Logger(mode)
    streaming_stt_model.StreamingSTTModel.on_validation_epoch_start(model)
    metrics, *_ = _metric_case(end_id)
    _append_epoch_metrics(model, metrics)
    streaming_stt_model.StreamingSTTModel.on_validation_epoch_end(model)

    assert model.logged["val_user_backchannel_sob_latency_p50_s"].item() == pytest.approx(0.08)
    assert model.logged[f"val_user_backchannel_{end_name}_latency_p50_s"].item() == pytest.approx(0.32)
    assert model.logged[f"val_user_backchannel_{end_name}_latency_p90_s"].item() == pytest.approx(0.704)


def test_undefined_latency_quantiles_are_nan():
    model = _Logger()
    streaming_stt_model.StreamingSTTModel.on_validation_epoch_start(model)
    zero = torch.tensor(0, dtype=torch.long)
    for name in (
        "num_samples",
        "event_count",
        "sob_correct_count",
        "sou_confusion_count",
        "no_start_marker_count",
        "sob_spurious_count",
        "end_in_collar_count",
        "end_late_count",
        "end_missing_count",
        "end_spurious_count",
        "eou_end_confusion_count",
        "paired_event_count",
    ):
        model._partial_user_backchannel_metrics[name].append(zero)
    model._partial_user_backchannel_metrics["sob_latency_frames"].append(torch.empty(0))
    model._partial_user_backchannel_metrics["end_latency_frames"].append(torch.empty(0))
    streaming_stt_model.StreamingSTTModel.on_validation_epoch_end(model)
    assert torch.isnan(model.logged["val_user_backchannel_sob_latency_p50_s"])
    assert torch.isnan(model.logged["val_user_backchannel_eou_latency_p90_s"])
