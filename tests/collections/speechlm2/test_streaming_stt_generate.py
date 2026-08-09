# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import inspect
from types import SimpleNamespace

import pytest

from examples.speechlm2 import streaming_stt_generate
from nemo.collections.speechlm2.models import BoundaryEvent, StreamingGenerationRecord


def _cut(cut_id: str, clip_id: str | None, duration: float):
    return SimpleNamespace(id=cut_id, duration=duration, custom={"clip_id": clip_id} if clip_id else {})


def _generation_record(text: str, token_id: int, token_piece: str, event: BoundaryEvent):
    return StreamingGenerationRecord(
        pred_text_unnormalized=text,
        sampled_token_ids=[token_id],
        sampled_token_pieces=[token_piece],
        boundary_events=[event],
    )


def test_output_records_follow_actual_yield_order_and_use_clean_schema():
    event_b = BoundaryEvent("sou", 10, "<sou>", 0, 2, 0.2)
    event_a = BoundaryEvent("eou", 11, "<eou>", 0, 4, 0.4)
    ordered_inputs = [
        (_cut("cut-b", "clip-b", 2.0), "b", "b", _generation_record("<sou> b", 10, "<sou>", event_b)),
        (_cut("cut-a", "clip-a", 1.0), "a", "a", _generation_record("a <eou>", 11, "<eou>", event_a)),
    ]

    records = [streaming_stt_generate._output_record(*args) for args in ordered_inputs]

    assert [record["id"] for record in records] == ["cut-b", "cut-a"]
    assert [record["clip_id"] for record in records] == ["clip-b", "clip-a"]
    assert records[0]["pred_text"] == "b"
    assert records[0]["pred_text_unnormalized"] == "<sou> b"
    assert records[0]["sampled_token_ids"] == [10]
    assert records[0]["sampled_token_pieces"] == ["<sou>"]
    assert records[0]["boundary_events"] == [
        {
            "boundary_type": "sou",
            "token_id": 10,
            "token_piece": "<sou>",
            "sampled_token_sequence_index": 0,
            "encoder_frames_consumed": 2,
            "emission_time_seconds": 0.2,
        }
    ]
    assert not {
        "pred_text_raw",
        "raw_pred_tokens",
        "raw_pred_text_with_special",
        "sou_emissions",
        "eou_emissions",
        "first_sou_time",
        "first_eou_time",
    }.intersection(records[0])


def test_text_only_output_preserves_established_fields_without_details():
    record = streaming_stt_generate._output_record(_cut("cut-a", None, 1.0), "hello", "hello")

    assert record == {
        "id": "cut-a",
        "clip_id": None,
        "duration": 1.0,
        "text": "hello",
        "pred_text": "hello",
        "wer": 0.0,
        "ins": 0.0,
        "del": 0.0,
        "sub": 0.0,
    }


def test_detailed_output_requires_source_clip_id():
    event = BoundaryEvent("sou", 10, "<sou>", 0, 2, 0.2)
    details = _generation_record("hello", 10, "<sou>", event)

    with pytest.raises(RuntimeError, match="source clip_id"):
        streaming_stt_generate._output_record(_cut("cut-a", None, 1.0), "hello", "hello", details)


def test_detailed_output_rejects_sampled_id_piece_length_mismatch():
    details = StreamingGenerationRecord(
        pred_text_unnormalized="hello",
        sampled_token_ids=[10],
        sampled_token_pieces=[],
        boundary_events=[],
    )

    with pytest.raises(RuntimeError, match="ID/piece length mismatch"):
        streaming_stt_generate._output_record(_cut("cut-a", "clip-a", 1.0), "hello", "hello", details)


def test_duration_batching_keeps_both_cut_and_duration_limits():
    """Guard the inline sampler setup without reintroducing a sampler helper."""
    source = inspect.getsource(streaming_stt_generate.main.__wrapped__)

    assert "max_cuts=cfg.batch_size" in source
    assert "max_duration=cfg.max_batch_duration" in source
    assert "num_batches = None if cfg.max_batch_duration is not None" in source
