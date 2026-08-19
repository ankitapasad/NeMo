# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

import nemo.collections.speechlm2.models.streaming_stt_model as streaming_stt_model


class _FakeTokenizer:
    def ids_to_tokens(self, token_ids):
        special = {10: "<sou>", 11: "<eou>", 20: "<soab>", 21: "<eoab>"}
        return [special.get(int(token_id), f"t{int(token_id)}") for token_id in token_ids]


class _FakeLLM:
    def __init__(self, emitted_ids: list[int], vocab_size: int = 128):
        self.config = SimpleNamespace(max_position_embeddings=64)
        self._emitted_ids = iter(emitted_ids)
        self._vocab_size = vocab_size

    def __call__(self, **kwargs):
        token_id = next(self._emitted_ids)
        batch_size = kwargs["inputs_embeds"].shape[0]
        logits = torch.full((batch_size, 1, self._vocab_size), -1000.0)
        logits[:, :, token_id] = 1000.0
        return SimpleNamespace(logits=logits, past_key_values=None)


def _bind_generation_helpers(model) -> None:
    model._get_boundary_token_info = MethodType(streaming_stt_model.StreamingSTTModel._get_boundary_token_info, model)
    model._append_sampled_token_and_boundary_event = MethodType(
        streaming_stt_model.StreamingSTTModel._append_sampled_token_and_boundary_event, model
    )
    model._build_generation_records = MethodType(
        streaming_stt_model.StreamingSTTModel._build_generation_records, model
    )


def _fake_dynamic_model(
    emitted_ids: list[int], *, user_footer_and_header=None, assistant_footer=None, backchannels=False
):
    model = SimpleNamespace()
    model.core_cfg = SimpleNamespace(
        chunk_size=2,
        frame_length_in_secs=0.1,
        sample_rate=10,
        add_utterance_boundary_tokens=True,
        utterance_start_token="<sou>",
        utterance_end_token="<eou>",
        enable_agent_backchannels=backchannels,
        agent_backchannel_start_token="<soab>",
        agent_backchannel_end_token="<eoab>",
        use_chunk_classifier=False,
    )
    model.embed_tokens = nn.Embedding(128, 4)
    model.llm = _FakeLLM(emitted_ids)
    model._user_header_ids = []
    model._user_footer_and_asst_header_ids = list(user_footer_and_header or [])
    model._asst_footer_ids = list(assistant_footer or [9])
    model._user_footer_first_id = None
    model._eos_id = 99
    model.text_pad_id = 0
    model.blank_token_id = 0
    model.blank_token = ""
    model.tokenizer = _FakeTokenizer()
    model.has_blank = False
    model._get_boundary_token_ids = lambda: (10, 11)
    model._get_agent_backchannel_token_ids = lambda: (20, 21) if backchannels else (None, None)
    model._sample_token = lambda logits, *_args, **_kwargs: logits.argmax(dim=-1)
    model.get_audio_feature_buffer = lambda **_kwargs: None
    model._build_offline_emb_chunks = lambda *_args, **_kwargs: [torch.zeros(1, 2, 4)]
    model.get_init_streaming_state = lambda *_args, **_kwargs: SimpleNamespace(
        cache=None,
        seq_lens=[0],
        attention_mask=None,
        audio_feature_buffer=None,
        audio_cache=None,
        aux_hidden_buffer=None,
    )
    model._dynamic_finish_generating = streaming_stt_model.StreamingSTTModel._dynamic_finish_generating
    _bind_generation_helpers(model)
    return model


def _run_dynamic(model, *, detailed=False):
    return streaming_stt_model.StreamingSTTModel._generate_dynamic_streaming(
        model,
        audios=torch.zeros(1, 2),
        n_samples_list=[2],
        system_prompt="prompt",
        max_new_tokens=10,
        use_offline_embs=True,
        return_generation_records=detailed,
    )


def test_state_machine_default_return_type_and_text_are_unchanged(monkeypatch):
    monkeypatch.setattr(
        streaming_stt_model,
        "decode_with_blank",
        lambda token_ids, *_args: ",".join(str(token_id) for token_id in token_ids),
    )

    result = _run_dynamic(_fake_dynamic_model([0, 10, 12, 11, 9, 0]))

    assert result == ["10,12,11,99"]
    assert all(isinstance(text, str) for text in result)


def test_state_machine_offline_embeddings_return_exact_tokens_and_boundaries(monkeypatch):
    monkeypatch.setattr(
        streaming_stt_model,
        "decode_with_blank",
        lambda token_ids, *_args: ",".join(str(token_id) for token_id in token_ids),
    )

    records = _run_dynamic(_fake_dynamic_model([0, 10, 12, 11, 9, 0]), detailed=True)

    assert records == [
        streaming_stt_model.StreamingGenerationRecord(
            pred_text_unnormalized="10,12,11,99",
            sampled_token_ids=[10, 12, 11, 9],
            sampled_token_pieces=["<sou>", "t12", "<eou>", "t9"],
            boundary_events=[
                streaming_stt_model.BoundaryEvent(
                    boundary_type="sou",
                    token_id=10,
                    token_piece="<sou>",
                    sampled_token_sequence_index=0,
                    encoder_frames_consumed=2,
                    emission_time_seconds=pytest.approx(0.2),
                ),
                streaming_stt_model.BoundaryEvent(
                    boundary_type="eou",
                    token_id=11,
                    token_piece="<eou>",
                    sampled_token_sequence_index=2,
                    encoder_frames_consumed=2,
                    emission_time_seconds=pytest.approx(0.2),
                ),
            ],
        )
    ]


def test_state_machine_preserves_two_ordered_sou_eou_pairs(monkeypatch):
    monkeypatch.setattr(streaming_stt_model, "decode_with_blank", lambda token_ids, *_args: list(token_ids))

    records = _run_dynamic(_fake_dynamic_model([0, 10, 12, 11, 10, 13, 11, 9, 0]), detailed=True)

    assert records[0].sampled_token_ids == [10, 12, 11, 10, 13, 11, 9]
    assert [event.boundary_type for event in records[0].boundary_events] == ["sou", "eou", "sou", "eou"]
    assert [event.sampled_token_sequence_index for event in records[0].boundary_events] == [0, 2, 3, 5]
    assert [event.encoder_frames_consumed for event in records[0].boundary_events] == [2, 2, 2, 2]


def test_empty_transition_and_single_token_assistant_footer_stop_cleanly(monkeypatch):
    monkeypatch.setattr(streaming_stt_model, "decode_with_blank", lambda token_ids, *_args: list(token_ids))
    records = _run_dynamic(
        _fake_dynamic_model([0, 9, 0], user_footer_and_header=[], assistant_footer=[9]), detailed=True
    )

    assert records[0].pred_text_unnormalized == [99]
    assert records[0].sampled_token_ids == [9]


def test_optional_agent_backchannel_events_are_retained_without_pairing(monkeypatch):
    monkeypatch.setattr(streaming_stt_model, "decode_with_blank", lambda token_ids, *_args: list(token_ids))
    records = _run_dynamic(_fake_dynamic_model([0, 20, 12, 9, 0], backchannels=True), detailed=True)

    assert records[0].sampled_token_ids == [20, 12, 9]
    assert [event.boundary_type for event in records[0].boundary_events] == ["soab"]
    assert records[0].boundary_events[0].sampled_token_sequence_index == 0


def test_disabled_agent_backchannels_are_not_interpreted_as_events(monkeypatch):
    monkeypatch.setattr(streaming_stt_model, "decode_with_blank", lambda token_ids, *_args: list(token_ids))
    records = _run_dynamic(_fake_dynamic_model([0, 20, 12, 9, 0], backchannels=False), detailed=True)

    assert records[0].sampled_token_ids == [20, 12, 9]
    assert records[0].boundary_events == []


def _fake_chunked_model():
    model = SimpleNamespace()
    model.core_cfg = SimpleNamespace(
        chunk_size=2,
        frame_length_in_secs=0.1,
        sample_rate=10,
        add_utterance_boundary_tokens=True,
        utterance_start_token="<sou>",
        utterance_end_token="<eou>",
        enable_agent_backchannels=False,
    )
    model.blank_token = "<blank>"
    model.tokenizer = _FakeTokenizer()
    model.get_init_streaming_state = lambda *_args, **_kwargs: object()
    model._get_boundary_token_ids = lambda: (10, 11)
    model._get_agent_backchannel_token_ids = lambda: (None, None)
    outputs = iter(
        (
            ([[10, 7], [11]], [[10, 7, 9], [11, 9]]),
            ([[11], []], [[11, 9], [9]]),
        )
    )
    model._chunked_streaming_step = lambda *_args, **_kwargs: next(outputs)
    _bind_generation_helpers(model)
    return model


def _run_chunked(model, *, detailed=False):
    return streaming_stt_model.StreamingSTTModel._generate_chunked_streaming(
        model,
        audios=torch.zeros(2, 4),
        n_samples_list=[4, 2],
        system_prompt="prompt",
        max_new_tokens=4,
        return_generation_records=detailed,
    )


def test_fixed_streaming_records_preserve_text_exact_tokens_and_stream_timestamps(monkeypatch):
    monkeypatch.setattr(streaming_stt_model, "decode_with_blank", lambda token_ids, *_args: list(token_ids))
    without_records = _run_chunked(_fake_chunked_model())
    records = _run_chunked(_fake_chunked_model(), detailed=True)

    assert without_records == [[10, 7, 11], [11]]
    assert [record.pred_text_unnormalized for record in records] == without_records
    assert records[0].sampled_token_ids == [10, 7, 9, 11, 9]
    assert records[1].sampled_token_ids == [11, 9]
    assert [
        (event.boundary_type, event.encoder_frames_consumed, event.sampled_token_sequence_index)
        for event in records[0].boundary_events
    ] == [("sou", 2, 0), ("eou", 4, 3)]
    assert records[0].boundary_events[0].emission_time_seconds == pytest.approx(0.2)
    assert records[0].boundary_events[1].emission_time_seconds == pytest.approx(0.4)
    assert records[1].boundary_events[0].emission_time_seconds == pytest.approx(0.2)


def test_fixed_streaming_preserves_two_ordered_sou_eou_pairs(monkeypatch):
    monkeypatch.setattr(streaming_stt_model, "decode_with_blank", lambda token_ids, *_args: list(token_ids))
    model = _fake_chunked_model()
    outputs = iter(
        (
            ([[10, 7]], [[10, 7, 9]]),
            ([[11, 10, 8]], [[11, 10, 8, 9]]),
            ([[11]], [[11, 9]]),
        )
    )
    model._chunked_streaming_step = lambda *_args, **_kwargs: next(outputs)

    records = streaming_stt_model.StreamingSTTModel._generate_chunked_streaming(
        model,
        audios=torch.zeros(1, 6),
        n_samples_list=[6],
        system_prompt="prompt",
        max_new_tokens=4,
        return_generation_records=True,
    )

    assert records[0].sampled_token_ids == [10, 7, 9, 11, 10, 8, 9, 11, 9]
    assert [event.boundary_type for event in records[0].boundary_events] == ["sou", "eou", "sou", "eou"]
    assert [event.sampled_token_sequence_index for event in records[0].boundary_events] == [0, 3, 4, 7]
    assert [event.encoder_frames_consumed for event in records[0].boundary_events] == [2, 4, 4, 6]


def test_generation_record_builder_rejects_sample_count_mismatch():
    model = _fake_dynamic_model([])

    with pytest.raises(RuntimeError, match="Generation output count mismatch"):
        model._build_generation_records(["one", "two"], [[10]], [[]])


def _fake_generate_dispatch_model(chunk_size: int):
    model = SimpleNamespace(core_cfg=SimpleNamespace(chunk_size=chunk_size))
    model._ensure_inference_cache = lambda: None
    model._generate_offline = lambda *_args, **_kwargs: ["offline"]
    model._generate_dynamic_streaming = lambda *_args, **kwargs: [
        f"state-machine:{kwargs['return_generation_records']}"
    ]
    model._generate_chunked_streaming = lambda *_args, **kwargs: [f"fixed:{kwargs['return_generation_records']}"]
    return model


def _call_generate(model, *, state_machine=False, detailed=False):
    return streaming_stt_model.StreamingSTTModel.generate.__wrapped__(
        model,
        audios=torch.zeros(1, 2),
        audio_lens=torch.tensor([2]),
        use_state_machine_inference=state_machine,
        return_generation_records=detailed,
    )


def test_generate_keeps_fixed_and_state_machine_text_dispatch_compatible(monkeypatch):
    monkeypatch.setattr(streaming_stt_model, "move_embedding", lambda _model: nullcontext())
    model = _fake_generate_dispatch_model(chunk_size=2)

    assert _call_generate(model) == ["fixed:False"]
    assert _call_generate(model, state_machine=True) == ["state-machine:False"]
    assert _call_generate(model, state_machine=True, detailed=True) == ["state-machine:True"]


@pytest.mark.parametrize("chunk_size", [-1, 0])
def test_detailed_records_reject_nonpositive_chunk_sizes(monkeypatch, chunk_size):
    monkeypatch.setattr(streaming_stt_model, "move_embedding", lambda _model: nullcontext())
    model = _fake_generate_dispatch_model(chunk_size=chunk_size)

    with pytest.raises(ValueError, match="positive streaming chunk_size"):
        _call_generate(model, detailed=True)
