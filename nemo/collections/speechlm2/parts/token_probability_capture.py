# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-generation raw probability capture for positive-chunk native FSM decoding."""
from collections import deque
import math

import torch

SCHEMA = 'ttm_token_probabilities_v1'


def tracked_tokens(model):
    cfg = model.core_cfg
    fields = {'sou': 'utterance_start_token', 'eou': 'utterance_end_token'}
    if cfg.user_backchannel_mode != 'ignore':
        fields['sob'] = 'user_backchannel_start_token'
        if cfg.user_backchannel_mode == 'sob_eob':
            fields['eob'] = 'user_backchannel_end_token'
    result = {}
    for name, field in fields.items():
        piece = getattr(cfg, field)
        ids = model.tokenizer.tokenizer.encode(piece, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f'probability token must be one vocabulary ID: {piece!r}')
        result[name] = {'token_id': int(ids[0]), 'token_piece': piece}
    if len({v['token_id'] for v in result.values()}) != len(result):
        raise ValueError('tracked probability tokens must be distinct')
    return result


def selected_probabilities(logits, token_ids):
    """Never normalize just the tracked subset or mutate decoding logits."""
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError('capture requires explicit per-stream FSM sampling')
    selected = torch.log_softmax(logits.float(), dim=-1)[:, token_ids]
    if not torch.isfinite(selected).all():
        raise ValueError('nonfinite raw tracked-token log probabilities')
    return selected.exp().detach().cpu().tolist()[0], selected.detach().cpu().tolist()[0]


class TokenProbabilityCapture:
    def __init__(self, tokens, frame_seconds, chunk_frames):
        if chunk_frames <= 0 or not math.isfinite(frame_seconds) or frame_seconds <= 0:
            raise ValueError('probability capture requires positive fixed chunks/frame duration')
        self.tokens = tokens
        self.names = list(tokens)
        self.ids = [tokens[k]['token_id'] for k in self.names]
        self.frame_seconds = frame_seconds
        self.chunk_frames = chunk_frames
        self.pending = deque()
        self.streams = {}

    def before_sample(self, logits):
        return selected_probabilities(logits, self.ids)

    def after_sample(self, selected, values):
        if selected.numel() != 1:
            raise ValueError('capture requires one selected token per stream')
        probs, logs = values
        self.pending.append(dict(selected_token_id=int(selected.item()),
                                 probabilities=dict(zip(self.names, probs)),
                                 log_probabilities=dict(zip(self.names, logs))))

    def append(self, token_id, frames, token_list):
        if not self.pending:
            raise ValueError('appended token was not actually sampled')
        point = self.pending.popleft()
        if point['selected_token_id'] != token_id:
            raise ValueError('probability/selected-token mismatch')
        if frames <= 0 or frames % self.chunk_frames:
            raise ValueError('unexpected probability frame coordinate')
        point.update(prediction_index=len(token_list), encoder_frames_consumed=frames)
        self.streams.setdefault(id(token_list), []).append(point)

    def finish_stream(self, token_list, token_pieces):
        points = self.streams.pop(id(token_list), [])
        if len(token_pieces) != len(token_list) or [p['selected_token_id'] for p in points] != list(token_list):
            raise ValueError('probability trace must match every selected token one-to-one')
        chunks = []
        for point, piece in zip(points, token_pieces):
            frame = point.pop('encoder_frames_consumed')
            if not chunks or chunks[-1]['encoder_frames_consumed'] != frame:
                if chunks and frame <= chunks[-1]['encoder_frames_consumed']:
                    raise ValueError('nonmonotonic sampled chunk frames')
                chunks.append(dict(chunk_index=frame//self.chunk_frames-1,
                                   encoder_frames_consumed=frame,
                                   audio_time_seconds=frame*self.frame_seconds, predictions=[]))
            point.update(selected_token=piece, position_in_chunk=len(chunks[-1]['predictions']))
            chunks[-1]['predictions'].append(point)
        return chunks

    def assert_consumed(self):
        if self.pending or self.streams:
            raise ValueError('orphaned sampled probabilities or streams')

    def metadata(self):
        return dict(schema_version=SCHEMA, tokens=self.tokens,
                    probability_definition='FP32 full-vocabulary softmax of raw logits before decoding transformations',
                    log_probability_definition='natural logarithm; FP32 log_softmax',
                    timestamp_definition='encoder_frames_consumed * frame_length_seconds; no delay subtraction',
                    frame_length_seconds=self.frame_seconds, chunk_size_frames=self.chunk_frames)
