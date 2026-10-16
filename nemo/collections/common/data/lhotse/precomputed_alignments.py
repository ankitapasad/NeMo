# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict reader for canonical DAI offline-word-alignment schema version 1.

Records remain in physical-member coordinates. Projection is performed after
native context sampling, without changing the source row or saved word records.
"""
from collections.abc import Mapping
import math
import unicodedata


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{field} must be a finite number')
    return float(value)


def _text(text):
    if not isinstance(text, str):
        raise ValueError('alignment/fragment text must be a string')
    return ''.join(c.casefold() for c in unicodedata.normalize('NFKC', text) if c.isalnum())


def project_precomputed_word_alignments(row):
    mode = row.get('word_alignment_mode', 'online')
    if mode == 'online':
        return row
    if mode != 'precomputed_required':
        raise ValueError(f'unsupported word_alignment_mode: {mode!r}')
    curation = row.get('curation') or {}
    if row.get('_skipme') or row.get('eligible_for_training') is False or curation.get('eligible_for_training') is False:
        raise ValueError('ineligible offline-aligned row reached projection')
    target = curation.get('target', row.get('target'))
    components = target.get('components') if isinstance(target, Mapping) else None
    provenance = curation.get('offline_word_alignment', row.get('offline_word_alignment'))
    if not components or not isinstance(components, list) or not isinstance(provenance, Mapping):
        raise ValueError('canonical offline alignment components/provenance are missing')
    if provenance.get('flagged_case_ids') or provenance.get('release_blockers') or provenance.get('preexisting_skip'):
        raise ValueError('flagged alignment reached the training loader')
    expected = provenance.get('case_ids')
    if (not isinstance(expected, list) or not expected
            or any(not isinstance(case, str) or not case.strip() for case in expected)
            or len(expected) != len(set(expected))):
        raise ValueError('invalid alignment case membership')
    offset = _number(row.get('offset', 0), 'offset')
    duration = _number(row.get('duration'), 'duration')
    if offset < 0 or duration <= 0:
        raise ValueError('invalid crop offset/duration')
    projected, cases = [], []
    model_identity = None
    for component in components:
        if not isinstance(component, Mapping) or component.get('type') not in {'substantive_turn', 'backchannel'}:
            raise ValueError('unsupported aligned component')
        record = component.get('precomputed_word_alignment')
        if not isinstance(record, Mapping) or record.get('schema_version') != 1:
            raise ValueError('missing canonical precomputed_word_alignment schema 1')
        if record.get('coordinate_system') != 'physical_audio_member_seconds':
            raise ValueError('unsupported saved word coordinates')
        case_id = record.get('case_id')
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError('invalid alignment case membership identifier')
        cases.append((component["type"], case_id))
        if record.get('status') not in {'qwen_aligned', 'annotated_span_fallback'}:
            raise ValueError('alignment has unusable status')
        if record['status'] == 'annotated_span_fallback' and component['type'] != 'backchannel':
            raise ValueError('substantive alignment cannot use annotated fallback')
        identity = (record.get('model'), record.get('model_revision'))
        if not all(identity):
            raise ValueError('alignment model provenance is incomplete')
        if model_identity is not None and identity != model_identity:
            raise ValueError('mixed aligner identities within one row')
        model_identity = identity
        words, fragments = record.get('words'), component.get('fragments')
        if not isinstance(words, list) or not words or not isinstance(fragments, list) or not fragments:
            raise ValueError('empty aligned words or source fragments')
        if _text(' '.join(w['word'] for w in words)) != _text(' '.join(f['text'] for f in fragments)):
            raise ValueError('saved word transcript differs from component transcript')
        previous = -math.inf
        for word in words:
            text = word.get('word')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('empty saved word')
            start, end = _number(word.get('start'), 'word.start'), _number(word.get('end'), 'word.end')
            if start < 0 or end < start or start < previous:
                raise ValueError('invalid word geometry/order')
            local_start, local_end = start-offset, end-offset
            if local_start < -0.001 or local_end > duration+0.001:
                raise ValueError('saved words fall outside sampled crop')
            projected.append({'text': text, 'start_time': min(duration, max(0., local_start)), 'end_time': max(0., min(duration, local_end))})
            previous = start
    ordered_cases = [case for kind, case in cases if kind == "substantive_turn"] + [case for kind, case in cases if kind == "backchannel"]
    if ordered_cases != expected:
        raise ValueError('component alignment membership/order differs from provenance')
    projected.sort(key=lambda w: (w['start_time'], w['end_time'], w['text']))
    result = dict(row)
    result['alignments'] = projected
    return result
