"""Canonical offline alignment projection, independent of audio/model libraries."""
import copy
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[3] / 'nemo/collections/common/data/lhotse/precomputed_alignments.py'
spec = importlib.util.spec_from_file_location('projection', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row():
    record = dict(schema_version=1, coordinate_system='physical_audio_member_seconds',
                  policy_sha256='a'*64, case_id='b'*64, alignment_record_sha256='c'*64,
                  raw_words_sha256='d'*64, status='qwen_aligned', model='aligner',
                  model_revision='revision', model_file_hashes={'model': 'e'*64},
                  words=[dict(word='Hello', start=10., end=11.)])
    return dict(word_alignment_mode='precomputed_required', offset=9., duration=3.,
                curation=dict(offline_word_alignment=dict(policy_sha256='a'*64, case_ids=['b'*64]),
                              target=dict(components=[dict(type='substantive_turn',
                                  fragments=[dict(text='Hello!')], precomputed_word_alignment=record)])))


def test_projection_uses_sampled_crop_and_preserves_source():
    source = row(); original = copy.deepcopy(source)
    assert module.project_precomputed_word_alignments(source)['alignments'] == [
        dict(text='Hello', start_time=1., end_time=2.)]
    assert source == original
    source['offset'] = 8.
    assert module.project_precomputed_word_alignments(source)['alignments'][0]['start_time'] == 2.


@pytest.mark.parametrize('include_audit_fields', [False, True])
def test_projection_does_not_require_or_validate_audit_hashes(include_audit_fields):
    source = row()
    expected = module.project_precomputed_word_alignments(source)['alignments']
    record = source['curation']['target']['components'][0]['precomputed_word_alignment']
    for field in ('alignment_record_sha256', 'raw_words_sha256', 'policy_sha256', 'model_file_hashes'):
        if include_audit_fields:
            record[field] = 'unused audit metadata'
        else:
            del record[field]
    provenance = source['curation']['offline_word_alignment']
    if include_audit_fields:
        provenance['policy_sha256'] = 'different unused audit metadata'
    else:
        del provenance['policy_sha256']
    record['case_id'] = 'alignment-case-1'
    provenance['case_ids'] = ['alignment-case-1']
    original = copy.deepcopy(source)
    assert module.project_precomputed_word_alignments(source)['alignments'] == expected
    assert source == original


@pytest.mark.parametrize('field,value,match', [
    ('coordinate_system', 'crop_seconds', 'coordinates'),
    ('case_id', 'f'*64, 'membership'), ('status', 'flagged', 'status'),
    ('case_id', None, 'membership'), ('case_id', '', 'membership'),
    ('words', [dict(word='Different', start=10., end=11.)], 'transcript'),
    ('words', [dict(word='Hello', start=8., end=11.)], 'outside'),
    ('words', [dict(word='Hello', start=10., end=float('nan'))], 'finite'),
])
def test_invalid_alignment_fails_closed(field, value, match):
    source = row()
    source['curation']['target']['components'][0]['precomputed_word_alignment'][field] = value
    with pytest.raises(ValueError, match=match):
        module.project_precomputed_word_alignments(source)


def test_online_passthrough_and_skip_rejection():
    online = dict(duration=2.)
    assert module.project_precomputed_word_alignments(online) is online
    source = row(); source['_skipme'] = True
    with pytest.raises(ValueError, match='ineligible'):
        module.project_precomputed_word_alignments(source)
