"""Native manifest -> context crop -> Lhotse estimator integration (CPU, real audio)."""
import copy
import io
import json
import tarfile

import numpy as np
import pytest
import soundfile as sf
from omegaconf import OmegaConf

from nemo.collections.common.data.lhotse.nemo_adapters import (
    LazyNeMoIterator, LazyNeMoTarredIterator, _lean_multiturn_context_geometry,
    _sample_lean_multiturn_context,
)
from nemo.collections.speechlm2.data.batch_estimation import (
    context_duration_rows, estimate_context_duration_bins,
)


@pytest.fixture
def source(tmp_path):
    rows = []
    manifest = tmp_path / 'manifest_0.jsonl'
    audio_tar = tmp_path / 'audio_0.tar'
    with tarfile.open(audio_tar, 'w') as tar:
        for i in range(12):
            core = 2 + i / 4
            row = dict(audio_filepath=f'a{i}-sub1.wav', sample_id=f's{i}', shard_id=0,
                       offset=2., duration=core + 2, source_audio_offset=102., text='hello',
                       curation=dict(schema_version='lean_multi_turn_v3',
                         physical_audio=dict(duration=10., source_start=100.,
                             stores_all_available_transcript_bounded_context=True),
                         audio_context=dict(leading_sil=1., trailing_sil=1.,
                             max_available_leading_sil=3., max_available_trailing_sil=7-core),
                         target=dict(utterance_regions=[dict(start=1., duration=core)],
                             components=[dict(fragments=[dict(start=1., duration=core)])], pause_regions=[])))
            rows.append(row)
            buffer = io.BytesIO()
            sf.write(buffer, np.zeros(160000, dtype=np.float32), 16000, format='WAV')
            payload = buffer.getvalue()
            entry = tarfile.TarInfo(f'a{i}.wav')
            entry.size = len(payload)
            tar.addfile(entry, io.BytesIO(payload))
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    context = dict(min_leading_s=.05, min_trailing_s=.4, max_leading_s=3., max_trailing_s=4., max_duration_s=9.)
    cfg = tmp_path / 'input.yaml'
    OmegaConf.save(OmegaConf.create([dict(type='nemo_tarred', manifest_filepath=str(manifest),
                          tarred_audio_filepaths=str(audio_tar), context_sampling=context)]), cfg)
    return manifest, audio_tar, cfg, rows, context


@pytest.mark.parametrize('seed', [0, 17, 1234])
def test_export_matches_real_tarred_training_crops(source, seed):
    manifest, audio_tar, cfg, original, context = source
    rows = context_duration_rows(cfg, seed=seed)
    native = list(LazyNeMoTarredIterator(str(manifest), str(audio_tar),
                  context_sampling=context, shard_seed=seed, shuffle_shards=False))
    metadata = list(LazyNeMoIterator(str(manifest), metadata_only=True,
                    context_sampling=context, shard_seed=seed))
    assert len(rows) == len(native) == len(metadata) == 12
    for row, train, meta in zip(rows, native, metadata):
        assert row['id'] == train.sample_id
        assert row['sampled_duration'] == pytest.approx(train.duration, abs=1/16000)
        assert meta.source_audio_offset == train.source_audio_offset
        assert meta.curation == train.curation
        assert row['min_duration'] <= row['sampled_duration'] <= row['max_duration']
        assert train.load_audio().shape[-1] == round(train.duration * 16000)
    from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets
    result = estimate_context_duration_bins(cfg, seed=seed, num_buckets=3)
    expected = estimate_duration_buckets(metadata, num_buckets=3)
    assert result['estimator_duration_bins'] == expected
    assert result['bucket_duration_bins'][-1] == max(r['max_duration'] for r in rows)


def test_bounds_are_analytic_and_seed_independent(source):
    manifest, audio_tar, cfg, rows, context = source
    first = context_duration_rows(cfg, seed=0)
    second = context_duration_rows(cfg, seed=1)
    assert [r['sampled_duration'] for r in first] != [r['sampled_duration'] for r in second]
    for i, (a, b) in enumerate(zip(first, second)):
        core = 2 + i / 4
        assert a['min_duration'] == b['min_duration'] == core + .05 + .4
        assert a['max_duration'] == b['max_duration'] == min(9, core + 3 + min(4, 7-core))
    config = OmegaConf.load(cfg)
    config[0].context_sampling.max_leading_s = .05
    config[0].context_sampling.max_trailing_s = .4
    OmegaConf.save(config, cfg)
    fixed = context_duration_rows(cfg)
    for row in fixed:
        assert row['min_duration'] == pytest.approx(row['sampled_duration'])
        assert row['max_duration'] == pytest.approx(row['sampled_duration'])


def test_saved_alignment_expands_required_context(source):
    _, _, _, rows, context = source
    row = copy.deepcopy(rows[0])
    row['word_alignment_mode'] = 'precomputed_required'
    row['curation']['target']['components'][0]['precomputed_word_alignment'] = {
        'words': [dict(start=2., end=6.)]}
    geometry = _lean_multiturn_context_geometry(row, recording_duration=10., context_sampling=context)
    assert geometry['min_duration'] == 4.
    for seed in range(20):
        crop = _sample_lean_multiturn_context(row, recording_duration=10., context_sampling=context, seed=seed)
        assert crop['offset'] <= 2.
        assert crop['offset'] + crop['duration'] >= 6.


def test_invalid_context_fails_and_skipme_follows_native_policy(source):
    manifest, _, cfg, rows, context = source
    rows[0]['curation']['physical_audio']['duration'] = None
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(TypeError, match='physical_audio.duration'):
        context_duration_rows(cfg)
    rows[0]['_skipme'] = True
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert len(context_duration_rows(cfg)) == 11


def test_impossible_budget_and_header_mismatch(source):
    _, _, _, rows, context = source
    with pytest.raises(ValueError, match='decoded tar member'):
        _lean_multiturn_context_geometry(rows[0], recording_duration=11., context_sampling=context)
    context['max_duration_s'] = 2.
    with pytest.raises(ValueError, match='does not fit'):
        _lean_multiturn_context_geometry(rows[0], recording_duration=10., context_sampling=context)


def test_no_context_keeps_manifest_durations(source):
    _, _, cfg, original, _ = source
    config = OmegaConf.load(cfg)
    del config[0]['context_sampling']
    OmegaConf.save(config, cfg)
    rows = context_duration_rows(cfg)
    assert [r['sampled_duration'] for r in rows] == [r['duration'] for r in original]
    assert all(r['min_duration'] == r['sampled_duration'] == r['max_duration'] for r in rows)


def test_multiple_sources_rejected(source):
    _, _, cfg, _, _ = source
    config = OmegaConf.load(cfg)
    config.append(copy.deepcopy(config[0]))
    OmegaConf.save(config, cfg)
    with pytest.raises(ValueError, match='exactly one'):
        context_duration_rows(cfg)


@pytest.mark.parametrize('schema', ['lean_multi_turn_v2', 'lean_multi_turn_v3'])
def test_multiturn_geometry_includes_internal_gaps(source, schema):
    _, _, _, rows, context = source
    row = copy.deepcopy(rows[0])
    row['curation']['schema_version'] = schema
    row['curation']['target']['utterance_regions'] = [
        dict(start=1., duration=.25), dict(start=2., duration=.25), dict(start=2.75, duration=.25)]
    bounds = _lean_multiturn_context_geometry(row, recording_duration=10., context_sampling=context)
    assert bounds['core_duration'] == 2.  # Includes internal silences, not just .75s of speech.
    assert bounds['min_duration'] == pytest.approx(2.45)


def test_full_context_reaches_physical_recording_bound(source):
    _, _, _, rows, context = source
    context.update(max_leading_s=-1., max_trailing_s=-1., max_duration_s=30.)
    geometry = _lean_multiturn_context_geometry(rows[0], recording_duration=10., context_sampling=context)
    assert geometry['max_duration'] == 10.


@pytest.mark.parametrize('entrypoint', ['nemo', 'data'])
def test_cli_writes_native_rows_and_refuses_overwrite(source, monkeypatch, tmp_path, entrypoint):
    import runpy
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    _, _, cfg, _, _ = source
    output = tmp_path / 'geometry.json'
    if entrypoint == 'nemo':
        script = root / 'scripts/speechlm2/estimate_duration_bins.py'
        args = ['--input-cfg', str(cfg)]
    else:
        script = root.parent / 'ttm-data-processing/scripts/batch_estimation/run.py'
        if not script.exists():
            pytest.skip('Paired ttm-data-processing checkout unavailable')
        args = ['--nemo-root', str(root), 'context-geometry', '--input', str(cfg)]
    monkeypatch.setattr(sys, 'argv', [str(script), *args, '--num-buckets', '3', '--output', str(output)])
    try:
        runpy.run_path(str(script), run_name='__main__')
    except SystemExit as exc:
        assert exc.code == 0
    result = json.loads(output.read_text())
    assert len(result['rows']) == 12
    assert result['validation_status'] == 'METADATA_ONLY_NOT_TRAINING_READY'
    before = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name='__main__')
    assert exc.value.code == 2
    assert output.read_bytes() == before


def test_input_config_tags_reach_generic_exposure_tracker(source, monkeypatch):
    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
    from nemo.collections.speechlm2.data.data_seen_tracker import exposure_metadata, DataSeenTracker
    _, _, cfg, original, _ = source
    sources = OmegaConf.load(cfg)
    sources[0]['tags'] = dict(data_seen_source='meeting_corpus', data_seen_category='three_turn')
    cuts, _ = read_cutset_from_config(OmegaConf.create(dict(
        input_cfg=sources, metadata_only=True, force_finite=True, shuffle=False, shard_seed=42)))
    monkeypatch.setenv(DataSeenTracker.ENV_PATH, '/unused/totals.json')
    monkeypatch.setenv(DataSeenTracker.ENV_LABEL_MODE, 'explicit')
    metadata = exposure_metadata(list(cuts))
    assert metadata['mixture_sources'] == ['meeting_corpus'] * 12
    assert metadata['turn_buckets'] == ['three_turn'] * 12
    assert metadata['sample_ids'] == [row['sample_id'] for row in original]
    assert metadata['accounting_durations_s'] == [row['duration'] for row in original]


@pytest.mark.parametrize('mode', ['minimum', 'maximum_leading', 'maximum_trailing'])
def test_analysis_corners_share_native_reader_without_mutating_input(source, mode):
    from nemo.collections.speechlm2.data.batch_estimation import cuts_for
    _, _, cfg, _, _ = source
    item = OmegaConf.to_container(OmegaConf.load(cfg)[0], resolve=True)
    original = copy.deepcopy(item)
    meta = list(cuts_for(item, mode, 42, True))
    real = list(cuts_for(item, mode, 42, False))
    assert item == original
    for m, c in zip(meta, real):
        assert m.duration == pytest.approx(c.duration, abs=1/16000)
        assert m.source_audio_offset == pytest.approx(c.source_audio_offset)
        bounds = m.context_duration_bounds
        expected = bounds['min_duration' if mode == 'minimum' else 'max_duration']
        assert c.duration == pytest.approx(expected, abs=1/16000)
        assert c.load_audio().shape[-1] == round(c.duration * 16000)
    assert list(cuts_for(item))[0].duration == context_duration_rows(cfg)[0]['sampled_duration']


def test_selected_crops_match_ids_when_manifest_and_tar_order_differ(source):
    from nemo.collections.speechlm2.data.batch_estimation import cuts_for
    from nemo.collections.speechlm2.data.batch_profiling import selected_crops
    manifest, _, cfg, _, _ = source
    lines = manifest.read_text().splitlines(keepends=True)
    manifest.write_text(''.join(reversed(lines)))
    item = OmegaConf.to_container(OmegaConf.load(cfg)[0], resolve=True)
    rows = context_duration_rows(cfg)
    requested = [dict(source_id=row['id'], duration=row['sampled_duration']) for row in rows[:3]]
    matched = list(selected_crops(cuts_for(item, metadata=False), requested))
    assert len(matched) == 3
    assert [cut.sample_id for _, cut in matched] != [x['source_id'] for x in requested]
    assert {cut.sample_id for _, cut in matched} == {x['source_id'] for x in requested}
    for selected, cut in matched:
        assert selected['source_id'] == cut.sample_id
    bad = [dict(requested[0], duration=requested[0]['duration'] + .1)]
    with pytest.raises(ValueError, match='Crop duration differs'):
        list(selected_crops(cuts_for(item, metadata=False), bad))
    with pytest.raises(ValueError, match='missing in real traversal'):
        list(selected_crops(cuts_for(item, metadata=False), [dict(source_id='absent', duration=1)]))


def test_plain_manifest_without_sample_id_matches_subsegments(source):
    from nemo.collections.speechlm2.data.batch_estimation import cuts_for, sample_identity
    from nemo.collections.speechlm2.data.batch_profiling import selected_crops
    manifest, _, cfg, rows, _ = source
    for row in rows:
        row.pop('sample_id')
    # Same tar member, two distinct manifest subsegments must retain distinct IDs.
    second = copy.deepcopy(rows[0])
    second.update(audio_filepath='a0-sub2.wav', offset=1., duration=2.)
    rows.append(second)
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in reversed(rows)))
    item = OmegaConf.to_container(OmegaConf.load(cfg)[0], resolve=True)
    item.pop('context_sampling')
    meta = list(cuts_for(item, metadata=True))
    real = list(cuts_for(item, metadata=False))
    assert meta[0].id != real[0].id
    requests = [dict(source_id=sample_identity(c), duration=c.duration) for c in meta]
    matched = list(selected_crops(real, requests))
    assert len(matched) == len(rows)
    assert len({sample_identity(c) for c in meta}) == len(rows)
    assert {sample_identity(c) for c in real} == {x['source_id'] for x in requests}
