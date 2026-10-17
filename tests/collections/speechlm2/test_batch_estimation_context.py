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


def test_selective_tar_collection_reads_only_selected_members_once(source, tmp_path, monkeypatch):
    from nemo.collections.speechlm2.data.batch_collection import SelectedTarIterator
    from nemo.collections.speechlm2.data.batch_estimation import cuts_for, sample_identity
    from nemo.collections.speechlm2.data.batch_profiling import selected_crops
    manifest, audio_tar, cfg, rows, _ = source
    item = OmegaConf.to_container(OmegaConf.load(cfg)[0], resolve=True)
    item['tags'] = dict(system_prompt='Transcribe with boundary events.', add_utterance_boundary_tokens=True, data_seen_source='arbitrary_source')
    modes = ('sampled', 'minimum', 'maximum_leading', 'maximum_trailing')
    requests = []
    expected = {}
    for mode in modes:
        real = list(cuts_for(item, mode, 42, False))
        for cut in real:
            if sample_identity(cut) in ('s1', 's9'):
                requests.append(dict(source_id=sample_identity(cut), mode=mode, duration=cut.duration))
                expected[mode, sample_identity(cut)] = cut
    # Reverse manifest order; selected membership must not depend on archive order.
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in reversed(rows)))
    reader = SelectedTarIterator(item, requests, 42, tmp_path/'cache')
    extracted = []
    original = tarfile.TarFile.extractfile
    def tracked(archive, member):
        extracted.append(member.name)
        return original(archive, member)
    monkeypatch.setattr(tarfile.TarFile, 'extractfile', tracked)
    for mode in reversed(modes):
        selected = [r for r in requests if r['mode'] == mode]
        for _, cut in selected_crops(reader.for_mode(mode), selected):
            baseline = expected[mode, sample_identity(cut)]
            assert cut.duration == baseline.duration
            assert cut.start == baseline.start
            assert cut.source_audio_offset == baseline.source_audio_offset
            for key, value in item['tags'].items():
                assert cut.custom[key] == baseline.custom[key] == value
            np.testing.assert_array_equal(cut.load_audio(), baseline.load_audio())
    assert extracted == ['a1.wav', 'a9.wav']
    assert reader.members_read == 2
    assert reader.shards_opened == 1
    # A new process-equivalent reader reuses validated cache without reopening tar.
    audio_tar.rename(audio_tar.with_suffix('.hidden'))
    resumed = SelectedTarIterator(item, requests, 42, tmp_path/'cache')
    assert len(list(resumed.for_mode('sampled'))) == 2
    assert resumed.members_read == resumed.shards_opened == 0
    cached = next(p for p in (tmp_path/'cache').iterdir() if p.suffix == '')
    cached.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Cached recording changed'):
        list(resumed.for_mode('sampled'))


def test_selective_collection_never_opens_unselected_shards(source, tmp_path):
    from nemo.collections.speechlm2.data.batch_collection import SelectedTarIterator
    manifest, audio_tar, cfg, rows, _ = source
    item = OmegaConf.to_container(OmegaConf.load(cfg)[0], resolve=True)
    other = tmp_path/'manifest_1.jsonl'
    row = dict(rows[0], sample_id='unselected', shard_id=1)
    other.write_text(json.dumps(row)+'\n')
    item['manifest_filepath'] = [str(manifest), str(other)]
    item['tarred_audio_filepaths'] = [str(audio_tar), str(tmp_path/'audio_1.tar')]
    reader = SelectedTarIterator(item, [dict(source_id='s0')], 42, tmp_path/'cache')
    assert len(list(reader.for_mode('sampled'))) == 1
    assert reader.shards_opened == reader.members_read == 1
    with pytest.raises(ValueError, match='absent from manifest'):
        SelectedTarIterator(item, [dict(source_id='missing')], 42, tmp_path/'other_cache')


def test_cpu_collection_resume_receipts_without_model_or_cuda(source, tmp_path, monkeypatch):
    """Real audio/cache orchestration; formatter stub isolates model dependencies."""
    import sys
    import types
    import torch
    from nemo.collections.speechlm2.data.batch_collection import collect
    from nemo.collections.speechlm2.data.batch_pipeline import write_json, load_json
    from nemo.collections.speechlm2.data.batch_estimation import cuts_for, sample_identity
    from nemo.collections.speechlm2.data.batch_profiling import worker
    _, _, cfg_path, _, _ = source
    source_cfg = OmegaConf.to_container(OmegaConf.load(cfg_path)[0], resolve=True)
    selected = []
    for mode in ('sampled', 'maximum_trailing'):
        cut = next(iter(cuts_for(source_cfg, mode, 42, True)))
        selected.append(dict(id='0:0:'+mode, source_index=0, source_id=sample_identity(cut),
                             mode=mode, duration=cut.duration, bucket=0))
    output = tmp_path/'output'
    write_json(output/'dai/geometry.json', dict(sources=[source_cfg], candidates=selected))
    model_cfg = tmp_path/'model.yaml'
    OmegaConf.save(OmegaConf.create(dict(model={}, data=dict(dataset=dict(sample_rate=16000)))), model_cfg)
    plan = dict(model_config=str(model_cfg), seed=42)
    args = types.SimpleNamespace(action='collect', output=output, profile='dai')
    calls = []
    class Dataset:
        def __init__(self, **kwargs): pass
        def __getitem__(self, cuts):
            calls.append('decode')
            for cut in cuts:
                cut.load_audio()
            return types.SimpleNamespace(cuts=cuts, audios=None, audio_lens=[1], text=['hello'])
        def get_batch_data(self, *args):
            return types.SimpleNamespace(audio_lens=[1], input_token_lens=[3], target_token_lens=[2])
    monkeypatch.setattr(sys.modules['nemo.collections.speechlm2'], 'StreamingSTTDataset', Dataset, raising=False)
    fake_model = types.ModuleType('nemo.collections.speechlm2.models.streaming_stt_model')
    fake_model.StreamingSTTModelConfig = object
    fake_model.setup_streaming_tokenizer = lambda cfg: (object(), '')
    monkeypatch.setitem(sys.modules, fake_model.__name__, fake_model)
    utils = types.ModuleType('nemo.collections.speechlm2.parts.utils')
    utils.to_dataclass = lambda cls, cfg: cfg
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    alignments = types.ModuleType('nemo.collections.speechlm2.parts.alignments')
    alignments.get_word_alignments_for_batch = lambda cuts: [['hello'] for _ in cuts]
    monkeypatch.setitem(sys.modules, alignments.__name__, alignments)
    def forbidden(*args, **kwargs):
        raise AssertionError('Collection attempted CUDA initialization')
    monkeypatch.setattr(torch.cuda, 'set_device', forbidden)
    assert worker(args, plan) == 0
    result = load_json(output/'dai/collect.json')
    assert result['collection_device'] == 'cpu'
    assert result['members_read'] == 1
    assert len(result['candidates']) == len(calls) == 2
    # Simulate interruption before the final report; receipts must avoid reformatting.
    (output/'dai/collect.json').unlink()
    calls.clear()
    assert collect(args, plan) == 0
    assert calls == []
    assert load_json(output/'dai/collect.json')['members_read'] == 0


def test_context_near_full_member_keeps_exact_crop(source):
    from nemo.collections.speechlm2.data.batch_estimation import cuts_for
    _, _, cfg, _, _ = source
    item = OmegaConf.to_container(OmegaConf.load(cfg)[0], resolve=True)
    item['context_sampling'].update(max_leading_s=2.86, max_trailing_s=10., max_duration_s=10.)
    metadata = list(cuts_for(item, 'maximum_leading', 42, True))
    actual = list(cuts_for(item, 'maximum_leading', 42, False))
    for meta, cut in zip(metadata, actual):
        assert meta.duration == pytest.approx(9.86)
        assert cut.duration == pytest.approx(meta.duration, abs=1/16000)
        assert cut.load_audio().shape[-1] == round(9.86*16000)


def test_exact_crop_preserves_audio_offset_within_legacy_codec_tolerance(tmp_path):
    from lhotse import Recording
    from nemo.collections.common.data.lhotse.nemo_adapters import make_cut_with_subset_inmemory_recording
    audio = np.linspace(-.9, .9, 16000, dtype=np.float32)
    path = tmp_path/'ramp.wav'
    sf.write(path, audio, 16000, subtype='FLOAT')
    recording = Recording.from_file(path)
    original = recording.load_audio()
    cut = make_cut_with_subset_inmemory_recording(recording, offset=.14, duration=.86, exact_crop=True)
    assert cut.duration == .86
    np.testing.assert_allclose(cut.load_audio(), original[:, 2240:], atol=4e-5)
    # Existing non-context codec tolerance remains unchanged.
    legacy = make_cut_with_subset_inmemory_recording(recording, offset=0., duration=.86)
    assert legacy.duration == 1.


@pytest.mark.parametrize('with_context', [False, True])
def test_header_only_reader_preserves_tar_order_epochs_and_crops(source, monkeypatch, with_context):
    from lhotse import Recording
    manifest, audio_tar, _, rows, context = source
    # Deliberately disagree with archive order and include an excluded example.
    rows[3]['_skipme'] = True
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in reversed(rows)))
    kwargs = dict(context_sampling=context if with_context else None, shard_seed=17,
                  shuffle_shards=True, slice_length=8)
    real = LazyNeMoTarredIterator(str(manifest), str(audio_tar), **kwargs)
    headers = LazyNeMoTarredIterator(str(manifest), str(audio_tar), audio_metadata_only=True, **kwargs)
    expected = [list(real), list(real)]
    def forbid_audio(*args, **kwargs):
        raise AssertionError('Header-only measurement must never decode audio')
    monkeypatch.setattr(Recording, 'load_audio', forbid_audio)
    actual = [list(headers), list(headers)]
    for audio_epoch, header_epoch in zip(expected, actual):
        assert len(audio_epoch) == len(header_epoch)
        for audio, header in zip(audio_epoch, header_epoch):
            assert (header.id, header.duration, header.start, header.num_samples) == (
                audio.id, audio.duration, audio.start, audio.num_samples)
            assert header.custom == audio.custom
            assert header.recording.sources[0].source == b''


def test_header_only_native_sampler_matches_audio_batches(source, tmp_path):
    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
    from nemo.collections.speechlm2.data.batch_pipeline import MeasureCuts
    manifest, audio_tar, _, rows, context = source
    manifest.write_text(''.join(json.dumps(row)+'\n' for row in reversed(rows)))
    entries = [dict(type='nemo_tarred',manifest_filepath=str(manifest),
                   tarred_audio_filepaths=str(audio_tar),context_sampling=context,
                   weight=weight,tags={'data_seen_source':name})
               for name, weight in [('short',1.),('long',2.)]]
    def batches(headers):
        current=copy.deepcopy(entries)
        for entry in current: entry['tarred_audio_metadata_only']=headers
        cfg=OmegaConf.create(dict(input_cfg=current,num_workers=0,force_iterable_dataset=True,
            shuffle=True,seed=42,shard_seed=42,use_bucketing=True,bucket_duration_bins=[5.,9.],
            bucket_batch_size=[3,1],bucket_buffer_size=20,shuffle_buffer_size=10,
            concurrent_bucketing=False))
        loader=get_lhotse_dataloader_from_config(cfg,global_rank=0,world_size=1,dataset=MeasureCuts())
        import itertools
        return list(itertools.islice(loader,100))
    assert batches(True) == batches(False)


def test_header_calibration_exports_audio_capable_input(source, tmp_path):
    from nemo.collections.speechlm2.data.source_weight_calibration import run
    manifest, audio_tar, _, _, context = source
    inputs=tmp_path/'sources.yaml'
    OmegaConf.save(OmegaConf.create([dict(type='nemo_tarred',manifest_filepath=str(manifest),
        tarred_audio_filepaths=str(audio_tar),context_sampling=context,weight=1.,
        tags={'data_seen_source':'only'})]),inputs)
    config=tmp_path/'training.yaml'
    OmegaConf.save(OmegaConf.create(dict(data=dict(train_ds=dict(input_cfg=str(inputs),num_workers=0,
        shuffle=True,shard_seed=42,seed=42,use_bucketing=True,bucket_duration_bins=[9.],
        bucket_batch_size=[2],bucket_buffer_size=8,shuffle_buffer_size=8,
        concurrent_bucketing=False)))),config)
    result=run(config,{'only':1.},tmp_path/'out',reader='headers',warmup_batches=5,
               measurement_batches=10,verification_batches=10,max_iterations=1)
    assert result['status']=='PASS' and result['reader']=='headers'
    assert result['history'][0]['verification_counts']=={'only':20}
    exported=OmegaConf.load(tmp_path/'out/input.yaml')
    assert 'tarred_audio_metadata_only' not in exported[0]
    assert OmegaConf.load(tmp_path/'out/input_iteration_0.yaml')[0].tarred_audio_metadata_only
