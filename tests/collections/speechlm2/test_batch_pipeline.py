"""Search parity, source grouping, OOM classification and native manifest connector."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


search_module = load('search_core', ROOT/'nemo/collections/speechlm2/data/batch_size_search.py')


@pytest.fixture
def pipeline(monkeypatch):
    core = load('estimation_core', ROOT/'nemo/collections/speechlm2/data/batch_estimation.py')
    monkeypatch.setitem(sys.modules, 'nemo.collections.speechlm2.data.batch_estimation', core)
    monkeypatch.setitem(sys.modules, 'nemo.collections.speechlm2.data.batch_size_search', search_module)
    return load('pipeline_under_test', ROOT/'nemo/collections/speechlm2/data/batch_pipeline.py')


@pytest.mark.parametrize('capacity', [1, 2, 3, 7, 31, 32, 33, 95, 100, 257])
def test_legacy_search_exact_trace(capacity):
    import json
    original = json.loads((Path(__file__).parent/'data/batch_size_search_legacy_traces.json').read_text())['traces'][str(capacity)]
    new = search_module.BatchSizeSearch()
    for size in original['trials']:
        assert new.current_batch_size == size
        assert new.max_batch_size is None
        new.advance(size > capacity)
    assert new.max_batch_size == original['result']


def test_groups_are_generic_and_preserve_labels(pipeline):
    items=list(pipeline.leaves([dict(type='group',tags={'data_seen_source':'alpha'},input_cfg=[
        dict(type='nemo',manifest_filepath='one'),dict(type='group',tags={'data_seen_category':'beta'},input_cfg=[
            dict(type='nemo_tarred',manifest_filepath='two',tags={'data_seen_source':'gamma'})])])]))
    assert [x['tags'] for x in items] == [{'data_seen_source':'alpha'}, {'data_seen_source':'gamma','data_seen_category':'beta'}]


def test_cap_is_not_reported_as_maximum(pipeline):
    calls=[]
    def test(n):
        calls.append(n);return {'status':'PASS'}
    result=pipeline.search(test, 7)
    assert calls==[1,2,4,7]
    assert result['batch_size']==7 and result['capped']


def test_non_oom_failure_stops(pipeline):
    with pytest.raises(RuntimeError, match='Non-memory'):
        pipeline.search(lambda n: {'status':'DATA_ERROR'}, 8)
    with pytest.raises(RuntimeError, match='Batch size 1'):
        pipeline.search(lambda n: {'status':'OOM'}, 8)


@pytest.mark.parametrize('mode', ['minimum','maximum_leading','maximum_trailing'])
def test_context_corners_respect_budget(pipeline, mode):
    from nemo.collections.common.data.lhotse.nemo_adapters import context_corner_pair
    lead,trail=context_corner_pair(mode,leading_low=.05,leading_high=5.,trailing_low=.4,trailing_high=4.,total_budget=6.)
    assert .05<=lead<=5 and .4<=trail<=4 and lead+trail<=6
    if mode != 'minimum': assert lead+trail==pytest.approx(6.)


def test_prepare_nested_real_native_manifests(tmp_path, pipeline):
    # Native integration test; no fake duration estimator or source-name catalog.
    import json
    import numpy as np
    import soundfile as sf
    from omegaconf import OmegaConf
    source_entries=[]
    for name in ['arbitrary_a','arbitrary_b']:
        manifest=tmp_path/(name+'.jsonl')
        rows=[]
        for i in range(3):
            path=tmp_path/f'{name}_{i}.wav';sf.write(path,np.zeros(16000*(i+1)),16000)
            rows.append(dict(audio_filepath=str(path),duration=i+1,text='hello',sample_id=str(i)))
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        source_entries.append(dict(type='group',tags={'data_seen_source':name},input_cfg=[dict(type='nemo',manifest_filepath=str(manifest))]))
    inputs=tmp_path/'input.yaml';OmegaConf.save(OmegaConf.create(source_entries),inputs)
    output=tmp_path/'out'
    plan=dict(profiles={'shared':{'input_cfg':str(inputs)}},seed=42,num_buckets=2,candidates_per_source_bin=1)
    pipeline.prepare(plan,output)
    geometry=pipeline.load_json(output/'shared/geometry.json')
    assert geometry['examples']==6
    assert {c['source_index'] for c in geometry['candidates']}=={0,1}
    assert geometry['requested_num_buckets']==2
    assert geometry['bucket_duration_bins'][-1]==3.
    assert geometry['bucket_duration_bins']==sorted(set(geometry['bucket_duration_bins']))


@pytest.mark.parametrize('draws', [None, 1])
def test_finalize_native_sampler_and_calibration(tmp_path, pipeline, draws):
    import os
    if os.environ.get('NEMO_NATIVE_TEST') != '1':
        pytest.skip('Full native sampler integration runs in the NeMo container')
    import json
    from omegaconf import OmegaConf
    # These are CPU sampler fixtures, explicitly not GPU-validation evidence.
    manifest=tmp_path/'metadata.jsonl'
    import numpy as np
    import soundfile as sf
    audio=tmp_path/'audio.wav';sf.write(audio,np.zeros(48000),16000)
    manifest.write_text(''.join(json.dumps(dict(audio_filepath=str(audio),duration=d,text='word',sample_id=str(i)))+'\n' for i,d in enumerate([1.,2.,1.5,2.5])))
    inp=tmp_path/'input.yaml'
    OmegaConf.save(OmegaConf.create([dict(type='nemo',manifest_filepath=str(manifest),tags=dict(data_seen_source='arbitrary'))]),inp)
    cfg=tmp_path/'model.yaml'
    OmegaConf.save(OmegaConf.create(dict(data=dict(train_ds=dict(num_workers=0,shuffle=False,concurrent_bucketing=False,bucket_buffer_size=10,shard_seed=42)),trainer=dict(accumulate_grad_batches=1))),cfg)
    plan=dict(model_config=str(cfg),profiles={name:dict(input_cfg=str(inp),target_share=share) for name,share in [('first',.8),('second',.2)]},measurement_batches=20,memory_fraction=.8,seed=42)
    plan['calibration_simulation_draws'] = draws
    out=tmp_path/'run'
    for name,size in [('first',2),('second',1)]:
        folder=out/name;folder.mkdir(parents=True)
        pipeline.write_json(folder/'collect.json',dict(status='PASS'))
        pipeline.write_json(folder/f'trial_0_{size}.json',dict(status='PASS'))
        selected=dict(bucket_duration_bins=[3.],bucket_batch_size=[size],trial_hashes={f'trial_0_{size}.json':pipeline.digest(folder/f'trial_0_{size}.json')},collect_sha256=pipeline.digest(folder/'collect.json'))
        pipeline.write_json(folder/'selected.json',selected)
        pipeline.write_json(folder/'validate.json',dict(status='PASS',selected_sha256=pipeline.digest(folder/'selected.json')))
    pipeline.finalize(plan,out)
    report=pipeline.load_json(out/'REPORT.json')
    assert report['profiles']['first']['measured_batches'] == 2
    assert report['profiles']['second']['measured_batches'] == 4
    assert all(p['sampler_exhausted'] for p in report['profiles'].values())
    assert report['calibration']['simulation_passed'] is (None if draws is None else False)
    assert report['calibration']['mean_emitted_batch_sizes']=={'first':2.,'second':1.}
    assert report['calibration']['selection_probabilities']['first']==pytest.approx(2/3)
    output=OmegaConf.load(out/'training.yaml')
    assert output.data.train_ds.multi_config
    assert output.data.train_ds.sampler_fusion == 'randomized_round_robin'
    assert set(output.data.train_ds.sampler_weights) == {'first', 'second'}
    assert output.data.train_ds.seed == 0  # Native default, not estimation seed 42.
    assert 'batch_mux_seed' not in output.data.train_ds
    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
    loader = get_lhotse_dataloader_from_config(output.data.train_ds, 0, 1, pipeline.MeasureCuts())
    batches = list(loader)
    assert sorted(map(len, batches)) == [1, 1, 1, 1, 2, 2]
    assert sum(map(len, batches)) == 8
    # Tampered GPU evidence must not be accepted on finalize/resume.
    pipeline.write_json(out/'first/trial_0_2.json',dict(status='OOM'))
    with pytest.raises(ValueError,match='trial report changed'):
        pipeline.finalize(plan,out)


def test_parallel_bins_match_serial_and_require_complete_evidence(tmp_path, pipeline, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    plan = dict(profiles={'generic': {}}, max_batch_size=4, memory_fraction=.8)
    monkeypatch.setattr(pipeline, 'check_hardware', lambda *a, **kw: None)
    def worker(plan_path, output, action, name, bucket=None, size=None):
        folder = output/name
        if action == 'trial':
            result = dict(status='PASS' if size <= [2, 4][bucket] else 'OOM')
            pipeline.write_json(folder/f'trial_{bucket}_{size}.json', result)
            return result
        assert action == 'validate'
        return dict(status='PASS', selected_sha256=pipeline.digest(folder/'selected.json'))
    monkeypatch.setattr(pipeline, 'run_worker', worker)
    for mode in ('serial', 'parallel'):
        output=tmp_path/mode
        pipeline.write_json(output/'generic/geometry.json', dict(bucket_duration_bins=[5., 10.]))
        pipeline.write_json(output/'generic/collect.json', dict(status='PASS'))
        if mode == 'serial':
            for bucket in range(2):
                pipeline.profile_bin(None, plan, output, 'generic', bucket)
        else:
            with ThreadPoolExecutor(2) as pool:
                list(pool.map(lambda b: pipeline.profile_bin(None, plan, output, 'generic', b), range(2)))
        pipeline.validate_all(None, plan, output)
    serial=pipeline.load_json(tmp_path/'serial/generic/selected.json')
    assert serial == pipeline.load_json(tmp_path/'parallel/generic/selected.json')
    assert serial['bucket_batch_size'] == [2, 4]
    folder=tmp_path/'parallel/generic'
    pipeline.write_json(folder/'trial_0_2.json', dict(status='OOM'))
    with pytest.raises(ValueError, match='report changed'):
        pipeline.validate_all(None, plan, tmp_path/'parallel')
    (folder/'bin_0.json').unlink()
    with pytest.raises(FileNotFoundError):
        pipeline.validate_all(None, plan, tmp_path/'parallel')


def test_bin_locks_allow_other_bins_but_block_duplicate_and_finalize(tmp_path, pipeline, monkeypatch):
    import fcntl
    output=tmp_path/'run';(output/'generic').mkdir(parents=True)
    monkeypatch.setattr(pipeline, 'read_plan', lambda _: dict(profiles={'generic': {}}))
    monkeypatch.setattr(pipeline, 'bind_inputs', lambda *a, **kw: None)
    calls=[]
    monkeypatch.setattr(pipeline, 'profile_bin', lambda *a: calls.append(a[-1]))
    args=['--plan', str(tmp_path/'plan'), '--output', str(output)]
    with (output/'pipeline.lock').open('a') as shared, (output/'generic/bin_0.lock').open('a') as own:
        fcntl.flock(shared, fcntl.LOCK_SH|fcntl.LOCK_NB)
        fcntl.flock(own, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            pipeline.main(['profile-bin', *args, '--profile', 'generic', '--bucket', '0'])
        pipeline.main(['profile-bin', *args, '--profile', 'generic', '--bucket', '1'])
        with pytest.raises(BlockingIOError):
            pipeline.main(['finalize', *args])
    assert calls == [1]


def test_search_copy_matches_untouched_legacy_methods():
    legacy = ast.parse((ROOT/'scripts/speechlm2/oomptimizer.py').read_text())
    copied = ast.parse((ROOT/'nemo/collections/speechlm2/data/batch_size_search.py').read_text())
    def methods(tree, name):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
        return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    old, new = methods(legacy, 'ProfilingBatchGenerator'), methods(copied, 'BatchSizeSearch')
    for name in ('max_batch_size', 'current_rel_gap', 'reset', 'advance'):
        assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False)
