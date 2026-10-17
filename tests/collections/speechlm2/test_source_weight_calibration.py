"""Shared-loader calibration adjusts source weights, not batch-selection weights."""
import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[3]/'nemo/collections/speechlm2/data/source_weight_calibration.py'
spec=importlib.util.spec_from_file_location('calibration',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_recovers_known_source_acceptance_bias():
    def measure(weights, iteration):
        counts={k:round(100000*weights[k]*factor) for k,factor in [('a',1),('b',.5)]}
        return counts,counts.copy()
    result=m.calibrate(measure,{'a':1,'b':1},{'a':.5,'b':.5},max_iterations=3,damping=1)
    assert result['status']=='PASS'
    assert result['weights']['b']/result['weights']['a']==pytest.approx(2,rel=1e-3)


def test_verification_must_pass_and_weights_stay_if_already_balanced():
    result=m.calibrate(lambda w,i:({'a':100,'b':100},{'a':100,'b':100}),{'a':1,'b':1},{'a':.5,'b':.5})
    assert result['status']=='PASS' and len(result['history'])==1
    result=m.calibrate(lambda w,i:({'a':100,'b':100},{'a':60,'b':140}),{'a':1,'b':1},{'a':.5,'b':.5},max_iterations=1)
    assert result['status']=='NOT_CONVERGED'


def test_missing_unknown_and_nonfinite_sources_fail():
    for counts in ({'a':10},{'a':10,'c':10},{'a':10,'b':0}):
        with pytest.raises(ValueError):m.adjusted_weights({'a':1,'b':1},{'a':.5,'b':.5},counts)
    with pytest.raises(ValueError):m.shares({'a':float('nan')},'targets')


def test_source_groups_remain_intact():
    child={'type':'nemo','manifest_filepath':'unchanged'}
    entries=[{'type':'group','tags':{'data_seen_source':'a'},'input_cfg':[child],'weight':1}]
    selected=m.source_entries(entries);selected['a']['weight']=2
    assert entries[0]['input_cfg']==[child]
    with pytest.raises(ValueError):m.source_entries(entries*2)


@pytest.mark.parametrize("targets, expected", [({"alpha": .5, "beta": .5}, "PASS"), ({"alpha": .99, "beta": .01}, "NOT_CONVERGED")])
def test_native_shared_loader_exports_unchanged_bins(tmp_path, targets, expected):
    import os
    if os.environ.get("NEMO_NATIVE_TEST") != "1":
        pytest.skip("Full native loader imports require the NeMo container")
    import json
    import numpy as np
    import soundfile as sf
    from omegaconf import OmegaConf
    audio=tmp_path/'clip.wav';sf.write(audio,np.zeros(1600),16000)
    import tarfile
    archive=tmp_path/'audio_0.tar'
    with tarfile.open(archive, 'w') as tar: tar.add(audio, arcname='clip.wav')
    sources=[]
    for name in ['alpha','beta']:
        manifest=tmp_path/f'{name}_manifest_0.jsonl'
        manifest.write_text(''.join(json.dumps(dict(audio_filepath='clip.wav',shard_id=0,duration=.1,text='hi',sample_id=f'{name}{i}'))+'\n' for i in range(100)))
        sources.append(dict(type='nemo_tarred',manifest_filepath=str(manifest),tarred_audio_filepaths=str(archive),weight=1.,tags={'data_seen_source':name}))
    inputs=tmp_path/'input.yaml';OmegaConf.save(OmegaConf.create(sources),inputs)
    config=tmp_path/'training.yaml'
    train=dict(input_cfg=str(inputs),num_workers=0,force_iterable_dataset=True,shuffle=True,
               shard_seed=42,seed=42,use_bucketing=True,bucket_duration_bins=[.2],bucket_batch_size=[4],
               bucket_buffer_size=16,shuffle_buffer_size=16,concurrent_bucketing=False)
    OmegaConf.save(OmegaConf.create(dict(model={'llm_gradient_checkpointing':True},data={'train_ds':train})),config)
    original=config.read_bytes()
    result=m.run(config,targets,tmp_path/'out',warmup_batches=20,
                 measurement_batches=100,verification_batches=100,max_iterations=1,tolerance_pp=10)
    assert result['reader']=='headers'
    measured=result['measurements'][-1]
    assert sum(measured['duration_seconds'].values()) == pytest.approx(40.)
    assert measured['hour_shares'] == pytest.approx(measured['sample_shares'])
    assert result['status']==expected
    if expected != 'PASS':
        assert not (tmp_path/'out/training.yaml').exists()
        assert not (tmp_path/'out/input.yaml').exists()
        assert config.read_bytes()==original
        return
    assert sum(result['history'][0]['calibration_counts'].values())==400
    assert sum(result['history'][0]['verification_counts'].values())==400
    exported=OmegaConf.load(tmp_path/'out/training.yaml')
    assert exported.data.train_ds.bucket_batch_size==[4]
    assert exported.data.train_ds.bucket_duration_bins==[.2]
    assert exported.model.llm_gradient_checkpointing is True
    assert config.read_bytes()==original


def test_audio_calibration_option_removed(tmp_path):
    import os
    if os.environ.get('NEMO_NATIVE_TEST') != '1':
        pytest.skip('Native imports require NeMo environment')
    with pytest.raises(ValueError, match='reader=headers'):
        m.run(tmp_path/'unused.yaml', {'only':1.}, tmp_path/'out', reader='audio')
