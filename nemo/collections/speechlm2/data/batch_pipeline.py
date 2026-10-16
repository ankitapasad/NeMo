# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resumable manifest-to-batching workflow. No scheduler or private paths.

GPU workers are disposable subprocesses: only explicit CUDA OOM is a search
observation; other failures stop the pipeline. Reports describe bounded coverage,
not a guarantee against OOM on unseen examples or distributed configurations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import Counter

from nemo.collections.speechlm2.data.batch_estimation import duration_geometry, bucket_index, calibrate, stress_batches, cuts_for, sample_identity


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def leaves(entries, tags=None):
    """Flatten groups only for finite source-stratified analysis, never runtime mixing."""
    for entry in entries:
        merged = dict(tags or {}, **entry.get('tags', {}))
        if entry['type'] == 'group':
            yield from leaves(entry['input_cfg'], merged)
        elif entry['type'] in ('nemo', 'nemo_tarred'):
            item = copy.deepcopy(entry)
            item['tags'] = merged
            yield item
        else:
            raise ValueError('Profiling supports native nemo/nemo_tarred inputs; unsupported type: ' + entry['type'])


MODES = ('sampled', 'minimum', 'maximum_leading', 'maximum_trailing')


def read_plan(path):
    from omegaconf import OmegaConf
    path = Path(path).resolve()
    plan = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    def absolute(p):
        return str((path.parent / p).resolve())
    plan['model_config'] = absolute(plan['model_config'])
    if not plan.get('profiles'):
        raise ValueError('profiles must contain at least one shared input_cfg')
    for name, profile in plan['profiles'].items():
        if not name.replace('_', '').isalnum():
            raise ValueError('Profile names must contain only letters, digits and underscores')
        profile['input_cfg'] = absolute(profile['input_cfg'])
    plan.setdefault('seed', 42)
    plan.setdefault('num_buckets', 10)
    plan.setdefault('candidates_per_source_bin', 8)
    plan.setdefault('max_batch_size', 128)
    plan.setdefault('memory_fraction', .80)
    plan.setdefault('measurement_batches', 1000)
    plan.setdefault('updates_per_case', 2)
    for key in ('num_buckets', 'candidates_per_source_bin', 'max_batch_size', 'measurement_batches', 'updates_per_case'):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(key + ' must be a positive integer')
    if not 0 < plan['memory_fraction'] <= 1:
        raise ValueError('memory_fraction must be in (0, 1]')
    if len(plan['profiles']) > 1:
        shares = [float(p['target_share']) for p in plan['profiles'].values()]
        if any(x <= 0 for x in shares) or not math.isclose(sum(shares), 1.):
            raise ValueError('Separate profiles require positive target_share values summing to 1')
    return plan


def bind_inputs(plan, output, *, verify_only=False):
    """Bind resume to config, concrete manifests and implementation content."""
    from omegaconf import OmegaConf
    from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths
    files = [Path(plan['model_config'])]
    model_cfg = OmegaConf.load(plan['model_config']); OmegaConf.resolve(model_cfg)
    if model_cfg.get('init_weights_from_checkpoint'):
        files.append(Path(model_cfg.init_weights_from_checkpoint))
    train = model_cfg.data.train_ds
    # These change duration/memory outside the source reader; reject rather than profile a different pipeline.
    for key in ('pad_extra_duration', 'pad_extra_duration_field', 'perturb_speed', 'noise_path', 'concatenate_samples', 'cut_into_windows_duration', 'truncate_duration', 'rir_enabled', 'use_multimodal_sampling'):
        if train.get(key):
            raise ValueError(f'Profiling does not yet support train_ds.{key}; remove it only if intended for training too')
    for p in plan['profiles'].values():
        files.append(Path(p['input_cfg']))
        for source in leaves(OmegaConf.to_container(OmegaConf.load(p['input_cfg']), resolve=True)):
            for manifest in expand_sharded_filepaths(source['manifest_filepath'], shard_strategy='replicate', world_size=1, global_rank=0):
                files.append(Path(manifest))
    root = Path(__file__).resolve().parents[3]
    files += list((root / 'collections/speechlm2/data').glob('*.py'))
    files += list((root / 'collections/common/data/lhotse').glob('*.py'))
    files += [root / 'collections/speechlm2/models/streaming_stt_model.py']
    files += list((root / 'collections/speechlm2/parts').glob('*.py'))
    files += list((root / 'collections/speechlm2/modules').glob('*.py'))
    binding = dict(plan=plan, files={str(f): digest(f) for f in sorted(set(files))})
    target = output / 'binding.json'
    if target.exists() and load_json(target) != binding:
        raise ValueError('Inputs or code changed: use a fresh output directory; refusing stale resume')
    if verify_only:
        if not target.exists():
            raise ValueError("Run prepare before parallel profiling")
    else:
        write_json(target, binding)


def prepare(plan, output):
    from omegaconf import OmegaConf
    for name, profile in plan['profiles'].items():
        target = output / name / 'geometry.json'
        if target.exists():
            continue
        sources = list(leaves(OmegaConf.to_container(OmegaConf.load(profile['input_cfg']), resolve=True)))
        rows, variants = [], []
        for source_index, source in enumerate(sources):
            for mode in (MODES if source.get('context_sampling') else ('sampled',)):
                seen = set()
                for ordinal, cut in enumerate(cuts_for(source, mode, plan['seed'], True)):
                    ident = sample_identity(cut)
                    if ident in seen:
                        raise ValueError(f'Duplicate example in source {source_index}: {ident}')
                    seen.add(ident)
                    key = f'{source_index}:{ordinal}'
                    bounds = (cut.custom or {}).get('context_duration_bounds', {})
                    if mode == 'sampled':
                        rows.append(dict(id=key, sampled_duration=cut.duration,
                                         min_duration=bounds.get('min_duration', cut.duration),
                                         max_duration=bounds.get('max_duration', cut.duration)))
                    variants.append(dict(id=f'{key}:{mode}', source_index=source_index, ordinal=ordinal,
                                         source_id=str(ident), mode=mode, duration=cut.duration,
                                         text_characters=sum(len(s.text or '') for s in cut.supervisions)))
        geometry = duration_geometry(rows, edges=profile.get('bucket_duration_bins'), num_buckets=plan['num_buckets'])
        edges = geometry['bucket_duration_bins']
        rng = random.Random(plan['seed'])
        pools = {}
        for row in variants:
            row['bucket'] = bucket_index(row['duration'], edges)
            pools.setdefault((row['source_index'], row['bucket']), []).append(row)
        chosen = {}
        for pool in pools.values():
            representative = pool.copy(); rng.shuffle(representative)
            # Broad metadata shortlist; actual formatted lengths rank stress batches on GPU.
            for ranking in (representative, sorted(pool, key=lambda r: -r['duration']),
                            sorted(pool, key=lambda r: -r['text_characters'])):
                for row in ranking[:plan['candidates_per_source_bin']]:
                    chosen[row['id']] = row
        missing = set(range(len(edges))) - {r['bucket'] for r in chosen.values()}
        if missing:
            raise ValueError(f'{name}: no real candidate for bins {sorted(missing)}; supply better fixed edges')
        geometry.update(sources=sources, candidates=list(chosen.values()),
                        coverage='Full metadata scan; bounded source-stratified shortlist, four context modes. Not exhaustive formatted-token extrema.',
                        input_cfg=profile['input_cfg'])
        write_json(target, geometry)


def run_worker(plan_path, output, action, name, bucket=None, size=None):
    suffix = action if bucket is None else f'{action}_{bucket}_{size}'
    report = output / name / (suffix + '.json')
    if report.exists():
        return load_json(report)
    command = [sys.executable, '-m', 'nemo.collections.speechlm2.data.batch_pipeline', '_worker', '--plan', str(plan_path), '--output', str(output),
               '--action', action, '--profile', name]
    if bucket is not None:
        command += ['--bucket', str(bucket), '--batch-size', str(size)]
    start = time.time()
    with (output / name / (suffix + '.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode not in (0, 86) or not report.exists():
        raise RuntimeError(f'{suffix} failed with exit {result.returncode}; inspect {report.with_suffix(".log")}')
    record = load_json(report)
    if (result.returncode == 86) != (record.get('status') == 'OOM'):
        raise RuntimeError('Worker exit/report mismatch')
    record.update(wall_seconds=time.time()-start, command=command)
    write_json(report, record)
    return record


def search(test, maximum, memory_fraction=.80, start_size=1):
    """Use the original OOMptimizer search; impose a caller cap without fake OOMs."""
    from nemo.collections.speechlm2.data.batch_size_search import BatchSizeSearch
    state = BatchSizeSearch(start_batch_size=min(start_size, maximum))
    for _ in range(64):
        size = min(state.current_batch_size, maximum)
        if size < 1:
            raise RuntimeError('Batch size 1 exceeds memory budget')
        result = test(size)
        if result['status'] not in ('PASS', 'OOM'):
            raise RuntimeError('Non-memory profiling failure: ' + str(result))
        if result['status'] == 'PASS' and size == maximum:
            return dict(batch_size=size, capped=True, stopping_reason='caller_cap')
        if result['status'] == 'OOM' and size == 1:
            raise RuntimeError('Batch size 1 exceeds memory budget')
        # If doubling crossed a non-power-of-two cap, this measured cap trial
        # becomes the actual current observation, rather than an invented OOM.
        state._current = size
        state.advance(result['status'] == 'OOM')
        if state.max_batch_size is not None:
            return dict(batch_size=state.max_batch_size, capped=False,
                        stopping_reason='legacy_oomptimizer_gap')
    raise RuntimeError('Search exceeded bounded trial count')


def check_hardware(output, *, create=False):
    import torch
    hardware = dict(gpu_model=torch.cuda.get_device_name(), gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory, torch_version=torch.__version__, cuda_version=torch.version.cuda)
    path = output/'hardware.json'
    if path.exists():
        if load_json(path) != hardware:
            raise ValueError('Profiling hardware/software changed: use matching GPUs or a fresh output directory')
    elif create:
        write_json(path, hardware)
    else:
        raise ValueError('Run collect before profiling bins')


def collect_all(plan_path, plan, output):
    check_hardware(output, create=True)
    tasks = []
    for name in plan['profiles']:
        geometry = load_json(output/name/'geometry.json')
        collected = run_worker(plan_path, output, 'collect', name)
        if collected['status'] != 'PASS':
            raise RuntimeError('Candidate collection failed')
        tasks.extend(dict(profile=name, bucket=i) for i in range(len(geometry['bucket_duration_bins'])))
    write_json(output/'bin_tasks.json', tasks)


def profile_bin(plan_path, plan, output, name, bucket):
    if name not in plan['profiles']:
        raise ValueError('Unknown profile')
    geometry = load_json(output/name/'geometry.json')
    if bucket is None or not 0 <= bucket < len(geometry['bucket_duration_bins']):
        raise ValueError('Invalid bucket index')
    check_hardware(output)
    if load_json(output/name/'collect.json')['status'] != 'PASS':
        raise ValueError('Candidate collection must pass first')
    selected = search(lambda size: run_worker(plan_path, output, 'trial', name, bucket, size), plan['max_batch_size'], plan['memory_fraction'])
    write_json(output/name/f'bin_{bucket}.json', dict(
        **selected, bucket=bucket, collect_sha256=digest(output/name/'collect.json'),
        trial_hashes={f.name:digest(f) for f in (output/name).glob(f'trial_{bucket}_*.json')}))


def validate_all(plan_path, plan, output):
    check_hardware(output)
    for name in plan['profiles']:
        geometry = load_json(output/name/'geometry.json')
        selected, hashes = [], {}
        for bucket in range(len(geometry['bucket_duration_bins'])):
            result = load_json(output/name/f'bin_{bucket}.json')
            if result['bucket'] != bucket or result['collect_sha256'] != digest(output/name/'collect.json'):
                raise ValueError('Stale bin result')
            for filename, expected in result['trial_hashes'].items():
                if digest(output/name/filename) != expected:
                    raise ValueError('GPU trial report changed')
            filename = f"trial_{bucket}_{result['batch_size']}.json"
            if filename not in result['trial_hashes'] or load_json(output/name/filename)['status'] != 'PASS':
                raise ValueError('Bin result lacks a successful selected trial')
            selected.append(result)
            hashes.update(result['trial_hashes'])
        write_json(output/name/'selected.json', dict(bucket_duration_bins=geometry['bucket_duration_bins'],
                   bucket_batch_size=[x['batch_size'] for x in selected], search=selected,
                   trial_hashes=hashes, collect_sha256=digest(output/name/'collect.json')))
        validation = run_worker(plan_path, output, 'validate', name)
        if validation['status'] != 'PASS' or validation['selected_sha256'] != digest(output/name/'selected.json'):
            raise RuntimeError('Combined selected-profile GPU validation failed; inspect validate.log')


def profile_all(plan_path, plan, output):
    collect_all(plan_path, plan, output)
    for task in load_json(output/'bin_tasks.json'):
        profile_bin(plan_path, plan, output, task['profile'], task['bucket'])
    validate_all(plan_path, plan, output)


class MeasureCuts:
    """CPU sampler probe: preserve cut membership without decoding or formatting."""
    def __getitem__(self, cuts):
        return [dict(id=c.id, duration=c.duration, source=(c.custom or {}).get('data_seen_source', 'unlabeled')) for c in cuts]


def finalize(plan, output):
    from omegaconf import OmegaConf, open_dict
    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
    cfg = OmegaConf.load(plan['model_config']); OmegaConf.resolve(cfg)
    train = OmegaConf.to_container(cfg.data.train_ds, resolve=True)
    if train.get('multi_config') or 'datasets' in train:
        raise ValueError('Use a flat train_ds template; declare analysis profiles in the estimation plan')
    multi = len(plan['profiles']) > 1
    if multi:
        from nemo.collections.common.data.lhotse.dataloader import LhotseDataLoadingConfig
        defaults = OmegaConf.structured(LhotseDataLoadingConfig)
        shared_keys = ('seed', 'shard_seed', 'shuffle', 'num_workers', 'pin_memory', 'metadata_only', 'force_finite')
        shared = {key: train.get(key, defaults[key]) for key in shared_keys}
        reserved = set(shared_keys) | {'multi_config', 'sampler_fusion', 'sampler_weights'}
        if reserved.intersection(plan['profiles']):
            raise ValueError('Profile names conflict with native multi_config options')
        train.update(shared)
        # Native fusion requires all sources to use the same iterable mode.
        train['force_iterable_dataset'] = True
    for key in ('input_cfg', 'manifest_filepath', 'tarred_audio_filepaths', 'batch_size', 'batch_duration'):
        train.pop(key, None)
    histograms, reports, children = {}, {}, {}
    for name, profile in plan['profiles'].items():
        selected = load_json(output/name/'selected.json')
        validation = load_json(output/name/'validate.json')
        if validation['status'] != 'PASS' or validation['selected_sha256'] != digest(output/name/'selected.json'):
            raise ValueError('Missing or stale full-profile GPU validation')
        if digest(output/name/'collect.json') != selected['collect_sha256']:
            raise ValueError('Candidate report changed')
        for filename, expected in selected['trial_hashes'].items():
            if digest(output/name/filename) != expected:
                raise ValueError('GPU trial report changed')
        for i, size in enumerate(selected['bucket_batch_size']):
            result = load_json(output/name/f'trial_{i}_{size}.json')
            if result['status'] != 'PASS':
                raise ValueError('Selected size lacks successful real-batch trial')
        child = dict(train, input_cfg=profile['input_cfg'], use_bucketing=True,
                     bucket_duration_bins=selected['bucket_duration_bins'], bucket_batch_size=selected['bucket_batch_size'])
        loader = get_lhotse_dataloader_from_config(OmegaConf.create(child), global_rank=0, world_size=1, dataset=MeasureCuts())
        sizes, source_counts = Counter(), Counter()
        iterator = iter(loader)
        exhausted = False
        for _ in range(plan['measurement_batches']):
            try:
                batch = next(iterator)
            except StopIteration:
                exhausted = True
                break
            if not batch:
                raise ValueError('Empty sampler batch')
            indices = {bucket_index(row['duration'], selected['bucket_duration_bins']) for row in batch}
            if len(indices) != 1 or len(batch) > selected['bucket_batch_size'][next(iter(indices))]:
                raise ValueError('Sampler bucket/batch contract differs from profiled configuration')
            sizes[len(batch)] += 1
            source_counts.update(row['source'] for row in batch)
        if not sizes:
            raise ValueError(f'Profile {name!r}: sampler emitted no batches; check manifests and filters')
        histograms[name] = dict(sizes)
        reports[name] = dict(batch_size_histogram=dict(sizes), examples_by_source=dict(source_counts),
                             measured_batches=sum(sizes.values()), requested_batches=plan['measurement_batches'],
                             sampler_exhausted=exhausted,
                             world_size=1, num_workers=child.get('num_workers', 0))
        children[name] = child
        del iterator, loader
    calibration = None
    if len(children) == 1:
        cfg.data.train_ds = OmegaConf.create(next(iter(children.values())))
    else:
        calibration = calibrate({k:p['target_share'] for k,p in plan['profiles'].items()}, histograms, seed=plan['seed'], draws=plan.get('calibration_simulation_draws'))
        cfg.data.train_ds = OmegaConf.create(dict(
            **shared, multi_config=True, sampler_fusion='randomized_round_robin',
            sampler_weights=calibration['selection_weights'], **children))
    OmegaConf.save(cfg, output/'training.yaml')
    write_json(output/'REPORT.json', dict(status='BOUNDED_SINGLE_GPU_VALIDATION_PASSED', profiles=reports,
               calibration=calibration, memory_fraction=plan['memory_fraction'],
               limitations=['Bounded real candidate set, not exhaustive token extrema or all stochastic crops',
                            'Sampler measurement uses one rank; verify production distributed mixture and resume separately',
                            'GPU trials test source-specific and mixed stress batches; finite samples cannot guarantee no future OOM'],
               training_yaml_sha256=digest(output/'training.yaml')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'collect', 'profile-bin', 'validate', 'profile', 'finalize', 'run', '_worker'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--action', choices=['collect', 'trial', 'validate'])
    parser.add_argument('--profile')
    parser.add_argument('--bucket', type=int)
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args(argv)
    args.plan = args.plan.resolve(); args.output = args.output.resolve()
    plan = read_plan(args.plan)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == '_worker':
        from nemo.collections.speechlm2.data.batch_profiling import worker
        return worker(args, plan)
    # Bin processes share the pipeline lock, but exclusively lock their own bin.
    # Collection, validation and finalization cannot overlap any bin process.
    import fcntl
    with (args.output/'pipeline.lock').open('a') as lock:
        parallel = args.stage == 'profile-bin'
        fcntl.flock(lock, (fcntl.LOCK_SH if parallel else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        bind_inputs(plan, args.output, verify_only=parallel)
        if parallel:
            if args.profile not in plan['profiles'] or args.bucket is None or args.bucket < 0:
                raise ValueError('profile-bin requires a valid --profile and --bucket')
            with (args.output/args.profile/f'bin_{args.bucket}.lock').open('a') as bin_lock:
                fcntl.flock(bin_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                profile_bin(args.plan, plan, args.output, args.profile, args.bucket)
            return 0
        if args.stage == 'collect':
            collect_all(args.plan, plan, args.output)
        if args.stage == 'validate':
            validate_all(args.plan, plan, args.output)
        if args.stage in ('prepare', 'run'):
            prepare(plan, args.output)
        if args.stage in ('profile', 'run'):
            profile_all(args.plan, plan, args.output)
        if args.stage in ('finalize', 'run'):
            finalize(plan, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
