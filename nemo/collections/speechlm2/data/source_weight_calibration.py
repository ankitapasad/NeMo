# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrate independently weighted sources inside one native bucketed loader.

Unlike whole-batch calibration, this measures emitted source counts and adjusts
input source weights. Duration bins and batch sizes remain fixed. Uses tar audio headers without waveform decoding, while preserving native
stream repetition. The older metadata_only loader mode changes those semantics.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path


def shares(values, label):
    if not values or any(not math.isfinite(float(v)) or float(v) <= 0 for v in values.values()):
        raise ValueError(f"{label} must contain finite positive values")
    total = sum(values.values())
    return {k: float(v) / total for k, v in values.items()}


def adjusted_weights(weights, targets, counts, damping=0.5):
    if set(weights) != set(targets) or set(counts) != set(targets):
        raise ValueError("Observed sources must exactly match targets; missing/unknown source")
    if not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")
    observed = shares(counts, "observed counts")
    targets = shares(targets, "targets")
    return shares({k: weights[k] * (targets[k] / observed[k]) ** damping for k in targets}, "weights")


def within_tolerance(counts, targets, tolerance_pp):
    if set(counts) != set(targets) or any(v <= 0 for v in counts.values()):
        return False
    actual = shares(counts, "counts")
    return max(abs(actual[k] - targets[k]) * 100 for k in targets) <= tolerance_pp


def source_entries(entries):
    # Weight the explicit top-level streams, preserving their internal recipe.
    result = {}
    for entry in entries:
        name = entry.get('tags', {}).get('data_seen_source')
        if not name or name in result:
            raise ValueError("Each top-level input_cfg entry needs a unique tags.data_seen_source")
        result[name] = entry
    return result


def calibrate(measure, initial, targets, *, max_iterations=3, tolerance_pp=1.0, damping=0.5, record=lambda value: None):
    weights, targets = shares(initial, 'weights'), shares(targets, 'targets')
    if set(weights) != set(targets):
        raise ValueError('Source names and target names must match exactly')
    history = []
    for iteration in range(max_iterations):
        # measure returns consecutive, disjoint post-warmup windows from one loader.
        calibration, verification = measure(weights, iteration)
        row = dict(iteration=iteration, weights=weights.copy(), calibration_counts=calibration,
                   verification_counts=verification)
        row['passed'] = (within_tolerance(calibration, targets, tolerance_pp) and
                         within_tolerance(verification, targets, tolerance_pp))
        history.append(row); record(row)
        if row['passed']:
            return dict(status='PASS', weights=weights, targets=targets, history=history,
                        tolerance_pp=tolerance_pp, verification_scope='Separate consecutive post-warmup window, one rank')
        if iteration + 1 < max_iterations:
            weights = adjusted_weights(weights, targets, calibration, damping)
    return dict(status='NOT_CONVERGED', weights=weights, targets=targets, history=history,
                tolerance_pp=tolerance_pp)


def run(config_path, targets, output, *, warmup_batches=1000, measurement_batches=2000,
        verification_batches=2000, max_iterations=3, tolerance_pp=1.0, damping=0.5, reader="headers"):
    started = time.monotonic()
    from omegaconf import OmegaConf
    from nemo.collections.common.data.lhotse.dataloader import get_lhotse_dataloader_from_config
    from nemo.collections.speechlm2.data.batch_pipeline import MeasureCuts, write_json
    if any(type(n) is not int or n < 1 for n in (warmup_batches, measurement_batches, verification_batches, max_iterations)):
        raise ValueError('Window lengths and max_iterations must be positive integers')
    if not math.isfinite(tolerance_pp) or tolerance_pp <= 0:
        raise ValueError('tolerance_pp must be finite and positive')
    if reader != 'headers':
        raise ValueError('Source calibration uses reader=headers; real audio remains in training/profiling')
    targets = shares(targets, 'targets')
    cfg = OmegaConf.load(config_path); OmegaConf.resolve(cfg)
    train = OmegaConf.to_container(cfg.data.train_ds, resolve=True)
    if train.get('multi_config') or train.get('metadata_only') or train.get('force_finite'):
        raise ValueError('Requires one shared, repeating native loader; metadata-only/finite/multi_config unsupported')
    if train.get('num_workers', 0) != 0:
        raise ValueError('Source calibration currently requires num_workers=0; report this single-rank scope')
    if not train.get('bucket_batch_size') or not train.get('bucket_duration_bins'):
        raise ValueError('Requires validated fixed bucket_duration_bins and bucket_batch_size')
    source_path = Path(train['input_cfg'])
    entries = OmegaConf.to_container(OmegaConf.load(source_path), resolve=True)
    sources = source_entries(entries)
    if set(sources) != set(targets):
        raise ValueError('Target names must match all top-level tags.data_seen_source values')
    for entry in entries:
        if not isinstance(entry.get('weight'), (float, int)):
            raise ValueError('Each source needs an explicit numeric weight')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    fingerprint = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    write_json(output/'INPUTS.json', dict(config=str(config_path),config_sha256=fingerprint(config_path),
               input_cfg=str(source_path),input_sha256=fingerprint(source_path), code_sha256=fingerprint(__file__),
               warmup_batches=warmup_batches,measurement_batches=measurement_batches,
               verification_batches=verification_batches,max_iterations=max_iterations,
               targets=targets,reader=reader,scope='Native reader and sampler, one rank, zero workers, no model'))

    measurements = []

    def measure(weights, iteration):
        current = copy.deepcopy(entries)
        for name, entry in source_entries(current).items():
            entry['weight'] = weights[name]
        if reader == 'headers':
            def set_headers(items):
                for entry in items:
                    if entry.get('type') == 'group':
                        set_headers(entry['input_cfg'])
                    elif entry.get('type') in ('nemo', 'nemo_tarred') and entry.get('tarred_audio_filepaths'):
                        entry['tarred_audio_metadata_only'] = True
                    else:
                        raise ValueError('Header-only calibration supports NeMo tarred leaves only')
            set_headers(current)
        path = output/f'input_iteration_{iteration}.yaml'
        OmegaConf.save(OmegaConf.create(current), path)
        child = dict(train, input_cfg=str(path))
        loader = get_lhotse_dataloader_from_config(OmegaConf.create(child), global_rank=0, world_size=1, dataset=MeasureCuts())
        iterator = iter(loader)
        windows = []
        for label, n in [('warmup', warmup_batches), ('calibration', measurement_batches), ('verification', verification_batches)]:
            counts = Counter()
            duration_s = Counter()
            for index in range(n):
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise ValueError('Loader exhausted before completing measurement; no weights exported') from error
                if not batch:
                    raise ValueError('Empty batch in mixture measurement')
                counts.update(row['source'] for row in batch)
                for row in batch:
                    seconds = float(row['duration'])
                    if not math.isfinite(seconds) or seconds <= 0:
                        raise ValueError('Sampled crop duration must be finite and positive')
                    duration_s[row['source']] += seconds
                if set(counts) - set(targets):
                    raise ValueError('Unknown emitted data_seen_source: '+str(set(counts)-set(targets)))
                if (index+1) % 100 == 0 or index+1 == n:
                    progress=dict(iteration=iteration,window=label,batches=index+1,counts=dict(counts))
                    write_json(output/'progress.json',progress)
                    print(json.dumps(progress),flush=True)
            measurements.append(dict(iteration=iteration, window=label, batches=n,
                counts=dict(counts), duration_seconds=dict(duration_s),
                sample_shares=shares(counts, 'counts'), hour_shares=shares(duration_s, 'durations'),
                hours={key: value / 3600 for key, value in duration_s.items()}))
            write_json(output/'measurements.json', measurements)
            windows.append(dict(counts))
        del iterator, loader
        return windows[1], windows[2]

    result = calibrate(measure, {k:e['weight'] for k,e in sources.items()}, targets,
                       max_iterations=max_iterations,tolerance_pp=tolerance_pp,damping=damping,
                       record=lambda row: write_json(output/f"iteration_{row['iteration']}.json",row))
    if result['status'] == 'PASS':
        final = copy.deepcopy(entries)
        for name, entry in source_entries(final).items(): entry['weight'] = result['weights'][name]
        OmegaConf.save(OmegaConf.create(final),output/'input.yaml')
        cfg.data.train_ds.input_cfg = str(output/'input.yaml')
        OmegaConf.save(cfg,output/'training.yaml')
        result['training_yaml_sha256']=fingerprint(output/'training.yaml')
        result['input_yaml_sha256']=fingerprint(output/'input.yaml')
    result['measurements'] = measurements
    result['hours_definition'] = 'Sum of emitted sampled crop durations including leading/trailing context; excludes batch padding; repeated draws count again'
    result['reader'] = reader
    result['elapsed_seconds'] = time.monotonic() - started
    write_json(output/'REPORT.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader',choices=['headers'],default='headers',
                        help='headers preserves tar streams but reads audio headers only (uncompressed seekable tars)')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--targets',type=Path,required=True,help='JSON object: data_seen_source to target share')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warmup-batches',type=int,default=1000)
    parser.add_argument('--measurement-batches',type=int,default=2000)
    parser.add_argument('--verification-batches',type=int,default=2000)
    parser.add_argument('--max-iterations',type=int,default=3)
    parser.add_argument('--tolerance-pp',type=float,default=1.0)
    args=parser.parse_args()
    result=run(args.config.resolve(),json.loads(args.targets.read_text()),args.output,
               warmup_batches=args.warmup_batches,measurement_batches=args.measurement_batches,
               verification_batches=args.verification_batches,max_iterations=args.max_iterations,
               tolerance_pp=args.tolerance_pp,reader=args.reader)
    return 0 if result['status']=='PASS' else 2
