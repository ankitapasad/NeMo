# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata helpers around Lhotse duration estimation and TTM mixture calibration.

No audio decoding or training happens here. Crop metadata must come from the
same native transform as training. Automatic bin estimation requires Lhotse;
fixed-bound checks, stress selection and calibration use only the standard library.
"""
from __future__ import annotations

import bisect
import math
import random
from collections import Counter
from numbers import Real

DEFAULT_MEMORY_FRACTION = 0.80


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def positive_integer(value, name):
    number = positive(value, name)
    if int(number) != number:
        raise ValueError(f"{name} must be an integer")
    return int(number)


def validate_edges(edges):
    values = [positive(x, 'duration edge') for x in edges]
    if not values or any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError('duration edges must be nonempty and strictly increasing')
    return values


def bucket_index(duration, edges):
    edges = validate_edges(edges)
    index = bisect.bisect_left(edges, positive(duration, 'crop duration'))
    if index == len(edges):
        raise ValueError(f'crop duration {duration} exceeds supported tail {edges[-1]}')
    return index


def duration_geometry(rows, edges=None, num_buckets=10):
    """Validate crop geometry and wrap NeMo's existing Lhotse bin estimator.

    Durations reflect native source crops. Context-envelope bounds are inputs,
    not reconstructed by adding silence to the original manifest duration.
    Lhotse estimates internal boundaries by approximately equal total duration.
    Preserve its raw result separately; deduplicate boundaries and append the
    feasible maximum to obtain finite upper bounds for fixed-size batch profiles.
    """
    num_buckets = positive_integer(num_buckets, 'num_buckets')
    checked = []
    ids = set()
    for row in rows:
        ident = row['id']
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError('empty or duplicate sample identity')
        ids.add(ident)
        low, sampled, high = [positive(row[k], k) for k in ('min_duration', 'sampled_duration', 'max_duration')]
        if not low <= sampled <= high:
            raise ValueError(f'invalid native crop interval for {ident}')
        checked.append((low, sampled, high, positive(row.get('weight', 1), 'weight')))
    if not checked:
        raise ValueError('no eligible examples')
    if len({row[3] for row in checked}) != 1:
        raise ValueError('Lhotse duration estimation requires equally weighted rows; supply a representative unweighted sample')
    supported_max = max(x[2] for x in checked)
    estimated_bins = None
    estimator_version = None
    if edges is None:
        if not 1 < num_buckets <= len(checked):
            raise ValueError('automatic estimation requires 2 <= num_buckets <= number of examples')
        import lhotse
        from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets

        cuts = (lhotse.MonoCut(id=str(i), start=0., duration=row[1], channel=0)
                for i, row in enumerate(checked))
        estimated_bins = [float(x) for x in estimate_duration_buckets(cuts, num_buckets=num_buckets)]
        estimator_version = lhotse.__version__
        edges = sorted(set(estimated_bins + [supported_max]))
        method = 'Lhotse equal-total-duration estimator; finite context-envelope upper bound appended'
    else:
        method = 'fixed supported crop-duration boundaries'
    edges = validate_edges(edges)
    if supported_max > edges[-1]:
        raise ValueError('fixed edges do not cover the full supported context envelope')
    occupancy = Counter(bucket_index(x[1], edges) for x in checked)
    lower = 0.0
    buckets = []
    for i, upper in enumerate(edges):
        reachable = sum(low <= upper and high > lower for low, _, high, _ in checked)
        buckets.append(dict(index=i, upper=upper, sampled_examples=occupancy[i], reachable_examples=reachable))
        lower = upper
    return dict(bucket_duration_bins=edges, buckets=buckets, examples=len(checked), method=method,
                estimator_duration_bins=estimated_bins, lhotse_version=estimator_version,
                requested_num_buckets=num_buckets if estimated_bins is not None else None,
                num_buckets=len(edges), supported_max_duration=supported_max,
                boundary_convention='inclusive finite upper bounds for fixed batch sizes',
                token_filtering=False, full_supported_duration_tail=True)


def calibrate(targets, histograms, draws=None, seed=42, tolerance_pp=0.25):
    """Convert desired example proportions to relative batch-selection weights.

    Optional simulation uses Python Random selection and an independent size RNG.
    It is diagnostic only; native runtime behavior must be checked separately.
    """
    if draws is not None:
        draws = positive_integer(draws, 'draws')
    tolerance_pp = positive(tolerance_pp, 'tolerance_pp')
    if set(targets) != set(histograms) or not targets:
        raise ValueError('target and histogram streams must match')
    targets = {k: positive(v, 'target share') for k, v in targets.items()}
    if not math.isclose(math.fsum(targets.values()), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError('target example shares must sum to 1')
    sizes, counts, means = {}, {}, {}
    for name, hist in histograms.items():
        normalized = {}
        for key, value in hist.items():
            k = positive_integer(float(key), 'batch size') if isinstance(key, str) else positive_integer(key, 'batch size')
            if k in normalized:
                raise ValueError('duplicate batch-size key after normalization')
            normalized[k] = positive_integer(value, 'batch frequency')
        if not normalized:
            raise ValueError('empty histogram')
        sizes[name], counts[name] = list(normalized), list(normalized.values())
        means[name] = sum(k*v for k, v in normalized.items()) / sum(normalized.values())
    weights = {k: targets[k] / means[k] for k in targets}
    denominator = math.fsum(weights.values())
    probabilities = {k: v / denominator for k, v in weights.items()}
    if draws is None:
        return dict(selection_weights=weights, selection_probabilities=probabilities,
                    target_example_shares=targets, mean_emitted_batch_sizes=means,
                    simulation_passed=None, draws=0,
                    simulation_scope='not requested; analytical weights from measured histograms')
    names = list(targets)
    selection = random.Random(seed)
    size_rng = random.Random(seed + 1)
    selected = Counter(selection.choices(names, weights=list(weights.values()), k=draws))
    emitted = {k: sum(size_rng.choices(sizes[k], weights=counts[k], k=selected[k])) for k in names}
    total = sum(emitted.values())
    shares = {k: emitted[k] / total for k in names}
    errors = {k: 100*(shares[k]-targets[k]) for k in names}
    return dict(selection_weights=weights, selection_probabilities=probabilities,
                target_example_shares=targets, mean_emitted_batch_sizes=means,
                simulated_example_shares=shares, deviation_percentage_points=errors,
                draws=draws, seed=seed, tolerance_percentage_points=tolerance_pp,
                simulation_passed=max(map(abs, errors.values())) <= tolerance_pp,
                simulation_scope='measured batch-size histograms; not a production guarantee')


def stress_batches(candidates, batch_size, seed=42):
    """Choose representative and distinct shape-extreme real-crop batches.

    Candidates belong to ONE bucket and carry actual native formatted lengths.
    Mixed audio/token extrema are retained. Duplicate membership is removed, but
    shape alone is not assumed to prove equivalent execution paths.
    """
    batch_size = positive_integer(batch_size, 'batch_size')
    if not candidates:
        raise ValueError('no candidates in bucket')
    identities = [x['id'] for x in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError('candidate crop identities must be unique')
    for row in candidates:
        for key in ('duration', 'input_tokens', 'target_tokens'):
            positive(row[key], key)
    rng = random.Random(seed)
    representative = list(candidates)
    rng.shuffle(representative)
    rankings = [representative]
    for key in ('duration', 'input_tokens', 'target_tokens'):
        rankings.append(sorted(candidates, key=lambda x: (-x[key], x['id'])))
    mixed, seen = [], set()
    for items in zip(*rankings[1:]):
        for item in items:
            if item['id'] not in seen:
                mixed.append(item)
                seen.add(item['id'])
    rankings.append(mixed)
    result, memberships = [], set()
    for label, ranked in zip(('representative', 'long_audio', 'long_input', 'long_target', 'mixed_extremes'), rankings):
        chosen = [ranked[i % len(ranked)] for i in range(batch_size)]
        signature = tuple(sorted(x['id'] for x in chosen))
        if signature in memberships:
            continue
        memberships.add(signature)
        result.append(dict(kind=label, crop_ids=[x['id'] for x in chosen],
                           unique_examples=len(set(signature)), batch_size=batch_size,
                           max_duration=max(x['duration'] for x in chosen),
                           max_input_tokens=max(x['input_tokens'] for x in chosen),
                           max_target_tokens=max(x['target_tokens'] for x in chosen)))
    return result


def sample_identity(cut):
    custom = cut.custom or {}
    return str(custom.get('sample_id', custom.get('_manifest_row_id', cut.id)))


def cuts_for(source, mode="sampled", seed=42, metadata=True):
    """Shared finite native traversal for duration export and real-crop profiling.

    Analysis mode is applied to a copy and never written into exported training YAML.
    """
    import copy
    from omegaconf import OmegaConf
    from nemo.collections.common.data.lhotse.cutset import read_cutset_from_config
    item = copy.deepcopy(source)
    controls = dict(metadata_only=metadata, force_finite=True, shuffle=False, shard_seed=int(seed))
    item.update(controls)
    if item.get('context_sampling'):
        item['context_sampling']['analysis_mode'] = mode
    elif mode != 'sampled':
        raise ValueError('Context corners require context_sampling')
    cuts, _ = read_cutset_from_config(OmegaConf.create(dict(input_cfg=[item], **controls)))
    yield from cuts


def context_duration_rows(input_cfg, *, seed=42):
    """Export finite native input-config crops; no audio decode or model forward.

    Use one source/category per estimation. Physical-member durations must have
    been validated separately against audio headers. Explicit _skipme rows follow
    native exclusion semantics. Other malformed rows fail rather than disappear.
    """
    from omegaconf import OmegaConf, ListConfig

    sources = OmegaConf.load(input_cfg)
    if not isinstance(sources, ListConfig) or len(sources) != 1:
        raise ValueError('Use a native input_cfg YAML list containing exactly one source/category')
    source = sources[0]
    if source.get('type') not in ('nemo', 'nemo_tarred'):
        raise ValueError('Context estimation currently supports native nemo/nemo_tarred manifests only')
    if not isinstance(source.get('manifest_filepath'), str):
        raise ValueError('Use one manifest path or native sharded-path expression per source')
    cuts = cuts_for(OmegaConf.to_container(source, resolve=True), seed=seed)
    rows = []
    seen = set()
    for cut in cuts:
        if not hasattr(cut, 'duration'):
            raise ValueError('Duration estimation requires audio-only input')
        custom = cut.custom or {}
        identifier = sample_identity(cut)
        if identifier in seen:
            raise ValueError(f'Duplicate example {identifier!r}; use one finite source/category')
        seen.add(identifier)
        bounds = custom.get('context_duration_bounds', {})
        rows.append(dict(id=identifier, sampled_duration=cut.duration,
                         min_duration=bounds.get('min_duration', cut.duration),
                         max_duration=bounds.get('max_duration', cut.duration)))
    if not rows:
        raise ValueError('No eligible examples in input configuration')
    return rows


def estimate_context_duration_bins(input_cfg, *, seed=42, num_buckets=10):
    rows = context_duration_rows(input_cfg, seed=seed)
    result = duration_geometry(rows, num_buckets=num_buckets)
    result.update(rows=rows, seed=seed, input_cfg=str(input_cfg),
                  physical_duration_validation='REQUIRES_SEPARATE_AUDIO_HEADER_VALIDATION',
                  validation_status='METADATA_ONLY_NOT_TRAINING_READY')
    return result
