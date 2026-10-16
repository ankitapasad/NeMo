"""CPU-only tests that intentionally avoid importing the training dependency tree."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[3] / 'nemo/collections/speechlm2/data/batch_estimation.py'
spec = importlib.util.spec_from_file_location('batch_estimation_core', path)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_crop_moves_bucket_and_tail_is_not_silently_dropped():
    assert core.bucket_index(9, [10, 15, 20]) == 0
    assert core.bucket_index(13, [10, 15, 20]) == 1
    assert core.bucket_index(15, [10, 15, 20]) == 1
    with pytest.raises(ValueError):
        core.bucket_index(21, [10, 15, 20])


def test_fixed_bounds_must_cover_context_envelope_not_only_seeded_draw():
    rows = [dict(id='x', min_duration=8, sampled_duration=9, max_duration=18)]
    with pytest.raises(ValueError, match='envelope'):
        core.duration_geometry(rows, edges=[10, 15])
    report = core.duration_geometry(rows, edges=[10, 15, 20])
    assert [x['reachable_examples'] for x in report['buckets']] == [1, 1, 1]
    assert [x['sampled_examples'] for x in report['buckets']] == [1, 0, 0]


def test_coincident_lhotse_boundaries_and_tail():
    report = core.duration_geometry([dict(id=str(i), min_duration=2, sampled_duration=5, max_duration=8) for i in range(10)])
    assert report['bucket_duration_bins'] == [5, 8]
    assert report['num_buckets'] == 2
    assert report['requested_num_buckets'] == 10


def test_automatic_bins_match_real_lhotse_and_keep_context_tail_separate():
    import lhotse
    from lhotse.dataset.sampling.dynamic_bucketing import estimate_duration_buckets

    durations = list(range(1, 101))
    rows = [dict(id=str(d), min_duration=d, sampled_duration=d, max_duration=d + 10) for d in durations]
    expected = estimate_duration_buckets(
        [lhotse.MonoCut(id=str(d), start=0., duration=d, channel=0) for d in durations], num_buckets=4)
    report = core.duration_geometry(rows, num_buckets=4)
    assert report['estimator_duration_bins'] == expected
    assert report['bucket_duration_bins'] == list(expected) + [110.]
    assert report['lhotse_version'] == lhotse.__version__
    assert expected != [25, 50, 75]  # Equal-example percentiles must not return.
    assert sum(bucket['sampled_examples'] for bucket in report['buckets']) == 100
    assert core.bucket_index(110, report['bucket_duration_bins']) == report['num_buckets'] - 1
    with pytest.raises(ValueError, match='tail'):
        core.bucket_index(111, report['bucket_duration_bins'])


@pytest.mark.parametrize('num_buckets', [1, 3])
def test_invalid_lhotse_bucket_count_is_explicit(num_buckets):
    rows = [dict(id=str(i), min_duration=1, sampled_duration=2, max_duration=3) for i in range(2)]
    with pytest.raises(ValueError, match='number of examples'):
        core.duration_geometry(rows, num_buckets=num_buckets)


def test_unequal_row_weights_are_not_silently_ignored():
    rows = [dict(id=str(i), min_duration=1, sampled_duration=2, max_duration=3, weight=i+1) for i in range(2)]
    with pytest.raises(ValueError, match='equally weighted'):
        core.duration_geometry(rows, num_buckets=2)


@pytest.mark.parametrize('duration', [0, -1, float('nan'), float('inf'), True])
def test_invalid_duration_rejected(duration):
    with pytest.raises(ValueError):
        core.bucket_index(duration, [10])


def test_mixture_compensates_for_batch_sizes():
    result = core.calibrate({'dai': .8, 'asr': .2}, {'dai': {'8': 5000}, 'asr': {'32': 5000}}, draws=100_000)
    assert result['selection_probabilities']['dai'] == pytest.approx(16 / 17)
    assert result['simulation_passed']
    assert result['simulated_example_shares']['dai'] == pytest.approx(.8, abs=.0025)


@pytest.mark.parametrize('targets,hist', [({'a': 1}, {'a': {0: 2}}), ({'a': 1}, {'a': {2: 1.5}}),
                                        ({'a': .8}, {'a': {2: 1}}), ({'a': 1}, {'a': {}})])
def test_invalid_calibration_rejected(targets, hist):
    with pytest.raises(ValueError):
        core.calibrate(targets, hist)


def test_cross_example_padding_extrema_and_sparse_reporting():
    rows = [dict(id='long', duration=15, input_tokens=100, target_tokens=90),
            dict(id='words', duration=11, input_tokens=500, target_tokens=490)]
    cases = core.stress_batches(rows, 8)
    assert all(x['unique_examples'] == 2 for x in cases)
    assert all(x['max_duration'] == 15 and x['max_input_tokens'] == 500 for x in cases)
    assert len({tuple(sorted(x['crop_ids'])) for x in cases}) == len(cases)
    single = core.stress_batches(rows, 1)
    assert {x['crop_ids'][0] for x in single} == {'long', 'words'}


def test_calibration_without_simulation_matches_analytical_example_shares():
    result = core.calibrate({'dai': .5, 'asr': .5}, {'dai': {8: 10}, 'asr': {4: 10}})
    assert result['draws'] == 0
    assert result['simulation_passed'] is None
    assert result['selection_probabilities'] == pytest.approx({'dai': 1/3, 'asr': 2/3})
