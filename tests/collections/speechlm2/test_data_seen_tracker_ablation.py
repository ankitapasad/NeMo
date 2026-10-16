import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code' if (ROOT / 'code').is_dir() else next(
    parent for parent in Path(__file__).resolve().parents if (parent / 'nemo/collections/speechlm2').is_dir())
spec = importlib.util.spec_from_file_location(
    'data_seen_tracker', CODE / 'nemo/collections/speechlm2/data/data_seen_tracker.py')
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
DataSeenTracker, CellTotal = module.DataSeenTracker, module.CellTotal
TOTALS = {('d7', 'complete_turn'): CellTotal(10, 10.), ('d7', 'two_turn_no_backchannel'): CellTotal(4, 4.)}
MEMBERSHIP = 'a' * 64


def checkpoint_hook(name):
    tree = ast.parse((CODE / 'nemo/collections/speechlm2/models/streaming_stt_model.py').read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'StreamingSTTModel')
    function = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-checkpoint-hook>', 'exec'), namespace)
    return namespace[name]


def batch(ids=('a',), durations=(3600.,), categories=None):
    return SimpleNamespace(sample_ids=list(ids), accounting_durations_s=list(durations),
                           mixture_sources=['d7'] * len(ids),
                           turn_buckets=list(categories or ['complete_turn'] * len(ids)))


def distributed_worker(rank, rendezvous, queue):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        tracker = DataSeenTracker(TOTALS, MEMBERSHIP, interval=100)
        tracker.update(batch())  # Same row consumed on both ranks.
        tracker.sync(torch.device('cpu'))
        tracker.update(batch(ids=(f'rank{rank}',), durations=(1800.,)))
        # Simulate checkpoint between periodic logging boundaries.
        saved = {}
        checkpoint_hook('on_save_checkpoint')(
            SimpleNamespace(data_seen_tracker=tracker, device=torch.device('cpu')), saved)
        restored = DataSeenTracker(TOTALS, MEMBERSHIP)
        checkpoint_hook('on_load_checkpoint')(SimpleNamespace(data_seen_tracker=restored), saved)
        restored.update(batch())
        restored.sync(torch.device('cpu'))
        queue.put((rank, restored.base_samples.tolist(), len(restored.unique),
                   restored.base_seconds.tolist()))
    finally:
        dist.destroy_process_group()


class TrackerTests(unittest.TestCase):
    def test_resume_hook_rejects_missing_counters(self):
        with self.assertRaisesRegex(ValueError, 'missing exposure counters'):
            checkpoint_hook('on_load_checkpoint')(
                SimpleNamespace(data_seen_tracker=DataSeenTracker(TOTALS, MEMBERSHIP)), {})

    def test_disabled_hooks_preserve_legacy_checkpoint(self):
        saved = {'state_dict': {'sentinel': 1}}
        checkpoint_hook('on_save_checkpoint')(SimpleNamespace(data_seen_tracker=None), saved)
        checkpoint_hook('on_load_checkpoint')(SimpleNamespace(data_seen_tracker=None), saved)
        self.assertEqual(saved, {'state_dict': {'sentinel': 1}})

    def test_default_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(DataSeenTracker.from_env())
            self.assertEqual(module.exposure_metadata([object()]), {})

    def test_repeated_unique_and_namespaced_categories(self):
        tracker = DataSeenTracker(TOTALS, MEMBERSHIP)
        tracker.update(batch(ids=('a', 'a', 'a'), durations=(3600.,) * 3,
                             categories=('complete_turn', 'complete_turn', 'two_turn_no_backchannel')))
        metrics = tracker.sync(torch.device('cpu'))
        self.assertEqual(metrics['data_seen/d7/complete_turn/samples (total=10)'], 2)
        self.assertEqual(metrics['data_unique/d7/complete_turn/samples (total=10)'], 1)
        self.assertEqual(metrics['data_seen/d7/complete_turn/hours (total=10 h)'], 2)
        self.assertEqual(len(tracker.unique), 2)

    def test_pending_checkpoint_refused_then_flush_restore(self):
        tracker = DataSeenTracker(TOTALS, MEMBERSHIP)
        tracker.update(batch())
        with self.assertRaisesRegex(RuntimeError, 'flushed'):
            tracker.state_dict()
        tracker.sync(torch.device('cpu'))
        restored = DataSeenTracker(TOTALS, MEMBERSHIP)
        restored.load_state_dict(tracker.state_dict())
        restored.update(batch())
        restored.sync(torch.device('cpu'))
        self.assertEqual(restored.base_samples.tolist(), [2, 0])
        self.assertEqual(len(restored.unique), 1)

    def test_identity_duration_mismatch_before_and_after_sync(self):
        for synced in (False, True):
            tracker = DataSeenTracker(TOTALS, MEMBERSHIP)
            tracker.update(batch())
            if synced:
                tracker.sync(torch.device('cpu'))
            with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
                tracker.update(batch(durations=(4.,)))

    def test_resume_rejects_membership_or_totals_change(self):
        state = DataSeenTracker(TOTALS, MEMBERSHIP).state_dict()
        with self.assertRaisesRegex(ValueError, 'membership'):
            DataSeenTracker(TOTALS, 'b' * 64).load_state_dict(state)
        with self.assertRaisesRegex(ValueError, 'totals'):
            DataSeenTracker({('d7', 'complete_turn'): CellTotal(11, 11.)}, MEMBERSHIP).load_state_dict(state)

    def test_fixed_duration_metadata_and_missing_category(self):
        cut = SimpleNamespace(id='unstable-crop-id', duration=9., custom={
            'shared_tar_category': 'complete_turn', 'sample_id': 'stable-row',
            'data_seen_accounting_duration_s': 12.})
        with patch.dict(os.environ, {DataSeenTracker.ENV_PATH: '/test/totals', DataSeenTracker.ENV_LABEL_MODE: 'legacy'}):
            data = module.exposure_metadata([cut])
            self.assertEqual(data['sample_ids'], ['stable-row'])
            self.assertEqual(data['accounting_durations_s'], [12.])
            cut.custom = {'taskname': 'davidai-d7-turn-taking'}
            with self.assertRaisesRegex(ValueError, 'shared_tar_category'):
                module.exposure_metadata([cut])

    def test_generic_labels_and_default_category(self):
        cuts = [SimpleNamespace(id='same-id', duration=5., custom={'data_seen_source': name})
                for name in ('meeting_audio', 'podcasts')]
        with patch.dict(os.environ, {DataSeenTracker.ENV_PATH: '/test/totals',
                                     DataSeenTracker.ENV_LABEL_MODE: 'explicit'}):
            metadata = module.exposure_metadata(cuts)
        self.assertEqual(metadata['mixture_sources'], ['meeting_audio', 'podcasts'])
        self.assertEqual(metadata['turn_buckets'], ['all', 'all'])
        totals = {(name, 'all'): CellTotal(10, 1.) for name in ('meeting_audio', 'podcasts')}
        tracker = DataSeenTracker(totals, MEMBERSHIP)
        tracker.update(SimpleNamespace(**metadata))
        metrics = tracker.sync(torch.device('cpu'))
        self.assertEqual(len(tracker.unique), 2)
        self.assertEqual(metrics['data_unique/podcasts/all/sample_coverage'], .1)
        restored = DataSeenTracker(totals, MEMBERSHIP)
        restored.load_state_dict(tracker.state_dict())
        self.assertEqual(restored.metrics(), metrics)

    def test_explicit_labels_override_legacy_metadata(self):
        cut = SimpleNamespace(id='row', duration=5., custom={
            'data_seen_source': 'new_corpus', 'data_seen_category': 'short',
            'shared_tar_category': 'complete_turn', 'mixture_source': 'd7'})
        with patch.dict(os.environ, {DataSeenTracker.ENV_PATH: '/test/totals',
                                     DataSeenTracker.ENV_LABEL_MODE: 'legacy'}):
            metadata = module.exposure_metadata([cut])
        self.assertEqual(metadata['mixture_sources'], ['new_corpus'])
        self.assertEqual(metadata['turn_buckets'], ['short'])

    def test_unlabelled_data_and_invalid_labels_rejected(self):
        with patch.dict(os.environ, {DataSeenTracker.ENV_PATH: '/test/totals',
                                     DataSeenTracker.ENV_LABEL_MODE: 'explicit'}):
            for custom in ({}, {'taskname': 'a_dataset'}, {'data_seen_category': 'short'},
                           {'data_seen_source': ''}, {'data_seen_source': 'a/b'},
                           {'data_seen_source': 'corpus', 'data_seen_category': ' '}):
                with self.assertRaises(ValueError):
                    module.exposure_metadata([SimpleNamespace(id='row', duration=2., custom=custom)])

    def test_generic_sampled_context_requires_fixed_identity_and_duration(self):
        custom = {'data_seen_source': 'new_corpus', 'curation': {'audio_context': {
            'selection': {'policy': 'dataloader_uniform_available_transcript_bounded'}}}}
        cut = SimpleNamespace(id='crop-dependent-id', duration=9., custom=custom)
        with patch.dict(os.environ, {DataSeenTracker.ENV_PATH: '/test/totals',
                                     DataSeenTracker.ENV_LABEL_MODE: 'explicit'}):
            with self.assertRaisesRegex(ValueError, 'stable'):
                module.exposure_metadata([cut])
            custom['sample_id'] = 'stable-row'
            with self.assertRaisesRegex(ValueError, 'accounting duration'):
                module.exposure_metadata([cut])
            custom['data_seen_accounting_duration_s'] = 5.
            self.assertEqual(module.exposure_metadata([cut])['accounting_durations_s'], [5.])

    def test_unknown_asr_is_not_silently_ignored(self):
        tracker = DataSeenTracker(TOTALS, MEMBERSHIP)
        item = batch()
        item.mixture_sources, item.turn_buckets = ['asr'], ['asr']
        with self.assertRaisesRegex(ValueError, 'Unregistered'):
            tracker.update(item)
        tracker = DataSeenTracker(TOTALS, MEMBERSHIP, ignored_cells=[('asr', 'asr')])
        tracker.update(item)
        self.assertEqual(tracker.delta_samples.sum().item(), 0)
        tracker.sync(torch.device('cpu'))
        with self.assertRaisesRegex(ValueError, 'ignored cells'):
            DataSeenTracker(TOTALS, MEMBERSHIP).load_state_dict(tracker.state_dict())

    def test_totals_file_explicit_exclusion_and_legacy_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            import json
            path = Path(tmp) / 'totals.json'
            payload = dict(schema_version='nemo_data_seen_totals_v2', membership_sha256=MEMBERSHIP,
                           cells={'meeting_audio/all': dict(samples=10, hours=1.)},
                           ignored_cells=['background/all'])
            path.write_text(json.dumps(payload))
            with patch.dict(os.environ, {DataSeenTracker.ENV_PATH: str(path),
                                         DataSeenTracker.ENV_LABEL_MODE: 'explicit'}):
                tracker = DataSeenTracker.from_env()
                self.assertEqual(tracker.ignored_cells, {('background', 'all')})
            with patch.dict(os.environ, {DataSeenTracker.ENV_PATH: str(path),
                                         DataSeenTracker.ENV_LABEL_MODE: 'legacy'}):
                tracker = DataSeenTracker.from_env()
                self.assertIn(('asr', 'asr'), tracker.ignored_cells)
                metadata = module.exposure_metadata([SimpleNamespace(id='asr-row', duration=1., custom={})])
                self.assertEqual(metadata['mixture_sources'], ['asr'])
            with self.assertRaisesRegex(ValueError, 'both tracked and ignored'):
                DataSeenTracker(TOTALS, MEMBERSHIP, ignored_cells=[('d7', 'complete_turn')])

    def test_actual_two_rank_global_dedup_and_checkpoint(self):
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            workers = [ctx.Process(target=distributed_worker, args=(rank, f'file://{tmp}/rendezvous', queue))
                       for rank in range(2)]
            for worker in workers:
                worker.start()
            try:
                results = [queue.get(timeout=45) for _ in workers]
                for worker in workers:
                    worker.join(timeout=10)
                    self.assertEqual(worker.exitcode, 0)
                for _, counts, unique_count, seconds in results:
                    self.assertEqual(counts, [6., 0.])
                    self.assertEqual(unique_count, 3)
                    self.assertEqual(seconds, [18000., 0.])
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=5)


if __name__ == '__main__':
    unittest.main()
