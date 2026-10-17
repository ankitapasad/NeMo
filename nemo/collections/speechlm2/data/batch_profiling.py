# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-data connector for legacy OOMptimizer batch-size search.

Training, membership and memory checks adapted from the validated D7 real-batch
profiler. All model/data paths and scientific settings come from the user's config.
"""
import copy
import os
from pathlib import Path
import time

from nemo.collections.speechlm2.data.batch_pipeline import cuts_for, digest, load_json, write_json
from nemo.collections.speechlm2.data.batch_estimation import stress_batches, sample_identity


def selected_crops(cuts, requested):
    """Match metadata selections by stable ID, independent of tar member order."""
    pending = {item['source_id']: item for item in requested}
    if len(pending) != len(requested):
        raise ValueError('Duplicate selected sample identity within a source/context mode')
    for cut in cuts:
        identity = sample_identity(cut)
        if identity not in pending:
            continue
        item = pending.pop(identity)
        # Native decoded cuts round durations to audio samples; metadata is continuous.
        tolerance = max(1e-5, 1.0 / cut.sampling_rate)
        if abs(cut.duration - item['duration']) > tolerance:
            raise ValueError(f"Crop duration differs for {identity!r}: metadata={item['duration']}, audio={cut.duration}")
        yield item, cut
        if not pending:
            return
    if pending:
        raise ValueError(f'Selected examples missing in real traversal: {sorted(pending)}')


def worker(args, plan):
    if args.action == 'collect':
        from nemo.collections.speechlm2.data.batch_collection import collect
        return collect(args, plan)
    import torch
    from lightning.pytorch import Trainer, Callback, seed_everything
    from omegaconf import OmegaConf
    from lhotse import CutSet
    from nemo.collections.speechlm2 import StreamingSTTModel, StreamingSTTDataset
    from nemo.core.classes.common import Serialization
    from nemo.utils.trainer_utils import resolve_trainer_cfg

    os.environ['WANDB_MODE'] = 'disabled'
    # Profiling has its own membership accounting; it must not modify production counters.
    os.environ.pop('NEMO_DATA_SEEN_TOTALS_PATH', None)
    seed_everything(plan['seed'], workers=True)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(plan['memory_fraction'])
    destination = args.output / args.profile
    suffix = args.action if args.bucket is None else f'{args.action}_{args.bucket}_{args.batch_size}'
    report_path = destination / (suffix + '.json')
    if report_path.exists():
        raise FileExistsError(report_path)
    cfg = OmegaConf.load(plan['model_config']); OmegaConf.resolve(cfg)
    if cfg.data.dataset.get('enable_turn_state_tokens', False):
        raise ValueError('This branch supports boundary training only')
    if cfg.trainer.get('accumulate_grad_batches', 1) != 1:
        raise ValueError('Real-data profiling currently requires accumulate_grad_batches=1; do not silently override training')
    trainer_cfg = resolve_trainer_cfg(cfg.trainer)
    trainer_cfg.update(devices=1, num_nodes=1, accelerator='gpu', strategy='auto', logger=False,
                       enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                       max_epochs=1, max_steps=1, limit_train_batches=1, limit_val_batches=1,
                       num_sanity_val_steps=1, val_check_interval=1, log_every_n_steps=1)
    trainer_cfg.pop('callbacks', None)
    records = []
    current = {}

    class Checks(Callback):
        def on_train_batch_start(self, trainer, model, batch, batch_idx):
            current['started'] = time.monotonic()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        def on_before_backward(self, trainer, model, loss):
            if not torch.isfinite(loss).all():
                raise ValueError('Nonfinite real-batch loss')
            current['loss'] = float(loss.detach())
        def on_after_backward(self, trainer, model):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('Missing or nonfinite real-batch gradients')
            current['gradient_tensors'] = len(grads)
        def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            external = max(0, total-free-torch.cuda.memory_reserved())
            peak = torch.cuda.max_memory_reserved()+external
            if peak > plan['memory_fraction']*total:
                raise torch.cuda.OutOfMemoryError('Reserved plus observed non-PyTorch memory exceeds profiling budget')
            records.append(dict(case=cases[batch_idx // plan['updates_per_case']],
                                loss=current['loss'], gradient_tensors=current['gradient_tensors'],
                                budgeted_peak_bytes=peak, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                elapsed_seconds=time.monotonic()-current['started']))

    try:
        trainer = Trainer(**trainer_cfg, callbacks=[Checks()], default_root_dir=str(destination/suffix))
        forced_aligner = Serialization.from_config_dict(cfg.forced_aligner) if cfg.get('forced_aligner') else None
        with trainer.init_module():
            model = StreamingSTTModel(OmegaConf.to_container(cfg.model, resolve=True), forced_aligner=forced_aligner,
                                      data_cfg=cfg.data.dataset, dataset_cls=StreamingSTTDataset)
            if cfg.get('init_weights_from_checkpoint'):
                from nemo.collections.speechlm2.parts.stage2_checkpoint import load_asr_stage2_weights
                load_asr_stage2_weights(model, cfg.init_weights_from_checkpoint, lora_merge=cfg.get("init_lora_merge"))
        dataset = StreamingSTTDataset(cfg=cfg.data.dataset, tokenizer=model.tokenizer, defer_get_batch=True)

        def formatted(cuts):
            batch = dataset[cuts]
            if batch is None or len(batch.cuts) != len(cuts):
                raise ValueError('Real audio loader dropped examples')
            if forced_aligner is None:
                from nemo.collections.speechlm2.parts.alignments import get_word_alignments_for_batch
                alignments = get_word_alignments_for_batch(batch.cuts)
                if any(text.strip() and not words for text, words in zip(batch.text, alignments)):
                    raise ValueError('Missing saved alignments and no forced_aligner configured')
            else:
                alignments = dataset.get_mixed_alignments(cuts=batch.cuts, audios=batch.audios, audio_lens=batch.audio_lens,
                                                          text=batch.text, forced_aligner=forced_aligner)
            result = dataset.get_batch_data(batch.cuts, batch.audios, batch.audio_lens, alignments, batch.text)
            if result is None or len(result.audio_lens) != len(cuts):
                raise ValueError('Real target formatter dropped examples')
            return result

        collected = load_json(destination/'collect.json')
        cuts_path = destination/'candidates.jsonl.gz'
        if digest(cuts_path) != collected['cuts_sha256']:
            raise ValueError('Candidate cuts changed')
        available = {c.id:c for c in CutSet.from_file(cuts_path)}
        selected = load_json(destination/'selected.json') if args.action == 'validate' else None
        bucket_sizes = (list(enumerate(selected['bucket_batch_size'])) if selected is not None
                        else [(args.bucket, args.batch_size)])
        cases = []
        for bucket, size in bucket_sizes:
            pool = [c for c in collected['candidates'] if c['bucket'] == bucket]
            for item in pool:
                audio_path = destination/'candidate_audio'/(item['id'].replace(':', '_')+'.wav')
                if digest(audio_path) != item['audio_sha256']:
                    raise ValueError('Candidate audio changed')
            begin = len(cases)
            memberships = set()
            pools = [pool] + [[c for c in pool if c['source_index']==source] for source in sorted({c['source_index'] for c in pool})]
            for subset in pools:
                for case in stress_batches(subset, size, plan['seed']):
                    signature = tuple(sorted(case['crop_ids']))
                    if signature not in memberships:
                        memberships.add(signature); cases.append(case)
            if len(pools) > 2 and size > 1:
                # Explicit source-mixed case, even if all global extrema rank one source first.
                source_pools = pools[1:]
                chosen = []
                for slot in range(size):
                    source_pool = source_pools[slot % len(source_pools)]
                    key = 'duration' if slot % 2 == 0 else 'input_tokens'
                    chosen.append(max(source_pool, key=lambda c: c[key])['id'])
                signature = tuple(sorted(chosen))
                if signature not in memberships:
                    cases.append(dict(kind='mixed_sources', crop_ids=chosen, batch_size=size,
                                      unique_examples=len(set(chosen))))

            if len(cases) == begin:
                raise ValueError('No real stress batches for bucket')
            for case in cases[begin:]:
                case['bucket'] = bucket

        class Batches(torch.utils.data.IterableDataset):
            def __iter__(self):
                for case in cases:
                    cuts = []
                    for i, ident in enumerate(case['crop_ids']):
                        cut = copy.deepcopy(available[ident]); cut.id = f'{ident}:slot{i}'
                        cuts.append(cut)
                    cutset = CutSet.from_cuts(cuts)
                    batch = dataset[cutset] if forced_aligner is not None else formatted(cutset)
                    if batch is None or len(batch.audio_lens) != len(cuts):
                        raise ValueError('Training loader dropped selected examples')
                    for _ in range(plan['updates_per_case']):
                        yield batch
            def __len__(self):
                return len(cases)*plan['updates_per_case']

        loader = torch.utils.data.DataLoader(Batches(), batch_size=None, num_workers=0)
        steps = len(loader)
        # Construct a fresh trainer around the same initialized model. Precision is unchanged.
        trainer_cfg.update(max_steps=steps, limit_train_batches=steps, val_check_interval=steps)
        trainer = Trainer(**trainer_cfg, callbacks=[Checks()], default_root_dir=str(destination/suffix))
        trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
        if trainer.global_step != steps or len(records) != steps:
            raise ValueError('Incomplete optimizer updates')
        write_json(report_path, dict(status='PASS', bucket=args.bucket, batch_size=args.batch_size,
                    optimizer_updates=steps, checks=records, gpu_model=torch.cuda.get_device_name(),
                    gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    budgeted_peak_bytes=max(r['budgeted_peak_bytes'] for r in records),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    llm_gradient_checkpointing=bool(model.llm.is_gradient_checkpointing),
                    selected_sha256=digest(destination/'selected.json') if selected is not None else None))
        return 0
    except torch.cuda.OutOfMemoryError as error:
        if args.action == 'collect':
            raise RuntimeError('Model/candidate construction cannot fit memory budget') from error
        write_json(report_path, dict(status='OOM', bucket=args.bucket, batch_size=args.batch_size, message=str(error)))
        return 86
