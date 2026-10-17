# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only materialization of metadata-selected profiling crops.

Filter manifest rows before opening audio. Read selected members once into a
private disk cache, then reuse the native adapter for every context variant.
Unselected tar shards are never opened. Local seekable archives can skip payloads;
streaming archives still traverse intervening bytes, but only once per shard.
"""
import copy
import hashlib
import json
import tarfile
from pathlib import Path

from nemo.collections.common.data.lhotse.nemo_adapters import (
    LazyNeMoTarredIterator, _manifest_row_identity, _validate_context_sampling_config,
)
from nemo.collections.speechlm2.data.batch_pipeline import digest, load_json, write_json


class SelectedTarIterator(LazyNeMoTarredIterator):
    """Analysis-only selection/cache; native crop and alignment logic is unchanged."""

    def __init__(self, source, requested, seed, cache):
        if source.get('extra_fields') or source.get('slice_length'):
            raise ValueError('Selective collection does not support extra_fields or slice_length')
        super().__init__(
            source['manifest_filepath'], source['tarred_audio_filepaths'],
            shard_seed=seed, shuffle_shards=False,
            text_field=source.get('text_field', 'text'), lang_field=source.get('lang_field', 'lang'),
            context_sampling=source.get('context_sampling'),
        )
        if self.use_ais_get_batch:
            raise ValueError('Selective collection requires USE_AIS_GET_BATCH=false')
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        wanted = {r['source_id'] for r in requested}
        found = set()
        selected = {}
        for sid, rows in self.shard_id_to_manifest.items():
            kept = []
            for row in rows:
                identity = str(row.get('sample_id', _manifest_row_identity(row)))
                if identity in wanted and not row.get('_skipme', False):
                    if identity in found:
                        raise ValueError(f'Duplicate selected sample identity: {identity}')
                    found.add(identity)
                    kept.append(row)
            if kept:
                selected[sid] = kept
        if wanted != found:
            raise ValueError(f'Selected examples absent from manifest: {sorted(wanted-found)}')
        self.shard_id_to_manifest = selected
        self.tags = copy.deepcopy(source.get('tags', {}))
        self.original_context = copy.deepcopy(source.get('context_sampling'))
        self.members_read = 0
        self.shards_opened = 0

    def for_mode(self, mode):
        context = copy.deepcopy(self.original_context)
        if context:
            context['analysis_mode'] = mode
        elif mode != 'sampled':
            raise ValueError('Context corners require context_sampling')
        self.context_sampling = _validate_context_sampling_config(context)
        self.epoch = 0  # Each analysis mode uses the recorded seed, not a new training epoch.
        from nemo.collections.common.data.lhotse.cutset import attach_tags
        return (attach_tags(cut, self.tags) for cut in self)

    def _iter_sequential(self, tar_path, shard_manifest, manifest_path, rng):
        from lhotse.serialization import open_best
        # Cache key includes the original archive and member, not just its basename.
        paths = {name: self.cache / hashlib.sha256(
            json.dumps([str(tar_path), name]).encode()).hexdigest() for name in shard_manifest}
        missing = set()
        for name, path in paths.items():
            receipt = path.with_suffix('.json')
            if not path.exists() or not receipt.exists():
                missing.add(name)
            elif digest(path) != load_json(receipt)['sha256']:
                raise ValueError(f'Cached recording changed: {path}')
        if missing:
            self.shards_opened += 1
            # r:* uses seeks for local files; pipe/object-store streams stay sequential.
            mode = 'r:*' if Path(str(tar_path)).is_file() else 'r|*'
            with open_best(tar_path, mode='rb') as stream, tarfile.open(fileobj=stream, mode=mode) as archive:
                for member in archive:
                    if member.name not in missing:
                        continue
                    if not member.isfile():
                        raise ValueError(f'Selected audio member is not a file: {member.name}')
                    path = paths[member.name]
                    tmp = path.with_suffix('.tmp')
                    with archive.extractfile(member) as audio, tmp.open('wb') as out:
                        import shutil
                        shutil.copyfileobj(audio, out)
                    tmp.replace(path)
                    write_json(path.with_suffix('.json'), dict(sha256=digest(path)))
                    self.members_read += 1
                    missing.remove(member.name)
                    if not missing:
                        break
            if missing:
                raise ValueError(f'Selected members absent from {tar_path}: {sorted(missing)}')
        for name, path in paths.items():
            info = tarfile.TarInfo(name)
            info.size = path.stat().st_size
            # The parent performs native physical-header, context and alignment checks.
            yield shard_manifest[name], path.read_bytes(), info


def collect(args, plan):
    from lhotse import CutSet, MonoCut
    from omegaconf import OmegaConf
    from nemo.collections.speechlm2 import StreamingSTTDataset
    from nemo.collections.speechlm2.models.streaming_stt_model import (
        StreamingSTTModelConfig, setup_streaming_tokenizer,
    )
    from nemo.collections.speechlm2.parts.utils import to_dataclass
    from nemo.collections.speechlm2.parts.alignments import get_word_alignments_for_batch
    from nemo.collections.speechlm2.data.batch_profiling import selected_crops
    from nemo.collections.speechlm2.data.batch_pipeline import cuts_for

    cfg = OmegaConf.load(plan['model_config']); OmegaConf.resolve(cfg)
    if cfg.data.dataset.get('enable_turn_state_tokens', False):
        raise ValueError('This branch supports boundary training only')
    tokenizer, _ = setup_streaming_tokenizer(to_dataclass(StreamingSTTModelConfig, cfg.model))
    dataset = StreamingSTTDataset(cfg=cfg.data.dataset, tokenizer=tokenizer, defer_get_batch=True)
    destination = args.output / args.profile
    geometry = load_json(destination/'geometry.json')
    audio_dir = destination/'candidate_audio'; audio_dir.mkdir(exist_ok=True)
    by_source = {}
    for item in geometry['candidates']:
        by_source.setdefault(item['source_index'], []).append(item)
    candidates, saved_cuts = [], []
    members_read = shards_opened = 0
    for index, requested in by_source.items():
        source = geometry['sources'][index]
        reader = (SelectedTarIterator(source, requested, plan['seed'], destination/'recording_cache'/str(index))
                  if source.get('tarred_audio_filepaths') else None)
        modes = dict.fromkeys(item['mode'] for item in requested)
        for mode in modes:
            selection = [item for item in requested if item['mode'] == mode]
            # Completed crops can be resumed without decoding/formatting them again.
            pending = []
            for item in selection:
                receipt = audio_dir/(item['id'].replace(':', '_')+'.json')
                if receipt.exists():
                    record = load_json(receipt)
                    if record['selection'] != item or digest(audio_dir/(item['id'].replace(':', '_')+'.wav')) != record['candidate']['audio_sha256']:
                        raise ValueError(f'Stale collected crop: {receipt}')
                    candidates.append(record['candidate']); saved_cuts.append(MonoCut.from_dict(record['cut']))
                else:
                    pending.append(item)
            if not pending:
                continue
            if reader:
                cuts = reader.for_mode(mode)
            else:
                # Plain file manifests: metadata selection first, then selected file headers only.
                from lhotse import Recording
                def plain_cuts():
                    for item, cut in selected_crops(cuts_for(source, mode, plan['seed'], True), pending):
                        # Metadata cut IDs contain the resolved audio path.
                        recording = Recording.from_file(cut.id)
                        cut.recording = recording
                        cut.channel = recording.channel_ids[0]
                        for supervision in cut.supervisions:
                            supervision.recording_id = recording.id
                        yield cut
                cuts = plain_cuts()
            for item, cut in selected_crops(cuts, pending):
                cut = cut.resample(cfg.data.dataset.sample_rate)
                path = audio_dir/(item['id'].replace(':', '_')+'.wav')
                cut = cut.save_audio(str(path)); cut.id = item['id']
                batch = dataset[CutSet.from_cuts([cut])]
                if batch is None or len(batch.cuts) != 1:
                    raise ValueError('Real audio loader dropped selected example')
                if cfg.get('forced_aligner') and any(
                    (c.custom or {}).get('word_alignment_mode', 'online') != 'precomputed_required' for c in batch.cuts
                ):
                    raise ValueError('CPU collection cannot run online forced alignment; supply precomputed_required alignments')
                alignments = get_word_alignments_for_batch(batch.cuts)
                if any(text.strip() and not words for text, words in zip(batch.text, alignments)):
                    raise ValueError('CPU collection requires saved word alignments for selected examples')
                formatted = dataset.get_batch_data(batch.cuts, batch.audios, batch.audio_lens, alignments, batch.text)
                if formatted is None or len(formatted.audio_lens) != 1:
                    raise ValueError('Real target formatter dropped selected example')
                candidate = dict(item, input_tokens=int(formatted.input_token_lens[0]),
                                 target_tokens=int(formatted.target_token_lens[0]), audio_sha256=digest(path))
                write_json(path.with_suffix('.json'), dict(selection=item, candidate=candidate, cut=cut.to_dict()))
                candidates.append(candidate); saved_cuts.append(cut)
                print(f'Collected {len(candidates)}/{len(geometry["candidates"])} crops: {item["id"]}', flush=True)
        if reader:
            members_read += reader.members_read; shards_opened += reader.shards_opened
    cuts_path = destination/'candidates.jsonl.gz'
    CutSet.from_cuts(saved_cuts).to_file(cuts_path)
    write_json(destination/'collect.json', dict(status='PASS', candidates=candidates, cuts_sha256=digest(cuts_path),
                                               collection_device='cpu', members_read=members_read, shards_opened=shards_opened))
    return 0
