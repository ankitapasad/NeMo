# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lhotse adapters for NeMo datasets including Parquet support."""
import copy
import hashlib
import math
import os
import random
import re
import tarfile
from collections.abc import Mapping, Sequence
from io import BytesIO
from numbers import Real
from pathlib import Path
from typing import Generator, Iterable, List, Literal

try:
    import pyarrow.parquet as pq

    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False
import soundfile
from cytoolz import groupby
from lhotse import AudioSource, MonoCut, Recording, SupervisionSegment
from lhotse.audio.backend import LibsndfileBackend
from lhotse.cut import Cut
from lhotse.dataset.dataloading import resolve_seed
from lhotse.lazy import LazyIteratorChain, LazyJsonlIterator
from lhotse.serialization import open_best
from lhotse.utils import compute_num_samples, ifnone

from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging
from nemo.utils.data_utils import is_datastore_path


LEAN_MULTI_TURN_SCHEMA_VERSION = "lean_multi_turn_v2"
_CONTEXT_SAMPLING_FIELDS = {
    "min_leading_s",
    "min_trailing_s",
    "max_duration_s",
}


def _finite_nonnegative_number(value, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"context_sampling.{field} must be a non-negative finite number; got {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"context_sampling.{field} must be a non-negative finite number; got {value!r}")
    return value


def _finite_number(value, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"context_sampling.{field} must be a finite number; got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"context_sampling.{field} must be a finite number; got {value!r}")
    return value


def _validate_context_sampling_config(config: Mapping | None) -> dict | None:
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise TypeError(f"context_sampling must be a mapping; got {type(config).__name__}")
    unknown = set(config) - _CONTEXT_SAMPLING_FIELDS
    if unknown:
        raise ValueError(f"context_sampling has unsupported fields: {sorted(unknown)}")
    missing = _CONTEXT_SAMPLING_FIELDS - set(config)
    if missing:
        raise ValueError(f"context_sampling is missing required fields: {sorted(missing)}")

    normalized = {key: _finite_nonnegative_number(config[key], field=key) for key in _CONTEXT_SAMPLING_FIELDS}
    if normalized["max_duration_s"] <= 0:
        raise ValueError("context_sampling.max_duration_s must be positive")
    return normalized


def _uniform_context_pair(
    rng: random.Random,
    *,
    leading_low: float,
    leading_high: float,
    trailing_low: float,
    trailing_high: float,
    total_budget: float,
) -> tuple[float, float]:
    """Sample uniformly from two context intervals conditioned on their sum fitting the budget."""
    tolerance = 1e-9
    if leading_low + trailing_low > total_budget + tolerance:
        raise ValueError(
            "Minimum available leading/trailing context does not fit max_duration_s: "
            f"{leading_low} + {trailing_low} > {total_budget}"
        )

    leading_high = min(leading_high, total_budget - trailing_low)
    if leading_high < leading_low:
        leading_high = leading_low

    leading_span = leading_high - leading_low
    trailing_span = trailing_high - trailing_low
    if leading_span <= tolerance and trailing_span <= tolerance:
        return leading_low, trailing_low
    if leading_span <= tolerance:
        return leading_low, rng.uniform(trailing_low, min(trailing_high, total_budget - leading_low))
    if trailing_span <= tolerance:
        return rng.uniform(leading_low, min(leading_high, total_budget - trailing_low)), trailing_low
    if leading_high + trailing_high <= total_budget + tolerance:
        return rng.uniform(leading_low, leading_high), rng.uniform(trailing_low, trailing_high)

    # The feasible region is a rectangle clipped by leading + trailing <= total_budget.
    # Its vertical width is constant, then decreases linearly. Sampling leading with
    # density proportional to that width and trailing uniformly inside it produces a
    # uniform joint sample over the clipped region.
    plateau_end = min(leading_high, total_budget - trailing_high)
    full_trailing_width = trailing_high - trailing_low
    plateau_area = max(0.0, plateau_end - leading_low) * full_trailing_width

    slope_start = max(leading_low, total_budget - trailing_high)
    slope_end = leading_high
    slope_width = max(0.0, slope_end - slope_start)
    width_at_start = max(0.0, total_budget - trailing_low - slope_start)
    slope_area = width_at_start * slope_width - 0.5 * slope_width * slope_width
    total_area = plateau_area + max(0.0, slope_area)
    if total_area <= tolerance:
        return leading_low, trailing_low

    area_sample = rng.random() * total_area
    if area_sample < plateau_area:
        leading = leading_low + area_sample / full_trailing_width
        trailing_upper = trailing_high
    else:
        slope_sample = area_sample - plateau_area
        discriminant = max(0.0, width_at_start * width_at_start - 2.0 * slope_sample)
        leading = slope_start + width_at_start - math.sqrt(discriminant)
        leading = min(max(leading, slope_start), slope_end)
        trailing_upper = min(trailing_high, total_budget - leading)
    trailing = rng.uniform(trailing_low, trailing_upper)
    return leading, trailing


def _shift_start_fields(items, *, shift: float, field: str) -> None:
    if items is None:
        return
    if not isinstance(items, list):
        raise TypeError(f"{field} must be a list for context sampling")
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field}[{idx}] must be a mapping for context sampling")
        if "start" not in item:
            raise ValueError(f"{field}[{idx}] is missing start for context sampling")
        shifted_start = _finite_nonnegative_number(item["start"], field=f"{field}[{idx}].start") + shift
        if shifted_start < -1e-5:
            raise ValueError(f"{field}[{idx}].start becomes negative after context sampling: {shifted_start}")
        item["start"] = max(0.0, shifted_start)


def _shift_and_filter_context_intervals(
    items,
    *,
    shift: float,
    clip_duration: float,
    field: str,
) -> None:
    """Shift signed contextual intervals and retain only full in-crop intervals."""
    if items is None:
        return
    if not isinstance(items, list):
        raise TypeError(f"{field} must be a list for context sampling")
    clip_duration = _finite_nonnegative_number(clip_duration, field="sampled_clip.duration")
    tolerance = 1e-5
    retained = []
    for idx, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field}[{idx}] must be a mapping for context sampling")
        if "start" not in item:
            raise ValueError(f"{field}[{idx}] is missing start for context sampling")
        if "duration" not in item:
            raise ValueError(f"{field}[{idx}] is missing duration for context sampling")
        start = _finite_number(item["start"], field=f"{field}[{idx}].start")
        duration = _finite_nonnegative_number(item["duration"], field=f"{field}[{idx}].duration")
        shifted_start = start + shift
        shifted_end = shifted_start + duration
        if shifted_start < -tolerance or shifted_end > clip_duration + tolerance:
            continue
        item["start"] = max(0.0, shifted_start)
        item["duration"] = min(duration, max(0.0, clip_duration - item["start"]))
        retained.append(item)
    items[:] = retained


def _sample_lean_multiturn_context(
    data: Mapping,
    *,
    recording_duration: float,
    context_sampling: Mapping,
    seed: int,
) -> dict:
    """Return a copied v2 row with a sampled real-audio crop and shifted relative timestamps."""
    data = copy.deepcopy(dict(data))
    sample_id = data.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("context_sampling requires a non-empty sample_id")
    curation = data.get("curation")
    if not isinstance(curation, Mapping) or curation.get("schema_version") != LEAN_MULTI_TURN_SCHEMA_VERSION:
        schema_version = curation.get("schema_version") if isinstance(curation, Mapping) else None
        raise ValueError(
            f"context_sampling is supported only for {LEAN_MULTI_TURN_SCHEMA_VERSION}; "
            f"sample {sample_id!r} has schema {schema_version!r}"
        )
    audio_context = curation.get("audio_context")
    physical_audio = curation.get("physical_audio")
    target = curation.get("target")
    if not all(isinstance(item, Mapping) for item in (audio_context, physical_audio, target)):
        raise ValueError(f"Sample {sample_id!r} is missing v2 audio_context, physical_audio, or target metadata")
    if physical_audio.get("stores_all_available_transcript_bounded_context") is not True:
        raise ValueError(f"Sample {sample_id!r} physical member does not guarantee all available context")

    declared_physical_duration = _finite_nonnegative_number(
        physical_audio.get("duration"), field="physical_audio.duration"
    )
    if abs(declared_physical_duration - recording_duration) > 0.2:
        raise ValueError(
            f"Sample {sample_id!r} declares physical duration {declared_physical_duration}, "
            f"but decoded tar member duration is {recording_duration}"
        )

    old_offset = _finite_nonnegative_number(data.get("offset", 0.0), field="manifest.offset")
    old_duration = _finite_nonnegative_number(data.get("duration"), field="manifest.duration")
    current_leading = _finite_nonnegative_number(audio_context.get("leading_sil"), field="audio_context.leading_sil")
    current_trailing = _finite_nonnegative_number(
        audio_context.get("trailing_sil"), field="audio_context.trailing_sil"
    )
    available_leading = _finite_nonnegative_number(
        audio_context.get("max_available_leading_sil"), field="audio_context.max_available_leading_sil"
    )
    available_trailing = _finite_nonnegative_number(
        audio_context.get("max_available_trailing_sil"), field="audio_context.max_available_trailing_sil"
    )
    source_start = _finite_nonnegative_number(physical_audio.get("source_start"), field="physical_audio.source_start")
    source_audio_offset = _finite_nonnegative_number(
        data.get("source_audio_offset"), field="manifest.source_audio_offset"
    )
    if abs(source_audio_offset - (source_start + old_offset)) > 0.2:
        raise ValueError(
            f"Sample {sample_id!r} source_audio_offset={source_audio_offset} is inconsistent with "
            f"physical source_start + offset={source_start + old_offset}"
        )

    regions = target.get("utterance_regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError(f"Sample {sample_id!r} has no target utterance_regions")
    first_start = _finite_nonnegative_number(regions[0].get("start"), field="target.utterance_regions[0].start")
    last_start = _finite_nonnegative_number(regions[-1].get("start"), field="target.utterance_regions[-1].start")
    last_duration = _finite_nonnegative_number(
        regions[-1].get("duration"), field="target.utterance_regions[-1].duration"
    )
    last_end = last_start + last_duration
    tolerance = 1e-5
    if abs(first_start - current_leading) > tolerance:
        raise ValueError(
            f"Sample {sample_id!r} leading_sil={current_leading} does not match first target start={first_start}"
        )
    if abs(old_duration - last_end - current_trailing) > tolerance:
        raise ValueError(
            f"Sample {sample_id!r} trailing_sil={current_trailing} does not match "
            f"duration-last_target_end={old_duration - last_end}"
        )

    target_start_in_member = old_offset + first_start
    target_end_in_member = old_offset + last_end
    physical_leading = target_start_in_member
    physical_trailing = recording_duration - target_end_in_member
    if available_leading > physical_leading + 0.2 or available_trailing > physical_trailing + 0.2:
        raise ValueError(f"Sample {sample_id!r} context availability exceeds physical member bounds")
    available_leading = min(available_leading, physical_leading)
    available_trailing = min(available_trailing, physical_trailing)

    leading_high = available_leading
    trailing_high = available_trailing
    if leading_high + tolerance < context_sampling["min_leading_s"]:
        raise ValueError(
            f"Sample {sample_id!r} has only {leading_high}s available leading context, below "
            f"context_sampling.min_leading_s={context_sampling['min_leading_s']}"
        )
    if trailing_high + tolerance < context_sampling["min_trailing_s"]:
        raise ValueError(
            f"Sample {sample_id!r} has only {trailing_high}s available trailing context, below "
            f"context_sampling.min_trailing_s={context_sampling['min_trailing_s']}"
        )
    leading_low = min(context_sampling["min_leading_s"], leading_high)
    trailing_low = min(context_sampling["min_trailing_s"], trailing_high)
    core_duration = last_end - first_start
    total_budget = context_sampling["max_duration_s"] - core_duration
    stable_seed = int.from_bytes(
        hashlib.blake2b(f"{seed}\0{sample_id}".encode("utf-8"), digest_size=8).digest(), "big"
    )
    leading, trailing = _uniform_context_pair(
        random.Random(stable_seed),
        leading_low=leading_low,
        leading_high=leading_high,
        trailing_low=trailing_low,
        trailing_high=trailing_high,
        total_budget=total_budget,
    )

    new_offset = target_start_in_member - leading
    new_duration = core_duration + leading + trailing
    new_end = new_offset + new_duration
    if new_offset < -tolerance or new_end > recording_duration + tolerance:
        raise ValueError(
            f"Sample {sample_id!r} sampled crop [{new_offset}, {new_end}] exceeds "
            f"physical duration {recording_duration}"
        )
    if new_duration > context_sampling["max_duration_s"] + tolerance:
        raise ValueError(f"Sample {sample_id!r} sampled duration {new_duration} exceeds max_duration_s")

    shift = old_offset - new_offset
    components = target.get("components")
    if not isinstance(components, list):
        raise TypeError(f"Sample {sample_id!r} target.components must be a list")
    for component_idx, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise TypeError(f"Sample {sample_id!r} target.components[{component_idx}] must be a mapping")
        _shift_start_fields(
            component.get("fragments"), shift=shift, field=f"target.components[{component_idx}].fragments"
        )
    _shift_start_fields(regions, shift=shift, field="target.utterance_regions")
    _shift_start_fields(target.get("pause_regions", []), shift=shift, field="target.pause_regions")

    other_speaker = curation.get("other_speaker")
    if other_speaker is not None:
        if not isinstance(other_speaker, Mapping):
            raise TypeError(f"Sample {sample_id!r} other_speaker must be a mapping")
        other_fragments = other_speaker.get("fragments", [])
        _shift_and_filter_context_intervals(
            other_fragments,
            shift=shift,
            clip_duration=new_duration,
            field="other_speaker.fragments",
        )
        for fragment_idx, fragment in enumerate(other_fragments):
            _shift_and_filter_context_intervals(
                fragment.get("overlap_with_target_active", []),
                shift=shift,
                clip_duration=new_duration,
                field=f"other_speaker.fragments[{fragment_idx}].overlap_with_target_active",
            )

    data["offset"] = max(0.0, new_offset)
    data["duration"] = new_duration
    data["source_audio_offset"] = source_start + data["offset"]
    audio_context["leading_sil"] = leading
    audio_context["trailing_sil"] = trailing
    selection = audio_context.get("selection")
    selection = dict(selection) if isinstance(selection, Mapping) else {}
    selection.update(
        {
            "policy": "dataloader_uniform_available_transcript_bounded",
            "selected_leading_s": leading,
            "selected_trailing_s": trailing,
            "max_total_duration_s": context_sampling["max_duration_s"],
        }
    )
    audio_context["selection"] = selection
    return data


class LazyNeMoIterator:
    """
    ``LazyNeMoIterator`` reads a NeMo (non-tarred) JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text" (overridable with ``text_field`` argument)

    Specially supported keys are:
    - [recommended] "sampling_rate" allows us to provide a valid Lhotse
     ``Recording`` object without checking the audio file
    - "offset" for partial recording reads
    - "lang" is mapped to Lhotse superivsion's language (overridable with ``lang_field`` argument)

    Every other key found in the manifest will be attached to Lhotse Cut and accessible via ``cut.custom[key]``.

    .. caution:: We will perform some I/O (as much as required by soundfile.info) to discover the sampling rate
        of the audio file. If this is not acceptable, convert the manifest to Lhotse format which contains
        sampling rate info. For pure metadata iteration purposes we also provide a ``metadata_only`` flag that
        will create only partially valid Lhotse objects (with metadata related to sampling rate / num samples missing).

    Example::

        >>> cuts = lhotse.CutSet(LazyNeMoIterator("nemo_manifests/train.json"))

    We allow attaching custom metadata to cuts from files other than the manifest via ``extra_fields`` argument.
    In the example below, we'll iterate file "questions.txt" together with the manifest and attach each line
    under ``cut.question`` using the field type ``text_iter``::

        >>> cuts = lhotse.CutSet(LazyNeMoIterator(
        ...     "nemo_manifests/train.json",
        ...     extra_fields=[{"type": "text_iter", "name": "question", "path": "questions.txt"}],
        ... ))

    We also support random sampling of lines with field type ``text_sample``::

        >>> cuts = lhotse.CutSet(LazyNeMoIterator(
        ...     "nemo_manifests/train.json",
        ...     extra_fields=[{"type": "text_sample", "name": "question", "path": "questions.txt"}],
        ... ))
    """

    def __init__(
        self,
        path: str | Path | list[str],
        text_field: str = "text",
        lang_field: str = "lang",
        metadata_only: bool = False,
        shuffle_shards: bool = False,
        shard_seed: int | Literal["randomized", "trng"] = "trng",
        extra_fields: list[dict[str, str]] | None = None,
    ) -> None:
        self.path = path
        self.shuffle_shards = shuffle_shards
        self.shard_seed = shard_seed
        paths = expand_sharded_filepaths(path)

        if len(paths) == 1:
            self.source = LazyJsonlIterator(paths[0])
        else:
            self.source = LazyIteratorChain(
                *(LazyJsonlIterator(p) for p in paths), shuffle_iters=self.shuffle_shards, seed=self.shard_seed
            )
        self.text_field = text_field
        self.lang_field = lang_field
        self.metadata_only = metadata_only
        self.extra_fields = extra_fields
        validate_extra_fields(self.extra_fields)

    def __iter__(self) -> Generator[Cut, None, None]:
        seed = resolve_seed(self.shard_seed)
        # Propagate the random seed
        extra_fields = [ExtraField.from_dict({"seed": seed, **field_cfg}) for field_cfg in self.extra_fields or ()]
        for data in self.source:
            # filter out entries with valid "_skipme" values.
            if data.get("_skipme", False):
                continue
            audio_path = get_full_path(str(data.pop("audio_filepath")), str(self.path), force_cache=False)
            duration = data.pop("duration")
            offset = data.pop("offset", None)
            cut = self._create_cut(
                audio_path=audio_path, offset=offset, duration=duration, sampling_rate=data.pop("sampling_rate", None)
            )
            # Note that start=0 and not start=offset because supervision's start if relative to the
            # start of the cut; and cut.start is already set to offset
            cut.supervisions.append(
                SupervisionSegment(
                    id=cut.id,
                    recording_id=cut.recording_id,
                    start=0,
                    duration=cut.duration,
                    channel=cut.channel,
                    text=data.get(self.text_field),
                    language=data.get(self.lang_field),
                )
            )
            cut.custom = data
            for extra_field in extra_fields:
                extra_field.attach_to(cut)
            yield cut

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)

    def _create_cut(
        self,
        audio_path: str,
        offset: float,
        duration: float,
        sampling_rate: int | None = None,
    ) -> Cut:
        if not self.metadata_only:
            recording = self._create_recording(audio_path, duration, sampling_rate)
            cut = recording.to_cut()
            if offset is not None:
                cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
                cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"
        else:
            # Only metadata requested.
            # We'll provide accurate metadata for Cut but inaccurate metadata for Recording to avoid
            # incurring IO penalty (note that Lhotse manifests contain more information than
            # NeMo manifests, so for actual dataloading we have to fill it using the audio file).
            sr = ifnone(sampling_rate, 16000)  # fake sampling rate
            offset = ifnone(offset, 0.0)
            cut = MonoCut(
                id=audio_path,
                start=offset,
                duration=duration,
                channel=0,
                supervisions=[],
                recording=Recording(
                    id=audio_path,
                    sources=[AudioSource(type="dummy", channels=[0], source="")],
                    sampling_rate=sr,
                    duration=offset + duration,
                    num_samples=compute_num_samples(offset + duration, sr),
                ),
            )
        return cut

    def _create_recording(
        self,
        audio_path: str,
        duration: float,
        sampling_rate: int | None = None,
    ) -> Recording:
        if sampling_rate is not None:
            # TODO(pzelasko): It will only work with single-channel audio in the current shape.

            source_type = "url" if is_datastore_path(audio_path) else "file"
            return Recording(
                id=audio_path,
                sources=[AudioSource(type=source_type, channels=[0], source=audio_path)],
                sampling_rate=sampling_rate,
                num_samples=compute_num_samples(duration, sampling_rate),
                duration=duration,
                channel_ids=[0],
            )
        else:
            return Recording.from_file(audio_path)


class LazyNeMoTarredIterator:
    r"""
    ``LazyNeMoTarredIterator`` reads a NeMo tarred JSON manifest and converts it on the fly to an ``Iterable[Cut]``.
    It's used to create a ``lhotse.CutSet``.

    Currently, it requires the following keys in NeMo manifests:
    - "audio_filepath"
    - "duration"
    - "text" (overridable with text_field argument)
    - "shard_id"

    Specially supported keys are:
    - "lang" is mapped to Lhotse superivsion's language (overridable with ``lang_field`` argument)

    Every other key found in the manifest will be attached to Lhotse Cut and accessible via ``cut.custom[key]``.

    Args ``manifest_path`` and ``tar_paths`` can be either a path/string to a single file, or a string in NeMo format
    that indicates multiple paths (e.g. "[[data/bucket0/tarred_audio_paths.json],[data/bucket1/...]]").
    We discover shard ids from sharded tar and json files by parsing the input specifier/path and
    searching for the following pattern: ``(manifest|audio)[^/]*_(\d+)[^/]*\.(json|tar)``.
    It allows filenames such as ``manifest_0.json``, ``manifest_0_normalized.json``, ``manifest_normalized_0.json``,
    ``manifest_0.jsonl.gz``, etc. (anologusly the same applies to tar files).

    We also support generalized input specifiers that imitate webdataset's pipes (also very similar to Kaldi's pipes).
    These are arbitrary shell commands to be lazily executed which yield manifest or tar audio contents.
    For example, ``tar_paths`` can be set to ``pipe:ais get ais://my-bucket/audio_{0..127}.tar -``
    to indicate that we want to read tarred audio data from shards on an AIStore bucket.
    This can be used for other cloud storage APIs such as S3, GCS, etc.
    The same mechanism applies to ``manifest_path``.

    If your data has been filtered so that the JSON manifests refer to just a subset of recordings,
    set ``skip_missing_manifest_entries` to ``True``.
    This will still read the tar files sequentially (very fast) and discard the audio files that
    are not present in the corresponding manifest.

    The ``shard_seed`` argument is used to seed the RNG shuffling the shards.
    By default, it's ``trng`` which samples a seed number from OS-provided TRNG (see Python ``secrets`` module).
    Seed is resolved lazily so that every dataloading worker may sample a different one.
    Override with an integer value for deterministic behaviour and consult Lhotse documentation for details:
    https://lhotse.readthedocs.io/en/latest/datasets.html#handling-random-seeds

    ``context_sampling`` is an opt-in policy for ``lean_multi_turn_v2`` rows whose tar member stores all
    transcript-bounded context. It samples each side from its configured minimum through that row's full
    advertised availability before the in-memory subset is created, shifts row-relative timestamps, and enforces
    a maximum total cut duration. It is intentionally unsupported by the AIS batch path because that path does
    not decode the physical member before constructing the cut.

    Set ``slice_length`` to enable random slicing mode: for each shard, we'll randomly select an offset K
    and skip the first K examples (but will actually read them first). Then, we'll yield only ``slice_length``
    examples. This setting can improve the sampling randomness when there are many datasets with many shards
    but only a limited run time.

    Example of CutSet with inter-shard shuffling enabled::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path=["nemo_manifests/sharded_manifests/manifest_0.json", ...],
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     shuffle_shards=True,
        ... ))

    We allow attaching custom metadata to cuts from files other than the manifest via ``extra_fields`` argument.
    In the example below, we'll iterate file "questions.txt" together with the manifest and attach each line
    under ``cut.question`` using the field type ``text_iter``::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path=["nemo_manifests/sharded_manifests/manifest_0.json", ...],
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     extra_fields=[{"type": "text_iter", "name": "question", "path": "questions.txt"}],
        ... ))

    We also support random sampling of lines with field type ``text_sample``::

        >>> cuts = lhotse.CutSet(LazyNeMoTarredIterator(
        ...     manifest_path=["nemo_manifests/sharded_manifests/manifest_0.json", ...],
        ...     tar_paths=["nemo_manifests/audio_0.tar", ...],
        ...     extra_fields=[{"type": "text_sample", "name": "question", "path": "questions.txt"}],
        ... ))
    """

    def __init__(
        self,
        manifest_path: str | Path | list[str],
        tar_paths: str | list,
        shuffle_shards: bool = False,
        shard_seed: int | Literal["trng", "randomized"] = "trng",
        text_field: str = "text",
        lang_field: str = "lang",
        skip_missing_manifest_entries: bool = False,
        extra_fields: list[dict[str, str]] | None = None,
        slice_length: int = None,
        context_sampling: Mapping | None = None,
    ) -> None:
        self.skip_missing_manifest_entries = skip_missing_manifest_entries
        self.shard_id_to_manifest: dict[int, Iterable[dict]]
        self.paths = expand_sharded_filepaths(manifest_path)
        if len(self.paths) == 1:
            logging.warning(
                f"You are using Lhotse dataloading for tarred audio with a non-sharded manifest. "
                f"This will incur significant memory overhead. To prevent this, please shard file "
                f"'{self.paths[0]}' using 'scripts/speech_recognition/convert_to_tarred_audio_dataset.py' "
                f"WITHOUT '--no_shard_manifest'"
            )
            self.source = LazyJsonlIterator(self.paths[0])
            self.shard_id_to_manifest = groupby("shard_id", self.source)
        else:
            json_pattern = re.compile(r"manifest[^/]*_(\d+)[^/]*\.json")
            shard_ids = []
            for p in self.paths:
                m = json_pattern.search(p)
                assert m is not None, (
                    f"Cannot determine shard_id from manifest input specified: "
                    f"we searched with regex '{json_pattern.pattern}' in input '{p}'"
                )
                shard_ids.append(int(m.group(1)))
            self.shard_id_to_manifest_path = dict(zip(shard_ids, self.paths))
            self.shard_id_to_manifest = {sid: LazyJsonlIterator(p) for sid, p in zip(shard_ids, self.paths)}
            self.source = LazyIteratorChain(*self.shard_id_to_manifest.values())

        self.tar_paths = expand_sharded_filepaths(tar_paths)
        tar_pattern = re.compile(r"audio[^/]*_(\d+)[^/]*\.tar")
        shard_ids = []
        for p in self.tar_paths:
            m = tar_pattern.search(p)
            assert m is not None, (
                f"Cannot determine shard_id from tar input specifier: "
                f"we searched with regex '{tar_pattern.pattern}' in input '{p}'"
            )
            shard_ids.append(int(m.group(1)))
        self.shard_id_to_tar_path = dict(zip(shard_ids, self.tar_paths))

        self.shuffle_shards = shuffle_shards
        self.shard_seed = shard_seed
        self.text_field = text_field
        self.lang_field = lang_field
        self.extra_fields = extra_fields
        self.slice_length = slice_length
        self.context_sampling = _validate_context_sampling_config(context_sampling)
        self.epoch = 0
        self._validate()
        self.use_ais_get_batch = os.environ.get("USE_AIS_GET_BATCH", "False").lower() == "true"

    def to_shards(self) -> List["LazyNeMoTarredIterator"]:
        """Convert this iterator to a list of separate iterators for each shard."""
        if len(self.paths) == 1:
            # Cannot do that if the JSON manifest is a single file for all shards;
            # just return self.
            return [self]
        else:
            return [
                LazyNeMoTarredIterator(
                    manifest_path=path,
                    tar_paths=tarpath,
                    shuffle_shards=False,
                    shard_seed=self.shard_seed,
                    text_field=self.text_field,
                    lang_field=self.lang_field,
                    context_sampling=self.context_sampling,
                )
                for path, tarpath in zip(self.paths, self.shard_id_to_tar_path.values())
            ]

    def _validate(self) -> None:
        shard_ids_tars = set(self.shard_id_to_tar_path)
        shard_ids_manifest = set(self.shard_id_to_manifest)
        assert shard_ids_tars == shard_ids_manifest, (
            f"Mismatch between shard IDs. Details:\n"
            f"* JSON manifest(s) {self.paths}\n"
            f"* Tar files: {self.tar_paths}\n"
            f"* JSON manifest(s) indicate(s) IDs: {sorted(shard_ids_manifest)}\n"
            f"* Tar path(s) indicate(s) IDs: {sorted(shard_ids_tars)}\n"
        )
        validate_extra_fields(self.extra_fields)

    def _get_seed(self) -> int:
        return resolve_seed(self.shard_seed) + self.epoch

    @property
    def shard_ids(self) -> List[int]:
        return sorted(self.shard_id_to_manifest.keys())

    def _iter_batch_for_ais_get_batch(
        self, tar_path, shard_manifest, manifest_path, rng, extra_fields
    ) -> Generator[Cut, None, None]:
        """
        Iterator for batch reading mode (AIS get batch).
        Yields cuts with URL-based recordings without opening tar files.
        """
        # Calculate slice offset for random skipping
        total_entries = sum(len(entries) for entries in shard_manifest.values())
        slice_offset = (
            rng.randint(0, total_entries - self.slice_length)
            if self.slice_length is not None and self.slice_length < total_entries
            else -1
        )
        cntr = 0
        entries_processed = 0

        for audio_filename, manifest_entries in shard_manifest.items():
            for data in manifest_entries:
                # Skip entries if we haven't reached the slice offset yet
                if entries_processed < slice_offset:
                    entries_processed += 1
                    continue
                # Stop if we've reached the slice length limit
                elif cntr == self.slice_length:
                    break

                # filter out entries with valid "_skipme" values.
                if data.get("_skipme", False):
                    entries_processed += 1
                    continue

                # Construct URL: tar_path/audio_filename
                audio_url = f"{tar_path.rstrip('/')}/{audio_filename.lstrip('/')}"

                # Get metadata from manifest
                duration = data.get("duration")
                if duration is None:
                    logging.warning(f"Skipping '{audio_filename}' - missing duration in manifest")
                    entries_processed += 1
                    continue

                offset = data.get("offset", 0.0)
                sampling_rate = data.get("sampling_rate", 16000)  # default to 16kHz if not specified

                # Create URL-based recording
                recording = Recording(
                    id=audio_filename,
                    sources=[AudioSource(type="url", channels=[0], source=audio_url)],
                    sampling_rate=sampling_rate,
                    num_samples=compute_num_samples(duration, sampling_rate),
                    duration=duration,
                )

                # Create cut from recording (audio will be loaded lazily from URL when needed)
                cut = recording.to_cut()
                if offset > 0:
                    cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
                    cut.id = f"{cut.id}-{round(offset * 1e2):06d}-{round(duration * 1e2):06d}"

                # Add supervision (transcript metadata)
                cut.supervisions.append(
                    SupervisionSegment(
                        id=cut.id,
                        recording_id=cut.recording_id,
                        start=0,
                        duration=cut.duration,
                        text=data.get(self.text_field),
                        language=data.get(self.lang_field),
                    )
                )

                # Attach custom fields and metadata
                cut.custom = _to_custom_attr_dict(data)
                cut.manifest_origin = manifest_path
                cut.tar_origin = tar_path
                for extra_field in extra_fields:
                    extra_field.attach_to(cut)

                cntr += 1
                entries_processed += 1
                yield cut

            # Break outer loop if we've reached the slice length limit
            if cntr == self.slice_length:
                break

    def _iter_sequential(
        self, tar_path, shard_manifest, manifest_path, rng
    ) -> Generator[tuple[dict, bytes], None, None]:
        slice_offset = (
            rng.randint(0, len(shard_manifest) - self.slice_length)
            if self.slice_length is not None and self.slice_length < len(shard_manifest)
            else -1
        )
        cntr = 0
        with tarfile.open(fileobj=open_best(tar_path, mode="rb"), mode="r|*") as tar:
            for idx, tar_info in enumerate(tar):
                if idx < slice_offset:
                    continue
                elif cntr == self.slice_length:
                    break
                try:
                    data = shard_manifest[tar_info.name]
                    raw_audio = tar.extractfile(tar_info).read()
                    yield data, raw_audio, tar_info
                    cntr += 1
                except KeyError as e:
                    if self.skip_missing_manifest_entries:
                        continue
                    else:
                        raise RuntimeError(
                            f"Mismatched entry between JSON manifest ('{manifest_path}') and tar file ('{tar_path}'). "
                            f"Cannot locate JSON entry for tar file '{tar_info.name}'"
                        ) from e

    def __iter__(self) -> Generator[Cut, None, None]:
        shard_ids = self.shard_ids

        seed = self._get_seed()
        rng = random.Random(seed)
        if self.shuffle_shards:
            rng.shuffle(shard_ids)

        # Propagate the random seed
        extra_fields = [ExtraField.from_dict({"seed": seed, **field_cfg}) for field_cfg in self.extra_fields or ()]

        # Handle NeMo tarred manifests with offsets.
        # They have multiple JSONL entries where audio paths end with '-sub1', '-sub2', etc. for each offset.
        offset_pattern = re.compile(r'^(?P<stem>.+)(?P<sub>-sub\d+)(?P<ext>\.\w+)?$')

        for sid in shard_ids:
            manifest_path = self.shard_id_to_manifest_path[sid] if len(self.paths) > 1 else self.paths[0]

            def basename(d: dict) -> str:
                return (
                    m.group("stem") + ifnone(m.group("ext"), "")
                    if (m := offset_pattern.match(k := d["audio_filepath"])) is not None
                    else k
                )

            shard_manifest: dict[str, list[dict]] = groupby(basename, self.shard_id_to_manifest[sid])
            tar_path = self.shard_id_to_tar_path[sid]

            if self.use_ais_get_batch:
                if self.context_sampling is not None:
                    raise RuntimeError("context_sampling is not supported with USE_AIS_GET_BATCH=true")
                # Use batch reading mode - URL-based recordings without opening tar files
                yield from self._iter_batch_for_ais_get_batch(
                    tar_path, shard_manifest, manifest_path, rng, extra_fields
                )
                continue
            try:
                for data, raw_audio, tar_info in self._iter_sequential(tar_path, shard_manifest, manifest_path, rng):
                    try:
                        meta = soundfile.info(BytesIO(raw_audio))
                    except Exception:
                        logging.warning(f"Skipped corrupted file '{tar_info.path}' in {tar_path=}.")
                        continue
                    recording = Recording(
                        id=tar_info.path,
                        sources=[AudioSource(type="memory", channels=list(range(meta.channels)), source=raw_audio)],
                        sampling_rate=int(meta.samplerate),
                        num_samples=meta.frames,
                        duration=meta.duration,
                    )
                    cuts_for_recording = []
                    for data in sorted(shard_manifest[tar_info.name], key=lambda d: d["audio_filepath"]):
                        # filter out entries with valid "_skipme" values.
                        if data.get("_skipme", False):
                            continue
                        if self.context_sampling is not None:
                            data = _sample_lean_multiturn_context(
                                data,
                                recording_duration=recording.duration,
                                context_sampling=self.context_sampling,
                                seed=seed,
                            )
                        # Cut the recording into corresponding segment and discard audio data outside the segment.
                        cut = make_cut_with_subset_inmemory_recording(
                            recording, offset=data.get("offset", 0.0), duration=data.get("duration")
                        )
                        cut.supervisions.append(
                            SupervisionSegment(
                                id=cut.id,
                                recording_id=cut.recording_id,
                                start=0,
                                duration=cut.duration,
                                text=data.get(self.text_field),
                                language=data.get(self.lang_field),
                            )
                        )
                        cut.custom = _to_custom_attr_dict(data)
                        cut.manifest_origin = manifest_path
                        cut.tar_origin = tar_path
                        for extra_field in extra_fields:
                            extra_field.attach_to(cut)
                        cuts_for_recording.append(cut)
                    del recording  # free the memory - helps with very large audio files
                    del raw_audio
                    yield from cuts_for_recording
            except tarfile.ReadError:
                logging.warning(
                    f"Skipping tar file due to read errors (unstable storage or bad file?): {tar_path=}",
                )

        self.epoch += 1

    def __len__(self) -> int:
        return len(self.source)

    def __add__(self, other):
        return LazyIteratorChain(self, other)


def make_cut_with_subset_inmemory_recording(
    recording: Recording, offset: float = 0.0, duration: float | None = None
) -> Cut:
    """
    This method is built specifically to optimize CPU memory usage during dataloading
    when reading tarfiles containing very long recordings (1h+).
    Normally each cut would hold a reference to the long in-memory recording and load
    the necessary subset of audio (there wouldn't be a separate copy of the long recording for each cut).
    This is fairly efficient already, but we don't actually need to hold the unused full recording in memory.
    Instead, we re-create each cut so that it only holds a reference to the subset of recording necessary.
    This allows us to discard unused data which would otherwise be held in memory as part of sampling buffering.
    """

    # Fast path: no offset and (almost) matching duration (within 200ms; leeway for different audio codec behavior).
    cut = recording.to_cut()
    if offset == 0.0 and duration is None or abs(duration - recording.duration) < 0.2:
        return cut

    # Otherwise, apply the memory optimization.
    try:
        cut = cut.truncate(offset=offset, duration=duration, preserve_id=True)
    except Exception as e:
        raise RuntimeError(
            f"Lhotse cut.truncate failed with offset={offset}, duration={duration}, recording={recording}: {e}"
        ) from e

    audiobytes = BytesIO()
    LibsndfileBackend().save_audio(audiobytes, cut.load_audio(), sampling_rate=cut.sampling_rate, format="wav")
    audiobytes.seek(0)
    new_recording = Recording(
        id=recording.id,
        sampling_rate=recording.sampling_rate,
        num_samples=cut.num_samples,
        duration=cut.duration,
        sources=[
            AudioSource(
                type="memory",
                channels=recording.channel_ids,
                source=audiobytes.getvalue(),
            )
        ],
    )
    return new_recording.to_cut()


class ExtraField:
    TYPE = None
    SUPPORTED_TYPES = {}

    def attach_to(self, cut):
        raise NotImplementedError()

    def __init_subclass__(cls, **kwargs):
        if cls.__name__ not in ExtraField.SUPPORTED_TYPES:
            ExtraField.SUPPORTED_TYPES[cls.TYPE] = cls
        super().__init_subclass__(**kwargs)

    @staticmethod
    def from_dict(data: dict) -> "ExtraField":
        assert data["type"] in ExtraField.SUPPORTED_TYPES, f"Unknown transform type: {data['type']}"
        return ExtraField.SUPPORTED_TYPES[data["type"]](**{k: v for k, v in data.items() if k != 'type'})

    @classmethod
    def is_supported(cls, field_type: str) -> bool:
        return field_type in cls.SUPPORTED_TYPES

    @classmethod
    def supported_types(cls) -> list[str]:
        return list(cls.SUPPORTED_TYPES)


class TextIteratorExtraField(ExtraField):
    TYPE = "text_iter"

    def __init__(self, name: str, path: str, seed=None):
        self.name = name
        self.path = path
        self.iterator = None

    def _maybe_init(self):
        if self.iterator is None:
            self.iterator = iter(map(str.strip, open_best(self.path)))

    def attach_to(self, cut):
        self._maybe_init()
        try:
            attached_value = next(self.iterator)
        except StopIteration:
            raise RuntimeError(f"Not enough lines in file {self.path} to attach to cuts under field {self.name}.")
        setattr(cut, self.name, attached_value)
        return cut


class TextSampleExtraField(ExtraField):
    TYPE = "text_sample"

    def __init__(self, name: str, path: str, seed: int | str):
        self.name = name
        self.path = path
        self.seed = seed
        self.population = None
        self.rng = None

    def _maybe_init(self):
        if self.population is None:
            self.population = list(map(str.strip, open_best(self.path)))
            self.rng = random.Random(resolve_seed(self.seed))

    def attach_to(self, cut):
        self._maybe_init()
        attached_value = self.rng.choice(self.population)
        setattr(cut, self.name, attached_value)
        return cut


def validate_extra_fields(extra_fields):
    if extra_fields is None:
        return
    assert isinstance(
        extra_fields, Sequence
    ), f"The argument provided to 'extra_fields' must be a list of dicts. We received {extra_fields=}"
    for field in extra_fields:
        assert isinstance(
            field, Mapping
        ), f"Each item in 'extra_fields' must be a dict. We received {field=} in {extra_fields=}"
        field_type = field.get("type")
        assert ExtraField.is_supported(field_type), (
            f"Each item in 'extra_fields' must contain a 'type' field with one of "
            f"the supported values ({ExtraField.supported_types()}). "
            f"We got {field_type=} in {extra_fields=}"
        )
        assert "name" in field, (
            f"Each item in 'extra_fields' must contain a 'name' field so that the field is available under cut.<name>."
            f"We found {field=} in {extra_fields=}"
        )


def expand_sharded_filepaths(paths: str | Path | list[str]) -> list[str]:
    # local import to avoid circular imports
    from nemo.collections.asr.data.audio_to_text import expand_sharded_filepaths as _expand_sharded_filepaths

    if isinstance(paths, Path):
        paths = str(paths)

    return _expand_sharded_filepaths(paths, shard_strategy="replicate", world_size=1, global_rank=0)


def _to_custom_attr_dict(d: dict, _excluded_fields: set[str] = {"duration", "audio_filepath"}) -> dict:
    return {k: v for k, v in d.items() if k not in _excluded_fields}


class LazyParquetIterator:
    """
    LazyParquetIterator reads a Parquet file (local or remote) and yields Lhotse Cut objects.
    It streams data using PyArrow's iter_batches to avoid loading the full file into memory.

    Args:
        path (str | Path): Path to the .parquet file.
        audio_field (str): Name of the column containing audio bytes (default: "audio").
        text_field (str): Name of the column containing transcript (default: "text").
        duration_field (str): Name of the column containing duration (default: "duration").
        lang_field (str): Name of the column containing language (default: "lang").
        sampling_rate (int): Fallback sampling rate if not found in metadata (default: 16000).
    """

    def __init__(
        self,
        path: str | Path,
        audio_field: str = "audio",
        text_field: str = "text",
        duration_field: str = "duration",
        lang_field: str = "lang",
        sampling_rate: int = 16000,
    ) -> None:
        # SAFETY CHECK: Ensure pyarrow is actually installed
        if not HAVE_PYARROW:
            raise ImportError(
                "PyArrow is required to read Parquet manifests. Please install it using: pip install pyarrow"
            )

        self.path = str(path)
        self.audio_field = audio_field
        self.text_field = text_field
        self.duration_field = duration_field
        self.lang_field = lang_field
        self.sampling_rate = sampling_rate

    def __iter__(self) -> Generator[Cut, None, None]:
        # Open Parquet file in streaming mode inside __iter__
        # This ensures each DataLoader worker gets its own file handle.
        try:
            parquet_file = pq.ParquetFile(self.path)
        except Exception as e:
            raise RuntimeError(f"Failed to open Parquet file: {self.path}") from e

        # Stream batches to keep memory usage low
        for batch in parquet_file.iter_batches():
            df = batch.to_pandas()

            for idx, row in df.iterrows():
                # 1. Extract Audio Bytes
                # Handle HuggingFace format: {'bytes': b'...', 'path': '...'} or raw bytes
                audio_data = row.get(self.audio_field)
                if isinstance(audio_data, dict) and 'bytes' in audio_data:
                    audio_bytes = audio_data['bytes']
                elif isinstance(audio_data, bytes):
                    audio_bytes = audio_data
                else:
                    logging.warning(f"Skipping row {idx}: Audio column '{self.audio_field}' format unrecognized.")
                    continue

                # 2. Extract Metadata
                text = row.get(self.text_field, "")
                language = row.get(self.lang_field, None)

                # 3. Create Unique ID
                # Use 'id' column if exists, else combine filename + index
                row_id = str(row.get('id', f"{Path(self.path).stem}_{idx}"))

                # 4. Create Lhotse Recording
                try:
                    recording = Recording.from_bytes(
                        data=audio_bytes,
                        recording_id=row_id,
                    )
                except (RuntimeError, ValueError, TypeError) as e:
                    logging.warning(f"Skipping row {row_id}: Failed to decode audio bytes. {e}")
                    continue

                # 5. Create Cut
                cut = recording.to_cut()

                # Add Supervision (Transcript)
                cut.supervisions.append(
                    SupervisionSegment(
                        id=row_id,
                        recording_id=row_id,
                        start=0.0,
                        duration=cut.duration,
                        channel=0,
                        text=text,
                        language=language,
                    )
                )

                # Attach any extra metadata from the row to cut.custom
                # (Exclude the heavy audio bytes to save RAM)
                cut.custom = {k: v for k, v in row.items() if k != self.audio_field}

                yield cut
