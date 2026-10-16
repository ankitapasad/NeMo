# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Lightweight distributed sample-index exposure accounting for W&B."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


def _sample_key(sample_id: str) -> tuple[int, int]:
    digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=16).digest()
    return tuple(int.from_bytes(digest[i : i + 8], "big", signed=True) for i in (0, 8))


@dataclass(frozen=True)
class CellTotal:
    samples: int
    hours: float


def _label(value, field):
    if not isinstance(value, str) or not value.strip() or value != value.strip() or "/" in value:
        raise ValueError(f"{field} must be a nonempty label without '/' or surrounding whitespace")
    return value


def exposure_metadata(cuts) -> dict:
    """Carry explicit source/category labels and fixed row durations.

    Set data_seen_source and optionally data_seen_category in manifest metadata or
    input_cfg tags. Legacy D7/ASR naming is available only via explicit opt-in.
    """
    if not os.environ.get(DataSeenTracker.ENV_PATH):
        return {}
    mode = os.environ.get(DataSeenTracker.ENV_LABEL_MODE, "explicit")
    if mode not in ("explicit", "legacy"):
        raise ValueError("NEMO_DATA_SEEN_LABEL_MODE must be explicit or legacy")
    fields = dict(sample_ids=[], accounting_durations_s=[], mixture_sources=[], turn_buckets=[])
    for cut in cuts:
        custom = cut.custom or {}
        if "data_seen_source" in custom:
            source = _label(custom["data_seen_source"], "data_seen_source")
            category = _label(custom.get("data_seen_category", "all"), "data_seen_category")
        elif mode == "legacy" and "data_seen_category" not in custom:
            # Compatibility adapter for historical campaigns, not dataset discovery.
            category = custom.get("shared_tar_category")
            if category is None:
                if "davidai" in str(custom.get("taskname", "")).lower():
                    raise ValueError("Legacy D7 accounting requires shared_tar_category")
                source, category = "asr", "asr"
            else:
                source = custom.get("mixture_source", "d7")
                if not custom.get("sample_id"):
                    raise ValueError("Legacy D7 accounting requires the stable manifest sample_id")
            source, category = _label(source, "source"), _label(category, "category")
        else:
            raise ValueError(
                "Data counting is enabled (NEMO_DATA_SEEN_TOTALS_PATH is set), but "
                f"example {custom.get('sample_id') or cut.id!r} is missing 'data_seen_source'. "
                "Set data_seen_source under this dataset source's tags in the YAML referenced "
                "by your training data config's input_cfg (or in the source entry if input_cfg "
                "is an inline list), for example "
                "tags: {data_seen_source: d7, data_seen_category: complete_turn}, "
                "Alternatively, add data_seen_source as a top-level field in each manifest JSONL row. "
                "Use your dataset's actual labels; "
                "data_seen_category is optional and defaults to 'all'. "
                "For existing D7/ASR configurations, set NEMO_DATA_SEEN_LABEL_MODE=legacy "
                "to keep the previous naming behavior. To disable data counting, "
                "unset NEMO_DATA_SEEN_TOTALS_PATH."
            )
        sample_id = custom.get("sample_id") or str(cut.id)
        sampled_context = (
            ((custom.get("curation") or {}).get("audio_context") or {}).get("selection") or {}
        ).get("policy") == "dataloader_uniform_available_transcript_bounded"
        if sampled_context:
            if not custom.get("sample_id"):
                raise ValueError("Sampled crop accounting requires a stable manifest sample_id")
            if "data_seen_accounting_duration_s" not in custom:
                raise ValueError("Sampled crop is missing its original accounting duration")
        duration = float(custom.get("data_seen_accounting_duration_s", cut.duration))
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Invalid fixed exposure duration")
        fields["sample_ids"].append(sample_id)
        fields["accounting_durations_s"].append(duration)
        fields["mixture_sources"].append(source)
        fields["turn_buckets"].append(category)
    return fields


class DataSeenTracker:
    """Track repeated and globally unique fixed-duration sample exposure."""

    ENV_PATH = "NEMO_DATA_SEEN_TOTALS_PATH"
    ENV_INTERVAL = "NEMO_DATA_SEEN_LOG_INTERVAL"
    ENV_LABEL_MODE = "NEMO_DATA_SEEN_LABEL_MODE"

    def __init__(self, totals: dict[tuple[str, str], CellTotal], membership_sha256: str, interval: int = 100, ignored_cells=()):
        if not totals:
            raise ValueError("data-seen totals must contain at least one source/category cell")
        if interval <= 0:
            raise ValueError("data-seen logging interval must be positive")
        if len(membership_sha256) != 64 or any(c not in '0123456789abcdef' for c in membership_sha256):
            raise ValueError("data-seen tracking requires a sealed membership SHA-256")
        if any(t.samples <= 0 or not math.isfinite(t.hours) or t.hours <= 0 for t in totals.values()):
            raise ValueError("data-seen totals must be finite and positive")
        self.ignored_cells = frozenset(tuple(cell) for cell in ignored_cells)
        for source, category in (*totals, *self.ignored_cells):
            _label(source, "source")
            _label(category, "category")
        if self.ignored_cells.intersection(totals):
            raise ValueError("A data-seen cell cannot be both tracked and ignored")
        self.membership_sha256 = membership_sha256
        self.cells = tuple(sorted(totals))
        self.cell_index = {cell: idx for idx, cell in enumerate(self.cells)}
        self.totals = totals
        self.interval = int(interval)
        n = len(self.cells)
        self.base_samples = torch.zeros(n, dtype=torch.float64)
        self.base_seconds = torch.zeros(n, dtype=torch.float64)
        self.delta_samples = torch.zeros(n, dtype=torch.float64)
        self.delta_seconds = torch.zeros(n, dtype=torch.float64)
        self.unique: dict[tuple[int, int], tuple[int, float]] = {}
        self.pending: dict[tuple[int, int], tuple[int, float]] = {}

    @classmethod
    def from_env(cls) -> DataSeenTracker | None:
        path = os.environ.get(cls.ENV_PATH)
        if not path:
            return None
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != "nemo_data_seen_totals_v2":
            raise ValueError(f"Unsupported data-seen totals schema in {path}")
        totals = {}
        for name, item in payload.get("cells", {}).items():
            source, category = name.split("/", 1)
            total = CellTotal(samples=int(item["samples"]), hours=float(item["hours"]))
            if total.samples <= 0 or total.hours <= 0:
                raise ValueError(f"Invalid data-seen total for {name}: {item!r}")
            totals[(source, category)] = total
        mode = os.environ.get(cls.ENV_LABEL_MODE, "explicit")
        if mode not in ("explicit", "legacy"):
            raise ValueError("NEMO_DATA_SEEN_LABEL_MODE must be explicit or legacy")
        ignored = payload.get("ignored_cells", [])
        if not isinstance(ignored, list) or any(not isinstance(name, str) or name.count("/") != 1 for name in ignored):
            raise ValueError("ignored_cells must be a list of source/category labels")
        ignored = [tuple(name.split("/")) for name in ignored]
        if mode == "legacy" and ("asr", "asr") not in totals and ("asr", "asr") not in ignored:
            ignored.append(("asr", "asr"))
        return cls(totals, membership_sha256=payload["membership_sha256"],
                   interval=int(os.environ.get(cls.ENV_INTERVAL, "100")), ignored_cells=ignored)

    def update(self, batch) -> None:
        fields = (
            batch.sample_ids,
            batch.accounting_durations_s,
            batch.mixture_sources,
            batch.turn_buckets,
        )
        if any(value is None for value in fields):
            raise ValueError("Data-seen tracking is enabled but batch accounting metadata is absent")
        lengths = {len(value) for value in fields}
        if len(lengths) != 1:
            raise ValueError(f"Batch accounting metadata lengths disagree: {sorted(lengths)}")
        for sample_id, duration, source, category in zip(*fields):
            cell = (source, category)
            if cell not in self.cell_index:
                if cell in self.ignored_cells:
                    continue
                raise ValueError(f"Unregistered data-seen cell {source}/{category}")
            index = self.cell_index[cell]
            seconds = float(duration)
            if not math.isfinite(seconds) or seconds <= 0 or not isinstance(sample_id, str) or not sample_id:
                raise ValueError("Invalid sample identity or fixed accounting duration")
            # Manifest-row identity is explicitly namespaced across data sources
            # and categories; repeated sliding windows remain distinct rows.
            key = _sample_key(json.dumps([source, category, sample_id], separators=(",", ":")))
            value = (index, seconds)
            previous = self.pending.get(key, self.unique.get(key))
            if previous is not None and previous != value:
                raise RuntimeError("Data-seen sample identity maps to inconsistent cell/duration")
            self.delta_samples[index] += 1
            self.delta_seconds[index] += seconds
            if previous is None:
                self.pending[key] = value

    def should_sync(self, global_step: int) -> bool:
        return (int(global_step) + 1) % self.interval == 0

    def sync(self, device: torch.device) -> dict[str, float]:
        delta_samples = self.delta_samples.to(device)
        delta_seconds = self.delta_seconds.to(device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(delta_samples)
            dist.all_reduce(delta_seconds)
        # Lightning may move tensors reachable from the model to its accelerator.
        # Keep the long-lived accounting state on CPU and only use ``device`` for
        # the tiny distributed reductions required by NCCL.
        self.base_samples = self.base_samples.cpu()
        self.base_seconds = self.base_seconds.cpu()
        self.base_samples.add_(delta_samples.detach().cpu())
        self.base_seconds.add_(delta_seconds.detach().cpu())
        self.delta_samples.zero_()
        self.delta_seconds.zero_()
        self._sync_unique()
        return self.metrics()

    def _sync_unique(self) -> None:
        records = [(a, b, cell, seconds) for (a, b), (cell, seconds) in self.pending.items()]
        gathered = [records]
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, records)
        for rank_records in gathered:
            for a, b, cell, seconds in rank_records:
                key = (a, b)
                previous = self.unique.get(key)
                value = (int(cell), float(seconds))
                if previous is not None and previous != value:
                    raise RuntimeError("Data-seen sample identity maps to inconsistent cell/duration")
                self.unique[key] = value
        self.pending.clear()

    def metrics(self) -> dict[str, float]:
        unique_samples = [0] * len(self.cells)
        unique_seconds = [0.0] * len(self.cells)
        for cell, seconds in self.unique.values():
            unique_samples[cell] += 1
            unique_seconds[cell] += seconds
        metrics = {}
        for idx, (source, category) in enumerate(self.cells):
            total = self.totals[(source, category)]
            sample_suffix = f"total={total.samples:,}"
            hour_suffix = f"total={round(total.hours):,} h"
            prefix = f"{source}/{category}"
            metrics[f"data_seen/{prefix}/samples ({sample_suffix})"] = float(self.base_samples[idx])
            metrics[f"data_seen/{prefix}/hours ({hour_suffix})"] = float(self.base_seconds[idx] / 3600.0)
            metrics[f"data_unique/{prefix}/samples ({sample_suffix})"] = float(unique_samples[idx])
            metrics[f"data_unique/{prefix}/hours ({hour_suffix})"] = float(unique_seconds[idx] / 3600.0)
            metrics[f"data_unique/{prefix}/sample_coverage"] = unique_samples[idx] / total.samples
            metrics[f"data_unique/{prefix}/hour_coverage"] = (unique_seconds[idx] / 3600.0) / total.hours
        return metrics

    def state_dict(self) -> dict:
        # Model checkpoint hook must call sync on ALL ranks, including between
        # log intervals. Silently dropping local deltas here is not permitted.
        if self.pending or self.delta_samples.count_nonzero() or self.delta_seconds.count_nonzero():
            raise RuntimeError("Data-seen counters must be collectively flushed before checkpoint save")
        return {
            "schema_version": "nemo_data_seen_tracker_v2",
            "membership_sha256": self.membership_sha256,
            "ignored_cells": [list(cell) for cell in sorted(self.ignored_cells)],
            "totals": [[self.totals[cell].samples, self.totals[cell].hours] for cell in self.cells],
            "cells": [list(cell) for cell in self.cells],
            "base_samples": self.base_samples.tolist(),
            "base_seconds": self.base_seconds.tolist(),
            "unique": [[a, b, cell, seconds] for (a, b), (cell, seconds) in self.unique.items()],
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("schema_version") != "nemo_data_seen_tracker_v2":
            raise ValueError("Unsupported data-seen checkpoint state")
        if state.get("membership_sha256") != self.membership_sha256:
            raise ValueError("Data-seen checkpoint membership does not match")
        if state.get("totals") != [[self.totals[cell].samples, self.totals[cell].hours] for cell in self.cells]:
            raise ValueError("Data-seen checkpoint totals do not match")
        if tuple(tuple(cell) for cell in state["cells"]) != self.cells:
            raise ValueError("Data-seen checkpoint cells do not match configured totals")
        if "ignored_cells" in state and frozenset(tuple(cell) for cell in state["ignored_cells"]) != self.ignored_cells:
            raise ValueError("Data-seen checkpoint ignored cells do not match")
        self.base_samples = torch.tensor(state["base_samples"], dtype=torch.float64)
        self.base_seconds = torch.tensor(state["base_seconds"], dtype=torch.float64)
        self.unique = {(a, b): (cell, seconds) for a, b, cell, seconds in state["unique"]}
        self.pending.clear()
        self.delta_samples.zero_()
        self.delta_seconds.zero_()
