# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path
from typing import Mapping

import torch

from nemo.utils import logging


_VOCAB_WEIGHT_SUFFIXES = ("embed_tokens.weight", "lm_head.weight")


def _single_key_ending_with(keys, suffix: str) -> str:
    matches = [key for key in keys if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one state-dict key ending in {suffix!r}; found {matches}")
    return matches[0]


def _token_ids(model, tokens: tuple[str, ...], *, label: str) -> list[int]:
    tokenizer = getattr(model.tokenizer, "tokenizer", model.tokenizer)
    token_ids = []
    for token in tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"{label} token {token!r} must encode to one token; got {encoded}")
        token_ids.append(encoded[0])
    return token_ids


def prepare_asr_stage2_state_dict(
    model,
    source_state: Mapping[str, torch.Tensor],
    *,
    boundary_tokens: tuple[str, ...] = ("<sou>", "<eou>"),
    text_tokens: tuple[str, str] = ("<|text_start|>", "<|text_end|>"),
    expected_source_vocab_size: int = 151672,
) -> dict[str, torch.Tensor]:
    """Expand tied vocabulary weights while remapping compact-text rows around new boundaries."""

    target_state = model.state_dict()
    source_keys = set(source_state)
    target_keys = set(target_state)
    if source_keys != target_keys:
        missing = sorted(target_keys - source_keys)
        unexpected = sorted(source_keys - target_keys)
        raise RuntimeError(f"Stage-2 state-dict keys differ: missing={missing}, unexpected={unexpected}")

    mismatch_keys = [key for key in target_state if source_state[key].shape != target_state[key].shape]
    expected_mismatch_keys = {_single_key_ending_with(target_state, suffix) for suffix in _VOCAB_WEIGHT_SUFFIXES}
    if set(mismatch_keys) != expected_mismatch_keys:
        mismatch_shapes = {
            key: (tuple(source_state[key].shape), tuple(target_state[key].shape)) for key in mismatch_keys
        }
        raise RuntimeError(
            "Only the embedding and LM-head vocabulary rows may differ during stage-2 initialization; "
            f"got {mismatch_shapes}"
        )

    source_shapes = {tuple(source_state[key].shape) for key in expected_mismatch_keys}
    target_shapes = {tuple(target_state[key].shape) for key in expected_mismatch_keys}
    if len(source_shapes) != 1 or len(target_shapes) != 1:
        raise RuntimeError(
            f"Tied vocabulary weights disagree: source_shapes={source_shapes}, target_shapes={target_shapes}"
        )
    source_shape = next(iter(source_shapes))
    target_shape = next(iter(target_shapes))
    if (
        len(source_shape) != 2
        or source_shape[0] != expected_source_vocab_size
        or not boundary_tokens
        or target_shape[0] != source_shape[0] + len(boundary_tokens)
        or target_shape[1] != source_shape[1]
    ):
        raise RuntimeError(
            "Unexpected vocabulary expansion for stage-2 initialization: "
            f"source={source_shape}, target={target_shape}, boundary_tokens={boundary_tokens}"
        )

    boundary_ids = _token_ids(model, boundary_tokens, label="Boundary")
    text_ids = _token_ids(model, text_tokens, label="Compact-text")
    source_text_ids = list(range(source_shape[0] - len(text_tokens), source_shape[0]))
    first_inserted_id = source_text_ids[0]
    expected_boundary_ids = list(
        range(first_inserted_id, first_inserted_id + len(boundary_tokens))
    )
    expected_text_ids = list(range(first_inserted_id + len(boundary_tokens), target_shape[0]))
    if boundary_ids != expected_boundary_ids:
        raise RuntimeError(
            f"Boundary tokens must occupy the source compact-text rows {expected_boundary_ids}; got {boundary_ids}"
        )
    if text_ids != expected_text_ids:
        raise RuntimeError(f"Compact-text tokens must move to appended rows {expected_text_ids}; got {text_ids}")

    expanded_state = dict(source_state)
    for key in expected_mismatch_keys:
        source_tensor = source_state[key].detach()
        target_tensor = target_state[key].detach()
        expanded = target_tensor.to(device="cpu", dtype=source_tensor.dtype).clone()
        first_source_text_id = source_text_ids[0]
        expanded[:first_source_text_id].copy_(source_tensor[:first_source_text_id].to(device="cpu"))
        for source_id, target_id in zip(source_text_ids, text_ids):
            expanded[target_id].copy_(source_tensor[source_id].to(device="cpu"))
        expanded_state[key] = expanded

    logging.info(
        "Prepared ASR stage-2 vocabulary expansion %d -> %d; preserved %s at IDs %s and remapped %s "
        "from IDs %s to %s",
        source_shape[0],
        target_shape[0],
        boundary_tokens,
        boundary_ids,
        text_tokens,
        source_text_ids,
        text_ids,
    )
    return expanded_state


def _source_lora_merge_config(checkpoint, fallback=None):
    """Read source scaling; never infer alpha from rank or silently ignore variants."""
    saved = checkpoint.get('hyper_parameters', {}).get('cfg', {}).get('lora', {}) or {}
    if not isinstance(saved, Mapping):
        raise ValueError('Checkpoint hyper_parameters.cfg.lora must be a mapping')
    # This initializer currently supports the historical standard Q/V adapter.
    # These settings change scaling, tensor layout or extra trained modules.
    for key in ('use_rslora', 'use_dora', 'fan_in_fan_out', 'lora_bias',
                'rank_pattern', 'alpha_pattern', 'modules_to_save', 'layer_replication',
                'target_parameters'):
        if saved.get(key):
            raise ValueError(f'Automatic ASR LoRA merge does not support source setting {key}={saved[key]!r}')
    if saved.get('bias', 'none') != 'none':
        raise ValueError('Automatic ASR LoRA merge supports source bias=none only')
    targets = saved.get('target_modules')
    if targets is not None and (isinstance(targets, str) or set(targets) != {'q_proj', 'v_proj'}):
        raise ValueError('Automatic ASR LoRA merge currently supports q_proj/v_proj source adapters only')
    fallback = {} if fallback is None else fallback
    if not isinstance(fallback, Mapping) or set(fallback) - {'rank', 'alpha'}:
        raise ValueError('Top-level init_lora_merge may contain only rank and alpha')
    resolved = {}
    for target, source in (('rank', 'r'), ('alpha', 'lora_alpha')):
        value = saved.get(source)
        if value is not None and target in fallback and fallback[target] != value:
            raise ValueError(f'init_lora_merge.{target} conflicts with checkpoint lora.{source}={value!r}')
        if value is None:
            value = fallback.get(target)
        if value is None:
            raise ValueError(
                f'ASR checkpoint contains LoRA weights but hyper_parameters.cfg.lora.{source} is missing. '
                f'Set init_lora_merge.{target} at the top level of the training YAML to the source '
                'checkpoint setting. Alpha cannot be inferred from adapter weights.'
            )
        resolved[target] = value
    return resolved


def load_asr_stage2_weights(model, checkpoint_path: str, *, lora_merge=None) -> None:
    # Fresh ASR initialization only; the entry point skips this on full-state resume.
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ASR stage-2 checkpoint is not a readable file: {checkpoint_path}")

    logging.info("Loading ASR weights for fresh stage-2 training from: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    if "state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint has no state_dict: {checkpoint_path}")

    source_has_lora = any('.lora_' in name for name in checkpoint['state_dict'])
    target_has_lora = any('.lora_' in name for name in model.state_dict())
    merge_config = None
    if lora_merge is not None and (not source_has_lora or target_has_lora):
        raise ValueError('init_lora_merge is only valid for a LoRA source and an adapter-free full-FT target')
    if source_has_lora and not target_has_lora:
        merge_config = _source_lora_merge_config(checkpoint, lora_merge)

    boundary_tokens = ("<sou>", "<eou>")
    user_backchannel_mode = getattr(model, "user_backchannel_mode", "ignore")
    if user_backchannel_mode in {"sob_eou", "sob_eob"}:
        boundary_tokens += (getattr(model, "user_backchannel_start_token", "<sob>"),)
    if user_backchannel_mode == "sob_eob":
        boundary_tokens += (getattr(model, "user_backchannel_end_token", "<eob>"),)
    from contextlib import nullcontext

    initialization = nullcontext()
    if merge_config is not None:
        from nemo.collections.speechlm2.parts.lora_checkpoint import full_finetuning_lora_init

        initialization = full_finetuning_lora_init(model, **merge_config)

    with initialization:
        expanded_state = prepare_asr_stage2_state_dict(
            model, checkpoint["state_dict"], boundary_tokens=boundary_tokens
        )
        incompatible = model.load_state_dict(expanded_state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict stage-2 load unexpectedly returned: {incompatible}")
        del expanded_state
    del checkpoint
    logging.info("ASR stage-2 weights loaded with a strict state-dict match%s",
                 " and PEFT adapters merged for full finetuning" if merge_config is not None else "")
