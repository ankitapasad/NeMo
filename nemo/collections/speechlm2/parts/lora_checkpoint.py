# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Use PEFT to load and merge ASR LoRA for adapter-free full finetuning."""
from contextlib import contextmanager
import math

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer

from nemo.collections.speechlm2.parts.pretrained import move_embedding


@contextmanager
def full_finetuning_lora_init(model, *, rank, alpha):
    """Temporarily install source Q/V adapters for a strict checkpoint load.

    The caller loads the checkpoint inside this context. PEFT then merges and
    unloads the adapters, computing adapted layers in FP32 before restoring their
    target dtype. The native freeze policy is reapplied before optimizer creation.
    This is fresh initialization only, never full-FT checkpoint resume.

    Rank/alpha must match the source standard (alpha/r), single-default-adapter
    LoRA config. Other variants require their complete source PEFT configuration;
    this helper deliberately does not infer it from checkpoint tensor shapes.
    """
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA alpha must be finite and positive")
    if isinstance(model.llm, PeftModel) or 'lora' in model.cfg:
        raise ValueError("init_lora_merge requires an adapter-free target: remove model.lora")
    if model.cfg.get('freeze_llm_model', False):
        raise ValueError("init_lora_merge requires model.freeze_llm_model=false")

    config = LoraConfig(task_type='CAUSAL_LM', r=rank, lora_alpha=alpha,
                        target_modules=['q_proj', 'v_proj'], lora_dropout=0.0, bias='none')
    with move_embedding(model):
        model.llm = get_peft_model(model.llm, config)

    # Casting before loading preserves source FP32 adapted weights even when the
    # target model was constructed in BF16. Do not upcast the whole language model.
    layers = []
    for module in model.llm.modules():
        if isinstance(module, LoraLayer):
            base = module.get_base_layer()
            layers.append((base, base.weight.dtype))
            module.float()
    if not layers:
        raise ValueError("No Q/V LoRA layers installed for the source checkpoint")

    yield

    # PEFT owns the merge arithmetic, scaling, wrapper removal and safe-merge check.
    device_type = layers[0][0].weight.device.type
    with torch.autocast(device_type=device_type, enabled=False), move_embedding(model):
        model.llm = model.llm.merge_and_unload(safe_merge=True)
    for base, dtype in layers:
        base.to(dtype=dtype)
    model._apply_freeze_config()
    if isinstance(model.llm, PeftModel) or any(isinstance(m, LoraLayer) for m in model.llm.modules()):
        raise RuntimeError("PEFT merge left adapters in the full-finetuning model")
