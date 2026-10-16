# Full LM finetuning from an ASR LoRA checkpoint

Use `examples/speechlm2/streaming_stt_train_stage2.py` with
`--config-name streaming_stt_full_ft`, your data/Trainer overrides, and
`init_weights_from_checkpoint=/path/to/asr.ckpt`.

The recipe starts an adapter-free model with a trainable LM, embedding/head and
modality projection. The speech encoder is frozen, matching historical Setup C;
set `model.freeze_speech_encoder=false` to train it too. Existing model freeze
flags and optimizer configuration still control trainability. No production
batch sizes, learning-rate schedule or dataset paths are supplied by this example.

```yaml
# Top-level settings for fresh stage-2 initialization:
init_weights_from_checkpoint: /path/to/asr.ckpt
model:
  freeze_llm_model: false
  freeze_speech_encoder: true
  freeze_llm_head: false
  freeze_embed_tokens: false
  freeze_params: []
  prevent_freeze_params: []
# Omit model.lora entirely; setting it to null is not equivalent in this model.
```

For a full-FT target, LoRA weights are detected automatically. Rank and alpha come
from `checkpoint["hyper_parameters"]["cfg"]["lora"]` (`r` and `lora_alpha`). The
real ASR step-12676 checkpoint stores both, so no `init_lora_merge` setting is needed.
A LoRA target keeps its adapters; an adapter-free ASR source loads without merging.

If required source metadata is absent, the error tells you which value to supply
at the top level of the training YAML. For example, **only when metadata is missing**:

```yaml
init_lora_merge:
  rank: 128   # Use the actual source rank.
  alpha: 256  # Use the actual source alpha; never infer it from rank.
```

You may supply just the missing field. Values conflicting with saved metadata are
rejected. Supported source: the historical ordinary alpha/r-scaled, bias-free Q/V
LoRA with one PEFT `default` adapter and unquantized weights. Known non-default
scaling/layout settings (including rsLoRA, DoRA and per-layer rank/alpha patterns)
are rejected. Missing optional settings use this supported historical format;
rank and alpha have no implicit defaults. Strict loading rejects unmatched tensors.

## Code path

1. The stage-2 entry point constructs the ordinary adapter-free target model.
2. On a fresh start, `load_asr_stage2_weights(..., lora_merge=...)` enters
   `parts/lora_checkpoint.py::full_finetuning_lora_init`. It temporarily installs
   real PEFT Q/V adapters using the saved source rank and alpha. NeMo's existing
   `move_embedding` context handles its separately managed embedding.
3. The existing strict ASR initializer loads the complete checkpoint, preserves
   pretrained vocabulary rows, moves compact-text marker rows, and retains new
   boundary rows. This expects the existing 151672-row compact-text ASR vocabulary;
   it is not a generic same-vocabulary TTM-checkpoint converter.
4. PEFT's `merge_and_unload(safe_merge=True)` performs the merge and removes the
   adapters. Adapted layers are upcast before checkpoint loading, so source FP32
   weights are not prematurely rounded into a BF16 target. Merge runs in FP32 with
   autocast disabled, then those layers return to the target dtype. No custom merge
   arithmetic or checkpoint-key translation remains.
5. The native `_apply_freeze_config()` restores trainability after PEFT's temporary
   adapter installation. The normal optimizer is created for the resulting full-FT
   target. The source LoRA optimizer state is not imported into fresh training.

When `exp_manager` selects a full-FT resume checkpoint (`trainer.ckpt_path`), the
entry point skips ASR initialization and merging entirely. Lightning restores the
adapter-free model, optimizer, scheduler and step. Keep the full-FT model/config
unchanged for resume. Do not pass a full-FT checkpoint as a fresh ASR initializer.

The batch profiler uses the same automatic detection and optional metadata fallback, so its
forward/backward tests measure the full-FT model. Re-estimate batch sizes when
changing trainability; LoRA batch sizes are not full-FT capacity measurements.
No legacy OOMptimizer, datamodule, or state-token behavior was changed.

## Validation scope

`tests/collections/speechlm2/test_lora_checkpoint.py` uses a tiny real Qwen3 model
and actual PEFT adapters. It covers FP32/BF16/FP16 merge precision, pre/post-merge
logit parity, external tied embeddings, adapter removal, full-LM trainability,
malformed/nonfinite adapter rejection, strict ASR vocabulary remapping, frozen and
trainable encoder choices, full-weight updates, and adapter-free PyTorch
save/reload with optimizer/scheduler state. Existing stage-2 tests also pass.

Local tests used PEFT 0.18.0 / Transformers 4.57.1, followed by full native
container validation on PEFT 0.18.1 / Transformers 4.57.6. IAD GPU smoke 14060864
passed 43 tests without skips and exercised the real ASR checkpoint, two-bin
full-FT profiling, generated YAML, two training updates, checkpoint save, and a
fresh-process Lightning resume to update 3. Model, optimizer and scheduler state
matched exactly at step 2. Ordinary LM weight updates and frozen encoder gradients
were checked, as were adapter removal and tied embeddings. No OOM occurred.

Runtime: 5m25s, 0.09028 A100 GPU-hours, plus 13 seconds queued. This is bounded
single-GPU acceptance: six original D7 examples, batch search capped at one,
1,722,141,696 trainable parameters. The native loader does not restore exact sample
position; its mid-epoch replay warning remains. Distributed resume, production
batch maxima and GPU training of the unfrozen encoder are not covered. Exact
reports and source hashes are in the companion review workspace under
`validation/full_ft_peft_gpu_20261016`.

Automatic metadata selection subsequently passed IAD GPU smoke **14069810**:
**56 native tests passed, zero skipped**, followed by full-model profiling and
training with `init_lora_merge` absent (asserted). The real ASR checkpoint supplied
rank and alpha automatically. Two updates, checkpoint save, and exact-state
fresh-process Lightning resume to update 3 passed with no OOM. Runtime: 5m25s,
0.09028 GPU-hours, 6-second queue. Current code matched the tested source bundle.
Evidence: `validation/full_ft_auto_gpu_20261016` in the companion review workspace.
The same single-GPU, bounded-fixture and sample-position limitations above apply.
