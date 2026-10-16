# Manifest-to-batching connector

Run with a NeMo training environment containing the same Torch, Lightning,
Lhotse, tokenizer and model dependencies as training:

```sh
PYTHONPATH=. python scripts/speechlm2/estimate_batch_sizes.py run \
  --plan /absolute/path/profile.yaml --output /absolute/path/fresh-run
```

The `prepare`, `profile` and `finalize` stages can also run separately. `prepare`
reads metadata on CPU. `profile` uses one GPU, real audio and native target
formatting, and launches each candidate trial in a fresh subprocess. `finalize`
measures the native CPU sampler, optionally calibrates separate-loader weights,
and writes `training.yaml` and `REPORT.json`. No scheduler is required or invoked.
Use the same environment and paths across stages. Input YAML manifest/audio paths
should be absolute. Existing outputs resume only when the input/code binding
matches; changed inputs require a fresh run directory. An advisory file lock
prevents concurrent controllers. Interrupted workers without a completed report
are rerun; successful trials are reused. Do not edit cached reports or artifacts.

Example plan (the named profile and source labels are arbitrary):

```yaml
model_config: /data/model_training.yaml
profiles:
  speech:
    input_cfg: /data/all_sources.yaml
    # Optional fixed upper bounds; otherwise use native Lhotse estimation:
    # bucket_duration_bins: [10, 20, 40, 80, 300]
num_buckets: 10
seed: 42
memory_fraction: 0.80
candidates_per_source_bin: 8
max_batch_size: 128
updates_per_case: 2
measurement_batches: 1000
```

`input_cfg` is the usual native list of `nemo`, `nemo_tarred`, or nested `group`
entries. Group tags are inherited by their children for profiling; native input
weights and grouping remain unchanged in the generated training recipe. Every
leaf source contributes candidates in its populated bins. No dataset names are
hardcoded. Bins use pooled manifest membership, not a weighted simulation of the
runtime mixture. Source weights therefore affect runtime mixture measurement but
not the pooled estimation scan. Profiles with zero-weight sources are not a way
to exclude validation: omit inputs that are not intended for this profile.

For intentionally separate runtime loaders, supply multiple profile entries with
positive `target_share` values summing to one. The generated recipe uses the
existing native `multi_config` path with `randomized_round_robin` and `sampler_weights`. A single profile uses the original
shared `train_ds.input_cfg` path. The model/precision/config is shared by all
profiles in one run. Use a separate run per trainability setup or hardware type.

## What is reused

`batch_estimation.py` calls the existing Lhotse duration estimator. The original
speechlm2 OOMptimizer search was copied into standalone `batch_size_search.py`
for this real-data connector. The original artificial-batch script is unchanged
and does not import the copy. The
connector does not replace the search with the D7 campaign's memory-guided loop.
Real-batch training checks and subprocess isolation derive from that campaign.
The existing OOMptimizer CLI keeps its original memory-fraction default; this new
workflow defaults to 80%.

## Coverage and interpretation

Metadata traversal is finite and source-stratified. For context-aware inputs it
considers the seeded sampled crop, minimum context, and maximal-duration corners
favoring either leading or trailing context. Geometry uses sampled durations
and preserves the supported envelope maximum. Candidate shortlisting uses audio
length, transcript character count and seeded representative selection; actual
formatted input and target lengths are measured on the selected real examples.
Character count is only a shortlist heuristic, never a target-token measurement.
This is bounded coverage, not exhaustive formatted-token extrema. Some feasible
intermediate crops may not occur in the four modes. Empty selected bins stop the
workflow instead of receiving an untested placeholder batch size.

GPU trials test representative, long-audio, long-input, long-target and combined
cases, deduplicating equivalent memberships. Each leaf source is checked as well
as explicit source-mixed batches when batch size permits. Trials preserve native
alignment and formatting; absent saved words require a configured forced aligner.
Saved audio crops are local profiling artifacts under the output directory and
are not transferred by the tool. Candidate audio/cut hashes are checked on use.

A passing trial includes optimizer updates, finite loss/gradients, exact batch
membership, sanity and post-update validation. Memory checks include peak Torch
reserved memory and observed non-Torch memory. Only explicit CUDA OOM causes
search reduction. A caller-capped result is a tested size, not proof of the
maximum feasible size. A bounded single-GPU PASS is not distributed training or
resume clearance, and does not guarantee no OOM on unseen examples or crops.

The CPU measurement uses the configured worker count on one rank and measures
batch-size histograms and labeled source counts. Multiple-profile calibration
uses these measured histograms, then simulates the requested example mixture.
Verify the production world-size mixture separately. `REPORT.json` records this
scope explicitly. The measurement has no model forward pass and does not prove
full-dataset decode validity.

## Current restrictions

- Boundary targets; native `nemo`/`nemo_tarred` sources; duration-only buckets.
- One GPU, `accumulate_grad_batches=1`; no checkpoint resume of profiler trials.
- `init_weights_from_checkpoint`, when supplied, must be a compatible Lightning
  state dict and loads strictly. Automatic LoRA merging/conversion is not done.
- Extra loader duration transforms (padding, speed perturbation, concatenation,
  windowing) must not silently differ from geometry; unsupported configured
  transforms are rejected. Source-native context sampling is supported.
- Manifest hashes bind local, readable manifest files. Remote manifests need a
  stable local manifest view. Source audio and model assets must remain immutable
  during a run; only selected materialized audio is fully hashed by this workflow.

Profiling can be expensive: each trial initializes the real model/aligner. Metadata
scan size, shortlist size, buckets, maximum batch size and OOM outcomes determine
cost; worker reports record elapsed time. Choose bounded smoke inputs before a
full profiling run. No production training starts automatically.

## Independently scheduled bins

The default `profile` stage remains serial. External orchestrators may instead
run `prepare`, then GPU `collect`, followed by independent one-GPU commands:

```bash
python scripts/speechlm2/estimate_batch_sizes.py profile-bin \
  --plan plan.yaml --output results --profile dai --bucket 0
```

`collect` writes `bin_tasks.json` listing the actual profile/bin pairs, including
any reduction in automatic bin count. Each pair must finish before GPU `validate`
and CPU `finalize`. Only separate bins may overlap. The same bin is exclusively
locked; pipeline-wide mutation is excluded during bin searches. All bin GPUs
must match the collected hardware/software identity. Per-bin results include
trial and candidate-report hashes. Missing or altered evidence stops validation.
Scheduler implementation belongs to the caller; this module never allocates GPUs
or submits jobs. Single-bin subprocess trials still run sequentially.

Multiple-profile export preserves existing loader seed/shard_seed settings (native
defaults if omitted). No batch_mux_seed or custom selection cursor is used.
Use a flat train_ds template for estimation; profiles belong in the analysis plan.
The generated multi_config YAML sets force_iterable_dataset on each child so
tarred and untarred inputs can share the native sampler fusion path.

## Shared readers and optional calibration diagnostics

Duration-only export and full profiling use `batch_estimation.cuts_for` and the
native adapter. Stress crops use explicit `context_sampling.analysis_mode` on a
copied analysis configuration; training inputs are not modified. The default
remains ordinary context sampling. Profiling reuses `load_asr_stage2_weights`
from the stage-2 training entry point for checkpoint initialization.

Mixture weights are calculated from measured native-loader mean batch sizes.
No simulation is required. An optional positive `calibration_simulation_draws`
in the plan enables a Python histogram diagnostic; even a diagnostic outside
its tolerance does not prevent export. It cannot certify production mixtures.

Checkpoint initialization follows the existing stage-2 training contract, including
its tokenizer layout. For an ASR checkpoint trained with compact text markers,
set `model.use_text_tokens: true` and `data.dataset.use_text_tokens: true` along
with compact-template mode; do not substitute the default chat-header layout.
This is fresh stage-2 initialization, not arbitrary checkpoint resume or LoRA merge.

Candidate matching uses explicit `sample_id` when available. Otherwise native
readers preserve the original audio filepath, offset and duration as a row key,
so tar subsegment naming and traversal order do not change candidate identity.
Duplicate row identities within one source are rejected during preparation.
