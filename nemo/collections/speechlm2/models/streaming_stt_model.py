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
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.asr.inference.streaming.buffering.cache_feature_bufferer import BatchedCacheFeatureBufferer
from nemo.collections.asr.inference.streaming.framing.request import Frame
from nemo.collections.asr.inference.utils.context_manager import CacheAwareContext
from nemo.collections.common.data.utils import move_data_to_device
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    AUDIO_TOKEN_IDX,
    IGNORE_INDEX,
    StreamingSTTBatch,
    StreamingSTTDataset,
    decode_with_blank,
    parse_chat_template_ids,
)
from nemo.collections.speechlm2.parts.alignments import ForcedAligner
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding, setup_perception
from nemo.collections.speechlm2.parts.utils import freeze_module, to_dataclass, unfreeze_module
from nemo.utils import logging


def _find_sublist(haystack: list, needle: list) -> int | None:
    """Return the start index of *needle* in *haystack*, or ``None``."""
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def interleave_embeddings(
    input_tokens: Tensor,
    audio_mask: Tensor,
    text_embeds: Tensor,
    audio_embs: Tensor,
    pad_id: int,
) -> dict[str, Tensor]:
    """
    Merge pre-computed text and audio embeddings into a single sequence,
    guided by ``audio_mask``.

    All operations are fully batched (no Python loops over batch items):

    1. ``cumsum`` on the audio mask gives a 0-based frame index per audio position.
    2. ``torch.gather`` selects the correct audio frame for each position.
    3. ``torch.where`` picks audio or text embeddings per position.

    Args:
        input_tokens: (B, L) token IDs — only used to derive the attention mask
            (non-``pad_id`` positions).
        audio_mask: (B, L) bool — True at ``AUDIO_TOKEN_IDX`` positions.
        text_embeds: (B, L, H) embeddings produced by the text embedding layer.
            Values at audio positions are unused and may be arbitrary.
        audio_embs: (B, T_enc, H) frame-level embeddings from the audio encoder.
            If there are more audio tokens than encoder frames (last-chunk
            ceiling), the tensor is zero-padded automatically.
        pad_id: token ID used for left-padding — these positions get
            ``attention_mask = False``.

    Returns:
        dict with:
            ``input_embeds`` — (B, L, H) interleaved embeddings.
            ``attention_mask`` — (B, L) bool, False only at left-padding positions.
    """
    B, L = input_tokens.shape

    if not audio_mask.any():
        # Pure text — nothing to interleave.
        attention_mask = input_tokens != pad_id
        return {"input_embeds": text_embeds, "attention_mask": attention_mask}

    # Sequential 0-based frame index for each audio-token position.
    frame_indices = audio_mask.long().cumsum(dim=1) - 1  # (B, L)

    # Pad encoder output if the dataset produced more audio tokens than
    # the encoder returned (last chunk ceiling).
    max_frame_idx = frame_indices.max().item()
    T_enc = audio_embs.shape[1]
    if max_frame_idx >= T_enc:
        audio_embs = F.pad(audio_embs, (0, 0, 0, max_frame_idx - T_enc + 1))

    # Gather the correct audio frame for every position in L.
    H = audio_embs.shape[2]
    gather_idx = frame_indices.clamp(min=0).unsqueeze(-1).expand(B, L, H)
    audio_at_all_pos = torch.gather(audio_embs, dim=1, index=gather_idx)  # (B, L, H)

    # Merge: audio embeddings at audio positions, text embeddings elsewhere.
    embeds = torch.where(audio_mask.unsqueeze(-1), audio_at_all_pos, text_embeds)

    # Attend to every non-padding position.
    # pad_id is ≥ 0 and AUDIO_TOKEN_IDX is −100, so this is safe.
    attention_mask = input_tokens != pad_id  # (B, L)

    return {"input_embeds": embeds, "attention_mask": attention_mask}


@dataclass
class StreamingSTTModelConfig:
    pretrained_llm: str
    pretrained_asr: str
    load_llm_weights: bool
    blank_token: str
    load_asr_weights: bool
    freeze_speech_encoder: bool
    freeze_modality_adapter: bool
    freeze_modality_proj: bool
    freeze_llm_model: bool
    freeze_llm_head: bool
    freeze_embed_tokens: bool
    chunk_size: int
    audio_tag: str = "<audio>"
    att_context_size: Optional[List[int]] = None
    audio_pad_to: Optional[int] = None
    sample_rate: int = 16000
    frame_length_in_secs: float = 0.08
    log_every_n_steps: int = 10
    dtype: str = "bfloat16"


@dataclass
class StreamingState:
    """Holds the KV cache and other state for one streaming audio session."""

    cache: tuple | None = None  # HF past_key_values
    generated_tokens: list[list[int]] = field(default_factory=list)  # per-chunk generated token IDs
    seq_len: int = 0  # total sequence length seen so far
    audio_cache: CacheAwareContext | None = None
    audio_feature_buffer: BatchedCacheFeatureBufferer | None = None


class StreamingSTTModel(LightningModule, HFHubMixin):

    def __init__(
        self,
        cfg: dict,
        forced_aligner: Optional[ForcedAligner] = None,
        data_cfg: Optional[DictConfig] = None,
        dataset_cls=StreamingSTTDataset,
    ) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to StreamingSTTModel as a Python dict to support hyperparameter "
            f"serialization in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        self.cfg = DictConfig(cfg)
        self.core_cfg: StreamingSTTModelConfig = to_dataclass(StreamingSTTModelConfig, cfg)

        # --- LLM ---
        self.tokenizer = AutoTokenizer(self.core_cfg.pretrained_llm, use_fast=True)
        self.llm = load_pretrained_hf(
            self.core_cfg.pretrained_llm,
            pretrained_weights=self.core_cfg.load_llm_weights,
        )

        # Ensure <blank> token is in the vocabulary.
        self.blank_token = self.core_cfg.blank_token
        if self.blank_token not in self.tokenizer.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.blank_token]})
            self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))

        # Separate embedding layer to avoid FSDP/TP conflicts (same pattern as SALM)
        self.embed_tokens = self.llm.model.embed_tokens
        del self.llm.model.embed_tokens

        # --- LoRA ---
        maybe_install_lora(self)

        # --- Speech encoder (perception module) ---
        self.perception = setup_perception(
            cfg=self.cfg,
            output_dim=self.llm.config.hidden_size,
            pretrained_asr=self.core_cfg.pretrained_asr,
            pretrained_weights=self.core_cfg.load_asr_weights,
            audio_pad_to=self.core_cfg.audio_pad_to,
            att_context_size=self.core_cfg.att_context_size,
        )

        self._apply_freeze_config()

        if forced_aligner is not None:
            assert data_cfg is not None, "Dataset config is required for online forced alignment"
            assert dataset_cls is not None, "Dataset class is required for online forced alignment"
            self.forced_aligner = forced_aligner
            self.dataset = dataset_cls(cfg=data_cfg, tokenizer=self.tokenizer)
        else:
            self.forced_aligner = None
            self.dataset = None

        logging.info("\n" + str(ModelSummary(self, max_depth=2)))

    def _apply_freeze_config(self) -> None:
        if self.core_cfg.freeze_speech_encoder:
            freeze_module(self.perception.encoder)
        else:
            unfreeze_module(self.perception.encoder)

        if self.core_cfg.freeze_modality_adapter:
            freeze_module(self.perception.modality_adapter)
        else:
            unfreeze_module(self.perception.modality_adapter)

        if self.core_cfg.freeze_modality_proj:
            freeze_module(self.perception.proj)
        else:
            unfreeze_module(self.perception.proj)

        # Freeze the LLM body (lm_head and embed_tokens are handled separately)
        if self.core_cfg.freeze_llm_model:
            freeze_module(self.llm.model)
        else:
            unfreeze_module(self.llm.model)

        # lm_head is inside self.llm, so re-apply after the LLM-wide freeze
        if self.core_cfg.freeze_llm_head:
            freeze_module(self.llm.lm_head)
        else:
            unfreeze_module(self.llm.lm_head)

        # embed_tokens is a separate top-level module (moved out of llm)
        if self.core_cfg.freeze_embed_tokens:
            freeze_module(self.embed_tokens)
        else:
            unfreeze_module(self.embed_tokens)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return int(self.embed_tokens.num_embeddings)

    @property
    def text_pad_id(self) -> int:
        pad_id = self.tokenizer.pad_id
        if pad_id is None:
            pad_id = self.tokenizer.unk_id
        if pad_id is None:
            warnings.warn(
                "The text tokenizer has no <pad> or <unk> token; using id 0 for "
                "padding (this may lead to silent bugs)."
            )
            pad_id = 0
        return pad_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def sampling_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    @property
    def sample_rate(self) -> int:
        return self.perception.preprocessor.featurizer.sample_rate

    @property
    def frame_duration(self) -> float:
        """Duration (in seconds) of one audio frame at the perception output."""
        return self.perception.token_equivalent_duration

    # ------------------------------------------------------------------
    # Core: efficient audio-text embedding interleaving
    # ------------------------------------------------------------------

    def _build_input_embeds(
        self,
        input_tokens: Tensor,
        audios: Tensor,
        audio_lens: Tensor,
    ) -> dict[str, Tensor]:
        """
        Encode audio, embed text tokens, then interleave them.

        This is the high-level entry point used by ``training_step`` and
        ``_eval_step``.  The pure-tensor interleaving logic lives in
        :func:`interleave_embeddings` so it can be tested without a model.

        Args:
            input_tokens: (B, L) token IDs with ``AUDIO_TOKEN_IDX`` at audio
                positions and ``text_pad_id`` at left-padding positions.
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
        Returns:
            dict with keys ``input_embeds`` (B, L, H), ``attention_mask`` (B, L).
        """
        audio_mask = input_tokens == AUDIO_TOKEN_IDX  # (B, L)

        # --- text embeddings ---
        # Zero-out audio positions so embed_tokens gets valid indices.
        text_tokens = input_tokens.where(~audio_mask, torch.zeros_like(input_tokens))
        text_embeds = self.embed_tokens(text_tokens)  # (B, L, H)

        # --- audio embeddings ---
        audio_embs, _audio_emb_lens = self.perception(
            input_signal=audios,
            input_signal_length=audio_lens,
        )  # audio_embs: (B, T_enc, H)

        # --- interleave & build attention mask ---
        return interleave_embeddings(
            input_tokens=input_tokens,
            audio_mask=audio_mask,
            text_embeds=text_embeds,
            audio_embs=audio_embs,
            pad_id=self.text_pad_id,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_embeds: Tensor,
        attention_mask: Tensor | None = None,
        cache=None,
    ) -> dict[str, Tensor]:
        """
        Forward pass:  embeddings → LLM → logits.
        """
        out = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
        )
        ans = {"logits": out["logits"]}  # (B, L, V)
        if cache is not None:
            ans["cache"] = out["past_key_values"]
        return ans

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: StreamingSTTBatch, batch_idx: int):
        # Keep frozen modules in eval mode (disables dropout / batch-norm updates).
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm):
            if is_frozen(m):
                m.eval()

        if self.forced_aligner is not None:
            alignments = self.forced_aligner.align(batch.audios, batch.audio_lens, batch.text)
            batch = self.dataset.get_batch_data(
                cuts=batch.cuts,
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                alignments=alignments,
                text=batch.text,
            )
            batch = move_data_to_device(batch, self.device)

        inputs = self._build_input_embeds(batch.input_tokens, batch.audios, batch.audio_lens)
        outputs = self.forward(inputs["input_embeds"], attention_mask=inputs["attention_mask"])

        target_ids = batch.target_tokens
        num_targets = (target_ids != IGNORE_INDEX).long().sum()

        if num_targets == 0:
            logging.warning("Batch %d: num_targets is 0 — skipping (returning zero loss).", batch_idx)
            return {"loss": torch.tensor(0.0, device=target_ids.device, requires_grad=True)}

        logits = outputs["logits"]

        # Diagnose NaN sources (remove once stable).
        if torch.isnan(inputs["input_embeds"]).any():
            logging.warning("Batch %d: NaN in input_embeds", batch_idx)
        if torch.isnan(logits).any():
            logging.warning("Batch %d: NaN in logits", batch_idx)

        with loss_parallel():
            loss = (
                F.cross_entropy(
                    logits.flatten(0, 1),
                    target_ids.flatten(0, 1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
                / num_targets
            )
        B, L = inputs["input_embeds"].shape[:2]
        self.log_dict(
            {
                "loss": loss,
                "learning_rate": torch.as_tensor(
                    self.trainer.optimizers[0].param_groups[0]["lr"] if self._trainer is not None else 0
                ),
                "batch_size": float(B),
                "sequence_length": float(L),
                "num_targets": num_targets.float(),
                "target_to_input_ratio": num_targets / (B * L),
            },
            on_step=True,
        )
        return {"loss": loss}

    def configure_optimizers(self):
        return configure_optimizers(self)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        self._partial_val_losses: dict[str, list] = defaultdict(list)
        self._partial_accuracies: dict[str, list] = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        val_losses = []
        for name, vals in self._partial_val_losses.items():
            val_loss = torch.stack(vals).mean()
            self.log(f"val_loss_{name}", val_loss, on_epoch=True, sync_dist=True)
            val_losses.append(val_loss)
        if val_losses:
            self.log("val_loss", torch.stack(val_losses).mean(), on_epoch=True, sync_dist=True)

        accuracies = []
        for name, accs in self._partial_accuracies.items():
            val_acc = torch.stack(accs).mean()
            self.log(f"val_acc_{name}", val_acc, on_epoch=True, sync_dist=True)
            accuracies.append(val_acc)
        if accuracies:
            self.log("val_acc", torch.stack(accuracies).mean(), on_epoch=True, sync_dist=True)

        self._partial_val_losses.clear()
        self._partial_accuracies.clear()

    def validation_step(self, batch, batch_idx: int):
        # Support multiple validation dataloaders ({name: batch} dict).
        if isinstance(batch, dict):
            for name, dataset_batch in batch.items():
                if dataset_batch is not None:
                    self._eval_step(dataset_batch, name, batch_idx)
        else:
            self._eval_step(batch, "val", batch_idx)

    def _eval_step(self, batch: StreamingSTTBatch, name: str, batch_idx: int = 0) -> None:
        if self.forced_aligner is not None:
            alignments = self.forced_aligner.align(batch.audios, batch.audio_lens, batch.text)
            batch = self.dataset.get_batch_data(
                cuts=batch.cuts,
                audios=batch.audios,
                audio_lens=batch.audio_lens,
                alignments=alignments,
                text=batch.text,
            )
            batch = move_data_to_device(batch, self.device)

        inputs = self._build_input_embeds(batch.input_tokens, batch.audios, batch.audio_lens)
        outputs = self.forward(inputs["input_embeds"], attention_mask=inputs["attention_mask"])

        target_ids = batch.target_tokens
        num_targets = (target_ids != IGNORE_INDEX).long().sum()

        with loss_parallel():
            loss = (
                F.cross_entropy(
                    outputs["logits"].flatten(0, 1),
                    target_ids.flatten(0, 1),
                    reduction="sum",
                    ignore_index=IGNORE_INDEX,
                )
                / num_targets
            )

        preds = outputs["logits"].argmax(dim=-1).view(-1)
        refs = target_ids.reshape(-1)
        preds = preds[refs != IGNORE_INDEX]
        refs = refs[refs != IGNORE_INDEX]
        accuracy = preds.eq(refs).float().mean()

        self._partial_val_losses[name].append(loss)
        self._partial_accuracies[name].append(accuracy)

        # Log decoded predictions vs references periodically (first sample in batch).
        if batch_idx % self.core_cfg.log_every_n_steps == 0:
            # Per-sample: decode only the first sample's non-IGNORE tokens.
            sample_target = batch.target_tokens[0]
            sample_logits = outputs["logits"][0]
            sample_preds = sample_logits.argmax(dim=-1)
            mask = sample_target != IGNORE_INDEX
            sample_ref_ids = sample_target[mask].tolist()
            sample_pred_ids = sample_preds[mask].tolist()

            ref_decoded = decode_with_blank(sample_ref_ids, self.blank_token, self.tokenizer)
            pred_decoded = decode_with_blank(sample_pred_ids, self.blank_token, self.tokenizer)
            ref_text = batch.text[0] if batch.text else ""
            logging.info(
                "[%s] batch %d\n  gt:         `%s`\n  ref_tokens: `%s`\n  pred:       `%s`",
                name,
                batch_idx,
                ref_text,
                ref_decoded,
                pred_decoded,
            )

    # ------------------------------------------------------------------
    # Test (delegates to validation logic)
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end()

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    # ------------------------------------------------------------------
    # Backward + OOMptimizer
    # ------------------------------------------------------------------

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    @property
    def oomptimizer_schema(self) -> dict:
        from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType

        return {
            "cls": StreamingSTTBatch,
            "inputs": [
                {
                    "name": "input_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": int(self.text_vocab_size),
                },
                {"name": "input_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": int(self.text_vocab_size),
                },
                {"name": "target_token_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "output"},
                {"name": "audios", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
            ],
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensure_inference_cache(self) -> None:
        """Lazily cache token templates and IDs needed for streaming inference.

        Uses ``apply_chat_template(tokenize=False)`` on a 4-message dummy
        conversation and splits the text around a sentinel to isolate
        user-header, user-footer + assistant-header, and assistant-footer tokens.

        The 4-message pattern (two user+assistant pairs) ensures the *first*
        assistant turn is not the last — this prevents Qwen3-style chat
        templates from injecting ``<think>``/``</think>`` tags, which only
        appear on the final assistant turn.
        """
        if hasattr(self, '_turn_template_ids'):
            return

        hf_tok = self.tokenizer.tokenizer
        chunk_size = self.core_cfg.chunk_size

        # --- Build turn template from the shared template parser ---
        user_header_ids, user_footer_and_asst_header_ids, asst_footer_ids = parse_chat_template_ids(hf_tok)
        self._asst_footer_ids = asst_footer_ids

        turn_ids = user_header_ids + [AUDIO_TOKEN_IDX] * chunk_size + user_footer_and_asst_header_ids
        self._turn_template_ids = turn_ids
        n_audio = turn_ids.count(AUDIO_TOKEN_IDX)
        logging.info(
            f"Streaming turn template ({len(turn_ids)} tokens, "
            f"{n_audio} audio slots, chunk_size={chunk_size}): {turn_ids}"
        )

        self._blank_id = hf_tok.convert_tokens_to_ids(self.blank_token)
        self._eos_id = getattr(hf_tok, 'eos_token_id', None)

        # When eos_token_id coincides with a token in the footer (e.g. Qwen3
        # where eos = <|im_end|> = footer[0]), detecting EOS acts as an
        # early-stop shortcut that avoids generating the remaining footer
        # tokens.  When eos_token_id is NOT in the footer it serves as a
        # safety-net stop only.
        self._eos_in_footer = self._eos_id is not None and self._eos_id in self._asst_footer_ids
        logging.info(
            f"Assistant footer IDs: {self._asst_footer_ids}, "
            f"blank ID: {self._blank_id}, EOS ID: {self._eos_id}, "
            f"EOS in footer: {self._eos_in_footer}"
        )

    def _sample_token(
        self,
        logits: Tensor,
        generated_ids: list[int] | None = None,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> Tensor:
        """Select the next token from logits.

        Applies the following transforms in order (each is skipped when the
        corresponding parameter is at its default/off value):

        1. **Suppress tokens** — force listed token IDs to ``-inf``.
        2. **No-repeat-ngram** — block n-grams that already appear in
           *generated_ids*.
        3. **Repetition penalty** — scale logits for tokens that already appear
           in *generated_ids*.
        4. **Temperature** — divide logits by temperature.
        5. **Top-k** — keep only the *k* highest-scoring tokens.
        6. **Top-p (nucleus)** — keep the smallest set of tokens whose
           cumulative probability is ≥ *top_p*.
        7. If ``do_sample`` is ``True``, sample from the filtered distribution;
           otherwise return the argmax.

        Parameters are read from *generation_kwargs* first, falling back to
        *generation_config*, then to HuggingFace defaults.

        Args:
            logits: ``(1, vocab_size)`` logits for the last position.
            generated_ids: Token IDs generated so far (for repetition-aware
                transforms).  May be ``None`` or empty.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(1,)`` tensor with the selected token ID.
        """
        # Fast path: no config → greedy
        if generation_config is None and not generation_kwargs:
            return logits.argmax(dim=-1)

        cfg = generation_config or GenerationConfig()
        do_sample = generation_kwargs.get('do_sample', cfg.do_sample)
        temperature = generation_kwargs.get('temperature', cfg.temperature)
        top_k = generation_kwargs.get('top_k', cfg.top_k)
        top_p = generation_kwargs.get('top_p', cfg.top_p)
        repetition_penalty = generation_kwargs.get('repetition_penalty', cfg.repetition_penalty)
        no_repeat_ngram_size = generation_kwargs.get('no_repeat_ngram_size', cfg.no_repeat_ngram_size)
        suppress_tokens = generation_kwargs.get('suppress_tokens', cfg.suppress_tokens)

        # --- logit manipulation (order matters) ---

        # 1. Suppress tokens
        if suppress_tokens:
            logits[..., suppress_tokens] = float('-inf')

        # 2. No-repeat-ngram blocking
        if no_repeat_ngram_size > 0 and generated_ids and len(generated_ids) >= no_repeat_ngram_size - 1:
            ngram_prefix = generated_ids[-(no_repeat_ngram_size - 1) :]
            for i in range(len(generated_ids) - no_repeat_ngram_size + 1):
                if generated_ids[i : i + no_repeat_ngram_size - 1] == ngram_prefix:
                    # The token that followed this prefix last time is banned
                    logits[..., generated_ids[i + no_repeat_ngram_size - 1]] = float('-inf')

        # 3. Repetition penalty
        if repetition_penalty != 1.0 and generated_ids:
            prev_token_ids = torch.tensor(list(set(generated_ids)), device=logits.device)
            scores = logits[..., prev_token_ids]
            # Penalize: divide positive scores, multiply negative scores
            logits[..., prev_token_ids] = torch.where(
                scores > 0, scores / repetition_penalty, scores * repetition_penalty
            )

        # Greedy fast path (no sampling-related transforms needed)
        if not do_sample:
            return logits.argmax(dim=-1)

        # 4. Temperature scaling
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature

        # 5. Top-k filtering
        if top_k > 0:
            k = min(top_k, logits.size(-1))
            kth_val = torch.topk(logits, k, dim=-1)[0][..., -1:]
            logits = logits.masked_fill(logits < kth_val, float('-inf'))

        # 6. Top-p (nucleus) filtering
        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Mark tokens whose cumulative probability (excluding themselves) >= top_p
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            indices_to_remove = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        # 7. Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _streaming_decode(
        self,
        logits: Tensor,
        cache: tuple,
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> tuple[list[int], tuple, bool]:
        """Autoregressive decoding for the streaming path.

        Token selection is delegated to :meth:`_sample_token`, which supports
        greedy (default), sampling (temperature / top-k / top-p), repetition
        penalty, no-repeat-ngram blocking, and token suppression.

        Generation stops when any of these conditions is met (checked in order):

        1. **EOS** — the tokenizer's ``eos_token_id`` is predicted.  The token is
           *not* fed to the LLM and *not* included in the output.  When
           ``eos_token_id`` is also part of the footer (e.g. Qwen3 where
           ``<|im_end|>`` = ``eos`` = ``footer[0]``), this acts as an early-stop
           shortcut that avoids generating the remaining footer tokens.
        2. **Blank** — the ``<blank>`` token is predicted.  It is fed to the LLM
           (so the KV cache stays consistent) and included in the output.
        3. **Footer sequence** — the last *N* generated tokens match
           ``self._asst_footer_ids`` (e.g. ``[<|im_end|>, \\n]``).  All *N*
           tokens have been fed to the LLM; they are stripped from the output
           and ``footer_consumed`` is set to ``True``.  This is the primary
           stop mechanism for models whose ``eos_token_id`` is not in the footer.
        4. **Max tokens** — ``max_new_tokens`` is reached.

        Returns:
            ``(generated_token_ids, updated_cache, footer_consumed)`` —
            *footer_consumed* is ``True`` when the footer sequence was detected
            and is already present in the KV cache.
        """
        footer = self._asst_footer_ids
        flen = len(footer)
        generated = []
        footer_consumed = False
        next_token = self._sample_token(logits[:, -1, :], generated, generation_config, **generation_kwargs)

        while len(generated) < max_new_tokens:
            tid = next_token.item()

            # EOS: stop without feeding to LLM
            if self._eos_id is not None and tid == self._eos_id:
                break

            # Append token and feed to LLM
            generated.append(tid)
            token_emb = self.embed_tokens(next_token.unsqueeze(0))
            out = self.llm(
                inputs_embeds=token_emb,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values

            # Blank: stop (token is in cache and in generated)
            if tid == self._blank_id:
                break

            # Footer sequence match
            if flen > 0 and len(generated) >= flen and generated[-flen:] == footer:
                generated = generated[:-flen]
                footer_consumed = True
                break

            next_token = self._sample_token(out.logits[:, -1, :], generated, generation_config, **generation_kwargs)

        return generated, cache, footer_consumed

    def get_audio_feature_buffer(self, batch_size: int) -> BatchedCacheFeatureBufferer:
        """Get the audio feature buffer for the streaming state."""
        preprocessor_cfg: DictConfig = self.perception.cfg.preprocessor
        window_stride_in_secs = preprocessor_cfg.window_stride
        pre_encode_cache_size = self.perception.encoder.streaming_cfg.pre_encode_cache_size
        if isinstance(pre_encode_cache_size, list):
            pre_encode_cache_size = pre_encode_cache_size[1]
        pre_encode_cache_size_in_secs = pre_encode_cache_size * window_stride_in_secs
        chunk_size_in_secs = self.core_cfg.chunk_size * self.core_cfg.frame_length_in_secs
        buffer_size_in_secs = pre_encode_cache_size_in_secs + chunk_size_in_secs

        audio_feature_buffer = BatchedCacheFeatureBufferer(
            num_slots=batch_size,
            sample_rate=self.core_cfg.sample_rate,
            buffer_size_in_secs=buffer_size_in_secs,
            chunk_size_in_secs=chunk_size_in_secs,
            preprocessor_cfg=preprocessor_cfg,
            device=self.device,
        )
        return audio_feature_buffer

    def get_init_streaming_state(self, system_prompt: str, device: torch.device) -> StreamingState:
        """Forward the system prompt through the LLM and return a fresh :class:`StreamingState`."""
        hf_tok = self.tokenizer.tokenizer
        sys_ids = hf_tok.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        dtype = self.embed_tokens.weight.dtype
        sys_embs = self.embed_tokens(torch.tensor(sys_ids, device=device, dtype=torch.long).unsqueeze(0))
        out = self.llm(inputs_embeds=sys_embs, use_cache=True, return_dict=True)
        cache_last_channel, cache_last_time, cache_last_channel_len = self.perception.get_initial_cache_state(
            batch_size=1, dtype=dtype, device=device
        )
        audio_feature_buffer = self.get_audio_feature_buffer(batch_size=1)
        audio_cache = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        return StreamingState(
            cache=out.past_key_values,
            generated_tokens=[],
            seq_len=sys_embs.shape[1],
            audio_cache=audio_cache,
            audio_feature_buffer=audio_feature_buffer,
        )

    @torch.no_grad()
    def generate_streaming(
        self,
        audio_chunk: Tensor,
        audio_chunk_len: Optional[Tensor] = None,
        state: Optional[StreamingState] = None,
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        _audio_embs: Optional[Tensor] = None,
        **generation_kwargs,
    ) -> list[int]:
        """
        Process one raw audio chunk and generate the assistant response.

        Encodes ``audio_chunk`` through the perception module (using the
        streaming encoder cache stored in ``state``), then builds the user turn
        using the same ``apply_chat_template`` + ``AUDIO_TOKEN_IDX`` replacement
        pattern as :class:`StreamingSTTDataset`, forwards through the LLM with
        the KV cache in ``state``, runs autoregressive decoding until
        end-of-turn or ``max_new_tokens``, then finalizes the assistant turn in
        the cache.

        Args:
            audio_chunk: (1, T_samples) raw waveform for one chunk.
            audio_chunk_len: (1,) number of valid samples.
            state: Mutable :class:`StreamingState` (updated in place).
            max_new_tokens: Maximum tokens to generate per chunk.
            generation_config: Optional HuggingFace ``GenerationConfig``.
                Supports ``temperature``, ``top_k``, ``top_p``, ``do_sample``.
            _audio_embs: Optional pre-computed audio embeddings (1, chunk_size, H).
                When provided, bypasses both the feature buffer and the
                perception module entirely.  Diagnostic use only.
            generation_kwargs: Per-call overrides for generation parameters.
        Returns:
            List of generated token IDs (excluding end-of-turn tokens).
        """

        self._ensure_inference_cache()
        device = audio_chunk.device

        if _audio_embs is not None:
            # Diagnostic: skip feature buffer + perception entirely.
            audio_chunk_embs = _audio_embs.type_as(self.embed_tokens.weight)
        else:
            # 0. Update audio feature buffer
            audio_chunk = audio_chunk.view(-1)
            if audio_chunk_len is None:
                audio_chunk_len = audio_chunk.shape[0]
            frame = Frame(
                samples=audio_chunk,
                length=audio_chunk_len,
                stream_id=0,
            )
            # features: list of (D, T) tensors
            # right_paddings: list of ints
            features, right_paddings = state.audio_feature_buffer.update([frame])
            processed_signal = features[0].unsqueeze(0).type_as(self.embed_tokens.weight)
            processed_signal_length = torch.tensor(
                [processed_signal.shape[-1] - int(right_paddings[0])], device=device
            ).long()

            # 1. Encode audio chunk with streaming cache
            outputs = self.perception(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=state.audio_cache.cache_last_channel,
                cache_last_time=state.audio_cache.cache_last_time,
                cache_last_channel_len=state.audio_cache.cache_last_channel_len,
                streaming=True,
            )
            audio_chunk_embs, audio_chunk_emb_lens, new_perception_cache = outputs

            # 2. Update streaming state with new perception cache
            if new_perception_cache is not None:
                state.audio_cache.cache_last_channel = new_perception_cache['cache_last_channel']
                state.audio_cache.cache_last_time = new_perception_cache['cache_last_time']
                state.audio_cache.cache_last_channel_len = new_perception_cache['cache_last_channel_len']

        # 3. Pad/trim to chunk_size frames (template expects exactly chunk_size audio slots)
        chunk_size = self.core_cfg.chunk_size
        n_frames = audio_chunk_embs.shape[1]
        if n_frames < chunk_size:
            audio_chunk_embs = F.pad(audio_chunk_embs, (0, 0, 0, chunk_size - n_frames))
        elif n_frames > chunk_size:
            audio_chunk_embs = audio_chunk_embs[:, :chunk_size, :]

        # 1. Build input embeddings from cached turn template
        turn_ids_t = torch.tensor(self._turn_template_ids, device=device).unsqueeze(0)  # (1, L)
        audio_mask = turn_ids_t == AUDIO_TOKEN_IDX  # (1, L)

        # Embed text tokens (zero out audio positions for valid embedding lookup)
        text_tokens = turn_ids_t.where(~audio_mask, torch.zeros_like(turn_ids_t))
        input_embeds = self.embed_tokens(text_tokens)  # (1, L, H)

        # Replace audio placeholder positions with actual audio embeddings
        input_embeds[audio_mask] = audio_chunk_embs.reshape(-1, audio_chunk_embs.shape[-1])

        # 2. Forward through LLM with cache
        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=state.cache,
            use_cache=True,
            return_dict=True,
        )
        state.cache = out.past_key_values
        state.seq_len += input_embeds.shape[1]

        # 3. Autoregressive generation loop
        generated, state.cache, footer_consumed = self._streaming_decode(
            out.logits, state.cache, max_new_tokens, generation_config, **generation_kwargs
        )
        state.seq_len += len(generated)

        # 4. Finalize turn — ensure end-of-turn tokens are in the cache.
        #    If _greedy_decode already consumed the footer (matched the full
        #    sequence), it is already in the KV cache; only update seq_len.
        #    Otherwise, feed the footer explicitly.
        if footer_consumed:
            state.seq_len += len(self._asst_footer_ids)
        elif self._asst_footer_ids:
            asst_footer_embs = self.embed_tokens(torch.tensor(self._asst_footer_ids, device=device).unsqueeze(0))
            out = self.llm(
                inputs_embeds=asst_footer_embs,
                past_key_values=state.cache,
                use_cache=True,
                return_dict=True,
            )
            state.cache = out.past_key_values
            state.seq_len += asst_footer_embs.shape[1]

        # 5. Store and return
        state.generated_tokens.append(generated)
        return generated

    def _generate_streaming_sample(
        self,
        audio_wav_b: Tensor,
        n_samples: int,
        system_prompt: str,
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        **generation_kwargs,
    ) -> str:
        """Chunk-by-chunk streaming generation for one sample.

        Args:
            audio_wav_b: (T_samples,) 1-D raw waveform for one sample.
            n_samples: Number of valid samples in ``audio_wav_b``.
            system_prompt: System prompt for the LLM.
            max_new_tokens: Maximum tokens to generate per chunk.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            use_offline_embs: When ``True``, pre-compute perception embeddings
                on the full audio offline and slice them into ``chunk_size``
                frame groups, bypassing both the feature buffer **and** the
                streaming encoder.  This isolates the LLM / generation logic
                from perception entirely.  Diagnostic use only.
            generation_kwargs: Per-call overrides for generation parameters.
        """
        if n_samples == 0:
            return ""
        device = audio_wav_b.device
        chunk_size = self.core_cfg.chunk_size
        chunk_samples = math.ceil(chunk_size * self.core_cfg.frame_length_in_secs * self.core_cfg.sample_rate)
        state = self.get_init_streaming_state(system_prompt, device=device)

        # Optionally pre-compute offline perception embeddings.
        offline_emb_chunks = None
        if use_offline_embs:
            offline_emb_chunks = self._build_offline_emb_chunks(audio_wav_b[:n_samples], n_samples, device)

        num_chunks = math.ceil(n_samples / chunk_samples) if n_samples > 0 else 0
        all_token_ids = []
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, n_samples)
            chunk_wav = audio_wav_b[start:end].unsqueeze(0)  # (1, T)
            chunk_len = torch.tensor([end - start], device=device)  # (1,)

            # zero padding to chunk_samples
            if chunk_wav.shape[1] < chunk_samples:
                chunk_wav = F.pad(chunk_wav, (0, chunk_samples - chunk_wav.shape[1]))
            extra_kwargs = {}
            if offline_emb_chunks is not None and i < len(offline_emb_chunks):
                extra_kwargs["_audio_embs"] = offline_emb_chunks[i]

            all_token_ids.extend(
                self.generate_streaming(
                    chunk_wav,
                    chunk_len,
                    state,
                    max_new_tokens,
                    generation_config,
                    **extra_kwargs,
                    **generation_kwargs,
                )
            )

        return decode_with_blank(all_token_ids, self.blank_token, self.tokenizer)

    def _build_offline_emb_chunks(
        self,
        audio_wav: Tensor,
        n_samples: int,
        device: torch.device,
    ) -> list[Tensor]:
        """Pre-compute offline perception embeddings and slice into chunk_size groups.

        Runs the full perception module on the complete audio (the same path
        used during training), then splits the resulting embeddings into
        ``chunk_size``-frame groups that can be fed directly to the LLM turn
        template.  This bypasses both the feature buffer and the streaming
        encoder, isolating the LLM / generation logic from perception.

        Returns a list of ``(1, chunk_size, H)`` tensors, one per chunk.
        """
        chunk_size = self.core_cfg.chunk_size
        with torch.no_grad():
            offline_embs, offline_emb_lens = self.perception(
                input_signal=audio_wav.unsqueeze(0),
                input_signal_length=torch.tensor([n_samples], device=device),
            )
        # offline_embs: (1, T_total, H)
        total_frames = offline_embs.shape[1]
        chunks: list[Tensor] = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = offline_embs[:, start:end, :]  # (1, <=chunk_size, H)
            # Pad last chunk if needed so the template always gets chunk_size slots.
            if chunk.shape[1] < chunk_size:
                chunk = F.pad(chunk, (0, 0, 0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)
        return chunks

    def _generate_oneshot_sample(
        self,
        audio_embs_b: Tensor,
        n_frames: int,
        system_prompt: str,
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> str:
        """One-shot generation: feed all audio in a single user turn.

        When *generation_config* or *generation_kwargs* are provided, delegates
        to ``self.llm.generate()`` for full HuggingFace generation support
        (beam search, diverse decoding, etc.).  Otherwise uses the custom
        :meth:`_decode` loop (greedy by default).
        """
        if n_frames == 0:
            return ""
        hf_tok = self.tokenizer.tokenizer
        device = audio_embs_b.device
        _SENTINEL = "XSENTINELX"

        # Build input with system prompt + user turn (sentinel placeholder) + generation prompt
        input_ids = list(
            hf_tok.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _SENTINEL},
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        sentinel_ids = hf_tok.encode(_SENTINEL, add_special_tokens=False)
        sentinel_start = _find_sublist(input_ids, sentinel_ids)
        assert sentinel_start is not None, "Could not find sentinel in one-shot input template"
        input_ids = (
            input_ids[:sentinel_start] + [AUDIO_TOKEN_IDX] * n_frames + input_ids[sentinel_start + len(sentinel_ids) :]
        )

        # Build embeddings — same interleave pattern as training
        input_ids_t = torch.tensor(input_ids, device=device).unsqueeze(0)
        audio_mask = input_ids_t == AUDIO_TOKEN_IDX
        text_tokens = input_ids_t.where(~audio_mask, torch.zeros_like(input_ids_t))
        input_embeds = self.embed_tokens(text_tokens)
        input_embeds[audio_mask] = audio_embs_b[:n_frames]

        if generation_config is not None or generation_kwargs:
            # Delegate to HF generate() for full generation support
            output_ids = self.llm.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
                **generation_kwargs,
            )
            input_len = input_embeds.shape[1]
            generated = output_ids[0, input_len:].tolist()
            # Strip trailing EOS / footer tokens that HF generate may include
            if generated and self._eos_id is not None and generated[-1] == self._eos_id:
                generated.pop()
            flen = len(self._asst_footer_ids)
            if flen > 0 and len(generated) >= flen and generated[-flen:] == self._asst_footer_ids:
                generated = generated[:-flen]
        else:
            # Default: custom decode loop (greedy)
            out = self.llm(inputs_embeds=input_embeds, use_cache=True, return_dict=True)
            generated, _, _ = self._streaming_decode(out.logits, out.past_key_values, max_new_tokens)

        return self.tokenizer.ids_to_text(generated)

    @torch.no_grad()
    def generate(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        system_prompt: Union[str, List[str]] = "Transcribe the audio into text.",
        simulate_streaming: bool = True,
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        **generation_kwargs,
    ) -> list[str]:
        """
        Transcribe full audio(s).

        Args:
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
            system_prompt: System prompt for the LLM.
            max_new_tokens: Maximum tokens to generate per chunk/turn.
            simulate_streaming: When ``True``, processes audio chunk-by-chunk
                (simulating real-time streaming with alternating user/assistant
                turns and ``<blank>`` tokens).  When ``False`` (default), feeds
                all audio in a single user turn for direct transcription.
            generation_config: Optional HuggingFace GenerationConfig object.
            use_offline_embs: When ``True`` and ``simulate_streaming`` is also
                ``True``, pre-compute perception embeddings on the full audio
                offline and slice them into ``chunk_size``-frame groups,
                bypassing both the feature buffer and the streaming encoder.
                Diagnostic use only.
            generation_kwargs: Keyword arguments passed directly to the underlying LLM's ``generate`` method.

        Returns:
            List of transcription strings, one per sample.
        """
        self._ensure_inference_cache()

        with move_embedding(self):
            B = audios.shape[0]
            if isinstance(system_prompt, str):
                system_prompt = [system_prompt] * B

            # Only encode all audio upfront for the one-shot path
            audio_embs, audio_emb_lens = None, None
            if not simulate_streaming:
                audio_embs, audio_emb_lens = self.perception(
                    input_signal=audios,
                    input_signal_length=audio_lens,
                )  # (B, T_total_frames, H)

            results = []
            for b in range(B):
                if simulate_streaming:
                    n_samples = int(audio_lens[b].item())
                    text = self._generate_streaming_sample(
                        audios[b],
                        n_samples,
                        system_prompt[b],
                        max_new_tokens,
                        generation_config,
                        use_offline_embs=use_offline_embs,
                        **generation_kwargs,
                    )
                else:
                    n_frames = int(audio_emb_lens[b].item())
                    text = self._generate_oneshot_sample(
                        audio_embs[b],
                        n_frames,
                        system_prompt[b],
                        max_new_tokens,
                        generation_config,
                        **generation_kwargs,
                    )
                results.append(text)

        return results
