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
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.model_summary import ModelSummary
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.tensor.parallel import loss_parallel
from transformers import GenerationConfig

from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.streaming_stt_dataset import (
    AUDIO_TOKEN_IDX,
    IGNORE_INDEX,
    StreamingSTTBatch,
    decode_with_blank,
)
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding, setup_perception
from nemo.collections.speechlm2.parts.utils import freeze_module, to_dataclass, unfreeze_module
from nemo.utils import logging


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
    att_context_size: Optional[List[int]] = None
    audio_pad_to: Optional[int] = None
    log_every_n_steps: int = 10


class StreamingSTTModel(LightningModule, HFHubMixin):

    def __init__(self, cfg: dict) -> None:
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
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_tokens: Tensor,
        audios: Tensor | None = None,
        audio_lens: Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        **generation_kwargs,
    ) -> Tensor:
        """
        Generate text from (optionally) audio-augmented token prompts.

        Args:
            input_tokens: (B, L) token IDs with ``AUDIO_TOKEN_IDX`` at audio
                positions (if audio is provided).
            audios: (B, T_samples) raw waveforms, or *None* for text-only.
            audio_lens: (B,) waveform lengths in samples.
            generation_config: HuggingFace ``GenerationConfig``.
            **generation_kwargs: forwarded to ``self.llm.generate()``.

        Returns:
            Generated token IDs tensor.
        """
        if audios is not None:
            inputs = self._build_input_embeds(input_tokens, audios, audio_lens)
            generation_inputs = {
                "inputs_embeds": inputs["input_embeds"],
                "attention_mask": inputs["attention_mask"],
            }
        else:
            generation_inputs = {
                "input_ids": input_tokens,
                "attention_mask": input_tokens != self.text_pad_id,
            }

        if generation_config is None:
            generation_config = GenerationConfig(
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                pad_token_id=self.text_pad_id,
            )

        # Temporarily restore embed_tokens inside the LLM so HF's generate()
        # can embed newly produced tokens during autoregressive decoding.
        with move_embedding(self):
            answer_tokens = self.llm.generate(
                **generation_inputs,
                **generation_kwargs,
                generation_config=generation_config,
            )
        return answer_tokens
