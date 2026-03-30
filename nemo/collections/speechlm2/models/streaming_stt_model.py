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
    """Holds the KV cache and other state for B streaming audio sessions.

    All tensors have batch dimension B (B=1 for single-stream inference).
    The LLM cache ``past_key_values`` has shape ``(layers, (B, heads, seq, dim))``
    for K and V.  The perception cache has batch dim on axis 1.
    """

    cache: tuple | None = None  # HF past_key_values with batch dim B
    generated_tokens: list[list[int]] = field(default_factory=list)  # B lists of per-chunk token IDs
    seq_lens: list[int] = field(default_factory=list)  # per-stream sequence lengths
    audio_cache: CacheAwareContext | None = None  # perception cache with batch dim B
    audio_feature_buffer: BatchedCacheFeatureBufferer | None = None
    batch_size: int = 1

    @property
    def seq_len(self) -> int:
        """Max seq_len across streams (= KV cache dimension)."""
        return max(self.seq_lens) if self.seq_lens else 0


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
        import pdb

        pdb.set_trace()
        if self.blank_token not in self.tokenizer.tokenizer.get_vocab():
            logging.info(f"Adding blank token `{self.blank_token}` to tokenizer")
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.blank_token]})
            self.llm.resize_token_embeddings(len(self.tokenizer.tokenizer))
        else:
            logging.info(
                f"Blank token `{self.blank_token}` already in tokenizer: {self.tokenizer.text_to_ids(self.blank_token)}"
            )

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
        generated_ids: list[list[int]] | list[int] | None = None,
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
            logits: ``(B, vocab_size)`` logits for the last position.
            generated_ids: Token IDs generated so far.  For B=1, a flat list.
                For B>1, a list of B lists (one per stream).  Used for
                repetition-aware transforms.  May be ``None`` or empty.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(B,)`` tensor with the selected token IDs.
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
        state: Optional['StreamingState'],
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        **generation_kwargs,
    ) -> tuple[list[list[int]], tuple, list[bool], int]:
        """Autoregressive decoding for the streaming path (supports B streams).

        Token selection is delegated to :meth:`_sample_token`, which supports
        greedy (default), sampling (temperature / top-k / top-p), repetition
        penalty, no-repeat-ngram blocking, and token suppression.

        Generation stops per stream when any of these conditions is met:

        1. **EOS** — the tokenizer's ``eos_token_id`` is predicted.
        2. **Blank** — the ``<blank>`` token is predicted.
        3. **Footer sequence** — the last *N* tokens match ``self._asst_footer_ids``.
        4. **Max tokens** — ``max_new_tokens`` is reached.

        Args:
            logits: ``(B, L, V)`` logits from the LLM forward pass.
            cache: HF ``past_key_values`` with batch dim B.
            max_new_tokens: Maximum tokens to generate per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            generation_kwargs: Per-call overrides.

        Returns:
            ``(generated_per_stream, updated_cache, footer_consumed_per_stream, num_feed_steps)``
            where ``generated_per_stream`` is a list of B token-ID lists and
            ``num_feed_steps`` is how many tokens were fed to the LLM cache.
        """
        B = logits.shape[0]
        footer = self._asst_footer_ids
        flen = len(footer)
        generated: list[list[int]] = [[] for _ in range(B)]
        footer_consumed = [False] * B
        finished = [False] * B
        num_feed_steps = 0
        # Track which streams need their token fed to the LLM this step.
        # EOS tokens must NOT be fed; blank/footer/normal tokens must be fed.
        feed_mask = [False] * B

        next_tokens = self._sample_token(logits[:, -1, :], None, generation_config, **generation_kwargs)  # (B,)

        for _ in range(max_new_tokens):
            for b in range(B):
                feed_mask[b] = False
                if finished[b]:
                    continue
                tid = next_tokens[b].item()

                # EOS: stop WITHOUT feeding to LLM (matches old behavior exactly)
                if self._eos_id is not None and tid == self._eos_id:
                    finished[b] = True
                    continue

                # All other tokens get appended and fed to LLM
                generated[b].append(tid)
                feed_mask[b] = True

                # Blank: stop (token IS fed to LLM, IS in generated)
                if tid == self._blank_id:
                    finished[b] = True

                # Footer sequence match
                elif flen > 0 and len(generated[b]) >= flen and generated[b][-flen:] == footer:
                    generated[b] = generated[b][:-flen]
                    footer_consumed[b] = True
                    finished[b] = True

            # If no stream needs feeding, we're done
            if not any(feed_mask):
                break

            # Feed tokens to LLM. For finished streams, feed the blank token
            # (which the model was trained on) instead of a pad token, so the
            # KV cache stays clean — no foreign tokens that corrupt attention.
            tokens_to_feed = next_tokens.clone()
            for b in range(B):
                if not feed_mask[b]:
                    tokens_to_feed[b] = self._blank_id

            # All tokens are "real" (blank is a valid token), so all seq_lens grow
            if state is not None:
                for b in range(B):
                    state.seq_lens[b] += 1

            token_emb = self.embed_tokens(tokens_to_feed.unsqueeze(1))  # (B, 1, H)
            out = self.llm(
                inputs_embeds=token_emb,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = out.past_key_values
            num_feed_steps += 1

            if all(finished):
                break

            next_tokens = self._sample_token(out.logits[:, -1, :], None, generation_config, **generation_kwargs)

        return generated, cache, footer_consumed, num_feed_steps

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

    def get_init_streaming_state(
        self,
        system_prompt: str,
        device: torch.device,
        batch_size: int = 1,
    ) -> StreamingState:
        """Forward the system prompt through the LLM and return a fresh :class:`StreamingState`.

        Args:
            system_prompt: System prompt (same for all B streams).
            device: Target device.
            batch_size: Number of parallel streams (B).
        """
        hf_tok = self.tokenizer.tokenizer
        sys_ids = hf_tok.apply_chat_template(
            [{"role": "system", "content": system_prompt}],
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        dtype = self.embed_tokens.weight.dtype
        # (1, L_sys, H) → expand to (B, L_sys, H)
        sys_embs = self.embed_tokens(torch.tensor(sys_ids, device=device, dtype=torch.long).unsqueeze(0))
        sys_embs = sys_embs.expand(batch_size, -1, -1)
        out = self.llm(inputs_embeds=sys_embs, use_cache=True, return_dict=True)
        cache_last_channel, cache_last_time, cache_last_channel_len = self.perception.get_initial_cache_state(
            batch_size=batch_size, dtype=dtype, device=device
        )
        audio_feature_buffer = self.get_audio_feature_buffer(batch_size=batch_size)
        audio_cache = CacheAwareContext(
            cache_last_channel=cache_last_channel,
            cache_last_time=cache_last_time,
            cache_last_channel_len=cache_last_channel_len,
        )
        sys_len = sys_embs.shape[1]
        return StreamingState(
            cache=out.past_key_values,
            generated_tokens=[[] for _ in range(batch_size)],
            seq_lens=[sys_len] * batch_size,
            audio_cache=audio_cache,
            audio_feature_buffer=audio_feature_buffer,
            batch_size=batch_size,
        )

    @torch.no_grad()
    def generate_streaming(
        self,
        audio_chunks: Tensor,
        audio_chunk_lens: Optional[Tensor] = None,
        state: Optional[StreamingState] = None,
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        _audio_embs: Optional[Tensor] = None,
        **generation_kwargs,
    ) -> list[list[int]]:
        """
        Process B raw audio chunks and generate the assistant responses.

        Args:
            audio_chunks: ``(B, T_samples)`` raw waveforms for one chunk per stream.
            audio_chunk_lens: ``(B,)`` number of valid samples per stream.
            state: Mutable :class:`StreamingState` with ``batch_size=B`` (updated in place).
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            _audio_embs: Optional pre-computed audio embeddings ``(B, chunk_size, H)``.
                Diagnostic use only.
            generation_kwargs: Per-call overrides for generation parameters.
        Returns:
            List of B token-ID lists (one per stream).
        """

        self._ensure_inference_cache()
        device = audio_chunks.device
        B = state.batch_size

        if _audio_embs is not None:
            audio_chunk_embs = _audio_embs.type_as(self.embed_tokens.weight)
        else:
            # 0. Update audio feature buffer — B frames, one per stream
            if audio_chunk_lens is None:
                audio_chunk_lens = torch.tensor([audio_chunks.shape[-1]] * B, device=device)
            frames = [
                Frame(
                    samples=audio_chunks[b] if audio_chunks.dim() == 2 else audio_chunks,
                    length=int(audio_chunk_lens[b].item()),
                    stream_id=b,
                )
                for b in range(B)
            ]
            features, right_paddings = state.audio_feature_buffer.update(frames)
            # Stack B feature buffers → (B, D, fbl)
            processed_signal = torch.stack(features).type_as(self.embed_tokens.weight)
            processed_signal_length = torch.tensor(
                [processed_signal.shape[-1] - int(rp) for rp in right_paddings],
                device=device,
            ).long()

            # 1. Encode audio chunks with streaming cache
            outputs = self.perception(
                processed_signal=processed_signal,
                processed_signal_length=processed_signal_length,
                cache_last_channel=state.audio_cache.cache_last_channel,
                cache_last_time=state.audio_cache.cache_last_time,
                cache_last_channel_len=state.audio_cache.cache_last_channel_len,
                streaming=True,
            )
            audio_chunk_embs, _, new_perception_cache = outputs

            # 2. Update streaming state with new perception cache
            if new_perception_cache is not None:
                state.audio_cache.cache_last_channel = new_perception_cache['cache_last_channel']
                state.audio_cache.cache_last_time = new_perception_cache['cache_last_time']
                state.audio_cache.cache_last_channel_len = new_perception_cache['cache_last_channel_len']

        # 3. Pad/trim to chunk_size frames
        chunk_size = self.core_cfg.chunk_size
        n_frames = audio_chunk_embs.shape[1]
        if n_frames < chunk_size:
            audio_chunk_embs = F.pad(audio_chunk_embs, (0, 0, 0, chunk_size - n_frames))
        elif n_frames > chunk_size:
            audio_chunk_embs = audio_chunk_embs[:, :chunk_size, :]

        # 4. Build input embeddings from cached turn template — (B, L, H)
        turn_ids_t = torch.tensor(self._turn_template_ids, device=device).unsqueeze(0).expand(B, -1)  # (B, L)
        audio_mask = turn_ids_t == AUDIO_TOKEN_IDX  # (B, L)

        text_tokens = turn_ids_t.where(~audio_mask, torch.zeros_like(turn_ids_t))
        input_embeds = self.embed_tokens(text_tokens)  # (B, L, H)

        # Replace audio placeholder positions with actual audio embeddings
        input_embeds[audio_mask] = audio_chunk_embs.reshape(-1, audio_chunk_embs.shape[-1])

        # 5. Forward through LLM with cache
        input_len = input_embeds.shape[1]
        out = self.llm(
            inputs_embeds=input_embeds,
            past_key_values=state.cache,
            use_cache=True,
            return_dict=True,
        )
        state.cache = out.past_key_values
        for b in range(B):
            state.seq_lens[b] += input_len

        # 6. Autoregressive generation loop
        generated_per_stream, state.cache, footer_consumed, num_feed_steps = self._streaming_decode(
            out.logits, state.cache, state, max_new_tokens, generation_config, **generation_kwargs
        )

        # 7. Finalize turn — ensure end-of-turn tokens are in the cache.
        any_needs_footer = any(not fc for fc in footer_consumed)
        if any_needs_footer and self._asst_footer_ids:
            flen = len(self._asst_footer_ids)
            asst_footer_embs = self.embed_tokens(
                torch.tensor(self._asst_footer_ids, device=device).unsqueeze(0).expand(B, -1)
            )
            out = self.llm(
                inputs_embeds=asst_footer_embs,
                past_key_values=state.cache,
                use_cache=True,
                return_dict=True,
            )
            state.cache = out.past_key_values
            for b in range(B):
                state.seq_lens[b] += flen
        elif all(footer_consumed):
            for b in range(B):
                state.seq_lens[b] += len(self._asst_footer_ids)

        # 8. Store and return
        for b in range(B):
            state.generated_tokens[b].extend(generated_per_stream[b])
        return generated_per_stream

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
            offline_embs, _ = self.perception(
                input_signal=audio_wav.unsqueeze(0),
                input_signal_length=torch.tensor([n_samples], device=device),
            )
        total_frames = offline_embs.shape[1]
        chunks: list[Tensor] = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = offline_embs[:, start:end, :]
            if chunk.shape[1] < chunk_size:
                chunk = F.pad(chunk, (0, 0, 0, chunk_size - chunk.shape[1]))
            chunks.append(chunk)
        return chunks

    def _generate_streaming_samples(
        self,
        audios: Tensor,
        n_samples_list: list[int],
        system_prompt: str,
        max_new_tokens: int,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        **generation_kwargs,
    ) -> list[str]:
        """Chunk-by-chunk streaming generation for B samples in lockstep.

        Args:
            audios: ``(B, T_samples)`` raw waveforms (zero-padded to max length).
            n_samples_list: List of B valid sample counts.
            system_prompt: System prompt for the LLM (same for all streams).
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace ``GenerationConfig``.
            use_offline_embs: When True, bypass streaming perception with offline embeddings.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of B transcription strings.
        """
        B = len(n_samples_list)
        if B == 0 or max(n_samples_list) == 0:
            return [""] * B
        device = audios.device
        chunk_size = self.core_cfg.chunk_size
        chunk_samples = math.ceil(chunk_size * self.core_cfg.frame_length_in_secs * self.core_cfg.sample_rate)
        state = self.get_init_streaming_state(system_prompt, device=device, batch_size=B)

        offline_emb_chunks_list = None
        if use_offline_embs:
            offline_emb_chunks_list = [
                self._build_offline_emb_chunks(audios[b, : n_samples_list[b]], n_samples_list[b], device)
                for b in range(B)
            ]

        num_chunks_per_stream = [math.ceil(ns / chunk_samples) if ns > 0 else 0 for ns in n_samples_list]
        max_chunks = max(num_chunks_per_stream)
        all_token_ids: list[list[int]] = [[] for _ in range(B)]

        for chunk_i in range(max_chunks):
            # Build B audio chunks (zero-pad finished streams)
            chunks = []
            chunk_lens = []
            for b in range(B):
                start = chunk_i * chunk_samples
                end = min(start + chunk_samples, n_samples_list[b])
                if start >= n_samples_list[b]:
                    # Stream b has finished — send zeros with zero valid length
                    chunks.append(torch.zeros(chunk_samples, device=device, dtype=audios.dtype))
                    chunk_lens.append(0)
                else:
                    wav = audios[b, start:end]
                    if wav.shape[0] < chunk_samples:
                        wav = F.pad(wav, (0, chunk_samples - wav.shape[0]))
                    chunks.append(wav)
                    chunk_lens.append(end - start)

            audio_batch = torch.stack(chunks)  # (B, chunk_samples)
            lens_batch = torch.tensor(chunk_lens, device=device)

            extra_kwargs = {}
            if offline_emb_chunks_list is not None:
                emb_chunks = []
                for b in range(B):
                    if chunk_i < len(offline_emb_chunks_list[b]):
                        emb_chunks.append(offline_emb_chunks_list[b][chunk_i])
                    else:
                        H = offline_emb_chunks_list[0][0].shape[-1]
                        emb_chunks.append(torch.zeros(1, chunk_size, H, device=device, dtype=audios.dtype))
                extra_kwargs["_audio_embs"] = torch.cat(emb_chunks, dim=0)

            chunk_tokens = self.generate_streaming(
                audio_batch,
                lens_batch,
                state,
                max_new_tokens,
                generation_config,
                **extra_kwargs,
                **generation_kwargs,
            )
            for b in range(B):
                # Only collect tokens for streams that are still active
                if chunk_i < num_chunks_per_stream[b]:
                    all_token_ids[b].extend(chunk_tokens[b])

        return [decode_with_blank(toks, self.blank_token, self.tokenizer) for toks in all_token_ids]

    @torch.no_grad()
    def generate(
        self,
        audios: Tensor,
        audio_lens: Tensor,
        system_prompt: Union[str, List[str]] = "Transcribe the audio into text.",
        max_new_tokens: int = 64,
        generation_config: Optional[GenerationConfig] = None,
        use_offline_embs: bool = False,
        **generation_kwargs,
    ) -> list[str]:
        """
        Transcribe full audio(s) using chunk-by-chunk streaming.

        All B samples are processed in lockstep — each step, every stream
        contributes one audio chunk, and the perception + LLM forward passes
        are batched across streams.

        Args:
            audios: (B, T_samples) raw waveforms.
            audio_lens: (B,) waveform lengths in samples.
            system_prompt: System prompt for the LLM (same for all streams).
            max_new_tokens: Maximum tokens to generate per chunk per stream.
            generation_config: Optional HuggingFace GenerationConfig object.
            use_offline_embs: When True, bypass streaming perception with
                offline embeddings. Diagnostic use only.
            generation_kwargs: Per-call overrides for generation parameters.

        Returns:
            List of transcription strings, one per sample.
        """
        self._ensure_inference_cache()

        with move_embedding(self):
            B = audios.shape[0]
            if isinstance(system_prompt, str):
                sys_prompt = system_prompt
            else:
                sys_prompt = system_prompt[0]

            n_samples_list = [int(audio_lens[b].item()) for b in range(B)]
            results = self._generate_streaming_samples(
                audios,
                n_samples_list,
                sys_prompt,
                max_new_tokens,
                generation_config,
                use_offline_embs=use_offline_embs,
                **generation_kwargs,
            )

        return results
