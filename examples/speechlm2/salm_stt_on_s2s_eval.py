# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass
import os
from time import perf_counter
from typing import Optional
from collections import defaultdict
import time
import json

import lhotse.dataset
import torch
from lhotse import CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.serialization import SequentialJsonlWriter
from omegaconf import OmegaConf
from transformers import GenerationConfig
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.metrics.wer import word_error_rate_detail
from nemo.collections.common.data.lhotse.text_adapters import (
    AudioTurn,
    TextTurn,
    NeMoMultimodalConversation
)
from nemo.collections.common.data.lhotse.cutset import guess_parse_cutset
from nemo.collections.speechlm2 import SALM
from nemo.core.config import hydra_runner
from nemo.utils import logging

def save_jsonl(data, json_fn):
    os.makedirs(os.path.dirname(json_fn), exist_ok=True)
    with open(json_fn, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

class ToAudio(torch.utils.data.Dataset):
    def __getitem__(self, cuts: CutSet):
        # return {"cuts": cuts}
        all_cuts = []
        ref_agent_text = []
        ref_user_text = []
        example_idx_to_audio_idxs = []
        cntr = 0
        for conversation in cuts:
            assert isinstance(conversation, NeMoMultimodalConversation)
            conv_id = conversation.id
            example_idx_to_audio_idxs.append([])
            ref_agent_text.extend([(conv_id, turn.value) for turn in conversation.turns if isinstance(turn, TextTurn)])
            ref_user_text.extend([(conv_id, turn.cut.supervisions[0].text) for turn in conversation.turns if isinstance(turn, AudioTurn)])
            for cut in conversation.list_cuts():
                all_cuts.append(cut)
                example_idx_to_audio_idxs[-1].append(cntr)
                cntr += 1
        audios, audio_lens = collate_audio(CutSet(all_cuts))
        return {
            "cuts": CutSet(all_cuts),
            "audios": audios,
            "audio_lens": audio_lens,
            "agent_ref": ref_agent_text,
            "user_ref": ref_user_text
            }

@dataclass
class SalmEvalConfig:
    pretrained_name: str
    inputs: str
    batch_size: int = 64
    max_new_tokens: int = 128
    output_manifest: Optional[str] = "generations.jsonl"
    verbose: bool = True
    use_normalizer: bool = True
    device: str = "cuda"
    extra_eos_tokens: Optional[list[str]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None


@hydra_runner(config_name="SalmEvalConfig", schema=SalmEvalConfig)
def main(cfg: SalmEvalConfig):
    logging.info(f'Hydra config:\n{OmegaConf.to_yaml(cfg)}')

    with torch.device(cfg.device):
        torch.set_default_dtype(torch.bfloat16)
        if cfg.pretrained_name.endswith('.ckpt'):
            # For local checkpoint files
            model = SALM.load_from_checkpoint(cfg.pretrained_name).eval().to(torch.bfloat16).to(torch.device(cfg.device))
        else:
            # For Hugging Face model identifiers
            model = SALM.from_pretrained(cfg.pretrained_name).eval().to(torch.bfloat16).to(torch.device(cfg.device))
        torch.set_default_dtype(torch.float32)

    # Add audio_locator_tag, input_roles, and output_roles to the input config
    input_config = OmegaConf.load(cfg.inputs)
    for cfg_entry in input_config:
        assert cfg_entry.type == "group"
        for entry in cfg_entry.input_cfg:
            if entry.type in ["s2s_as_conversation", "lhotse_as_conversation"]:
                entry.audio_locator_tag = model.cfg.audio_locator_tag
                entry.input_roles = ["user", "User"]
                entry.output_roles = ["agent", "Assistant", "assistant"]
                entry.force_finite = True  # Make the dataset finite
    
    # Save modified config
    modified_inputs = cfg.inputs.replace(".yaml", "_modified.yaml")
    with open(modified_inputs, "w") as f:
        f.write(OmegaConf.to_yaml(input_config))
    # cuts = guess_parse_cutset(cfg.inputs).sort_by_duration()  # Not compatible with lazy loading of SHAR data
    cuts = guess_parse_cutset(modified_inputs)
    
    dloader = torch.utils.data.DataLoader(
        dataset=ToAudio(),
        sampler=lhotse.dataset.DynamicCutSampler(
            cuts, 
            max_cuts=cfg.batch_size,
            shuffle=False,  # Ensure deterministic ordering
            drop_last=False,  # Don't drop the last batch even if it's smaller
        ),
        num_workers=1,
        batch_size=None,
    )
    logging.info(f"Created DataLoader with batch_size={cfg.batch_size}")

    if cfg.use_normalizer:
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = lambda x: x

    eos_tokens = [model.text_eos_id]
    if cfg.extra_eos_tokens is not None:
        for t in cfg.extra_eos_tokens:
            tid = model.tokenizer.token_to_id(t)
            assert tid is not None, f"Token '{t}' is not in the model's vocabulary."
            eos_tokens.append(tid)

    system_prompt = []
    if cfg.system_prompt is not None:
        system_prompt.append({"role": "system", "slots": {"message": cfg.system_prompt}})
    if cfg.user_prompt is not None:
        user_content = cfg.user_prompt # Example: "Repeat after me, typing in lowercase. "
    else:
        user_content = ""
    refs = []
    hyps = []
    input_durations = []
    infer_durations = []
    per_conv_data = {}
    jsonl_data_single_turn = []
    total_samples_processed = 0
    processed_ids = set()  # Track unique sample IDs
    print("start")
    for batch_idx, batch in enumerate(dloader):
        # breakpoint()
        # if batch_idx > 2:
        #     break
        ts = perf_counter()
        # Track unique samples in this batch
        batch_ids = {cut.id for cut in batch['cuts']}
        processed_ids.update(batch_ids)
        logging.info(f"Processing batch {batch_idx} with {len(batch['cuts'])} cuts")
        logging.info(f"Unique samples processed so far: {len(processed_ids)}")
        total_samples_processed += len(batch['cuts'])
        logging.info(f"Total samples processed so far: {total_samples_processed}")
        if user_prompt := cfg.user_prompt:
            system_prompt.append({"role": "user", "slots": {"message": user_prompt}})
        answer_ids = model.generate(
            prompts=[
                system_prompt
                + [
                    {
                        "role": "user",
                        "content": f"{user_content}{model.audio_locator_tag}",
                    }
                ]
            ]
            * len(batch["cuts"]),
            audios=batch["audios"].to(model.device, non_blocking=True),
            audio_lens=batch["audio_lens"].to(model.device, non_blocking=True),
            generation_config=GenerationConfig(
                max_new_tokens=cfg.max_new_tokens,
                bos_token_id=model.text_bos_id,
                eos_token_id=eos_tokens,
                pad_token_id=model.text_pad_id,
            ),
        )
        answer_ids = answer_ids.cpu()
        batch_infer_duration = perf_counter() - ts

        batch_duration = sum(c.duration for c in batch["cuts"])
        # batch_user_refs = [normalizer(cut.supervisions[0].text) for cut in batch["cuts"]]
        batch_user_refs = [item[1] for item in batch["user_ref"]]
        batch_agent_refs = [item[1] for item in batch["agent_ref"]]
        batch_hyps = [
            normalizer(model.tokenizer.ids_to_text(parse_hyp(ans, eos_tokens)).strip()) for ans in answer_ids
        ]
        assert len(batch_hyps) == len(batch_user_refs) == len(batch_agent_refs)
        if cfg.verbose:
            batch_wer, _, nins, ndel, nsub = word_error_rate_detail(batch_hyps, batch_agent_refs)
            batch_rtfx = batch_duration / batch_infer_duration
            logging.info(
                f"Batch {batch_idx}: {len(batch_hyps)} samples, WER={batch_wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}] RTFx={batch_rtfx:.1f}"
            )
        
        # collect data for saving to jsonl
        for item_idx, item_user_ref in enumerate(batch["user_ref"]):
            conv_id = item_user_ref[0]
            ref = item_user_ref[1]
            _ = per_conv_data.setdefault(conv_id, {"id": conv_id, "ref_user_text": "", "ref_agent_text": "", "pred_agent_text": ""})
            per_conv_data[conv_id]["ref_user_text"] += f"|{ref}"
            per_conv_data[conv_id]["pred_agent_text"] += f"|{batch_hyps[item_idx]}"
            jsonl_data_single_turn.append({
                "id": conv_id,
                "ref_user_text": ref,
                "pred_agent_text": batch_hyps[item_idx],
                "ref_agent_text": batch["agent_ref"][item_idx][1], # assuming paired data
            })
        
        for item_idx, item_agent_ref in enumerate(batch["agent_ref"]):
            conv_id = item_agent_ref[0]
            ref = item_agent_ref[1]
            try:
                assert conv_id in per_conv_data
            except:
                print(f"conv_id {conv_id} not found in per_conv_data")
                breakpoint()
            per_conv_data[conv_id]["ref_agent_text"] += f"|{ref}"

        # refs.extend(batch_agent_refs)
        # hyps.extend(batch_hyps)

        # input_durations.append(batch_duration)
        # infer_durations.append(batch_infer_duration)

    # wer, _, nins, ndel, nsub = word_error_rate_detail(hypotheses=hyps, references=refs, use_cer=False)
    # rtfx = sum(input_durations) / sum(infer_durations)
    # logging.info(f"WER: {wer:.2%} [ins={nins:.2%} del={ndel:.2%} sub={nsub:.2%}]")
    # logging.info(f"RTFx: {rtfx:.1f}")

    # if cfg.output_manifest is not None:
    #     with SequentialJsonlWriter(cfg.output_manifest) as writer:
    #         for cut, ref, hyp in zip(cuts, refs, hyps):
    #             writer.write({"id": cut.id, "duration": cut.duration, "text": ref, "pred_text": hyp})
    
    # convert per_conv_data to jsonl_data
    
    jsonl_data = []
    for conv_id, conv_data in per_conv_data.items():
        jsonl_data.append({
            "id": conv_id,
            "ref_user_text": conv_data["ref_user_text"],
            "pred_agent_text": conv_data["pred_agent_text"],
            "ref_agent_text": conv_data["ref_agent_text"],
        })
    output_manifest_single_turn = cfg.output_manifest.replace(".jsonl", "_single_turn.jsonl")
    if cfg.output_manifest is not None:
        save_jsonl(jsonl_data, cfg.output_manifest)
        save_jsonl(jsonl_data_single_turn, output_manifest_single_turn)


def parse_hyp(answer: torch.Tensor, eos_tokens: list[int]):
    end = (answer == torch.isin(answer, torch.tensor(eos_tokens))).nonzero(as_tuple=True)[0]
    if end.numel() == 0:
        return answer
    end = end[0]
    return answer[:end]


if __name__ == '__main__':
    main()
