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

"""
# ASR transcribe/inference with multi-GPU/multi-node support for large datasets
# It supports both tarred and non-tarred datasets
# Arguments
#    model: path to a nemo/PTL checkpoint file or name of a pretrained model
#    predict_ds: config of the dataset/dataloader
#    output_path: path to store the predictions
#    return_predictions: whether to return the predictions as output other than writing into the files
#    use_cer: whether to calculate the error in terms of CER or use the default WER
#
# Results of each GPU/worker is written into a file named 'predictions_{rank}.json, and aggregated results of all workers are written into 'predictions_all.json'

Example for non-tarred datasets:

python transcribe_speech_parallel.py \
    model=stt_en_conformer_ctc_large \
    predict_ds.manifest_filepath=/dataset/manifest_file.json \
    predict_ds.batch_size=16 \
    output_path=/tmp/

Example for Hybrid-CTC/RNNT models with non-tarred datasets:

python transcribe_speech_parallel.py \
    model=stt_en_fastconformer_hybrid_large \
    decoder_type=ctc \
    predict_ds.manifest_filepath=/dataset/manifest_file.json \
    predict_ds.batch_size=16 \
    output_path=/tmp/

Example for tarred datasets:

python transcribe_speech_parallel.py \
    predict_ds.is_tarred=true \
    predict_ds.manifest_filepath=/tarred_dataset/tarred_audio_manifest.json \
    predict_ds.tarred_audio_filepaths=/tarred_dataset/audio__OP_0..127_CL_.tar \
    ...

By default the trainer uses all the GPUs available and default precision is FP32.
By setting the trainer config you may control these configs. For example to do the predictions with AMP on just two GPUs:

python transcribe_speech_parallel.py \
    trainer.precision=16 \
    trainer.devices=2 \
    ...

You may control the dataloader's config by setting the predict_ds:

python transcribe_speech_parallel.py \
    predict_ds.num_workers=8 \
    predict_ds.min_duration=2.0 \
    predict_ds.sample_rate=16000 \
    model=stt_en_conformer_ctc_small \
    ...

"""

import itertools
import json
import os
from dataclasses import dataclass, field, is_dataclass
from typing import Optional

import lightning.pytorch as ptl
import torch
from omegaconf import MISSING, OmegaConf

from nemo.collections.asr.data.audio_to_text_dataset import ASRPredictionWriter
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel, EncDecHybridRNNTCTCModel
from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
from nemo.collections.asr.models.configs import ASRDatasetConfig
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
from nemo.collections.asr.parts.submodules.rnnt_greedy_decoding import GreedyBatchedRNNTInferConfig
from nemo.core.config import TrainerConfig, hydra_runner
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero


@dataclass
class ParallelTranscriptionConfig:
    model: Optional[str] = None  # name
    predict_ds: ASRDatasetConfig = field(
        default_factory=lambda: ASRDatasetConfig(return_sample_id=True, num_workers=4, min_duration=0, max_duration=40)
    )
    output_path: str = MISSING

    # when return_predictions is enabled, the prediction call would keep all the predictions in memory and return them when prediction is done
    return_predictions: bool = False
    use_cer: bool = False

    # decoding strategy for RNNT models
    # Double check whether fused_batch_size=-1 is right
    rnnt_decoding: RNNTDecodingConfig = field(default_factory=lambda: RNNTDecodingConfig(fused_batch_size=-1))

    # Decoding strategy for CTC models
    ctc_decoding: CTCDecodingConfig = field(default_factory=CTCDecodingConfig)

    # decoder type: ctc or rnnt, can be used to switch between CTC and RNNT decoder for Hybrid RNNT/CTC models
    decoder_type: Optional[str] = None
    # att_context_size can be set for cache-aware streaming models with multiple look-aheads
    att_context_size: Optional[list] = None

    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(devices=-1, accelerator="gpu", strategy="ddp")
    )


def match_train_config(predict_ds, train_ds):
    # It copies the important configurations from the train dataset of the model
    # into the predict_ds to be used for prediction. It is needed to match the training configurations.
    if train_ds is None:
        return

    predict_ds.sample_rate = train_ds.get("sample_rate", 16000)
    cfg_name_list = [
        "int_values",
        "use_start_end_token",
        "blank_index",
        "unk_index",
        "normalize",
        "parser",
        "eos_id",
        "bos_id",
        "pad_id",
    ]

    if is_dataclass(predict_ds):
        predict_ds = OmegaConf.structured(predict_ds)
    for cfg_name in cfg_name_list:
        if hasattr(train_ds, cfg_name):
            setattr(predict_ds, cfg_name, getattr(train_ds, cfg_name))

    return predict_ds


@hydra_runner(config_name="TranscriptionConfig", schema=ParallelTranscriptionConfig)
def main(cfg: ParallelTranscriptionConfig):
    if cfg.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = ASRModel.restore_from(restore_path=cfg.model, map_location="cpu")
    elif cfg.model.endswith(".ckpt"):
        logging.info("Attempting to initialize from .ckpt file")
        model = ASRModel.load_from_checkpoint(checkpoint_path=cfg.model, map_location="cpu")
    else:
        logging.info(
            "Attempting to initialize from a pretrained model as the model name does not have the extension of .nemo or .ckpt"
        )
        model = ASRModel.from_pretrained(model_name=cfg.model, map_location="cpu")

    # Setup decoding strategy
    if hasattr(model, 'change_decoding_strategy') and hasattr(model, 'decoding'):
        if cfg.decoder_type is not None:
            decoding_cfg = cfg.rnnt_decoding if cfg.decoder_type == 'rnnt' else cfg.ctc_decoding
            if hasattr(model, 'cur_decoder'):
                model.change_decoding_strategy(decoding_cfg, decoder_type=cfg.decoder_type)
            else:
                model.change_decoding_strategy(decoding_cfg)

        # Check if ctc or rnnt model
        elif hasattr(model, 'joint'):  # RNNT model
            model.change_decoding_strategy(cfg.rnnt_decoding)
        else:
            model.change_decoding_strategy(cfg.ctc_decoding)

    cfg.predict_ds.return_sample_id = True
    cfg.predict_ds = match_train_config(predict_ds=cfg.predict_ds, train_ds=model.cfg.train_ds)

    if cfg.predict_ds.use_lhotse:
        OmegaConf.set_struct(cfg.predict_ds, False)
        cfg.trainer.use_distributed_sampler = False
        cfg.predict_ds.force_finite = True
        cfg.predict_ds.force_map_dataset = True
        cfg.predict_ds.do_transcribe = True
        OmegaConf.set_struct(cfg.predict_ds, True)

    if isinstance(model, EncDecMultiTaskModel):
        cfg.trainer.use_distributed_sampler = False
        OmegaConf.set_struct(cfg.predict_ds, False)
        cfg.predict_ds.use_lhotse = True
        cfg.predict_ds.lang_field = "target_lang"
        OmegaConf.set_struct(cfg.predict_ds, True)

    trainer = ptl.Trainer(**cfg.trainer)

    if cfg.predict_ds.use_lhotse:
        OmegaConf.set_struct(cfg.predict_ds, False)
        cfg.predict_ds.global_rank = trainer.global_rank
        cfg.predict_ds.world_size = trainer.world_size
        OmegaConf.set_struct(cfg.predict_ds, True)

    data_loader = model._setup_dataloader_from_config(cfg.predict_ds)

    os.makedirs(cfg.output_path, exist_ok=True)
    # trainer.global_rank is not valid before predict() is called. Need this hack to find the correct global_rank.
    global_rank = trainer.node_rank * trainer.num_devices + int(os.environ.get("LOCAL_RANK", 0))
    output_file = os.path.join(cfg.output_path, f"predictions_{global_rank}.json")
    predictor_writer = ASRPredictionWriter(dataset=data_loader.dataset, output_file=output_file)
    trainer.callbacks.extend([predictor_writer])

    predictions = trainer.predict(model=model, dataloaders=data_loader, return_predictions=cfg.return_predictions)
    if predictions is not None:
        predictions = list(itertools.chain.from_iterable(predictions))
    samples_num = predictor_writer.close_output_file()

    logging.info(
        f"Prediction on rank {global_rank} is done for {samples_num} samples and results are stored in {output_file}."
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    samples_num = 0
    pred_text_list = []
    text_list = []
    if is_global_rank_zero():
        output_file = os.path.join(cfg.output_path, f"predictions_all.json")
        logging.info(f"Prediction files are being aggregated in {output_file}.")
        with open(output_file, 'w') as outf:
            for rank in range(trainer.world_size):
                input_file = os.path.join(cfg.output_path, f"predictions_{rank}.json")
                with open(input_file, 'r') as inpf:
                    lines = inpf.readlines()
                    for line in lines:
                        item = json.loads(line)
                        pred_text_list.append(item["pred_text"])
                        text_list.append(item["text"])
                        outf.write(json.dumps(item) + "\n")
                        samples_num += 1
        wer_cer = word_error_rate(hypotheses=pred_text_list, references=text_list, use_cer=cfg.use_cer)
        logging.info(
            f"Prediction is done for {samples_num} samples in total on all workers and results are aggregated in {output_file}."
        )
        logging.info("{} for all predictions is {:.4f}.".format("CER" if cfg.use_cer else "WER", wer_cer))


if __name__ == '__main__':
    main()
