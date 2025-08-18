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
import json
import os
import shutil
from collections import defaultdict

import torch
import torchaudio
from typing import List, Optional

from nemo.utils import logging

from whisper_normalizer.english import EnglishTextNormalizer

def safe_remove_path(path):
    try:
        shutil.rmtree(path)
    except:
        pass  # File was already deleted by another thread


class ResultsLogger:
    """
    Saves audios and a json file with the model outputs.
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.audio_save_path = os.path.join(save_path, "pred_wavs")
        os.makedirs(self.audio_save_path, exist_ok=True)
        self.matadata_save_path = os.path.join(save_path, "metadatas")
        os.makedirs(self.matadata_save_path, exist_ok=True)
        self.normalizer = EnglishTextNormalizer()
        self.cached_results = defaultdict(list)

    def reset(self):
        # ensures that we are cleaning the metadata files
        # metadata_files = os.listdir(self.matadata_save_path)
        # for f in metadata_files:
        #     open(os.path.join(self.matadata_save_path, f), 'w').close()
        self.cached_results = defaultdict(list)
        return self

    @staticmethod
    def merge_and_save_audio(
        out_audio_path: str, pred_audio: torch.Tensor, pred_audio_sr: int, user_audio: torch.Tensor, user_audio_sr: int
    ) -> None:
        user_audio = torchaudio.functional.resample(user_audio.float(), user_audio_sr, pred_audio_sr)
        T1, T2 = pred_audio.shape[0], user_audio.shape[0]
        max_len = max(T1, T2)
        pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
        user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)

        # combine audio in a multichannel audio
        combined_wav = torch.cat(
            [
                user_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                pred_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
            ],
            dim=0,
        )

        # save audio
        torchaudio.save(out_audio_path, combined_wav.squeeze(), pred_audio_sr)
        logging.info(f"Audio saved at: {out_audio_path}")

    def update(
        self,
        name: str,
        refs: list[str],
        hyps: list[str],
        asr_hyps: list[str],
        samples_id: list[str],
        pred_audio: torch.Tensor,
        pred_audio_sr: int,
        user_audio: torch.Tensor,
        user_audio_sr: int,
        eou_pred: torch.Tensor = None,
        fps: float = None,
        results=None,
        tokenizer=None,
        decode_audio=True,
    ) -> None:

        out_json_path = os.path.join(self.matadata_save_path, f"{name}.json")
        out_dicts = []
        for i in range(len(refs)):
            # save audio
            sample_id = samples_id[i][:150]  # make sure that sample id is not too big
            out_audio_path = os.path.join(self.audio_save_path, f"{name}_{sample_id}.wav")
            if decode_audio:
                self.merge_and_save_audio(out_audio_path, pred_audio[i], pred_audio_sr, user_audio[i], user_audio_sr)
                # create a wav with eou prediction for debug purposes
                if eou_pred is not None:
                    out_audio_path_eou = os.path.join(self.audio_save_path, f"{name}_{sample_id}_eou.wav")
                    repeat_factor = int(pred_audio_sr / fps)
                    eou_pred_wav = (
                        eou_pred[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, repeat_factor)
                    )  # (B, T, repeat_factor)
                    eou_pred_wav = eou_pred_wav.view(1, -1)  # (B, T * repeat_factor)
                    eou_pred_wav = eou_pred_wav.float() * 0.8  #  make 1 audible and keep 0 as total silence
                    torchaudio.save(out_audio_path_eou, eou_pred_wav.squeeze().unsqueeze(0).detach().cpu(), pred_audio_sr)

            # cache metadata
            out_dict = {
                "target_text": refs[i],
                "pred_text": hyps[i],
            }
            if decode_audio:
                out_dict["speech_pred_transcribed"] = asr_hyps[i]
                out_dict['audio_path'] = os.path.relpath(out_audio_path, self.save_path)
            if results is not None:
                if tokenizer is not None:
                    out_dict['tokens_text'] = " ".join(tokenizer.ids_to_tokens(results['tokens_text'][i]))
                else:
                    out_dict['tokens_text'] = results['tokens_text'][i].tolist()
            out_dicts.append(out_dict)
            self.cached_results[name].append(out_dict)

        # uses append here to avoid needs to cache
        with open(out_json_path, 'a+', encoding='utf-8') as fout:
            for out_dict in out_dicts:
                json.dump(out_dict, fout, indent=4, ensure_ascii=False)

        logging.info(f"Metadata file for {name} dataset updated at: {out_json_path}")
    
    def compute_and_save(self, special_subset_names: Optional[List[str]] = None):
        """
        Saves all cached results. If special_subset_names are provided, it also
        computes and returns the accuracy and empty rate for each of those subsets.

        Args:
            special_subset_names: A list of validation subset names to compute accuracy for.

        Returns:
            A dictionary of calculated metrics (accuracy and empty_rate) for the special subsets.
            E.g., {'web-q': {'acc': 0.8, 'empty_rate': 0.1}, ...}
        """
        if special_subset_names is None:
            special_subset_names = ['llama-qa']
            # special_subset_names = ['web-qa', 'llama-qa','trivia-qa']


        metrics_results = {}

        for name, results_list in self.cached_results.items():

            out_json_path = os.path.join(self.matadata_save_path, f"{name}.json")
            with open(out_json_path, 'w', encoding='utf-8') as fout:
                json.dump(results_list, fout, indent=4, ensure_ascii=False)
            logging.info(f"Metadata file for {name} dataset saved at: {out_json_path}")

            if name in special_subset_names and results_list:
                correct_count = 0
                correct_count_phrase = 0
                empty_count = 0
                total_count = len(results_list)

                for item in results_list:

                    pred_text = item["pred_text"].strip()
                    normalized_pred = self.normalizer(pred_text)

                    if not normalized_pred:
                        empty_count += 1
                        continue

                    pred_words = set(normalized_pred.split())


                    target_text = item["target_text"]
                    possible_targets = target_text.split(';')

                    is_correct = False
                    for target_option in possible_targets:

                        normalized_target_option = self.normalizer(target_option.strip())
                        if normalized_target_option in normalized_pred:
                            correct_count_phrase += 1
                            break

                        target_words = set(normalized_target_option.split())

                        # if not target_words:
                        #     breakpoint()

                        if target_words.issubset(pred_words):
                            is_correct = True
                            break

                    if is_correct:
                        correct_count += 1

                acc_phrase = correct_count_phrase / total_count if total_count > 0 else 0.0
                acc = correct_count / total_count if total_count > 0 else 0.0
                empty_rate = empty_count / total_count if total_count > 0 else 0.0

                metrics_results[name] = {'acc': torch.tensor(acc), 'acc_phrase': torch.tensor(acc_phrase), 'empty_rate': torch.tensor(empty_rate)}
                logging.info(f"Metrics for special subset '{name}': Accuracy (split)={acc}, Accuracy (phrase)={acc_phrase}, Empty Rate={empty_rate}")

        return metrics_results
