from collections import defaultdict

import sacrebleu
import torch
from whisper_normalizer.english import EnglishTextNormalizer

from nemo.collections.asr.models import ASRModel
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.collections.duplex_s2s.parts.pretrained import load_pretrained_nemo
from nemo.utils import logging


class ASRBLEU:
    """
    Computes BLEU scores on ASR predictions on generated audio with pretrained NeMo ASR.
    By default, uses Whisper's EnglishTextNormalizer on hypotheses and references.
    """

    def __init__(self, pretrained_asr: str, normalize: bool = True, normalizer=None, verbose: bool = True) -> None:
        self.asr = None  # load into memory on reset()
        self.pretrained_asr_name = pretrained_asr
        self.verbose = verbose
        if normalize:
            if normalizer is None:
                self.normalizer = EnglishTextNormalizer()
            else:
                self.normalizer = normalizer
        else:
            self.normalizer = _identity

        self._refs = defaultdict(list)
        self._hyps = defaultdict(list)

    def reset(self):
        # Cleaning up GPU memory before we load ASRModel, because it may already
        # be quite fragmented and close to the limit after observing many
        # dynamic shapes during the training epoch.
        torch.cuda.memory.empty_cache()
        self.asr = load_pretrained_nemo(ASRModel, self.pretrained_asr_name).eval()
        WithOptionalCudaGraphs.disable_cuda_graphs_recursive(self.asr, attribute_path="decoding.decoding")
        return self

    def update(
        self, name: str, refs: list[str], pred_audio: torch.Tensor, pred_audio_lens: torch.Tensor = None
    ) -> None:
        if self.asr is None:
            self.reset()

        if pred_audio_lens is None:
            pred_audio_lens = [pred_audio.shape[1]] * pred_audio.shape[0]

        asr_hyps = self.asr.transcribe(
            [audio[:alen] for audio, alen in zip(pred_audio, pred_audio_lens)],
            batch_size=pred_audio.shape[0],
            verbose=False,
        )

        for ref, asr_hyp in zip(refs, asr_hyps):
            asr_hyp = asr_hyp.text
            self._refs[name].append([self.normalizer(ref)])
            self._hyps[name].append(self.normalizer(asr_hyp))
            if self.verbose:
                asrb = sacrebleu.sentence_bleu(asr_hyp, [ref]).score
                logging.info(f"[REF]\t{ref}\n[ASR]\t{asr_hyp} [{asrb:.2f}]")

    def compute(self) -> dict[str, torch.Tensor]:
        """Computes the final score and deallocates ASR and partial results."""
        corpus_metric = {}
        for name in self._refs.keys():
            metric = torch.tensor(sacrebleu.corpus_bleu(self._hyps[name], self._refs[name]).score)
            corpus_metric[f"asr_bleu_{name}"] = metric
        corpus_metric["asr_bleu"] = torch.stack(list(corpus_metric.values())).mean()
        self._refs.clear()
        self._hyps.clear()
        self.asr = None  # free up GPU memory
        torch.cuda.memory.empty_cache()
        return corpus_metric


def _identity(x):
    return x
