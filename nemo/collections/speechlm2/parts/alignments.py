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


from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from lhotse import CutSet
from torch import Tensor


@dataclass
class WordAlignment:
    """Word-level alignment result from forced aligner."""

    text: str
    start_time: float
    end_time: float


class ForcedAligner(ABC):
    """
    Base class for forced aligners.
    """

    @abstractmethod
    def align(self, audio: Tensor, audio_lens: Tensor, texts: List[str]) -> List[List[WordAlignment]]:
        """
        Align audio and text for a batch of audio and text.

        Args:
            audio: Tensor containing all audio in the batch
            audio_lens: Tensor containing all audio lengths in the batch
            texts: List of strings containing all text in the batch

        Returns:
            List[List[WordAlignment]]: A list of lists of word alignments for each audio in the batch.
        """
        raise NotImplementedError("Subclasses must implement this method.")


def get_word_alignments_for_batch(
    cuts: Optional[CutSet] = None,
    audios: Optional[Tensor] = None,
    audio_lens: Optional[Tensor] = None,
    texts: Optional[List[str]] = None,
    forced_aligner: Optional[ForcedAligner] = None,
) -> List[List[WordAlignment]]:
    """
    Get word-level alignments for a batch of audio and text.

    Args:
        cuts: CutSet containing all cuts in the batch
        audios: Tensor containing all audio in the batch
        audio_lens: Tensor containing all audio lengths in the batch
        texts: List of strings containing all text in the batch
        forced_aligner: ForcedAligner to use for alignment

    Returns:
        List[List[WordAlignment]]: A list of lists of word alignments for each audio in the batch.
    """
    batch_alignments = []
    if cuts is not None:
        for cut in cuts:
            custom = cut.custom or {}
            raw_alignments = custom.get("alignments", [])
            cut_alignments = []
            for alignment in raw_alignments:
                cut_alignments.append(
                    WordAlignment(
                        text=alignment["text"], start_time=alignment["start_time"], end_time=alignment["end_time"]
                    )
                )
            batch_alignments.append(cut_alignments)
    else:
        assert forced_aligner is not None, "Either cuts or forced_aligner must be provided."
        assert (
            audios is not None and audio_lens is not None and texts is not None
        ), "audios, audio_lens, and texts must be provided if cuts is not provided."
        batch_alignments = forced_aligner.align(audios, audio_lens, texts)
    return batch_alignments
