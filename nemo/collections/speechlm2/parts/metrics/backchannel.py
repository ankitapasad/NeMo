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

"""Output-only validation metrics for agent backchannel marker sequences."""

import re
from collections.abc import Callable

import torch
from torch import Tensor


# Kept in sync with ttm-data-processing
# src/davidai_data_processing/generate_sample_candidates.py at
# 00704969049bf30edb52b61bc30545c573b0e61f.
HARDCODED_COMMON_BACKCHANNEL_PHRASES = {
    "ah",
    "all right",
    "alright",
    "good",
    "gotcha",
    "hmm",
    "huh",
    "i see",
    "okay",
    "ok",
    "mm",
    "mm-hmm",
    "mhm",
    "no",
    "oh",
    "oh okay",
    "oh wow",
    "right",
    "uh-huh",
    "interesting",
    "wow",
    "yeah",
    "yes",
}

BACKCHANNEL_SPELLING_VARIANT_TO_CANONICAL = {
    "mhmm": "mm-hmm",
    "mm hmm": "mm-hmm",
    "mmhmm": "mm-hmm",
    "uh huh": "uh-huh",
    "uhhuh": "uh-huh",
}

CONFIRMED_BACKCHANNEL_PHRASES = (
    HARDCODED_COMMON_BACKCHANNEL_PHRASES | set(BACKCHANNEL_SPELLING_VARIANT_TO_CANONICAL)
)


def normalize_backchannel_unit(text: str) -> str:
    lowered = text.strip().lower().replace("\u2019", "'")
    cleaned = re.sub(r"[^a-z0-9'-]+", " ", lowered)
    return " ".join(cleaned.split())


def backchannel_units(text: str) -> list[str]:
    units = [normalize_backchannel_unit(unit) for unit in re.split(r"[.!?,;:]+", text)]
    return [unit for unit in units if unit]


def is_hardcoded_backchannel_phrase(text: str) -> bool:
    """Match one curated phrase or a punctuation-separated sequence of them."""
    units = backchannel_units(text)
    return bool(units) and all(unit in CONFIRMED_BACKCHANNEL_PHRASES for unit in units)


def compute_agent_backchannel_output_metrics(
    pred_ids: Tensor,
    output_mask: Tensor,
    *,
    soab_id: int,
    eoab_id: int,
    decode_token_ids: Callable[[list[int]], str],
) -> dict[str, Tensor]:
    """Count well-formed markers and lexicon-valid phrases without GT matching.

    The parser runs independently per utterance over teacher-forced output
    positions. A nested SOAB leaves the preceding SOAB isolated and opens a new
    span. An EOAB outside a span is isolated. A span still open at utterance end
    contributes one isolated SOAB.
    """
    if pred_ids.shape != output_mask.shape:
        raise ValueError(
            f"pred_ids and output_mask must have the same shape; got {pred_ids.shape} and {output_mask.shape}"
        )
    if pred_ids.ndim != 2:
        raise ValueError(f"pred_ids must have shape (batch, sequence); got {pred_ids.shape}")
    if soab_id == eoab_id:
        raise ValueError("soab_id and eoab_id must be different")

    paired = 0
    unpaired_soab = 0
    unpaired_eoab = 0
    hardcoded = 0

    for sample_idx in range(pred_ids.shape[0]):
        sample_ids = pred_ids[sample_idx][output_mask[sample_idx].bool()].tolist()
        open_phrase: list[int] | None = None
        for token_id in sample_ids:
            if open_phrase is None:
                if token_id == soab_id:
                    open_phrase = []
                elif token_id == eoab_id:
                    unpaired_eoab += 1
                continue

            if token_id == soab_id:
                unpaired_soab += 1
                open_phrase = []
            elif token_id == eoab_id:
                paired += 1
                if is_hardcoded_backchannel_phrase(decode_token_ids(open_phrase)):
                    hardcoded += 1
                open_phrase = None
            else:
                open_phrase.append(token_id)

        if open_phrase is not None:
            unpaired_soab += 1

    device = pred_ids.device
    return {
        "num_samples": torch.as_tensor(pred_ids.shape[0], dtype=torch.long, device=device),
        "paired_backchannel_count": torch.as_tensor(paired, dtype=torch.long, device=device),
        "unpaired_soab_count": torch.as_tensor(unpaired_soab, dtype=torch.long, device=device),
        "unpaired_eoab_count": torch.as_tensor(unpaired_eoab, dtype=torch.long, device=device),
        "hardcoded_backchannel_count": torch.as_tensor(hardcoded, dtype=torch.long, device=device),
    }
