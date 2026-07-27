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

import pytest
import torch

from nemo.collections.speechlm2.parts.metrics.backchannel import (
    compute_agent_backchannel_output_metrics,
    is_hardcoded_backchannel_phrase,
)


@pytest.mark.parametrize(
    "text",
    [
        "Okay.",
        "uh huh",
        "uhhuh",
        "mhmm",
        "mm hmm",
        "mmhmm",
        "Oh wow, yeah!",
    ],
)
def test_hardcoded_backchannel_accepts_inventory_variants_and_sequences(text):
    assert is_hardcoded_backchannel_phrase(text)


@pytest.mark.parametrize("text", ["", "certainly", "okay, certainly"])
def test_hardcoded_backchannel_rejects_empty_or_unknown_units(text):
    assert not is_hardcoded_backchannel_phrase(text)


def test_output_metrics_count_pairs_isolated_markers_and_valid_phrases_per_sample():
    soab_id = 100
    eoab_id = 101
    pred_ids = torch.tensor(
        [
            [soab_id, 1, eoab_id, soab_id, 2, 0],
            [eoab_id, soab_id, 3, eoab_id, 0, 0],
        ]
    )
    output_mask = torch.tensor(
        [
            [True, True, True, True, True, False],
            [True, True, True, True, False, False],
        ]
    )
    decoded = {(1,): "uh huh", (3,): "certainly"}

    metrics = compute_agent_backchannel_output_metrics(
        pred_ids,
        output_mask,
        soab_id=soab_id,
        eoab_id=eoab_id,
        decode_token_ids=lambda ids: decoded.get(tuple(ids), "unknown"),
    )

    assert metrics["num_samples"].item() == 2
    assert metrics["paired_backchannel_count"].item() == 2
    assert metrics["unpaired_soab_count"].item() == 1
    assert metrics["unpaired_eoab_count"].item() == 1
    assert metrics["hardcoded_backchannel_count"].item() == 1


def test_nested_start_leaves_previous_start_isolated():
    metrics = compute_agent_backchannel_output_metrics(
        torch.tensor([[100, 1, 100, 2, 101]]),
        torch.ones(1, 5, dtype=torch.bool),
        soab_id=100,
        eoab_id=101,
        decode_token_ids=lambda ids: "okay",
    )

    assert metrics["paired_backchannel_count"].item() == 1
    assert metrics["unpaired_soab_count"].item() == 1
    assert metrics["unpaired_eoab_count"].item() == 0
