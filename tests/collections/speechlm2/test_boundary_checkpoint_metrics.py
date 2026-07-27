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

from nemo.collections.speechlm2.parts.metrics.boundary import boundary_collar_precision_recall_f1


@pytest.mark.parametrize(
    ("hits", "predictions", "targets", "precision", "recall", "f1"),
    [
        (8, 10, 8, 0.8, 1.0, 8 / 9),
        (4, 4, 8, 1.0, 0.5, 2 / 3),
        (0, 0, 8, 0.0, 0.0, 0.0),
        (0, 0, 0, 0.0, 0.0, 0.0),
    ],
)
def test_boundary_collar_precision_recall_f1(hits, predictions, targets, precision, recall, f1):
    actual = boundary_collar_precision_recall_f1(torch.tensor(hits), torch.tensor(predictions), torch.tensor(targets))
    assert actual[0].item() == pytest.approx(precision)
    assert actual[1].item() == pytest.approx(recall)
    assert actual[2].item() == pytest.approx(f1)


def test_extra_pause_boundary_predictions_reduce_f1():
    precision, recall, f1 = boundary_collar_precision_recall_f1(
        collar_hits=torch.tensor(5),
        prediction_count=torch.tensor(10),
        target_count=torch.tensor(5),
    )
    assert precision.item() == pytest.approx(0.5)
    assert recall.item() == pytest.approx(1.0)
    assert f1.item() == pytest.approx(2 / 3)
