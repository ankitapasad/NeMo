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

import torch
from torch import Tensor


def count_boundary_collar_hits(
    target_mask: Tensor,
    pred_mask: Tensor,
    frame_idx: Tensor,
    before_frames: int,
    after_frames: int,
) -> Tensor:
    hits = torch.zeros((), dtype=torch.long, device=target_mask.device)
    for sample_idx in range(target_mask.shape[0]):
        target_frames = frame_idx[sample_idx][target_mask[sample_idx]]
        pred_frames = frame_idx[sample_idx][pred_mask[sample_idx]]
        if target_frames.numel() == 0 or pred_frames.numel() == 0:
            continue
        in_window = (pred_frames[:, None] >= target_frames[None, :] - before_frames) & (
            pred_frames[:, None] <= target_frames[None, :] + after_frames
        )
        hits = hits + in_window.any(dim=0).long().sum()
    return hits


def boundary_collar_precision_recall_f1(
    collar_hits: Tensor, prediction_count: Tensor, target_count: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute zero-safe collar precision, recall, and F1 from aggregate counts."""
    hits = collar_hits.float()
    precision = hits / prediction_count.clamp(min=1).float()
    recall = hits / target_count.clamp(min=1).float()
    denom = precision + recall
    f1 = torch.where(denom > 0, 2 * precision * recall / denom, torch.zeros_like(denom))
    return precision, recall, f1


def compute_boundary_token_metrics(
    pred_ids: Tensor,
    target_ids: Tensor,
    input_tokens: Tensor,
    sou_id: int,
    eou_id: int,
    ignore_index: int,
    audio_token_idx: int,
    sou_before_frames: int = 4,
    sou_after_frames: int = 4,
    eou_before_frames: int = 2,
    eou_after_frames: int = 4,
) -> dict[str, Tensor]:
    valid_mask = target_ids != ignore_index
    sou_targets = valid_mask & (target_ids == sou_id)
    eou_targets = valid_mask & (target_ids == eou_id)
    sou_preds = valid_mask & (pred_ids == sou_id)
    eou_preds = valid_mask & (pred_ids == eou_id)
    frame_idx = (input_tokens == audio_token_idx).long().cumsum(dim=1)

    sou_collar_hit = count_boundary_collar_hits(
        target_mask=sou_targets,
        pred_mask=sou_preds,
        frame_idx=frame_idx,
        before_frames=sou_before_frames,
        after_frames=sou_after_frames,
    )
    eou_collar_hit = count_boundary_collar_hits(
        target_mask=eou_targets,
        pred_mask=eou_preds,
        frame_idx=frame_idx,
        before_frames=eou_before_frames,
        after_frames=eou_after_frames,
    )

    return {
        "num_samples": torch.as_tensor(target_ids.shape[0], dtype=torch.long, device=target_ids.device),
        "sou_target_count": sou_targets.long().sum(),
        "eou_target_count": eou_targets.long().sum(),
        "sou_pred_count": sou_preds.long().sum(),
        "eou_pred_count": eou_preds.long().sum(),
        "sou_collar_hit": sou_collar_hit,
        "eou_collar_hit": eou_collar_hit,
    }
