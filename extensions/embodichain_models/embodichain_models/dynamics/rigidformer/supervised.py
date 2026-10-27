# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
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
# ----------------------------------------------------------------------------

"""Shifted author supervision and valid-anchor loss reduction for unified SL."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
from embodichain.learning.sl import LossResult, SupervisedBatch

__all__ = ["rigidformer_batch", "RigidFormerObjective"]


def rigidformer_batch(
    values: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None
) -> SupervisedBatch:
    """Separate shifted conditioning history and next-frame supervision.

    Args:
        values: Author batch including an aligned sequence of at least 3 frames.
        metadata: Optional immutable data and sampling provenance.

    Returns:
        Supervision for frames 2..T-1 with causal conditioning through T-2.
        Earlier targets also occur as subsequent teacher-forcing inputs.
    """
    positions = values["obj_positions"]
    if positions.ndim != 5 or positions.shape[1] < 3:
        raise ValueError("Supervision requires obj_positions [B,T>=3,O,P,3]")
    inputs = {k: v for k, v in values.items() if isinstance(v, torch.Tensor)}
    inputs["obj_positions"] = positions[:, :-1]
    inputs["valid_vtx_mask"] = values["valid_vtx_mask"][:, :-1]
    return SupervisedBatch(
        inputs=inputs, targets=positions[:, 2:], metadata=dict(metadata or {})
    )


class RigidFormerObjective:
    """Reproduce the author's four anchor-only smooth-L1 losses.

    Args:
        weights: Position/acceleration before/after Kabsch weights.
        time_step_scale: Divide errors by step_size squared before smooth-L1.
        augmentation: Optional author geometry augmentation, applied in FP32
            before the network's numerical context.
    """

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        time_step_scale: bool = True,
        augmentation: Any = None,
    ) -> None:
        self.weights = {
            "position_before_icp": 10.0,
            "position_after_icp": 10.0,
            "acceleration_before_icp": 1.0,
            "acceleration_after_icp": 1.0,
            **dict(weights or {}),
        }
        if set(self.weights) != {
            "position_before_icp",
            "position_after_icp",
            "acceleration_before_icp",
            "acceleration_after_icp",
        } or any(
            not torch.isfinite(torch.tensor(v)) or v < 0 for v in self.weights.values()
        ):
            raise ValueError(
                "Only the four finite nonnegative author anchor loss weights are supported"
            )
        self.time_step_scale = time_step_scale
        self.augmentation = augmentation

    def compute_loss(self, model: nn.Module, batch: SupervisedBatch) -> LossResult:
        """Compute one local anchor mean; the shared learner owns DDP reduction.

        Args:
            model: Native RigidFormer model.
            batch: Shifted conditioning and owned target geometry.

        Returns:
            Weighted scalar loss, diagnostic terms and valid-anchor count.
        """
        values = dict(batch.inputs)
        history = values["obj_positions"]
        target = batch.targets
        if self.augmentation is not None:
            raw = dict(values)
            raw["obj_positions"] = torch.cat((history[:, :2], target), dim=1)
            raw["valid_vtx_mask"] = torch.cat(
                (values["valid_vtx_mask"], values["valid_vtx_mask"][:, -1:]), dim=1
            )
            # The reference augments FP32 geometry before entering model autocast.
            with torch.autocast(history.device.type, enabled=False):
                augmented = rigidformer_batch(self.augmentation(raw, split="train"))
            values = dict(augmented.inputs)
            history, target = values["obj_positions"], augmented.targets
        if target.shape != history[:, 1:].shape or batch.masks:
            raise ValueError(
                "Target layout must align with shifted author history; masks use geometry fields"
            )
        # The last frame is never a conditioning input. Fill it without labels.
        values["obj_positions"] = torch.cat(
            (history, torch.zeros_like(history[:, -1:])), dim=1
        )
        values["valid_vtx_mask"] = torch.cat(
            (values["valid_vtx_mask"], values["valid_vtx_mask"][:, -1:]), dim=1
        )
        predictions = model(values)
        b, t, o, p, _ = target.shape
        anchors = values["anchor_idx"][:, None, :, :, None].expand(b, t, o, -1, 3)

        def at_anchor(value: torch.Tensor) -> torch.Tensor:
            return torch.gather(value, 3, anchors)

        gt_pos = at_anchor(target)
        gt_acc = at_anchor(target - 2 * history[:, 1:] + history[:, :-1])
        errors = {
            "position_before_icp": at_anchor(predictions["obj_positions_norm"])
            - gt_pos,
            "position_after_icp": at_anchor(predictions["obj_positions_icp_norm"])
            - gt_pos,
            "acceleration_before_icp": predictions[
                "obj_accelerations_no_icp_norm_anchor"
            ]
            - gt_acc,
            "acceleration_after_icp": at_anchor(predictions["obj_accelerations_norm"])
            - gt_acc,
        }
        mask = (
            values["valid_obj_mask"][:, None, :, None]
            .expand(b, t, o, anchors.shape[3])
            .float()
        )
        count = mask.sum()
        if count <= 0:
            raise ValueError("At least one valid supervised anchor is required")
        scale = (
            values["skip_step_size"][:, None, None, None, None].float().square()
            if self.time_step_scale
            else 1.0
        )
        terms = {
            name: (
                F.smooth_l1_loss(
                    error / scale, torch.zeros_like(error), reduction="none"
                )
                .sum(-1)
                .float()
                * mask
            ).sum()
            / (count + 1e-8)
            for name, error in errors.items()
        }
        loss = sum(terms[name] * weight for name, weight in self.weights.items())
        return LossResult(loss=loss, metrics=terms, weight=float(count.detach()))
