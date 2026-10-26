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

"""NeRD objectives and batch adaptation for the shared supervised learner."""

from __future__ import annotations

from typing import Any

import torch

from embodichain.learning.sl import LossResult, SupervisedBatch

__all__ = ["NeRDObjective", "make_nerd_batch"]


class NeRDObjective:
    """Preserve the NeRD normalized physical-target loss.

    Args:
        include_prefix: Include valid startup positions using the batch mask.
            False retains the reference full-window MSE reduction order.

    Input and target normalization remain owned by the NeRD model and codec.
    Computing this objective never updates their statistics.
    """

    def __init__(self, *, include_prefix: bool = False) -> None:
        self.include_prefix = include_prefix

    def compute_loss(
        self, model: torch.nn.Module, batch: SupervisedBatch
    ) -> LossResult:
        """Compute the normalized state-increment error without detaching it.

        Args:
            model: NeRD model with its frozen normalization buffers.
            batch: Encoded inputs, physical targets and valid-position mask.

        Returns:
            Graph-connected loss for one shared supervised update.
        """
        # Share the physical-target reduction with offline evaluation.
        from .integration import _model_variance, normalized_mse

        prediction = model(batch.inputs)
        loss = normalized_mse(
            prediction,
            batch.targets,
            _model_variance(model),
            batch.masks["valid"] if self.include_prefix else None,
        )
        count = batch.metadata.get("valid_positions")
        if count is None:
            count = (
                int(batch.masks["valid"].sum())
                if self.include_prefix
                else prediction.numel() // prediction.shape[-1]
            )
        return LossResult(loss, weight=float(count))


def make_nerd_batch(
    batch: dict[str, Any], codec: Any, device: str | torch.device
) -> SupervisedBatch:
    """Encode a raw NeRD sequence without changing its sampling contract.

    Args:
        batch: Owned trajectory windows from ``DynamicsWindowDataset``.
        codec: The matching physical state/contact/effort codec.
        device: Device on which encoding and model execution take place.

    Returns:
        Encoded history inputs, physical increment targets and validity mask.
    """
    from .integration import encode_batch

    inputs, targets, valid = encode_batch(batch, codec, device)
    return SupervisedBatch(
        inputs,
        targets,
        masks={"valid": valid},
        metadata={
            "valid_positions": int(batch["valid_mask"].sum()),
            "trajectory_id": tuple(batch["trajectory_id"]),
            "start": batch["start"],
        },
    )
