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

"""Tensor-tree batches and objective results for supervised learning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import nn

__all__ = ["SupervisedBatch", "LossResult", "SupervisedObjective"]


def _map_tensors(value: Any, function: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, Mapping):
        return {key: _map_tensors(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [_map_tensors(item, function) for item in value]
    return value


@dataclass(frozen=True)
class SupervisedBatch:
    """Separate measured inputs, supervision and masks without imposing axes.

    Args:
        inputs: Tensor or nested tensor tree supplied to the objective.
        targets: Supervision tree. Its history may differ from the input history.
        masks: Named masks interpreted by the objective, not the optimizer.
        metadata: Source identities, units, timing and other host-side context.

    The producer owns storage and causal alignment. This container does not
    detach tensors: differentiable multi-step objectives retain their graph.
    """

    inputs: Any
    targets: Any
    masks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> SupervisedBatch:
        """Move tensor trees, preserving dtypes and host metadata.

        Args:
            device: Destination for inputs, targets and masks.

        Returns:
            A batch whose differentiable tensor transfers preserve gradients.
        """
        move = lambda tensor: tensor.to(device)
        return SupervisedBatch(
            _map_tensors(self.inputs, move),
            _map_tensors(self.targets, move),
            _map_tensors(self.masks, move),
            self.metadata,
        )


@dataclass(frozen=True)
class LossResult:
    """An objective-owned scalar loss and sample-weighted observations.

    Args:
        loss: Scalar connected to the parameters to optimize.
        metrics: Scalar diagnostics; they do not enter backward automatically.
        weight: Effective observation count. Under DDP, ``loss`` and metrics
            must be local means over this positive count; the learner uses it
            to produce a global weighted mean and the corresponding gradient.
            Single-device loss reduction remains the objective's decision.
    """

    loss: torch.Tensor
    metrics: Mapping[str, torch.Tensor | float] = field(default_factory=dict)
    weight: float = 1.0


class SupervisedObjective(Protocol):
    """Compute task-specific losses without owning optimizer or data lifecycle."""

    def compute_loss(self, model: nn.Module, batch: SupervisedBatch) -> LossResult:
        """Build a scalar objective, optionally unrolling a differentiable model.

        Args:
            model: Trainable module, already on the batch's device.
            batch: Explicit inputs, targets, masks and sample context.

        Returns:
            Loss and statistics using this objective's numerical conventions.
        """
        ...
