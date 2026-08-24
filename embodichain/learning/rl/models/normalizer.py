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

"""Running observation normalization for reinforcement-learning policies."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["EmpiricalNormalizer"]


class EmpiricalNormalizer(nn.Module):
    """Normalize observations with online mean and variance estimates.

    Statistics are stored as module buffers so policy checkpoints preserve the
    exact transformation used during training and evaluation.

    Args:
        feature_dim: Number of features in the final observation dimension.
        epsilon: Stabilizer added to the running standard deviation.
    """

    def __init__(self, feature_dim: int, epsilon: float = 1e-2) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.epsilon = float(epsilon)
        self.register_buffer("mean", torch.zeros(1, feature_dim))
        self.register_buffer("variance", torch.ones(1, feature_dim))
        self.register_buffer("standard_deviation", torch.ones(1, feature_dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return observations normalized by the current running moments."""
        return (value - self.mean) / (self.standard_deviation + self.epsilon)

    @torch.no_grad()
    def update(self, value: torch.Tensor) -> None:
        """Merge one observation batch into the running moments."""
        if not self.training:
            return
        if value.ndim != 2 or value.shape[-1] != self.mean.shape[-1]:
            raise ValueError(
                "normalizer input must have shape [batch, feature_dim], got "
                f"{tuple(value.shape)}"
            )
        batch_count = int(value.shape[0])
        if batch_count == 0:
            return
        batch_mean = value.mean(dim=0, keepdim=True)
        batch_variance = value.var(dim=0, unbiased=False, keepdim=True)
        self.count.add_(batch_count)
        rate = batch_count / self.count
        delta = batch_mean - self.mean
        self.mean.add_(rate * delta)
        self.variance.add_(
            rate * (batch_variance - self.variance + delta * (batch_mean - self.mean))
        )
        self.variance.clamp_min_(0.0)
        self.standard_deviation.copy_(torch.sqrt(self.variance))
