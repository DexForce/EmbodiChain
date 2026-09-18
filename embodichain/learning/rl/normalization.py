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

"""Running statistics for learning-environment observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

__all__ = ["RunningObservationNormalizer"]


class RunningObservationNormalizer:
    """Normalize continuous observation fields with running Welford statistics.

    The optional mask leaves semantic fields such as one-hot encodings and
    validity bits unchanged while still tracking a single flat observation.
    Statistics are updated only when :meth:`update` is called; :meth:`normalize`
    is side-effect free so a complete differentiable rollout uses one frozen
    normalization transform.

    Args:
        observation_dim: Flat observation dimension.
        device: Device used for the running statistics.
        normalize_mask: Boolean mask where ``True`` selects normalized fields.
        initial_count: Positive pseudo-count used to stabilize the first update.
    """

    def __init__(
        self,
        observation_dim: int,
        device: torch.device | str,
        normalize_mask: torch.Tensor | None = None,
        *,
        initial_count: float = 1.0e-4,
    ) -> None:
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive.")
        if initial_count <= 0.0:
            raise ValueError("initial_count must be positive.")

        self.observation_dim = int(observation_dim)
        self.device = torch.device(device)
        self.mean = torch.zeros(self.observation_dim, device=self.device)
        self.var = torch.ones(self.observation_dim, device=self.device)
        self.count = float(initial_count)
        if normalize_mask is None:
            normalize_mask = torch.ones(
                self.observation_dim,
                dtype=torch.bool,
                device=self.device,
            )
        normalize_mask = torch.as_tensor(
            normalize_mask,
            dtype=torch.bool,
            device=self.device,
        )
        if normalize_mask.shape != (self.observation_dim,):
            raise ValueError(
                "normalize_mask must have shape "
                f"({self.observation_dim},), got {tuple(normalize_mask.shape)}."
            )
        self.normalize_mask = normalize_mask.clone()

    @torch.no_grad()
    def update(self, observations: torch.Tensor) -> None:
        """Merge one observation batch into the running statistics.

        Args:
            observations: Finite tensor shaped ``[batch, observation_dim]``.

        Raises:
            ValueError: If the shape is incompatible or values are non-finite.
        """
        observations = torch.as_tensor(observations, device=self.device)
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(
                "observations must have shape [batch, observation_dim], got "
                f"{tuple(observations.shape)}."
            )
        if observations.shape[0] == 0:
            return
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("observations must contain only finite values.")

        batch_mean = observations.mean(dim=0)
        batch_var = observations.var(dim=0, unbiased=False)
        batch_count = int(observations.shape[0])
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean.add_(delta * batch_count / total_count)
        merged_m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta.square() * self.count * batch_count / total_count
        )
        self.var.copy_(merged_m2 / total_count)
        self.count = float(total_count)

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        """Apply the frozen running transform without detaching observations.

        Args:
            observations: Tensor ending in the configured observation dimension.

        Returns:
            Tensor with continuous fields normalized and semantic fields intact.

        Raises:
            ValueError: If the trailing observation dimension is incompatible.
        """
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(
                "observations must end with observation_dim "
                f"{self.observation_dim}, got {tuple(observations.shape)}."
            )
        normalized = (observations - self.mean) / (self.var.sqrt() + 1.0e-8)
        return torch.where(self.normalize_mask, normalized, observations)

    def state_dict(self) -> dict[str, Any]:
        """Return a device-independent checkpoint payload.

        Returns:
            Mapping containing mean, variance, count, and normalization mask.
        """
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": self.count,
            "normalize_mask": self.normalize_mask.detach().cpu(),
        }

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore statistics while validating their observation layout.

        Args:
            state_dict: Payload produced by :meth:`state_dict`.

        Raises:
            ValueError: If dimensions or the pseudo-count are invalid.
        """
        mean = torch.as_tensor(state_dict["mean"], device=self.device)
        var = torch.as_tensor(state_dict["var"], device=self.device)
        mask = torch.as_tensor(
            state_dict.get("normalize_mask", self.normalize_mask),
            dtype=torch.bool,
            device=self.device,
        )
        expected_shape = (self.observation_dim,)
        if mean.shape != expected_shape or var.shape != expected_shape:
            raise ValueError(
                "Normalizer checkpoint shape does not match observation_dim "
                f"{self.observation_dim}."
            )
        if mask.shape != expected_shape:
            raise ValueError("Normalizer checkpoint mask has an incompatible shape.")
        count = float(state_dict["count"])
        if count <= 0.0:
            raise ValueError("Normalizer checkpoint count must be positive.")
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.normalize_mask.copy_(mask)
        self.count = count
