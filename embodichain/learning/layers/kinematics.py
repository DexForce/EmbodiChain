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

"""Torch module interface to fixed robot geometry, independent of simulation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from embodichain.compute.kinematics.serial_chain import SerialChainKinematics

__all__ = ["KinematicsLayer"]


class KinematicsLayer(nn.Module):
    """Expose FK and differentiable Jacobians as a reusable network layer.

    Args:
        model: Fixed kinematics model, copied so device moves are independent.
        tcp: Optional flange-to-tool rigid transform.
        lower: Optional physical lower limits; defaults to the chain limits.
        upper: Optional physical upper limits; defaults to the chain limits.

    Note:
        Geometry is configured, not trained. Checkpoints store TCP and limits
        and verify the identity of the model supplied at reconstruction time.
        ``load_state_dict(assign=True)`` adopts checkpoint device and precision
        for both tensor buffers and fixed geometry, including nested loads.
        ``train()`` and ``eval()`` do not disable input gradients.
    """

    def __init__(
        self,
        model: SerialChainKinematics,
        tcp: torch.Tensor | None = None,
        lower: torch.Tensor | None = None,
        upper: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model.to(model.chain.device, model.chain.dtype)
        options = {"device": model.chain.device, "dtype": model.chain.dtype}
        tcp_tensor = (
            torch.eye(4, **options)
            if tcp is None
            else torch.as_tensor(tcp, **options).detach().clone()
        )
        if tcp_tensor.shape != (4, 4) or not torch.isfinite(tcp_tensor).all():
            raise ValueError("TCP must be a finite 4x4 rigid transform")
        rotation = tcp_tensor[:3, :3]
        if (
            not torch.allclose(
                rotation.T @ rotation, torch.eye(3, **options), atol=1e-5, rtol=1e-5
            )
            or not torch.allclose(
                torch.linalg.det(rotation), torch.ones((), **options), atol=1e-5
            )
            or not torch.allclose(tcp_tensor[3], torch.tensor([0, 0, 0, 1], **options))
        ):
            raise ValueError("TCP must be a rigid SE(3) transform")
        lo = (
            torch.as_tensor(model.chain.low if lower is None else lower, **options)
            .detach()
            .clone()
        )
        hi = (
            torch.as_tensor(model.chain.high if upper is None else upper, **options)
            .detach()
            .clone()
        )
        if (
            lo.shape != (model.dof,)
            or hi.shape != lo.shape
            or torch.isnan(lo).any()
            or torch.isnan(hi).any()
            or (lo > hi).any()
        ):
            raise ValueError("Invalid physical joint limits")
        self.register_buffer("tcp", tcp_tensor)
        self.register_buffer("lower", lo)
        self.register_buffer("upper", hi)
        self.register_load_state_dict_post_hook(self._sync_model_after_load)

    @property
    def dof(self) -> int:
        """Number of movable joints in solver order."""
        return self.model.dof

    def _apply(self, fn: Any, recurse: bool = True) -> KinematicsLayer:
        # PK is not an nn.Module: explicitly migrate all of its geometry tensors.
        probe = fn(self.tcp)
        if probe.dtype not in (torch.float32, torch.float64):
            raise ValueError("KinematicsLayer supports float32 and float64")
        super()._apply(fn, recurse=recurse)
        self.model = self.model.to(self.tcp.device, self.tcp.dtype)
        return self

    def _sync_model_after_load(
        self, _module: nn.Module, _incompatible_keys: Any
    ) -> None:
        # assign=True replaces buffers without calling _apply, also when a parent
        # network loads this layer recursively. PK geometry is not a buffer.
        if self.tcp.dtype not in (torch.float32, torch.float64):
            raise ValueError("KinematicsLayer supports float32 and float64")
        if (
            self.model.chain.device != self.tcp.device
            or self.model.chain.dtype != self.tcp.dtype
        ):
            self.model = self.model.to(self.tcp.device, self.tcp.dtype)

    def get_extra_state(self) -> dict[str, str]:
        """Record fixed geometry identity alongside tensor buffers."""
        return {"model_fingerprint": self.model.fingerprint}

    def set_extra_state(self, state: dict[str, str]) -> None:
        """Reject checkpoints belonging to a different kinematic model."""
        if state != self.get_extra_state():
            raise ValueError(
                "Kinematics checkpoint geometry does not match the configured model"
            )

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        """Return root-frame TCP transforms for ``(..., dof)`` joints."""
        return self.evaluate(qpos)[0]

    def evaluate(self, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return TCP transforms and root-frame geometric Jacobians."""
        return self.model.evaluate(qpos, self.tcp)

    def jacobian(self, qpos: torch.Tensor) -> torch.Tensor:
        """Return differentiable TCP Jacobians, shape ``(..., 6, dof)``."""
        return self.evaluate(qpos)[1]

    def default_seed(self) -> torch.Tensor:
        """Return bounded joint midpoints, or projected zero for unbounded joints."""
        bounded = torch.isfinite(self.lower) & torch.isfinite(self.upper)
        midpoint = (self.lower.where(bounded, 0) + self.upper.where(bounded, 0)) * 0.5
        return midpoint.clamp(min=self.lower, max=self.upper)
