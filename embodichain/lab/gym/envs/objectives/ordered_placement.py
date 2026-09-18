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

"""Batched physical measurements independent of expert execution authority."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .config import OrderedPlacementObjectiveCfg, decode_objective

__all__ = ["MeasuredRigidObjectState", "OrderedPlacementObjective"]


@dataclass(frozen=True)
class MeasuredRigidObjectState:
    """Environment-local position and measured velocities, each shaped ``(N, 3)``."""

    position: torch.Tensor
    linear_velocity: torch.Tensor
    angular_velocity: torch.Tensor


class OrderedPlacementObjective:
    """Observe ordered stable regions without changing termination or acceptance.

    Call :meth:`update` exactly once after each control step's interval events.
    Snapshots clone every tensor and never sample or advance the objective.
    """

    def __init__(
        self,
        cfg: OrderedPlacementObjectiveCfg,
        num_envs: int,
        device: torch.device | str,
    ) -> None:
        """Create isolated progress buffers for the environment batch.

        Args:
            cfg: Ordered regions and physical stability thresholds.
            num_envs: Positive number of independent environment rows.
            device: Device shared by progress buffers and measured state.
        """
        cfg.validate()
        self.cfg = decode_objective(cfg.to_dict())
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer.")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._lower = torch.tensor(
            [r.lower for r in self.cfg.regions], device=self.device, dtype=torch.float64
        )
        self._upper = torch.tensor(
            [r.upper for r in self.cfg.regions], device=self.device, dtype=torch.float64
        )
        self._progress = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self._success = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._stable = torch.zeros_like(self._success)
        self._valid = torch.zeros_like(self._success)
        self._hold = torch.zeros(num_envs, device=self.device, dtype=torch.float64)
        self._position = torch.zeros(
            num_envs, 3, device=self.device, dtype=torch.float64
        )
        self._linear_speed = torch.zeros_like(self._hold)
        self._angular_speed = torch.zeros_like(self._hold)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear only selected episode rows, without taking a measurement.

        Args:
            env_ids: Rows to reset, or ``None`` to reset the entire batch.
        """
        ids = (
            slice(None)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        for tensor in (
            self._progress,
            self._success,
            self._stable,
            self._valid,
            self._hold,
            self._position,
            self._linear_speed,
            self._angular_speed,
        ):
            tensor[ids] = 0

    @torch.no_grad()
    def update(self, measured_state: MeasuredRigidObjectState, dt: float) -> None:
        """Consume one control sample, advancing at most one milestone per row.

        Args:
            measured_state: Batched local position and measured velocities.
            dt: Positive duration of this control step in seconds.
        """
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        tensors = (
            measured_state.position,
            measured_state.linear_velocity,
            measured_state.angular_velocity,
        )
        if any(
            tensor.shape != (self.num_envs, 3)
            or tensor.device != self._position.device
            or not tensor.is_floating_point()
            for tensor in tensors
        ):
            raise ValueError(
                "Measured state must contain floating (num_envs, 3) tensors on the objective device."
            )
        position, linear, angular = tensors
        self._position.copy_(position)
        linear_speed = torch.linalg.vector_norm(linear, dim=-1)
        angular_speed = torch.linalg.vector_norm(angular, dim=-1)
        self._linear_speed.copy_(linear_speed)
        self._angular_speed.copy_(angular_speed)
        self._valid.copy_(
            torch.stack([torch.isfinite(t).all(dim=-1) for t in tensors]).all(dim=0)
        )
        target = self._progress.clamp(max=len(self.cfg.regions) - 1)
        stable = (
            self._valid
            & (position >= self._lower[target].to(dtype=position.dtype)).all(dim=-1)
            & (position <= self._upper[target].to(dtype=position.dtype)).all(dim=-1)
            & (linear_speed <= self.cfg.max_linear_speed)
            & (angular_speed <= self.cfg.max_angular_speed)
        )
        self._stable.copy_(stable)
        self._hold.copy_(
            torch.where(stable, (self._hold + dt).clamp(max=self.cfg.hold_seconds), 0.0)
        )
        # Avoid one extra control step when repeated decimal dt additions land
        # a few floating-point ulps below the configured duration.
        held = (self._hold >= self.cfg.hold_seconds) | torch.isclose(
            self._hold,
            torch.full_like(self._hold, self.cfg.hold_seconds),
            rtol=1e-12,
            atol=0.0,
        )
        reached = (self._progress < len(self.cfg.regions)) & held
        self._progress.add_(reached.long())
        self._success.copy_((self._progress == len(self.cfg.regions)) & stable & held)
        self._hold[reached & ~self._success] = 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return an isolated batch snapshot of history and current physical truth.

        Returns:
            Predicate identity, milestone progress, current success and metrics.
            All tensor values are clones; reading them does not advance time.
        """
        return {
            "predicate": self.cfg.type,
            "object_uid": self.cfg.object_uid,
            "goal_count": len(self.cfg.regions),
            "success": self._success.clone(),
            "progress": self._progress.clone(),
            "metrics": {
                "stable": self._stable.clone(),
                "measurement_valid": self._valid.clone(),
                "hold_seconds": self._hold.clone(),
                "position": self._position.clone(),
                "linear_speed": self._linear_speed.clone(),
                "angular_speed": self._angular_speed.clone(),
            },
        }

    def snapshot_row(self, env_id: int) -> dict[str, Any]:
        """Return one JSON-compatible row; nonfinite measurements become null.

        Args:
            env_id: Zero-based environment row to inspect.

        Returns:
            Isolated physical result with tensors converted to JSON values.
        """
        if type(env_id) is not int or not 0 <= env_id < self.num_envs:
            raise ValueError("env_id is outside the objective batch.")

        def convert(value: Any) -> Any:
            if isinstance(value, torch.Tensor):
                return convert(value[env_id].detach().cpu().tolist())
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        return convert(self.snapshot())
