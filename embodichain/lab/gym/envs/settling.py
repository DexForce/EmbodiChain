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

"""Reusable per-environment dynamic-settling state machine."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch

from embodichain.utils import configclass


@configclass
class DynamicSettleMonitorCfg:
    """Threshold and cadence policy for :class:`DynamicSettleMonitor`.

    The monitor never advances an environment. Callers own the stepping path
    and provide raw velocity samples after the configured minimum/cadence.
    This lets reset events and demonstration post-policies share exactly the
    same state transition rules while using different stepping ports.
    """

    linear_velocity_threshold: float = 0.03
    """Maximum stable linear speed in metres per second."""

    angular_velocity_threshold: float = 0.20
    """Maximum stable angular speed in radians per second."""

    min_steps: int = 10
    """Minimum number of environment steps before the first check."""

    max_steps: int = 240
    """Maximum elapsed environment steps before unresolved rows time out."""

    check_interval_steps: int = 2
    """Minimum number of steps between independent evidence checks."""

    required_stable_checks: int = 3
    """Consecutive stable checks required independently for each row."""

    def __post_init__(self) -> None:
        for name in (
            "min_steps",
            "max_steps",
            "check_interval_steps",
            "required_stable_checks",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if self.min_steps < 0:
            raise ValueError("min_steps must be non-negative.")
        if self.max_steps < self.min_steps:
            raise ValueError("max_steps must be greater than or equal to min_steps.")
        if self.check_interval_steps < 1:
            raise ValueError("check_interval_steps must be at least 1.")
        if self.required_stable_checks < 1:
            raise ValueError("required_stable_checks must be at least 1.")
        for name in (
            "linear_velocity_threshold",
            "angular_velocity_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        available_checks = (
            1
            + (self.max_steps - self.min_steps + self.check_interval_steps - 1)
            // self.check_interval_steps
        )
        if self.required_stable_checks > available_checks:
            raise ValueError(
                "required_stable_checks cannot be reached within the configured "
                f"step budget; at most {available_checks} checks are possible."
            )

    def snapshot(self) -> DynamicSettleMonitorCfg:
        """Return an independently owned configuration value."""
        return DynamicSettleMonitorCfg(
            linear_velocity_threshold=self.linear_velocity_threshold,
            angular_velocity_threshold=self.angular_velocity_threshold,
            min_steps=self.min_steps,
            max_steps=self.max_steps,
            check_interval_steps=self.check_interval_steps,
            required_stable_checks=self.required_stable_checks,
        )


@dataclass(frozen=True, slots=True, eq=False)
class DynamicSettleSample:
    """Raw per-body speed evidence for one registered scene entity.

    Args:
        entity_id: Stable entity identifier used in metadata and diagnostics.
        linear_speed: Per-row body speeds with shape ``(B, N)``.
        angular_speed: Per-row body speeds with shape ``(B, N)``.
    """

    entity_id: str
    linear_speed: torch.Tensor
    angular_speed: torch.Tensor

    def __post_init__(self) -> None:
        if (
            type(self.entity_id) is not str
            or not self.entity_id
            or self.entity_id != self.entity_id.strip()
        ):
            raise ValueError(
                "entity_id must be a non-empty string without outer whitespace."
            )
        for name in ("linear_speed", "angular_speed"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if not value.is_floating_point() or value.dim() != 2:
                raise ValueError(f"{name} must be a floating tensor with shape (B, N).")
            if value.shape[0] == 0 or value.shape[1] == 0:
                raise ValueError(f"{name} must contain at least one row and body.")
        if self.linear_speed.shape != self.angular_speed.shape:
            raise ValueError("linear_speed and angular_speed must have equal shapes.")
        if self.linear_speed.device != self.angular_speed.device:
            raise ValueError("linear_speed and angular_speed must share a device.")
        object.__setattr__(self, "linear_speed", self.linear_speed.clone())
        object.__setattr__(self, "angular_speed", self.angular_speed.clone())

    def snapshot(self) -> DynamicSettleSample:
        """Return an independently owned raw evidence sample."""
        return DynamicSettleSample(
            entity_id=self.entity_id,
            linear_speed=self.linear_speed,
            angular_speed=self.angular_speed,
        )


@dataclass(frozen=True, slots=True, eq=False)
class DynamicSettleState:
    """Owned state emitted after one monitor observation."""

    env_ids: torch.Tensor
    elapsed_steps: int
    observation_count: int
    checked: bool
    stable_counts: torch.Tensor
    settled_mask: torch.Tensor
    timeout_mask: torch.Tensor
    max_linear_speed: torch.Tensor
    max_angular_speed: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if self.env_ids.dtype != torch.long or self.env_ids.dim() != 1:
            raise ValueError("env_ids must be a one-dimensional torch.long tensor.")
        if self.env_ids.numel() == 0:
            raise ValueError("env_ids must contain at least one row.")
        if type(self.elapsed_steps) is not int or self.elapsed_steps < 0:
            raise ValueError("elapsed_steps must be a non-negative integer.")
        if type(self.observation_count) is not int or self.observation_count < 0:
            raise ValueError("observation_count must be a non-negative integer.")
        if type(self.checked) is not bool:
            raise TypeError("checked must be a bool.")
        row_count = self.env_ids.numel()
        for name, dtype in (
            ("stable_counts", torch.long),
            ("settled_mask", torch.bool),
            ("timeout_mask", torch.bool),
        ):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if value.dtype != dtype or value.shape != (row_count,):
                raise ValueError(f"{name} must have shape (B,) and dtype {dtype}.")
            if value.device != self.env_ids.device:
                raise ValueError(f"{name} and env_ids must share a device.")
        if (self.settled_mask & self.timeout_mask).any():
            raise ValueError("settled_mask and timeout_mask must not overlap.")
        for name in ("max_linear_speed", "max_angular_speed"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if not value.is_floating_point() or value.shape != (row_count,):
                raise ValueError(f"{name} must be a floating tensor with shape (B,).")
            if value.device != self.env_ids.device:
                raise ValueError(f"{name} and env_ids must share a device.")
        for name in (
            "env_ids",
            "stable_counts",
            "settled_mask",
            "timeout_mask",
            "max_linear_speed",
            "max_angular_speed",
        ):
            object.__setattr__(self, name, getattr(self, name).clone())

    @property
    def complete(self) -> bool:
        """Whether every row has either settled or timed out."""
        return bool((self.settled_mask | self.timeout_mask).all().item())

    def to_metadata(self) -> dict[str, object]:
        """Return deterministic, JSON-compatible post-policy metadata."""
        return {
            "elapsed_steps": self.elapsed_steps,
            "observation_count": self.observation_count,
            "env_ids": self.env_ids.detach().to("cpu").tolist(),
            "stable_counts": self.stable_counts.detach().to("cpu").tolist(),
            "settled_mask": self.settled_mask.detach().to("cpu").tolist(),
            "timeout_mask": self.timeout_mask.detach().to("cpu").tolist(),
            "max_linear_speed": self.max_linear_speed.detach().to("cpu").tolist(),
            "max_angular_speed": self.max_angular_speed.detach().to("cpu").tolist(),
        }


class DynamicSettleMonitor:
    """Track settling independently for stable environment IDs.

    Duplicate observations at the same ``elapsed_steps`` value are idempotent.
    Regressing step counters are rejected, and a jump across multiple cadence
    boundaries counts as one fresh observation rather than replaying one sample.
    """

    def __init__(
        self,
        cfg: DynamicSettleMonitorCfg,
        env_ids: torch.Tensor,
    ) -> None:
        if not isinstance(cfg, DynamicSettleMonitorCfg):
            raise TypeError("cfg must be a DynamicSettleMonitorCfg.")
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor.")
        if env_ids.dtype != torch.long or env_ids.dim() != 1:
            raise ValueError("env_ids must be a one-dimensional torch.long tensor.")
        if env_ids.numel() == 0 or torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must contain unique environment IDs.")
        self.cfg = cfg.snapshot()
        self._env_ids = env_ids.clone()
        self._stable_counts = torch.zeros_like(env_ids)
        self._settled = torch.zeros_like(env_ids, dtype=torch.bool)
        self._timeout = torch.zeros_like(env_ids, dtype=torch.bool)
        self._max_linear = torch.full(
            env_ids.shape,
            float("inf"),
            dtype=torch.float32,
            device=env_ids.device,
        )
        self._max_angular = self._max_linear.clone()
        self._last_elapsed_steps = -1
        self._last_checked_steps = -1
        self._observation_count = 0

    @property
    def env_ids(self) -> torch.Tensor:
        """Return the stable row IDs owned by this monitor."""
        return self._env_ids.clone()

    def observe(
        self,
        samples: Sequence[DynamicSettleSample],
        *,
        elapsed_steps: int,
    ) -> DynamicSettleState:
        """Consume one raw speed observation when the configured cadence is due.

        Args:
            samples: One speed sample per monitored entity.
            elapsed_steps: Steps advanced by the caller since post-policy start.

        Returns:
            Per-row stable, settled, timeout, and velocity metadata.
        """
        if type(elapsed_steps) is not int or elapsed_steps < 0:
            raise ValueError("elapsed_steps must be a non-negative integer.")
        if elapsed_steps < self._last_elapsed_steps:
            raise ValueError("elapsed_steps must be monotonic.")
        normalized = tuple(samples)
        if not normalized or not all(
            isinstance(sample, DynamicSettleSample) for sample in normalized
        ):
            raise ValueError("samples must contain DynamicSettleSample values.")
        if len({sample.entity_id for sample in normalized}) != len(normalized):
            raise ValueError("samples must use unique entity IDs.")
        for sample in normalized:
            if sample.linear_speed.shape[0] != self._env_ids.numel():
                raise ValueError("Every sample batch must match env_ids length.")
            if sample.linear_speed.device != self._env_ids.device:
                raise ValueError("Samples and env_ids must share a device.")

        duplicate = elapsed_steps == self._last_elapsed_steps
        due = elapsed_steps >= self.cfg.min_steps and (
            self._last_checked_steps < 0
            or elapsed_steps - self._last_checked_steps >= self.cfg.check_interval_steps
            or elapsed_steps >= self.cfg.max_steps
        )
        checked = due and not duplicate and not self._timeout.all()
        if checked:
            linear = torch.cat([sample.linear_speed for sample in normalized], dim=1)
            angular = torch.cat([sample.angular_speed for sample in normalized], dim=1)
            finite = torch.isfinite(linear).all(dim=1) & torch.isfinite(angular).all(
                dim=1
            )
            self._max_linear = torch.where(
                torch.isfinite(linear), linear, torch.full_like(linear, float("inf"))
            ).amax(dim=1)
            self._max_angular = torch.where(
                torch.isfinite(angular),
                angular,
                torch.full_like(angular, float("inf")),
            ).amax(dim=1)
            stable = (
                finite
                & (self._max_linear <= self.cfg.linear_velocity_threshold)
                & (self._max_angular <= self.cfg.angular_velocity_threshold)
            )
            active = ~self._settled & ~self._timeout
            self._stable_counts = torch.where(
                active & stable,
                self._stable_counts + 1,
                torch.where(
                    active, torch.zeros_like(self._stable_counts), self._stable_counts
                ),
            )
            self._settled |= active & (
                self._stable_counts >= self.cfg.required_stable_checks
            )
            self._observation_count += 1
            self._last_checked_steps = elapsed_steps

        if elapsed_steps >= self.cfg.max_steps:
            self._timeout |= ~self._settled
        self._last_elapsed_steps = elapsed_steps
        return self._state(elapsed_steps=elapsed_steps, checked=checked)

    def _state(self, *, elapsed_steps: int, checked: bool) -> DynamicSettleState:
        """Build an owned state snapshot."""
        return DynamicSettleState(
            env_ids=self._env_ids,
            elapsed_steps=elapsed_steps,
            observation_count=self._observation_count,
            checked=checked,
            stable_counts=self._stable_counts,
            settled_mask=self._settled,
            timeout_mask=self._timeout,
            max_linear_speed=self._max_linear,
            max_angular_speed=self._max_angular,
        )


__all__ = [
    "DynamicSettleMonitor",
    "DynamicSettleMonitorCfg",
    "DynamicSettleSample",
    "DynamicSettleState",
]
