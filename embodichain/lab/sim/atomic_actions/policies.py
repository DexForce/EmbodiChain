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

"""Motion-generation and bounded-recovery policies for atomic actions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import PlanOptions


@dataclass(frozen=True, slots=True)
class MotionPolicy:
    """Immutable motion-generation policy for one action invocation.

    The policy is a runtime value object rather than application configuration.
    ``plan_opts`` is copied on construction so a caller-owned planner config
    cannot change an invocation after it has been created.
    """

    planner: str | None = None
    """Optional required planner backend name; ``None`` accepts the configured one."""

    motion_source: str = "ik_interp"
    """Trajectory source: ``ik_interp`` or ``motion_gen``."""

    interpolation: str = "linear"
    """Interpolation policy. Only linear interpolation is currently supported."""

    sample_count: int = 50
    """Requested trajectory sample count when the backend does not preserve samples."""

    control_dt: float = 1.0 / 60.0
    """Fallback command period in seconds when a planner supplies no timing."""

    velocity_limit: float | None = None
    """Optional planner velocity limit."""

    acceleration_limit: float | None = None
    """Optional planner acceleration limit."""

    collision_check: bool = True
    """Whether collision-aware backends should enable collision checking."""

    plan_opts: PlanOptions | None = None
    """Optional typed planner-specific options."""

    def __post_init__(self) -> None:
        valid_sources = {"ik_interp", "motion_gen"}
        if self.motion_source not in valid_sources:
            raise ValueError(
                f"motion_source must be one of {sorted(valid_sources)}, "
                f"got {self.motion_source!r}."
            )
        if self.interpolation != "linear":
            raise ValueError(
                "interpolation currently supports only 'linear', "
                f"got {self.interpolation!r}."
            )
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2.")
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be greater than zero.")
        if self.velocity_limit is not None and self.velocity_limit <= 0.0:
            raise ValueError("velocity_limit must be greater than zero when set.")
        if self.acceleration_limit is not None and self.acceleration_limit <= 0.0:
            raise ValueError("acceleration_limit must be greater than zero when set.")
        object.__setattr__(self, "plan_opts", deepcopy(self.plan_opts))


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """Bounded local recovery policy used by the execution runtime."""

    max_replans: int = 3
    """Maximum current-phase replans."""

    max_phase_retries: int = 2
    """Maximum retries of a failed phase."""

    tracking_error_threshold: float = 0.05
    """Joint tracking-error threshold in radians."""

    goal_translation_threshold: float = 0.02
    """Dynamic-goal translation threshold in metres."""

    goal_rotation_threshold: float = 0.0872664626
    """Dynamic-goal rotation threshold in radians (five degrees by default)."""

    phase_timeout: float = 30.0
    """Maximum phase execution time in seconds."""

    def __post_init__(self) -> None:
        if self.max_replans < 0:
            raise ValueError("max_replans must be non-negative.")
        if self.max_phase_retries < 0:
            raise ValueError("max_phase_retries must be non-negative.")
        threshold_fields = (
            "tracking_error_threshold",
            "goal_translation_threshold",
            "goal_rotation_threshold",
            "phase_timeout",
        )
        for name in threshold_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")


__all__ = ["MotionPolicy", "RecoveryPolicy"]
