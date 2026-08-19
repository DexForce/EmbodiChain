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
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

    from embodichain.lab.sim.planners import MotionGenOptions, PlanOptions


class DynamicCollisionMode(str, Enum):
    """Policy for consuming a live dynamic collision world.

    This mode controls scene-snapshot obstacle binding and collision-world
    revision recovery. It does not enable or disable a planner's configured
    static-world or self-collision checks.
    """

    OFF = "off"
    """Ignore scene-snapshot collision entities and their revisions."""

    AUTO = "auto"
    """Use live collision entities when the selected motion strategy supports them."""

    REQUIRED = "required"
    """Require live collision entities and a compatible motion planner."""


@dataclass(frozen=True, slots=True)
class MotionPolicy:
    """Immutable motion-generation policy for one action invocation.

    The policy is a runtime value object rather than application configuration.
    ``plan_opts`` is copied on construction so a caller-owned planner config
    cannot change an invocation after it has been created.
    """

    planner: str | None = None
    """Optional required planner backend name; ``None`` accepts the configured one."""

    strategy: Literal["motion_gen", "ik_interp"] = "ik_interp"
    """Motion strategy: ``motion_gen`` or ``ik_interp``."""

    sample_count: int = 50
    """Requested trajectory sample count when the backend does not preserve samples."""

    control_dt: float = 1.0 / 60.0
    """Fallback command period in seconds when a planner supplies no timing."""

    velocity_limit: float | None = None
    """Optional planner velocity limit."""

    acceleration_limit: float | None = None
    """Optional planner acceleration limit."""

    dynamic_collision_mode: DynamicCollisionMode = DynamicCollisionMode.AUTO
    """How this invocation consumes live scene-snapshot collision entities."""

    plan_opts: PlanOptions | None = None
    """Optional typed planner-specific options."""

    def __post_init__(self) -> None:
        valid_strategies = {"motion_gen", "ik_interp"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {sorted(valid_strategies)}, "
                f"got {self.strategy!r}."
            )
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2.")
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be greater than zero.")
        if self.velocity_limit is not None and self.velocity_limit <= 0.0:
            raise ValueError("velocity_limit must be greater than zero when set.")
        if self.acceleration_limit is not None and self.acceleration_limit <= 0.0:
            raise ValueError("acceleration_limit must be greater than zero when set.")
        mode = self.dynamic_collision_mode
        if isinstance(mode, str):
            try:
                mode = DynamicCollisionMode(mode)
            except ValueError as exc:
                raise ValueError(
                    "dynamic_collision_mode must be one of "
                    f"{[item.value for item in DynamicCollisionMode]}, got {mode!r}."
                ) from exc
        elif not isinstance(mode, DynamicCollisionMode):
            raise TypeError(
                "dynamic_collision_mode must be a DynamicCollisionMode or string."
            )
        object.__setattr__(self, "dynamic_collision_mode", mode)
        object.__setattr__(self, "plan_opts", deepcopy(self.plan_opts))

    def to_motion_gen_options(
        self,
        *,
        start_qpos: "torch.Tensor",
        control_part: str,
        sample_count: int | None = None,
        cartesian_linear: bool = False,
    ) -> "MotionGenOptions":
        """Translate this atomic policy into motion-generator options.

        Args:
            start_qpos: Observed controlled-joint start positions.
            control_part: Bound robot control-part name.
            sample_count: Optional segment-local sample-count override.
            cartesian_linear: Whether every supplied Cartesian keyframe is a
                required linear-path sample rather than a sparse endpoint.

        Returns:
            Independently owned options for :class:`MotionGenerator`.
        """
        from embodichain.lab.sim.planners.motion_generator import MotionGenOptions

        return MotionGenOptions(
            strategy=self.strategy,
            sample_count=self.sample_count if sample_count is None else sample_count,
            velocity_limit=self.velocity_limit,
            acceleration_limit=self.acceleration_limit,
            start_qpos=start_qpos,
            control_part=control_part,
            plan_opts=self.plan_opts,
            is_interpolate=True,
            is_linear=cartesian_linear,
            preserve_cartesian_samples=cartesian_linear,
        )


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """Bounded local recovery policy used by the execution runtime."""

    max_replans: int = 3
    """Maximum replans within one action attempt."""

    max_action_retries: int = 2
    """Maximum whole-action retries after planning, execution, or effect failure."""

    tracking_error_threshold: float = 0.05
    """Joint tracking-error threshold in radians."""

    goal_translation_threshold: float = 0.02
    """Dynamic-goal translation threshold in metres."""

    goal_rotation_threshold: float = 0.0872664626
    """Dynamic-goal rotation threshold in radians (five degrees by default)."""

    action_timeout: float = 30.0
    """Maximum execution time for one action attempt in seconds."""

    def __post_init__(self) -> None:
        if self.max_replans < 0:
            raise ValueError("max_replans must be non-negative.")
        if self.max_action_retries < 0:
            raise ValueError("max_action_retries must be non-negative.")
        threshold_fields = (
            "tracking_error_threshold",
            "goal_translation_threshold",
            "goal_rotation_threshold",
            "action_timeout",
        )
        for name in threshold_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")


__all__ = ["DynamicCollisionMode", "MotionPolicy", "RecoveryPolicy"]
