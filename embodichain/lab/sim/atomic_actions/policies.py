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

import math
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch

    from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
    from embodichain.lab.sim.motion.planners import PlanOptions


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

    strategy: Literal["motion_gen", "ik_interp"] = "ik_interp"
    """Motion strategy: ``motion_gen`` or ``ik_interp``."""

    sample_count: int = 50
    """Minimum trajectory sample count when the backend does not preserve samples."""

    velocity_targets: Literal["auto", "zero"] = "auto"
    """Use trajectory velocities or explicit zero velocity targets."""

    dynamic_collision_mode: DynamicCollisionMode = DynamicCollisionMode.AUTO
    """How this invocation consumes live scene-snapshot collision entities."""

    plan_opts: PlanOptions | None = None
    """Optional typed planner-specific options."""

    velocity_limit: float | None = None
    """Optional scalar speed request, capped by effective Robot joint limits."""

    acceleration_limit: float | None = 0.5
    """Scalar acceleration budget in joint units per second squared.

    For ``motion_gen``, ``None`` retains the selected backend's configured
    budget. Adaptive ``ik_interp`` requires a positive scalar budget; explicit
    ``fixed_grid`` requires ``None`` and does not certify acceleration.
    """

    interpolation_mode: Literal["adaptive", "fixed_grid"] = field(
        default="adaptive", kw_only=True
    )
    """Select adaptive interpolation or an unchanged count/interval grid."""

    ik_seed_mode: Literal["batched", "sequential"] = field(
        default="batched", kw_only=True
    )
    """Select batched IK or previous-valid-posture waypoint seeding."""

    sample_count_retries: tuple[int, ...] = field(default=(), kw_only=True)
    """Explicit denser whole-plan alternatives after successful-row overspeed.

    The engine tries only counts above the current count, in declared order,
    after action composition and any plan transform. This best-effort retry
    policy alone retains the original plan when alternatives are exhausted;
    it does not certify final velocity or enable external nominal motion.
    """

    segment_sample_count_retries: tuple[int, ...] = field(default=(), kw_only=True)
    """Ordered denser alternatives within one sampled Cartesian segment request.

    These counts are independent of complete-action retries. The first count
    also bounds eligibility of the original request, preserving the source
    coarse-grid boundary. Only the explicit Cartesian segment helper consumes
    them; ordinary ``generate`` calls reject non-default segment recovery.
    """

    segment_continuous_ik_recovery: bool = field(default=False, kw_only=True)
    """Try unchanged-target/time continuous IK after segment samples fail.

    Dense attempts advance the existing random stream. The continuous resolver
    retains its own Torch-only fork; no outer action counter is reused.
    """

    mask_generation_overspeed: bool = field(default=False, kw_only=True)
    """Mask normalized motion-generation rows using ordered authored/runtime caps.

    This opt-in check retains generated positions and timing and does not
    replace the separate complete-action velocity gate.
    """

    require_final_velocity: bool = field(default=False, kw_only=True)
    """Require authored/runtime speed validity of final full-robot commands.

    This separate opt-in gate runs after any declared sampling retries and plan
    transforms. Unknown command-only qualification is unsupported. Rejected rows
    cannot dispatch; their representation is explicit. This is planned evidence,
    not physical success.
    """

    final_velocity_rejection: Literal["hold", "mask_only"] = field(
        default="hold", kw_only=True
    )
    """Represent rejected final-velocity rows as holds or retain their proposal.

    ``hold`` preserves shared command authorization and rejected-candidate storage.
    Explicit ``mask_only`` waits for the complete action request, including its
    owned candidate recovery and continuations, then changes validity/classification
    without changing public command or trajectory tensors. Candidate masks stay
    untouched until that boundary; execution still cannot dispatch failed rows.
    """

    def __post_init__(self) -> None:
        if type(self.mask_generation_overspeed) is not bool:
            raise TypeError("mask_generation_overspeed must be a boolean.")
        if self.final_velocity_rejection not in {"hold", "mask_only"}:
            raise ValueError("final_velocity_rejection must be hold or mask_only.")
        segment_retries = self.segment_sample_count_retries
        if (
            type(segment_retries) is not tuple
            or any(type(value) is not int or value < 2 for value in segment_retries)
            or any(a >= b for a, b in zip(segment_retries, segment_retries[1:]))
        ):
            raise ValueError(
                "segment_sample_count_retries must be a strictly increasing tuple of integer counts >=2."
            )
        if type(self.segment_continuous_ik_recovery) is not bool:
            raise TypeError("segment_continuous_ik_recovery must be a boolean.")
        if (segment_retries or self.segment_continuous_ik_recovery) and (
            self.strategy != "ik_interp"
            or self.interpolation_mode != "fixed_grid"
            or self.ik_seed_mode != "sequential"
        ):
            raise ValueError(
                "Cartesian segment recovery requires fixed-grid sequential ik_interp."
            )
        if type(self.require_final_velocity) is not bool:
            raise TypeError("require_final_velocity must be a boolean.")
        retries = self.sample_count_retries
        if (
            type(retries) is not tuple
            or any(type(value) is not int or value < 2 for value in retries)
            or any(first >= second for first, second in zip(retries, retries[1:]))
        ):
            raise ValueError(
                "sample_count_retries must be a strictly increasing tuple of integer counts >=2."
            )
        if self.interpolation_mode not in {"adaptive", "fixed_grid"}:
            raise ValueError("interpolation_mode must be adaptive or fixed_grid.")
        if self.ik_seed_mode not in {"batched", "sequential"}:
            raise ValueError("ik_seed_mode must be batched or sequential.")
        if self.strategy != "ik_interp" and (
            self.interpolation_mode != "adaptive" or self.ik_seed_mode != "batched"
        ):
            raise ValueError(
                "Explicit interpolation modes require strategy='ik_interp'."
            )
        if self.interpolation_mode == "fixed_grid" and (
            self.velocity_limit is not None or self.acceleration_limit is not None
        ):
            raise ValueError(
                "fixed_grid requires velocity_limit=None and acceleration_limit=None."
            )
        valid_strategies = {"motion_gen", "ik_interp"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"strategy must be one of {sorted(valid_strategies)}, "
                f"got {self.strategy!r}."
            )
        if self.strategy == "ik_interp" and self.plan_opts is not None:
            raise ValueError(
                "plan_opts requires strategy='motion_gen'; ik_interp bypasses the backend."
            )
        if self.velocity_targets not in {"auto", "zero"}:
            raise ValueError("velocity_targets must be auto or zero.")
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least 2.")
        for name in ("velocity_limit", "acceleration_limit"):
            value = getattr(self, name)
            if value is None:
                if (
                    name == "velocity_limit"
                    or self.strategy == "motion_gen"
                    or self.interpolation_mode == "fixed_grid"
                ):
                    continue
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and greater than zero.")
            object.__setattr__(self, name, float(value))
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
        interpolation_dt: float | None = None,
        cartesian_linear: bool = False,
        cartesian_keyframes: bool = False,
        cartesian_path_sampler: (
            Callable[["torch.Tensor"], "torch.Tensor"] | None
        ) = None,
    ) -> "MotionGenOptions":
        """Translate this atomic policy into motion-generator options.

        Args:
            start_qpos: Observed controlled-joint start positions.
            control_part: Bound robot control-part name.
            sample_count: Optional segment-local sample-count override.
            interpolation_dt: Explicit waypoint interval used only by
                deterministic interpolation.
            cartesian_linear: Whether every supplied Cartesian keyframe is a
                required linear-path sample rather than a sparse endpoint.
            cartesian_keyframes: Supply unsampled linear keyframes to the public
                MotionGenerator sampling/recovery path. This is distinct from
                already sampled ``cartesian_linear`` input; the two selections
                cannot be combined.
            cartesian_path_sampler: Optional exact-path sampler receiving
                normalized progress values. Use this for constrained curves
                such as rotations about an external hinge.

        Returns:
            Independently owned options for :class:`MotionGenerator`.

        Raises:
            ValueError: If a fixed Cartesian path is requested with a planner
                strategy that does not guarantee those exact samples.
        """
        from embodichain.lab.sim.motion.motion_generator import (
            MotionGenOptions,
        )

        if type(cartesian_keyframes) is not bool:
            raise TypeError("cartesian_keyframes must be a boolean.")
        if cartesian_linear and cartesian_keyframes:
            raise ValueError("Cartesian input cannot be both samples and keyframes.")
        if (cartesian_linear or cartesian_keyframes) and self.strategy != "ik_interp":
            raise ValueError(
                "cartesian_linear skill segments require strategy='ik_interp'; "
                "motion_gen backends do not guarantee exact Cartesian samples."
            )
        return MotionGenOptions(
            strategy=self.strategy,
            sample_count=self.sample_count if sample_count is None else sample_count,
            velocity_limit=self.velocity_limit,
            acceleration_limit=self.acceleration_limit,
            start_qpos=start_qpos,
            control_part=control_part,
            plan_opts=self.plan_opts,
            is_interpolate=self.strategy == "motion_gen",
            interpolation_dt=interpolation_dt,
            is_linear=cartesian_linear or cartesian_keyframes,
            preserve_cartesian_samples=cartesian_linear,
            cartesian_path_sampler=cartesian_path_sampler,
            interpolation_mode=self.interpolation_mode,
            ik_seed_mode=self.ik_seed_mode,
            mask_generation_overspeed=self.mask_generation_overspeed,
            segment_sample_count_retries=self.segment_sample_count_retries,
            segment_continuous_ik_recovery=self.segment_continuous_ik_recovery,
        )


@dataclass(frozen=True, slots=True)
class PlanningRecovery:
    """Immutable, callable-free declaration owned by one concrete skill.

    Args:
        kind: Stable declaration kind selected by an allowlisted decoder.
        skill_id: Exact Atomic skill allowed to consume this declaration.
        revision: Positive declaration revision included in its fingerprint.

    Subclasses contain only immutable scalar/tuple values. Runtime services,
    solver objects and callable configuration do not belong in this value.
    """

    kind: str
    skill_id: str
    revision: int

    def __post_init__(self) -> None:
        for name in ("kind", "skill_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(
                    f"PlanningRecovery.{name} must be an exact identifier."
                )
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("PlanningRecovery.revision must be a positive integer.")
        _validate_planning_recovery(self)


def _validate_planning_recovery(value: PlanningRecovery) -> None:
    if (
        not isinstance(value, PlanningRecovery)
        or not is_dataclass(value)
        or not type(value).__dataclass_params__.frozen
    ):
        raise TypeError("planning_recovery must be a frozen PlanningRecovery value.")

    def scalar(item: object) -> None:
        if item is None or type(item) in (bool, int, str):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is tuple:
            for element in item:
                scalar(element)
            return
        raise TypeError(
            "Planning recovery declarations accept only immutable finite scalar/tuple data."
        )

    for item in fields(value):
        scalar(getattr(value, item.name))


@dataclass(frozen=True, slots=True)
class PlanningBudget:
    """Explicit complete-request measurements and optional cooperative deadline.

    Args:
        wall_seconds: Positive wall-clock seconds for one outer planning request,
            including its sampling retries and nested previews. None enables
            measurements without a deadline. Native operations are not preempted.
    """

    wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.wall_seconds is not None and (
            type(self.wall_seconds) not in (int, float)
            or not math.isfinite(self.wall_seconds)
            or self.wall_seconds <= 0
        ):
            raise ValueError("Planning wall budget must be finite and positive.")


@dataclass(frozen=True, slots=True)
class RecoveryPolicy:
    """Bounded local recovery policy used by the execution runtime."""

    max_replans: int = 3
    """Maximum replans within one action attempt."""

    max_action_retries: int = 2
    """Maximum whole-action retries after planning, execution, or effect failure."""

    goal_translation_threshold: float = 0.02
    """Dynamic-goal translation threshold in metres."""

    goal_rotation_threshold: float = 0.0872664626
    """Dynamic-goal rotation threshold in radians (five degrees by default)."""

    action_timeout: float = 30.0
    """Maximum time for one action attempt, including terminal effect verification."""

    retry_all_failed_plans: bool = field(default=True, kw_only=True)
    """Allow ordinary action retries when a complete plan has no successful row.

    False preserves its failure code but makes the all-failed result terminal.
    Candidate searches inside the complete planning boundary retain their own
    bounded recovery; partial successes keep their existing classification.
    """

    planning_budget: PlanningBudget | None = field(default=None, kw_only=True)
    """Optional outer planning ledger/deadline; None preserves shared defaults.

    An already-active parent ledger owns nested requests and their deadline.
    This is separate from action_timeout, which governs execution/verification.
    """

    planning_recovery: PlanningRecovery | None = field(default=None, kw_only=True)
    """Optional skill-owned planning declaration; None enables no new recovery.

    Its complete immutable contents participate in policy/registration identity.
    This scope is separate from execution replans and whole-action retries.
    """

    def __post_init__(self) -> None:
        if type(self.retry_all_failed_plans) is not bool:
            raise TypeError("retry_all_failed_plans must be bool.")
        if (
            self.planning_budget is not None
            and type(self.planning_budget) is not PlanningBudget
        ):
            raise TypeError("planning_budget must be a PlanningBudget value or None.")
        if self.planning_recovery is not None:
            _validate_planning_recovery(self.planning_recovery)
            object.__setattr__(
                self, "planning_recovery", deepcopy(self.planning_recovery)
            )
        if self.max_replans < 0:
            raise ValueError("max_replans must be non-negative.")
        if self.max_action_retries < 0:
            raise ValueError("max_action_retries must be non-negative.")
        threshold_fields = (
            "goal_translation_threshold",
            "goal_rotation_threshold",
            "action_timeout",
        )
        for name in threshold_fields:
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")

    def validate_skill(self, skill_id: str) -> None:
        """Reject a planning declaration selected for a different Atomic skill.

        Args:
            skill_id: Skill selected by the invocation or static descriptor.

        Raises:
            ValueError: The declaration belongs to another skill.
        """
        if (
            self.planning_recovery is not None
            and self.planning_recovery.skill_id != skill_id
        ):
            raise ValueError(
                f"Planning recovery for {self.planning_recovery.skill_id!r} cannot be used by skill {skill_id!r}."
            )


__all__ = [
    "DynamicCollisionMode",
    "MotionPolicy",
    "PlanningRecovery",
    "PlanningBudget",
    "RecoveryPolicy",
]
