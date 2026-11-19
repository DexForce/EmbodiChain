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

"""Contact manipulation goals and observed phase planning shared by PickUp/Place."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

import torch

from embodichain.lab.sim.atomic_skills.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_skills.continuation import PlanContinuationResult
from embodichain.lab.sim.atomic_skills.plans import SkillPlan, TimedTrajectory
from embodichain.lab.sim.atomic_skills.primitives._helpers import (
    plan_named_arm_trajectory,
    require_shared_task_state_key,
)
from embodichain.lab.sim.atomic_skills.invocation import ResolvedSkillRequest
from embodichain.lab.sim.atomic_skills.state import PlanningContext

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_skills.core import AtomicSkill


@dataclass(frozen=True, slots=True)
class ContactPostcondition:
    """Named physical check or explicit settling target in a contact path.

    Args:
        kind: Physical checkpoint or explicit command-hold duration.
        target: Identifier resolved by the episode's observation source.
    """

    kind: Literal["hold", "check"]
    target: str

    def __post_init__(self) -> None:
        if (
            self.kind not in {"hold", "check"}
            or not isinstance(self.target, str)
            or not self.target
        ):
            raise ValueError("Contact postconditions require a hold/check target.")


@dataclass(frozen=True, slots=True)
class ContactPhase:
    """One contact path waypoint, grasp, or hand command with named evidence.

    Args:
        kind: Cartesian path, pickup contact closure, or named joint path.
        target: Observation-source target resolved immediately before planning.
        after: Ordered settling and physical checkpoints on the command grid.
    """

    kind: Literal["pose", "grasp", "joint"]
    target: str
    after: tuple[ContactPostcondition, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.kind not in {"pose", "grasp", "joint"}
            or not isinstance(self.target, str)
            or not self.target
        ):
            raise ValueError("Invalid contact phase.")
        if not all(type(v) is ContactPostcondition for v in self.after):
            raise TypeError("Contact phase tails must be typed postconditions.")
        object.__setattr__(self, "after", tuple(self.after))


def _validate_path(
    source: str, phases: tuple[ContactPhase, ...], *, pickup: bool
) -> None:
    if (
        not isinstance(source, str)
        or not source
        or not 1 <= len(phases) <= 128
        or not all(type(p) is ContactPhase for p in phases)
    ):
        raise ValueError(
            "Contact goals require a bound source and a finite typed path."
        )
    if sum(p.kind == "grasp" for p in phases) != (1 if pickup else 0):
        raise ValueError(
            "Pickup paths need exactly one grasp; placement paths cannot grasp."
        )


@dataclass(frozen=True, slots=True)
class ContactPickUpGoal:
    """Approach, establish contact grasp and lift using an observed path source.

    Args:
        source: Engine-local source identifier; its episode host retains ownership.
        phases: Bounded path containing exactly one grasp and its required evidence.

    The source supplies physical acceptance; completion projects no rigid relation.
    """

    goal_kind: ClassVar[str] = "contact_pick_up"
    source: str
    phases: tuple[ContactPhase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        _validate_path(self.source, self.phases, pickup=True)


@dataclass(frozen=True, slots=True)
class ContactPlaceGoal:
    """Place a contact-held subject without projecting a rigid object/tool relation.

    Args:
        source: Engine-local source that observes the existing physical grasp.
        phases: Bounded transport, release and retreat path, with no new grasp.
    """

    goal_kind: ClassVar[str] = "contact_place"
    source: str
    phases: tuple[ContactPhase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "phases", tuple(self.phases))
        _validate_path(self.source, self.phases, pickup=False)


@dataclass(frozen=True, slots=True, eq=False)
class ResolvedContactPhase:
    """Owned same-observation geometric inputs; contains no controller handles.

    Args:
        values: Cartesian poses or joint path in the source's selected row order.
        held_qpos: Full loaded robot commands preserved on nonmoving joints.
        valid: Boolean row validity from geometry and contact preconditions.
        endpoint: Endpoint receiving joint paths; Cartesian paths use motion.
        data: Owned tensor/scalar metadata committed only on plan installation.
    """

    values: torch.Tensor
    held_qpos: torch.Tensor
    valid: torch.Tensor
    endpoint: Literal["motion", "grasp"] = "motion"
    data: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.held_qpos.is_floating_point()
            or not self.values.is_floating_point()
            or self.values.dtype != self.held_qpos.dtype
            or self.values.ndim < 2
            or self.held_qpos.ndim != 2
            or self.valid.dtype != torch.bool
            or self.valid.shape != self.held_qpos.shape[:1]
        ):
            raise ValueError(
                "Contact snapshots require full joint rows and matching boolean validity."
            )
        if (
            self.values.shape[0] != self.held_qpos.shape[0]
            or self.values.device != self.held_qpos.device
            or self.valid.device != self.held_qpos.device
        ):
            raise ValueError("Contact snapshot tensors must share batch/device.")
        if (
            not torch.isfinite(self.values).all()
            or not torch.isfinite(self.held_qpos).all()
        ):
            raise ValueError(
                "Resolved contact commands must be finite, including inactive holds."
            )
        if self.endpoint not in {"motion", "grasp"} or len(
            {k for k, _ in self.data}
        ) != len(self.data):
            raise ValueError("Invalid contact endpoint or duplicate snapshot keys.")
        if any(
            not isinstance(k, str)
            or not k
            or not (isinstance(v, (torch.Tensor, str, bool, int, float)) or v is None)
            for k, v in self.data
        ):
            raise TypeError(
                "Contact snapshot metadata must contain scalar values or owned tensors."
            )
        for name in ["values", "held_qpos", "valid"]:
            object.__setattr__(self, name, getattr(self, name).clone())
        object.__setattr__(
            self,
            "data",
            tuple(
                (k, v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in self.data
            ),
        )

    def snapshot(self) -> ResolvedContactPhase:
        """Return owned observation data."""
        return replace(self)


@dataclass(frozen=True, slots=True, eq=False)
class ContactContinuation:
    """Finite observed path cursor with immutable phase/checkpoint evidence.

    Args:
        position: One-based index of the current phase.
        limit: Total declared phase count in this invocation.
        phase_name: Auditable current target identifier.
        phase: Declarative target and ordered postconditions.
        resolved: Owned geometric inputs from the planning observation.
        checkpoints: Command indices and target identifiers requiring host evidence.
    """

    position: int
    limit: int
    phase_name: str
    phase: ContactPhase
    resolved: ResolvedContactPhase
    checkpoints: tuple[tuple[int, str], ...] = ()

    def snapshot(self) -> ContactContinuation:
        """Copy the phase's derived tensors, retaining only declarative references."""
        return replace(self, resolved=self.resolved.snapshot())


@runtime_checkable
class ContactPhaseSource(Protocol):
    """Engine-bound observation/geometry adapter; never plans or steps physics."""

    def resolve(
        self,
        phase: ContactPhase,
        request: ResolvedSkillRequest,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> ResolvedContactPhase:
        """Resolve one target from context and active rows without committing state."""
        ...

    def hold_steps(self, target: str) -> int:
        """Return a declared positive number of settling command samples."""
        ...

    def begin(
        self,
        cursor: ContactContinuation,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> None:
        """Install reference bookkeeping after the engine accepts a phase plan."""
        ...

    def finish(
        self,
        cursor: ContactContinuation,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        """Return per-row acceptance only after all physical checkpoints are observed."""
        ...


def _source(
    action: AtomicSkill, goal: ContactPickUpGoal | ContactPlaceGoal
) -> ContactPhaseSource:
    return action.planning_services.contact_source(goal.source)


def _trajectory(context: PlanningContext, positions: torch.Tensor) -> TimedTrajectory:
    dt = torch.full(
        positions.shape[:2],
        context.require_control_dt(),
        dtype=torch.float64,
        device=positions.device,
    )
    dt[:, 0] = 0
    return TimedTrajectory(positions, None, None, dt, context.env_ids)


def _plan_contact_phase(
    action: AtomicSkill,
    request: ResolvedSkillRequest,
    context: PlanningContext,
    position: int = 0,
    active_mask: torch.Tensor | None = None,
) -> SkillPlan:
    """Plan one observed chunk directly, without nested atomic invocations."""
    goal = request.goal
    if request.recovery_policy.planning_budget is not None:
        raise ValueError(
            "Observed contact paths do not support external planning budgets."
        )
    if request.recovery_policy.max_replans or request.recovery_policy.max_skill_retries:
        raise ValueError(
            "Contact paths use normal continuation, not replay of partially executed skills."
        )
    if (
        request.motion_policy.sample_count_retries
        or request.motion_policy.segment_sample_count_retries
        or request.motion_policy.segment_continuous_ik_recovery
        or request.motion_policy.phase_paths
    ):
        raise ValueError("Observed contact paths require a fixed sampling policy.")
    if (
        request.motion_policy.dynamic_collision_mode != "off"
        or request.motion_policy.strategy != "ik_interp"
    ):
        raise ValueError(
            "Observed contact paths require the declared IK grid and native contact qualification."
        )
    active = active_mask if active_mask is not None else context.planning_mask
    if active is None:
        active = torch.ones(
            context.batch_size, dtype=torch.bool, device=context.env_ids.device
        )
    key = require_shared_task_state_key(
        request.binding.endpoint("primary", "motion"),
        request.binding.endpoint("primary", "grasp"),
        participant="Contact manipulation",
    )
    if key is not None:
        active = active & ~context.task.held_object_mask(key)
        for pair, held in context.task.coordinated_held_objects.items():
            if key in pair:
                active &= ~(
                    torch.ones_like(active) if held.env_mask is None else held.env_mask
                )
    phase = goal.phases[position]
    source = _source(action, goal)
    resolved = source.resolve(phase, request, context, active)
    valid = active & resolved.valid
    motion = request.binding.endpoint("primary", "motion").require_target(
        JointPositionTarget
    )
    hand = request.binding.endpoint("primary", "grasp").require_target(
        JointPositionTarget
    )
    if phase.kind == "pose":
        success, arm = plan_named_arm_trajectory(
            action.motion_generator,
            motion.control_part,
            context.robot.qpos[:, list(motion.joint_ids)],
            resolved.values,
            resolved.values.shape[1] + 1,
            request.motion_policy,
            context.require_control_dt(),
            active_mask=valid,
            cartesian_linear=True,
        )
        positions = resolved.held_qpos[:, None].expand(-1, arm.shape[1], -1).clone()
        positions[:, :, list(motion.joint_ids)] = arm
        segments = {"move_end_effector": arm.shape[1]}
    elif phase.kind == "joint":
        target = motion if resolved.endpoint == "motion" else hand
        positions = (
            resolved.held_qpos[:, None].expand(-1, resolved.values.shape[1], -1).clone()
        )
        positions[:, :, list(target.joint_ids)] = resolved.values
        success = valid & action.motion_generator.joint_steps_within_velocity_limits(
            context.robot.qpos[:, list(target.joint_ids)],
            resolved.values,
            target.control_part,
            velocity_limit=request.motion_policy.velocity_limit,
            step_dt=context.require_control_dt(),
        )
        segments = {"move_joints": positions.shape[1]}
    else:
        opened = request.binding.endpoint("primary", "grasp").joint_positions(
            "open",
            num_envs=context.batch_size,
            device=action.device,
            dtype=context.robot.qpos.dtype,
        )
        closed = dict(resolved.data)["closed_qpos"]
        direction = request.skill_options.approach_direction.to(context.robot.qpos)
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        )
        success, positions, segments = action._get_full_pickup_trajectory(
            resolved.values,
            context.robot.qpos[:, list(motion.joint_ids)],
            context.robot.qpos,
            request.motion_policy,
            request.skill_options,
            direction,
            motion,
            hand,
            opened,
            closed,
            context.require_control_dt(),
            contact_grasp=True,
            env_ids=context.env_ids,
            active_mask=valid,
        )
        success &= valid
    checkpoints = []
    count = positions.shape[1]
    for post in phase.after:
        if post.kind == "hold":
            n = source.hold_steps(post.target)
        else:
            checkpoints.append((count - 1, post.target))
            n = 2
        if type(n) is not int or n < 1:
            raise ValueError(
                "Contact settling targets must supply positive integer steps."
            )
        positions = torch.cat((positions, positions[:, -1:].expand(-1, n, -1)), 1)
        segments[f"{post.kind}:{post.target}:{len(segments)}"] = n
        count += n
    plan = action.build_plan(
        request,
        context,
        success=success & valid,
        trajectory=_trajectory(context, positions),
        segment_lengths=segments,
        replannable=False,
    )
    return replace(
        plan,
        continuation=ContactContinuation(
            position + 1,
            len(goal.phases),
            phase.target,
            phase,
            resolved,
            tuple(checkpoints),
        ),
    )


def _continue_contact_phase(
    action: AtomicSkill,
    request: ResolvedSkillRequest,
    context: PlanningContext,
    plan: SkillPlan,
    active_mask: torch.Tensor,
) -> PlanContinuationResult:
    """Validate the observed chunk and keep the same invocation identity."""
    cursor = plan.continuation
    if not isinstance(cursor, ContactContinuation):
        raise TypeError("Expected contact path continuation.")
    accepted, messages = _source(action, request.goal).finish(
        cursor, context, active_mask
    )
    accepted &= active_mask
    next_plan = None
    if cursor.position < len(request.goal.phases) and bool(accepted.any()):
        next_plan = _plan_contact_phase(
            action, request, context, cursor.position, accepted
        )
    return PlanContinuationResult(accepted, next_plan, messages)


def _begin_contact_phase(
    action: AtomicSkill,
    request: ResolvedSkillRequest,
    context: PlanningContext,
    plan: SkillPlan,
) -> None:
    """Commit only observation/reference bookkeeping after plan acceptance."""
    if isinstance(request.goal, (ContactPickUpGoal, ContactPlaceGoal)):
        _source(action, request.goal).begin(
            plan.continuation,
            context,
            plan.plan_success
            & (
                context.planning_mask
                if context.planning_mask is not None
                else torch.ones_like(plan.plan_success)
            ),
        )


__all__ = [
    "ContactPhase",
    "ContactPostcondition",
    "ContactPickUpGoal",
    "ContactPlaceGoal",
    "ContactPhaseSource",
    "ContactContinuation",
    "ResolvedContactPhase",
]
