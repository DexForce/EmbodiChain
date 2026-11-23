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

"""MoveJoints atomic skill implementation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar, Protocol, runtime_checkable

import torch

from embodichain.lab.sim.atomic_skills.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_skills.continuation import PlanContinuationResult
from embodichain.lab.sim.atomic_skills.core import AtomicSkill
from embodichain.lab.sim.atomic_skills.invocation import (
    SkillOptions,
    ResolvedSkillRequest,
)
from embodichain.lab.sim.atomic_skills.plans import SkillPlan, TimedTrajectory
from embodichain.lab.sim.atomic_skills.policies import DynamicCollisionMode
from embodichain.lab.sim.atomic_skills.requirements import (
    JOINT_POSITION_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_skills.state import PlanningContext
from embodichain.lab.sim.atomic_skills.trajectory_ops import (
    build_joint_plan_states,
    resolve_joint_target,
    to_full_robot_trajectory,
)
from embodichain.lab.sim.atomic_skills.primitives._binding_contracts import (
    make_motion_slot,
)


@dataclass(frozen=True, slots=True)
class ObservedJointTrajectory:
    """One timed joint motion resolved at declared observation boundaries.

    Args:
        source_id: Engine-local identifier owned by the episode's command source.
        phase: Source-local name for this motion.
        chunk_lengths: One positive command count per chunk, in execution order.
    """

    source_id: str
    phase: str
    chunk_lengths: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("An observed trajectory needs a nonempty source ID.")
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise ValueError("An observed trajectory needs a nonempty phase name.")
        if (
            not isinstance(self.chunk_lengths, tuple)
            or not 1 <= len(self.chunk_lengths) <= 256
            or any(
                type(length) is not int or length < 1 for length in self.chunk_lengths
            )
        ):
            raise ValueError("Observed trajectories need 1–256 positive chunk lengths.")


@runtime_checkable
class ObservedJointTrajectorySource(Protocol):
    """Episode-owned source of live full-robot commands for MoveJoints."""

    def resolve(
        self,
        phase: str,
        start: int,
        stop: int,
        context: PlanningContext,
        active_mask: torch.Tensor,
    ) -> TimedTrajectory:
        """Resolve timed commands without dispatching or advancing the simulation.

        Args:
            phase: Source-local motion name from the goal.
            start: First command index, zero-based within the phase.
            stop: Exclusive end command index.
            context: Fresh observations, or an isolated nominal preview context.
            active_mask: Rows eligible to execute the chunk.

        Returns:
            One full-robot command per requested control step and row.
        """
        ...


@dataclass(frozen=True, slots=True)
class _JointTrajectoryCursor:
    """Value-owned progress of an observed trajectory within one invocation."""

    position: int
    limit: int
    phase_name: str
    stop: int

    def snapshot(self) -> _JointTrajectoryCursor:
        return replace(self)


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionGoal:
    """Explicit or named joint-space goal for a bound robot resource."""

    target: torch.Tensor | str | TimedTrajectory | ObservedJointTrajectory
    """Joint waypoints, a named command, or a timed full-robot trajectory.

    Authored and observed timed motion require dynamic collision planning to be
    disabled. An observed motion resolves later samples after earlier commands
    have executed.
    """

    def __post_init__(self) -> None:
        if isinstance(self.target, ObservedJointTrajectory):
            return
        if isinstance(self.target, TimedTrajectory):
            if self.target.waypoint_count == 0:
                raise ValueError(
                    "A timed joint replay must contain at least one command."
                )
            return
        if isinstance(self.target, str):
            if not self.target.strip():
                raise ValueError("Named joint-position target must not be empty.")
            return
        if not isinstance(self.target, torch.Tensor):
            raise TypeError(
                "target must be a torch.Tensor, str, TimedTrajectory or "
                "ObservedJointTrajectory, "
                f"got {type(self.target).__name__}."
            )
        if self.target.dim() not in (1, 2, 3) or self.target.shape[-1] == 0:
            raise ValueError(
                "Tensor target must have shape (control_dof,), "
                "(num_envs, control_dof), "
                "or (num_envs, n_waypoint, control_dof), "
                f"got {tuple(self.target.shape)}."
            )


@dataclass(frozen=True, slots=True, eq=False)
class MoveJointsOptions(SkillOptions):
    """Per-invocation behavior for :class:`MoveJoints`."""

    preserve_samples: bool = False
    """Keep authored joint samples at host cadence, rejecting overspeed rows."""

    preserve_velocity_samples: bool = False
    """Retain explicit arrival velocities, including the ends of consecutive timed phases."""

    def __post_init__(self) -> None:
        if type(self.preserve_velocity_samples) is not bool:
            raise TypeError("preserve_velocity_samples must be a bool.")
        if type(self.preserve_samples) is not bool:
            raise TypeError("preserve_samples must be a bool.")


class MoveJoints(AtomicSkill[JointPositionGoal, MoveJointsOptions]):
    """Plan joint motion from the observed state to one or more waypoints."""

    skill_id: ClassVar[str] = "move_joints"
    GoalType: ClassVar[type] = JointPositionGoal
    OptionsType: ClassVar[type] = MoveJointsOptions
    agent_visible: ClassVar[bool] = True
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_motion_slot(
                "primary",
                capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
            ),
        ),
    )

    def _plan(
        self,
        request: ResolvedSkillRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
    ) -> SkillPlan:
        """Plan a joint-space goal without mutating the robot or task state."""
        goal = request.goal
        if isinstance(goal.target, ObservedJointTrajectory):
            recovery = request.recovery_policy
            if (
                recovery.planning_budget
                or recovery.max_replans
                or recovery.max_skill_retries
            ):
                raise ValueError(
                    "Observed joint trajectories require a fixed execution grid "
                    "without planning budgets or skill retries."
                )
            active = context.planning_mask
            if active is None:
                active = torch.ones_like(context.env_ids, dtype=torch.bool)
            return self._plan_observed_chunk(
                request, context, goal.target, 0, 0, active
            )

        motion = request.binding.endpoint("primary", "motion")
        motion_target = motion.require_target(JointPositionTarget)
        control_part = motion_target.control_part
        joint_ids = list(motion_target.joint_ids)
        if request.skill_options.preserve_velocity_samples and (
            not isinstance(goal.target, TimedTrajectory)
            or goal.target.velocities is None
        ):
            raise ValueError(
                "Preserving velocity samples requires an explicitly timed velocity trajectory."
            )
        if isinstance(goal.target, TimedTrajectory):
            return self._build_timed_plan(request, context, goal.target, True)

        joint_dof = len(motion_target.joint_ids)
        target_qpos = resolve_joint_target(
            self._resolve_target_qpos(
                goal,
                request=request,
                context=context,
            ),
            num_envs=context.batch_size,
            joint_dof=joint_dof,
            control_part=control_part,
            device=self.device,
        )
        start_qpos = context.robot.qpos[:, joint_ids]
        if request.skill_options.preserve_samples:
            if request.motion_policy.strategy != "ik_interp":
                raise ValueError("preserve_samples requires strategy='ik_interp'.")
            from embodichain.lab.sim.motion.planners import PlanResult

            if target_qpos.ndim != 3:
                raise ValueError("preserve_samples requires batched joint samples.")
            step_dt = context.require_control_dt()
            dt = torch.full(
                target_qpos.shape[:2],
                step_dt,
                dtype=torch.float64,
                device=target_qpos.device,
            )
            dt[:, 0] = 0
            result = PlanResult(
                success=self.motion_generator.joint_steps_within_velocity_limits(
                    start_qpos,
                    target_qpos,
                    control_part,
                    velocity_limit=request.motion_policy.velocity_limit,
                    step_dt=step_dt,
                ),
                positions=target_qpos,
                dt=dt,
            )
        else:
            result = self.motion_generator.generate(
                build_joint_plan_states(target_qpos),
                options=request.motion_policy.to_motion_gen_options(
                    start_qpos=start_qpos,
                    control_part=control_part,
                    interpolation_dt=context.control_dt,
                ),
            )
        success, trajectory = to_full_robot_trajectory(
            result,
            base_qpos=context.robot.qpos,
            joint_ids=joint_ids,
            env_ids=context.env_ids,
            control_dt=context.require_control_dt(),
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=trajectory,
        )

    def _build_timed_plan(
        self,
        request: ResolvedSkillRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
        trajectory: TimedTrajectory,
        success: bool | torch.Tensor,
    ) -> SkillPlan:
        """Apply the same full-robot command grid to authored and observed motion."""
        motion = request.binding.endpoint("primary", "motion")
        joint_ids = motion.require_target(JointPositionTarget).joint_ids
        if sorted(joint_ids) != list(range(context.robot.robot_dof)):
            raise ValueError("Timed replay requires a full-robot motion binding.")
        if not torch.equal(trajectory.env_ids, context.env_ids):
            raise ValueError("Timed replay must match the selected environment rows.")
        if trajectory.positions.device != context.robot.qpos.device:
            raise ValueError("Timed replay must be on the bound robot device.")
        if request.motion_policy.dynamic_collision_mode is not DynamicCollisionMode.OFF:
            raise ValueError(
                "Authored replay requires dynamic collision planning to be off."
            )
        if not trajectory.dt.is_floating_point() or not torch.allclose(
            trajectory.dt,
            torch.full_like(trajectory.dt, context.require_control_dt()),
            rtol=1e-6,
            atol=0,
        ):
            raise ValueError("Each timed replay command must occupy one control step.")
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=trajectory,
            replannable=False,
            preserve_velocity_samples=request.skill_options.preserve_velocity_samples,
        )

    def _plan_observed_chunk(
        self,
        request: ResolvedSkillRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
        target: ObservedJointTrajectory,
        chunk: int,
        start: int,
        active_mask: torch.Tensor,
    ) -> SkillPlan:
        """Resolve the next range after the preceding command is acknowledged."""
        stop = (
            sum(target.chunk_lengths)
            if context.is_preview
            else start + target.chunk_lengths[chunk]
        )
        source = self.planning_services.joint_trajectory_source(target.source_id)
        trajectory = source.resolve(target.phase, start, stop, context, active_mask)
        if not isinstance(trajectory, TimedTrajectory):
            raise TypeError(
                "An observed joint trajectory source must return TimedTrajectory."
            )
        if trajectory.waypoint_count != stop - start:
            raise ValueError(
                "An observed trajectory must supply the declared command count."
            )
        if (
            request.skill_options.preserve_velocity_samples
            and trajectory.velocities is None
        ):
            raise ValueError(
                "Preserving velocity samples requires a timed velocity trajectory."
            )
        plan = self._build_timed_plan(request, context, trajectory, active_mask)
        if context.is_preview or len(target.chunk_lengths) == 1:
            return plan
        return replace(
            plan,
            continuation=_JointTrajectoryCursor(
                chunk + 1, len(target.chunk_lengths), target.phase, stop
            ),
        )

    def continue_plan(
        self,
        request: ResolvedSkillRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
        plan: SkillPlan,
        active_mask: torch.Tensor,
    ) -> PlanContinuationResult:
        """Resolve another observed range within the existing MoveJoints invocation.

        Args:
            request: The unchanged grounded invocation.
            context: Observations from the last acknowledged control step.
            plan: Completed chunk and its command cursor.
            active_mask: Rows that executed the completed chunk.

        Returns:
            Accepted rows and their next timed plan, if more commands remain.
        """
        target = request.goal.target
        if not isinstance(target, ObservedJointTrajectory):
            return super().continue_plan(request, context, plan, active_mask)
        cursor = plan.continuation
        if not isinstance(cursor, _JointTrajectoryCursor):
            raise TypeError("Expected an observed joint trajectory continuation.")
        next_plan = None
        if cursor.position < cursor.limit and bool(active_mask.any()):
            next_plan = self._plan_observed_chunk(
                request, context, target, cursor.position, cursor.stop, active_mask
            )
        return PlanContinuationResult(active_mask, next_plan)

    def _resolve_target_qpos(
        self,
        goal: JointPositionGoal,
        *,
        request: ResolvedSkillRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
    ) -> torch.Tensor:
        """Resolve an explicit or named joint goal to a tensor."""
        if isinstance(goal.target, torch.Tensor):
            return goal.target
        return request.binding.endpoint("primary", "motion").joint_positions(
            goal.target,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )


__all__ = [
    "JointPositionGoal",
    "MoveJoints",
    "MoveJointsOptions",
    "ObservedJointTrajectory",
    "ObservedJointTrajectorySource",
]
