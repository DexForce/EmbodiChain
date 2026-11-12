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

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_skills.bindings import JointPositionTarget
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


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionGoal:
    """Explicit or named joint-space goal for a bound robot resource."""

    target: torch.Tensor | str | TimedTrajectory
    """Joint waypoints, a named command, or an explicitly timed full-joint replay.

    Authored timed replay requires dynamic collision planning to be disabled.
    """

    def __post_init__(self) -> None:
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
                "target must be a torch.Tensor, str or TimedTrajectory, "
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
        motion = request.binding.endpoint("primary", "motion")
        motion_target = motion.require_target(JointPositionTarget)
        control_part = motion_target.control_part
        joint_ids = list(motion_target.joint_ids)
        if isinstance(goal.target, TimedTrajectory):
            trajectory = goal.target
            if sorted(joint_ids) != list(range(context.robot.robot_dof)):
                raise ValueError("Timed replay requires a full-robot motion binding.")
            if not torch.equal(trajectory.env_ids, context.env_ids):
                raise ValueError(
                    "Timed replay must match the selected environment rows."
                )
            if trajectory.positions.device != context.robot.qpos.device:
                raise ValueError("Timed replay must be on the bound robot device.")
            if (
                request.motion_policy.dynamic_collision_mode
                is not DynamicCollisionMode.OFF
            ):
                raise ValueError(
                    "Authored replay requires dynamic collision planning to be off."
                )
            if not trajectory.dt.is_floating_point() or not torch.allclose(
                trajectory.dt,
                torch.full_like(trajectory.dt, context.require_control_dt()),
                rtol=1e-6,
                atol=0,
            ):
                raise ValueError(
                    "Each timed replay command must occupy one control step."
                )
            return self.build_plan(
                request, context, success=True, trajectory=trajectory, replannable=False
            )

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
]
