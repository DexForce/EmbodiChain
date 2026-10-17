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

"""MoveJoints atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.sim.atomic_actions.trajectory_assembly import (
    SkillPlanSemantics,
    SkillTrajectoryContract,
    SkillTrajectoryInput,
)
from embodichain.lab.sim.atomic_actions.requirements import (
    JOINT_POSITION_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_joint_plan_states,
    resolve_joint_target,
    to_full_robot_trajectory,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_motion_slot,
)


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionGoal:
    """Explicit or named joint-space goal for a bound robot resource."""

    target: torch.Tensor | str
    """Joint qpos/waypoints or a named control-part profile command."""

    def __post_init__(self) -> None:
        if isinstance(self.target, str):
            if not self.target.strip():
                raise ValueError("Named joint-position target must not be empty.")
            return
        if not isinstance(self.target, torch.Tensor):
            raise TypeError(
                "target must be a torch.Tensor or str, "
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
class MoveJointsOptions(ActionOptions):
    """Per-invocation behavior for :class:`MoveJoints`."""


class MoveJoints(AtomicAction[JointPositionGoal, MoveJointsOptions]):
    """Plan joint motion from the observed state to one or more waypoints."""

    skill_id: ClassVar[str] = "move_joints"
    GoalType: ClassVar[type] = JointPositionGoal
    OptionsType: ClassVar[type] = MoveJointsOptions
    agent_visible: ClassVar[bool] = True
    trajectory_contract: ClassVar[SkillTrajectoryContract] = SkillTrajectoryContract(
        schema_id="move_joints.qpos",
        revision="1",
        landmark_names=("goal",),
        landmark_stages={"goal": None},
    )
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
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a joint-space goal without mutating the robot or task state."""
        goal = request.goal
        motion = request.binding.endpoint("primary", "motion")
        motion_target = motion.require_target(JointPositionTarget)
        control_part = motion_target.control_part
        joint_ids = list(motion_target.joint_ids)
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
        return self.assemble_trajectory_plan(
            request,
            context,
            SkillTrajectoryInput(
                schema_id=self.trajectory_contract.schema_id,
                schema_revision=self.trajectory_contract.revision,
                trajectory=trajectory,
                landmarks={"goal": trajectory.waypoint_count - 1},
                planning_success=success,
                replannable=True,
            ),
        )

    def _trajectory_semantics(
        self,
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
        motion: SkillTrajectoryInput,
    ) -> SkillPlanSemantics:
        """Check ordered joint goals without introducing symbolic task effects."""
        target = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        goals = resolve_joint_target(
            self._resolve_target_qpos(request.goal, request=request, context=context),
            num_envs=context.batch_size,
            joint_dof=len(target.joint_ids),
            control_part=target.control_part,
            device=self.device,
        )
        if goals.ndim == 2:
            goals = goals[:, None]
        positions = motion.trajectory.positions[:, :, list(target.joint_ids)]
        if motion.landmarks["goal"] != motion.trajectory.waypoint_count - 1:
            raise ValueError("MoveJoints goal event must be the terminal sample")
        # This checks the reference path, not measured controller acceptance.
        tolerance = 1e-4
        for row in motion.planning_success.nonzero().flatten().tolist():
            if not torch.allclose(
                positions[row, -1], goals[row, -1].to(positions), atol=tolerance, rtol=0
            ):
                raise ValueError(
                    "MoveJoints trajectory does not reach its requested joint goal"
                )
            path = positions[row]
            if len(path) == 1:
                if not torch.allclose(
                    goals[row].to(path),
                    path.expand_as(goals[row]),
                    atol=tolerance,
                    rtol=0,
                ):
                    raise ValueError(
                        "MoveJoints trajectory must visit requested waypoints in order"
                    )
                continue
            edges = path[1:] - path[:-1]
            squared_lengths = edges.square().sum(dim=-1)
            cursor = 0.0
            for waypoint in goals[row]:
                # Control-grid retiming can put an interior waypoint between
                # samples. Check the ordered reference polyline, not only exact
                # sample equality; this still does not certify physical motion.
                delta = waypoint.to(path) - path[:-1]
                fractions = (
                    (delta * edges).sum(dim=-1)
                    / squared_lengths.clamp_min(torch.finfo(path.dtype).tiny)
                ).clamp(0, 1)
                closest = path[:-1] + fractions[:, None] * edges
                progress = torch.arange(
                    len(edges), device=path.device, dtype=torch.float64
                ) + fractions.to(torch.float64)
                matches = (
                    (
                        ((closest - waypoint.to(path)).abs() <= tolerance).all(dim=-1)
                        & (progress >= cursor - 1e-8)
                    )
                    .nonzero()
                    .flatten()
                )
                if not len(matches):
                    raise ValueError(
                        "MoveJoints trajectory must visit requested waypoints in order"
                    )
                cursor = float(progress[matches[0]])
        return SkillPlanSemantics()

    def _resolve_target_qpos(
        self,
        goal: JointPositionGoal,
        *,
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
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
