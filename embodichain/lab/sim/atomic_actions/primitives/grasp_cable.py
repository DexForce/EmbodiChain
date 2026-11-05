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

"""Plan a frictional cable pinch without a rigid-object attachment effect."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import ClassVar
import torch
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.requirements import (
    SkillBindingContract,
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
)
from embodichain.lab.sim.atomic_actions.control import (
    OPEN_COMMAND,
    GRASP_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    resolve_pose_target,
    to_full_robot_trajectory,
)
from embodichain.lab.sim.motion.planners.utils import interpolate_xpos_batched
from embodichain.lab.sim.atomic_actions.plans import TimedTrajectory

__all__ = ["CableGraspGoal", "GraspCableOptions", "GraspCable"]


@dataclass(frozen=True, slots=True, eq=False)
class CableGraspGoal:
    """Resolved pinch pose plus a stable material-coordinate identity.

    Args:
        cable_uid: Cable identity retained for the caller's physical validator.
        material_distance: Resolved arc length from the proximal end, in metres.
        grasp_pose: ``(4,4)`` or ``(B,4,4)`` TCP pose in Robot FK's Arena-local
            frame. The caller converts world cable observations before planning.
    """

    goal_kind: ClassVar[str] = "cable_grasp"
    cable_uid: str
    material_distance: float
    grasp_pose: torch.Tensor

    def __post_init__(self) -> None:
        from embodichain.lab.sim.atomic_actions.goals import validate_pose_goal

        if (
            not isinstance(self.cable_uid, str)
            or not self.cable_uid
            or not math.isfinite(self.material_distance)
            or self.material_distance < 0
        ):
            raise ValueError(
                "Cable identity and finite nonnegative material distance are required."
            )
        validate_pose_goal(self.grasp_pose, "grasp_pose", allow_waypoints=False)


@dataclass(frozen=True, slots=True, eq=False)
class GraspCableOptions(ActionOptions):
    """Pinch dwell and world-Z lift behavior; timing grid is engine-owned."""

    close_steps: int = 48
    close_hold_duration: float = 0.0
    """Seconds to maintain the closed command before lifting.

    Rounded up to the control grid and included in the ``close`` phase.
    """
    lift_height: float = 0.0185
    lift_speed: float = 0.05
    """World-Z lift speed in metres per second, sampled on the control grid."""

    def __post_init__(self) -> None:
        if (
            isinstance(self.close_steps, bool)
            or not isinstance(self.close_steps, int)
            or self.close_steps < 2
            or isinstance(self.close_hold_duration, bool)
            or not math.isfinite(self.close_hold_duration)
            or self.close_hold_duration < 0
            or not math.isfinite(self.lift_height)
            or self.lift_height <= 0
            or not math.isfinite(self.lift_speed)
            or self.lift_speed <= 0
        ):
            raise ValueError(
                "Cable grasp requires finite positive lift height/speed, "
                "nonnegative close_hold_duration and integer close_steps >= 2."
            )


class GraspCable(AtomicAction[CableGraspGoal, GraspCableOptions]):
    """Descend, close and lift using contact alone.

    The caller supplies a fresh cable-derived pose and verifies material-point
    lift/slip after execution. Planning does not step physics or assert a grasp.
    """

    skill_id: ClassVar[str] = "grasp_cable"
    GoalType: ClassVar[type] = CableGraspGoal
    OptionsType: ClassVar[type] = GraspCableOptions
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset(
                    {CARTESIAN_POSE_CAPABILITY, FORWARD_KINEMATICS_CAPABILITY}
                ),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
        )
    )

    def _plan(
        self,
        request: ResolvedActionRequest[CableGraspGoal, GraspCableOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        goal = self.require_goal(request)
        motion = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        hand = request.binding.endpoint("primary", "grasp")
        fingers = list(hand.require_target(JointPositionTarget).joint_ids)
        ids = list(motion.joint_ids)
        open_q = hand.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        closed_q = hand.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        target = resolve_pose_target(
            goal.grasp_pose, num_envs=context.batch_size, device=self.device
        ).to(context.robot.qpos)

        def move(start_full: torch.Tensor, end: torch.Tensor, *, sample_count: int):
            start = self.robot.compute_fk(
                start_full[:, ids], name=motion.control_part, to_matrix=True
            )
            poses = interpolate_xpos_batched(start, end, sample_count)[:, 1:]
            result = self.motion_generator.generate(
                build_pose_plan_states(poses),
                options=replace(
                    request.motion_policy, sample_count=sample_count
                ).to_motion_gen_options(
                    start_qpos=start_full[:, ids],
                    control_part=motion.control_part,
                    interpolation_dt=context.require_control_dt(),
                    cartesian_linear=True,
                ),
            )
            success, trajectory = to_full_robot_trajectory(
                result,
                base_qpos=start_full,
                joint_ids=ids,
                env_ids=context.env_ids,
                control_dt=context.require_control_dt(),
            )
            return success, trajectory.positions.clone()

        ok1, down = move(
            context.robot.qpos, target, sample_count=request.motion_policy.sample_count
        )
        down[:, :, fingers] = open_q[:, None, :]
        close = down[:, -1:].expand(-1, request.skill_options.close_steps, -1).clone()
        weights = torch.linspace(
            0, 1, request.skill_options.close_steps, device=self.device
        )[None, :, None]
        close[:, :, fingers] = (
            open_q[:, None, :] * (1 - weights) + closed_q[:, None, :] * weights
        )
        hold_steps = math.ceil(
            request.skill_options.close_hold_duration / context.require_control_dt()
        )
        if hold_steps:
            close = torch.cat((close, close[:, -1:].expand(-1, hold_steps, -1)), dim=1)
        lifted = target.clone()
        lifted[:, 2, 3] += request.skill_options.lift_height
        lift_samples = max(
            2,
            math.ceil(
                request.skill_options.lift_height
                / (request.skill_options.lift_speed * context.require_control_dt())
            )
            + 1,
        )
        ok2, up = move(close[:, -1], lifted, sample_count=lift_samples)
        up[:, :, fingers] = closed_q[:, None, :]
        return self.build_plan(
            request,
            context,
            success=ok1 & ok2,
            trajectory=TimedTrajectory.from_uniform_step(
                torch.cat((down, close, up), 1),
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            segment_lengths={
                "descend": down.shape[1],
                "close": close.shape[1],
                "lift": up.shape[1],
            },
        )
