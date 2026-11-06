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

"""MoveEndEffector atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
from collections.abc import Callable
from embodichain.lab.sim.atomic_actions.policies import PlanningRecovery
from embodichain.lab.sim.atomic_actions.released_continuation import (
    ReleasedHandClearanceRecovery,
)
from embodichain.lab.sim.atomic_actions.core import CompletePlanBuilder

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.goals import (
    PoseGoalValue,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    resolve_pose_target,
    to_full_robot_trajectory,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_motion_slot,
)


@dataclass(frozen=True, slots=True, eq=False)
class EndEffectorPoseGoal:
    """End-effector pose goal with optional batched intermediate waypoints."""

    xpos: PoseGoalValue
    """Homogeneous pose with shape ``(4,4)``, ``(B,4,4)`` or ``(B,N,4,4)``."""

    def __post_init__(self) -> None:
        validate_pose_goal(self.xpos, "xpos", allow_waypoints=True)


@dataclass(frozen=True, slots=True, eq=False)
class MoveEndEffectorOptions(ActionOptions):
    """Per-invocation behavior for :class:`MoveEndEffector`."""

    linear_segments: bool = field(default=False, kw_only=True)
    """Select source fixed-grid Cartesian sampling for a free end-effector move."""

    def __post_init__(self) -> None:
        if type(self.linear_segments) is not bool:
            raise TypeError("linear_segments must be bool.")


class MoveEndEffector(AtomicAction[EndEffectorPoseGoal, MoveEndEffectorOptions]):
    """Plan a free-space move for a bound manipulator."""

    skill_id: ClassVar[str] = "move_end_effector"
    GoalType: ClassVar[type] = EndEffectorPoseGoal
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_motion_slot(
                "primary",
                capabilities=frozenset({CARTESIAN_POSE_CAPABILITY}),
            ),
        ),
    )
    OptionsType: ClassVar[type] = MoveEndEffectorOptions
    PlanningRecoveryType: ClassVar[type] = ReleasedHandClearanceRecovery

    def plan_complete_request(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, MoveEndEffectorOptions],
        context: PlanningContext,
        *,
        build: CompletePlanBuilder,
        external_nominal: bool = False,
    ) -> ActionPlan:
        """Qualify the entire cleanup and its exact declared future Park sequence."""
        if (
            type(request.recovery_policy.planning_recovery)
            is not ReleasedHandClearanceRecovery
        ):
            return build(request)
        if external_nominal:
            raise ValueError(
                "Released cleanup does not support external nominal motion."
            )
        provider = self.planning_services.released_hand_provider
        query = self.planning_services.free_motion_query
        if provider is None or query is None:
            raise ValueError(
                "Released cleanup requires declared suffix and collision services."
            )
        session = provider.prepare(request, context)
        if session is None:
            raise ValueError("Released cleanup has no exact declared occurrence.")
        from embodichain.lab.sim.atomic_actions.primitives._released_clearance import (
            recover,
        )

        def complete(selected, current):
            if current is not context:
                raise ValueError("Cleanup alternatives changed their observed context.")
            return build(
                selected,
                planner=lambda exact, observed: self._plan_nominal(exact, observed),
            )

        original = complete(request, context)
        return recover(
            request,
            context,
            original,
            build=complete,
            session=session,
            robot=self.robot,
            query=query,
            plan_preview=build.plan_preview,
        )

    def _validate_segment_motion_recovery(self, request, context) -> None:
        if request.skill_options.linear_segments:
            if (
                context.batch_size != 1
                or context.env_ids.tolist() != [0]
                or (
                    request.recovery_policy.planning_recovery is not None
                    and type(request.recovery_policy.planning_recovery)
                    is not ReleasedHandClearanceRecovery
                )
                or request.motion_policy.dynamic_collision_mode != "off"
                or request.motion_policy.segment_sample_count_retries != (260, 320)
                or request.motion_policy.segment_continuous_ik_recovery
            ):
                raise ValueError(
                    "Free Cartesian end-effector segments require the source B1 grid without articulation recovery."
                )
            return

        raise ValueError(
            "End-effector segment recovery requires explicit linear_segments in this layer."
        )

    def _plan(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, MoveEndEffectorOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        return self._plan_nominal(request, context)

    def _qualify_final_plan(self, request, context, plan):
        return None

    def _plan_nominal(
        self,
        request: ResolvedActionRequest[EndEffectorPoseGoal, MoveEndEffectorOptions],
        context: PlanningContext,
        *,
        generate: Callable | None = None,
    ) -> ActionPlan:
        """Plan an end-effector pose goal from the observed joint state."""
        goal = request.goal
        motion_target = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        control_part = motion_target.control_part
        joint_ids = list(motion_target.joint_ids)
        move_xpos = resolve_pose_target(
            resolve_pose_goal(goal.xpos, context, name="xpos"),
            num_envs=context.batch_size,
            device=self.device,
        )
        start_qpos = context.robot.qpos[:, joint_ids]
        targets = build_pose_plan_states(move_xpos)
        motion_options = request.motion_policy.to_motion_gen_options(
            start_qpos=start_qpos,
            control_part=control_part,
            interpolation_dt=context.control_dt,
            cartesian_keyframes=request.skill_options.linear_segments
            or generate is not None,
        )
        if request.skill_options.linear_segments:
            if (
                context.batch_size != 1
                or generate is not None
                or (
                    request.recovery_policy.planning_recovery is not None
                    and type(request.recovery_policy.planning_recovery)
                    is not ReleasedHandClearanceRecovery
                )
            ):
                raise ValueError(
                    "Free Cartesian end-effector segments require B1 without articulation recovery."
                )
            result = self.motion_generator.generate(targets, options=motion_options)
        else:
            result = (self.motion_generator.generate if generate is None else generate)(
                targets, options=motion_options
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


__all__ = [
    "EndEffectorPoseGoal",
    "MoveEndEffector",
    "MoveEndEffectorOptions",
]
