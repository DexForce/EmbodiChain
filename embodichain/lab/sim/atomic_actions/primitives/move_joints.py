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

from dataclasses import dataclass, field, replace
from typing import ClassVar
from ..articulation_continuation import ArticulationParkPlanningRecovery
from embodichain.lab.sim.atomic_actions.released_continuation import (
    ReleasedHandParkRecovery,
)
from embodichain.lab.sim.atomic_actions.core import CompletePlanBuilder
from embodichain.lab.sim.atomic_actions.policies import PlanningRecovery

import torch
import math

from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import ActionPlan
from embodichain.lab.sim.atomic_actions.plans import (
    PlannerDiagnostics,
    PlanQualification,
    PlanningFailure,
)
from embodichain.compute.trajectory.limits import joint_velocity_validity
from embodichain.lab.sim.motion.limits import authored_joint_velocity_limits
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
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


@dataclass(frozen=True, slots=True)
class JointOffsetTarget:
    """Explicit relative joint target with optional model-link clearance search.

    The protected link is a geometric identity, not a resource inferred from its
    spelling. Motion owns the bounded search; normal offsets require no solver.

    Args:
        joint_name: Exact model joint inside the bound motion endpoint.
        offset: Finite nonzero displacement from the current observed position.
        protected_link_name: Optional explicit model link protected by Motion's
            bounded clearance search. Omission uses the authored offset.
    """

    joint_name: str
    offset: float
    protected_link_name: str | None = None

    def __post_init__(self) -> None:
        for name in ("joint_name", "protected_link_name"):
            value = getattr(self, name)
            if value is None and name == "protected_link_name":
                continue
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Joint offset geometry identities must be exact.")
        if (
            type(self.offset) not in (int, float)
            or not math.isfinite(self.offset)
            or self.offset == 0
        ):
            raise ValueError("Joint offset must be finite and nonzero.")


@dataclass(frozen=True, slots=True, eq=False)
class JointPositionGoal:
    """Explicit or named joint-space goal for a bound robot resource."""

    target: torch.Tensor | str | JointOffsetTarget
    """Joint qpos/waypoints or a named control-part profile command."""

    def __post_init__(self) -> None:
        if type(self.target) is JointOffsetTarget:
            return
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

    require_free_motion: bool = field(default=False, kw_only=True)
    """Require an empty B=1/env0 resource and discrete robot/scene qualification.

    The exact full-robot start, passive joints and authored position/velocity
    bounds are checked before dispatch, including after a plan transform.
    A configured :class:`~embodichain.lab.sim.motion.mesh_collision.FreeMotionQuery`
    is mandatory. This does not certify continuous collision or physical contact.
    """

    def __post_init__(self) -> None:
        if type(self.require_free_motion) is not bool:
            raise TypeError("require_free_motion must be a boolean.")


class MoveJoints(AtomicAction[JointPositionGoal, MoveJointsOptions]):
    """Plan joint motion from the observed state to one or more waypoints."""

    skill_id: ClassVar[str] = "move_joints"
    GoalType: ClassVar[type] = JointPositionGoal
    OptionsType: ClassVar[type] = MoveJointsOptions
    PlanningRecoveryType: ClassVar[tuple[type[PlanningRecovery], ...]] = (
        ArticulationParkPlanningRecovery,
        ReleasedHandParkRecovery,
    )
    agent_visible: ClassVar[bool] = True
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_motion_slot(
                "primary",
                capabilities=frozenset({JOINT_POSITION_CAPABILITY}),
            ),
        ),
    )

    def plan_complete_request(
        self,
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
        *,
        build: CompletePlanBuilder,
        external_nominal: bool = False,
    ) -> ActionPlan:
        """Qualify an explicitly bound empty-hand Park after full planning."""
        if (
            type(request.recovery_policy.planning_recovery)
            is not ReleasedHandParkRecovery
        ):
            return build(request)
        if external_nominal:
            raise ValueError("Released Park does not support external nominal motion.")
        provider = self.planning_services.released_hand_provider
        if provider is None or self.planning_services.free_motion_query is None:
            raise ValueError(
                "Released Park requires declared authority and collision services."
            )
        provider.require_park(request, context)
        from embodichain.lab.sim.atomic_actions.primitives._released_park import (
            qualify_and_recover_park,
        )

        def complete(selected):
            return build(
                selected,
                planner=lambda exact, observed: self._plan_nominal(exact, observed),
            )

        return qualify_and_recover_park(
            self, request, context, complete(request), complete
        )

    def _plan(
        self,
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        if (
            request.recovery_policy.planning_recovery is None
            and self.planning_services.articulation_continuation_provider is None
        ):
            return self._plan_nominal(request, context)
        from embodichain.lab.sim.atomic_actions.primitives._articulation_recovery import (
            plan_articulation,
        )

        return plan_articulation(
            self,
            request,
            context,
            lambda selected, generate: self._plan_nominal(selected, context),
            kind="park",
        )

    def _plan_nominal(
        self,
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a joint-space goal without mutating the robot or task state."""
        if request.skill_options.require_free_motion:
            try:
                self._require_empty_resource(request, context)
            except ValueError as error:
                return self.failed_plan(
                    request,
                    context,
                    message=str(error),
                    failure_code="idle_clearance_unqualified",
                    retryable=False,
                )
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
        plan = self.build_plan(
            request,
            context,
            success=success,
            trajectory=trajectory,
        )
        if not request.skill_options.require_free_motion or not plan.plan_success.all():
            return plan
        qualification = self._qualify_free_motion(request, context, plan)
        if not qualification.valid_mask.all():
            return self.failed_plan(
                request,
                context,
                message="; ".join(qualification.diagnostics.messages),
                failure_code="idle_clearance_unqualified",
                retryable=False,
            )
        return replace(
            plan,
            diagnostics=replace(
                plan.diagnostics,
                metadata={
                    **plan.diagnostics.metadata,
                    **qualification.diagnostics.metadata,
                },
            ),
        )

    @staticmethod
    def _require_empty_resource(request, context: PlanningContext) -> None:
        if context.batch_size != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("Free-motion qualification requires B1/env0.")
        resource = request.binding.endpoint("primary", "motion").task_state_key
        held = context.get_held_object(resource)
        if held is not None and context.task.held_object_mask(resource).any():
            raise ValueError("Free motion cannot move a held object.")
        if any(
            resource in pair and (state.env_mask is None or state.env_mask.any())
            for pair, state in context.task.coordinated_held_objects.items()
        ):
            raise ValueError("Free motion cannot move a coordinated-held resource.")

    def _qualify_free_motion(self, request, context, plan) -> PlanQualification:
        try:
            self._require_empty_resource(request, context)
            endpoint = request.binding.endpoint("primary", "motion")
            arm = endpoint.require_target(JointPositionTarget)
            path = plan.joint_trajectory
            if (
                path is None
                or not path.waypoint_count
                or not torch.equal(path.positions[:, 0], context.robot.qpos)
            ):
                raise ValueError("Free motion must retain its exact full-robot start.")
            passive = [i for i in range(path.robot_dof) if i not in arm.joint_ids]
            if not torch.equal(
                path.positions[:, :, passive],
                context.robot.qpos[:, None, passive].expand(
                    -1, path.waypoint_count, -1
                ),
            ):
                raise ValueError("Free motion changed an uncommanded joint.")
            bounds = self.robot.get_qpos_limits(name=arm.control_part).to(
                path.positions
            )
            active = path.positions[:, :, list(arm.joint_ids)]
            if not (
                (active >= bounds[:, None, :, 0]) & (active <= bounds[:, None, :, 1])
            ).all():
                raise ValueError("Free motion violates original joint limits.")
            if not joint_velocity_validity(
                path.positions,
                path.dt,
                authored_joint_velocity_limits(self.robot, None).to(path.positions),
            ).all():
                raise ValueError("Free motion violates original velocity limits.")
            query = self.planning_services.free_motion_query
            if query is None:
                raise ValueError("Free-motion collision query is unavailable.")
            checked = query.inspect(path.positions[0], control_part=arm.control_part)
            if not isinstance(checked, MotionCollisionReport):
                raise TypeError("Free-motion query returned an invalid report.")
            if checked.scope != "discrete_free_motion_meshes":
                raise ValueError("Free motion requires robot and scene qualification.")
            if not checked.valid:
                raise ValueError(
                    f"Robot/scene free motion rejected: {checked.metadata()}"
                )
            if checked.frames != path.waypoint_count:
                raise ValueError(
                    "Free-motion query did not qualify every emitted sample."
                )
            return PlanQualification(
                plan.plan_success,
                PlannerDiagnostics(
                    self.planning_services.planner_name,
                    metadata={
                        "idle_clearance": {
                            "resource": endpoint.task_state_key,
                            "qualification": checked.metadata(),
                        }
                    },
                ),
            )
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            return PlanQualification(
                torch.zeros_like(plan.plan_success),
                PlannerDiagnostics(
                    self.planning_services.planner_name,
                    messages=(str(error),),
                    failure=PlanningFailure(
                        "idle_final_plan_unqualified", retryable=False
                    ),
                ),
            )

    def _qualify_final_plan(self, request, context, plan) -> PlanQualification | None:
        if type(request.recovery_policy.planning_recovery) is ReleasedHandParkRecovery:
            return None
        if request.recovery_policy.planning_recovery is not None:
            from embodichain.lab.sim.atomic_actions.primitives._articulation_recovery import (
                reject_articulation_transform,
            )

            return reject_articulation_transform(self, request, plan)
        if not request.skill_options.require_free_motion or not plan.plan_success.all():
            return None
        return self._qualify_free_motion(request, context, plan)

    def _resolve_target_qpos(
        self,
        goal: JointPositionGoal,
        *,
        request: ResolvedActionRequest[JointPositionGoal, MoveJointsOptions],
        context: PlanningContext,
    ) -> torch.Tensor:
        """Resolve an explicit or named joint goal to a tensor."""
        if type(goal.target) is JointOffsetTarget:
            if not request.skill_options.require_free_motion:
                raise ValueError(
                    "Joint offset goals require explicit empty-arm free-motion qualification."
                )
            self._require_empty_resource(request, context)
            endpoint = request.binding.endpoint("primary", "motion")
            target = endpoint.require_target(JointPositionTarget)
            names = [self.robot.joint_names[index] for index in target.joint_ids]
            if goal.target.joint_name not in names:
                raise ValueError("Idle joint is outside the bound motion endpoint.")
            offset = goal.target.offset
            if goal.target.protected_link_name is not None:
                from embodichain.lab.sim.motion.joint_clearance import (
                    select_clearance_joint_offset,
                )

                query = self.planning_services.free_motion_query
                if query is None:
                    raise ValueError(
                        "Idle search requires an explicit free-motion query."
                    )
                offset = select_clearance_joint_offset(
                    self.robot,
                    context.robot.qpos,
                    active_control_part=target.control_part,
                    joint_name=goal.target.joint_name,
                    maximum_offset=offset,
                    protected_link_name=goal.target.protected_link_name,
                    query=query,
                )
            result = context.robot.qpos[:, list(target.joint_ids)].clone()
            result[:, names.index(goal.target.joint_name)] += offset
            return result
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
    "JointOffsetTarget",
    "MoveJoints",
    "MoveJointsOptions",
]
