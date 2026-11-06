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

"""MoveHeldObject atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import ClassVar, Literal

import torch

from embodichain.compute.geometry.rigid import rotate_final_pose_world_z
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    arm_qpos_from_state,
    require_shared_task_state_key,
)
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction, CompletePlanBuilder
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import (
    PoseGoalValue,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.nominal import NominalMotionSource
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    PlannerDiagnostics,
    PlanningFailure,
    PlanQualification,
)
from embodichain.lab.sim.atomic_actions.policies import PlanningRecovery
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    build_pose_plan_states,
    resolve_pose_target,
    to_full_robot_trajectory,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)


@dataclass(frozen=True, slots=True, eq=False)
class HeldObjectPoseGoal:
    """Desired pose for the object held by this action's control part."""

    object_target_pose: PoseGoalValue
    """Object target pose or ordered waypoints.

    Accepts ``(4, 4)``, ``(num_envs, 4, 4)`` or
    ``(num_envs, n_waypoint, 4, 4)``; every waypoint retains the held grasp.
    """

    world_yaw_free: bool = False
    """Permit world-Z yaw changes at the final waypoint, retaining its position.

    Intermediate waypoints remain exact. Planning tries the nominal heading
    first, then seven 45-degree heading alternatives; each complete path must
    pass the motion generator's normal checks. This is not an exhaustive search.
    """

    object_id: str | None = field(default=None, kw_only=True)
    """Optional exact held-object identity and retained-attachment effect.

    Named goals reject another held object before planning and request terminal
    effect verification without changing its verified attachment transform.
    Omission retains the existing effectless transport contract.
    """

    def __post_init__(self) -> None:
        if type(self.world_yaw_free) is not bool:
            raise TypeError("world_yaw_free must be a boolean.")
        if self.object_id is not None and (
            type(self.object_id) is not str
            or not self.object_id
            or self.object_id != self.object_id.strip()
        ):
            raise ValueError("object_id must be an exact non-empty identifier or None.")
        validate_pose_goal(
            self.object_target_pose,
            "object_target_pose",
            allow_waypoints=True,
        )


@dataclass(frozen=True, slots=True, eq=False)
class MoveHeldObjectOptions(ActionOptions):
    """Per-invocation held-object transport behavior."""

    drawer_transport: bool = field(default=False, kw_only=True)
    """Select explicit source drawer planning with its declared physical provider."""

    preserve_vertical: bool = field(default=False, kw_only=True)
    """Keep drawer payload targets upright instead of above-rim drop release."""

    linear_segments: bool = field(default=False, kw_only=True)
    """Plan the declared Cartesian grid with source per-heading density retries."""

    lift_clearance: float | None = field(default=None, kw_only=True)
    """Optional height above both TCP endpoints for a single-pose lift route.

    Ordered object-pose waypoints remain exact. A single target is approached
    through a vertical lift and a translation above the target, in metres.
    """

    hand_hold: Literal["grasp_command", "observed"] = field(
        default="grasp_command", kw_only=True
    )
    """Hold the resolved grasp command, or the current observed hand posture.

    Observed mode deliberately takes precedence over command overrides and is
    resolved again from each planning context. It does not establish retention.
    """

    report_retained_attachment: bool = field(default=False, kw_only=True)
    """Report the unchanged held relation as an expected terminal effect.

    This neither commits physical state nor changes the verified grasp transform.
    The execution integration still owns effect verification.
    """

    def __post_init__(self) -> None:
        if type(self.drawer_transport) is not bool:
            raise TypeError("drawer_transport must be bool.")
        if type(self.preserve_vertical) is not bool:
            raise TypeError("preserve_vertical must be bool.")
        if self.drawer_transport:
            raise ValueError(
                "Drawer transport is not installed in this held-motion layer."
            )
        if self.preserve_vertical and not self.drawer_transport:
            raise ValueError("preserve_vertical requires explicit drawer planning.")
        if type(self.linear_segments) is not bool:
            raise TypeError("linear_segments must be bool.")
        if self.lift_clearance is not None and (
            isinstance(self.lift_clearance, bool)
            or not isinstance(self.lift_clearance, (int, float))
            or not math.isfinite(self.lift_clearance)
            or self.lift_clearance < 0
        ):
            raise ValueError("lift_clearance must be finite and nonnegative or None.")
        if self.hand_hold not in {"grasp_command", "observed"}:
            raise ValueError("hand_hold must be 'grasp_command' or 'observed'.")
        if type(self.report_retained_attachment) is not bool:
            raise TypeError("report_retained_attachment must be a boolean.")


@dataclass(frozen=True, slots=True)
class MoveHeldObjectPlanningRecovery(PlanningRecovery):
    """Explicit eight-heading screening and unchanged-grid final-heading fallback.

    Optional staging keeps the original B/C distal-envelope hints after that
    complete attempt fails. Host bindings select rigid subjects and an optional
    articulation-only query; command certificates never imply physical retention.
    """

    place_continuation: bool = field(default=False, kw_only=True)
    """Qualify the actual adjacent Place after bounded free-yaw alignment."""

    staging: bool = field(default=False, kw_only=True)
    """Permit declared two-pose rigid staging hints after the original failure."""

    def __post_init__(self) -> None:
        PlanningRecovery.__post_init__(self)
        if type(self.staging) is not bool or type(self.place_continuation) is not bool:
            raise TypeError("Held alignment recovery selections must be bool.")
        if self.place_continuation and not self.staging:
            raise ValueError(
                "Held Place continuation requires declared staging recovery."
            )

    kind: str = field(default="held_alignment_bounded", init=False)
    skill_id: str = field(default="move_held_object", init=False)
    revision: int = field(default=1, init=False)


class MoveHeldObject(AtomicAction[HeldObjectPoseGoal, MoveHeldObjectOptions]):
    """Move the held object through exact target poses with a closed hand.

    Requested orientations are exact unless final world yaw is explicitly free.
    Callers that need a
    transport orientation must encode it in :class:`HeldObjectPoseGoal`; this
    action never substitutes an implicit end-effector orientation.
    """

    skill_id: ClassVar[str] = "move_held_object"
    GoalType: ClassVar[type] = HeldObjectPoseGoal
    PlanningRecoveryType: ClassVar[type] = MoveHeldObjectPlanningRecovery
    OptionsType: ClassVar[type] = MoveHeldObjectOptions
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "primary",
                motion_capabilities=frozenset(
                    {
                        CARTESIAN_POSE_CAPABILITY,
                        FORWARD_KINEMATICS_CAPABILITY,
                    }
                ),
                grasp_commands={GRASP_COMMAND: JointPositionCommand},
            ),
        ),
    )

    def plan_complete_request(
        self,
        request: ResolvedActionRequest[HeldObjectPoseGoal, MoveHeldObjectOptions],
        context: PlanningContext,
        *,
        build: CompletePlanBuilder,
        external_nominal: bool = False,
    ) -> ActionPlan:
        """Dispatch explicit drawer transport or canonical alignment continuation."""
        if request.skill_options.drawer_transport:
            raise ValueError(
                "Drawer transport is not installed in this held-motion layer."
            )
        rule = request.recovery_policy.planning_recovery
        if rule is None or not rule.place_continuation:
            return build(request)
        if external_nominal:
            raise ValueError(
                "Held Place continuation does not support external nominal motion."
            )
        provider = self.planning_services.place_continuation_provider
        if provider is None:
            raise ValueError(
                "Held Place continuation requires its exact occurrence provider."
            )
        preview = provider.prepare(request, context)
        if preview is None:
            raise ValueError("Held Place continuation occurrence is unavailable.")
        from embodichain.lab.sim.atomic_actions.primitives._alignment_continuation import (
            eligible,
            effective_sample_count,
            recover,
        )
        from embodichain.lab.sim.motion.search import PlanningRandomState

        if not eligible(self, request, context, preview):
            raise ValueError(
                "Declared held Place continuation ownership is unavailable."
            )
        before = PlanningRandomState.capture(
            (self.robot.device, context.robot.qpos.device)
        )
        original = build(request)
        if effective_sample_count(original, request) != 260:
            return original
        return recover(
            self,
            request,
            context,
            original,
            preview=preview,
            build=lambda selected: build(
                selected,
                planner=lambda exact, current: self._plan_transport(exact, current),
                action_qualifier=lambda exact, current, plan: None,
            ),
            before_original=before,
        )

    def _plan_from_nominal(
        self,
        request: ResolvedActionRequest[HeldObjectPoseGoal, MoveHeldObjectOptions],
        context: PlanningContext,
        nominal_source: NominalMotionSource,
    ) -> ActionPlan:
        """Assemble externally supplied held-object transport."""
        options = request.skill_options
        if (
            request.recovery_policy.planning_recovery is not None
            or options.linear_segments
            or options.lift_clearance is not None
            or options.hand_hold != "grasp_command"
            or options.report_retained_attachment
        ):
            raise ValueError(
                "External nominal transport does not support lift routes, "
                "observed hand holds, attachment-retention effects or alignment recovery."
            )
        target = self.require_goal(request)
        self._validate_named_hold(request, context)
        motion = self._validate_nominal_motion(
            request, context, nominal_source, segments=("transport",)
        )
        if not motion.success.any():
            return self.failed_plan(
                request, context, message="External transport motion failed."
            )
        arm = request.binding.endpoint("primary", "motion")
        hand = request.binding.endpoint("primary", "grasp")
        endpoint = arm.require_target(JointPositionTarget)
        hand_target = hand.require_target(JointPositionTarget)
        key = require_shared_task_state_key(
            arm, hand, participant="MoveHeldObject primary participant"
        )
        held = context.get_held_object(key)
        if held is None:
            raise ValueError("external transport requires a held object")
        eligible = context.task.exclusive_held_object_mask(key)
        if not eligible.all():
            return self.failed_plan(
                request,
                context,
                message="Held object is not exclusive to the control part.",
            )
        goal = resolve_pose_target(
            resolve_pose_goal(
                target.object_target_pose, context, name="object_target_pose"
            ),
            num_envs=context.batch_size,
            device=self.device,
        )
        if goal.ndim == 4:
            if goal.shape[1] != 1:
                raise ValueError(
                    "external transport requires a single target pose; "
                    "ordered waypoints are unsupported"
                )
            goal = goal[:, 0]
        relation = held.object_to_eef.to(goal)
        if relation.ndim == 2:
            relation = relation.unsqueeze(0)
        self._validate_nominal_anchor(
            motion.trajectory,
            index=motion.trajectory.waypoint_count - 1,
            endpoint=endpoint,
            target=goal @ relation,
        )
        grasp_command = hand.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_path = motion.trajectory.positions[:, :, list(hand_target.joint_ids)]
        if not torch.allclose(
            hand_path, grasp_command[:, None, :].expand_as(hand_path), atol=1e-5, rtol=0
        ):
            raise ValueError(
                "external transport must preserve the configured grasp command"
            )
        return self.build_plan(
            request,
            context,
            success=motion.success & eligible,
            trajectory=motion.trajectory,
            expected_effects=(
                StateDelta(held_object_updates={key: held})
                if target.object_id is not None
                else None
            ),
            segment_lengths=dict(motion.segment_lengths),
            replannable=False,
            diagnostics=PlannerDiagnostics(
                backend="external_nominal", metadata=motion.provenance()
            ),
        )

    def _validate_segment_motion_recovery(self, request, context) -> None:
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or not request.skill_options.linear_segments
            or request.motion_policy.strategy != "ik_interp"
            or request.motion_policy.dynamic_collision_mode != "off"
            or request.motion_policy.segment_continuous_ik_recovery
        ):
            raise ValueError(
                "Held Cartesian density recovery requires B1/env0 linear ik_interp/off; continuity belongs to final-heading recovery."
            )

    def _plan(
        self,
        request: ResolvedActionRequest[HeldObjectPoseGoal, MoveHeldObjectOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        return self._plan_alignment(request, context, allow_staging=True)

    def _plan_alignment(
        self, request, context, *, allow_staging: bool, staging_preparation=None
    ) -> ActionPlan:
        if request.recovery_policy.planning_recovery is None:
            return self._plan_transport(request, context)
        if not request.goal.world_yaw_free:
            raise ValueError(
                "Held alignment recovery requires final world-yaw freedom."
            )
        from embodichain.lab.sim.atomic_actions.primitives._held_alignment import (
            AlignmentCollisionUnavailable,
            alignment_planning_scope,
            recover,
            qualify_alignment_plan,
        )

        from embodichain.lab.sim.atomic_actions.primitives._held_staging import (
            prepare_staging,
            recover_staging,
        )

        before = (
            prepare_staging(self, request, context)
            if allow_staging and request.recovery_policy.planning_recovery.staging
            else staging_preparation
        )
        try:
            with alignment_planning_scope(self.robot, request, context) as scope:
                if before is not None and before.binding.articulation_scene:
                    from embodichain.lab.sim.atomic_actions.planning_probes import (
                        HeldArticulationStagingProvider,
                    )
                    from embodichain.lab.sim.motion.mesh_collision import (
                        RobotArticulationCollisionSnapshot,
                        RobotMeshCollisionSnapshot,
                    )

                    provider = self.planning_services.held_staging_provider
                    if not isinstance(provider, HeldArticulationStagingProvider):
                        raise AlignmentCollisionUnavailable(
                            "Declared articulation staging requires its exact query capability."
                        )
                    query = provider.prepare_articulation_query(request, context)
                    if type(query) is not RobotArticulationCollisionSnapshot:
                        if isinstance(query, RobotMeshCollisionSnapshot):
                            query.close()
                        raise TypeError(
                            "Articulation staging requires RobotArticulationCollisionSnapshot."
                        )
                    scope.articulation_query = query
                first = self._plan_transport(request, context, scope=scope)
                plan = recover(
                    scope,
                    request,
                    context,
                    first,
                    lambda selected: self._plan_transport(
                        selected, context, scope=scope
                    ),
                )
                if before is not None and allow_staging:
                    plan = recover_staging(
                        self,
                        plan,
                        request,
                        context,
                        scope,
                        before.before_original,
                        lambda selected: self._plan_alignment(
                            selected,
                            context,
                            allow_staging=False,
                            staging_preparation=before,
                        ),
                    )
                return qualify_alignment_plan(request, context, plan, self.robot, scope)
        except AlignmentCollisionUnavailable as error:
            failed = self.failed_plan(request, context, message=str(error))
            return replace(
                failed,
                diagnostics=replace(
                    failed.diagnostics,
                    failure=PlanningFailure(
                        "held_alignment_collision_unavailable", retryable=False
                    ),
                ),
            )

    def _qualify_final_plan(self, request, context, plan) -> PlanQualification | None:
        if request.recovery_policy.planning_recovery is None:
            return None
        from embodichain.lab.sim.atomic_actions.primitives._held_alignment import (
            alignment_planning_scope,
            qualify_alignment_plan,
        )

        with alignment_planning_scope(self.robot, request, context) as scope:
            checked = qualify_alignment_plan(request, context, plan, self.robot, scope)
        staging = checked.diagnostics.metadata.get("staging_recovery", {})
        if staging.get("accepted") is True and checked.plan_success.any():
            from embodichain.lab.sim.atomic_actions.primitives._held_staging import (
                inspect_staging_plan,
            )

            certificate = inspect_staging_plan(
                self,
                request,
                context,
                checked,
                require_retained_command=staging.get("retained_hand_boundary", {}).get(
                    "accepted"
                )
                is True,
            )
            checked = replace(
                checked,
                plan_success=checked.plan_success & certificate["valid"],
                diagnostics=replace(
                    checked.diagnostics,
                    metadata={
                        **checked.diagnostics.metadata,
                        "staging_recovery": {
                            **staging,
                            "post_transform_check": certificate,
                        },
                    },
                    **(
                        {}
                        if certificate["valid"]
                        else {
                            "failure": PlanningFailure(
                                "held_staging_safety_failed", retryable=False
                            )
                        }
                    ),
                ),
            )
        return PlanQualification(checked.plan_success, checked.diagnostics)

    def _plan_transport(
        self,
        request: ResolvedActionRequest[HeldObjectPoseGoal, MoveHeldObjectOptions],
        context: PlanningContext,
        *,
        scope=None,
        motion_build=None,
    ) -> ActionPlan:
        """Plan exact object-space transport without changing the attachment."""
        target = request.goal
        self._validate_named_hold(request, context)
        options = request.skill_options
        if options.linear_segments and (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or request.motion_policy.strategy != "ik_interp"
            or request.motion_policy.dynamic_collision_mode != "off"
            or request.motion_policy.interpolation_mode != "fixed_grid"
            or request.motion_policy.ik_seed_mode != "sequential"
        ):
            raise ValueError(
                "Held Cartesian segments require explicit B1/env0 fixed-grid sequential ik_interp/off."
            )
        binding = request.binding
        motion = binding.endpoint("primary", "motion")
        grasp = binding.endpoint("primary", "grasp")
        motion_target = motion.require_target(JointPositionTarget)
        grasp_target = grasp.require_target(JointPositionTarget)
        task_state_key = require_shared_task_state_key(
            motion,
            grasp,
            participant="MoveHeldObject primary participant",
        )
        control_part = motion_target.control_part
        arm_joint_ids = list(motion_target.joint_ids)
        hand_joint_ids = list(grasp_target.joint_ids)
        state = context
        held_object = state.get_held_object(task_state_key)
        if held_object is None:
            raise ValueError(
                "MoveHeldObject requires an object held by task-state resource "
                f"{task_state_key!r} - run PickUp first."
            )
        eligible = context.task.exclusive_held_object_mask(task_state_key)
        if not eligible.any():
            return self.failed_plan(
                request,
                context,
                message="Held object is not exclusive to the control part.",
            )
        object_target_pose = resolve_pose_target(
            resolve_pose_goal(
                target.object_target_pose,
                context,
                name="object_target_pose",
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        start_arm_qpos = arm_qpos_from_state(state, arm_joint_ids)
        object_to_eef = held_object.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.num_envs, 1, 1)
        if options.lift_clearance is not None and object_target_pose.ndim == 3:
            from embodichain.utils.math import pose_inv

            start = self.robot.compute_fk(
                qpos=start_arm_qpos, name=control_part, to_matrix=True
            )
            target_eef = object_target_pose @ object_to_eef
            lifted = start.clone()
            lifted[..., 2, 3] = (
                torch.maximum(start[..., 2, 3], target_eef[..., 2, 3])
                + options.lift_clearance
            )
            above_eef = target_eef.clone()
            above_eef[..., 2, 3] = lifted[..., 2, 3]
            eef_to_object = pose_inv(object_to_eef)
            object_target_pose = torch.stack(
                (lifted @ eef_to_object, above_eef @ eef_to_object, object_target_pose),
                dim=1,
            )
        if scope is not None and scope.replay_result is None:
            scope.prepared_targets = object_target_pose.clone()
        if object_target_pose.ndim == 4:
            object_to_eef = object_to_eef.unsqueeze(1)
        motion_options = request.motion_policy.to_motion_gen_options(
            start_qpos=start_arm_qpos,
            control_part=control_part,
            interpolation_dt=context.control_dt,
            cartesian_keyframes=options.linear_segments,
        )

        def plan_target(poses: torch.Tensor) -> PlanResult:
            targets = build_pose_plan_states(torch.matmul(poses, object_to_eef))
            if scope is not None and scope.replay_result is not None:
                return scope.replay(targets, motion_options)
            if motion_build is not None:
                result = motion_build(targets, motion_options)
            else:
                result = self.motion_generator.generate(targets, options=motion_options)
            if scope is not None:
                from embodichain.lab.sim.atomic_actions.primitives._held_alignment import (
                    screen_alignment_result,
                )

                result = screen_alignment_result(
                    scope,
                    self.motion_generator,
                    result,
                    motion_options,
                    targets=targets,
                )
            return result

        result = plan_target(object_target_pose)
        assert isinstance(result.success, torch.Tensor)
        selected_yaws = start_arm_qpos.new_zeros(self.num_envs)
        attempts = 1
        if target.world_yaw_free:
            for fraction in (0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0):
                missing = eligible & ~result.success
                if not missing.any():
                    break
                angle = math.pi * fraction
                poses = rotate_final_pose_world_z(object_target_pose, angle)
                candidate = plan_target(poses)
                attempts += 1
                accepted = missing & candidate.success
                if accepted.any():
                    result = _merge_yaw_paths(result, candidate, accepted)
                    selected_yaws[accepted] = angle
        if result.positions is None:
            return self.failed_plan(
                request, context, message="No feasible held-object transport path."
            )
        success = result.success & eligible

        base_qpos = state.last_qpos.clone()
        if options.hand_hold == "grasp_command":
            base_qpos[:, hand_joint_ids] = grasp.joint_positions(
                GRASP_COMMAND,
                num_envs=context.batch_size,
                device=self.device,
                dtype=context.robot.qpos.dtype,
            )
        _, timed = to_full_robot_trajectory(
            result,
            base_qpos=base_qpos,
            joint_ids=arm_joint_ids,
            env_ids=context.env_ids,
            control_dt=context.require_control_dt(),
        )

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=timed,
            expected_effects=(
                StateDelta(held_object_updates={task_state_key: held_object})
                if (options.report_retained_attachment or target.object_id is not None)
                and success.any()
                else None
            ),
            diagnostics=(
                PlannerDiagnostics(
                    backend=self.planning_services.planner_name,
                    metadata={
                        "world_yaw_offsets_rad": selected_yaws.tolist(),
                        "yaw_plan_attempts": attempts,
                    },
                )
                if target.world_yaw_free
                else None
            ),
            segment_lengths={"transport": timed.waypoint_count},
        )

    def _validate_named_hold(
        self,
        request: ResolvedActionRequest[HeldObjectPoseGoal, MoveHeldObjectOptions],
        context: PlanningContext,
    ) -> None:
        """Validate named input ownership before native or nominal motion planning."""
        expected = request.goal.object_id
        if expected is None:
            return
        key = require_shared_task_state_key(
            request.binding.endpoint("primary", "motion"),
            request.binding.endpoint("primary", "grasp"),
            participant="MoveHeldObject primary participant",
        )
        held = context.get_held_object(key)
        if held is None or held.semantics.entity_id != expected:
            raise ValueError(
                f"MoveHeldObject requires object {expected!r} held under resource {key!r}."
            )


def _merge_yaw_paths(
    previous: PlanResult, candidate: PlanResult, rows: torch.Tensor
) -> PlanResult:
    """Retain accepted rows while selecting newly feasible, independently timed paths."""
    assert candidate.positions is not None and candidate.dt is not None
    if previous.positions is None:
        return candidate
    assert previous.dt is not None
    count = max(previous.positions.shape[1], candidate.positions.shape[1])

    def pad_positions(value: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [value, value[:, -1:].expand(-1, count - value.shape[1], -1)], dim=1
        )

    # Zero-time terminal padding preserves each row's duration. The action
    # recomputes derivatives on the control grid in to_full_robot_trajectory.
    return PlanResult(
        success=torch.where(rows, candidate.success, previous.success),
        waypoint_indices=(
            torch.where(
                rows[:, None], candidate.waypoint_indices, previous.waypoint_indices
            )
            if candidate.waypoint_indices is not None
            and previous.waypoint_indices is not None
            else None
        ),
        positions=torch.where(
            rows[:, None, None],
            pad_positions(candidate.positions),
            pad_positions(previous.positions),
        ),
        dt=torch.where(
            rows[:, None],
            torch.nn.functional.pad(candidate.dt, (0, count - candidate.dt.shape[1])),
            torch.nn.functional.pad(previous.dt, (0, count - previous.dt.shape[1])),
        ),
    )


__all__ = [
    "HeldObjectPoseGoal",
    "MoveHeldObject",
    "MoveHeldObjectOptions",
    "MoveHeldObjectPlanningRecovery",
]
