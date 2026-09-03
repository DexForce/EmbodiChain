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

"""Pick-up-and-handover atomic action implementation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from embodichain.utils import logger
from embodichain.utils.math import get_relative_rotation, pose_inv

from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.bindings import (
    EndpointBinding,
    JointPositionTarget,
)
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    OPEN_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction
from embodichain.lab.sim.atomic_actions.effects import StateDelta
from embodichain.lab.sim.atomic_actions.goals import (
    ObjectActionGoal,
    PoseGoalValue,
    _resolve_object_pose,
    resolve_pose_goal,
    validate_pose_goal,
)
from embodichain.lab.sim.atomic_actions.invocation import (
    ActionOptions,
    ResolvedActionRequest,
)
from embodichain.lab.sim.atomic_actions.plans import (
    ActionPlan,
    TimedTrajectory,
    normalize_success_mask,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    assemble_full_robot_trajectory,
    plan_named_arm_trajectory,
    repeat_qpos,
    require_shared_task_state_key,
    resolve_batched_pose,
)
from embodichain.lab.sim.atomic_actions.requirements import (
    CARTESIAN_POSE_CAPABILITY,
    DisjointResourceSlots,
    FORWARD_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    interpolate_hand_qpos,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True, eq=False)
class HandOverGoal(ObjectActionGoal):
    """Object to pick and hand over, plus its final object pose."""

    target_pose: PoseGoalValue
    """Final object pose after the receiving arm lowers and releases it."""

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(self.target_pose, "target_pose", allow_waypoints=False)


@dataclass(frozen=True, slots=True, eq=False)
class HandOverOptions(ActionOptions):
    """Per-invocation pick-up, handover, and placement behavior."""

    pre_grasp_distance: float = 0.10
    """Distance from each grasp pose to its approach pose, in metres."""

    lift_height: float = 0.10
    """World-Z distance used to lift the object after the first grasp."""

    hand_interp_steps: int = 10
    """Waypoints used by every gripper open/close interpolation."""

    hold_steps: int = 4
    """Closed-hand waypoints used to settle a receiving grasp."""

    retreat_steps: int = 24
    """Waypoints used while the source hand retreats after release."""

    retreat_distance: float = 0.10
    """Planar clearance travelled away from the receiving grasp before lift."""

    receive_pick_object_part: Literal["center", "top", "bottom"] = "bottom"
    """Object end selected by the receiving gripper for an existing hold."""

    release_at_target: bool = True
    """Whether the receiving hand places and releases after the transfer.

    When false, execution ends after the source hand opens and the receiving
    resource remains the verified holder. This supports a semantic handover
    followed by a later Place call without moving that workflow into Task Engine.
    """

    arm_selection: Literal["nearest", "bound"] = "nearest"
    """How the transfer participant is selected.

    ``"nearest"`` preserves the low-level Atomic Action default for direct
    callers.  Semantic Task Program bindings should select ``"bound"`` so the
    explicit ``source`` and ``destination`` resource slots are authoritative.
    """

    def __post_init__(self) -> None:
        for name in ("pre_grasp_distance", "lift_height", "retreat_distance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        for name in ("hold_steps", "retreat_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.retreat_steps < 2:
            raise ValueError("retreat_steps must be at least 2.")
        if self.receive_pick_object_part not in ("center", "top", "bottom"):
            raise ValueError(
                "receive_pick_object_part must be exactly 'center', 'top', or "
                "'bottom'."
            )
        if type(self.release_at_target) is not bool:
            raise TypeError("release_at_target must be a bool.")
        if self.arm_selection not in ("nearest", "bound"):
            raise ValueError("arm_selection must be exactly 'nearest' or 'bound'.")


@dataclass(frozen=True, slots=True, eq=False)
class _Participant:
    """One candidate arm, its hand, and resolved semantic hand commands."""

    task_state_key: str
    arm: JointPositionTarget
    hand: JointPositionTarget
    hand_open_qpos: torch.Tensor
    hand_grasp_qpos: torch.Tensor


@dataclass(frozen=True, slots=True, eq=False)
class _HandOverResources:
    """The two invocation-bound candidate participants."""

    first: _Participant
    second: _Participant


@dataclass(frozen=True, slots=True, eq=False)
class _DirectionalPlan:
    """One fixed handover-arm/receiving-arm assignment."""

    success: torch.Tensor
    trajectory: torch.Tensor
    segment_lengths: dict[str, int]
    handover_object_to_eef: torch.Tensor
    handover_grasp_xpos: torch.Tensor
    receive_object_to_eef: torch.Tensor
    receive_grasp_xpos: torch.Tensor


class HandOver(AtomicAction[HandOverGoal, HandOverOptions]):
    """Pick an object with the nearer arm and transfer it to the other arm.

    For each environment, the action chooses the arm whose root link is closer
    to the observed object pose. It samples at most 1000 mesh-surface points and
    applies SVD in the current object pose to find ``obj_longest_axis``. When
    that axis is closer to world Z than to the horizontal plane, both grasp
    approaches point toward the object horizontally and tilt downward by 45
    degrees. Otherwise both approaches are world-Z downward.

    The first arm grasps the projected end of ``obj_longest_axis`` nearest its
    current TCP; the receiving arm grasps the opposite end at the predicted
    middle object pose. This keeps the two hands from selecting the same object
    region regardless of whether a long object is standing or lying down.

    After each grasp waypoint, subsequent EEF waypoints preserve that grasp
    rotation and change translation only. With ``release_at_target=True``, the
    receiving arm additionally moves horizontally at handover height, lowers to
    the final target pose, and releases. Transfer-only mode stops with the
    receiving arm recorded as the verified holder for a later Semantic Call.
    """

    skill_id: ClassVar[str] = "hand_over"
    GoalType: ClassVar[type] = HandOverGoal
    OptionsType: ClassVar[type] = HandOverOptions
    open_loop: ClassVar[bool] = True
    _SURFACE_POINT_COUNT: ClassVar[int] = 1000
    _VERTICAL_MODE_MIN_ABS_Z: ClassVar[float] = math.sqrt(0.5)
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            make_manipulation_slot(
                "source",
                motion_capabilities=frozenset(
                    {
                        CARTESIAN_POSE_CAPABILITY,
                        FORWARD_KINEMATICS_CAPABILITY,
                    }
                ),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
            make_manipulation_slot(
                "destination",
                motion_capabilities=frozenset(
                    {
                        CARTESIAN_POSE_CAPABILITY,
                        FORWARD_KINEMATICS_CAPABILITY,
                    }
                ),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            ),
        ),
        constraints=(DisjointResourceSlots(("source", "destination")),),
    )

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[HandOverGoal, HandOverOptions],
    ) -> tuple[str, ...]:
        """Track both the initial object and any late-bound final target."""
        dependencies = set(super()._scene_dependencies(request))
        entity_id = request.goal.semantics.entity_id
        if entity_id is not None:
            dependencies.add(entity_id)
        return tuple(sorted(dependencies))

    def _resolve_resources(
        self,
        request: ResolvedActionRequest[HandOverGoal, HandOverOptions],
    ) -> _HandOverResources:
        """Resolve the two bound candidate arm/hand pairs."""
        binding = request.binding
        first_motion = binding.endpoint("source", "motion")
        second_motion = binding.endpoint("destination", "motion")
        first_grasp = binding.endpoint("source", "grasp")
        second_grasp = binding.endpoint("destination", "grasp")
        first_arm = first_motion.require_target(JointPositionTarget)
        second_arm = second_motion.require_target(JointPositionTarget)
        first_hand = first_grasp.require_target(JointPositionTarget)
        second_hand = second_grasp.require_target(JointPositionTarget)
        if first_arm.control_part == second_arm.control_part:
            raise ValueError("HandOver requires two different manipulator parts.")
        if first_hand.control_part == second_hand.control_part:
            raise ValueError("HandOver requires two different end-effector parts.")

        def participant(
            arm: JointPositionTarget,
            hand: JointPositionTarget,
            motion_endpoint: EndpointBinding,
            grasp_endpoint: EndpointBinding,
            participant_name: str,
        ) -> _Participant:
            return _Participant(
                task_state_key=require_shared_task_state_key(
                    motion_endpoint,
                    grasp_endpoint,
                    participant=participant_name,
                ),
                arm=arm,
                hand=hand,
                hand_open_qpos=grasp_endpoint.joint_positions(
                    OPEN_COMMAND,
                    num_envs=self.num_envs,
                    device=self.device,
                    dtype=torch.float32,
                ),
                hand_grasp_qpos=grasp_endpoint.joint_positions(
                    GRASP_COMMAND,
                    num_envs=self.num_envs,
                    device=self.device,
                    dtype=torch.float32,
                ),
            )

        return _HandOverResources(
            first=participant(
                first_arm,
                first_hand,
                first_motion,
                first_grasp,
                "HandOver source participant",
            ),
            second=participant(
                second_arm,
                second_hand,
                second_motion,
                second_grasp,
                "HandOver destination participant",
            ),
        )

    def _plan(
        self,
        request: ResolvedActionRequest[HandOverGoal, HandOverOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan pickup and transfer, with optional placement and release."""
        goal = self.require_goal(request)
        options = request.skill_options
        resources = self._resolve_resources(request)
        if (
            request.motion_policy.strategy == "motion_gen"
            and self.motion_generator.planner.cfg.planner_type == "curobo"
        ):
            raise ValueError(
                "Coordinated dual-arm planning is not supported by the cuRobo backend."
            )
        if not isinstance(goal.semantics.affordance, AntipodalAffordance):
            raise ValueError("HandOver requires an AntipodalAffordance.")

        # A semantic handover is also the continuation point after an explicit
        # Pick call.  In that case the source attachment is already verified by
        # TaskState and the action must transfer that attachment rather than
        # silently attempting a second pickup.  Keep the legacy unified route
        # below for direct low-level callers that start with two free arms.
        source_held = context.task.get_held_object(resources.first.task_state_key)
        source_mask = context.task.held_object_mask(resources.first.task_state_key)
        if source_held is not None and source_mask.any():
            if self._same_object(goal.semantics, source_held):
                if options.release_at_target:
                    raise ValueError(
                        "HandOver cannot place an already-held object in the same "
                        "invocation; use a following Place call."
                    )
                return self._plan_existing_hold(
                    request, context, resources, source_held
                )

        object_pose = _resolve_object_pose(
            goal.semantics,
            context,
            name="handover_object_pose",
        )
        obj_longest_axis = goal.semantics.affordance.get_object_longest_axis(
            object_pose,
            max_points=self._SURFACE_POINT_COUNT,
        )
        final_object_pose = resolve_batched_pose(
            resolve_pose_goal(
                goal.target_pose,
                context,
                name="handover_target_pose",
            ),
            num_envs=self.num_envs,
            device=self.device,
            name="handover_target_pose",
        )
        first_root_pose = self._root_link_pose(resources.first.arm, context.env_ids)
        second_root_pose = self._root_link_pose(resources.second.arm, context.env_ids)
        if options.arm_selection == "bound":
            # Semantic bindings are authoritative: ``source`` acquires and
            # ``destination`` receives.  Do not silently invert an explicit
            # request merely because the object starts nearer the other arm.
            first_is_handover = torch.ones(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        else:
            first_distance = torch.linalg.vector_norm(
                object_pose[:, :3, 3] - first_root_pose[:, :3, 3], dim=1
            )
            second_distance = torch.linalg.vector_norm(
                object_pose[:, :3, 3] - second_root_pose[:, :3, 3], dim=1
            )
            first_is_handover = first_distance <= second_distance

        # This unified action starts before pickup. Rows where either bound arm
        # already holds an object are therefore ineligible and remain at the
        # observed robot state.
        eligible = ~context.task.held_object_mask(
            resources.first.task_state_key
        ) & ~context.task.held_object_mask(resources.second.task_state_key)
        self._report_waypoint_failure(
            context,
            "candidate_arms_unoccupied",
            ~eligible,
            "one or both candidate arms already hold an object",
        )
        if not eligible.any():
            return self.failed_plan(
                request,
                context,
                message="HandOver requires both candidate arms to start unoccupied.",
            )

        segment_lengths = self._compute_segment_lengths(
            request.motion_policy.sample_count,
            options,
        )
        if first_is_handover.all():
            selected = self._plan_direction(
                context,
                request,
                goal.semantics.affordance,
                object_pose,
                obj_longest_axis,
                final_object_pose,
                first_root_pose,
                second_root_pose,
                resources.first,
                resources.second,
                segment_lengths,
                eligible,
            )
            success = selected.success & eligible
            full = selected.trajectory
            first_object_to_eef = selected.handover_object_to_eef
            first_grasp_xpos = selected.handover_grasp_xpos
            second_object_to_eef = selected.receive_object_to_eef
            second_grasp_xpos = selected.receive_grasp_xpos
        elif (~first_is_handover).all():
            selected = self._plan_direction(
                context,
                request,
                goal.semantics.affordance,
                object_pose,
                obj_longest_axis,
                final_object_pose,
                second_root_pose,
                first_root_pose,
                resources.second,
                resources.first,
                segment_lengths,
                eligible,
            )
            success = selected.success & eligible
            full = selected.trajectory
            first_object_to_eef = selected.receive_object_to_eef
            first_grasp_xpos = selected.receive_grasp_xpos
            second_object_to_eef = selected.handover_object_to_eef
            second_grasp_xpos = selected.handover_grasp_xpos
        else:
            first_to_second = self._plan_direction(
                context,
                request,
                goal.semantics.affordance,
                object_pose,
                obj_longest_axis,
                final_object_pose,
                first_root_pose,
                second_root_pose,
                resources.first,
                resources.second,
                segment_lengths,
                first_is_handover & eligible,
            )
            second_to_first = self._plan_direction(
                context,
                request,
                goal.semantics.affordance,
                object_pose,
                obj_longest_axis,
                final_object_pose,
                second_root_pose,
                first_root_pose,
                resources.second,
                resources.first,
                segment_lengths,
                ~first_is_handover & eligible,
            )
            if first_to_second.trajectory.shape != second_to_first.trajectory.shape:
                raise ValueError(
                    "Both HandOver arm assignments must produce matching trajectory shapes."
                )
            success = (
                torch.where(
                    first_is_handover,
                    first_to_second.success,
                    second_to_first.success,
                )
                & eligible
            )
            full = torch.where(
                first_is_handover[:, None, None],
                first_to_second.trajectory,
                second_to_first.trajectory,
            )
            pose_selector = first_is_handover[:, None, None]
            first_object_to_eef = torch.where(
                pose_selector,
                first_to_second.handover_object_to_eef,
                second_to_first.receive_object_to_eef,
            )
            first_grasp_xpos = torch.where(
                pose_selector,
                first_to_second.handover_grasp_xpos,
                second_to_first.receive_grasp_xpos,
            )
            second_object_to_eef = torch.where(
                pose_selector,
                first_to_second.receive_object_to_eef,
                second_to_first.handover_object_to_eef,
            )
            second_grasp_xpos = torch.where(
                pose_selector,
                first_to_second.receive_grasp_xpos,
                second_to_first.handover_grasp_xpos,
            )

        first_candidate = HeldObjectState(
            semantics=goal.semantics,
            object_to_eef=first_object_to_eef,
            grasp_xpos=first_grasp_xpos,
            env_mask=(None if options.release_at_target else ~first_is_handover),
        )
        second_candidate = HeldObjectState(
            semantics=goal.semantics,
            object_to_eef=second_object_to_eef,
            grasp_xpos=second_grasp_xpos,
            env_mask=(None if options.release_at_target else first_is_handover),
        )
        terminal_updates = (
            {
                resources.first.task_state_key: None,
                resources.second.task_state_key: None,
            }
            if options.release_at_target
            else {
                resources.first.task_state_key: first_candidate,
                resources.second.task_state_key: second_candidate,
            }
        )

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(
                held_object_updates=terminal_updates,
            ),
            effect_candidates=StateDelta(
                held_object_updates={
                    resources.first.task_state_key: first_candidate,
                    resources.second.task_state_key: second_candidate,
                },
            ),
            segment_lengths=segment_lengths,
            # The object may move from contact as soon as the pickup gripper
            # starts closing.  Keep dynamic-target monitoring active through
            # the approach, but do not classify expected pickup motion as an
            # external scene revision.
            scene_dependency_monitor_until=(
                {}
                if goal.semantics.entity_id is None
                else {goal.semantics.entity_id: segment_lengths["pickup_approach"]}
            ),
        )

    def _plan_existing_hold(
        self,
        request: ResolvedActionRequest[HandOverGoal, HandOverOptions],
        context: PlanningContext,
        resources: _HandOverResources,
        held: HeldObjectState,
    ) -> ActionPlan:
        """Transfer a verified source attachment to the destination hand.

        This is the canonical continuation used by ``Pick -> HandOver``.  It
        deliberately lives in the Atomic Action so Task Engine never owns
        grasp poses, hand timing, or a second physical execution loop.
        """
        goal = self.require_goal(request)
        options = request.skill_options
        source_mask = context.task.exclusive_held_object_mask(
            resources.first.task_state_key
        )
        destination_mask = context.task.held_object_mask(
            resources.second.task_state_key
        )
        eligible = source_mask & ~destination_mask
        self._report_waypoint_failure(
            context,
            "existing_source_attachment",
            ~source_mask,
            "source participant does not own the requested object",
        )
        self._report_waypoint_failure(
            context,
            "destination_unoccupied",
            destination_mask,
            "destination participant already holds an object",
        )
        if not eligible.any():
            return self.failed_plan(
                request,
                context,
                message=(
                    "HandOver requires an exclusive source attachment and an "
                    "unoccupied destination."
                ),
            )

        source_start_qpos = context.last_qpos[:, list(resources.first.arm.joint_ids)]
        destination_start_qpos = context.last_qpos[
            :, list(resources.second.arm.joint_ids)
        ]
        source_object_to_eef = held.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if source_object_to_eef.dim() == 2:
            source_object_to_eef = source_object_to_eef.unsqueeze(0).expand(
                self.num_envs, -1, -1
            )
        source_eef = self.robot.compute_fk(
            qpos=source_start_qpos,
            name=resources.first.arm.control_part,
            to_matrix=True,
        )
        current_object_pose = torch.bmm(source_eef, pose_inv(source_object_to_eef))

        # The provider target supplies the exchange x/y location and a safe
        # *absolute* exchange height.  Do not add the lift twice: a configured
        # provider may already choose a staging height above the table.  The
        # measured attachment still wins when it is higher than that hint.
        # Keeping this as an object-frame target (rather than an EEF target)
        # is important: the Task Program owns only semantic intent while the
        # Atomic Action owns the physical transfer geometry.
        exchange_pose = current_object_pose.clone()
        target_pose = resolve_batched_pose(
            resolve_pose_goal(
                goal.target_pose,
                context,
                name="handover_exchange_pose",
            ),
            num_envs=self.num_envs,
            device=self.device,
            name="handover_exchange_pose",
        )
        exchange_pose[:, :2, 3] = target_pose[:, :2, 3]
        exchange_pose[:, 2, 3] = torch.maximum(
            current_object_pose[:, 2, 3],
            target_pose[:, 2, 3],
        )
        exchange_pose[:, :3, :3] = current_object_pose[:, :3, :3]

        source_exchange_eef = torch.bmm(exchange_pose, source_object_to_eef)
        destination_eef = self.robot.compute_fk(
            qpos=destination_start_qpos,
            name=resources.second.arm.control_part,
            to_matrix=True,
        )
        source_root = self._root_link_pose(resources.first.arm, context.env_ids)
        destination_root = self._root_link_pose(
            resources.second.arm,
            context.env_ids,
        )
        # Approach diagonally from the receiver's side of the embodiment.  A
        # top-down receiver places two bulky parallel grippers in the same
        # vertical envelope and can squeeze the object out while the source
        # opens.  Root-to-root direction is stable, robot-generic role geometry
        # and reproduces the successful inward approach for either transfer
        # direction without embedding left/right names.
        approach_direction, approach_direction_valid = (
            self._downward_diagonal_approach_direction(
                destination_root[:, :3, 3],
                source_root[:, :3, 3],
            )
        )

        affordance = goal.semantics.affordance
        assert isinstance(affordance, AntipodalAffordance)
        longest_axis = affordance.get_object_longest_axis(
            exchange_pose,
            max_points=self._SURFACE_POINT_COUNT,
        )
        if options.receive_pick_object_part != "center":
            local_axis = exchange_pose.new_tensor([0.0, 0.0, 1.0])
            receive_axis = torch.matmul(exchange_pose[:, :3, :3], local_axis)
            receive_positive = torch.full(
                (self.num_envs,),
                options.receive_pick_object_part == "top",
                dtype=torch.bool,
                device=self.device,
            )
        else:
            receive_axis = longest_axis
            receive_positive = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        destination_grasp, grasp_success = self._resolve_grasp(
            affordance,
            exchange_pose,
            approach_direction,
            resources.second.hand.target_id,
            obj_longest_axis=receive_axis,
            is_positive_part=receive_positive,
        )
        destination_pre_grasp = translate_pose_world(
            destination_grasp,
            -destination_grasp[:, :3, 2] * options.pre_grasp_distance,
        )
        destination_object_to_eef = torch.bmm(
            pose_inv(exchange_pose), destination_grasp
        )
        source_retreat_waypoints = self._source_retreat_waypoints(
            source_exchange_eef,
            destination_grasp,
            source_fallback=source_eef,
            destination_fallback=destination_eef,
            planar_distance=options.retreat_distance,
            lift_height=options.lift_height,
        )

        lengths = self._compute_existing_hold_segment_lengths(
            request.motion_policy.sample_count,
            options,
        )
        success = (
            normalize_success_mask(
                grasp_success,
                num_envs=self.num_envs,
                device=self.device,
                name="HandOver receiving-grasp success",
            )
            & approach_direction_valid
            & eligible
        )
        self._report_waypoint_failure(
            context,
            "receive_approach_direction",
            eligible & ~approach_direction_valid,
            "source and destination roots have no horizontal separation",
        )
        self._report_waypoint_failure(
            context,
            "receive_grasp",
            eligible & ~success,
            "no finite receiving grasp was found",
        )

        phase_success, source_transfer = plan_named_arm_trajectory(
            self.motion_generator,
            resources.first.arm.control_part,
            source_start_qpos,
            source_exchange_eef.unsqueeze(1),
            lengths["transfer"],
            request.motion_policy,
            context.control_dt,
        )
        success &= normalize_success_mask(
            phase_success,
            num_envs=self.num_envs,
            device=self.device,
            name="HandOver existing-hold source transfer success",
        )
        source_hold_qpos = source_transfer[:, -1]
        phase_success, destination_approach = plan_named_arm_trajectory(
            self.motion_generator,
            resources.second.arm.control_part,
            destination_start_qpos,
            torch.stack((destination_pre_grasp, destination_grasp), dim=1),
            lengths["approach"],
            request.motion_policy,
            context.control_dt,
        )
        success &= normalize_success_mask(
            phase_success,
            num_envs=self.num_envs,
            device=self.device,
            name="HandOver existing-hold destination approach success",
        )
        destination_hold_qpos = destination_approach[:, -1]
        phase_success, source_retreat = plan_named_arm_trajectory(
            self.motion_generator,
            resources.first.arm.control_part,
            source_hold_qpos,
            source_retreat_waypoints,
            lengths["retreat"],
            request.motion_policy,
            context.control_dt,
        )
        success &= normalize_success_mask(
            phase_success,
            num_envs=self.num_envs,
            device=self.device,
            name="HandOver existing-hold source retreat success",
        )

        segment_values: list[tuple[str, torch.Tensor]] = [
            (
                "transfer",
                self._assemble_segment(
                    context,
                    source_transfer,
                    repeat_qpos(destination_start_qpos, lengths["transfer"]),
                    repeat_qpos(resources.first.hand_grasp_qpos, lengths["transfer"]),
                    repeat_qpos(resources.second.hand_open_qpos, lengths["transfer"]),
                    resources.first,
                    resources.second,
                ),
            ),
            (
                "receive_approach",
                self._assemble_segment(
                    context,
                    repeat_qpos(source_hold_qpos, lengths["approach"]),
                    destination_approach,
                    repeat_qpos(resources.first.hand_grasp_qpos, lengths["approach"]),
                    repeat_qpos(resources.second.hand_open_qpos, lengths["approach"]),
                    resources.first,
                    resources.second,
                ),
            ),
            (
                "receive_close",
                self._assemble_segment(
                    context,
                    repeat_qpos(source_hold_qpos, lengths["close"]),
                    repeat_qpos(destination_hold_qpos, lengths["close"]),
                    repeat_qpos(resources.first.hand_grasp_qpos, lengths["close"]),
                    interpolate_hand_qpos(
                        resources.second.hand_open_qpos,
                        resources.second.hand_grasp_qpos,
                        n_waypoints=lengths["close"],
                    ),
                    resources.first,
                    resources.second,
                ),
            ),
        ]
        if lengths["hold"]:
            segment_values.append(
                (
                    "receive_hold",
                    self._assemble_segment(
                        context,
                        repeat_qpos(source_hold_qpos, lengths["hold"]),
                        repeat_qpos(destination_hold_qpos, lengths["hold"]),
                        repeat_qpos(resources.first.hand_grasp_qpos, lengths["hold"]),
                        repeat_qpos(resources.second.hand_grasp_qpos, lengths["hold"]),
                        resources.first,
                        resources.second,
                    ),
                )
            )
        segment_values.extend(
            (
                (
                    "handover_release",
                    self._assemble_segment(
                        context,
                        repeat_qpos(source_hold_qpos, lengths["release"]),
                        repeat_qpos(destination_hold_qpos, lengths["release"]),
                        interpolate_hand_qpos(
                            resources.first.hand_grasp_qpos,
                            resources.first.hand_open_qpos,
                            n_waypoints=lengths["release"],
                        ),
                        repeat_qpos(
                            resources.second.hand_grasp_qpos, lengths["release"]
                        ),
                        resources.first,
                        resources.second,
                    ),
                ),
                (
                    "source_retreat",
                    self._assemble_segment(
                        context,
                        source_retreat,
                        repeat_qpos(destination_hold_qpos, lengths["retreat"]),
                        repeat_qpos(resources.first.hand_open_qpos, lengths["retreat"]),
                        repeat_qpos(
                            resources.second.hand_grasp_qpos, lengths["retreat"]
                        ),
                        resources.first,
                        resources.second,
                    ),
                ),
            )
        )
        trajectory = torch.cat([value for _, value in segment_values], dim=1)
        received = HeldObjectState(
            semantics=held.semantics,
            object_to_eef=destination_object_to_eef,
            grasp_xpos=destination_grasp,
            env_mask=eligible,
        )
        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=TimedTrajectory.from_uniform_step(
                trajectory,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(
                held_object_updates={
                    resources.first.task_state_key: None,
                    resources.second.task_state_key: received,
                }
            ),
            effect_candidates=StateDelta(
                held_object_updates={
                    resources.first.task_state_key: held,
                    resources.second.task_state_key: received,
                }
            ),
            segment_lengths={name: value.shape[1] for name, value in segment_values},
            scene_dependency_monitor_until={
                entity_id: 0 for entity_id in self._scene_dependencies(request)
            },
        )

    @staticmethod
    def _source_retreat_waypoints(
        source_exchange_eef: torch.Tensor,
        destination_grasp: torch.Tensor,
        *,
        source_fallback: torch.Tensor,
        destination_fallback: torch.Tensor,
        planar_distance: float,
        lift_height: float,
    ) -> torch.Tensor:
        """Clear the receiving grasp laterally before lifting the source TCP.

        A single upward IK target can produce a joint-space interpolation that
        first bows inward between the two grippers.  Once the source fingers
        open, that transient motion can knock the object out of an otherwise
        valid receiving grasp.  Resolve the horizontal separation axis from
        the exchange geometry and constrain the planner with intermediate
        Cartesian waypoints before adding vertical clearance.
        """
        if source_exchange_eef.shape != destination_grasp.shape:
            raise ValueError(
                "Source exchange and destination grasp poses must have matching "
                "shapes."
            )
        if source_fallback.shape != source_exchange_eef.shape:
            raise ValueError("Source fallback poses must match exchange poses.")
        if destination_fallback.shape != destination_grasp.shape:
            raise ValueError("Destination fallback poses must match grasp poses.")

        direction = source_exchange_eef[:, :3, 3] - destination_grasp[:, :3, 3]
        direction[:, 2] = 0.0
        fallback = source_fallback[:, :3, 3] - destination_fallback[:, :3, 3]
        fallback[:, 2] = 0.0
        direction_norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True)
        fallback_norm = torch.linalg.vector_norm(fallback, dim=1, keepdim=True)
        direction = torch.where(
            direction_norm > 1.0e-6,
            direction,
            torch.where(
                fallback_norm > 1.0e-6,
                fallback,
                direction.new_tensor([1.0, 0.0, 0.0]).expand_as(direction),
            ),
        )
        direction = direction / torch.linalg.vector_norm(
            direction, dim=1, keepdim=True
        ).clamp_min(1.0e-6)

        planar_fractions = source_exchange_eef.new_tensor([1.0 / 3.0, 2.0 / 3.0, 1.0])
        planar = source_exchange_eef[:, None].repeat(1, 3, 1, 1)
        planar[:, :, :3, 3] += (
            direction[:, None] * planar_fractions[None, :, None] * planar_distance
        )
        lifted = planar[:, -1:].repeat(1, 2, 1, 1)
        lifted[:, :, 2, 3] += lifted.new_tensor([0.5, 1.0])[None] * lift_height
        local_clearance = torch.linalg.vector_norm(
            lifted[:, -1, :2, 3] - destination_grasp[:, :2, 3],
            dim=1,
        )
        fallback_clearance = torch.linalg.vector_norm(
            source_fallback[:, :2, 3] - destination_fallback[:, :2, 3],
            dim=1,
        )
        # Returning to the pre-transfer source pose is both already reachable
        # and a better shared-workspace exit when the two starting endpoints
        # were farther apart than a short local retreat.  Keep the local
        # lifted endpoint otherwise (for example when transfer began in the
        # exchange region already).
        use_source_workspace = fallback_clearance > local_clearance
        lifted[:, -1] = torch.where(
            use_source_workspace[:, None, None],
            source_fallback,
            lifted[:, -1],
        )
        return torch.cat((planar, lifted), dim=1)

    @staticmethod
    def _same_object(
        requested: object,
        held: HeldObjectState,
    ) -> bool:
        """Return whether semantic object identity matches a held relation."""
        requested_entity_id = getattr(requested, "entity_id", None)
        held_entity_id = held.semantics.entity_id
        if requested_entity_id is not None and held_entity_id is not None:
            return requested_entity_id == held_entity_id
        requested_entity = getattr(requested, "entity", None)
        held_entity = held.semantics.entity
        if requested_entity is not None and held_entity is not None:
            return requested_entity is held_entity
        requested_label = getattr(requested, "label", None)
        return bool(requested_label) and requested_label == held.semantics.label

    @staticmethod
    def _compute_existing_hold_segment_lengths(
        sample_count: int,
        options: HandOverOptions,
    ) -> dict[str, int]:
        """Split one existing-hold transfer into motion and hand phases."""
        close = max(2, options.hand_interp_steps)
        release = max(2, options.hand_interp_steps)
        hold = options.hold_steps
        retreat = max(2, options.retreat_steps)
        reserved = close + release + hold + retreat
        remaining = sample_count - reserved
        if remaining < 4:
            raise ValueError(
                "Not enough HandOver waypoints for an existing held-object "
                "transfer; increase sample_count or reduce hand phases."
            )
        transfer = max(2, remaining // 2)
        approach = remaining - transfer
        if approach < 2:
            raise ValueError(
                "Not enough HandOver waypoints for the receiving approach."
            )
        return {
            "transfer": transfer,
            "approach": approach,
            "close": close,
            "hold": hold,
            "release": release,
            "retreat": retreat,
        }

    def _find_symmetric_nearest_xpos(
        self, target_xpos: torch.Tensor, reference_xpos: torch.Tensor
    ) -> torch.Tensor:
        """Find the nearest symmetric pose to the reference pose."""
        symmetric_xpos = target_xpos.clone()
        symmetric_xpos[:, :3, 0] = -symmetric_xpos[:, :3, 0]
        symmetric_xpos[:, :3, 1] = -symmetric_xpos[:, :3, 1]
        angle_a = get_relative_rotation(
            reference_xpos[:, :3, :3], target_xpos[:, :3, :3]
        )
        angle_b = get_relative_rotation(
            reference_xpos[:, :3, :3], symmetric_xpos[:, :3, :3]
        )
        choose_target = (angle_a < angle_b)[..., None, None]
        target_xpos = torch.where(choose_target, target_xpos, symmetric_xpos)
        return target_xpos

    def _plan_direction(
        self,
        context: PlanningContext,
        request: ResolvedActionRequest[HandOverGoal, HandOverOptions],
        affordance: AntipodalAffordance,
        object_pose: torch.Tensor,
        obj_longest_axis: torch.Tensor,
        final_object_pose: torch.Tensor,
        handover_root_pose: torch.Tensor,
        receive_root_pose: torch.Tensor,
        handover: _Participant,
        receive: _Participant,
        segment_lengths: dict[str, int],
        active_mask: torch.Tensor,
    ) -> _DirectionalPlan:
        """Plan one concrete handover-arm to receiving-arm assignment.

        The pickup grasp is sampled on the observed object pose, whereas the
        receiving grasp is sampled on the predicted pose after lift and middle
        transfer. Vertical-mode approaches tilt down by 45 degrees; horizontal
        mode approaches vertically downward. The two grasps use opposite ends
        of the SVD-derived longest object axis.
        """
        options = request.skill_options
        state = context
        start_qpos = state.last_qpos.to(device=self.device, dtype=torch.float32)
        handover_start_qpos = start_qpos[:, list(handover.arm.joint_ids)]
        handover_start_eef = self.robot.compute_fk(
            qpos=handover_start_qpos,
            name=handover.arm.control_part,
            to_matrix=True,
        )
        receive_start_qpos = start_qpos[:, list(receive.arm.joint_ids)]
        receive_start_eef = self.robot.compute_fk(
            qpos=receive_start_qpos,
            name=receive.arm.control_part,
            to_matrix=True,
        )

        vertical_mode = (
            torch.abs(obj_longest_axis[:, 2]) >= self._VERTICAL_MODE_MIN_ABS_Z
        )
        vertical_down = torch.tensor(
            [0.0, 0.0, -1.0],
            dtype=torch.float32,
            device=self.device,
        ).expand(self.num_envs, -1)

        # In vertical mode, point from the pickup TCP toward the observed
        # object horizontally and tilt down by 45 degrees. In horizontal mode,
        # approach vertically downward. Only vertical rows require nonzero
        # horizontal TCP-to-object separation.
        handover_diagonal, handover_diagonal_valid = (
            self._downward_diagonal_approach_direction(
                handover_start_eef[:, :3, 3], object_pose[:, :3, 3]
            )
        )
        handover_direction = torch.where(
            vertical_mode[:, None], handover_diagonal, vertical_down
        )
        handover_direction_valid = ~vertical_mode | handover_diagonal_valid

        # SVD axes have arbitrary sign. Selecting the sign whose projected end
        # points toward the pickup TCP makes the physical choice sign-invariant;
        # the receiving arm always takes the opposite projected end.
        handover_is_positive_part = (
            torch.sum(
                (handover_start_eef[:, :3, 3] - object_pose[:, :3, 3])
                * obj_longest_axis,
                dim=1,
            )
            >= 0.0
        )
        handover_grasp, handover_grasp_success = self._resolve_grasp(
            affordance,
            object_pose,
            handover_direction,
            handover.hand.target_id,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=handover_is_positive_part,
        )
        handover_grasp = self._find_symmetric_nearest_xpos(
            handover_grasp, handover_start_eef
        )
        handover_pre_grasp = translate_pose_world(
            handover_grasp,
            -handover_direction * options.pre_grasp_distance,
        )
        handover_object_to_eef = torch.bmm(pose_inv(object_pose), handover_grasp)

        lifted_object_pose = object_pose.clone()
        lifted_object_pose[:, 2, 3] += options.lift_height
        middle_object_pose = self._middle_object_pose(
            lifted_object_pose,
            handover_root_pose,
            receive_root_pose,
        )
        handover_lift_eef = torch.bmm(
            lifted_object_pose,
            handover_object_to_eef,
        )
        handover_middle_eef = torch.bmm(
            middle_object_pose,
            handover_object_to_eef,
        )
        # From the grasp waypoint through lift and transfer, only translation
        # may change. Pinning the rotations explicitly also avoids numerical
        # drift from the object/EFF transform multiplications.
        handover_lift_eef[:, :3, :3] = handover_grasp[:, :3, :3]
        handover_middle_eef[:, :3, :3] = handover_grasp[:, :3, :3]

        # Apply the same mode to receiving. The object rotation is unchanged by
        # lift and middle transfer, so its world-space longest axis is unchanged.
        receive_diagonal, receive_diagonal_valid = (
            self._downward_diagonal_approach_direction(
                receive_start_eef[:, :3, 3], middle_object_pose[:, :3, 3]
            )
        )
        receive_direction = torch.where(
            vertical_mode[:, None], receive_diagonal, vertical_down
        )
        receive_direction_valid = ~vertical_mode | receive_diagonal_valid
        receive_grasp, receive_grasp_success = self._resolve_grasp(
            affordance,
            middle_object_pose,
            receive_direction,
            receive.hand.target_id,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=~handover_is_positive_part,
        )
        receive_grasp = self._find_symmetric_nearest_xpos(
            receive_grasp, receive_start_eef
        )

        receive_pre_grasp = translate_pose_world(
            receive_grasp,
            -receive_direction * options.pre_grasp_distance,
        )
        receive_object_to_eef = torch.bmm(
            pose_inv(middle_object_pose),
            receive_grasp,
        )
        lowering_direction_valid = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        receive_above_eef: torch.Tensor | None = None
        receive_final_eef: torch.Tensor | None = None
        if options.release_at_target:
            placed_object_pose = final_object_pose.clone()
            placed_object_pose[:, :3, :3] = middle_object_pose[:, :3, :3]
            above_object_pose = placed_object_pose.clone()
            # Move to the target's horizontal coordinates while preserving the
            # middle handover height exactly. The following target performs the
            # only vertical motion and reaches the requested final object pose.
            above_object_pose[:, 2, 3] = middle_object_pose[:, 2, 3]
            lowering_direction_valid = (
                above_object_pose[:, 2, 3] - placed_object_pose[:, 2, 3] > 1.0e-6
            )
            receive_above_eef = torch.bmm(above_object_pose, receive_object_to_eef)
            receive_final_eef = torch.bmm(placed_object_pose, receive_object_to_eef)
            # Likewise, receiving-grasp through final lowering reuses the same EEF
            # rotation and changes translation only.
            receive_above_eef[:, :3, :3] = receive_grasp[:, :3, :3]
            receive_final_eef[:, :3, :3] = receive_grasp[:, :3, :3]
        self._report_waypoint_failure(
            context,
            "pickup_grasp",
            active_mask & ~handover_grasp_success,
            "no finite grasp candidate on the pickup-side object end for arm "
            f"{handover.arm.control_part!r}",
        )
        self._report_waypoint_failure(
            context,
            "pickup_approach_direction",
            active_mask & ~handover_direction_valid,
            "handover TCP and observed object position have no horizontal "
            f"separation for arm {handover.arm.control_part!r}",
        )
        self._report_waypoint_failure(
            context,
            "receive_approach_direction",
            active_mask & ~receive_direction_valid,
            "receive TCP and predicted object position have no horizontal "
            f"separation for arm {receive.arm.control_part!r}",
        )
        self._report_waypoint_failure(
            context,
            "receive_grasp",
            active_mask & ~receive_grasp_success,
            "no finite grasp candidate on the opposite object end for arm "
            f"{receive.arm.control_part!r}",
        )
        if options.release_at_target:
            self._report_waypoint_failure(
                context,
                "target_final",
                active_mask & ~lowering_direction_valid,
                "final target is not below the horizontal-transfer height",
            )

        success = (
            handover_direction_valid
            & handover_grasp_success
            & receive_direction_valid
            & receive_grasp_success
            & lowering_direction_valid
        )
        pickup_approach_targets = torch.stack(
            [handover_pre_grasp, handover_grasp], dim=1
        )
        phase_success, pickup_approach = plan_named_arm_trajectory(
            self.motion_generator,
            handover.arm.control_part,
            handover_start_qpos,
            pickup_approach_targets,
            segment_lengths["pickup_approach"],
            request.motion_policy,
            context.control_dt,
        )
        pickup_approach_success = normalize_success_mask(
            phase_success,
            num_envs=self.num_envs,
            device=self.device,
            name="HandOver pickup-approach success",
        )
        self._report_phase_failure(
            context,
            phase_name="pickup_approach",
            waypoint_names=("pickup_pre_grasp", "pickup_grasp"),
            target_poses=pickup_approach_targets,
            start_qpos=handover_start_qpos,
            arm=handover.arm,
            failed_mask=active_mask & ~pickup_approach_success,
        )
        success &= pickup_approach_success
        handover_grasp_qpos = pickup_approach[:, -1]

        pickup_transport_targets = torch.stack(
            [handover_lift_eef, handover_middle_eef], dim=1
        )
        phase_success, pickup_transport = plan_named_arm_trajectory(
            self.motion_generator,
            handover.arm.control_part,
            handover_grasp_qpos,
            pickup_transport_targets,
            segment_lengths["pickup_transport"],
            request.motion_policy,
            context.control_dt,
        )
        pickup_transport_success = normalize_success_mask(
            phase_success,
            num_envs=self.num_envs,
            device=self.device,
            name="HandOver pickup-transport success",
        )
        self._report_phase_failure(
            context,
            phase_name="pickup_transport",
            waypoint_names=("pickup_lift", "handover_middle"),
            target_poses=pickup_transport_targets,
            start_qpos=handover_grasp_qpos,
            arm=handover.arm,
            failed_mask=active_mask & ~pickup_transport_success,
        )
        success &= pickup_transport_success
        handover_middle_qpos = pickup_transport[:, -1]

        receive_approach_targets = torch.stack(
            [receive_pre_grasp, receive_grasp], dim=1
        )
        phase_success, receive_approach = plan_named_arm_trajectory(
            self.motion_generator,
            receive.arm.control_part,
            receive_start_qpos,
            receive_approach_targets,
            segment_lengths["receive_approach"],
            request.motion_policy,
            context.control_dt,
        )
        receive_approach_success = normalize_success_mask(
            phase_success,
            num_envs=self.num_envs,
            device=self.device,
            name="HandOver receive-approach success",
        )
        self._report_phase_failure(
            context,
            phase_name="receive_approach",
            waypoint_names=("receive_pre_grasp", "receive_grasp"),
            target_poses=receive_approach_targets,
            start_qpos=receive_start_qpos,
            arm=receive.arm,
            failed_mask=active_mask & ~receive_approach_success,
        )
        success &= receive_approach_success
        receive_grasp_qpos = receive_approach[:, -1]

        receive_place: torch.Tensor | None = None
        receive_final_qpos: torch.Tensor | None = None
        if options.release_at_target:
            assert receive_above_eef is not None and receive_final_eef is not None
            placement_targets = torch.stack(
                [receive_above_eef, receive_final_eef], dim=1
            )
            phase_success, receive_place = plan_named_arm_trajectory(
                self.motion_generator,
                receive.arm.control_part,
                receive_grasp_qpos,
                placement_targets,
                segment_lengths["place"],
                request.motion_policy,
                context.control_dt,
            )
            placement_success = normalize_success_mask(
                phase_success,
                num_envs=self.num_envs,
                device=self.device,
                name="HandOver placement success",
            )
            self._report_phase_failure(
                context,
                phase_name="place",
                waypoint_names=("target_above", "target_final"),
                target_poses=placement_targets,
                start_qpos=receive_grasp_qpos,
                arm=receive.arm,
                failed_mask=active_mask & ~placement_success,
            )
            success &= placement_success
            receive_final_qpos = receive_place[:, -1]

        segments = [
            self._assemble_segment(
                state,
                pickup_approach,
                repeat_qpos(receive_start_qpos, segment_lengths["pickup_approach"]),
                repeat_qpos(
                    handover.hand_open_qpos, segment_lengths["pickup_approach"]
                ),
                repeat_qpos(receive.hand_open_qpos, segment_lengths["pickup_approach"]),
                handover,
                receive,
            ),
            self._assemble_segment(
                state,
                repeat_qpos(handover_grasp_qpos, segment_lengths["pickup_close"]),
                repeat_qpos(receive_start_qpos, segment_lengths["pickup_close"]),
                interpolate_hand_qpos(
                    handover.hand_open_qpos,
                    handover.hand_grasp_qpos,
                    n_waypoints=segment_lengths["pickup_close"],
                ),
                repeat_qpos(receive.hand_open_qpos, segment_lengths["pickup_close"]),
                handover,
                receive,
            ),
            self._assemble_segment(
                state,
                pickup_transport,
                repeat_qpos(receive_start_qpos, segment_lengths["pickup_transport"]),
                repeat_qpos(
                    handover.hand_grasp_qpos,
                    segment_lengths["pickup_transport"],
                ),
                repeat_qpos(
                    receive.hand_open_qpos, segment_lengths["pickup_transport"]
                ),
                handover,
                receive,
            ),
            self._assemble_segment(
                state,
                repeat_qpos(handover_middle_qpos, segment_lengths["receive_approach"]),
                receive_approach,
                repeat_qpos(
                    handover.hand_grasp_qpos,
                    segment_lengths["receive_approach"],
                ),
                repeat_qpos(
                    receive.hand_open_qpos, segment_lengths["receive_approach"]
                ),
                handover,
                receive,
            ),
            self._assemble_segment(
                state,
                repeat_qpos(handover_middle_qpos, segment_lengths["receive_close"]),
                repeat_qpos(receive_grasp_qpos, segment_lengths["receive_close"]),
                repeat_qpos(handover.hand_grasp_qpos, segment_lengths["receive_close"]),
                interpolate_hand_qpos(
                    receive.hand_open_qpos,
                    receive.hand_grasp_qpos,
                    n_waypoints=segment_lengths["receive_close"],
                ),
                handover,
                receive,
            ),
            self._assemble_segment(
                state,
                repeat_qpos(handover_middle_qpos, segment_lengths["handover_release"]),
                repeat_qpos(receive_grasp_qpos, segment_lengths["handover_release"]),
                interpolate_hand_qpos(
                    handover.hand_grasp_qpos,
                    handover.hand_open_qpos,
                    n_waypoints=segment_lengths["handover_release"],
                ),
                repeat_qpos(
                    receive.hand_grasp_qpos,
                    segment_lengths["handover_release"],
                ),
                handover,
                receive,
            ),
        ]
        if options.release_at_target:
            assert receive_place is not None and receive_final_qpos is not None
            segments.extend(
                (
                    self._assemble_segment(
                        state,
                        repeat_qpos(handover_middle_qpos, segment_lengths["place"]),
                        receive_place,
                        repeat_qpos(handover.hand_open_qpos, segment_lengths["place"]),
                        repeat_qpos(receive.hand_grasp_qpos, segment_lengths["place"]),
                        handover,
                        receive,
                    ),
                    self._assemble_segment(
                        state,
                        repeat_qpos(
                            handover_middle_qpos,
                            segment_lengths["receive_release"],
                        ),
                        repeat_qpos(
                            receive_final_qpos,
                            segment_lengths["receive_release"],
                        ),
                        repeat_qpos(
                            handover.hand_open_qpos,
                            segment_lengths["receive_release"],
                        ),
                        interpolate_hand_qpos(
                            receive.hand_grasp_qpos,
                            receive.hand_open_qpos,
                            n_waypoints=segment_lengths["receive_release"],
                        ),
                        handover,
                        receive,
                    ),
                )
            )
        trajectory = torch.cat(segments, dim=1)
        actual_lengths = {
            name: segment.shape[1]
            for name, segment in zip(segment_lengths, segments, strict=True)
        }
        if actual_lengths != segment_lengths:
            logger.log_warning(
                "HandOver planner returned segment lengths that differ from the request."
            )
        return _DirectionalPlan(
            success=success,
            trajectory=trajectory,
            segment_lengths=actual_lengths,
            handover_object_to_eef=handover_object_to_eef,
            handover_grasp_xpos=handover_grasp,
            receive_object_to_eef=receive_object_to_eef,
            receive_grasp_xpos=receive_grasp,
        )

    @staticmethod
    def _report_waypoint_failure(
        context: PlanningContext,
        waypoint_name: str,
        failed_mask: torch.Tensor,
        reason: str,
    ) -> None:
        """Log one semantic waypoint failure with affected environment IDs."""
        if not failed_mask.any():
            return
        env_ids = context.env_ids.to(failed_mask.device)[failed_mask]
        logger.log_warning(
            f"HandOver waypoint '{waypoint_name}' failed for "
            f"env_ids={env_ids.detach().cpu().tolist()}: {reason}."
        )

    def _report_phase_failure(
        self,
        context: PlanningContext,
        *,
        phase_name: str,
        waypoint_names: tuple[str, ...],
        target_poses: torch.Tensor,
        start_qpos: torch.Tensor,
        arm: JointPositionTarget,
        failed_mask: torch.Tensor,
    ) -> None:
        """Identify failed waypoint IK, or report a path/collision failure."""
        if not failed_mask.any():
            return
        identified = torch.zeros_like(failed_mask)
        joint_seed = start_qpos
        try:
            for waypoint_index, waypoint_name in enumerate(waypoint_names):
                ik_success, waypoint_qpos = self.robot.compute_ik(
                    pose=target_poses[:, waypoint_index],
                    name=arm.control_part,
                    joint_seed=joint_seed,
                )
                ik_success = normalize_success_mask(
                    ik_success,
                    num_envs=self.num_envs,
                    device=self.device,
                    name=f"HandOver diagnostic IK for {waypoint_name}",
                )
                waypoint_qpos = torch.as_tensor(
                    waypoint_qpos,
                    dtype=joint_seed.dtype,
                    device=self.device,
                )
                if waypoint_qpos.shape != joint_seed.shape:
                    raise ValueError(
                        "diagnostic IK returned qpos with shape "
                        f"{tuple(waypoint_qpos.shape)}, expected "
                        f"{tuple(joint_seed.shape)}"
                    )
                waypoint_failed = failed_mask & ~ik_success
                self._report_waypoint_failure(
                    context,
                    waypoint_name,
                    waypoint_failed,
                    f"IK failed for arm {arm.control_part!r}",
                )
                identified |= waypoint_failed
                joint_seed = torch.where(ik_success[:, None], waypoint_qpos, joint_seed)
        except Exception as exc:
            env_ids = context.env_ids.to(failed_mask.device)[failed_mask]
            logger.log_warning(
                f"HandOver phase '{phase_name}' failed for "
                f"arm {arm.control_part!r}, "
                f"env_ids={env_ids.detach().cpu().tolist()}, "
                "but waypoint IK "
                f"diagnostics could not run: {exc}."
            )
            return

        unresolved = failed_mask & ~identified
        if unresolved.any():
            env_ids = context.env_ids.to(unresolved.device)[unresolved]
            logger.log_warning(
                f"HandOver phase '{phase_name}' failed between waypoints "
                f"{list(waypoint_names)} for "
                f"arm {arm.control_part!r}, "
                f"env_ids={env_ids.detach().cpu().tolist()}; "
                "individual "
                "waypoint IK succeeded, so the likely cause is path or "
                "collision planning."
            )

    def _root_link_pose(
        self,
        arm: JointPositionTarget,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Read the root-link pose configured for ``arm``."""
        robot_cfg = getattr(self.robot, "cfg", None)
        solver_cfg = getattr(robot_cfg, "solver_cfg", None)
        if not isinstance(solver_cfg, Mapping) or arm.control_part not in solver_cfg:
            raise ValueError(
                "HandOver requires " f"solver_cfg[{arm.control_part!r}].root_link_name."
            )
        root_link_name = getattr(solver_cfg[arm.control_part], "root_link_name", None)
        if not isinstance(root_link_name, str) or not root_link_name:
            raise ValueError(
                "HandOver requires a root_link_name for arm " f"{arm.control_part!r}."
            )
        pose = self.robot.get_link_pose(
            link_name=root_link_name,
            env_ids=env_ids.tolist(),
            to_matrix=True,
        )
        return resolve_batched_pose(
            pose,
            num_envs=self.num_envs,
            device=self.device,
            name=f"{arm.control_part} root-link pose",
        )

    def _resolve_grasp(
        self,
        affordance: AntipodalAffordance,
        object_pose: torch.Tensor,
        approach_direction: torch.Tensor,
        grasp_target_id: str,
        *,
        obj_longest_axis: torch.Tensor,
        is_positive_part: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the lowest-cost grasp on one projected end of the object."""
        if object_pose.shape != (self.num_envs, 4, 4):
            raise ValueError(
                "HandOver grasp object_pose must have shape "
                f"({self.num_envs}, 4, 4)."
            )
        if approach_direction.shape != (self.num_envs, 3):
            raise ValueError(
                "HandOver grasp approach_direction must have shape "
                f"({self.num_envs}, 3)."
            )
        if obj_longest_axis.shape != (self.num_envs, 3):
            raise ValueError(
                f"HandOver obj_longest_axis must have shape ({self.num_envs}, 3)."
            )
        if is_positive_part.dtype != torch.bool or is_positive_part.shape != (
            self.num_envs,
        ):
            raise ValueError(
                "HandOver is_positive_part must be a bool tensor with shape "
                f"({self.num_envs},)."
            )

        generator = self.planning_services.grasp_pose_generator(grasp_target_id)
        sampled = generator.get_valid_grasp_poses(
            mesh_vertices=affordance.mesh_vertices,
            mesh_triangles=affordance.mesh_triangles,
            obj_poses=object_pose,
            approach_direction=approach_direction,
            obj_longest_axis=obj_longest_axis,
            is_positive_part=is_positive_part,
        )
        if len(sampled) != self.num_envs:
            raise ValueError(
                "HandOver expected exactly one grasp-sampling result per environment."
            )
        poses = torch.eye(
            4,
            dtype=torch.float32,
            device=self.device,
        ).repeat(self.num_envs, 1, 1)
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for env_index, (candidates, costs) in enumerate(sampled):
            candidates = candidates.to(device=self.device, dtype=torch.float32)
            costs = costs.to(device=self.device, dtype=torch.float32)
            if candidates.shape[0] == 0 or not torch.isfinite(costs).any():
                continue
            finite_costs = torch.where(
                torch.isfinite(costs),
                costs,
                torch.full_like(costs, torch.inf),
            )
            poses[env_index] = candidates[torch.argmin(finite_costs)]
            success[env_index] = True
        return poses, success

    @staticmethod
    def _downward_diagonal_approach_direction(
        start_position: torch.Tensor,
        target_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return TCP-to-target horizontal directions tilted down by 45 degrees.

        The direction is valid only when the TCP and target have nonzero
        horizontal separation; callers report the corresponding semantic
        approach waypoint when that construction is undefined.
        """
        horizontal_delta = target_position[:, :2] - start_position[:, :2]
        horizontal_norm = torch.linalg.vector_norm(horizontal_delta, dim=1)
        valid = horizontal_norm > 1.0e-6
        horizontal_unit = horizontal_delta / horizontal_norm.clamp_min(
            1.0e-6
        ).unsqueeze(1)
        component = math.sqrt(0.5)
        direction = torch.zeros(
            (start_position.shape[0], 3),
            dtype=start_position.dtype,
            device=start_position.device,
        )
        direction[:, :2] = horizontal_unit * component
        direction[:, 2] = -component
        return direction, valid

    @staticmethod
    def _middle_object_pose(
        lifted_object_pose: torch.Tensor,
        handover_root_pose: torch.Tensor,
        receive_root_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Move only the dominant root-separation coordinate to its midpoint."""
        handover_root_position = handover_root_pose[:, :3, 3]
        receive_root_position = receive_root_pose[:, :3, 3]
        dominant_axis = torch.argmax(
            torch.abs(handover_root_position - receive_root_position), dim=1
        )
        root_midpoint = 0.5 * (handover_root_position + receive_root_position)
        middle = lifted_object_pose.clone()
        middle_position = middle[:, :3, 3]
        selected_midpoint = root_midpoint.gather(1, dominant_axis[:, None])
        middle_position.scatter_(1, dominant_axis[:, None], selected_midpoint)
        return middle

    @staticmethod
    def _compute_segment_lengths(
        sample_count: int,
        options: HandOverOptions,
    ) -> dict[str, int]:
        """Split the sample budget across enabled arm and hand phases."""
        hand_count = options.hand_interp_steps
        hand_phase_count = 4 if options.release_at_target else 3
        motion_phase_count = 4 if options.release_at_target else 3
        motion_budget = sample_count - hand_phase_count * hand_count
        if motion_budget < 2 * motion_phase_count:
            raise ValueError(
                "Not enough HandOver waypoints. Increase sample_count or decrease "
                "hand_interp_steps."
            )
        motion_counts = [motion_budget // motion_phase_count] * motion_phase_count
        for index in range(motion_budget % motion_phase_count):
            motion_counts[index] += 1
        result = {
            "pickup_approach": motion_counts[0],
            "pickup_close": hand_count,
            "pickup_transport": motion_counts[1],
            "receive_approach": motion_counts[2],
            "receive_close": hand_count,
            "handover_release": hand_count,
        }
        if options.release_at_target:
            result.update(
                {
                    "place": motion_counts[3],
                    "receive_release": hand_count,
                }
            )
        return result

    @staticmethod
    def _assemble_segment(
        state: PlanningContext,
        handover_arm_trajectory: torch.Tensor,
        receive_arm_trajectory: torch.Tensor,
        handover_hand_trajectory: torch.Tensor,
        receive_hand_trajectory: torch.Tensor,
        handover: _Participant,
        receive: _Participant,
    ) -> torch.Tensor:
        """Embed both arm and hand paths in the full robot joint order."""
        return assemble_full_robot_trajectory(
            state.last_qpos,
            (
                (handover.arm.joint_ids, handover_arm_trajectory),
                (receive.arm.joint_ids, receive_arm_trajectory),
                (handover.hand.joint_ids, handover_hand_trajectory),
                (receive.hand.joint_ids, receive_hand_trajectory),
            ),
        )


__all__ = ["HandOver", "HandOverGoal", "HandOverOptions"]
