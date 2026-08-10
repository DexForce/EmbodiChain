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

"""Place atomic action implementation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from embodichain.utils.math import quat_error_magnitude, quat_from_matrix

from ._helpers import arm_qpos_from_state, resolve_object_target
from ..affordance import AssembleAffordance
from ..control import GRASP_COMMAND, OPEN_COMMAND, JointPositionCommand
from ..core import AtomicAction
from ..effects import StateDelta
from ..goals import (
    PoseGoalValue,
    SceneEntityPose,
    resolve_pose_goal,
    validate_pose_goal,
)
from ..invocation import ActionOptions, ResolvedActionRequest
from ..plans import ActionPlan
from ..requirements import (
    ActionBindingRoute,
    CARTESIAN_POSE_CAPABILITY,
    DisjointSlotEndpoints,
    FORWARD_KINEMATICS_CAPABILITY,
    GRASP_CAPABILITY,
    SkillBindingContract,
    SkillEndpointRequirement,
    SkillResourceSlot,
)
from ..state import PlanningContext
from ..trajectory_ops import (
    build_pose_plan_states,
    interpolate_hand_qpos,
    resolve_pose_target,
    split_three_segments,
)

TcpSymmetry = Literal["none", "z_roll_180"]


@dataclass(frozen=True, slots=True, eq=False)
class PlaceGoal:
    """End-effector release-pose target used by :class:`Place`."""

    xpos: PoseGoalValue
    """Target end-effector release pose.

    Accepts ``(4, 4)``, ``(num_envs, 4, 4)``, or
    ``(num_envs, n_waypoint, 4, 4)``.
    """

    tcp_symmetry: TcpSymmetry = "none"
    """Optional TCP-frame symmetry allowed by the placement semantics.

    ``"none"`` preserves the pose exactly. ``"z_roll_180"`` lets placement
    choose between the pose and its TCP z-roll 180 equivalent, which flips TCP
    x/y while preserving TCP z and translation.
    """

    def __post_init__(self) -> None:
        validate_pose_goal(self.xpos, "xpos", allow_waypoints=True)
        if self.tcp_symmetry not in ("none", "z_roll_180"):
            raise ValueError(
                "tcp_symmetry must be one of 'none' or 'z_roll_180', "
                f"but got {self.tcp_symmetry!r}"
            )


@dataclass(frozen=True, slots=True, eq=False)
class AssembleGoal:
    """Place a held assemble object onto a base object at a relative pose.

    The preferred base object pose is a late-bound :class:`SceneEntityPose`.
    Omitting it temporarily falls back to
    :attr:`AssembleAffordance.base_object_entity`. The assemble object's target
    pose is ``base_pose @ assemble_to_base_pose``. The held-object transform
    (``object_to_eef``) is read from :class:`PlanningContext` for the place
    control part, which a prior :class:`PickUp` populates.
    """

    affordance: AssembleAffordance
    """Assembly affordance anchoring the assemble object to the base object."""

    base_pose: SceneEntityPose | None = None
    """Late-bound base-object pose used for snapshot-consistent planning."""

    def __post_init__(self) -> None:
        if not isinstance(self.affordance, AssembleAffordance):
            raise TypeError("affordance must be an AssembleAffordance instance.")
        if self.base_pose is not None and not isinstance(
            self.base_pose,
            SceneEntityPose,
        ):
            raise TypeError("base_pose must be a SceneEntityPose or None.")


@dataclass(frozen=True, slots=True, eq=False)
class PlaceOptions(ActionOptions):
    """Per-invocation placement behavior."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper-open interpolation segment."""

    lift_height: float = 0.1
    """Height (m) to retract the end-effector after opening the gripper."""

    max_approach_retract_z: float | None = None
    """Optional maximum world-frame TCP z for approach and retract poses (m)."""

    cartesian_waypoint_count: int = 1
    """Number of fixed-orientation Cartesian keyframes per translation segment."""

    def __post_init__(self) -> None:
        if self.hand_interp_steps < 1:
            raise ValueError("hand_interp_steps must be at least 1.")
        if self.lift_height < 0.0:
            raise ValueError("lift_height must be non-negative.")
        if self.cartesian_waypoint_count < 1:
            raise ValueError("cartesian_waypoint_count must be at least 1.")


class Place(AtomicAction[PlaceGoal | AssembleGoal, PlaceOptions]):
    """Lower the held object to a place pose, open the gripper, retract.

    The :class:`PlaceGoal` may carry either a single waypoint
    ``(num_envs, 4, 4)`` (or a broadcastable ``(4, 4)``) or a multi-waypoint
    trajectory ``(num_envs, n_waypoint, 4, 4)``. In the multi-waypoint case the
    approach segment visits every waypoint in order; approaching from above the
    first waypoint, descending through each waypoint, then opening the gripper
    at the final waypoint and retracting to above the last waypoint. Starting
    joint positions are inherited from :class:`PlanningContext`.

    An :class:`AssembleGoal` replaces the explicit EEF pose with an assembly
    affordance: the place pose is derived from the base object's snapshot pose
    (or deprecated live fallback) and ``assemble_to_base_pose``, converted to an
    EEF pose through the held object's ``object_to_eef`` (read from
    :class:`PlanningContext`).
    """

    skill_id: ClassVar[str] = "place"
    GoalType: ClassVar[type | tuple[type, ...]] = (
        PlaceGoal,
        AssembleGoal,
    )
    OptionsType: ClassVar[type] = PlaceOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("primary",)
    end_effector_roles: ClassVar[tuple[str, ...]] = ("primary",)
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=(
            SkillResourceSlot(
                slot_id="primary",
                endpoints=(
                    SkillEndpointRequirement(
                        endpoint_id="motion",
                        capabilities=frozenset(
                            {
                                CARTESIAN_POSE_CAPABILITY,
                                FORWARD_KINEMATICS_CAPABILITY,
                            }
                        ),
                        route=ActionBindingRoute("manipulator", "primary"),
                    ),
                    SkillEndpointRequirement(
                        endpoint_id="grasp",
                        capabilities=frozenset({GRASP_CAPABILITY}),
                        required_commands={
                            OPEN_COMMAND: JointPositionCommand,
                            GRASP_COMMAND: JointPositionCommand,
                        },
                        route=ActionBindingRoute("end_effector", "primary"),
                    ),
                ),
                constraints=(DisjointSlotEndpoints(("motion", "grasp")),),
            ),
        ),
    )

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[PlaceGoal | AssembleGoal, PlaceOptions],
    ) -> tuple[str, ...]:
        """Include an explicitly snapshot-grounded assembly base."""
        dependencies = set(super()._scene_dependencies(request))
        target = request.goal
        if isinstance(target, AssembleGoal) and target.base_pose is not None:
            dependencies.add(target.base_pose.entity_id)
        return tuple(sorted(dependencies))

    def _plan(
        self,
        request: ResolvedActionRequest[PlaceGoal | AssembleGoal, PlaceOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan approach, release, and retract without committing detachment."""
        target = request.goal
        options = request.skill_options
        binding = request.binding
        manipulator = binding.manipulator()
        end_effector = binding.end_effector()
        control_part = manipulator.name
        arm_joint_ids = list(manipulator.joint_ids)
        hand_joint_ids = list(end_effector.joint_ids)
        hand_open_qpos = end_effector.joint_positions(
            OPEN_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        hand_grasp_qpos = end_effector.joint_positions(
            GRASP_COMMAND,
            num_envs=context.batch_size,
            device=self.device,
            dtype=context.robot.qpos.dtype,
        )
        state = context
        held_mask = context.task.held_object_mask(control_part)
        exclusive_mask = context.task.exclusive_held_object_mask(control_part)
        eligible = (
            exclusive_mask
            if isinstance(target, AssembleGoal)
            else ~held_mask | exclusive_mask
        )
        place_xpos = self._resolve_place_xpos(target, state, control_part)
        if not eligible.any():
            return self.failed_plan(
                request,
                context,
                message="Held object is shared with another control part.",
            )
        if place_xpos.dim() == 3:
            place_xpos = place_xpos.unsqueeze(1)

        start_arm_qpos = arm_qpos_from_state(state, arm_joint_ids)
        if isinstance(target, PlaceGoal) and target.tcp_symmetry == "z_roll_180":
            place_xpos = self._select_tcp_symmetric_place_variant(
                place_xpos, start_arm_qpos, control_part
            )
        n_down, n_open, n_back = split_three_segments(
            request.motion_policy.sample_count,
            options.hand_interp_steps,
            first_segment_name="approach",
            third_segment_name="back",
        )

        approach_xpos = self._lifted_pose(place_xpos[:, 0], options)
        retract_xpos = self._lifted_pose(place_xpos[:, -1], options)

        start_xpos = self.robot.compute_fk(
            qpos=start_arm_qpos,
            name=control_part,
            to_matrix=True,
        )
        down_xpos = torch.cat([approach_xpos.unsqueeze(1), place_xpos], dim=1)
        down_xpos = self._translation_keyframes(start_xpos, down_xpos, options)

        down_result = self.motion_generator.generate(
            build_pose_plan_states(down_xpos),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=start_arm_qpos,
                control_part=control_part,
                sample_count=n_down,
            ),
        )
        assert isinstance(down_result.success, torch.Tensor)
        assert down_result.positions is not None
        down_success = down_result.success
        down_arm = down_result.positions
        reach_arm_qpos = down_arm[:, -1, :]

        back_xpos = self._translation_keyframes(
            place_xpos[:, -1], retract_xpos.unsqueeze(1), options
        )
        back_result = self.motion_generator.generate(
            build_pose_plan_states(back_xpos),
            options=request.motion_policy.to_motion_gen_options(
                start_qpos=reach_arm_qpos,
                control_part=control_part,
                sample_count=n_back,
            ),
        )
        assert isinstance(back_result.success, torch.Tensor)
        assert back_result.positions is not None
        back_success = back_result.success
        back_arm = back_result.positions
        success = down_success & back_success & eligible

        hand_open_path = interpolate_hand_qpos(
            hand_grasp_qpos, hand_open_qpos, n_waypoints=n_open
        )

        # Allocate from the actually returned segment lengths so collision-aware
        # planners (which preserve their own sample count) are accommodated.
        n_down_actual = down_arm.shape[1]
        n_back_actual = back_arm.shape[1]
        full = torch.empty(
            (self.num_envs, n_down_actual + n_open + n_back_actual, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_down_actual, arm_joint_ids] = down_arm
        full[:, :n_down_actual, hand_joint_ids] = hand_grasp_qpos.unsqueeze(1)
        full[:, n_down_actual : n_down_actual + n_open, arm_joint_ids] = (
            reach_arm_qpos.unsqueeze(1)
        )
        full[:, n_down_actual : n_down_actual + n_open, hand_joint_ids] = hand_open_path
        full[:, n_down_actual + n_open :, arm_joint_ids] = back_arm
        full[:, n_down_actual + n_open :, hand_joint_ids] = hand_open_qpos.unsqueeze(1)

        return self.build_plan(
            request,
            context,
            success=success,
            trajectory=full,
            expected_effects=StateDelta(held_object_updates={control_part: None}),
            segment_lengths={
                "approach": n_down_actual,
                "release": n_open,
                "retract": n_back_actual,
            },
        )

    def _resolve_place_xpos(
        self,
        target: PlaceGoal | AssembleGoal,
        state: PlanningContext,
        control_part: str,
    ) -> torch.Tensor:
        """Resolve the place EEF poses from a typed target.

        Args:
            target: Either an explicit EEF pose target or an assembly target.
            state: World state carrying the held-object transform.

        Returns:
            Place EEF poses with shape ``(num_envs, 4, 4)`` or
            ``(num_envs, n_waypoint, 4, 4)``.
        """
        if isinstance(target, PlaceGoal):
            return resolve_pose_target(
                resolve_pose_goal(target.xpos, state, name="xpos"),
                num_envs=self.num_envs,
                device=self.device,
            )
        return self._resolve_assemble_place_xpos(target, state, control_part)

    def _resolve_assemble_place_xpos(
        self,
        target: AssembleGoal,
        state: PlanningContext,
        control_part: str,
    ) -> torch.Tensor:
        """Derive the place EEF pose from an assembly affordance.

        The assemble object target pose is ``base_pose @ assemble_to_base_pose``;
        the EEF pose is that target posed through the held object's
        ``object_to_eef``.

        Args:
            target: Assembly target carrying the base/assemble affordance.
            state: World state carrying the held-object transform.

        Returns:
            Place EEF poses with shape ``(num_envs, 4, 4)``.

        Raises:
            ValueError: If no held object or base-pose source is available.
        """
        held = state.get_held_object(control_part)
        if held is None:
            raise ValueError(
                "Place with AssembleGoal requires an object held by control "
                f"part {control_part!r} (run PickUp first)."
            )
        affordance = target.affordance
        if target.base_pose is not None:
            base_pose = resolve_object_target(
                resolve_pose_goal(
                    target.base_pose,
                    state,
                    name="base_pose",
                ),
                num_envs=self.num_envs,
                device=self.device,
                name="base_pose",
            )
        else:
            if affordance.base_object_entity is None:
                raise ValueError(
                    "AssembleGoal requires base_pose or "
                    "AssembleAffordance.base_object_entity."
                )
            warnings.warn(
                "AssembleGoal without base_pose reads "
                "AssembleAffordance.base_object_entity live; provide "
                "base_pose=SceneEntityPose(...) instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            base_pose = resolve_object_target(
                affordance.base_object_entity.get_local_pose(to_matrix=True),
                num_envs=self.num_envs,
                device=self.device,
                name="legacy_base_pose",
            )
        assemble_object_pose = affordance.get_assemble_object_pose(base_pose)
        object_to_eef = resolve_object_target(
            held.object_to_eef,
            num_envs=self.num_envs,
            device=self.device,
            name="object_to_eef",
        )
        return torch.bmm(assemble_object_pose, object_to_eef)

    def _lifted_pose(
        self, release_xpos: torch.Tensor, options: PlaceOptions
    ) -> torch.Tensor:
        """Build an above-release pose while respecting the optional TCP z cap."""
        lifted_xpos = release_xpos.clone()
        lifted_z = release_xpos[:, 2, 3] + options.lift_height
        if options.max_approach_retract_z is not None:
            max_z = torch.as_tensor(
                options.max_approach_retract_z,
                dtype=release_xpos.dtype,
                device=release_xpos.device,
            )
            lifted_z = torch.maximum(
                release_xpos[:, 2, 3],
                torch.clamp_max(lifted_z, max_z),
            )
        lifted_xpos[:, 2, 3] = lifted_z
        return lifted_xpos

    def _translation_keyframes(
        self,
        start_xpos: torch.Tensor,
        target_xpos: torch.Tensor,
        options: PlaceOptions,
    ) -> torch.Tensor:
        """Interpolate translations while holding each segment's target rotation."""
        count = options.cartesian_waypoint_count
        if count == 1:
            return target_xpos

        segment_starts = torch.cat(
            [start_xpos.unsqueeze(1), target_xpos[:, :-1]], dim=1
        )
        alpha = torch.linspace(
            1.0 / count,
            1.0,
            count,
            dtype=target_xpos.dtype,
            device=self.device,
        )
        keyframes = target_xpos.unsqueeze(2).repeat(1, 1, count, 1, 1)
        start_position = segment_starts[..., :3, 3].unsqueeze(2)
        target_position = target_xpos[..., :3, 3].unsqueeze(2)
        keyframes[..., :3, 3] = start_position + alpha[None, None, :, None] * (
            target_position - start_position
        )
        return keyframes.flatten(1, 2)

    def _select_tcp_symmetric_place_variant(
        self,
        place_xpos: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
    ) -> torch.Tensor:
        """Choose the closest TCP z-roll variant for an opt-in place target."""
        mirrored_place_xpos = place_xpos.clone()
        mirrored_place_xpos[..., :3, 0] = -mirrored_place_xpos[..., :3, 0]
        mirrored_place_xpos[..., :3, 1] = -mirrored_place_xpos[..., :3, 1]
        place_variants = torch.stack([place_xpos, mirrored_place_xpos], dim=2)

        start_xpos = self.robot.compute_fk(
            qpos=start_qpos,
            name=control_part,
            to_matrix=True,
        )
        start_quat = quat_from_matrix(start_xpos[:, :3, :3])
        first_waypoint_quat = quat_from_matrix(place_variants[:, 0, :, :3, :3])
        start_quat = start_quat[:, None, :].expand_as(first_waypoint_quat)
        rotation_error = quat_error_magnitude(
            first_waypoint_quat.reshape(-1, 4),
            start_quat.reshape(-1, 4),
        ).reshape(self.num_envs, 2)
        best_variant_idx = rotation_error.argmin(dim=1)

        env_idx = torch.arange(self.num_envs, device=self.device)[:, None]
        waypoint_idx = torch.arange(place_xpos.shape[1], device=self.device)[None, :]
        return place_variants[
            env_idx,
            waypoint_idx,
            best_variant_idx[:, None],
        ]


__all__ = ["AssembleGoal", "Place", "PlaceGoal", "PlaceOptions"]
