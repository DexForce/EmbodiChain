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

"""CoordinatedPickment atomic action implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from embodichain.utils import logger
from embodichain.utils.math import matrix_from_quat, pose_inv, quat_from_matrix

from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.bindings import ResolvedControlPart
from embodichain.lab.sim.atomic_actions.control import (
    GRASP_COMMAND,
    OPEN_COMMAND,
    JointPositionCommand,
)
from embodichain.lab.sim.atomic_actions.core import AtomicAction, ObjectSemantics
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
from embodichain.lab.sim.atomic_actions.requirements import (
    DisjointResourceSlots,
    INVERSE_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    assemble_full_robot_trajectory,
    repeat_qpos,
    resolve_batched_pose,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    interpolate_joint_trajectory,
    translate_pose_world,
)


@dataclass(frozen=True, slots=True, eq=False)
class CoordinatedPickGoal(ObjectActionGoal):
    """Object-centric target for picking and moving one object with two hands.

    The left/right grasp poses are not supplied by the caller; they are sampled
    from :meth:`AntipodalAffordance.get_dual_arm_valid_grasp_poses` at planning
    time using the dual-arm direction and approach direction declared on
    :class:`CoordinatedPickmentOptions`.
    """

    object_target_pose: PoseGoalValue
    """Target pose for the shared object, shape ``(4, 4)`` or ``(num_envs, 4, 4)``."""

    object_initial_pose: PoseGoalValue | None = None
    """Optional initial object pose.

    When omitted, the pose is grounded through the semantic object's stable
    scene identity, with its live entity retained only as a legacy fallback.
    """

    def __post_init__(self) -> None:
        ObjectActionGoal.__post_init__(self)
        validate_pose_goal(
            self.object_target_pose,
            "object_target_pose",
            allow_waypoints=False,
        )
        if self.object_initial_pose is not None:
            validate_pose_goal(
                self.object_initial_pose,
                "object_initial_pose",
                allow_waypoints=False,
            )


@dataclass(frozen=True, slots=True, eq=False)
class CoordinatedPickmentOptions(ActionOptions):
    """Per-invocation coordinated pickup behavior.

    Left/right grasps are sampled from the target affordance using
    :meth:`AntipodalAffordance.get_dual_arm_valid_grasp_poses`. The dual-arm
    direction splits the object into left/right grasp regions and the approach
    direction filters the sampled antipodal pairs.
    """

    object_motion_keyframes: int = 6
    """Number of object-pose keyframes solved by IK before joint-space interpolation."""

    pre_grasp_distance: float = 0.10
    """World distance to retreat from each grasp pose along negative TCP z."""

    lift_height: float = 0.08
    """World-Z lift distance before moving to the object target pose."""

    hand_interp_steps: int = 10
    """Number of waypoints used for the simultaneous hand-close segment."""

    hold_steps: int = 4
    """Number of waypoints to hold the final object target pose."""

    approach_direction: torch.Tensor = torch.tensor(
        [0.0, 0.0, -1.0], dtype=torch.float32
    )
    """World-frame direction used to sample and approach both grasps, shape ``(3,)``."""

    left_to_right_arm_direction: torch.Tensor = torch.tensor(
        [1.0, 0.0, 0.0], dtype=torch.float32
    )
    """World-frame direction from the left arm base to the right arm base, shape
    ``(3,)``. It partitions the object into left/right grasp regions and should be
    a finite, non-zero vector; it is normalized at planning time."""

    middle_empty_ratio: float = 0.4
    """Fraction of the object's left-to-right extent left grasp-free in the middle
    so the two grippers pinch opposite ends. Must be in ``[0, 1]``."""

    def __post_init__(self) -> None:
        if self.object_motion_keyframes < 2:
            raise ValueError("object_motion_keyframes must be at least 2.")
        if self.pre_grasp_distance < 0.0:
            raise ValueError("pre_grasp_distance must be non-negative.")
        if self.lift_height < 0.0:
            raise ValueError("lift_height must be non-negative.")
        for name in ("hand_interp_steps", "hold_steps"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        for name in ("approach_direction", "left_to_right_arm_direction"):
            value = getattr(self, name)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,).")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain finite values.")
            if torch.linalg.vector_norm(value) <= 1.0e-6:
                raise ValueError(f"{name} must be non-zero.")
            object.__setattr__(self, name, value.clone())
        if not 0.0 <= self.middle_empty_ratio <= 1.0:
            raise ValueError("middle_empty_ratio must be in [0, 1].")


@dataclass(frozen=True, slots=True, eq=False)
class _CoordinatedPickResources:
    """Invocation-bound control parts and compatible hand commands."""

    left_arm: ResolvedControlPart
    right_arm: ResolvedControlPart
    left_hand: ResolvedControlPart
    right_hand: ResolvedControlPart
    left_hand_open_qpos: torch.Tensor
    left_hand_close_qpos: torch.Tensor
    right_hand_open_qpos: torch.Tensor
    right_hand_close_qpos: torch.Tensor


class _DualArmHelpers:
    """Shared trajectory helpers for dual-arm coordinated actions."""

    def _expand_qpos(self, qpos: torch.Tensor, dof: int, name: str) -> torch.Tensor:
        qpos = qpos.to(device=self.device, dtype=torch.float32)
        if qpos.shape == (dof,):
            return qpos.unsqueeze(0).repeat(self.num_envs, 1)
        if qpos.shape == (self.num_envs, dof):
            return qpos
        raise ValueError(
            f"{name} must have shape ({dof},) or "
            f"({self.num_envs}, {dof}), but got {qpos.shape}"
        )

    def _resolve_pose(self, pose: torch.Tensor, name: str) -> torch.Tensor:
        return resolve_batched_pose(
            pose,
            num_envs=self.num_envs,
            device=self.device,
            name=name,
        )

    def _resolve_dual_arm_start(
        self,
        state: PlanningContext,
        resources: _CoordinatedPickResources,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_qpos = state.last_qpos.to(device=self.device, dtype=torch.float32)
        return (
            start_qpos[:, list(resources.left_arm.joint_ids)],
            start_qpos[:, list(resources.right_arm.joint_ids)],
        )

    def _assemble_segment(
        self,
        state: PlanningContext,
        first_arm_traj: torch.Tensor,
        second_arm_traj: torch.Tensor,
        first_hand_traj: torch.Tensor,
        second_hand_traj: torch.Tensor,
        *,
        resources: _CoordinatedPickResources,
    ) -> torch.Tensor:
        return assemble_full_robot_trajectory(
            state.last_qpos,
            (
                (resources.left_arm.joint_ids, first_arm_traj),
                (resources.right_arm.joint_ids, second_arm_traj),
                (resources.left_hand.joint_ids, first_hand_traj),
                (resources.right_hand.joint_ids, second_hand_traj),
            ),
        )

    def _interpolate_qpos(
        self,
        start_qpos: torch.Tensor,
        end_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        weights = torch.linspace(
            0.0,
            1.0,
            steps=n_waypoints,
            device=self.device,
            dtype=start_qpos.dtype,
        )
        return torch.lerp(
            start_qpos.unsqueeze(1),
            end_qpos.unsqueeze(1),
            weights[None, :, None],
        )

    def _interpolate_keyframe_qpos(
        self, keyframe_qpos: torch.Tensor, n_waypoints: int
    ) -> torch.Tensor:
        n_keyframes = keyframe_qpos.shape[1]
        keyframe_indices = (
            torch.linspace(
                0,
                n_waypoints - 1,
                steps=n_keyframes,
                device=self.device,
            )
            .round()
            .to(dtype=torch.long)
        )
        return self._interpolate_qpos_keyframes(
            keyframe_qpos, keyframe_indices, n_waypoints
        )

    def _interpolate_qpos_keyframes(
        self,
        keyframe_qpos: torch.Tensor,
        keyframe_indices: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        trajectory = torch.zeros(
            (self.num_envs, n_waypoints, keyframe_qpos.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        for segment_idx in range(len(keyframe_indices) - 1):
            start_idx = int(keyframe_indices[segment_idx].item())
            end_idx = int(keyframe_indices[segment_idx + 1].item())
            n_segment = end_idx - start_idx + 1
            weights = torch.linspace(
                0.0,
                1.0,
                steps=n_segment,
                dtype=keyframe_qpos.dtype,
                device=self.device,
            )
            segment = torch.lerp(
                keyframe_qpos[:, segment_idx : segment_idx + 1],
                keyframe_qpos[:, segment_idx + 1 : segment_idx + 2],
                weights[None, :, None],
            )
            trajectory[:, start_idx : end_idx + 1] = segment
        return trajectory

    def _interpolate_object_pose(
        self,
        start_pose: torch.Tensor,
        end_pose: torch.Tensor,
        n_waypoints: int,
        *,
        include_orientation: bool,
    ) -> torch.Tensor:
        weights = torch.linspace(
            0.0,
            1.0,
            steps=n_waypoints,
            device=self.device,
            dtype=start_pose.dtype,
        )
        poses = start_pose.unsqueeze(1).repeat(1, n_waypoints, 1, 1)
        poses[:, :, :3, 3] = torch.lerp(
            start_pose[:, None, :3, 3],
            end_pose[:, None, :3, 3],
            weights[None, :, None],
        )
        if not include_orientation:
            return poses

        start_quat = quat_from_matrix(start_pose[:, :3, :3])
        end_quat = quat_from_matrix(end_pose[:, :3, :3])
        quat_dot = torch.sum(start_quat * end_quat, dim=-1, keepdim=True)
        end_quat = torch.where(quat_dot < 0.0, -end_quat, end_quat)
        quat = torch.lerp(
            start_quat.unsqueeze(1),
            end_quat.unsqueeze(1),
            weights[None, :, None],
        )
        quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-8)
        poses[:, :, :3, :3] = matrix_from_quat(quat.reshape(-1, 4)).reshape(
            self.num_envs, n_waypoints, 3, 3
        )
        return poses


class CoordinatedPickment(
    AtomicAction[CoordinatedPickGoal, CoordinatedPickmentOptions]
):
    """Pick and move a single object pinched by two hands."""

    skill_id: ClassVar[str] = "coordinated_pickment"
    GoalType: ClassVar[type] = CoordinatedPickGoal
    OptionsType: ClassVar[type] = CoordinatedPickmentOptions
    manipulator_roles: ClassVar[tuple[str, ...]] = ("left", "right")
    end_effector_roles: ClassVar[tuple[str, ...]] = ("left", "right")
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=tuple(
            make_manipulation_slot(
                role,
                motion_capabilities=frozenset({INVERSE_KINEMATICS_CAPABILITY}),
                grasp_commands={
                    OPEN_COMMAND: JointPositionCommand,
                    GRASP_COMMAND: JointPositionCommand,
                },
            )
            for role in ("left", "right")
        ),
        constraints=(DisjointResourceSlots(("left", "right")),),
    )

    _assemble_segment = _DualArmHelpers._assemble_segment
    _expand_qpos = _DualArmHelpers._expand_qpos
    _interpolate_keyframe_qpos = _DualArmHelpers._interpolate_keyframe_qpos
    _interpolate_object_pose = _DualArmHelpers._interpolate_object_pose
    _interpolate_qpos = _DualArmHelpers._interpolate_qpos
    _interpolate_qpos_keyframes = _DualArmHelpers._interpolate_qpos_keyframes
    _repeat_qpos = staticmethod(repeat_qpos)
    _resolve_dual_arm_start = _DualArmHelpers._resolve_dual_arm_start
    _resolve_pose = _DualArmHelpers._resolve_pose

    def _scene_dependencies(
        self,
        request: ResolvedActionRequest[
            CoordinatedPickGoal,
            CoordinatedPickmentOptions,
        ],
    ) -> tuple[str, ...]:
        """Track the semantic object only when it supplies the initial pose."""
        dependencies = set(super()._scene_dependencies(request))
        target = request.goal
        if target.object_initial_pose is None:
            entity_id = target.semantics.entity_id
            if entity_id is not None:
                dependencies.add(entity_id)
        return tuple(sorted(dependencies))

    def _resolve_resources(
        self,
        request: ResolvedActionRequest[CoordinatedPickGoal, CoordinatedPickmentOptions],
    ) -> _CoordinatedPickResources:
        """Resolve left/right roles from robot control parts."""
        binding = request.binding
        left_arm = binding.manipulator("left")
        right_arm = binding.manipulator("right")
        left_hand = binding.end_effector("left")
        right_hand = binding.end_effector("right")
        if left_arm.name == right_arm.name:
            raise ValueError(
                "CoordinatedPickment left and right roles must use different "
                "manipulator control parts."
            )
        if left_hand.name == right_hand.name:
            raise ValueError(
                "CoordinatedPickment left and right roles must use different "
                "end-effector control parts."
            )
        return _CoordinatedPickResources(
            left_arm=left_arm,
            right_arm=right_arm,
            left_hand=left_hand,
            right_hand=right_hand,
            left_hand_open_qpos=left_hand.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            left_hand_close_qpos=left_hand.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            right_hand_open_qpos=right_hand.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            right_hand_close_qpos=right_hand.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def _resolve_object_initial_pose(
        self,
        target: CoordinatedPickGoal,
        context: PlanningContext,
    ) -> torch.Tensor:
        if target.object_initial_pose is not None:
            return self._resolve_pose(
                resolve_pose_goal(
                    target.object_initial_pose,
                    context,
                    name="object_initial_pose",
                ),
                "object_initial_pose",
            )
        return self._resolve_pose(
            _resolve_object_pose(
                target.semantics,
                context,
                name="object_initial_pose",
            ),
            "object_initial_pose",
        )

    def _resolve_target(
        self,
        target: CoordinatedPickGoal,
        context: PlanningContext,
        options: CoordinatedPickmentOptions,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[HeldObjectState, HeldObjectState],
        torch.Tensor,
    ]:
        object_initial_pose = self._resolve_object_initial_pose(target, context)
        object_target_pose = self._resolve_pose(
            resolve_pose_goal(
                target.object_target_pose,
                context,
                name="object_target_pose",
            ),
            "object_target_pose",
        )
        left_grasp_xpos, right_grasp_xpos, grasp_success = (
            self._resolve_dual_arm_grasp_poses(
                target.semantics, object_initial_pose, options
            )
        )
        left_object_to_eef = torch.bmm(pose_inv(object_initial_pose), left_grasp_xpos)
        right_object_to_eef = torch.bmm(pose_inv(object_initial_pose), right_grasp_xpos)
        left_target_xpos = torch.bmm(object_target_pose, left_object_to_eef)
        right_target_xpos = torch.bmm(object_target_pose, right_object_to_eef)
        held_states = (
            HeldObjectState(
                semantics=target.semantics,
                object_to_eef=left_object_to_eef,
                grasp_xpos=left_grasp_xpos,
            ),
            HeldObjectState(
                semantics=target.semantics,
                object_to_eef=right_object_to_eef,
                grasp_xpos=right_grasp_xpos,
            ),
        )
        return (
            object_initial_pose,
            object_target_pose,
            left_grasp_xpos,
            right_grasp_xpos,
            left_target_xpos,
            right_target_xpos,
            held_states,
            grasp_success,
        )

    def _resolve_dual_arm_grasp_poses(
        self,
        semantics: ObjectSemantics,
        object_poses: torch.Tensor,
        options: CoordinatedPickmentOptions,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample left/right grasp poses from the target antipodal affordance.

        Args:
            semantics: Object semantics carrying an :class:`AntipodalAffordance`.
            object_poses: Object poses with shape ``(num_envs, 4, 4)``.
            options: Coordinated pickment options carrying the dual-arm and
                approach directions used by the affordance sampler.

        Returns:
            ``(left_grasp_xpos, right_grasp_xpos, success_mask)``. The grasp poses
            have shape ``(num_envs, 4, 4)`` and the success mask has shape
            ``(num_envs,)``. Environments without a valid left or right grasp hold
            the identity pose and are marked ``False``.
        """
        if not isinstance(semantics.affordance, AntipodalAffordance):
            raise ValueError(
                "CoordinatedPickment requires an AntipodalAffordance to sample "
                "dual-arm grasps."
            )
        num_envs = object_poses.shape[0]
        identity = torch.eye(4, dtype=torch.float32, device=self.device)
        left_grasp_xpos = identity.unsqueeze(0).repeat(num_envs, 1, 1)
        right_grasp_xpos = identity.unsqueeze(0).repeat(num_envs, 1, 1)
        success_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        approach_direction = options.approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        left_to_right_arm_direction = options.left_to_right_arm_direction.to(
            device=self.device, dtype=torch.float32
        )
        left_to_right_arm_direction = left_to_right_arm_direction / (
            torch.linalg.vector_norm(left_to_right_arm_direction).clamp_min(1.0e-6)
        )
        dual_results = semantics.affordance.get_dual_arm_valid_grasp_poses(
            obj_poses=object_poses,
            left_to_right_arm_direction=left_to_right_arm_direction,
            approach_direction=approach_direction,
            middle_empty_ratio=options.middle_empty_ratio,
        )
        for env_idx, result in enumerate(dual_results):
            if result is None:
                logger.log_warning(
                    f"Failed to sample dual-arm grasps for environment {env_idx}."
                )
                continue
            left_grasp = self._select_best_grasp(result["left"])
            right_grasp = self._select_best_grasp(result["right"])
            if left_grasp is None or right_grasp is None:
                logger.log_warning(
                    f"No valid left/right grasp for environment {env_idx}."
                )
                continue
            left_grasp_xpos[env_idx] = left_grasp.to(
                device=self.device, dtype=torch.float32
            )
            right_grasp_xpos[env_idx] = right_grasp.to(
                device=self.device, dtype=torch.float32
            )
            success_mask[env_idx] = True
        return left_grasp_xpos, right_grasp_xpos, success_mask

    @staticmethod
    def _select_best_grasp(arm_result: dict) -> torch.Tensor | None:
        """Return the lowest-cost grasp pose from one arm's sampler result.

        Args:
            arm_result: One ``"left"``/``"right"`` entry of the dict returned by
                :meth:`AntipodalAffordance.get_dual_arm_valid_grasp_poses`.

        Returns:
            The selected ``(4, 4)`` grasp pose, or ``None`` when the sampler
            reports no valid grasp for this arm.
        """
        if not arm_result.get("is_success", False):
            return None
        grasp_poses = arm_result["grasp_poses"].to(dtype=torch.float32)
        costs = arm_result["total_cost"].to(dtype=torch.float32)
        if grasp_poses.dim() == 2:
            # The sampler returns a single eye(4) placeholder when it finds no
            # valid pair; is_success should already cover this, but stay robust.
            grasp_poses = grasp_poses.unsqueeze(0)
            costs = costs.unsqueeze(0)
        if grasp_poses.shape[0] == 0:
            return None
        best_idx = torch.argmin(costs)
        if not torch.isfinite(costs[best_idx]):
            return None
        return grasp_poses[best_idx]

    def _compute_segment_lengths(
        self, sample_count: int, options: CoordinatedPickmentOptions
    ) -> dict[str, int]:
        """Split the invocation sample budget across coordinated-pick segments."""
        n_close = max(2, options.hand_interp_steps)
        n_hold = max(0, options.hold_steps)
        n_motion = sample_count - n_close - n_hold
        n_approach = n_motion // 3
        n_lift = n_motion // 3
        n_move = n_motion - n_approach - n_lift
        if min(n_approach, n_lift, n_move) < 2:
            raise ValueError(
                "Not enough waypoints for coordinated pickment. Please increase "
                "sample_count or decrease hand_interp_steps/hold_steps."
            )
        return {
            "approach": n_approach,
            "close": n_close,
            "lift": n_lift,
            "move": n_move,
            "hold": n_hold,
        }

    def _compute_pre_grasp_xpos(
        self, grasp_xpos: torch.Tensor, options: CoordinatedPickmentOptions
    ) -> torch.Tensor:
        grasp_z = grasp_xpos[:, :3, 2]
        return translate_pose_world(grasp_xpos, -grasp_z * options.pre_grasp_distance)

    def _select_motion_keyframe_indices(
        self, n_waypoints: int, options: CoordinatedPickmentOptions
    ) -> torch.Tensor:
        n_keyframes = min(max(2, options.object_motion_keyframes), n_waypoints)
        return (
            torch.linspace(
                0,
                n_waypoints - 1,
                steps=n_keyframes,
                device=self.device,
            )
            .round()
            .to(dtype=torch.long)
        )

    def _log_ik_failures(
        self,
        control_part: str,
        target_name: str,
        failed_mask: torch.Tensor,
    ) -> None:
        failed_env_ids = torch.nonzero(failed_mask, as_tuple=False).flatten().tolist()
        if failed_env_ids:
            logger.log_warning(
                f"Failed to compute IK for {control_part} {target_name} in "
                f"environment(s) {failed_env_ids}."
            )

    def _plan_masked_arm_trajectory(
        self,
        control_part: str,
        start_qpos: torch.Tensor,
        target_poses: torch.Tensor,
        n_waypoints: int,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_state = target_poses.shape[1]
        keyframe_qpos = torch.zeros(
            (self.num_envs, n_state, start_qpos.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        success_mask = active_mask.clone()
        qpos_seed = start_qpos
        for target_idx in range(n_state):
            ik_success, qpos = self.robot.compute_ik(
                pose=target_poses[:, target_idx],
                name=control_part,
                joint_seed=qpos_seed,
            )
            ik_success = normalize_success_mask(
                ik_success,
                num_envs=self.num_envs,
                device=self.device,
                name=f"IK success for {control_part} target state {target_idx}",
            )
            failed_mask = success_mask & ~ik_success
            self._log_ik_failures(
                control_part, f"target state {target_idx}", failed_mask
            )
            success_mask &= ik_success
            qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
            qpos_seed = torch.where(success_mask[:, None], qpos, qpos_seed)
            keyframe_qpos[:, target_idx] = qpos_seed

        keyframe_qpos = torch.cat([start_qpos.unsqueeze(1), keyframe_qpos], dim=1)
        trajectory = (
            interpolate_joint_trajectory(
                keyframe_qpos[:, 0], keyframe_qpos[:, -1], n_waypoints
            )
            if n_state == 1
            else self._interpolate_keyframe_qpos(keyframe_qpos, n_waypoints)
        )
        return success_mask, trajectory

    def _plan_synchronized_object_motion(
        self,
        left_start_qpos: torch.Tensor,
        right_start_qpos: torch.Tensor,
        object_pose_traj: torch.Tensor,
        left_object_to_eef: torch.Tensor,
        right_object_to_eef: torch.Tensor,
        active_mask: torch.Tensor,
        resources: _CoordinatedPickResources,
        options: CoordinatedPickmentOptions,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_waypoints = object_pose_traj.shape[1]
        keyframe_indices = self._select_motion_keyframe_indices(n_waypoints, options)
        left_traj = torch.zeros(
            (self.num_envs, len(keyframe_indices), left_start_qpos.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        right_traj = torch.zeros(
            (self.num_envs, len(keyframe_indices), right_start_qpos.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        left_qpos_seed = left_start_qpos
        right_qpos_seed = right_start_qpos
        success_mask = active_mask.clone()
        for keyframe_col, waypoint_idx in enumerate(keyframe_indices.tolist()):
            left_xpos = torch.bmm(object_pose_traj[:, waypoint_idx], left_object_to_eef)
            right_xpos = torch.bmm(
                object_pose_traj[:, waypoint_idx], right_object_to_eef
            )
            left_success, left_qpos = self.robot.compute_ik(
                pose=left_xpos,
                name=resources.left_arm.name,
                joint_seed=left_qpos_seed,
            )
            right_success, right_qpos = self.robot.compute_ik(
                pose=right_xpos,
                name=resources.right_arm.name,
                joint_seed=right_qpos_seed,
            )
            left_success = normalize_success_mask(
                left_success,
                num_envs=self.num_envs,
                device=self.device,
                name=(
                    f"IK success for {resources.left_arm.name} object waypoint "
                    f"{waypoint_idx}"
                ),
            )
            right_success = normalize_success_mask(
                right_success,
                num_envs=self.num_envs,
                device=self.device,
                name=(
                    f"IK success for {resources.right_arm.name} object waypoint "
                    f"{waypoint_idx}"
                ),
            )
            self._log_ik_failures(
                resources.left_arm.name,
                f"object waypoint {waypoint_idx}",
                success_mask & ~left_success,
            )
            self._log_ik_failures(
                resources.right_arm.name,
                f"object waypoint {waypoint_idx}",
                success_mask & ~right_success,
            )
            success_mask &= left_success & right_success
            left_qpos = torch.as_tensor(
                left_qpos, dtype=torch.float32, device=self.device
            )
            right_qpos = torch.as_tensor(
                right_qpos, dtype=torch.float32, device=self.device
            )
            left_qpos_seed = torch.where(
                success_mask[:, None], left_qpos, left_qpos_seed
            )
            right_qpos_seed = torch.where(
                success_mask[:, None], right_qpos, right_qpos_seed
            )
            left_traj[:, keyframe_col] = left_qpos_seed
            right_traj[:, keyframe_col] = right_qpos_seed

        return (
            success_mask,
            self._interpolate_qpos_keyframes(left_traj, keyframe_indices, n_waypoints),
            self._interpolate_qpos_keyframes(right_traj, keyframe_indices, n_waypoints),
        )

    def _plan(
        self,
        request: ResolvedActionRequest[CoordinatedPickGoal, CoordinatedPickmentOptions],
        context: PlanningContext,
    ) -> ActionPlan:
        """Plan a coordinated pick without committing the dual attachment."""
        target = request.goal
        options = request.skill_options
        resources = self._resolve_resources(request)
        if (
            request.motion_policy.strategy == "motion_gen"
            and self.motion_generator.planner.cfg.planner_type == "curobo"
        ):
            raise ValueError(
                "Coordinated dual-arm planning is not supported by the cuRobo backend."
            )
        state = context
        (
            object_initial_pose,
            object_target_pose,
            left_grasp_xpos,
            right_grasp_xpos,
            left_target_xpos,
            right_target_xpos,
            held_states,
            grasp_success,
        ) = self._resolve_target(target, context, options)
        left_held_state, right_held_state = held_states
        if not grasp_success.any():
            logger.log_warning("CoordinatedPickment failed to resolve dual-arm grasps.")
            return self.failed_plan(
                request,
                context,
                message="Failed to resolve dual-arm grasps.",
            )
        left_start_qpos, right_start_qpos = self._resolve_dual_arm_start(
            state, resources
        )
        segments = self._compute_segment_lengths(
            request.motion_policy.sample_count, options
        )
        left_pre_grasp_xpos = self._compute_pre_grasp_xpos(left_grasp_xpos, options)
        right_pre_grasp_xpos = self._compute_pre_grasp_xpos(right_grasp_xpos, options)
        left_approach_targets = torch.stack(
            [left_pre_grasp_xpos, left_grasp_xpos], dim=1
        )
        right_approach_targets = torch.stack(
            [right_pre_grasp_xpos, right_grasp_xpos], dim=1
        )
        success_mask = grasp_success.clone()
        success_mask, left_approach_traj = self._plan_masked_arm_trajectory(
            resources.left_arm.name,
            left_start_qpos,
            left_approach_targets,
            segments["approach"],
            success_mask,
        )
        success_mask, right_approach_traj = self._plan_masked_arm_trajectory(
            resources.right_arm.name,
            right_start_qpos,
            right_approach_targets,
            segments["approach"],
            success_mask,
        )

        left_grasp_qpos = left_approach_traj[:, -1]
        right_grasp_qpos = right_approach_traj[:, -1]
        approach_trajectory = self._assemble_segment(
            state,
            left_approach_traj,
            right_approach_traj,
            self._repeat_qpos(resources.left_hand_open_qpos, segments["approach"]),
            self._repeat_qpos(resources.right_hand_open_qpos, segments["approach"]),
            resources=resources,
        )

        close_trajectory = self._assemble_segment(
            state,
            self._repeat_qpos(left_grasp_qpos, segments["close"]),
            self._repeat_qpos(right_grasp_qpos, segments["close"]),
            self._interpolate_qpos(
                resources.left_hand_open_qpos,
                resources.left_hand_close_qpos,
                segments["close"],
            ),
            self._interpolate_qpos(
                resources.right_hand_open_qpos,
                resources.right_hand_close_qpos,
                segments["close"],
            ),
            resources=resources,
        )

        lift_object_pose = translate_pose_world(
            object_initial_pose,
            torch.tensor([0.0, 0.0, options.lift_height], device=self.device),
        )
        lift_object_traj = self._interpolate_object_pose(
            object_initial_pose,
            lift_object_pose,
            segments["lift"],
            include_orientation=False,
        )
        success_mask, left_lift_traj, right_lift_traj = (
            self._plan_synchronized_object_motion(
                left_grasp_qpos,
                right_grasp_qpos,
                lift_object_traj,
                left_held_state.object_to_eef,
                right_held_state.object_to_eef,
                success_mask,
                resources,
                options,
            )
        )

        left_lift_qpos = left_lift_traj[:, -1]
        right_lift_qpos = right_lift_traj[:, -1]
        lift_trajectory = self._assemble_segment(
            state,
            left_lift_traj,
            right_lift_traj,
            self._repeat_qpos(resources.left_hand_close_qpos, segments["lift"]),
            self._repeat_qpos(resources.right_hand_close_qpos, segments["lift"]),
            resources=resources,
        )

        move_object_traj = self._interpolate_object_pose(
            lift_object_pose,
            object_target_pose,
            segments["move"],
            include_orientation=True,
        )
        success_mask, left_move_traj, right_move_traj = (
            self._plan_synchronized_object_motion(
                left_lift_qpos,
                right_lift_qpos,
                move_object_traj,
                left_held_state.object_to_eef,
                right_held_state.object_to_eef,
                success_mask,
                resources,
                options,
            )
        )

        left_target_qpos = left_move_traj[:, -1]
        right_target_qpos = right_move_traj[:, -1]
        move_trajectory = self._assemble_segment(
            state,
            left_move_traj,
            right_move_traj,
            self._repeat_qpos(resources.left_hand_close_qpos, segments["move"]),
            self._repeat_qpos(resources.right_hand_close_qpos, segments["move"]),
            resources=resources,
        )

        hold_trajectory = torch.empty(
            (self.num_envs, 0, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        if segments["hold"] > 0:
            hold_trajectory = self._assemble_segment(
                state,
                self._repeat_qpos(left_target_qpos, segments["hold"]),
                self._repeat_qpos(right_target_qpos, segments["hold"]),
                self._repeat_qpos(resources.left_hand_close_qpos, segments["hold"]),
                self._repeat_qpos(resources.right_hand_close_qpos, segments["hold"]),
                resources=resources,
            )

        full = torch.cat(
            [
                approach_trajectory,
                close_trajectory,
                lift_trajectory,
                move_trajectory,
                hold_trajectory,
            ],
            dim=1,
        )
        left_held_object = HeldObjectState(
            semantics=left_held_state.semantics,
            object_to_eef=left_held_state.object_to_eef,
            grasp_xpos=left_target_xpos,
        )
        right_held_object = HeldObjectState(
            semantics=right_held_state.semantics,
            object_to_eef=right_held_state.object_to_eef,
            grasp_xpos=right_target_xpos,
        )
        return self.build_plan(
            request,
            context,
            success=success_mask,
            trajectory=TimedTrajectory.from_uniform_step(
                full,
                env_ids=context.env_ids,
                step_dt=context.require_control_dt(),
            ),
            expected_effects=StateDelta(
                held_object_updates={
                    resources.left_arm.name: left_held_object,
                    resources.right_arm.name: right_held_object,
                },
            ),
            segment_lengths={
                "approach": approach_trajectory.shape[1],
                "close": close_trajectory.shape[1],
                "lift": lift_trajectory.shape[1],
                "move": move_trajectory.shape[1],
                "hold": hold_trajectory.shape[1],
            },
        )


__all__ = [
    "CoordinatedPickGoal",
    "CoordinatedPickment",
    "CoordinatedPickmentOptions",
]
