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
import math
from typing import ClassVar

import torch

from embodichain.toolkits.graspkit import ParallelJawGraspPoseGenerator
from embodichain.utils import logger
from embodichain.utils.math import matrix_from_quat, pose_inv, quat_from_matrix

from embodichain.lab.sim.atomic_actions.affordance import AntipodalAffordance
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
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
    BATCH_INVERSE_KINEMATICS_CAPABILITY,
    DisjointResourceSlots,
    INVERSE_KINEMATICS_CAPABILITY,
    SkillBindingContract,
)
from embodichain.lab.sim.atomic_actions.state import HeldObjectState, PlanningContext
from embodichain.lab.sim.atomic_actions.trajectory_ops import (
    interpolate_joint_trajectory,
    translate_pose_world,
)
from embodichain.lab.sim.atomic_actions.primitives._helpers import (
    assemble_full_robot_trajectory,
    require_shared_task_state_key,
    repeat_qpos,
    resolve_batched_pose,
)
from embodichain.lab.sim.atomic_actions.primitives._binding_contracts import (
    make_manipulation_slot,
)


@dataclass(frozen=True, slots=True, eq=False)
class CoordinatedPickGoal(ObjectActionGoal):
    """Object-centric target for picking and moving one object with two hands.

    The left/right grasp poses are not supplied by the caller; they are sampled
    by the parallel-jaw grasp-pose service at planning time using the dual-arm
    direction and approach direction declared on
    :class:`CoordinatedPickmentOptions`.
    """

    object_target_pose: PoseGoalValue
    """Target pose for the shared object, shape ``(4, 4)`` or ``(num_envs, 4, 4)``."""

    object_initial_pose: PoseGoalValue | None = None
    """Optional initial object pose.

    When omitted, the pose is grounded through the semantic object's stable
    scene identity.
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

    Left/right grasps are sampled from target-local affordance geometry by the
    engine's parallel-jaw grasp-pose service. The dual-arm direction splits the
    object into left/right grasp regions and the approach direction filters the
    sampled antipodal pairs.
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

    release: bool = False
    """Whether both hands open after reaching the shared object target pose."""

    release_steps: int = 10
    """Number of waypoints used for the simultaneous hand-open segment."""

    retreat_distance: float = 0.08
    """World-Z retreat distance after a coordinated release."""

    retreat_steps: int = 12
    """Number of waypoints used for the simultaneous post-release retreat."""

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

    grasp_seed: int = 17_393
    """Deterministic seed isolated around coordinated grasp sampling."""

    def __post_init__(self) -> None:
        if self.object_motion_keyframes < 2:
            raise ValueError("object_motion_keyframes must be at least 2.")
        if self.pre_grasp_distance < 0.0:
            raise ValueError("pre_grasp_distance must be non-negative.")
        if self.lift_height < 0.0:
            raise ValueError("lift_height must be non-negative.")
        if not isinstance(self.release, bool):
            raise TypeError("release must be a bool.")
        if self.retreat_distance < 0.0:
            raise ValueError("retreat_distance must be non-negative.")
        for name in (
            "hand_interp_steps",
            "hold_steps",
            "release_steps",
            "retreat_steps",
        ):
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
        if type(self.grasp_seed) is not int or self.grasp_seed < 0:
            raise ValueError("grasp_seed must be a non-negative integer.")


@dataclass(frozen=True, slots=True, eq=False)
class _CoordinatedPickResources:
    """Invocation-bound control parts and compatible hand commands."""

    left_task_state_key: str
    right_task_state_key: str
    left_arm: JointPositionTarget
    right_arm: JointPositionTarget
    left_hand: JointPositionTarget
    right_hand: JointPositionTarget
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
    _MAX_REACHABILITY_CANDIDATES: ClassVar[int] = 32
    binding_contract: ClassVar[SkillBindingContract] = SkillBindingContract(
        slots=tuple(
            make_manipulation_slot(
                role,
                motion_capabilities=frozenset(
                    {
                        INVERSE_KINEMATICS_CAPABILITY,
                        BATCH_INVERSE_KINEMATICS_CAPABILITY,
                    }
                ),
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
        left_motion = binding.endpoint("left", "motion")
        right_motion = binding.endpoint("right", "motion")
        left_grasp = binding.endpoint("left", "grasp")
        right_grasp = binding.endpoint("right", "grasp")
        left_arm = left_motion.require_target(JointPositionTarget)
        right_arm = right_motion.require_target(JointPositionTarget)
        left_hand = left_grasp.require_target(JointPositionTarget)
        right_hand = right_grasp.require_target(JointPositionTarget)
        left_task_state_key = require_shared_task_state_key(
            left_motion,
            left_grasp,
            participant="CoordinatedPickment left participant",
        )
        right_task_state_key = require_shared_task_state_key(
            right_motion,
            right_grasp,
            participant="CoordinatedPickment right participant",
        )
        if left_task_state_key == right_task_state_key:
            raise ValueError(
                "CoordinatedPickment left and right participants must use "
                "different task_state_key values."
            )
        if left_arm.control_part == right_arm.control_part:
            raise ValueError(
                "CoordinatedPickment left and right roles must use different "
                "manipulator control parts."
            )
        if left_hand.control_part == right_hand.control_part:
            raise ValueError(
                "CoordinatedPickment left and right roles must use different "
                "end-effector control parts."
            )
        return _CoordinatedPickResources(
            left_task_state_key=left_task_state_key,
            right_task_state_key=right_task_state_key,
            left_arm=left_arm,
            right_arm=right_arm,
            left_hand=left_hand,
            right_hand=right_hand,
            left_hand_open_qpos=left_grasp.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            left_hand_close_qpos=left_grasp.joint_positions(
                GRASP_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            right_hand_open_qpos=right_grasp.joint_positions(
                OPEN_COMMAND,
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ),
            right_hand_close_qpos=right_grasp.joint_positions(
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
        resources: _CoordinatedPickResources,
        left_start_qpos: torch.Tensor,
        right_start_qpos: torch.Tensor,
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
                target.semantics,
                object_initial_pose,
                object_target_pose,
                options,
                resources.left_hand.target_id,
                resources.right_hand.target_id,
                left_start_qpos,
                right_start_qpos,
                resources.left_arm.control_part,
                resources.right_arm.control_part,
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
        object_target_poses: torch.Tensor,
        options: CoordinatedPickmentOptions,
        left_grasp_target_id: str,
        right_grasp_target_id: str,
        left_start_qpos: torch.Tensor,
        right_start_qpos: torch.Tensor,
        left_control_part: str,
        right_control_part: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample left/right grasp poses from the target antipodal affordance.

        Args:
            semantics: Object semantics carrying an :class:`AntipodalAffordance`.
            object_poses: Object poses with shape ``(num_envs, 4, 4)``.
            object_target_poses: Requested terminal object poses with shape
                ``(num_envs, 4, 4)``.
            options: Coordinated pickment options carrying the dual-arm and
                approach directions used by the grasp-pose generator.
            left_grasp_target_id: Left grasp endpoint target ID.
            right_grasp_target_id: Right grasp endpoint target ID.
            left_start_qpos: Current left-arm joint positions.
            right_start_qpos: Current right-arm joint positions.
            left_control_part: Bound left-arm control-part name.
            right_control_part: Bound right-arm control-part name.

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
        approach_direction = options.approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        left_to_right_arm_direction = options.left_to_right_arm_direction.to(
            device=self.device, dtype=torch.float32
        )
        left_to_right_arm_direction = left_to_right_arm_direction / (
            torch.linalg.vector_norm(left_to_right_arm_direction).clamp_min(1.0e-6)
        )
        left_generator = self.planning_services.grasp_pose_generator(
            left_grasp_target_id
        )
        right_generator = self.planning_services.grasp_pose_generator(
            right_grasp_target_id
        )
        if not isinstance(
            left_generator, ParallelJawGraspPoseGenerator
        ) or not isinstance(right_generator, ParallelJawGraspPoseGenerator):
            raise TypeError(
                "CoordinatedPickment requires ParallelJawGraspPoseGenerator "
                "services for both grasp endpoints."
            )
        if left_generator.gripper_model != right_generator.gripper_model:
            raise ValueError(
                "CoordinatedPickment requires matching left and right parallel-jaw "
                "gripper geometry."
            )
        partition_ratios = self._candidate_middle_empty_ratios(
            semantics.affordance,
            object_poses,
            left_to_right_arm_direction,
            base_ratio=options.middle_empty_ratio,
        )
        approach_directions = self._candidate_approach_directions(
            semantics.affordance,
            object_poses,
            left_to_right_arm_direction,
            requested=approach_direction,
        )
        identity = torch.eye(4, dtype=torch.float32, device=self.device).repeat(
            self.num_envs,
            1,
            1,
        )
        left_grasp_xpos = identity.clone()
        right_grasp_xpos = identity.clone()
        success_mask = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        sampling_device = torch.device(self.device)
        cuda_devices = (
            [
                (
                    torch.cuda.current_device()
                    if sampling_device.index is None
                    else sampling_device.index
                )
            ]
            if sampling_device.type == "cuda"
            else []
        )
        selected_candidate: tuple[int, int] | None = None
        for approach_index, candidate_approach in enumerate(approach_directions):
            for partition_index, partition_ratio in enumerate(partition_ratios):
                # GraspKit perturbs approach directions while building candidates.
                # Isolate that randomness so retries and alternative partitions do
                # not consume or depend on application-global RNG state.  Reusing
                # one seed also keeps the sampled surface realization fixed while
                # geometry policies are the only variables under evaluation.
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(options.grasp_seed)
                    if cuda_devices:
                        torch.cuda.manual_seed_all(options.grasp_seed)
                    dual_results = left_generator.get_dual_arm_valid_grasp_poses(
                        mesh_vertices=semantics.affordance.mesh_vertices,
                        mesh_triangles=semantics.affordance.mesh_triangles,
                        obj_poses=object_poses,
                        left_to_right_arm_direction=left_to_right_arm_direction,
                        approach_direction=candidate_approach,
                        middle_empty_ratio=partition_ratio,
                    )
                candidate_left, left_success = self._select_reachable_arm_grasp(
                    dual_results,
                    role="left",
                    object_poses=object_poses,
                    object_target_poses=object_target_poses,
                    start_qpos=left_start_qpos,
                    control_part=left_control_part,
                    options=options,
                    log_failure=False,
                )
                candidate_right, right_success = self._select_reachable_arm_grasp(
                    dual_results,
                    role="right",
                    object_poses=object_poses,
                    object_target_poses=object_target_poses,
                    start_qpos=right_start_qpos,
                    control_part=right_control_part,
                    options=options,
                    log_failure=False,
                )
                selected = ~success_mask & left_success & right_success
                left_grasp_xpos = torch.where(
                    selected[:, None, None],
                    candidate_left,
                    left_grasp_xpos,
                )
                right_grasp_xpos = torch.where(
                    selected[:, None, None],
                    candidate_right,
                    right_grasp_xpos,
                )
                success_mask |= selected
                if bool(selected.any().item()) and selected_candidate is None:
                    selected_candidate = (approach_index, partition_index)
                if success_mask.all():
                    break
            if success_mask.all():
                break
        if not success_mask.all():
            failed = torch.nonzero(~success_mask, as_tuple=False).flatten().tolist()
            logger.log_warning(
                "No jointly reachable coordinated grasp for environment(s) "
                f"{failed}; tried {len(approach_directions)} approach directions "
                f"and middle-empty ratios {list(partition_ratios)}."
            )
        elif selected_candidate is not None:
            approach_index, partition_index = selected_candidate
            logger.log_info(
                "Selected coordinated grasp candidate with approach direction "
                f"{approach_directions[approach_index].detach().cpu().tolist()} "
                f"and middle-empty ratio {partition_ratios[partition_index]}."
            )
        return left_grasp_xpos, right_grasp_xpos, success_mask

    @staticmethod
    def _candidate_middle_empty_ratios(
        affordance: AntipodalAffordance,
        object_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        *,
        base_ratio: float,
    ) -> tuple[float, ...]:
        """Rank deterministic dual-grasp partitions from live object geometry.

        The live-axis candidate respects the current object orientation.  An
        axis-aligned geometry candidate remains in the bounded search because
        a rotated container can otherwise make the projected span too narrow
        for both grippers even though its physical end regions remain usable.
        """
        vertices = affordance.mesh_vertices
        if (
            not isinstance(vertices, torch.Tensor)
            or vertices.dim() != 2
            or vertices.shape[0] < 3
            or vertices.shape[1] != 3
            or not bool(torch.isfinite(vertices).all().item())
        ):
            return (float(base_ratio),)
        local_vertices = vertices.to(
            device=object_poses.device,
            dtype=object_poses.dtype,
        )
        centered = local_vertices - local_vertices.mean(dim=0, keepdim=True)
        covariance = (
            centered.transpose(0, 1) @ centered / float(local_vertices.shape[0])
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        second = float(eigenvalues[-2].clamp_min(1.0e-12).item())
        longest = float(eigenvalues[-1].clamp_min(1.0e-12).item())
        elongation_ratio = math.sqrt(longest / second)
        elongation_confidence = min(
            1.0,
            max(0.0, (elongation_ratio - 1.0) / 1.5),
        )
        world_axes = torch.matmul(object_poses[:, :3, :3], eigenvectors)
        principal_world = world_axes[:, :, -1]
        principal_world = principal_world / torch.linalg.vector_norm(
            principal_world,
            dim=1,
            keepdim=True,
        ).clamp_min(1.0e-6)
        arm_alignment = torch.abs(
            torch.sum(
                principal_world * left_to_right_arm_direction[None],
                dim=1,
            )
        )
        geometric_ratio = 0.25 + 0.45 * float(arm_alignment.mean().item())
        preferred_ratio = (1.0 - elongation_confidence) * float(
            base_ratio
        ) + elongation_confidence * geometric_ratio
        axis_aligned_ratio = (1.0 - elongation_confidence) * float(
            base_ratio
        ) + elongation_confidence * 0.70
        ratios: list[float] = []
        for raw_ratio in (
            preferred_ratio,
            axis_aligned_ratio,
            float(base_ratio),
            preferred_ratio - 0.15,
            preferred_ratio + 0.15,
        ):
            ratio = min(0.90, max(0.05, raw_ratio))
            if not any(abs(ratio - existing) <= 1.0e-6 for existing in ratios):
                ratios.append(ratio)
        return tuple(ratios)

    @staticmethod
    def _candidate_approach_directions(
        affordance: AntipodalAffordance,
        object_poses: torch.Tensor,
        left_to_right_arm_direction: torch.Tensor,
        *,
        requested: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Build a bounded world-frame approach search for coordinated grasps."""
        candidates: list[torch.Tensor] = []

        def add(direction: torch.Tensor) -> None:
            value = direction.to(device=object_poses.device, dtype=object_poses.dtype)
            norm = torch.linalg.vector_norm(value)
            if not bool(torch.isfinite(value).all().item()) or float(norm) <= 1.0e-6:
                return
            value = value / norm
            if any(
                float(torch.dot(value, existing).item()) >= 1.0 - 1.0e-5
                for existing in candidates
            ):
                return
            candidates.append(value)

        down = requested.new_tensor([0.0, 0.0, -1.0])
        add(requested)
        add(down)
        horizontal_arm = left_to_right_arm_direction.clone()
        horizontal_arm[2] = 0.0
        horizontal_norm = torch.linalg.vector_norm(horizontal_arm)
        if float(horizontal_norm) > 1.0e-6:
            horizontal_arm = horizontal_arm / horizontal_norm
            robot_forward = torch.stack(
                (-horizontal_arm[1], horizontal_arm[0], horizontal_arm.new_tensor(0.0))
            )
            add(robot_forward + down)
            add(-robot_forward + down)
            add(robot_forward)
            add(-robot_forward)

        vertices = affordance.mesh_vertices
        if (
            isinstance(vertices, torch.Tensor)
            and vertices.dim() == 2
            and vertices.shape[0] >= 3
            and vertices.shape[1] == 3
            and bool(torch.isfinite(vertices).all().item())
        ):
            local_vertices = vertices.to(
                device=object_poses.device,
                dtype=object_poses.dtype,
            )
            centered = local_vertices - local_vertices.mean(dim=0, keepdim=True)
            covariance = (
                centered.transpose(0, 1) @ centered / float(local_vertices.shape[0])
            )
            _, eigenvectors = torch.linalg.eigh(covariance)
            world_axes = torch.matmul(object_poses[:, :3, :3], eigenvectors)
            for axis_index in range(3):
                axes = world_axes[:, :, axis_index]
                if float(torch.mean(torch.abs(axes[:, 2])).item()) > 0.75:
                    continue
                reference = axes[0]
                consistency = torch.abs(torch.matmul(axes, reference))
                if bool((consistency < 0.90).any().item()):
                    continue
                add(reference + down)
                add(-reference + down)
                add(reference)
                add(-reference)
                break
        return tuple(candidates)

    def _select_reachable_arm_grasp(
        self,
        dual_results: list[dict[str, dict[str, object]] | None],
        *,
        role: str,
        object_poses: torch.Tensor,
        object_target_poses: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        options: CoordinatedPickmentOptions,
        log_failure: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select the lowest-cost candidate with a reachable transport route."""
        live_eef_pose = self.robot.compute_fk(
            qpos=start_qpos,
            name=control_part,
            to_matrix=True,
        )
        live_eef_pose = torch.as_tensor(
            live_eef_pose,
            dtype=torch.float32,
            device=self.device,
        )
        if live_eef_pose.shape != (self.num_envs, 4, 4):
            raise ValueError(
                f"Current {control_part} pose must have shape "
                f"({self.num_envs}, 4, 4), got {tuple(live_eef_pose.shape)}."
            )
        candidates, costs, sampled = self._ranked_arm_grasp_candidates(
            dual_results,
            role=role,
            live_eef_pose=live_eef_pose,
        )
        candidate_count = candidates.shape[1]
        object_to_eef = torch.matmul(
            pose_inv(object_poses)[:, None],
            candidates,
        )
        pre_grasp = candidates.clone()
        pre_grasp[..., :3, 3] -= pre_grasp[..., :3, 2] * options.pre_grasp_distance
        lifted_object = translate_pose_world(
            object_poses,
            torch.tensor(
                [0.0, 0.0, options.lift_height],
                dtype=object_poses.dtype,
                device=self.device,
            ),
        )
        lifted_eef = torch.matmul(lifted_object[:, None], object_to_eef)
        stages: list[tuple[str, torch.Tensor]] = [
            ("pre_grasp", pre_grasp),
            ("grasp", candidates),
            ("lift", lifted_eef),
        ]

        # Screen the transport with the same continuation used by the final
        # synchronized plan.  Solving the terminal pose directly from the lift
        # seed can reject a reachable route when a sparse IK sampler needs the
        # intermediate solutions to stay on one joint-space branch.
        transport_keyframes = self._interpolate_object_pose(
            lifted_object,
            object_target_poses,
            max(2, options.object_motion_keyframes),
            include_orientation=True,
        )
        for keyframe_index in range(1, transport_keyframes.shape[1]):
            stages.append(
                (
                    f"transport_{keyframe_index}",
                    torch.matmul(
                        transport_keyframes[:, keyframe_index, None],
                        object_to_eef,
                    ),
                )
            )
        target_eef = stages[-1][1]
        if options.release and options.retreat_distance > 0.0:
            retreat_eef = target_eef.clone()
            retreat_eef[..., 2, 3] += options.retreat_distance
            stages.append(("retreat", retreat_eef))

        seed = start_qpos[:, None, :].expand(-1, candidate_count, -1).clone()
        feasible = sampled.clone()
        stage_counts: dict[str, list[int]] = {}
        for stage_name, stage_poses in stages:
            result = self.robot.compute_batch_ik(
                pose=stage_poses,
                name=control_part,
                joint_seed=seed,
            )
            if type(result) is not tuple or len(result) != 2:
                raise TypeError(
                    "CoordinatedPickment batch IK must return (success, qpos)."
                )
            stage_success = torch.as_tensor(
                result[0],
                dtype=torch.bool,
                device=self.device,
            )
            stage_qpos = torch.as_tensor(
                result[1],
                dtype=torch.float32,
                device=self.device,
            )
            if stage_success.shape != sampled.shape:
                raise ValueError(
                    f"Batch IK success for {control_part} {stage_name} must have "
                    f"shape {tuple(sampled.shape)}, got {tuple(stage_success.shape)}."
                )
            if stage_qpos.shape != seed.shape:
                raise ValueError(
                    f"Batch IK qpos for {control_part} {stage_name} must have "
                    f"shape {tuple(seed.shape)}, got {tuple(stage_qpos.shape)}."
                )
            feasible &= stage_success
            seed = torch.where(feasible[..., None], stage_qpos, seed)
            stage_counts[stage_name] = feasible.sum(dim=1).tolist()

        feasible_costs = torch.where(
            feasible,
            costs,
            torch.full_like(costs, torch.inf),
        )
        best_cost, best_index = feasible_costs.min(dim=1)
        success = torch.isfinite(best_cost)
        env_index = torch.arange(candidates.shape[0], device=self.device)
        selected = candidates[env_index, best_index]
        if log_failure and not success.all():
            failed = torch.nonzero(~success, as_tuple=False).flatten().tolist()
            logger.log_warning(
                f"CoordinatedPickment {role} route screening failed for "
                f"environment(s) {failed}: sampled={sampled.sum(dim=1).tolist()}, "
                f"reachable={stage_counts}."
            )
        return selected, success

    def _ranked_arm_grasp_candidates(
        self,
        dual_results: list[dict[str, dict[str, object]] | None],
        *,
        role: str,
        live_eef_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad wrist-canonicalized candidates in ascending total-cost order."""
        if role not in {"left", "right"}:
            raise ValueError("Coordinated grasp role must be 'left' or 'right'.")
        if live_eef_pose.shape != (len(dual_results), 4, 4):
            raise ValueError(
                "live_eef_pose must provide one 4x4 pose per dual-grasp result."
            )
        ranked: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_candidates = 0
        for row_index, result in enumerate(dual_results):
            if result is None:
                poses = torch.empty((0, 4, 4), device=self.device)
                candidate_costs = torch.empty((0,), device=self.device)
            else:
                arm_result = result.get(role)
                if not isinstance(arm_result, dict) or not arm_result.get(
                    "is_success", False
                ):
                    poses = torch.empty((0, 4, 4), device=self.device)
                    candidate_costs = torch.empty((0,), device=self.device)
                else:
                    poses = torch.as_tensor(
                        arm_result["grasp_poses"],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    candidate_costs = torch.as_tensor(
                        arm_result["total_cost"],
                        dtype=torch.float32,
                        device=self.device,
                    ).reshape(-1)
                    if poses.shape == (4, 4):
                        poses = poses.unsqueeze(0)
                    if poses.dim() != 3 or poses.shape[1:] != (4, 4):
                        raise ValueError(
                            f"Coordinated {role} grasp poses must have shape "
                            f"(N, 4, 4), got {tuple(poses.shape)}."
                        )
                    if poses.shape[0] != candidate_costs.shape[0]:
                        raise ValueError(
                            f"Coordinated {role} grasp poses and costs must have "
                            "equal candidate counts."
                        )
                    finite = torch.isfinite(candidate_costs)
                    poses, wrist_rotation_cost = self._canonicalize_parallel_jaw_poses(
                        poses[finite],
                        live_eef_pose[row_index],
                    )
                    candidate_costs = (
                        candidate_costs[finite] + wrist_rotation_cost / math.pi
                    )
                    order = torch.argsort(candidate_costs)
                    poses = poses[order][: self._MAX_REACHABILITY_CANDIDATES]
                    candidate_costs = candidate_costs[order][
                        : self._MAX_REACHABILITY_CANDIDATES
                    ]
            ranked.append((poses, candidate_costs))
            max_candidates = max(max_candidates, poses.shape[0])

        padded_count = max(1, max_candidates)
        identity = torch.eye(4, dtype=torch.float32, device=self.device)
        poses = identity.repeat(len(ranked), padded_count, 1, 1)
        costs = torch.full(
            (len(ranked), padded_count),
            torch.inf,
            dtype=torch.float32,
            device=self.device,
        )
        sampled = torch.zeros_like(costs, dtype=torch.bool)
        for env_index, (env_poses, env_costs) in enumerate(ranked):
            count = env_poses.shape[0]
            if count == 0:
                continue
            poses[env_index, :count] = env_poses
            costs[env_index, :count] = env_costs
            sampled[env_index, :count] = True
        return poses, costs, sampled

    @staticmethod
    def _canonicalize_parallel_jaw_poses(
        poses: torch.Tensor,
        live_eef_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose each grasp's local-z half-turn nearest the live wrist.

        A parallel-jaw grasp is physically unchanged by a 180-degree rotation
        around its TCP z axis.  Canonicalizing that symmetry before IK avoids
        rejecting an otherwise reachable top-down grasp solely because the
        sampler returned the opposite wrist roll.
        """
        if poses.dim() != 3 or poses.shape[1:] != (4, 4):
            raise ValueError("poses must have shape (N, 4, 4).")
        if live_eef_pose.shape != (4, 4):
            raise ValueError("live_eef_pose must have shape (4, 4).")
        half_turn = torch.eye(
            4,
            dtype=poses.dtype,
            device=poses.device,
        )
        half_turn[0, 0] = -1.0
        half_turn[1, 1] = -1.0
        alternatives = torch.matmul(poses, half_turn)
        live_rotation = (
            live_eef_pose[:3, :3]
            .unsqueeze(0)
            .expand(
                poses.shape[0],
                -1,
                -1,
            )
        )

        def rotation_distance(candidate: torch.Tensor) -> torch.Tensor:
            relative = torch.matmul(
                live_rotation.transpose(-1, -2),
                candidate[:, :3, :3],
            )
            cosine = (
                torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
            ) * 0.5
            return torch.acos(torch.clamp(cosine, -1.0, 1.0))

        original_cost = rotation_distance(poses)
        alternative_cost = rotation_distance(alternatives)
        use_alternative = alternative_cost < original_cost
        return (
            torch.where(use_alternative[:, None, None], alternatives, poses),
            torch.where(use_alternative, alternative_cost, original_cost),
        )

    def _compute_segment_lengths(
        self, sample_count: int, options: CoordinatedPickmentOptions
    ) -> dict[str, int]:
        """Split the invocation sample budget across coordinated-pick segments."""
        n_close = max(2, options.hand_interp_steps)
        n_hold = max(0, options.hold_steps)
        n_release = max(2, options.release_steps) if options.release else 0
        n_retreat = max(2, options.retreat_steps) if options.release else 0
        n_motion = sample_count - n_close - n_hold - n_release - n_retreat
        n_approach = n_motion // 3
        n_lift = n_motion // 3
        n_move = n_motion - n_approach - n_lift
        if min(n_approach, n_lift, n_move) < 2:
            raise ValueError(
                "Not enough waypoints for coordinated pickment. Please increase "
                "sample_count or decrease close/hold/release/retreat steps."
            )
        return {
            "approach": n_approach,
            "close": n_close,
            "lift": n_lift,
            "move": n_move,
            "hold": n_hold,
            "release": n_release,
            "retreat": n_retreat,
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
                name=resources.left_arm.control_part,
                joint_seed=left_qpos_seed,
            )
            right_success, right_qpos = self.robot.compute_ik(
                pose=right_xpos,
                name=resources.right_arm.control_part,
                joint_seed=right_qpos_seed,
            )
            left_success = normalize_success_mask(
                left_success,
                num_envs=self.num_envs,
                device=self.device,
                name=(
                    f"IK success for {resources.left_arm.control_part} object waypoint "
                    f"{waypoint_idx}"
                ),
            )
            right_success = normalize_success_mask(
                right_success,
                num_envs=self.num_envs,
                device=self.device,
                name=(
                    f"IK success for {resources.right_arm.control_part} object waypoint "
                    f"{waypoint_idx}"
                ),
            )
            self._log_ik_failures(
                resources.left_arm.control_part,
                f"object waypoint {waypoint_idx}",
                success_mask & ~left_success,
            )
            self._log_ik_failures(
                resources.right_arm.control_part,
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
        left_start_qpos, right_start_qpos = self._resolve_dual_arm_start(
            state, resources
        )
        (
            object_initial_pose,
            object_target_pose,
            left_grasp_xpos,
            right_grasp_xpos,
            left_target_xpos,
            right_target_xpos,
            held_states,
            grasp_success,
        ) = self._resolve_target(
            target,
            context,
            options,
            resources,
            left_start_qpos,
            right_start_qpos,
        )
        left_held_state, right_held_state = held_states
        if not grasp_success.any():
            logger.log_warning("CoordinatedPickment failed to resolve dual-arm grasps.")
            return self.failed_plan(
                request,
                context,
                message="Failed to resolve dual-arm grasps.",
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
            resources.left_arm.control_part,
            left_start_qpos,
            left_approach_targets,
            segments["approach"],
            success_mask,
        )
        success_mask, right_approach_traj = self._plan_masked_arm_trajectory(
            resources.right_arm.control_part,
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

        release_trajectory = torch.empty(
            (self.num_envs, 0, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        retreat_trajectory = torch.empty(
            (self.num_envs, 0, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        if options.release:
            release_trajectory = self._assemble_segment(
                state,
                self._repeat_qpos(left_target_qpos, segments["release"]),
                self._repeat_qpos(right_target_qpos, segments["release"]),
                self._interpolate_qpos(
                    resources.left_hand_close_qpos,
                    resources.left_hand_open_qpos,
                    segments["release"],
                ),
                self._interpolate_qpos(
                    resources.right_hand_close_qpos,
                    resources.right_hand_open_qpos,
                    segments["release"],
                ),
                resources=resources,
            )
            retreat_delta = torch.tensor(
                [0.0, 0.0, options.retreat_distance],
                dtype=torch.float32,
                device=self.device,
            )
            left_retreat_xpos = translate_pose_world(left_target_xpos, retreat_delta)
            right_retreat_xpos = translate_pose_world(right_target_xpos, retreat_delta)
            success_mask, left_retreat_traj = self._plan_masked_arm_trajectory(
                resources.left_arm.control_part,
                left_target_qpos,
                left_retreat_xpos.unsqueeze(1),
                segments["retreat"],
                success_mask,
            )
            success_mask, right_retreat_traj = self._plan_masked_arm_trajectory(
                resources.right_arm.control_part,
                right_target_qpos,
                right_retreat_xpos.unsqueeze(1),
                segments["retreat"],
                success_mask,
            )
            retreat_trajectory = self._assemble_segment(
                state,
                left_retreat_traj,
                right_retreat_traj,
                self._repeat_qpos(resources.left_hand_open_qpos, segments["retreat"]),
                self._repeat_qpos(resources.right_hand_open_qpos, segments["retreat"]),
                resources=resources,
            )

        full = torch.cat(
            [
                approach_trajectory,
                close_trajectory,
                lift_trajectory,
                move_trajectory,
                hold_trajectory,
                release_trajectory,
                retreat_trajectory,
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
                    resources.left_task_state_key: (
                        None if options.release else left_held_object
                    ),
                    resources.right_task_state_key: (
                        None if options.release else right_held_object
                    ),
                },
            ),
            segment_lengths={
                "approach": approach_trajectory.shape[1],
                "close": close_trajectory.shape[1],
                "lift": lift_trajectory.shape[1],
                "move": move_trajectory.shape[1],
                "hold": hold_trajectory.shape[1],
                "release": release_trajectory.shape[1],
                "retreat": retreat_trajectory.shape[1],
            },
            # The approach has three evenly spaced keyframes: current,
            # pre-grasp, and grasp.  Contact may move the coordinated object
            # during the pre-grasp-to-grasp leg, so stop treating that
            # expected self-motion as an external scene revision once both
            # grippers reach the pre-grasp keyframe.  Independent late-bound
            # destination dependencies remain monitored for transport.
            scene_dependency_monitor_until=(
                {}
                if (
                    target.object_initial_pose is not None
                    or target.semantics.entity_id is None
                )
                else {
                    target.semantics.entity_id: max(
                        1,
                        math.ceil(approach_trajectory.shape[1] / 2),
                    )
                }
            ),
        )


__all__ = [
    "CoordinatedPickGoal",
    "CoordinatedPickment",
    "CoordinatedPickmentOptions",
]
