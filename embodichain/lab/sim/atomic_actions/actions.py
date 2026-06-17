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

from __future__ import annotations

import torch
from typing import Optional, Union, TYPE_CHECKING

from embodichain.lab.sim.planners import PlanResult, PlanState, MoveType
from embodichain.lab.sim.planners.motion_generator import MotionGenOptions
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
from .core import (
    AtomicAction,
    ObjectSemantics,
    AntipodalAffordance,
    ActionCfg,
    HeldObjectState,
    PlaceTarget,
)
from embodichain.utils import logger
from embodichain.utils import configclass
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
import numpy as np

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator
    from embodichain.lab.sim.objects import Robot


@configclass
class MoveActionCfg(ActionCfg):
    name: str = "move"
    """Name of the action, used for identification and logging."""

    sample_interval: int = 50
    """Number of waypoints to sample for the motion trajectory. Should be large enough to ensure smooth motion, but not too large to cause unnecessary computation overhead."""


@configclass
class GraspActionCfg(MoveActionCfg):
    """Shared configuration for actions that involve gripper open/close motions."""

    hand_open_qpos: torch.Tensor | None = None
    """[hand_dof,] of float. Joint positions for open hand state."""

    hand_close_qpos: torch.Tensor | None = None
    """[hand_dof,] of float. Joint positions for closed hand state."""

    hand_control_part: str = "hand"
    """Name of the robot part that controls the hand joints."""

    lift_height: float = 0.1
    """Height (m) to lift the end-effector after the gripper phase."""

    sample_interval: int = 80
    """Number of waypoints for the full trajectory (approach + hand + lift/back)."""

    hand_interp_steps: int = 5
    """Number of waypoints for the gripper open/close interpolation phase."""


class MoveAction(AtomicAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: MoveActionCfg | None = None,
    ):
        """
        Initialize the atomic action.
        Args:
            motion_generator: The motion generator instance to use for planning.
            cfg: Configuration for the action.
        """
        super().__init__(
            motion_generator, cfg=cfg if cfg is not None else MoveActionCfg()
        )

        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.dof = len(self.arm_joint_ids)

    def _resolve_pose_target(
        self,
        target: Union[ObjectSemantics, torch.Tensor],
        *,
        action_name: str,
    ) -> tuple[bool, torch.Tensor]:
        """Resolve a pose target into a batched homogeneous transform tensor."""
        if isinstance(target, ObjectSemantics):
            logger.log_error(
                f"{action_name} currently does not support ObjectSemantics target. "
                f"Please provide target pose as torch.Tensor of shape (4, 4) or "
                f"(n_envs, 4, 4)",
                NotImplementedError,
            )
        if not isinstance(target, torch.Tensor):
            logger.log_error(
                "Target must be either ObjectSemantics or torch.Tensor of shape "
                f"(4, 4) or ({self.n_envs}, 4, 4)",
                TypeError,
            )

        if target.shape == (4, 4):
            target = target.unsqueeze(0).repeat(self.n_envs, 1, 1)
        if target.shape != (self.n_envs, 4, 4):
            logger.log_error(
                f"Target tensor must have shape (4, 4) or ({self.n_envs}, 4, 4), but got {target.shape}",
                ValueError,
            )
        return True, target

    def _resolve_start_qpos(
        self,
        start_qpos: Optional[torch.Tensor],
        arm_dof: Optional[int] = None,
    ) -> torch.Tensor:
        """Resolve planning start joint positions into batched arm joint positions."""
        arm_dof = self.dof if arm_dof is None else arm_dof
        if start_qpos is None:
            start_qpos = self.robot.get_qpos(name=self.cfg.control_part)
        if start_qpos.shape == (arm_dof,):
            start_qpos = start_qpos.unsqueeze(0).repeat(self.n_envs, 1)
        if start_qpos.shape != (self.n_envs, arm_dof):
            logger.log_error(
                f"start_qpos must have shape ({self.n_envs}, {arm_dof}), but got {start_qpos.shape}",
                ValueError,
            )
        return start_qpos

    def _compute_three_phase_waypoints(
        self,
        hand_interp_steps: int,
        *,
        first_phase_name: str,
        third_phase_name: str,
        first_phase_ratio: float = 0.6,
    ) -> tuple[int, int, int]:
        """Split total sample interval into motion, hand interpolation, and motion phases."""
        first_phase_waypoint = int(
            np.round(self.cfg.sample_interval - hand_interp_steps) * first_phase_ratio
        )
        if first_phase_waypoint < 2:
            logger.log_error(
                f"Not enough waypoints for {first_phase_name} trajectory. "
                "Please increase sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        second_phase_waypoint = hand_interp_steps
        third_phase_waypoint = (
            self.cfg.sample_interval - first_phase_waypoint - second_phase_waypoint
        )
        if third_phase_waypoint < 2:
            logger.log_error(
                f"Not enough waypoints for {third_phase_name} trajectory. "
                "Please increase sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        return first_phase_waypoint, second_phase_waypoint, third_phase_waypoint

    def _build_motion_gen_options(
        self,
        start_qpos: torch.Tensor,
        sample_interval: int,
    ) -> MotionGenOptions:
        """Build default motion generation options for an atomic action."""
        return MotionGenOptions(
            start_qpos=start_qpos[0],
            control_part=self.cfg.control_part,
            is_interpolate=True,
            is_linear=False,
            interpolate_position_step=0.001,
            plan_opts=ToppraPlanOptions(
                sample_interval=sample_interval,
            ),
        )

    def _plan_arm_trajectory(
        self,
        target_states_list: list[list[PlanState]],
        start_qpos: torch.Tensor,
        n_waypoints: int,
        arm_dof: Optional[int] = None,
    ) -> tuple[bool, torch.Tensor]:
        """Plan batched arm trajectories for all environments."""
        arm_dof = self.dof if arm_dof is None else arm_dof

        n_state = len(target_states_list[0])
        xpos_traj = torch.zeros(
            size=(self.n_envs, n_state, 4, 4), dtype=torch.float32, device=self.device
        )
        for i, target_states in enumerate(target_states_list):
            for j, target_state in enumerate(target_states):
                # [env_i, state_j, 4, 4]
                xpos_traj[i, j] = target_state.xpos

        trajectory = torch.zeros(
            size=(self.n_envs, n_state, arm_dof),
            dtype=torch.float32,
            device=self.device,
        )
        qpos_seed = start_qpos
        for j in range(n_state):
            is_success, qpos = self.robot.compute_ik(
                pose=xpos_traj[:, j], name=self.cfg.control_part, joint_seed=qpos_seed
            )
            if not is_success:
                logger.log_warning(
                    f"Failed to compute IK for target state {j} in some environments. "
                    "The resulting trajectory may be invalid."
                )
                return False, trajectory
            else:
                trajectory[:, j] = qpos
                qpos_seed = qpos
        trajectory = torch.concatenate([start_qpos.unsqueeze(1), trajectory], dim=1)
        interp_traj = interpolate_with_distance(
            trajectory=trajectory, interp_num=n_waypoints, device=self.device
        )
        return True, interp_traj

    def _interpolate_hand_qpos(
        self,
        start_hand_qpos: torch.Tensor,
        end_hand_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate hand joint positions between two gripper states."""
        start_hand_qpos = start_hand_qpos.to(self.device)
        end_hand_qpos = end_hand_qpos.to(self.device)

        if start_hand_qpos.dim() == 1:
            start_hand_qpos = start_hand_qpos.unsqueeze(0)
        if end_hand_qpos.dim() == 1:
            end_hand_qpos = end_hand_qpos.unsqueeze(0)

        weights = torch.linspace(
            0,
            1,
            steps=n_waypoints,
            device=self.device,
            dtype=start_hand_qpos.dtype,
        )
        return torch.lerp(
            start_hand_qpos.unsqueeze(1),
            end_hand_qpos.unsqueeze(1),
            weights[None, :, None],
        )

    @staticmethod
    def _invert_pose(pose: torch.Tensor) -> torch.Tensor:
        """Invert a batched homogeneous transform."""
        inv_pose = pose.clone()
        rot_t = pose[:, :3, :3].transpose(1, 2)
        inv_pose[:, :3, :3] = rot_t
        inv_pose[:, :3, 3] = -torch.bmm(rot_t, pose[:, :3, 3:4]).squeeze(-1)
        return inv_pose

    def execute(
        self,
        target: Union[ObjectSemantics, torch.Tensor],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list[float]]:
        """execute pick up action

        Args:
            target (ObjectSemantics): object semantics containing grasp affordance and entity information
            start_qpos (Optional[torch.Tensor], optional): Planning start qpos. Defaults to None.

        Returns:
            tuple[bool, torch.Tensor, list[float]]:
            is_success,
            trajectory of shape (n_envs, n_waypoints, dof),
            joint_ids corresponding to trajectory
        """
        is_success, move_xpos = self._resolve_pose_target(
            target, action_name=self.__class__.__name__
        )
        start_qpos = self._resolve_start_qpos(start_qpos)

        # TODO: warning and fallback if no valid grasp pose found
        if not is_success:
            logger.log_warning("Failed to resolve move target pose.")
            return False, torch.empty(0), self.arm_joint_ids

        target_states_list = [
            [
                PlanState(xpos=move_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        is_plan_success, trajectory = self._plan_arm_trajectory(
            target_states_list, start_qpos, self.cfg.sample_interval
        )
        return is_plan_success, trajectory, self.arm_joint_ids

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True


@configclass
class PickUpActionCfg(GraspActionCfg):
    name: str = "pick_up"
    """Name of the action, used for identification and logging."""

    pre_grasp_distance: float = 0.15
    """Distance to offset back from the grasp pose along the approach direction to get
    the pre-grasp pose. Should be large enough to avoid collision during approach."""

    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)
    """Direction from which the gripper approaches the object for grasping, expressed
    in the object local frame. Default [0, 0, -1] means approaching from above."""


class PickUpAction(MoveAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: PickUpActionCfg | None = None,
    ):
        """
        Initialize the atomic action.
        Args:
            motion_generator: The motion generator instance to use for planning.
            cfg: Configuration for the action.
        """
        super().__init__(
            motion_generator, cfg=cfg if cfg is not None else PickUpActionCfg()
        )
        self.cfg = cfg if cfg is not None else self.cfg
        self._held_object_state: HeldObjectState | None = None
        self.approach_direction = self.cfg.approach_direction.to(self.device)
        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PickUpActionCfg")
        if self.cfg.hand_close_qpos is None:
            logger.log_error("hand_close_qpos must be specified in PickUpActionCfg")
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.joint_ids = self.arm_joint_ids + self.hand_joint_ids
        self.arm_dof = len(self.arm_joint_ids)
        self.dof = len(self.joint_ids)

    def _expand_hand_qpos(self, hand_qpos: torch.Tensor) -> torch.Tensor:
        """Resolve hand qpos to batched shape ``(n_envs, hand_dof)``."""
        hand_dof = len(self.hand_joint_ids)
        hand_qpos = hand_qpos.to(device=self.device, dtype=torch.float32)
        if hand_qpos.shape == (hand_dof,):
            return hand_qpos.unsqueeze(0).repeat(self.n_envs, 1)
        if hand_qpos.shape == (self.n_envs, hand_dof):
            return hand_qpos
        logger.log_error(
            f"hand_qpos must have shape ({hand_dof},) or "
            f"({self.n_envs}, {hand_dof}), but got {hand_qpos.shape}",
            ValueError,
        )

    def _repeat_hand_qpos(
        self, hand_qpos: torch.Tensor, n_waypoints: int
    ) -> torch.Tensor:
        """Repeat hand qpos across trajectory waypoints."""
        return self._expand_hand_qpos(hand_qpos).unsqueeze(1).repeat(1, n_waypoints, 1)

    def execute(
        self,
        target: Union[ObjectSemantics, torch.Tensor],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list[float]]:
        """execute pick up action

        Args:
            target (Union[ObjectSemantics, torch.Tensor]): target object semantics or target pose for grasping
            start_qpos (Optional[torch.Tensor], optional): Planning start qpos. Defaults to None.

        Returns:
            tuple[bool, torch.Tensor, list[float]]:
            is_success,
            trajectory of shape (n_envs, n_waypoints, dof),
            joint_ids corresponding to trajectory
        """

        # Resolve grasp pose
        self._held_object_state = None
        target_semantics = target if isinstance(target, ObjectSemantics) else None
        if target_semantics is not None:
            is_success, grasp_xpos = self._resolve_grasp_pose(target)
        else:
            is_success, grasp_xpos = self._resolve_pose_target(
                target, action_name=self.__class__.__name__
            )

        if isinstance(is_success, torch.Tensor):
            is_success = torch.all(is_success).item()
        if not is_success:
            logger.log_warning("Failed to resolve grasp pose for all environments.")
            return False, torch.empty(0), self.joint_ids

        if target_semantics is not None:
            obj_poses = target_semantics.entity.get_local_pose(to_matrix=True)
            object_to_eef = torch.bmm(self._invert_pose(obj_poses), grasp_xpos)
            self._held_object_state = HeldObjectState(
                semantics=target_semantics,
                object_to_eef=object_to_eef,
                grasp_xpos=grasp_xpos,
            )

        # Compute pre-grasp pose
        # TODO: only for parallel gripper, approach in negative grasp z direction
        grasp_z = grasp_xpos[:, :3, 2]
        pre_grasp_xpos = self._apply_offset(
            pose=grasp_xpos,
            offset=-grasp_z * self.cfg.pre_grasp_distance,
        )
        # Compute lift pose
        start_qpos = self._resolve_start_qpos(start_qpos, self.arm_dof)

        # compute waypoint number for each phase
        n_approach_waypoint, n_close_waypoint, n_lift_waypoint = (
            self._compute_three_phase_waypoints(
                self.cfg.hand_interp_steps,
                first_phase_name="approach",
                third_phase_name="lift",
            )
        )

        # get pick trajectory
        target_states_list = [
            [
                PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        pick_trajectory = torch.zeros(
            size=(self.n_envs, n_approach_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            start_qpos,
            n_approach_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan approach trajectory.")
            return False, pick_trajectory, self.joint_ids
        pick_trajectory[:, :, : self.arm_dof] = plan_traj
        # Padding hand open qpos to pick trajectory
        pick_trajectory[:, :, self.arm_dof :] = self.hand_open_qpos

        # get hand closing trajectory
        grasp_qpos = pick_trajectory[
            :, -1, : self.arm_dof
        ]  # Assuming the last point of pick trajectory is the grasp pose
        hand_close_path = self._interpolate_hand_qpos(
            self.hand_open_qpos,
            self.hand_close_qpos,
            n_close_waypoint,
        )
        hand_close_trajectory = torch.zeros(
            size=(self.n_envs, n_close_waypoint, self.dof),
            device=self.device,
        )
        hand_close_trajectory[:, :, : self.arm_dof] = grasp_qpos
        hand_close_trajectory[:, :, self.arm_dof :] = hand_close_path

        # get lift trajectory
        lift_trajectory = torch.zeros(
            size=(self.n_envs, n_lift_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        # lift_xpos = self._compute_lift_xpos(grasp_xpos)
        lift_xpos = self._apply_offset(
            pose=grasp_xpos,
            offset=torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )
        target_states_list = [
            [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            grasp_qpos,
            n_lift_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan lift trajectory.")
            return False, lift_trajectory, self.joint_ids
        lift_trajectory[:, :, : self.arm_dof] = plan_traj
        # padding hand close qpos to lift trajectory
        lift_trajectory[:, :, self.arm_dof :] = self.hand_close_qpos

        # concatenate trajectories
        trajectory = torch.cat(
            [pick_trajectory, hand_close_trajectory, lift_trajectory], dim=1
        )
        return True, trajectory, self.joint_ids

    def get_held_object_state(self) -> HeldObjectState | None:
        """Return the held-object state produced by the latest successful pickup."""
        return self._held_object_state

    def _resolve_grasp_pose(
        self, semantics: ObjectSemantics
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(semantics.affordance, AntipodalAffordance):
            logger.log_error(
                "Grasp pose affordance must be of type AntipodalAffordance"
            )
        if semantics.entity is None:
            logger.log_error(
                "ObjectSemantics must be associated with an entity to get object pose"
            )
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)

        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=obj_poses, approach_direction=self.approach_direction
        )

        # Get best grasp pose for each object
        n_envs = obj_poses.shape[0]
        init_qpos = self.robot.get_qpos(name=self.cfg.control_part)
        n_max_pose = 0
        for result in grasp_poses_result:
            n_pose = result[0].shape[0]
            if n_pose > n_max_pose:
                n_max_pose = n_pose

        grasp_xpos_padding = torch.zeros(
            (n_envs, n_max_pose, 4, 4), dtype=torch.float32, device=self.device
        )
        grasp_cost_padding = torch.full(
            (n_envs, n_max_pose), float("inf"), dtype=torch.float32, device=self.device
        )
        for i in range(n_envs):
            n_pose = grasp_poses_result[i][0].shape[0]
            grasp_xpos_padding[i, :n_pose] = grasp_poses_result[i][0]
            grasp_cost_padding[i, :n_pose] = grasp_poses_result[i][1]
            # padding with the first grasp pose, which is usually the best one, to ensure that the padded grasp poses are valid for IK computation, although they may not be optimal.
            grasp_xpos_padding[i, n_pose:] = grasp_poses_result[i][0][0]
            grasp_cost_padding[i, n_pose:] = grasp_poses_result[i][1][0]

        init_qpos_repeat = init_qpos[:, None, :].repeat(1, n_max_pose, 1)
        ik_success, qpos = self.robot.compute_batch_ik(
            pose=grasp_xpos_padding,
            name=self.cfg.control_part,
            joint_seed=init_qpos_repeat,
        )
        grasp_cost_masked = torch.where(ik_success, grasp_cost_padding, 10000.0)
        best_cost, best_idx = grasp_cost_masked.min(dim=1)
        is_success = best_cost < 9999.0  # usually cost < 1.0
        best_grasp_xpos = grasp_xpos_padding[
            torch.arange(n_envs, device=self.device), best_idx
        ]

        return is_success, best_grasp_xpos

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True


@configclass
class UprightActionCfg(PickUpActionCfg):
    name: str = "upright"
    """Name of the action, used for identification and logging."""

    place_clearance: float = 0.005
    """Clearance (m) between the upright object bottom and the support plane."""

    upright_axis_sign: float = 1.0
    """Direction of the object's local Z axis after upright placement.

    Use ``1.0`` to align local +Z with world +Z. Use ``-1.0`` when the mesh's
    local +Z points toward the physical bottom and local -Z should face upward.
    """

    place_press_depth: float = 0.002
    """Additional downward displacement (m) after pre-place to make support contact."""

    place_press_steps: int = 4
    """Number of closed-hand waypoints used for the downward place press."""

    upright_hold_steps: int = 0
    """Number of closed-hand waypoints to hold after upright placement."""

    place_hold_steps: int = 8
    """Number of closed-hand waypoints to hold the object after pressing it down."""

    release_interp_steps: int = 12
    """Number of waypoints for the slow hand release phase."""

    release_retreat_distance: float = 0.08
    """Horizontal distance (m) to retreat after releasing the upright object."""

    release_retreat_lift: float = 0.01
    """Small upward offset (m) added during release retreat."""

    use_grasp_width_qpos: bool = False
    """Whether to map selected grasp open length into a dynamic hand close qpos."""

    gripper_max_open_width: float = 0.088
    """Maximum total gripper opening width (m) used for width-to-qpos mapping."""

    grasp_squeeze_width: float = 0.003
    """Width margin (m) subtracted from the selected grasp width before closing."""

    final_approach_steps: int = 12
    """Number of waypoints for the slow final approach from pre-grasp to grasp."""

    final_approach_preclose_width_margin: float = 0.010
    """Extra opening width (m) kept around the selected grasp width during final approach."""

    grasp_hold_steps: int = 4
    """Number of closed-hand waypoints to hold the grasp before lifting."""

    min_dynamic_hand_close_qpos: torch.Tensor | None = None
    """Optional minimum hand qpos used when mapping grasp width into close qpos."""

    max_grasp_open_length: float | None = None
    """Optional maximum selected grasp opening length (m) for upright placement."""

    max_grasp_axis_approach_dot: float | None = None
    """Optional maximum absolute dot between grasp X axis and approach direction."""

    max_grasp_axis_upright_axis_dot: float | None = None
    """Optional maximum absolute dot between grasp X axis and object upright axis."""

    upright_yaw_offsets: tuple[float, ...] = (
        0.0,
        0.5 * np.pi,
        -0.5 * np.pi,
        np.pi,
    )
    """Yaw offsets (rad) to try after aligning the object upright axis."""


class UprightAction(PickUpAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: UprightActionCfg | None = None,
    ):
        """
        Initialize the atomic action.
        Args:
            motion_generator: The motion generator instance to use for planning.
            cfg: Configuration for the action.
        """
        super().__init__(
            motion_generator, cfg=cfg if cfg is not None else UprightActionCfg()
        )

    def _resolve_grasp_pose(
        self, semantics: ObjectSemantics
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(semantics.affordance, AntipodalAffordance):
            logger.log_error(
                "Grasp pose affordance must be of type AntipodalAffordance"
            )
        if semantics.entity is None:
            logger.log_error(
                "ObjectSemantics must be associated with an entity to get object pose"
            )
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        if semantics.affordance.generator is None:
            semantics.affordance._init_generator()
        generator = semantics.affordance.generator
        if generator is None:
            logger.log_error("Failed to initialize antipodal grasp generator")

        n_envs = obj_poses.shape[0]
        approach_direction = self.approach_direction.to(
            device=self.device, dtype=torch.float32
        )
        approach_direction = approach_direction / approach_direction.norm().clamp(
            min=1e-6
        )
        max_open_length = self.cfg.max_grasp_open_length
        max_approach_axis_dot = self.cfg.max_grasp_axis_approach_dot
        max_upright_axis_dot = self.cfg.max_grasp_axis_upright_axis_dot

        is_success = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
        grasp_xpos = torch.eye(4, dtype=torch.float32, device=self.device).repeat(
            n_envs, 1, 1
        )
        open_length = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
        init_qpos = self.robot.get_qpos(name=self.cfg.control_part)
        world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        upright_obj_pose_candidates = self._build_upright_object_pose_candidates(
            semantics, obj_poses
        )
        selected_upright_obj_poses = upright_obj_pose_candidates[:, 0].clone()

        for env_idx in range(n_envs):
            (
                has_candidates,
                candidate_grasp_xpos,
                candidate_open_length,
                candidate_cost,
            ) = generator.get_valid_grasp_poses(obj_poses[env_idx], approach_direction)
            if not has_candidates:
                logger.log_warning(
                    f"No valid grasp candidates found for {env_idx}-th object."
                )
                continue

            candidate_grasp_xpos = candidate_grasp_xpos.to(
                device=self.device, dtype=torch.float32
            )
            candidate_open_length = candidate_open_length.to(
                device=self.device, dtype=torch.float32
            )
            candidate_cost = candidate_cost.to(device=self.device, dtype=torch.float32)
            candidate_mask = torch.ones(
                candidate_grasp_xpos.shape[0], dtype=torch.bool, device=self.device
            )
            if max_open_length is not None:
                candidate_mask &= candidate_open_length <= max_open_length

            grasp_axis_dot = torch.abs(
                (candidate_grasp_xpos[:, :3, 0] * approach_direction).sum(dim=1)
            )
            if max_approach_axis_dot is not None:
                candidate_mask &= grasp_axis_dot <= max_approach_axis_dot

            upright_axis = torch.nn.functional.normalize(
                obj_poses[env_idx, :3, 2], dim=0
            )
            grasp_upright_axis_dot = torch.abs(
                (candidate_grasp_xpos[:, :3, 0] * upright_axis).sum(dim=1)
            )
            if max_upright_axis_dot is not None:
                candidate_mask &= grasp_upright_axis_dot <= max_upright_axis_dot

            if not bool(torch.any(candidate_mask).item()):
                logger.log_warning(
                    "No grasp candidates remain after upright grasp filtering "
                    f"for {env_idx}-th object."
                )
                continue

            candidate_grasp_xpos = candidate_grasp_xpos[candidate_mask]
            candidate_open_length = candidate_open_length[candidate_mask]
            candidate_cost = candidate_cost[candidate_mask]
            n_candidate = candidate_grasp_xpos.shape[0]

            pre_grasp_xpos = self._apply_offset(
                pose=candidate_grasp_xpos,
                offset=-candidate_grasp_xpos[:, :3, 2] * self.cfg.pre_grasp_distance,
            )
            lift_xpos = self._apply_offset(
                pose=candidate_grasp_xpos,
                offset=world_z * self.cfg.lift_height,
            )
            obj_pose_repeat = obj_poses[env_idx].unsqueeze(0).repeat(n_candidate, 1, 1)
            obj_to_grasp = torch.bmm(
                self._invert_pose(obj_pose_repeat), candidate_grasp_xpos
            )

            base_ik_success = torch.ones(
                n_candidate, dtype=torch.bool, device=self.device
            )
            qpos_seed = init_qpos[env_idx : env_idx + 1, None, :].repeat(
                1, n_candidate, 1
            )
            for target_xpos in (
                pre_grasp_xpos,
                candidate_grasp_xpos,
                lift_xpos,
            ):
                target_success, target_qpos = self.robot.compute_batch_ik(
                    pose=target_xpos.unsqueeze(0),
                    name=self.cfg.control_part,
                    joint_seed=qpos_seed,
                    env_ids=[env_idx],
                )
                base_ik_success &= target_success[0]
                qpos_seed = target_qpos

            n_upright_pose = upright_obj_pose_candidates.shape[1]
            upright_obj_pose_repeat = (
                upright_obj_pose_candidates[env_idx]
                .unsqueeze(1)
                .repeat(1, n_candidate, 1, 1)
                .reshape(-1, 4, 4)
            )
            obj_to_grasp_repeat = (
                obj_to_grasp.unsqueeze(0)
                .repeat(n_upright_pose, 1, 1, 1)
                .reshape(-1, 4, 4)
            )
            upright_lift_obj_xpos = self._apply_offset(
                pose=upright_obj_pose_repeat,
                offset=world_z * self.cfg.lift_height,
            )
            upright_lift_xpos = torch.bmm(upright_lift_obj_xpos, obj_to_grasp_repeat)
            upright_place_xpos = torch.bmm(upright_obj_pose_repeat, obj_to_grasp_repeat)
            press_xpos = self._apply_offset(
                pose=upright_place_xpos,
                offset=-world_z
                * (self.cfg.place_clearance + self.cfg.place_press_depth),
            )

            ik_success = base_ik_success.repeat(n_upright_pose)
            upright_qpos_seed = qpos_seed.repeat(1, n_upright_pose, 1)
            for target_xpos in (upright_lift_xpos, upright_place_xpos, press_xpos):
                target_success, target_qpos = self.robot.compute_batch_ik(
                    pose=target_xpos.unsqueeze(0),
                    name=self.cfg.control_part,
                    joint_seed=upright_qpos_seed,
                    env_ids=[env_idx],
                )
                ik_success &= target_success[0]
                upright_qpos_seed = target_qpos

            flat_candidate_cost = candidate_cost.repeat(n_upright_pose)
            masked_cost = torch.where(
                ik_success,
                flat_candidate_cost,
                torch.full_like(flat_candidate_cost, float("inf")),
            )
            best_cost, best_flat_idx = masked_cost.min(dim=0)
            best_upright_idx = torch.div(
                best_flat_idx, n_candidate, rounding_mode="floor"
            )
            best_idx = best_flat_idx % n_candidate

            if not torch.isfinite(best_cost):
                logger.log_warning(
                    "No upright grasp candidates remain after IK feasibility "
                    f"filtering for {env_idx}-th object."
                )
                continue

            is_success[env_idx] = True
            grasp_xpos[env_idx] = candidate_grasp_xpos[best_idx]
            open_length[env_idx] = candidate_open_length[best_idx]
            selected_upright_obj_poses[env_idx] = upright_obj_pose_candidates[
                env_idx, best_upright_idx
            ]

        self._selected_upright_obj_xpos = selected_upright_obj_poses
        return is_success, grasp_xpos, open_length

    @staticmethod
    def _invert_pose(pose: torch.Tensor) -> torch.Tensor:
        """Invert a batched homogeneous transform."""
        inv_pose = pose.clone()
        rot_t = pose[:, :3, :3].transpose(1, 2)
        inv_pose[:, :3, :3] = rot_t
        inv_pose[:, :3, 3] = -torch.bmm(rot_t, pose[:, :3, 3:4]).squeeze(-1)
        return inv_pose

    def _build_upright_object_pose(
        self, semantics: ObjectSemantics, obj_poses: torch.Tensor
    ) -> torch.Tensor:
        """Build a target object pose whose configured local Z direction is upright."""
        world_z = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        axis_sign = 1.0 if self.cfg.upright_axis_sign >= 0.0 else -1.0
        projected_x = obj_poses[:, :3, 0].clone()
        projected_x[:, 2] = 0.0
        projected_x_norm = projected_x.norm(dim=1, keepdim=True)

        fallback_x = obj_poses[:, :3, 1].clone()
        fallback_x[:, 2] = 0.0
        fallback_x_norm = fallback_x.norm(dim=1, keepdim=True)
        fallback_x = fallback_x / fallback_x_norm.clamp(min=1e-6)

        default_x = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            self.n_envs, 1
        )
        upright_x = torch.where(
            projected_x_norm > 1e-6,
            projected_x / projected_x_norm.clamp(min=1e-6),
            torch.where(fallback_x_norm > 1e-6, fallback_x, default_x),
        )
        upright_z = axis_sign * world_z.repeat(self.n_envs, 1)
        upright_y = torch.cross(upright_z, upright_x, dim=1)
        upright_y = upright_y / upright_y.norm(dim=1, keepdim=True).clamp(min=1e-6)
        upright_x = torch.cross(upright_y, upright_z, dim=1)
        upright_x = upright_x / upright_x.norm(dim=1, keepdim=True).clamp(min=1e-6)

        upright_pose = obj_poses.clone()
        upright_pose[:, :3, 0] = upright_x
        upright_pose[:, :3, 1] = upright_y
        upright_pose[:, :3, 2] = upright_z

        mesh_vertices = semantics.geometry.get("mesh_vertices")
        if isinstance(mesh_vertices, torch.Tensor) and mesh_vertices.numel() > 0:
            mesh_vertices = mesh_vertices.to(device=self.device, dtype=torch.float32)
            vertical_offsets = torch.matmul(
                mesh_vertices, upright_pose[:, 2, :3].transpose(0, 1)
            )
            local_bottom_z = vertical_offsets.min(dim=0).values
            upright_pose[:, 2, 3] = self.cfg.place_clearance - local_bottom_z
        return upright_pose

    def _build_upright_object_pose_candidates(
        self, semantics: ObjectSemantics, obj_poses: torch.Tensor
    ) -> torch.Tensor:
        """Build upright target poses with alternative yaw rotations."""
        base_pose = self._build_upright_object_pose(semantics, obj_poses)
        yaw_offsets = torch.as_tensor(
            self.cfg.upright_yaw_offsets, device=self.device, dtype=torch.float32
        )
        cos_yaw = torch.cos(yaw_offsets).view(1, -1, 1)
        sin_yaw = torch.sin(yaw_offsets).view(1, -1, 1)

        base_x = base_pose[:, None, :3, 0]
        base_y = base_pose[:, None, :3, 1]
        candidates = base_pose[:, None, :, :].repeat(1, yaw_offsets.numel(), 1, 1)
        candidates[:, :, :3, 0] = cos_yaw * base_x + sin_yaw * base_y
        candidates[:, :, :3, 1] = -sin_yaw * base_x + cos_yaw * base_y
        return candidates

    def _compute_hand_qpos_for_width(self, target_width: torch.Tensor) -> torch.Tensor:
        """Map desired total gripper width to batched hand qpos."""
        target_width = target_width.to(device=self.device, dtype=torch.float32).view(
            self.n_envs, 1
        )
        target_width = target_width.clamp(min=0.0, max=self.cfg.gripper_max_open_width)
        closing_distance = 0.5 * (self.cfg.gripper_max_open_width - target_width).clamp(
            min=0.0
        )
        hand_qpos_limits = self.robot.get_qpos_limits(
            name=self.cfg.hand_control_part
        ).to(self.device)
        lower_limits = hand_qpos_limits[:, :, 0]
        upper_limits = hand_qpos_limits[:, :, 1]
        hand_open_qpos = self._expand_hand_qpos(self.hand_open_qpos)
        dynamic_qpos = hand_open_qpos + closing_distance.repeat(
            1, len(self.hand_joint_ids)
        )
        return torch.max(torch.min(dynamic_qpos, upper_limits), lower_limits)

    def _compute_dynamic_hand_close_qpos(
        self, grasp_open_length: torch.Tensor
    ) -> torch.Tensor:
        """Map selected grasp width to batched hand close qpos for parallel grippers."""
        fallback_qpos = self._expand_hand_qpos(self.hand_close_qpos)
        if not self.cfg.use_grasp_width_qpos:
            return fallback_qpos

        grasp_open_length = grasp_open_length.to(
            device=self.device, dtype=torch.float32
        ).view(self.n_envs, 1)
        target_width = (grasp_open_length - self.cfg.grasp_squeeze_width).clamp(min=0.0)
        dynamic_qpos = self._compute_hand_qpos_for_width(target_width)
        if self.cfg.min_dynamic_hand_close_qpos is not None:
            min_close_qpos = self._expand_hand_qpos(
                self.cfg.min_dynamic_hand_close_qpos
            )
            dynamic_qpos = torch.max(dynamic_qpos, min_close_qpos)
        return dynamic_qpos

    def _compute_final_approach_hand_qpos(
        self, grasp_open_length: torch.Tensor, hand_close_qpos: torch.Tensor
    ) -> torch.Tensor:
        """Pre-close the gripper during final approach without reaching squeeze force."""
        hand_open_qpos = self._expand_hand_qpos(self.hand_open_qpos)
        if not self.cfg.use_grasp_width_qpos:
            return hand_open_qpos

        grasp_open_length = grasp_open_length.to(
            device=self.device, dtype=torch.float32
        ).view(self.n_envs, 1)
        target_width = grasp_open_length + self.cfg.final_approach_preclose_width_margin
        preclose_qpos = self._compute_hand_qpos_for_width(target_width)
        return torch.max(torch.min(preclose_qpos, hand_close_qpos), hand_open_qpos)

    def execute(
        self,
        target: Union[ObjectSemantics, torch.Tensor],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list[float]]:
        """Pick up an object, rotate it upright, place it down, and release it."""
        if not isinstance(target, ObjectSemantics):
            return super().execute(target=target, start_qpos=start_qpos, **kwargs)

        is_success, grasp_xpos, grasp_open_length = self._resolve_grasp_pose(target)
        obj_poses = target.entity.get_local_pose(to_matrix=True)
        if not torch.all(is_success).item():
            logger.log_warning(
                "Failed to resolve upright grasp pose for all environments."
            )
            return False, torch.empty(0), self.joint_ids

        world_z = torch.tensor([0, 0, 1], device=self.device, dtype=torch.float32)
        hand_close_qpos = self._compute_dynamic_hand_close_qpos(grasp_open_length)
        final_approach_hand_qpos = self._compute_final_approach_hand_qpos(
            grasp_open_length, hand_close_qpos
        )
        pre_grasp_xpos = self._apply_offset(
            pose=grasp_xpos,
            offset=-grasp_xpos[:, :3, 2] * self.cfg.pre_grasp_distance,
        )
        lift_xpos = self._apply_offset(
            pose=grasp_xpos,
            offset=world_z * self.cfg.lift_height,
        )

        obj_to_grasp = torch.bmm(self._invert_pose(obj_poses), grasp_xpos)
        upright_obj_xpos = getattr(self, "_selected_upright_obj_xpos", None)
        if upright_obj_xpos is None or upright_obj_xpos.shape != obj_poses.shape:
            upright_obj_xpos = self._build_upright_object_pose(target, obj_poses)
        upright_lift_obj_xpos = self._apply_offset(
            pose=upright_obj_xpos,
            offset=world_z * self.cfg.lift_height,
        )
        upright_lift_xpos = torch.bmm(upright_lift_obj_xpos, obj_to_grasp)
        upright_place_xpos = torch.bmm(upright_obj_xpos, obj_to_grasp)
        press_down_distance = self.cfg.place_clearance + self.cfg.place_press_depth
        press_xpos = self._apply_offset(
            pose=upright_place_xpos,
            offset=-world_z * press_down_distance,
        )
        retreat_direction = -press_xpos[:, :3, 2]
        retreat_direction[:, 2] = 0.0
        retreat_direction_norm = retreat_direction.norm(dim=1, keepdim=True)
        retreat_direction = torch.where(
            retreat_direction_norm > 1e-6,
            retreat_direction / retreat_direction_norm.clamp(min=1e-6),
            -press_xpos[:, :3, 0],
        )
        retreat_direction[:, 2] = 0.0
        retreat_direction = retreat_direction / retreat_direction.norm(
            dim=1, keepdim=True
        ).clamp(min=1e-6)
        retreat_offset = (
            retreat_direction * self.cfg.release_retreat_distance
            + world_z * self.cfg.release_retreat_lift
        )
        retreat_xpos = self._apply_offset(
            pose=press_xpos,
            offset=retreat_offset,
        )

        start_qpos = self._resolve_start_qpos(start_qpos, self.arm_dof)
        n_close_waypoint = self.cfg.hand_interp_steps
        n_final_approach_waypoint = max(2, self.cfg.final_approach_steps)
        n_grasp_hold_waypoint = max(0, self.cfg.grasp_hold_steps)
        n_press_waypoint = max(2, self.cfg.place_press_steps)
        n_upright_hold_waypoint = max(0, self.cfg.upright_hold_steps)
        n_hold_waypoint = max(0, self.cfg.place_hold_steps)
        n_open_waypoint = max(2, self.cfg.release_interp_steps)
        motion_waypoints = (
            self.cfg.sample_interval
            - n_close_waypoint
            - n_final_approach_waypoint
            - n_grasp_hold_waypoint
            - n_upright_hold_waypoint
            - n_press_waypoint
            - n_hold_waypoint
            - n_open_waypoint
        )
        if motion_waypoints < 6:
            logger.log_error(
                "Not enough waypoints for upright action. Please increase "
                "sample_interval or decrease hand/press/upright-hold/hold/release "
                "steps.",
                ValueError,
            )
        n_pre_approach_waypoint = max(2, int(np.round(motion_waypoints * 0.25)))
        n_upright_waypoint = max(2, int(np.round(motion_waypoints * 0.60)))
        n_retreat_waypoint = (
            self.cfg.sample_interval
            - n_close_waypoint
            - n_final_approach_waypoint
            - n_grasp_hold_waypoint
            - n_upright_hold_waypoint
            - n_press_waypoint
            - n_hold_waypoint
            - n_open_waypoint
            - n_pre_approach_waypoint
            - n_upright_waypoint
        )
        if n_retreat_waypoint < 2:
            retreat_deficit = 2 - n_retreat_waypoint
            n_retreat_waypoint = 2
            n_upright_waypoint = max(2, n_upright_waypoint - retreat_deficit)
        target_states_list = [
            [
                PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            start_qpos,
            n_pre_approach_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan approach trajectory.")
            return False, torch.empty(0), self.joint_ids
        approach_trajectory = torch.zeros(
            size=(self.n_envs, n_pre_approach_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        approach_trajectory[:, :, : self.arm_dof] = plan_traj
        approach_trajectory[:, :, self.arm_dof :] = self._repeat_hand_qpos(
            self.hand_open_qpos, n_pre_approach_waypoint
        )

        pre_grasp_qpos = approach_trajectory[:, -1, : self.arm_dof]
        target_states_list = [
            [PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            pre_grasp_qpos,
            n_final_approach_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan final approach trajectory.")
            return False, torch.empty(0), self.joint_ids
        final_approach_trajectory = torch.zeros(
            size=(self.n_envs, n_final_approach_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        final_approach_trajectory[:, :, : self.arm_dof] = plan_traj
        final_approach_hand_path = self._interpolate_hand_qpos(
            self.hand_open_qpos,
            final_approach_hand_qpos,
            n_final_approach_waypoint,
        )
        final_approach_trajectory[:, :, self.arm_dof :] = final_approach_hand_path

        grasp_qpos = final_approach_trajectory[:, -1, : self.arm_dof]
        hand_close_path = self._interpolate_hand_qpos(
            final_approach_hand_qpos,
            hand_close_qpos,
            n_close_waypoint,
        )
        close_trajectory = torch.zeros(
            size=(self.n_envs, n_close_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        close_trajectory[:, :, : self.arm_dof] = grasp_qpos.unsqueeze(1)
        close_trajectory[:, :, self.arm_dof :] = hand_close_path

        closed_grasp_qpos = close_trajectory[:, -1, : self.arm_dof]
        grasp_hold_trajectory = torch.zeros(
            size=(self.n_envs, n_grasp_hold_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        if n_grasp_hold_waypoint > 0:
            grasp_hold_trajectory[:, :, : self.arm_dof] = closed_grasp_qpos.unsqueeze(1)
            grasp_hold_trajectory[:, :, self.arm_dof :] = self._repeat_hand_qpos(
                hand_close_qpos, n_grasp_hold_waypoint
            )

        target_states_list = [
            [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=upright_lift_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=upright_place_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            closed_grasp_qpos,
            n_upright_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan upright trajectory.")
            return False, torch.empty(0), self.joint_ids
        upright_trajectory = torch.zeros(
            size=(self.n_envs, n_upright_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        upright_trajectory[:, :, : self.arm_dof] = plan_traj
        upright_trajectory[:, :, self.arm_dof :] = self._repeat_hand_qpos(
            hand_close_qpos, n_upright_waypoint
        )

        place_qpos = upright_trajectory[:, -1, : self.arm_dof]
        upright_hold_trajectory = torch.zeros(
            size=(self.n_envs, n_upright_hold_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        if n_upright_hold_waypoint > 0:
            upright_hold_trajectory[:, :, : self.arm_dof] = place_qpos.unsqueeze(1)
            upright_hold_trajectory[:, :, self.arm_dof :] = self._repeat_hand_qpos(
                hand_close_qpos, n_upright_hold_waypoint
            )

        target_states_list = [
            [PlanState(xpos=press_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            place_qpos,
            n_press_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan place press trajectory.")
            return False, torch.empty(0), self.joint_ids
        press_trajectory = torch.zeros(
            size=(self.n_envs, n_press_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        press_trajectory[:, :, : self.arm_dof] = plan_traj
        press_trajectory[:, :, self.arm_dof :] = self._repeat_hand_qpos(
            hand_close_qpos, n_press_waypoint
        )

        press_qpos = press_trajectory[:, -1, : self.arm_dof]
        hold_trajectory = torch.zeros(
            size=(self.n_envs, n_hold_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        if n_hold_waypoint > 0:
            hold_trajectory[:, :, : self.arm_dof] = press_qpos.unsqueeze(1)
            hold_trajectory[:, :, self.arm_dof :] = self._repeat_hand_qpos(
                hand_close_qpos, n_hold_waypoint
            )

        hand_open_path = self._interpolate_hand_qpos(
            hand_close_qpos,
            self.hand_open_qpos,
            n_open_waypoint,
        )
        open_trajectory = torch.zeros(
            size=(self.n_envs, n_open_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        open_trajectory[:, :, : self.arm_dof] = press_qpos.unsqueeze(1)
        open_trajectory[:, :, self.arm_dof :] = hand_open_path

        target_states_list = [
            [PlanState(xpos=retreat_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            press_qpos,
            n_retreat_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan retreat trajectory.")
            return False, torch.empty(0), self.joint_ids
        retreat_trajectory = torch.zeros(
            size=(self.n_envs, n_retreat_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        retreat_trajectory[:, :, : self.arm_dof] = plan_traj
        retreat_trajectory[:, :, self.arm_dof :] = self._repeat_hand_qpos(
            self.hand_open_qpos, n_retreat_waypoint
        )

        trajectory = torch.cat(
            [
                approach_trajectory,
                final_approach_trajectory,
                close_trajectory,
                grasp_hold_trajectory,
                upright_trajectory,
                upright_hold_trajectory,
                press_trajectory,
                hold_trajectory,
                open_trajectory,
                retreat_trajectory,
            ],
            dim=1,
        )
        return True, trajectory, self.joint_ids


@configclass
class PlaceActionCfg(GraspActionCfg):
    name: str = "place"
    """Name of the action, used for identification and logging."""

    release: bool = True
    """Whether to open the gripper after reaching the airborne place target."""

    place_height_offset: float = 0.2
    """World-Z height offset (m) above the object target pose for object-centric place."""


class PlaceAction(MoveAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: PlaceActionCfg | None = None,
    ):
        """
        Initialize the atomic action.
        Args:
            motion_generator: The motion generator instance to use for planning.
            cfg: Configuration for the action.
        """
        super().__init__(
            motion_generator, cfg=cfg if cfg is not None else PlaceActionCfg()
        )
        self.cfg = cfg if cfg is not None else self.cfg
        self._held_object_state: HeldObjectState | None = None
        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PlaceActionCfg")
        if self.cfg.hand_close_qpos is None:
            logger.log_error("hand_close_qpos must be specified in PlaceActionCfg")
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.joint_ids = self.arm_joint_ids + self.hand_joint_ids
        self.arm_dof = len(self.arm_joint_ids)
        self.dof = len(self.joint_ids)

    def _resolve_place_target(
        self,
        target: Union[ObjectSemantics, torch.Tensor, PlaceTarget],
        action_context: dict | None = None,
        held_object_state: HeldObjectState | None = None,
    ) -> tuple[bool, torch.Tensor, bool]:
        """Resolve place target into an end-effector target pose and release flag."""
        if not isinstance(target, PlaceTarget):
            is_success, place_xpos = self._resolve_pose_target(
                target, action_name=self.__class__.__name__
            )
            release = self.cfg.release
            held_state = held_object_state
            if held_state is None and action_context is not None:
                held_state = action_context.get("held_object_state")
            self._held_object_state = None if release else held_state
            return is_success, place_xpos, release

        held_state = target.held_object or held_object_state
        if held_state is None and action_context is not None:
            held_state = action_context.get("held_object_state")
        if held_state is None:
            logger.log_error(
                "PlaceTarget requires a HeldObjectState from a prior PickUpAction "
                "or target.held_object.",
                ValueError,
            )

        object_target_pose = target.object_target_pose.to(
            device=self.device, dtype=torch.float32
        )
        if object_target_pose.shape == (4, 4):
            object_target_pose = object_target_pose.unsqueeze(0).repeat(
                self.n_envs, 1, 1
            )
        if object_target_pose.shape != (self.n_envs, 4, 4):
            logger.log_error(
                f"object_target_pose must have shape (4, 4) or "
                f"({self.n_envs}, 4, 4), but got {object_target_pose.shape}",
                ValueError,
            )

        height_offset = (
            self.cfg.place_height_offset
            if target.height_offset is None
            else target.height_offset
        )
        object_place_xpos = self._apply_offset(
            pose=object_target_pose,
            offset=torch.tensor(
                [0.0, 0.0, height_offset], dtype=torch.float32, device=self.device
            ),
        )

        object_to_eef = held_state.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.n_envs, 1, 1)
        if object_to_eef.shape != (self.n_envs, 4, 4):
            logger.log_error(
                f"object_to_eef must have shape (4, 4) or "
                f"({self.n_envs}, 4, 4), but got {object_to_eef.shape}",
                ValueError,
            )

        place_xpos = torch.bmm(object_place_xpos, object_to_eef)
        release = self.cfg.release if target.release is None else target.release
        self._held_object_state = None if release else held_state
        return True, place_xpos, release

    def execute(
        self,
        target: Union[ObjectSemantics, torch.Tensor, PlaceTarget],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list[float]]:
        """Execute object-centric or end-effector-centric place action.

        Args:
            target: PlaceTarget for object-centric place, or tensor EEF pose for
                the legacy place path.
            start_qpos (Optional[torch.Tensor], optional): Planning start qpos. Defaults to None.

        Returns:
            tuple[bool, torch.Tensor, list[float]]:
            is_success,
            trajectory of shape (n_envs, n_waypoints, dof),
            joint_ids corresponding to trajectory
        """
        is_success, place_xpos, release = self._resolve_place_target(
            target,
            action_context=kwargs.get("action_context"),
            held_object_state=kwargs.get("held_object_state"),
        )
        start_qpos = self._resolve_start_qpos(start_qpos, self.arm_dof)

        if not is_success:
            logger.log_warning("Failed to resolve place target pose.")
            return False, torch.empty(0), self.joint_ids

        if not release:
            target_states_list = [
                [
                    PlanState(xpos=place_xpos[i], move_type=MoveType.EEF_MOVE),
                ]
                for i in range(self.n_envs)
            ]
            hold_trajectory = torch.zeros(
                size=(self.n_envs, self.cfg.sample_interval, self.dof),
                dtype=torch.float32,
                device=self.device,
            )
            is_success, plan_traj = self._plan_arm_trajectory(
                target_states_list,
                start_qpos,
                self.cfg.sample_interval,
                self.arm_dof,
            )
            if not is_success:
                logger.log_warning("Failed to plan place hold trajectory.")
                return False, hold_trajectory, self.joint_ids
            hold_trajectory[:, :, : self.arm_dof] = plan_traj
            hold_trajectory[:, :, self.arm_dof :] = self.hand_close_qpos
            return True, hold_trajectory, self.joint_ids

        # compute waypoint number for each phase
        n_down_waypoint, n_open_waypoint, n_lift_waypoint = (
            self._compute_three_phase_waypoints(
                self.cfg.hand_interp_steps,
                first_phase_name="approach",
                third_phase_name="lift",
            )
        )

        down_trajectory = torch.zeros(
            size=(self.n_envs, n_down_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        lift_xpos = self._apply_offset(
            pose=place_xpos,
            offset=torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )
        target_states_list = [
            [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=place_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            start_qpos,
            n_down_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan down trajectory.")
            return False, down_trajectory, self.joint_ids
        down_trajectory[:, :, : self.arm_dof] = plan_traj
        # Padding hand open qpos to pick trajectory
        down_trajectory[:, :, self.arm_dof :] = self.hand_close_qpos

        # get hand closing trajectory
        reach_qpos = down_trajectory[
            :, -1, : self.arm_dof
        ]  # Assuming the last point of pick trajectory is the grasp pose
        hand_open_path = self._interpolate_hand_qpos(
            self.hand_close_qpos,
            self.hand_open_qpos,
            n_open_waypoint,
        )
        hand_open_trajectory = torch.zeros(
            size=(self.n_envs, n_open_waypoint, self.dof),
            device=self.device,
        )
        hand_open_trajectory[:, :, : self.arm_dof] = reach_qpos
        hand_open_trajectory[:, :, self.arm_dof :] = hand_open_path

        # get lift trajectory
        back_trajectory = torch.zeros(
            size=(self.n_envs, n_lift_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        target_states_list = [
            [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        is_success, plan_traj = self._plan_arm_trajectory(
            target_states_list,
            reach_qpos,
            n_lift_waypoint,
            self.arm_dof,
        )
        if not is_success:
            logger.log_warning("Failed to plan back trajectory.")
            return False, back_trajectory, self.joint_ids
        back_trajectory[:, :, : self.arm_dof] = plan_traj
        # padding hand open qpos to back trajectory
        back_trajectory[:, :, self.arm_dof :] = self.hand_open_qpos

        # concatenate trajectories
        trajectory = torch.cat(
            [down_trajectory, hand_open_trajectory, back_trajectory], dim=1
        )
        return True, trajectory, self.joint_ids

    def get_held_object_state(self) -> HeldObjectState | None:
        """Return the held-object state after place, if the gripper stayed closed."""
        return self._held_object_state

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True
