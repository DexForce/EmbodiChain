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
from typing import Optional, Union, TYPE_CHECKING, Any

from embodichain.lab.sim.planners import PlanResult, PlanState, MoveType
from embodichain.lab.sim.planners.motion_generator import MotionGenOptions
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
from .core import AtomicAction, ObjectSemantics, AntipodalAffordance, ActionCfg
from embodichain.utils import logger
from embodichain.utils import configclass
import numpy as np

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator
    from embodichain.lab.sim.objects import Robot


@configclass
class MoveActionCfg(ActionCfg):
    sample_interval: int = 50


class MoveAction(AtomicAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: MoveActionCfg | None = None,
    ):
        super().__init__(motion_generator)
        self.cfg = cfg if cfg is not None else MoveActionCfg()

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
    ) -> torch.Tensor:
        """Plan batched arm trajectories for all environments."""
        arm_dof = self.dof if arm_dof is None else arm_dof
        trajectory = torch.zeros(
            size=(self.n_envs, n_waypoints, arm_dof),
            dtype=torch.float32,
            device=self.device,
        )
        options = self._build_motion_gen_options(start_qpos, n_waypoints)
        for i, target_states in enumerate(target_states_list):
            options.start_qpos = start_qpos[i]
            result = self.plan_trajectory(target_states, options)
            trajectory[i, :, :] = result.positions
        return trajectory

    def _interpolate_hand_qpos(
        self,
        start_hand_qpos: torch.Tensor,
        end_hand_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate hand joint positions between two gripper states."""
        weights = torch.linspace(0, 1, steps=n_waypoints, device=self.device)
        hand_qpos_list = [
            torch.lerp(start_hand_qpos, end_hand_qpos, weight) for weight in weights
        ]
        return torch.stack(hand_qpos_list, dim=0)

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
            logger.log_warning(
                "Failed to resolve grasp pose, using default approach pose"
            )
            return False, torch.empty(0), self.arm_joint_ids

        target_states_list = [
            [
                PlanState(xpos=move_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        trajectory = self._plan_arm_trajectory(
            target_states_list, start_qpos, self.cfg.sample_interval
        )
        return True, trajectory, self.arm_joint_ids

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True


@configclass
class PickUpActionCfg(MoveActionCfg):
    hand_open_qpos: torch.Tensor | None = None
    hand_close_qpos: torch.Tensor | None = None
    hand_control_part: str = "hand"
    pre_grasp_distance: float = 0.15
    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)
    lift_height: float = 0.1
    sample_interval: int = 80
    hand_interp_steps: int = 5


class PickUpAction(MoveAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: PickUpActionCfg | None = None,
    ):
        cfg = cfg if cfg is not None else PickUpActionCfg()
        super().__init__(motion_generator, cfg)
        self.cfg = cfg
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
        if isinstance(target, ObjectSemantics):
            is_success, grasp_xpos, open_length = self._resolve_grasp_pose(target)
        else:
            is_success, grasp_xpos = self._resolve_pose_target(
                target, action_name=self.__class__.__name__
            )

        # TODO: warning and fallback if no valid grasp pose found
        if not is_success:
            logger.log_warning(
                "Failed to resolve grasp pose, using default approach pose"
            )
            return False, torch.empty(0), self.joint_ids

        # Compute pre-grasp pose
        pre_grasp_xpos = self._compute_pre_grasp_xpos(grasp_xpos)

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
        pick_trajectory[:, :, : self.arm_dof] = self._plan_arm_trajectory(
            target_states_list,
            start_qpos,
            n_approach_waypoint,
            self.arm_dof,
        )
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
        lift_xpos = self._compute_lift_xpos(grasp_xpos)
        target_states_list = [
            [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        lift_trajectory[:, :, : self.arm_dof] = self._plan_arm_trajectory(
            target_states_list,
            grasp_qpos,
            n_lift_waypoint,
            self.arm_dof,
        )
        # padding hand close qpos to lift trajectory
        lift_trajectory[:, :, self.arm_dof :] = self.hand_close_qpos

        # concatenate trajectories
        trajectory = torch.cat(
            [pick_trajectory, hand_close_trajectory, lift_trajectory], dim=1
        )
        return True, trajectory, self.joint_ids

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

        is_success, grasp_xpos, open_length = semantics.affordance.get_best_grasp_poses(
            obj_poses=obj_poses, approach_direction=self.approach_direction
        )
        return is_success, grasp_xpos, open_length

    def _compute_pre_grasp_xpos(self, grasp_xpos: torch.Tensor) -> torch.Tensor:
        offsets = -self.approach_direction * self.cfg.pre_grasp_distance
        pre_grasp_xpos = grasp_xpos.clone()
        pre_grasp_xpos[:, :3, 3] += offsets
        return pre_grasp_xpos

    def _compute_lift_xpos(self, xpos: torch.Tensor) -> torch.Tensor:
        lift_xpos = xpos.clone()
        lift_xpos[:, 2, 3] += self.cfg.lift_height
        return lift_xpos

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True


@configclass
class PlaceActionCfg(MoveActionCfg):
    hand_open_qpos: torch.Tensor | None = None
    hand_close_qpos: torch.Tensor | None = None
    hand_control_part: str = "hand"
    lift_height: float = 0.1
    sample_interval: int = 80
    hand_interp_steps: int = 5


class PlaceAction(MoveAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: PlaceActionCfg | None = None,
    ):
        cfg = cfg if cfg is not None else PlaceActionCfg()
        super().__init__(motion_generator, cfg)
        self.cfg = cfg
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
        is_success, place_xpos = self._resolve_pose_target(
            target, action_name=self.__class__.__name__
        )
        start_qpos = self._resolve_start_qpos(start_qpos, self.arm_dof)

        # TODO: warning and fallback if no valid grasp pose found
        if not is_success:
            logger.log_warning(
                "Failed to resolve grasp pose, using default approach pose"
            )
            return False, torch.empty(0), self.joint_ids

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
        lift_xpos = self._compute_lift_xpos(place_xpos)
        target_states_list = [
            [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=place_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        down_trajectory[:, :, : self.arm_dof] = self._plan_arm_trajectory(
            target_states_list,
            start_qpos,
            n_down_waypoint,
            self.arm_dof,
        )
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
        lift_trajectory = torch.zeros(
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
        lift_trajectory[:, :, : self.arm_dof] = self._plan_arm_trajectory(
            target_states_list,
            reach_qpos,
            n_lift_waypoint,
            self.arm_dof,
        )
        # padding hand open qpos to lift trajectory
        lift_trajectory[:, :, self.arm_dof :] = self.hand_open_qpos

        # concatenate trajectories
        trajectory = torch.cat(
            [down_trajectory, hand_open_trajectory, lift_trajectory], dim=1
        )
        return True, trajectory, self.joint_ids

    def _compute_lift_xpos(self, xpos: torch.Tensor) -> torch.Tensor:
        lift_xpos = xpos.clone()
        lift_xpos[:, 2, 3] += self.cfg.lift_height
        return lift_xpos

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True
