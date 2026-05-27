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
from .semantic_grasp import SemanticGraspCandidatePlan, select_ranked_semantic_grasp
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

    def _resolve_qpos_target(
        self,
        target: torch.Tensor,
        *,
        name: str,
    ) -> torch.Tensor:
        """Resolve a joint-space target into batched control-part qpos."""
        target = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        if target.shape == (self.dof,):
            target = target.unsqueeze(0).repeat(self.n_envs, 1)
        if target.shape != (self.n_envs, self.dof):
            logger.log_error(
                f"{name} must have shape ({self.dof},) or "
                f"({self.n_envs}, {self.dof}), but got {target.shape}",
                ValueError,
            )
        return target

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
        weights = torch.linspace(0, 1, steps=n_waypoints, device=self.device)
        hand_qpos_list = [
            torch.lerp(start_hand_qpos, end_hand_qpos, weight) for weight in weights
        ]
        return torch.stack(hand_qpos_list, dim=0)

    def _interpolate_qpos(
        self,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        """Interpolate any control-part qpos target."""
        if n_waypoints < 1:
            logger.log_error("sample_interval must be >= 1.", ValueError)
        weights = torch.linspace(0, 1, steps=n_waypoints, device=self.device)
        return torch.stack(
            [torch.lerp(start_qpos, target_qpos, weight) for weight in weights],
            dim=1,
        )

    def execute(
        self,
        target: Union[ObjectSemantics, torch.Tensor],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[bool, torch.Tensor, list[float]]:
        """Execute an end-effector pose move or control-part qpos interpolation.

        Args:
            target (ObjectSemantics): object semantics containing grasp affordance and entity information
            start_qpos (Optional[torch.Tensor], optional): Planning start qpos. Defaults to None.

        Returns:
            tuple[bool, torch.Tensor, list[float]]: Success flag, trajectory of
            shape ``(n_envs, n_waypoints, dof)``, and joint ids corresponding
            to the trajectory columns.
        """
        if isinstance(target, torch.Tensor) and target.shape in {
            (self.dof,),
            (self.n_envs, self.dof),
        }:
            start_qpos = self._resolve_start_qpos(start_qpos)
            target_qpos = self._resolve_qpos_target(target, name="target_qpos")
            trajectory = self._interpolate_qpos(
                start_qpos,
                target_qpos,
                int(self.cfg.sample_interval),
            )
            return True, trajectory, self.arm_joint_ids

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
    in the world frame. Default [0, 0, -1] means approaching from above."""

    grasp_candidate_num: int = 8
    """Number of semantic grasp candidates to retry before reporting failure."""

    grasp_pose_offset_world: torch.Tensor = torch.tensor([0, 0, 0], dtype=torch.float32)
    """Optional world-frame xyz offset applied to resolved grasp poses before planning."""

    grasp_pose_offset_along_approach: float = 0.0
    """Optional scalar offset along approach_direction before planning."""

    ranked_grasp_selection: bool = False
    """Whether to rank planned semantic grasp candidates before selecting one."""

    grasp_approach_directions: list[tuple[str, torch.Tensor]] | None = None
    """Optional labeled approach directions used by ranked semantic grasp selection."""

    grasp_rank_options: dict[str, Any] = {}
    """Additional ranked semantic grasp selection options."""


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
        cfg = cfg if cfg is not None else PickUpActionCfg()
        super().__init__(motion_generator, cfg=cfg)
        self.cfg = cfg
        self.approach_direction = self.cfg.approach_direction.to(self.device)
        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PickUpActionCfg")
        if self.cfg.hand_close_qpos is None:
            logger.log_error("hand_close_qpos must be specified in PickUpActionCfg")
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)
        self.grasp_pose_offset_world = self.cfg.grasp_pose_offset_world.to(self.device)

        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.joint_ids = self.arm_joint_ids + self.hand_joint_ids
        self.arm_dof = len(self.arm_joint_ids)
        self.dof = len(self.joint_ids)
        self.last_selected_grasp: SemanticGraspCandidatePlan | None = None
        self.last_grasp_failures: list[str] = []

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

        self.last_selected_grasp = None
        self.last_grasp_failures = []

        if isinstance(target, ObjectSemantics) and self.cfg.ranked_grasp_selection:
            start_qpos = self._resolve_start_qpos(start_qpos, self.arm_dof)
            options = dict(self.cfg.grasp_rank_options or {})
            if self.cfg.grasp_approach_directions:
                options.setdefault(
                    "approach_directions",
                    self.cfg.grasp_approach_directions,
                )
            selection = select_ranked_semantic_grasp(
                action=self,
                target=target,
                start_qpos=start_qpos,
                options=options,
            )
            self.last_selected_grasp = selection.candidate
            self.last_grasp_failures = list(selection.failures)
            if selection.is_success:
                return True, selection.trajectory, selection.joint_ids
            logger.log_warning(
                "Failed to plan pickup with ranked semantic grasp candidates."
            )
            return False, selection.trajectory, self.joint_ids

        # Resolve grasp pose
        if isinstance(target, ObjectSemantics):
            is_success, grasp_xpos, open_length = self._resolve_grasp_pose_candidates(
                target
            )
        else:
            is_success, grasp_xpos = self._resolve_pose_target(
                target, action_name=self.__class__.__name__
            )
            grasp_xpos = grasp_xpos.unsqueeze(1)
            open_length = torch.ones(
                self.n_envs, 1, dtype=torch.float32, device=self.device
            )

        # TODO: warning and fallback if no valid grasp pose found
        if not bool(torch.as_tensor(is_success, device=self.device).all().item()):
            logger.log_warning(
                "Failed to resolve grasp pose, using default approach pose"
            )
            return False, torch.empty(0), self.joint_ids

        start_qpos = self._resolve_start_qpos(start_qpos, self.arm_dof)
        candidate_num = grasp_xpos.shape[1]
        last_failed_trajectory = torch.empty(0, device=self.device)
        successful_candidates: list[tuple[float, int, torch.Tensor]] = []
        for candidate_idx in range(candidate_num):
            if not torch.all(open_length[:, candidate_idx] > 0):
                continue
            candidate_grasp_xpos = grasp_xpos[:, candidate_idx]
            is_plan_success, trajectory = self._plan_candidate_pickup(
                candidate_grasp_xpos,
                start_qpos,
                candidate_idx=candidate_idx,
            )
            if is_plan_success:
                final_arm_qpos = trajectory[:, -1, : self.arm_dof]
                qpos_distance = torch.linalg.norm(final_arm_qpos - start_qpos, dim=-1)
                score = float(qpos_distance.mean().item()) + 0.01 * candidate_idx
                successful_candidates.append((score, candidate_idx, trajectory))
            last_failed_trajectory = trajectory

        if successful_candidates:
            _, selected_idx, selected_trajectory = min(
                successful_candidates, key=lambda item: item[0]
            )
            logger.log_info(
                f"Selected grasp candidate {selected_idx} from "
                f"{len(successful_candidates)} planned candidates."
            )
            return True, selected_trajectory, self.joint_ids

        logger.log_warning(
            f"Failed to plan pickup with {candidate_num} grasp candidates."
        )
        return False, last_failed_trajectory, self.joint_ids

    def _plan_candidate_pickup(
        self,
        grasp_xpos: torch.Tensor,
        start_qpos: torch.Tensor,
        *,
        candidate_idx: int,
    ) -> tuple[bool, torch.Tensor]:
        """Plan pickup for one resolved grasp pose candidate."""
        grasp_xpos = self._apply_grasp_pose_offset(grasp_xpos)
        # TODO: only for parallel gripper, approach in negative grasp z direction
        grasp_z = grasp_xpos[:, :3, 2]
        pre_grasp_xpos = self._apply_offset(
            pose=grasp_xpos,
            offset=-grasp_z * self.cfg.pre_grasp_distance,
        )

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
            logger.log_warning(
                f"Failed to plan approach trajectory for grasp candidate {candidate_idx}."
            )
            return False, pick_trajectory
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
            logger.log_warning(
                f"Failed to plan lift trajectory for grasp candidate {candidate_idx}."
            )
            return False, lift_trajectory
        lift_trajectory[:, :, : self.arm_dof] = plan_traj
        # padding hand close qpos to lift trajectory
        lift_trajectory[:, :, self.arm_dof :] = self.hand_close_qpos

        # concatenate trajectories
        trajectory = torch.cat(
            [pick_trajectory, hand_close_trajectory, lift_trajectory], dim=1
        )
        return True, trajectory

    def _apply_grasp_pose_offset(self, grasp_xpos: torch.Tensor) -> torch.Tensor:
        offset = self.grasp_pose_offset_world.to(
            dtype=grasp_xpos.dtype,
            device=grasp_xpos.device,
        )
        if self.cfg.grasp_pose_offset_along_approach:
            approach_direction = self.approach_direction.to(
                dtype=grasp_xpos.dtype,
                device=grasp_xpos.device,
            )
            approach_direction = approach_direction / torch.linalg.norm(
                approach_direction
            ).clamp_min(1e-6)
            offset = offset + approach_direction * float(
                self.cfg.grasp_pose_offset_along_approach
            )
        if torch.linalg.norm(offset).item() == 0.0:
            return grasp_xpos
        shifted_grasp_xpos = grasp_xpos.clone()
        shifted_grasp_xpos[:, :3, 3] = shifted_grasp_xpos[:, :3, 3] + offset
        return shifted_grasp_xpos

    def _resolve_grasp_pose_candidates(
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

        is_success, grasp_xpos, open_length = (
            semantics.affordance.get_grasp_pose_candidates(
                obj_poses=obj_poses,
                approach_direction=self.approach_direction,
                top_k=self.cfg.grasp_candidate_num,
            )
        )
        return is_success, grasp_xpos, open_length

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True


@configclass
class PlaceActionCfg(GraspActionCfg):
    name: str = "place"
    """Name of the action, used for identification and logging."""

    post_open_wait_steps: int = 0
    """Number of waypoints to hold the arm still after opening the gripper.

    Keeping the end-effector stationary for a short period after release avoids
    immediately dragging or knocking an object over during the retreat phase.
    """


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

        post_open_wait_steps = max(0, int(self.cfg.post_open_wait_steps))
        post_open_wait_trajectory = torch.empty(
            size=(self.n_envs, 0, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        if post_open_wait_steps > 0:
            post_open_wait_trajectory = torch.zeros(
                size=(self.n_envs, post_open_wait_steps, self.dof),
                dtype=torch.float32,
                device=self.device,
            )
            post_open_wait_trajectory[:, :, : self.arm_dof] = reach_qpos
            post_open_wait_trajectory[:, :, self.arm_dof :] = self.hand_open_qpos

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
            [
                down_trajectory,
                hand_open_trajectory,
                post_open_wait_trajectory,
                back_trajectory,
            ],
            dim=1,
        )
        return True, trajectory, self.joint_ids

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True
