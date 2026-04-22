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


# =============================================================================
# Reach Action
# =============================================================================


class ReachAction(AtomicAction):
    """Atomic action for reaching a target pose or object."""

    def __init__(
        self,
        motion_generator: "MotionGenerator",
        robot: "Robot",
        control_part: str,
        device: torch.device = torch.device("cuda"),
        interpolation_type: str = "linear",  # "linear", "cubic", "toppra"
    ):
        super().__init__(motion_generator)
        self.interpolation_type = interpolation_type

    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        approach_offset: Optional[torch.Tensor] = None,
        use_affordance: bool = True,
        **kwargs,
    ) -> PlanResult:
        """Execute reach action.

        Args:
            target: Target pose [4, 4] or ObjectSemantics
            start_qpos: Starting joint configuration
            approach_offset: Offset for pre-grasp approach [x, y, z]
            use_affordance: Whether to use object's affordance data

        Returns:
            PlanResult with trajectory and execution status
        """
        # Resolve target pose from ObjectSemantics if needed
        if isinstance(target, ObjectSemantics):
            target_pose = self._resolve_target_pose(target, use_affordance)
        else:
            target_pose = target

        # Apply approach offset if specified
        if approach_offset is not None:
            approach_pose = self._apply_offset(target_pose, approach_offset)
        else:
            approach_pose = target_pose

        # Get current state if not provided
        if start_qpos is None:
            start_qpos = self._get_current_qpos()

        # Create plan states
        target_states = [
            PlanState(qpos=start_qpos, move_type=MoveType.JOINT_MOVE),
            PlanState(xpos=approach_pose, move_type=MoveType.EEF_MOVE),
        ]

        # Plan trajectory
        options = MotionGenOptions(
            control_part=self.control_part,
            is_interpolate=True,
            is_linear=self.interpolation_type == "linear",
            interpolate_position_step=0.002,
            plan_opts=ToppraPlanOptions(
                sample_interval=kwargs.get("sample_interval", 30),
            ),
        )

        result = self.plan_trajectory(target_states, options)

        # Return PlanResult directly
        return result

    def validate(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> bool:
        """Check if the reach action is feasible."""
        try:
            # Quick IK feasibility check
            if isinstance(target, ObjectSemantics):
                target_pose = self._resolve_target_pose(target, use_affordance=True)
            else:
                target_pose = target

            # Attempt IK
            qpos_seed = (
                start_qpos if start_qpos is not None else self._get_current_qpos()
            )
            success, _ = self.robot.compute_ik(
                pose=target_pose.unsqueeze(0),
                qpos_seed=qpos_seed.unsqueeze(0),
                name=self.control_part,
            )
            return success.all().item()
        except Exception:
            return False

    def _resolve_target_pose(
        self, semantics: ObjectSemantics, use_affordance: bool
    ) -> torch.Tensor:
        """Resolve target pose from object semantics."""
        from .core import GraspPose

        if use_affordance and isinstance(semantics.affordance, GraspPose):
            # Use precomputed grasp pose from affordance data
            grasp_pose = semantics.affordance.get_best_grasp()
            object_pose = self._get_object_pose(semantics.label)
            target_pose = object_pose @ grasp_pose
        else:
            # Default to object center with approach direction
            object_pose = self._get_object_pose(semantics.label)
            approach_offset = torch.tensor([0, 0, 0.05], device=self.device)
            target_pose = object_pose.clone()
            target_pose[:3, 3] += approach_offset

        return target_pose

    def _get_object_pose(self, label: str) -> torch.Tensor:
        """Get current pose of object by label."""
        # Implementation depends on environment's object management
        # This is a placeholder - should be implemented based on environment
        raise NotImplementedError(
            "_get_object_pose must be implemented by subclass or "
            "provided with environment-specific object management"
        )


@configclass
class PickUpActionCfg(ActionCfg):
    hand_open_qpos: torch.Tensor | None = None
    hand_close_qpos: torch.Tensor | None = None
    hand_control_part: str = "hand"
    pre_grasp_distance: float = 0.15
    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)
    lift_height: float = 0.1
    sample_interval: int = 80
    hand_interp_steps: int = 5


class PickUpAction(AtomicAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: PickUpActionCfg | None = None,
    ):
        super().__init__(motion_generator)
        self.cfg = cfg if cfg is not None else PickUpActionCfg()
        self.approach_direction = self.cfg.approach_direction.to(self.device)
        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PickUpActionCfg")
        if self.cfg.hand_close_qpos is None:
            logger.log_error("hand_close_qpos must be specified in PickUpActionCfg")
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
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
        elif isinstance(target, torch.Tensor):
            if target.shape == (4, 4):
                target = target.unsqueeze(0).repeat(self.n_envs, 1, 1)  # (n_envs, 4, 4)
            if target.shape != (self.n_envs, 4, 4):
                logger.log_error(
                    f"Target tensor must have shape (4, 4) or ({self.n_envs}, 4, 4), but got {target.shape}",
                    ValueError,
                )
            is_success = True
            grasp_xpos = target
        else:
            logger.log_error(
                "Target must be either ObjectSemantics or torch.Tensor of shape (4, 4) or (n_envs, 4, 4)",
                TypeError,
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
        if start_qpos is None:
            start_qpos = self.robot.get_qpos(name=self.cfg.control_part)
        if start_qpos.shape == (self.arm_dof,):
            # expand to (n_envs, arm_dof)
            start_qpos = start_qpos.unsqueeze(0).repeat(self.n_envs, 1)
        if start_qpos.shape != (self.n_envs, self.arm_dof):
            logger.log_error(
                f"start_qpos must have shape ({self.n_envs}, {self.arm_dof}), "
                f"but got {start_qpos.shape}",
                ValueError,
            )

        # n_waypoint = kwargs.get("sample_interval", 30)
        n_approach_waypoint = int(
            np.round(self.cfg.sample_interval - self.cfg.hand_interp_steps) * 0.6
        )
        n_close_waypoint = self.cfg.hand_interp_steps
        n_lift_waypoint = (
            self.cfg.sample_interval - n_approach_waypoint - n_close_waypoint
        )

        # TODO: passing options from arguments
        options = MotionGenOptions(
            start_qpos=start_qpos[0],
            control_part=self.cfg.control_part,
            is_interpolate=True,
            is_linear=False,
            interpolate_position_step=0.001,
            plan_opts=ToppraPlanOptions(
                sample_interval=n_approach_waypoint,
            ),
        )

        # get pick trajectory
        pick_trajectory = torch.zeros(
            size=(self.n_envs, n_approach_waypoint, self.dof),
            dtype=torch.float32,
            device=self.device,
        )
        options.plan_opts = ToppraPlanOptions(sample_interval=n_approach_waypoint)
        for i in range(self.n_envs):
            target_states = [
                PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            # Use specified start_qpos for each environment robot instance
            options.start_qpos = start_qpos[i]
            result = self.plan_trajectory(target_states, options)
            pick_trajectory[i, :, : self.arm_dof] = result.positions
        # Padding hand open qpos to pick trajectory
        pick_trajectory[:, :, self.arm_dof :] = self.hand_open_qpos

        # get hand closing trajectory
        grasp_qpos = pick_trajectory[
            :, -1, : self.arm_dof
        ]  # Assuming the last point of pick trajectory is the grasp pose
        weights = torch.linspace(
            0, 1, steps=self.cfg.hand_interp_steps, device=self.device
        )
        hand_qpos_list = [
            torch.lerp(self.hand_open_qpos, self.hand_close_qpos, w) for w in weights
        ]
        hand_close_path = torch.stack(
            hand_qpos_list, dim=0
        )  # (hand_interp_steps, hand_dof)
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
        options.plan_opts = ToppraPlanOptions(sample_interval=n_lift_waypoint)
        # TODO: batch planning
        for i in range(self.n_envs):
            target_states = [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            # Use specified start_qpos for each environment robot instance
            options.start_qpos = grasp_qpos[i]
            result = self.plan_trajectory(target_states, options)
            lift_trajectory[i, :, : self.arm_dof] = result.positions
        # padding hand close qpos to lift trajectory
        lift_trajectory[:, :, self.arm_dof :] = self.hand_close_qpos

        # concatenate trajectories
        trajectory = torch.cat(
            [pick_trajectory, hand_close_trajectory, lift_trajectory], dim=1
        )
        return True, trajectory, self.joint_ids

    def _pack_trajectory(self):
        pass

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
class PlaceActionCfg(ActionCfg):
    hand_open_qpos: torch.Tensor | None = None
    hand_close_qpos: torch.Tensor | None = None
    hand_control_part: str = "hand"
    lift_height: float = 0.1
    sample_interval: int = 80
    hand_interp_steps: int = 5


class PlaceAction(AtomicAction):
    def __init__(
        self,
        motion_generator: MotionGenerator,
        cfg: PlaceActionCfg | None = None,
    ):
        super().__init__(motion_generator)
        self.cfg = cfg if cfg is not None else PlaceActionCfg()
        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PlaceActionCfg")
        if self.cfg.hand_close_qpos is None:
            logger.log_error("hand_close_qpos must be specified in PlaceActionCfg")
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
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
        if isinstance(target, ObjectSemantics):
            # TODO: place semantics
            logger.log_error(
                "PlaceAction currently does not support ObjectSemantics target. Please provide target pose as torch.Tensor of shape (4, 4) or (n_envs, 4, 4)",
                NotImplementedError,
            )
        elif isinstance(target, torch.Tensor):
            if target.shape == (4, 4):
                target = target.unsqueeze(0).repeat(self.n_envs, 1, 1)  # (n_envs, 4, 4)
            if target.shape != (self.n_envs, 4, 4):
                logger.log_error(
                    f"Target tensor must have shape (4, 4) or ({self.n_envs}, 4, 4), but got {target.shape}",
                    ValueError,
                )
            is_success = True
            place_xpos = target
        else:
            logger.log_error(
                "Target must be either ObjectSemantics or torch.Tensor of shape (4, 4) or (n_envs, 4, 4)",
                TypeError,
            )

        # TODO: warning and fallback if no valid grasp pose found
        if not is_success:
            logger.log_warning(
                "Failed to resolve grasp pose, using default approach pose"
            )
            return False, torch.empty(0), self.joint_ids

        lift_xpos = self._compute_lift_xpos(place_xpos)

    def _compute_lift_xpos(self, xpos: torch.Tensor) -> torch.Tensor:
        lift_xpos = xpos.clone()
        lift_xpos[:, 2, 3] += self.cfg.lift_height
        return lift_xpos

    def validate(self, target, start_qpos=None, **kwargs):
        # TODO: implement proper validation logic for pick up action
        return True


class GraspAction(AtomicAction):
    """Atomic action for grasping objects."""

    def __init__(
        self,
        motion_generator: MotionGenerator,
        control_part: str,
        pre_grasp_distance: float = 0.05,
        approach_direction: str = "z",  # "x", "y", "z", or "custom"
    ):
        super().__init__(motion_generator)
        self.pre_grasp_distance = pre_grasp_distance
        self.approach_direction = approach_direction

    def execute(
        self,
        target: ObjectSemantics,
        start_qpos: Optional[torch.Tensor] = None,
        use_affordance: bool = True,
        grasp_type: str = "default",  # "default", "pinch", "power"
        **kwargs,
    ) -> PlanResult:
        """Execute grasp action.

        Args:
            target: ObjectSemantics with grasp affordances
            start_qpos: Starting joint configuration
            use_affordance: Whether to use precomputed grasp poses
            grasp_type: Type of grasp to execute
        """
        # Resolve grasp pose
        grasp_pose = self._resolve_grasp_pose(target, use_affordance, grasp_type)

        # Compute pre-grasp pose (approach position)
        pre_grasp_pose = self._compute_pre_grasp_pose(grasp_pose)

        # Get current state
        if start_qpos is None:
            start_qpos = self._get_current_qpos()

        # Build trajectory plan states
        target_states = [
            PlanState(qpos=start_qpos, move_type=MoveType.JOINT_MOVE),
            PlanState(xpos=pre_grasp_pose, move_type=MoveType.EEF_MOVE),
            PlanState(xpos=grasp_pose, move_type=MoveType.EEF_MOVE),
        ]

        # Plan trajectory
        options = MotionGenOptions(
            control_part=self.control_part,
            is_interpolate=True,
            is_linear=False,
            interpolate_position_step=0.001,
            plan_opts=ToppraPlanOptions(
                sample_interval=kwargs.get("sample_interval", 30),
            ),
        )

        result = self.plan_trajectory(target_states, options)

        # Return PlanResult directly - it contains all trajectory data
        return result

    def validate(
        self,
        target: ObjectSemantics,
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> bool:
        """Validate if grasp is feasible."""
        try:
            grasp_pose = self._resolve_grasp_pose(
                target, use_affordance=True, grasp_type="default"
            )
            qpos_seed = (
                start_qpos if start_qpos is not None else self._get_current_qpos()
            )
            success, _ = self.robot.compute_ik(
                pose=grasp_pose.unsqueeze(0),
                qpos_seed=qpos_seed.unsqueeze(0),
                name=self.control_part,
            )
            return success.all().item()
        except Exception:
            return False

    def _resolve_grasp_pose(
        self, semantics: ObjectSemantics, use_affordance: bool, grasp_type: str
    ) -> torch.Tensor:
        """Resolve grasp pose from object semantics."""
        from .core import GraspPose

        if use_affordance and isinstance(semantics.affordance, GraspPose):
            grasp_pose_affordance = semantics.affordance
            grasp_pose = grasp_pose_affordance.get_grasp_by_type(grasp_type)
            if grasp_pose is None:
                grasp_pose = grasp_pose_affordance.get_best_grasp()

            # Transform to world frame
            object_pose = self._get_object_pose(semantics.label)
            return object_pose @ grasp_pose

        # Fallback: compute grasp pose from geometry
        return self._compute_grasp_from_geometry(semantics)

    def _compute_pre_grasp_pose(self, grasp_pose: torch.Tensor) -> torch.Tensor:
        """Compute pre-grasp pose with offset."""
        offset = torch.zeros(3, device=self.device)
        if self.approach_direction == "z":
            offset[2] = -self.pre_grasp_distance
        elif self.approach_direction == "x":
            offset[0] = -self.pre_grasp_distance
        elif self.approach_direction == "y":
            offset[1] = -self.pre_grasp_distance

        pre_grasp = grasp_pose.clone()
        pre_grasp[:3, 3] += grasp_pose[:3, :3] @ offset
        return pre_grasp

    def _get_object_pose(self, label: str) -> torch.Tensor:
        """Get current pose of object by label."""
        raise NotImplementedError(
            "_get_object_pose must be implemented by subclass or "
            "provided with environment-specific object management"
        )

    def _compute_grasp_from_geometry(self, semantics: ObjectSemantics) -> torch.Tensor:
        """Compute grasp pose from object geometry."""
        # Get object pose
        object_pose = self._get_object_pose(semantics.label)

        # Get bounding box from geometry
        bbox = semantics.geometry.get("bounding_box", [0.1, 0.1, 0.1])

        # Default top-down grasp
        grasp_offset = torch.eye(4, device=self.device)
        grasp_offset[2, 3] = bbox[2] / 2 + 0.02  # Slightly above object

        return object_pose @ grasp_offset


# =============================================================================
# Move Action
# =============================================================================


class MoveAction(AtomicAction):
    """Atomic action for moving to a target position."""

    def __init__(
        self,
        motion_generator: "MotionGenerator",
        robot: "Robot",
        control_part: str,
        device: torch.device = torch.device("cuda"),
        move_type: str = "cartesian",  # "cartesian", "joint"
        interpolation: str = "linear",  # "linear", "cubic", "toppra"
    ):
        super().__init__(motion_generator)
        self.move_type = move_type
        self.interpolation = interpolation

    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        velocity_limit: Optional[float] = None,
        acceleration_limit: Optional[float] = None,
    ) -> PlanResult:
        """Execute move action.

        Args:
            target: Target pose [4, 4] or ObjectSemantics
            start_qpos: Starting joint configuration
            offset: Optional offset from target
            velocity_limit: Max velocity for trajectory
            acceleration_limit: Max acceleration for trajectory
        """
        # Resolve target
        if isinstance(target, ObjectSemantics):
            target_pose = self._get_object_pose(target.label)
        else:
            target_pose = target

        # Apply offset if specified
        if offset is not None:
            target_pose = self._apply_offset(target_pose, offset)

        # Get start state
        if start_qpos is None:
            start_qpos = self._get_current_qpos()

        # Create plan states based on move type
        if self.move_type == "cartesian":
            target_states = [
                PlanState(qpos=start_qpos, move_type=MoveType.JOINT_MOVE),
                PlanState(xpos=target_pose, move_type=MoveType.EEF_MOVE),
            ]
            is_linear = self.interpolation == "linear"
        else:  # joint space
            target_qpos = self._ik_solve(target_pose, start_qpos)
            target_states = [
                PlanState(qpos=start_qpos, move_type=MoveType.JOINT_MOVE),
                PlanState(qpos=target_qpos, move_type=MoveType.JOINT_MOVE),
            ]
            is_linear = False

        # Configure motion generation
        options = MotionGenOptions(
            control_part=self.control_part,
            is_interpolate=True,
            is_linear=is_linear,
            interpolate_position_step=0.002,
            plan_opts=ToppraPlanOptions(
                sample_interval=kwargs.get("sample_interval", 30),
                velocity_limit=velocity_limit,
                acceleration_limit=acceleration_limit,
            ),
        )

        result = self.plan_trajectory(target_states, options)

        # Return PlanResult directly - it contains all trajectory data
        return result

    def validate(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> bool:
        """Validate if move action is feasible."""
        try:
            if isinstance(target, ObjectSemantics):
                target_pose = self._get_object_pose(target.label)
            else:
                target_pose = target

            qpos_seed = (
                start_qpos if start_qpos is not None else self._get_current_qpos()
            )

            if self.move_type == "joint":
                # For joint space moves, we need IK solvability
                self._ik_solve(target_pose, qpos_seed)
            else:
                # For cartesian moves, just check IK
                success, _ = self.robot.compute_ik(
                    pose=target_pose.unsqueeze(0),
                    qpos_seed=qpos_seed.unsqueeze(0),
                    name=self.control_part,
                )
                if not success.all():
                    return False

            return True
        except Exception:
            return False

    def _get_object_pose(self, label: str) -> torch.Tensor:
        """Get current pose of object by label."""
        raise NotImplementedError(
            "_get_object_pose must be implemented by subclass or "
            "provided with environment-specific object management"
        )


# =============================================================================
# Release Action
# =============================================================================


class ReleaseAction(AtomicAction):
    """Atomic action for releasing an object."""

    def execute(
        self,
        target: Optional[Union[torch.Tensor, ObjectSemantics]] = None,
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PlanResult:
        """Execute release action.

        Args:
            target: Optional target pose after release (for place operations)
            start_qpos: Starting joint configuration

        Returns:
            PlanResult with trajectory (may be empty for simple release)
        """
        # Get current state
        if start_qpos is None:
            start_qpos = self._get_current_qpos()

        # If target is specified, move to that pose first
        if target is not None:
            if isinstance(target, ObjectSemantics):
                # Move above the object
                target_pose = self._get_object_pose(target.label)
                approach_offset = torch.tensor([0, 0, 0.1], device=self.device)
                target_pose = self._apply_offset(target_pose, approach_offset)
            else:
                target_pose = target

            target_states = [
                PlanState(qpos=start_qpos, move_type=MoveType.JOINT_MOVE),
                PlanState(xpos=target_pose, move_type=MoveType.EEF_MOVE),
            ]

            options = MotionGenOptions(
                control_part=self.control_part,
                is_interpolate=True,
                is_linear=False,
                interpolate_position_step=0.002,
                plan_opts=ToppraPlanOptions(
                    sample_interval=kwargs.get("sample_interval", 30),
                ),
            )

            result = self.plan_trajectory(target_states, options)
        else:
            # Simple release - return success with current state
            result = PlanResult(
                success=True,
                positions=start_qpos.unsqueeze(0),
            )

        # Open gripper (if applicable)
        # This would be robot-specific and should be implemented by subclasses
        self._open_gripper()

        return result

    def validate(
        self,
        target: Optional[Union[torch.Tensor, ObjectSemantics]] = None,
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> bool:
        """Validate if release action is feasible."""
        # Release is generally always feasible
        # If target is specified, validate that we can reach it
        if target is not None and isinstance(target, torch.Tensor):
            try:
                qpos_seed = (
                    start_qpos if start_qpos is not None else self._get_current_qpos()
                )
                success, _ = self.robot.compute_ik(
                    pose=target.unsqueeze(0),
                    qpos_seed=qpos_seed.unsqueeze(0),
                    name=self.control_part,
                )
                return success.all().item()
            except Exception:
                return False
        return True

    def _open_gripper(self) -> None:
        """Open the gripper to release the object.

        This is a placeholder method that should be implemented by subclasses
        based on the specific robot hardware or simulation environment.
        """
        # Placeholder - should be implemented by subclass
        pass

    def _get_object_pose(self, label: str) -> torch.Tensor:
        """Get current pose of object by label."""
        raise NotImplementedError(
            "_get_object_pose must be implemented by subclass or "
            "provided with environment-specific object management"
        )
