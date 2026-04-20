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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from embodichain.lab.sim.planners import PlanResult, PlanState, MoveType
from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator
    from embodichain.lab.sim.objects import Robot


# =============================================================================
# Affordance Classes
# =============================================================================


@dataclass
class Affordance:
    """Base class for affordance data.

    Affordance represents interaction possibilities for an object.
    This is the base class for specific affordance types.
    """

    object_label: str = ""
    """Label of the object this affordance belongs to."""

    def get_batch_size(self) -> int:
        """Return the batch size of this affordance data."""
        return 1


@dataclass
class GraspPose(Affordance):
    """Grasp pose affordance containing a batch of 4x4 transformation matrices.

    Each grasp pose represents a valid end-effector pose for grasping the object.
    Multiple poses may be available for different grasp types (pinch, power, etc.)
    or approach directions.
    """

    poses: torch.Tensor = field(default_factory=lambda: torch.eye(4).unsqueeze(0))
    """Batch of grasp poses with shape [B, 4, 4].

    Each pose is a 4x4 homogeneous transformation matrix representing
    the end-effector pose in the object's local coordinate frame.
    """

    grasp_types: List[str] = field(default_factory=lambda: ["default"])
    """List of grasp type labels for each pose in the batch.

    Examples: "pinch", "power", "hook", "spherical", etc.
    Length must match the batch dimension of `poses`.
    """

    confidence_scores: torch.Tensor | None = None
    """Optional confidence scores for each grasp pose with shape [B].

    Higher values indicate more reliable/ stable grasps.
    Used for grasp selection when multiple options exist.
    """

    def get_batch_size(self) -> int:
        """Return the number of grasp poses in this affordance."""
        return self.poses.shape[0]

    def get_grasp_by_type(self, grasp_type: str) -> Optional[torch.Tensor]:
        """Get grasp pose by type label.

        Args:
            grasp_type: Type of grasp (e.g., "pinch", "power")

        Returns:
            4x4 pose tensor if found, None otherwise
        """
        if grasp_type in self.grasp_types:
            idx = self.grasp_types.index(grasp_type)
            return self.poses[idx]
        return None

    def get_best_grasp(self) -> torch.Tensor:
        """Get the best grasp pose based on confidence scores.

        Returns:
            4x4 pose tensor with highest confidence
        """
        if self.confidence_scores is not None:
            best_idx = torch.argmax(self.confidence_scores)
            return self.poses[best_idx]
        return self.poses[0]  # Default to first if no scores available


@dataclass
class InteractionPoints(Affordance):
    """Interaction points affordance containing a batch of 3D positions.

    Interaction points define specific locations on an object surface
    that can be used for contact-based interactions (pushing, poking,
    touching) rather than full grasping.
    """

    points: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 3))
    """Batch of 3D interaction points with shape [B, 3].

    Each point is a 3D coordinate in the object's local coordinate frame.
    """

    normals: torch.Tensor | None = None
    """Optional surface normals at each interaction point with shape [B, 3].

    Normals indicate the surface orientation at each point,
    useful for determining approach directions.
    """

    point_types: List[str] = field(default_factory=list)
    """Optional labels for each point's interaction type.

    Examples: "push", "poke", "touch", "pinch"
    """

    def get_points_by_type(self, point_type: str) -> torch.Tensor | None:
        """Get points by their interaction type.

        Args:
            point_type: Type of interaction (e.g., "push", "poke")

        Returns:
            Tensor of points if found, None otherwise
        """
        if point_type in self.point_types:
            indices = [i for i, t in enumerate(self.point_types) if t == point_type]
            return self.points[indices]
        return None

    def get_batch_size(self) -> int:
        """Return the number of interaction points in this affordance."""
        return self.points.shape[0]

    def get_approach_direction(self, point_idx: int) -> torch.Tensor:
        """Get recommended approach direction for a given point.

        Args:
            point_idx: Index of the point

        Returns:
            3D approach direction vector (normalized)
        """
        if self.normals is not None:
            # Approach from the opposite direction of the surface normal
            return -self.normals[point_idx]
        # Default: approach from positive z
        return torch.tensor(
            [0, 0, 1], dtype=self.points.dtype, device=self.points.device
        )


# =============================================================================
# ObjectSemantics
# =============================================================================


@dataclass
class ObjectSemantics:
    """Semantic information about interaction target.

    This class encapsulates all semantic and geometric information about
    an object needed for intelligent interaction planning.
    """

    affordance: Affordance
    """Affordance data (GraspPose, InteractionPoints, etc.)."""

    geometry: Dict[str, Any]
    """Geometric information including bounding box, mesh data."""

    properties: Dict[str, Any] = field(default_factory=dict)
    """Physical properties: mass, friction, etc."""

    label: str = "none"
    """Object category label (e.g., 'apple', 'bottle')."""

    uid: Optional[str] = None
    """Optional unique identifier for the object instance."""


# =============================================================================
# ActionCfg and AtomicAction
# =============================================================================


@configclass
class ActionCfg:
    """Configuration for atomic actions."""

    control_part: str = "left_arm"
    """Control part name for the action."""

    interpolation_type: str = "linear"
    """Interpolation type: 'linear', 'cubic'."""

    velocity_limit: Optional[float] = None
    """Optional velocity limit for the motion."""

    acceleration_limit: Optional[float] = None
    """Optional acceleration limit for the motion."""


class AtomicAction(ABC):
    """Abstract base class for atomic actions.

    All atomic actions use PlanResult from embodichain.lab.sim.planners
    as the return type for execute() method, ensuring consistency with
    the existing motion planning infrastructure.
    """

    def __init__(
        self,
        motion_generator: MotionGenerator,
    ):
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = self.robot.device

    @abstractmethod
    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PlanResult:
        """Execute the atomic action.

        Args:
            target: Target pose [4, 4] or ObjectSemantics
            start_qpos: Starting joint configuration [DOF]
            **kwargs: Additional action-specific parameters

        Returns:
            PlanResult with trajectory (positions, velocities, accelerations),
            end-effector poses (xpos_list), and success status.
            Use result.positions for joint trajectory [T, DOF].
            Use result.xpos_list for EE poses [T, 4, 4].
        """
        pass

    @abstractmethod
    def validate(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> bool:
        """Validate if the action is feasible without executing.

        This method performs a quick feasibility check (e.g., IK solvability)
        without generating a full trajectory.

        Returns:
            True if action appears feasible, False otherwise
        """
        pass

    def _ik_solve(
        self, target_pose: torch.Tensor, qpos_seed: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Solve IK for target pose.

        Args:
            target_pose: Target pose [4, 4]
            qpos_seed: Seed configuration [DOF]

        Returns:
            Joint configuration [DOF]

        Raises:
            RuntimeError: If IK fails to find a solution
        """
        if qpos_seed is None:
            qpos_seed = self.robot.get_qpos()

        success, qpos = self.robot.compute_ik(
            pose=target_pose.unsqueeze(0),
            qpos_seed=qpos_seed.unsqueeze(0),
            name=self.control_part,
        )

        if not success.all():
            raise RuntimeError(f"IK failed for target pose: {target_pose}")

        return qpos.squeeze(0)

    def _fk_compute(self, qpos: torch.Tensor) -> torch.Tensor:
        """Compute forward kinematics.

        Args:
            qpos: Joint configuration [DOF] or [B, DOF]

        Returns:
            End-effector pose [4, 4] or [B, 4, 4]
        """
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)

        xpos = self.robot.compute_fk(
            qpos=qpos,
            name=self.control_part,
            to_matrix=True,
        )

        return xpos.squeeze(0) if xpos.shape[0] == 1 else xpos

    def _apply_offset(self, pose: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Apply offset to pose in local frame.

        Args:
            pose: Base pose [4, 4]
            offset: Offset in local frame [3]

        Returns:
            Pose with offset applied [4, 4]
        """
        result = pose.clone()
        result[:3, 3] += pose[:3, :3] @ offset
        return result

    def plan_trajectory(
        self,
        target_states: List[PlanState],
        options: Optional["MotionGenOptions"] = None,
    ) -> "PlanResult":
        """Plan trajectory using motion generator."""
        from embodichain.lab.sim.planners import MotionGenOptions

        if options is None:
            options = MotionGenOptions(control_part=self.control_part)
        return self.motion_generator.generate(target_states, options)
