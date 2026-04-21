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

from embodichain.toolkits.graspkit.pg_grasp import (
    GraspGenerator,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.lab.sim.common import BatchEntity
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator, MotionGenOptions
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

    geometry: Dict[str, Any] = field(default_factory=dict)
    """Geometry dictionary shared with ObjectSemantics.

    The mesh payload is expected to be stored in:
    - ``mesh_vertices``: torch.Tensor with shape [N, 3]
    - ``mesh_triangles``: torch.Tensor with shape [M, 3]
    """

    custom_config: Dict[str, Any] = field(default_factory=dict)
    """User-defined configuration payload for affordance creation and usage."""

    @property
    def mesh_vertices(self) -> torch.Tensor | None:
        """Get mesh vertices from geometry.

        Returns:
            Mesh vertices tensor [N, 3], or None if unavailable.

        Raises:
            TypeError: If ``mesh_vertices`` exists but is not a torch tensor.
        """
        vertices = self.geometry.get("mesh_vertices")
        if vertices is None:
            return None
        if not isinstance(vertices, torch.Tensor):
            raise TypeError("geometry['mesh_vertices'] must be a torch.Tensor")
        return vertices

    @property
    def mesh_triangles(self) -> torch.Tensor | None:
        """Get mesh triangles from geometry.

        Returns:
            Mesh triangle index tensor [M, 3], or None if unavailable.

        Raises:
            TypeError: If ``mesh_triangles`` exists but is not a torch tensor.
        """
        triangles = self.geometry.get("mesh_triangles")
        if triangles is None:
            return None
        if not isinstance(triangles, torch.Tensor):
            raise TypeError("geometry['mesh_triangles'] must be a torch.Tensor")
        return triangles

    def set_custom_config(self, key: str, value: Any) -> None:
        """Set a custom affordance configuration value."""
        self.custom_config[key] = value

    def get_custom_config(self, key: str, default: Any = None) -> Any:
        """Get a custom affordance configuration value."""
        return self.custom_config.get(key, default)

    def get_batch_size(self) -> int:
        """Return the batch size of this affordance data."""
        return 1


@dataclass
class AntipodalAffordance(Affordance):
    generator: GraspGenerator | None = None
    """Grasp generator instance, initialized lazily when needed."""

    force_reannotate: bool = False
    """Whether to force re-annotation of grasp generator on each access."""

    def _init_generator(self):
        if (
            self.geometry.get("mesh_vertices", None) is None
            or self.geometry.get("mesh_triangles", None) is None
        ):
            logger.log_error(
                "Mesh vertices and triangles must be provided in geometry to initialize AntipodalAffordance."
            )
        self.generator = GraspGenerator(
            vertices=self.geometry.get("mesh_vertices"),
            triangles=self.geometry.get("mesh_triangles"),
            cfg=self.custom_config.get("generator_cfg", None),
            gripper_collision_cfg=self.custom_config.get("gripper_collision_cfg", None),
        )
        if self.force_reannotate:
            self.generator.annotate()
        else:
            if self.generator._hit_point_pairs is None:
                self.generator.annotate()

    def get_best_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.generator is None:
            self._init_generator()

        grasp_xpos_list = []
        is_success_list = []
        open_length_list = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_xpos, open_length = self.generator.get_grasp_poses(
                obj_pose, approach_direction
            )
            if is_success:
                grasp_xpos_list.append(grasp_xpos.unsqueeze(0))
            else:
                logger.log_warning(f"No valid grasp pose found for {i}-th object.")
                grasp_xpos_list.append(
                    torch.eye(
                        4, dtype=torch.float32, device=self.generator.device
                    ).unsqueeze(0)
                )  # Default to identity pose if no grasp found
            is_success_list.append(is_success)
            open_length_list.append(open_length)
        is_success = torch.tensor(
            is_success_list, dtype=torch.bool, device=self.generator.device
        )
        grasp_xpos = torch.concatenate(grasp_xpos_list, dim=0)  # [B, 4, 4]
        open_length = torch.tensor(
            open_length_list, dtype=torch.float32, device=self.generator.device
        )
        return is_success, grasp_xpos, open_length


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

    entity: BatchEntity | None = None
    """Optional reference to the underlying simulation entity representing this object."""

    def __post_init__(self) -> None:
        """Bind affordance metadata to this semantic object.

        The affordance shares the same geometry dict instance as
        ``ObjectSemantics.geometry`` so mesh tensors are authored in one place.
        """
        self.affordance.object_label = self.label
        self.affordance.geometry = self.geometry


# =============================================================================
# ActionCfg and AtomicAction
# =============================================================================


@configclass
class ActionCfg:
    """Configuration for atomic actions."""

    control_part: str = "arm"
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
        cfg: ActionCfg = ActionCfg(),
    ):
        self.motion_generator = motion_generator
        self.cfg = cfg
        self.robot = motion_generator.robot
        self.control_part = cfg.control_part
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
