# Atomic Action Abstraction Design for Embodied AI Motion Generation

**Date:** 2026-04-17
**Status:** Design Draft

---

## 1. Overview

This document describes the design of an **atomic action abstraction layer** for embodied AI motion generation. The design supports both demo simulation and data generation within gym environments, providing a consistent interface for agentic workflows while maintaining extensibility for new motion primitives.

---

## 2. Design Principles

### 2.1 Leverage Existing Infrastructure
- **MotionGenerator**: Use existing motion planning with TOPPRA and interpolation
- **Solvers**: Integrate IK/FK solvers (BaseSolver, PinkSolver, etc.)
- **Warp**: Utilize GPU-accelerated trajectory interpolation from `warp` module

### 2.2 Object Semantics
Atomic actions should consider:
- **Semantic labels**: Object categories (e.g., "apple", "bottle")
- **Affordance data**: Grasp poses, interaction points
- **Geometry**: Shape, dimensions, collision boundaries

### 2.3 Consistent Interface
- Unified API across all motion primitives
- Standard parameter types (torch.Tensor, np.ndarray)
- Predictable return types (PlanResult from motion planning)

### 2.4 Extensibility
- Registry-based motion primitive registration
- Plugin architecture for custom actions
- Strategy pattern for interaction behaviors

---

## 3. Architecture

### 3.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    AtomicActionEngine                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Semantic   │  │   Motion     │  │   Affordance     │  │
│  │   Analyzer   │  │   Planner    │  │   Provider       │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌────────────────┐   ┌──────────────────┐
│ ReachAction   │   │ GraspAction    │   │  MoveAction      │
└───────────────┘   └────────────────┘   └──────────────────┘
```

### 3.2 Class Hierarchy

```python
# Base class for all atomic actions
AtomicAction(ABC)
    ├── ReachAction
    ├── GraspAction
    ├── ReleaseAction
    ├── MoveAction
    ├── RotateAction
    ├── PushAction
    └── CustomAction

# Configuration classes
ActionConfig(ABC)
    ├── ReachConfig
    ├── GraspConfig
    └── ...

# Uses existing PlanResult from embodichain.lab.sim.planners
PlanResult (existing)
    ├── success: bool
    ├── positions: torch.Tensor  # [T, DOF]
    ├── xpos_list: torch.Tensor  # [T, 4, 4]
    ├── final_pose: torch.Tensor
    └── execution_time: float
```

---

## 4. Core API Design

### 4.1 Base AtomicAction Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import torch
import numpy as np

from embodichain.lab.sim.planners import PlanResult, PlanState, MoveType
from embodichain.utils import configclass


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

    confidence_scores: Optional[torch.Tensor] = None
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

    normals: Optional[torch.Tensor] = None
    """Optional surface normals at each interaction point with shape [B, 3].

    Normals indicate the surface orientation at each point,
    useful for determining approach directions.
    """

    point_types: List[str] = field(default_factory=lambda: ["contact"])
    """List of interaction types for each point.

    Examples: "push", "poke", "touch", "support", "handle", etc.
    Length must match the batch dimension of `points`.
    """

    contact_regions: Optional[List[str]] = None
    """Optional labels for object regions each point belongs to.

    Examples: "handle", "face", "edge", "corner", "center"
    """

    def get_batch_size(self) -> int:
        """Return the number of interaction points in this affordance."""
        return self.points.shape[0]

    def get_points_by_type(self, point_type: str) -> Optional[torch.Tensor]:
        """Get all points of a specific interaction type.

        Args:
            point_type: Type of interaction (e.g., "push", "handle")

        Returns:
            Tensor of points with shape [N, 3] if found, None otherwise
        """
        indices = [i for i, t in enumerate(self.point_types) if t == point_type]
        if indices:
            return self.points[indices]
        return None

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
        return torch.tensor([0, 0, 1], dtype=self.points.dtype, device=self.points.device)


# =============================================================================
# ObjectSemantics
# =============================================================================

@dataclass
class ObjectSemantics:
    """Semantic information about interaction target.

    This class encapsulates all semantic and geometric information about
    an object needed for intelligent interaction planning.
    """
    label: str
    """Object category label (e.g., 'apple', 'bottle')."""

    affordance: Affordance
    """Affordance data (GraspPose, InteractionPoints, etc.)."""

    geometry: Dict[str, Any]
    """Geometric information including bounding box, mesh data."""

    properties: Dict[str, Any]
    """Physical properties: mass, friction, etc."""

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
    """Interpolation type: 'linear', 'cubic', or 'toppra'."""

    velocity_limit: Optional[float] = None
    """Maximum velocity for trajectory."""

    acceleration_limit: Optional[float] = None
    """Maximum acceleration for trajectory."""


class AtomicAction(ABC):
    """Abstract base class for atomic actions.

    All atomic actions use PlanResult from embodichain.lab.sim.planners
    as the return type for execute() method, ensuring consistency with
    the existing motion planning infrastructure.
    """

    def __init__(
        self,
        motion_generator: "MotionGenerator",
        robot: "Robot",
        control_part: str,
        device: torch.device = torch.device("cuda"),
    ):
        self.motion_generator = motion_generator
        self.robot = robot
        self.control_part = control_part
        self.device = device

    @abstractmethod
    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        **kwargs
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
        **kwargs
    ) -> bool:
        """Validate if the action is feasible without executing.

        This method performs a quick feasibility check (e.g., IK solvability)
        without generating a full trajectory.

        Returns:
            True if action appears feasible, False otherwise
        """
        pass

    def _get_current_qpos(self) -> torch.Tensor:
        """Get current joint configuration from robot."""
        return self.robot.get_qpos()[0]  # Assuming single environment

    def _ik_solve(
        self,
        target_pose: torch.Tensor,
        qpos_seed: Optional[torch.Tensor] = None
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
            qpos_seed = self._get_current_qpos()

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
        target_states: List["PlanState"],
        options: Optional["MotionGenOptions"] = None,
    ) -> "PlanResult":
        """Plan trajectory using motion generator."""
        if options is None:
            options = MotionGenOptions(control_part=self.control_part)
        return self.motion_generator.generate(target_states, options)
```

### 4.2 ReachAction Implementation

```python
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
        super().__init__(motion_generator, robot, control_part, device)
        self.interpolation_type = interpolation_type

    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        approach_offset: Optional[torch.Tensor] = None,
        use_affordance: bool = True,
        **kwargs
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
            PlanState(
                qpos=start_qpos,
                move_type=MoveType.JOINT_MOVE
            ),
            PlanState(
                xpos=approach_pose,
                move_type=MoveType.EEF_MOVE
            ),
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
        **kwargs
    ) -> bool:
        """Check if the reach action is feasible."""
        try:
            # Quick IK feasibility check
            if isinstance(target, ObjectSemantics):
                target_pose = self._resolve_target_pose(target, use_affordance=True)
            else:
                target_pose = target

            # Attempt IK
            qpos_seed = start_qpos if start_qpos is not None else self._get_current_qpos()
            success, _ = self.robot.compute_ik(
                pose=target_pose.unsqueeze(0),
                qpos_seed=qpos_seed.unsqueeze(0),
                name=self.control_part,
            )
            return success.all().item()
        except Exception:
            return False

    def _resolve_target_pose(
        self,
        semantics: ObjectSemantics,
        use_affordance: bool
    ) -> torch.Tensor:
        """Resolve target pose from object semantics."""
        if use_affordance and "grasp_pose" in semantics.affordance_data:
            # Use precomputed grasp pose from affordance data
            grasp_pose = semantics.affordance_data["grasp_pose"]
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
        pass

    def _get_current_qpos(self) -> torch.Tensor:
        """Get current joint configuration."""
        return self.robot.get_qpos()[0]  # Assuming single environment

    def _apply_offset(self, pose: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Apply offset to pose in local frame."""
        result = pose.clone()
        result[:3, 3] += pose[:3, :3] @ offset
        return result
```

### 4.3 GraspAction Implementation

```python
class GraspAction(AtomicAction):
    """Atomic action for grasping objects."""

    def __init__(
        self,
        motion_generator: "MotionGenerator",
        robot: "Robot",
        control_part: str,
        device: torch.device = torch.device("cuda"),
        pre_grasp_distance: float = 0.05,
        approach_direction: str = "z",  # "x", "y", "z", or "custom"
    ):
        super().__init__(motion_generator, robot, control_part, device)
        self.pre_grasp_distance = pre_grasp_distance
        self.approach_direction = approach_direction

    def execute(
        self,
        target: ObjectSemantics,
        start_qpos: Optional[torch.Tensor] = None,
        use_affordance: bool = True,
        grasp_type: str = "default",  # "default", "pinch", "power"
        **kwargs
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
        **kwargs
    ) -> bool:
        """Validate if grasp is feasible."""
        try:
            grasp_pose = self._resolve_grasp_pose(target, use_affordance=True, grasp_type="default")
            qpos_seed = start_qpos if start_qpos is not None else self._get_current_qpos()
            success, _ = self.robot.compute_ik(
                pose=grasp_pose.unsqueeze(0),
                qpos_seed=qpos_seed.unsqueeze(0),
                name=self.control_part,
            )
            return success.all().item()
        except Exception:
            return False

    def _resolve_grasp_pose(
        self,
        semantics: ObjectSemantics,
        use_affordance: bool,
        grasp_type: str
    ) -> torch.Tensor:
        """Resolve grasp pose from object semantics."""
        if use_affordance and "grasp_poses" in semantics.affordance_data:
            grasp_poses = semantics.affordance_data["grasp_poses"]
            if grasp_type in grasp_poses:
                grasp_offset = grasp_poses[grasp_type]
            else:
                grasp_offset = grasp_poses["default"]

            object_pose = self._get_object_pose(semantics.label)
            return object_pose @ grasp_offset

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
```

### 4.4 MoveAction Implementation

```python
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
        super().__init__(motion_generator, robot, control_part, device)
        self.move_type = move_type
        self.interpolation = interpolation

    def execute(
        self,
        target: Union[torch.Tensor, ObjectSemantics],
        start_qpos: Optional[torch.Tensor] = None,
        offset: Optional[torch.Tensor] = None,
        velocity_limit: Optional[float] = None,
        acceleration_limit: Optional[float] = None,
        **kwargs
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
```

### 4.5 RigidObject Extension

```python
class RigidObject(BatchEntity):
    """RigidObject with ObjectSemantics support."""

    def get_object_semantics(self, env_id: int = 0) -> ObjectSemantics:
        """Return ObjectSemantics for this rigid object.

        This method aggregates object label, affordance data, geometry,
        and physical properties into a unified semantic representation
        for intelligent interaction planning.

        Args:
            env_id: Environment index for batched objects (default: 0)

        Returns:
            ObjectSemantics containing all semantic information

        Example:
            >>> obj = sim.get_rigid_object("apple_01")
            >>> semantics = obj.get_object_semantics()
            >>> print(semantics.label)  # "apple"
            >>> print(semantics.affordance.poses.shape)  # [B, 4, 4]
        """
        # Get object label from configuration or user data
        label = self._get_object_label(env_id)

        # Get or compute affordance data
        affordance = self._get_affordance_data(env_id)

        # Get geometry information
        geometry = self._get_geometry_data(env_id)

        # Get physical properties
        properties = self._get_physical_properties(env_id)

        return ObjectSemantics(
            label=label,
            affordance=affordance,
            geometry=geometry,
            properties=properties,
            uid=self._entities[env_id].get_user_id() if env_id < len(self._entities) else None,
        )

    def _get_object_label(self, env_id: int) -> str:
        """Extract object label from configuration or metadata."""
        # Try to get from object user data
        entity = self._entities[env_id] if env_id < len(self._entities) else None
        if entity:
            # Check for label in user data
            label = entity.get_user_data().get("label", None) if hasattr(entity, "get_user_data") else None
            if label:
                return label

        # Fallback: derive from object name or class
        class_name = self.__class__.__name__.lower()
        if "cube" in class_name:
            return "cube"
        elif "sphere" in class_name:
            return "sphere"
        elif "cylinder" in class_name:
            return "cylinder"

        return "unknown"

    def _get_affordance_data(self, env_id: int) -> Affordance:
        """Get or compute affordance data for the object."""
        # Check if precomputed affordance data exists
        entity = self._entities[env_id] if env_id < len(self._entities) else None
        if entity and hasattr(entity, "get_user_data"):
            user_data = entity.get_user_data()
            if "affordance" in user_data:
                return user_data["affordance"]

        # Compute default affordance based on geometry
        geometry = self._get_geometry_data(env_id)
        bbox = geometry.get("bounding_box", [0.1, 0.1, 0.1])

        # Create default grasp poses based on bounding box
        # Generate poses for top, front, and side grasps
        poses = []
        # Top grasp
        pose_top = torch.eye(4)
        pose_top[2, 3] = bbox[2] / 2 + 0.05  # Offset above object
        poses.append(pose_top)

        # Front grasp
        pose_front = torch.eye(4)
        pose_front[0, 3] = bbox[0] / 2 + 0.05
        poses.append(pose_front)

        poses_tensor = torch.stack(poses)

        return GraspPose(
            poses=poses_tensor,
            grasp_types=["top", "front"],
            object_label=self._get_object_label(env_id),
        )

    def _get_geometry_data(self, env_id: int) -> Dict[str, Any]:
        """Get geometric information about the object."""
        # Get bounding box from entity
        entity = self._entities[env_id] if env_id < len(self._entities) else None
        bbox = [0.1, 0.1, 0.1]  # Default

        if entity:
            try:
                # Try to get bounding box from entity
                if hasattr(entity, "get_bounding_box"):
                    bbox = entity.get_bounding_box()
            except Exception:
                pass

        # Get mesh information if available
        mesh_info = {}
        if entity and hasattr(entity, "get_mesh_count"):
            mesh_info["mesh_count"] = entity.get_mesh_count()

        return {
            "bounding_box": bbox,
            "volume": bbox[0] * bbox[1] * bbox[2],
            "mesh_info": mesh_info,
        }

    def _get_physical_properties(self, env_id: int) -> Dict[str, Any]:
        """Get physical properties of the object."""
        entity = self._entities[env_id] if env_id < len(self._entities) else None

        properties = {
            "mass": 1.0,
            "friction": 0.5,
            "restitution": 0.1,
            "body_type": self.body_type,
        }

        if entity and hasattr(entity, "get_physical_body"):
            try:
                phys_body = entity.get_physical_body()
                properties["mass"] = phys_body.get_mass()
                properties["friction"] = phys_body.get_dynamic_friction()
            except Exception:
                pass

        return properties


# =============================================================================
# Action Engine
# =============================================================================

class AtomicActionEngine:
    """Central engine for managing and executing atomic actions."""

    def __init__(
        self,
        robot: "Robot",
        motion_generator: "MotionGenerator",
        device: torch.device = torch.device("cuda"),
    ):
        self.robot = robot
        self.motion_generator = motion_generator
        self.device = device

        # Registry of action instances
        self._actions: Dict[str, AtomicAction] = {}

        # Semantic analyzer for object understanding
        self._semantic_analyzer = SemanticAnalyzer()

        # Initialize default actions
        self._init_default_actions()

    def _init_default_actions(self):
        """Initialize default atomic action instances."""
        control_parts = self.robot.control_parts or ["default"]

        for part in control_parts:
            self.register_action(
                f"reach_{part}",
                ReachAction(
                    motion_generator=self.motion_generator,
                    robot=self.robot,
                    control_part=part,
                    device=self.device,
                )
            )
            self.register_action(
                f"grasp_{part}",
                GraspAction(
                    motion_generator=self.motion_generator,
                    robot=self.robot,
                    control_part=part,
                    device=self.device,
                )
            )
            self.register_action(
                f"move_{part}",
                MoveAction(
                    motion_generator=self.motion_generator,
                    robot=self.robot,
                    control_part=part,
                    device=self.device,
                )
            )

    def register_action(self, name: str, action: AtomicAction):
        """Register a custom atomic action."""
        self._actions[name] = action

    def execute(
        self,
        action_name: str,
        target: Union[torch.Tensor, str, ObjectSemantics],
        control_part: Optional[str] = None,
        **kwargs
    ) -> PlanResult:
        """Execute an atomic action.

        Args:
            action_name: Name of registered action
            target: Target pose, object label, or ObjectSemantics
            control_part: Robot control part to use
            **kwargs: Additional action parameters

        Returns:
            PlanResult with trajectory (positions, velocities, accelerations),
            end-effector poses (xpos_list), and success status.
        """
        # Resolve action
        if control_part:
            action_key = f"{action_name}_{control_part}"
        else:
            action_key = action_name

        if action_key not in self._actions:
            raise ValueError(f"Unknown action: {action_key}")

        action = self._actions[action_key]

        # Resolve target to ObjectSemantics if string label provided
        if isinstance(target, str):
            target = self._semantic_analyzer.analyze(target)

        # Execute action - returns PlanResult directly
        return action.execute(target, **kwargs)

    def validate(
        self,
        action_name: str,
        target: Union[torch.Tensor, str, ObjectSemantics],
        control_part: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Validate if an action is feasible without executing."""
        if control_part:
            action_key = f"{action_name}_{control_part}"
        else:
            action_key = action_name

        if action_key not in self._actions:
            return False

        action = self._actions[action_key]

        if isinstance(target, str):
            target = self._semantic_analyzer.analyze(target)

        return action.validate(target, **kwargs)
```

---

## 5. Usage Examples

### 5.1 Basic Usage

```python
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.gym.envs import BaseEnv

# Initialize components
env: BaseEnv = ...  # Your environment
robot: Robot = env.robot
motion_gen = MotionGenerator(
    cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
)

# Create action engine
engine = AtomicActionEngine(
    robot=robot,
    motion_generator=motion_gen,
    device=env.device,
)

# Execute reach action
result = engine.execute(
    action_name="reach",
    target=target_pose,  # [4, 4] tensor
    control_part="left_arm",
    approach_offset=torch.tensor([0, 0, -0.05]),
)

if result.success:
    trajectory = result.trajectory  # [T, DOF]
    env.execute_trajectory(trajectory)
```

### 5.2 Using Object Semantics

```python
# Define object semantics
apple_semantics = ObjectSemantics(
    label="apple",
    affordance_data={
        "grasp_poses": {
            "default": grasp_pose_1,  # [4, 4]
            "pinch": grasp_pose_2,
        },
        "interaction_points": [point_1, point_2],
    },
    geometry={
        "bounding_box": [0.08, 0.08, 0.08],
        "mesh": "apple_mesh.obj",
    },
    properties={
        "mass": 0.15,
        "friction": 0.5,
    },
)

# Execute grasp with semantics
result = engine.execute(
    action_name="grasp",
    target=apple_semantics,
    control_part="right_arm",
    grasp_type="pinch",
)
```

### 5.3 Custom Atomic Action

```python
class RotateEEFAction(AtomicAction):
    """Custom action to rotate end-effector."""

    def execute(
        self,
        target: torch.Tensor,  # Rotation angle in degrees
        start_qpos: Optional[torch.Tensor] = None,
        axis: str = "z",
        **kwargs
    ) -> PlanResult:
        # Get current pose
        current_qpos = start_qpos or self._get_current_qpos()
        current_pose = self.robot.compute_fk(
            qpos=current_qpos.unsqueeze(0),
            name=self.control_part,
            to_matrix=True,
        ).squeeze(0)

        # Apply rotation
        angle_rad = torch.deg2rad(target)
        rotation = self._create_rotation_matrix(angle_rad, axis)
        target_pose = current_pose.clone()
        target_pose[:3, :3] = rotation @ current_pose[:3, :3]

        # Plan motion
        target_states = [
            PlanState(qpos=current_qpos, move_type=MoveType.JOINT_MOVE),
            PlanState(xpos=target_pose, move_type=MoveType.EEF_MOVE),
        ]

        result = self.plan_trajectory(target_states)

        # Return PlanResult directly - it contains all trajectory data
        return result

# Register custom action
engine.register_action(
    "rotate_eef",
    RotateEEFAction(
        motion_generator=motion_gen,
        robot=robot,
        control_part="left_arm",
        device=device,
    )
)
```

---

## 6. Integration with Agentic Workflows

### 6.1 LLM Action Selection

```python
class ActionSelector:
    """Select actions based on LLM reasoning."""

    def __init__(self, action_engine: AtomicActionEngine):
        self.engine = action_engine

    def select_action(
        self,
        task_description: str,
        scene_observation: Dict,
        available_objects: List[ObjectSemantics],
    ) -> Tuple[str, Dict]:
        """Select appropriate action based on task and scene."""
        # Parse task to determine required action
        if "pick" in task_description.lower():
            action_name = "grasp"
            target = self._resolve_object(task_description, available_objects)
            params = {"grasp_type": "default"}
        elif "reach" in task_description.lower():
            action_name = "reach"
            target = self._resolve_target_pose(task_description, scene_observation)
            params = {}
        elif "place" in task_description.lower():
            action_name = "move"
            target = self._resolve_placement(task_description, scene_observation)
            params = {}
        else:
            raise ValueError(f"Cannot determine action for task: {task_description}")

        return action_name, {"target": target, **params}

    def _resolve_object(
        self,
        task_description: str,
        available_objects: List[ObjectSemantics],
    ) -> ObjectSemantics:
        """Resolve object reference from task description."""
        # Use LLM or keyword matching to identify object
        pass
```

### 6.2 Action Composition

```python
class ActionComposer:
    """Compose multiple atomic actions into complex behaviors."""

    def __init__(self, engine: AtomicActionEngine):
        self.engine = engine

    def pick_and_place(
        self,
        pick_object: ObjectSemantics,
        place_pose: torch.Tensor,
        control_part: str = "right_arm",
    ) -> List[PlanResult]:
        """Compose pick and place behavior."""
        results = []

        # 1. Reach pre-grasp
        result = self.engine.execute(
            "reach",
            target=pick_object,
            control_part=control_part,
            approach_offset=torch.tensor([0, 0, -0.08]),
        )
        results.append(result)
        if not result.success:
            return results

        # 2. Grasp
        result = self.engine.execute(
            "grasp",
            target=pick_object,
            control_part=control_part,
        )
        results.append(result)
        if not result.success:
            return results

        # 3. Lift
        current_qpos = result.trajectory[-1] if result.trajectory is not None else None
        lift_pose = self._get_lift_pose(pick_object)
        result = self.engine.execute(
            "move",
            target=lift_pose,
            control_part=control_part,
            start_qpos=current_qpos,
        )
        results.append(result)
        if not result.success:
            return results

        # 4. Move to place
        result = self.engine.execute(
            "move",
            target=place_pose,
            control_part=control_part,
        )
        results.append(result)
        if not result.success:
            return results

        # 5. Release
        result = self.engine.execute(
            "release",
            target=place_pose,
            control_part=control_part,
        )
        results.append(result)

        return results
```

---

## 7. Extensibility

### 7.1 Custom Action Registration

```python
# Define custom action
class PushAction(AtomicAction):
    def execute(self, target, **kwargs):
        # Implementation
        pass

# Register at runtime
from embodichain.lab.sim.atomic_actions import register_action

register_action(
    name="push",
    action_class=PushAction,
    motion_generator=motion_gen,
    robot=robot,
    control_part="right_arm",
)

# Use in engine
engine.execute("push", target=object_semantics)
```

### 7.2 Plugin Architecture

```python
class ActionPlugin(ABC):
    """Base class for action plugins."""

    @abstractmethod
    def get_actions(self) -> Dict[str, Type[AtomicAction]]:
        """Return mapping of action names to classes."""
        pass

    @abstractmethod
    def get_configurators(self) -> Dict[str, Any]:
        """Return configuration helpers."""
        pass

# Example plugin
class ManipulationActionsPlugin(ActionPlugin):
    def get_actions(self):
        return {
            "twist": TwistAction,
            "slide": SlideAction,
            "insert": InsertAction,
        }
```

---

## 8. Summary

This atomic action abstraction design provides:

1. **Unified Interface**: All atomic actions inherit from `AtomicAction` with consistent `execute()` and `validate()` methods

2. **Semantic Awareness**: Object semantics (label, affordance, geometry) are first-class citizens

3. **Motion Planning Integration**: Leverages existing `MotionGenerator`, solvers, and warp interpolation

4. **Agentic Workflow Support**: Easy composition into complex behaviors and LLM integration

5. **Extensibility**: Registry-based action registration and plugin architecture

The design bridges low-level motion planning with high-level agent reasoning, enabling both precise control and semantic task execution.
