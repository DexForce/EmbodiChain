# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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
"""
Gizmo: A reusable controller for interactive manipulation of simulation elements (object, robot, camera, etc.)
"""


import numpy as np
import torch
import dexsim
from typing import Callable
from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.sensors import Camera
from embodichain.utils import configclass, logger

from dexsim.types import (
    AxisOption,
    RotationRingsOption,
    AxisArrowType,
    AxisCornerType,
    AxisTagType,
    TransformMask,
    ActorType,
    RigidBodyShape,
    PhysicalAttr,
)

from embodichain.lab.sim.utility.gizmo_utils import create_gizmo_callback


@configclass
class GizmoCfg:
    """Configuration class for Gizmo parameters.

    This class defines the visual and interaction parameters for gizmo controllers,
    including axis appearance and rotation rings settings.
    """

    # Axis configuration
    axis_length_x: float = 0.2
    """Length of X-axis arrow."""
    axis_length_y: float = 0.2
    """Length of Y-axis arrow."""
    axis_length_z: float = 0.2
    """Length of Z-axis arrow."""
    axis_size: float = 0.01
    """Thickness of axis lines."""
    arrow_type: AxisArrowType = AxisArrowType.CONE
    """Type of arrow head."""
    corner_type: AxisCornerType = AxisCornerType.SPHERE
    """Type of axis corner."""
    tag_type: AxisTagType = AxisTagType.PLANE
    """Type of axis label."""

    # Rotation rings configuration
    rings_radius: float = 0.15
    """Radius of rotation rings."""
    rings_size: float = 0.01
    """Thickness of rotation rings."""

    def to_options_dict(self) -> dict:
        """Convert configuration to options dictionary format expected by gizmo creation.

        Returns:
            Dictionary containing AxisOption and RotationRingsOption objects.
        """
        return {
            "axis": AxisOption(
                lx=self.axis_length_x,
                ly=self.axis_length_y,
                lz=self.axis_length_z,
                size=self.axis_size,
                arrow_type=self.arrow_type,
                corner_type=self.corner_type,
                tag_type=self.tag_type,
            ),
            "rings": RotationRingsOption(
                radius=self.rings_radius, size=self.rings_size
            ),
        }


class Gizmo:
    """
    Generic Gizmo controller for simulation elements.
    Supports RigidObject, Robot, and Camera with type-specific handling.

    Note:
        Gizmo can only be used in single environment mode (num_envs=1).
        Will raise RuntimeError if used with multiple environments.
    """

    def __init__(
        self,
        target: BatchEntity,
        cfg: GizmoCfg | None = None,
        control_part: str | None = "arm",
    ):
        """
        Args:
            target: The simulation element to control (RigidObject, Robot, or Camera)
            cfg: Gizmo configuration parameters (optional, uses default if None)
            control_part: For robots, specifies which control part to use (optional, default: "arm")
        """
        self.target = target
        self._target_type = self._detect_target_type(target)
        self._control_part = control_part
        self._env = dexsim.default_world().get_env()
        self._windows = dexsim.default_world().get_windows()

        # Check if running in single environment (num_env must be 1)
        num_envs = dexsim.get_world_num()
        if num_envs > 1:
            raise RuntimeError(
                f"Gizmo can only be used in single environment mode (num_env=1), "
                f"but current num_envs={num_envs}. Please create simulation with num_envs=1."
            )

        # Use provided config or get default
        if cfg is None:
            cfg = self._get_default_cfg()
        self.cfg = cfg
        self._gizmo = self._create_gizmo(self.cfg)
        self._callback = None
        self._state = "active"
        self._setup_gizmo_follow()

    def _detect_target_type(self, target: BatchEntity) -> str:
        """Detect target type: 'rigidobject', 'robot', or 'camera' using isinstance only."""
        if Robot is not None and isinstance(target, Robot):
            return "robot"
        if Camera is not None and isinstance(target, Camera):
            return "camera"
        if RigidObject is not None and isinstance(target, RigidObject):
            return "rigidobject"

        raise ValueError(
            f"Unsupported target type: {type(target)}. Only RigidObject, Robot, and Camera are supported."
        )

    def _get_default_cfg(self) -> GizmoCfg:
        """Get default gizmo configuration (same for all target types)"""
        return GizmoCfg()

    def _create_gizmo(self, cfg: GizmoCfg):
        """Create gizmo using configuration object"""
        options = cfg.to_options_dict()
        axis = options["axis"]
        rings = options["rings"]
        return self._env.create_gizmo(axis, rings)

    def _compute_ee_pose_fk(self):
        """Compute end-effector pose using forward kinematics"""
        # Get current joint positions for this arm
        proprioception = self.target.get_proprioception()
        current_qpos_full = proprioception["qpos"]
        current_joint_ids = self.target.get_joint_ids(self._robot_arm_name)

        joint_positions = current_qpos_full[:, current_joint_ids]
        if joint_positions.dim() > 1:
            joint_positions = joint_positions[0]

        # Compute forward kinematics
        ee_pose = self.target.compute_fk(
            joint_positions, name=self._control_part, to_matrix=True
        )

        return ee_pose

    def _create_proxy_cube(
        self, position: np.ndarray, rotation_matrix: np.ndarray, name: str
    ):
        """Create a proxy cube for gizmo tracking"""
        # Convert rotation matrix to euler angles
        euler = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)

        # Create small proxy cube at specified position
        proxy_cube = self._env.create_cube(0.02, 0.02, 0.02)  # 2cm cube
        proxy_cube.set_location(position[0], position[1], position[2])
        proxy_cube.set_rotation_euler(euler[0], euler[1], euler[2])

        # Connect gizmo to proxy cube
        self._gizmo.node.update_gizmo_follow(proxy_cube.node)

        logger.log_info(f"{name} gizmo proxy created at position: {position}")
        return proxy_cube

    def _setup_camera_gizmo(self):
        """Setup gizmo for Camera by creating a proxy RigidObject at camera position"""
        # Get current camera pose
        camera_pose = self.target.get_local_pose(to_matrix=True)[0]  # Get first camera
        camera_pos = camera_pose[:3, 3].cpu().numpy()
        camera_rot_matrix = camera_pose[:3, :3].cpu().numpy()

        # Create proxy cube and set callback
        self._proxy_cube = self._create_proxy_cube(
            camera_pos, camera_rot_matrix, "Camera"
        )
        self._gizmo.node.set_flush_transform_callback(self._proxy_gizmo_callback)

    def _proxy_gizmo_callback(self, node, translation, rotation, flag):
        """Generic callback for proxy-based gizmo: only updates proxy cube transform, defers actual updates"""
        if node is None:
            return

        # Check if proxy cube still exists (not destroyed)
        if not hasattr(self, "_proxy_cube") or self._proxy_cube is None:
            return

        # Update proxy cube transform
        if flag == (TransformMask.TRANSFORM_LOCAL | TransformMask.TRANSFORM_T):
            node.set_translation(translation)
        elif flag == (TransformMask.TRANSFORM_LOCAL | TransformMask.TRANSFORM_R):
            node.set_rotation_rpy(rotation)

        # Mark that target needs to be updated, save target transform
        proxy_pos = self._proxy_cube.get_location()
        proxy_rot = self._proxy_cube.get_rotation_euler()
        target_transform = torch.eye(4, dtype=torch.float32)
        target_transform[:3, 3] = torch.tensor(
            [proxy_pos[0], proxy_pos[1], proxy_pos[2]], dtype=torch.float32
        )
        target_transform[:3, :3] = torch.tensor(
            R.from_euler("xyz", proxy_rot).as_matrix(), dtype=torch.float32
        )
        # Ensure _pending_target_transform is (1, 4, 4)
        if isinstance(target_transform, torch.Tensor) and target_transform.shape == (
            4,
            4,
        ):
            target_transform = target_transform.unsqueeze(0)
        self._pending_target_transform = target_transform

    def _update_camera_pose(self, target_transform: torch.Tensor):
        """Update camera pose to match target transform"""
        try:
            # Set camera pose using set_local_pose method
            self.target.set_local_pose(target_transform)
            return True
        except Exception as e:
            logger.log_error(f"Error updating camera pose: {e}")
            return False

    def _setup_robot_gizmo(self):
        """Setup gizmo for Robot by creating a proxy RigidObject at end-effector"""
        # Get end-effector pose using specified control part
        if self.target.cfg.solver_cfg is None:
            raise ValueError(
                "Robot has no solver configured for IK/FK computations for gizmo"
            )

        arm_names = list(self.target.control_parts.keys())
        if not arm_names:
            raise ValueError("Robot has no control parts defined")

        # Use specified control part or fall back to first available
        if self._control_part and self._control_part in arm_names:
            self._robot_arm_name = self._control_part
        else:
            logger.log_error(f"Control part '{self._control_part}' not found.")

        logger.log_info(f"Using control part: {self._robot_arm_name}")

        # Get end-effector pose using forward kinematics
        ee_pose = self._compute_ee_pose_fk()[0]  # remove batch dimension

        ee_pos = ee_pose[:3, 3].cpu().numpy()
        ee_rot_matrix = ee_pose[:3, :3].cpu().numpy()

        # Create proxy cube and set callback
        self._proxy_cube = self._create_proxy_cube(ee_pos, ee_rot_matrix, "Robot")
        self._gizmo.node.set_flush_transform_callback(self._proxy_gizmo_callback)

    def _update_robot_ik(self, target_transform: torch.Tensor):
        """Update robot joints using IK to reach target transform"""
        try:
            # Get robot solver for the arm
            solver = self.target.get_solver(self._robot_arm_name)
            if solver is None:
                logger.log_warning(f"No solver found for arm: {self._robot_arm_name}")
                return False

            # Get current joint positions as seed using proprioception
            proprioception = self.target.get_proprioception()
            current_qpos_full = proprioception["qpos"]  # Full joint positions

            # Get joint IDs for this arm
            current_joint_ids = self.target.get_joint_ids(self._robot_arm_name)

            # Extract joint positions for this specific arm
            if len(current_joint_ids) > 0:
                joint_seed = current_qpos_full[
                    :, current_joint_ids
                ]  # Select arm joints
                if joint_seed.dim() > 1:
                    joint_seed = joint_seed[0]  # Take first batch element
            else:
                logger.log_warning(
                    f"No joint IDs found for arm: {self._robot_arm_name}"
                )
                return False

            # Solve IK
            ik_success, new_qpos = solver.get_ik(
                target_xpos=target_transform, joint_seed=joint_seed
            )

            if ik_success:
                # Ensure correct dimensions for setting qpos
                # new_qpos from IK solver may be (1, N, dof) or (N, dof), flatten to (dof,) for single env
                if new_qpos.dim() > 1:
                    new_qpos = new_qpos.squeeze()  # Remove all singleton dimensions
                if new_qpos.dim() == 1:
                    new_qpos = new_qpos.unsqueeze(0)  # Make it (1, dof) for set_qpos

                # Update robot joint positions
                self.target.set_qpos(qpos=new_qpos[0], joint_ids=current_joint_ids)
                return True
            else:
                logger.log_warning("IK solution not found")
                return False

        except Exception as e:
            logger.log_error(f"Error in robot IK: {e}")
            return False

    def _setup_gizmo_follow(self):
        """Setup gizmo based on target type"""
        if self._target_type == "rigidobject":
            # RigidObject: direct node access through MeshObject
            self._gizmo.node.update_gizmo_follow(self.target._entities[0].node)
            self._gizmo.node.set_flush_transform_callback(create_gizmo_callback())
        elif self._target_type == "robot":
            # Robot: create proxy object at end-effector position
            self._setup_robot_gizmo()
        elif self._target_type == "camera":
            # Camera: create proxy object at camera position
            self._setup_camera_gizmo()

    def attach(self, target: BatchEntity):
        """Attach gizmo to a new simulation element."""
        self.target = target
        self._target_type = self._detect_target_type(target)
        self._setup_gizmo_follow()

    def detach(self):
        """Detach gizmo from current element."""
        self.target = None
        # Use detach_parent to properly disconnect gizmo
        try:
            self._gizmo.node.detach_parent()
        except Exception as e:
            logger.log_warning(f"Failed to detach gizmo parent: {e}")

    def set_transform_callback(self, callback: Callable):
        """Set callback for gizmo transform events (translation/rotation)."""
        self._callback = callback
        self._gizmo.node.set_flush_transform_callback(callback)

    def set_world_pose(self, pose):
        """Set gizmo's world pose."""
        self._gizmo.node.set_world_pose(pose)

    def set_local_pose(self, pose):
        """Set gizmo's local pose."""
        self._gizmo.node.set_local_pose(pose)

    def set_line_width(self, width: float):
        """Set gizmo line width."""
        self._gizmo.node.set_line_width(width)

    def enable_collision(self, enabled: bool):
        """Enable or disable gizmo collision."""
        self._gizmo.node.enable_collision(enabled)

    def get_world_pose(self):
        """Get gizmo's world pose."""
        return self._gizmo.node.get_world_pose()

    def get_local_pose(self):
        """Get gizmo's local pose."""
        return self._gizmo.node.get_local_pose()

    def get_name(self):
        """Get gizmo node name."""
        return self._gizmo.node.get_name()

    def get_parent(self):
        """Get gizmo's parent node."""
        return self._gizmo.node.get_parent()

    def toggle_visibility(self) -> bool:
        """
        Toggle the visibility of the gizmo.

        Returns:
            bool: The new visibility state (True = visible, False = hidden)
        """
        if not hasattr(self, "_is_visible"):
            self._is_visible = True  # Default to visible

        # Toggle the state
        self._is_visible = not self._is_visible

        # Apply the visibility setting to the gizmo node
        if self._gizmo and hasattr(self._gizmo, "node"):
            self._gizmo.node.set_visible(self._is_visible)

        return self._is_visible

    def set_visible(self, visible: bool):
        """
        Set the visibility of the gizmo.

        Args:
            visible (bool): True to show, False to hide the gizmo
        """
        self._is_visible = visible

        # Apply the visibility setting to the gizmo node
        if self._gizmo and hasattr(self._gizmo, "node"):
            self._gizmo.node.set_visible(self._is_visible)

    def is_visible(self) -> bool:
        """
        Check if the gizmo is currently visible.

        Returns:
            bool: True if visible, False if hidden
        """
        return getattr(self, "_is_visible", True)

    def update(self):
        """Synchronize gizmo with target's current transform, and handle IK solving here."""
        if self._target_type == "rigidobject":
            self._gizmo.node.update_gizmo_follow(self.target._entities[0].node)
        elif self._target_type == "robot":
            # If there is a pending target, solve IK and clear it
            if (
                hasattr(self, "_pending_target_transform")
                and self._pending_target_transform is not None
            ):
                self._update_robot_ik(self._pending_target_transform)
                self._pending_target_transform = None
        elif self._target_type == "camera":
            # Update proxy cube position to match current camera pose
            if hasattr(self, "_proxy_cube") and self._proxy_cube:
                camera_pose = self.target.get_local_pose(to_matrix=True)[0]
                camera_pos = camera_pose[:3, 3].cpu().numpy()
                self._proxy_cube.set_location(
                    camera_pos[0], camera_pos[1], camera_pos[2]
                )

            # If there is a pending camera target, update camera pose and clear it
            if (
                hasattr(self, "_pending_target_transform")
                and self._pending_target_transform is not None
            ):
                self._update_camera_pose(self._pending_target_transform)
                self._pending_target_transform = None

    def apply_transform(self, translation, rotation):
        """Apply transform based on target type"""
        if self._target_type == "rigidobject":
            self.target.set_location(*translation)
            self.target.set_rotation_euler(*rotation)
        elif self._target_type == "robot":
            # Robot transforms are handled by IK in the gizmo callback
            if hasattr(self, "_proxy_cube") and self._proxy_cube:
                self._proxy_cube.set_location(*translation)
                self._proxy_cube.set_rotation_euler(*rotation)
        elif self._target_type == "camera":
            # Camera transforms are handled by pose update in the gizmo callback
            if hasattr(self, "_proxy_cube") and self._proxy_cube:
                self._proxy_cube.set_location(*translation)
                self._proxy_cube.set_rotation_euler(*rotation)
        else:
            # Other target types
            pass

    def destroy(self):
        """Clean up gizmo resources and release references."""
        # Clear transform callback first to avoid bad_function_call
        if hasattr(self, "_gizmo") and self._gizmo and hasattr(self._gizmo, "node"):
            try:
                # Clear transform callback before any other cleanup
                self._gizmo.node.set_flush_transform_callback(None)
                logger.log_info("Cleared gizmo transform callback")
            except Exception as e:
                logger.log_warning(f"Failed to clear gizmo callback: {e}")

        # Remove proxy cube if exists (before detaching gizmo)
        if hasattr(self, "_proxy_cube") and self._proxy_cube:
            try:
                # Detach gizmo from proxy cube first
                if (
                    hasattr(self, "_gizmo")
                    and self._gizmo
                    and hasattr(self._gizmo, "node")
                ):
                    self._gizmo.node.detach_parent()
                # Then remove the proxy cube
                self._env.remove_actor(self._proxy_cube)
                logger.log_info("Successfully removed proxy cube from environment")
            except Exception as e:
                logger.log_warning(f"Failed to remove proxy cube: {e}")
            self._proxy_cube = None

        # Final gizmo cleanup
        if hasattr(self, "_gizmo") and self._gizmo and hasattr(self._gizmo, "node"):
            try:
                # Ensure detach_parent is called if not done above
                if self._target_type in ["robot", "camera"]:
                    pass  # Already detached above
                else:
                    self._gizmo.node.detach_parent()
                logger.log_info("Successfully cleaned up gizmo node")
            except Exception as e:
                logger.log_warning(f"Failed to cleanup gizmo node: {e}")

        # Clear pending transform
        if hasattr(self, "_pending_target_transform"):
            self._pending_target_transform = None

        # Directly release references
        self._gizmo = None
        self.target = None
