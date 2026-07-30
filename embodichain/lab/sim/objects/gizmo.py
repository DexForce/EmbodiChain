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

"""Backend-neutral Gizmo target control with an optional DexSim handle."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import dexsim
import numpy as np
import torch
from dexsim.types import (
    AxisArrowType,
    AxisCornerType,
    AxisOption,
    AxisTagType,
    RotationRingsOption,
)
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.sensors import Camera
from embodichain.utils import configclass, logger

__all__ = ["Gizmo", "GizmoCfg"]


@configclass
class GizmoCfg:
    """Configure native and Viser Gizmo appearance."""

    axis_length_x: float = 0.2
    """Length of the X-axis arrow."""

    axis_length_y: float = 0.2
    """Length of the Y-axis arrow."""

    axis_length_z: float = 0.2
    """Length of the Z-axis arrow."""

    axis_size: float = 0.01
    """Thickness of the native axis lines."""

    arrow_type: AxisArrowType = AxisArrowType.CONE
    """Native axis arrow-head style."""

    corner_type: AxisCornerType = AxisCornerType.SPHERE
    """Native axis corner style."""

    tag_type: AxisTagType = AxisTagType.PLANE
    """Native axis-label style."""

    rings_radius: float = 0.15
    """Radius of the rotation rings."""

    rings_size: float = 0.01
    """Thickness of the native rotation rings."""

    def to_options_dict(self) -> dict[str, AxisOption | RotationRingsOption]:
        """Convert this configuration to DexSim Gizmo options."""
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
                radius=self.rings_radius,
                size=self.rings_size,
            ),
        }


class Gizmo:
    """Control a rigid object, robot end-effector, or camera.

    Target mutation is backend-neutral: both DexSim callbacks and Viser commands
    submit a local target pose, and :meth:`update` applies it on the simulation
    thread. A DexSim Gizmo handle is optional, which allows the same controller
    to work in a headless Viser process.

    .. attention::
        Gizmo control currently supports exactly one simulation environment.

    Args:
        target: Simulation element controlled by this Gizmo.
        cfg: Appearance configuration.
        control_part: Robot control part used for FK and IK.
        enable_native: Whether to create a DexSim Gizmo and proxy actor.
    """

    def __init__(
        self,
        target: BatchEntity,
        cfg: GizmoCfg | None = None,
        control_part: str | None = "arm",
        *,
        enable_native: bool = True,
    ) -> None:
        num_envs = dexsim.get_world_num()
        if num_envs > 1:
            raise RuntimeError(
                "Gizmo can only be used in single environment mode "
                f"(num_envs=1), but current num_envs={num_envs}."
            )

        self.target: BatchEntity | None = target
        self.cfg = cfg or GizmoCfg()
        self._control_part = control_part
        self._target_type = self._detect_target_type(target)
        self._env = dexsim.default_world().get_env()
        self._gizmo: object | None = None
        self._proxy_cube: object | None = None
        self._callback: Callable[..., Any] | None = None
        self._is_visible = True
        self._state_lock = threading.RLock()
        self._interaction_owner: str | None = None
        self._pending_target_transform: torch.Tensor | None = None
        self._desired_target_transform: torch.Tensor | None = None
        self._robot_arm_name: str | None = None

        if self._target_type == "robot":
            self._configure_robot()
        self._desired_target_transform = self._read_target_pose()

        if enable_native:
            self._gizmo = self._create_native_gizmo(self.cfg)
            self._setup_native_gizmo()

    @property
    def target_type(self) -> str:
        """Return ``rigid_object``, ``robot``, or ``camera``."""
        return self._target_type

    @property
    def control_part(self) -> str | None:
        """Return the robot control part, if applicable."""
        return self._control_part

    @property
    def native_enabled(self) -> bool:
        """Whether this controller owns a DexSim Gizmo handle."""
        return self._gizmo is not None

    def _detect_target_type(self, target: BatchEntity) -> str:
        if isinstance(target, Robot):
            return "robot"
        if isinstance(target, Camera):
            return "camera"
        if isinstance(target, RigidObject):
            return "rigid_object"
        raise ValueError(
            f"Unsupported Gizmo target type {type(target)!r}; expected "
            "RigidObject, Robot, or Camera."
        )

    def _configure_robot(self) -> None:
        if self.target is None or not isinstance(self.target, Robot):
            raise RuntimeError("Robot Gizmo has no attached Robot.")
        if self.target.cfg.solver_cfg is None:
            raise ValueError("Robot has no solver configured for Gizmo IK/FK.")
        arm_names = list(self.target.control_parts.keys())
        if not arm_names:
            raise ValueError("Robot has no control parts defined.")
        if self._control_part is None:
            self._robot_arm_name = arm_names[0]
            self._control_part = self._robot_arm_name
        elif self._control_part in arm_names:
            self._robot_arm_name = self._control_part
        else:
            raise ValueError(
                f"Control part {self._control_part!r} was not found; "
                f"available parts are {arm_names}."
            )

    def _target_device(self) -> torch.device:
        if self.target is None:
            return torch.device("cpu")
        return torch.device(getattr(self.target, "device", "cpu"))

    @staticmethod
    def _as_pose_matrix(pose: object, device: torch.device) -> torch.Tensor:
        matrix = torch.as_tensor(pose, dtype=torch.float32, device=device)
        if matrix.shape == (4, 4):
            matrix = matrix.unsqueeze(0)
        if matrix.shape != (1, 4, 4):
            raise ValueError(
                "Gizmo target pose must have shape (4, 4) or (1, 4, 4), "
                f"received {tuple(matrix.shape)}."
            )
        if not bool(torch.isfinite(matrix).all().item()):
            raise ValueError("Gizmo target pose must contain only finite values.")
        return matrix.detach().clone()

    def _compute_ee_pose_fk(self) -> torch.Tensor:
        if self.target is None or not isinstance(self.target, Robot):
            raise RuntimeError("Robot Gizmo has no attached Robot.")
        if self._robot_arm_name is None:
            raise RuntimeError("Robot Gizmo control part is not configured.")
        current_qpos = self.target.get_proprioception()["qpos"]
        joint_ids = self.target.get_joint_ids(self._robot_arm_name)
        joint_positions = current_qpos[:, joint_ids]
        pose = self.target.compute_fk(
            joint_positions,
            name=self._robot_arm_name,
            env_ids=[0],
            to_matrix=True,
        )
        if pose is None:
            raise RuntimeError("Robot forward kinematics returned no pose.")
        return self._as_pose_matrix(pose, self._target_device())

    def _read_target_pose(self) -> torch.Tensor:
        if self.target is None:
            raise RuntimeError("Gizmo is detached.")
        if self._target_type == "robot":
            return self._compute_ee_pose_fk()
        pose = self.target.get_local_pose(to_matrix=True)
        return self._as_pose_matrix(pose[0], self._target_device())

    def get_control_pose(self) -> torch.Tensor:
        """Return the Gizmo pose in the local arena frame as ``(1, 4, 4)``."""
        with self._state_lock:
            if (
                self._target_type == "robot"
                and self._desired_target_transform is not None
            ):
                return self._desired_target_transform.detach().clone()
        return self._read_target_pose()

    def begin_interaction(self, source_id: str) -> bool:
        """Acquire this Gizmo for one native or Viser drag source."""
        if not source_id:
            raise ValueError("source_id must not be empty.")
        with self._state_lock:
            if self._interaction_owner not in {None, source_id}:
                return False
            self._interaction_owner = source_id
            return True

    def request_local_pose(self, pose: object, *, source_id: str) -> bool:
        """Queue a local target pose for application by :meth:`update`.

        Returns:
            ``False`` if another client currently owns the drag.
        """
        matrix = self._as_pose_matrix(pose, self._target_device())
        with self._state_lock:
            if self._interaction_owner not in {None, source_id}:
                return False
            self._pending_target_transform = matrix
            self._desired_target_transform = matrix.detach().clone()
            return True

    def end_interaction(self, source_id: str) -> bool:
        """Release a drag source's ownership of this Gizmo."""
        with self._state_lock:
            if self._interaction_owner != source_id:
                return False
            self._interaction_owner = None
            return True

    def cancel_interaction(self, source_prefix: str | None = None) -> bool:
        """Cancel the active owner, optionally only for a source prefix."""
        with self._state_lock:
            owner = self._interaction_owner
            if owner is None:
                return False
            if source_prefix is not None and not owner.startswith(source_prefix):
                return False
            self._interaction_owner = None
            return True

    def _create_native_gizmo(self, cfg: GizmoCfg) -> object:
        options = cfg.to_options_dict()
        return self._env.create_gizmo(options["axis"], options["rings"])

    def _create_proxy_cube(self, pose: torch.Tensor, name: str) -> object:
        matrix = pose[0].detach().cpu().numpy()
        euler = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=False)
        proxy_cube = self._env.create_cube(0.02, 0.02, 0.02)
        proxy_cube.set_location(*matrix[:3, 3].tolist())
        proxy_cube.set_rotation_euler(*euler.tolist())
        self._require_native().follow(proxy_cube.node)
        logger.log_info(
            f"{name} Gizmo proxy created at position: {matrix[:3, 3].tolist()}"
        )
        return proxy_cube

    def _set_proxy_pose(self, pose: torch.Tensor) -> None:
        if self._proxy_cube is None:
            return
        matrix = pose[0].detach().cpu().numpy()
        euler = Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz", degrees=False)
        self._proxy_cube.set_location(*matrix[:3, 3].tolist())
        self._proxy_cube.set_rotation_euler(*euler.tolist())

    def _native_pose_callback(self, *args: object) -> None:
        if len(args) != 3 or args[0] is None:
            return
        try:
            pose = self._as_pose_matrix(args[1], self._target_device())
            if self._proxy_cube is not None:
                self._set_proxy_pose(pose)
            if not self.request_local_pose(pose, source_id="native"):
                self._set_proxy_pose(self.get_control_pose())
        except (TypeError, ValueError) as error:
            logger.log_warning(f"Ignoring invalid native Gizmo pose: {error}")

    def _setup_native_gizmo(self) -> None:
        native = self._require_native()
        if self.target is None:
            raise RuntimeError("Gizmo is detached.")
        if self._target_type == "rigid_object":
            native.follow(self.target._entities[0].node)
        else:
            label = "Robot" if self._target_type == "robot" else "Camera"
            self._proxy_cube = self._create_proxy_cube(
                self.get_control_pose(),
                label,
            )
        native.set_flush_localpose_callback(self._native_pose_callback)

    def _update_camera_pose(self, target_transform: torch.Tensor) -> bool:
        if self.target is None or not isinstance(self.target, Camera):
            return False
        try:
            self.target.set_local_pose(target_transform, env_ids=[0])
            return True
        except Exception as error:
            logger.log_error(f"Error updating camera pose: {error}")
            return False

    def _update_rigid_object_pose(self, target_transform: torch.Tensor) -> bool:
        if self.target is None or not isinstance(self.target, RigidObject):
            return False
        try:
            self.target.set_local_pose(target_transform, env_ids=[0])
            return True
        except Exception as error:
            logger.log_error(f"Error updating rigid object pose: {error}")
            return False

    def _update_robot_ik(self, target_transform: torch.Tensor) -> bool:
        if self.target is None or not isinstance(self.target, Robot):
            return False
        if self._robot_arm_name is None:
            return False
        try:
            current_qpos = self.target.get_proprioception()["qpos"]
            joint_ids = self.target.get_joint_ids(self._robot_arm_name)
            if len(joint_ids) == 0:
                logger.log_warning(
                    f"No joint IDs found for control part {self._robot_arm_name!r}."
                )
                return False
            joint_seed = current_qpos[:, joint_ids]
            result = self.target.compute_ik(
                pose=target_transform,
                name=self._robot_arm_name,
                joint_seed=joint_seed,
                env_ids=[0],
            )
            if result is None:
                return False
            success, new_qpos = result
            if not bool(torch.as_tensor(success).reshape(-1)[0].item()):
                logger.log_warning("Gizmo IK solution not found.")
                return False
            new_qpos = torch.as_tensor(
                new_qpos,
                dtype=torch.float32,
                device=self._target_device(),
            ).reshape(1, -1)
            self.target.set_qpos(
                qpos=new_qpos,
                joint_ids=joint_ids,
                env_ids=[0],
            )
            return True
        except Exception as error:
            logger.log_error(f"Error in Gizmo robot IK: {error}")
            return False

    def update(self) -> None:
        """Apply the latest queued target pose on the simulation thread."""
        with self._state_lock:
            pending = self._pending_target_transform
            self._pending_target_transform = None

        if pending is not None:
            if self._target_type == "rigid_object":
                self._update_rigid_object_pose(pending)
            elif self._target_type == "robot":
                self._update_robot_ik(pending)
            elif self._target_type == "camera":
                self._update_camera_pose(pending)
            self._set_proxy_pose(pending)

        if self._gizmo is None or self.target is None:
            return
        if self._target_type == "rigid_object":
            self._gizmo.follow(self.target._entities[0].node)
        elif self._target_type == "camera" and pending is None:
            self._set_proxy_pose(self._read_target_pose())

    def attach(self, target: BatchEntity) -> None:
        """Attach this Gizmo to a supported target."""
        self.target = target
        self._target_type = self._detect_target_type(target)
        self._robot_arm_name = None
        if self._target_type == "robot":
            self._configure_robot()
        with self._state_lock:
            self._interaction_owner = None
            self._pending_target_transform = None
            self._desired_target_transform = self._read_target_pose()
        if self._gizmo is not None:
            self._remove_proxy_cube()
            self._setup_native_gizmo()

    def detach(self) -> None:
        """Detach this Gizmo from its current target."""
        if self._gizmo is not None:
            self._gizmo.detach_parent()
        self._remove_proxy_cube()
        with self._state_lock:
            self._interaction_owner = None
            self._pending_target_transform = None
        self.target = None

    def _require_native(self) -> object:
        if self._gizmo is None:
            raise RuntimeError("This Gizmo was created without a native DexSim handle.")
        return self._gizmo

    def set_transform_callback(self, callback: Callable[..., Any]) -> None:
        """Set a callback directly on the native transform handle."""
        self._callback = callback
        self._require_native().set_transform_flush_callback(callback)

    def set_world_pose(self, pose: object) -> None:
        """Set the native Gizmo world pose."""
        self._require_native().set_world_pose(pose)

    def set_local_pose(self, pose: object) -> None:
        """Set the native Gizmo pose or queue it for a headless controller."""
        if self._gizmo is None:
            self.request_local_pose(pose, source_id="api")
        else:
            self._gizmo.set_local_pose(pose)

    def set_line_width(self, width: float) -> None:
        """Set the native Gizmo line width."""
        self._require_native().set_line_width(width)

    def enable_collision(self, enabled: bool) -> None:
        """Enable or disable native Gizmo collision."""
        self._require_native().enable_collision(enabled)

    def get_world_pose(self) -> object:
        """Return the native Gizmo world pose."""
        return self._require_native().get_world_pose()

    def get_local_pose(self) -> object:
        """Return the native pose, or the logical local control pose."""
        if self._gizmo is None:
            return self.get_control_pose()
        return self._gizmo.get_local_pose()

    def get_name(self) -> object:
        """Return the native Gizmo node name."""
        return self._require_native().get_name()

    def get_parent(self) -> object:
        """Return the native Gizmo parent node."""
        return self._require_native().get_parent()

    def toggle_visibility(self) -> bool:
        """Toggle visibility and return the new state."""
        self.set_visible(not self._is_visible)
        return self._is_visible

    def set_visible(self, visible: bool) -> None:
        """Set native and Viser Gizmo visibility."""
        self._is_visible = bool(visible)
        if self._gizmo is not None:
            self._gizmo.set_visible(self._is_visible)

    def is_visible(self) -> bool:
        """Return whether this Gizmo should be visible."""
        return self._is_visible

    def apply_transform(
        self,
        translation: object,
        rotation: object,
    ) -> None:
        """Apply a translation and XYZ Euler rotation through the shared path."""
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = np.asarray(translation, dtype=np.float32)
        matrix[:3, :3] = Rotation.from_euler(
            "xyz",
            np.asarray(rotation, dtype=np.float32),
        ).as_matrix()
        self.request_local_pose(matrix, source_id="api")

    def _remove_proxy_cube(self) -> None:
        if self._proxy_cube is None:
            return
        try:
            if self._gizmo is not None:
                self._gizmo.detach_parent()
            self._env.remove_actor(self._proxy_cube)
        except Exception as error:
            logger.log_warning(f"Failed to remove Gizmo proxy: {error}")
        self._proxy_cube = None

    def destroy(self) -> None:
        """Release native resources and target references."""
        if self._gizmo is not None and hasattr(self._gizmo, "node"):
            try:
                self._gizmo.node.set_flush_transform_callback(None)
            except Exception as error:
                logger.log_warning(f"Failed to clear Gizmo callback: {error}")
        self._remove_proxy_cube()
        if self._gizmo is not None:
            try:
                self._gizmo.detach_parent()
            except Exception as error:
                logger.log_warning(f"Failed to detach Gizmo: {error}")
        with self._state_lock:
            self._interaction_owner = None
            self._pending_target_transform = None
            self._desired_target_transform = None
        self._gizmo = None
        self.target = None
