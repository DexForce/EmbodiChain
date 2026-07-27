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
"""Interactive gizmos for simulation objects, robots, and cameras."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import dexsim
import numpy as np
import torch
import warp as wp
from dexsim.types import (
    AxisArrowType,
    AxisCornerType,
    AxisOption,
    AxisTagType,
    InputKey,
    RotationRingsOption,
)
from scipy.spatial.transform import Rotation as R

from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.objects.rigid_object import RigidObject
from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.sensors import Camera
from embodichain.lab.sim.utility.gizmo_utils import create_gizmo_callback
from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    from dexsim.kit.ik import IKGizmoController, NewtonChainIK

__all__ = ["Gizmo", "GizmoCfg"]


@configclass
class GizmoCfg:
    """Configure gizmo appearance and robot Newton IK behavior."""

    axis_length_x: float = 0.2
    """Length of the X-axis arrow."""

    axis_length_y: float = 0.2
    """Length of the Y-axis arrow."""

    axis_length_z: float = 0.2
    """Length of the Z-axis arrow."""

    axis_size: float = 0.01
    """Thickness of the axis lines."""

    arrow_type: AxisArrowType = AxisArrowType.CONE
    """Type of arrow head."""

    corner_type: AxisCornerType = AxisCornerType.SPHERE
    """Type of axis corner."""

    tag_type: AxisTagType = AxisTagType.PLANE
    """Type of axis label."""

    rings_radius: float = 0.15
    """Radius of the rotation rings."""

    rings_size: float = 0.01
    """Thickness of the rotation rings."""

    ik_root_link_name: str | None = None
    """Robot IK chain root link.

    When omitted, the value is read from the selected control part's configured
    EmbodiChain solver.
    """

    ik_end_link_name: str | None = None
    """Robot IK chain end link.

    When omitted, the value is read from the selected control part's configured
    EmbodiChain solver.
    """

    ik_tcp_pose: torch.Tensor | np.ndarray | list[list[float]] | None = None
    """End-link-to-TCP transform for robot IK.

    When omitted, the configured EmbodiChain solver TCP is used if available;
    otherwise the identity transform is used.
    """

    ik_iterations: int = 24
    """Number of Newton IK iterations per changed target."""

    ik_device: str | None = None
    """Warp device for the Newton IK model, or the robot device when omitted."""

    ik_gizmo_scale: float = 1.5
    """Isotropic scale of dexsim's robot IK target gizmo."""

    ik_toggle_key: InputKey = InputKey.SCANCODE_I
    """Window key used by dexsim to toggle the robot IK gizmo."""

    def to_options_dict(self) -> dict[str, object]:
        """Convert the visual configuration to dexsim gizmo options.

        Returns:
            The axis and rotation-ring options used by rigid-object and camera
            gizmos.
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
                radius=self.rings_radius,
                size=self.rings_size,
            ),
        }


class _RobotGizmoAdapter:
    """Expose one EmbodiChain robot control part to dexsim's IK controller."""

    def __init__(self, robot: Robot, control_part: str, env_id: int = 0) -> None:
        """Create the adapter.

        Args:
            robot: EmbodiChain robot whose state is synchronized.
            control_part: Robot control part driven by the IK solution.
            env_id: Environment instance exposed to the interactive controller.

        Raises:
            ValueError: If the control part, environment, or joint selection is
                invalid.
        """
        if not robot.control_parts or control_part not in robot.control_parts:
            raise ValueError(
                f"Control part {control_part!r} is not defined. Available parts: "
                f"{list(robot.control_parts or {})}."
            )
        if env_id < 0 or env_id >= robot.num_instances:
            raise ValueError(
                f"Robot gizmo env_id={env_id} is outside [0, {robot.num_instances})."
            )

        joint_ids = robot.get_joint_ids(control_part, remove_mimic=True)
        if not joint_ids:
            raise ValueError(
                f"Control part {control_part!r} has no non-mimic active joints."
            )

        self.robot = robot
        self.control_part = control_part
        self.env_id = env_id
        self.joint_ids = list(joint_ids)
        self.joint_names = [robot.joint_names[index] for index in self.joint_ids]

    def get_current_qpos(self) -> np.ndarray:
        """Return current selected joint positions in dexsim joint-name order."""
        return self._selected_qpos(target=False)

    def get_target_qpos(self) -> np.ndarray:
        """Return target selected joint positions in dexsim joint-name order."""
        return self._selected_qpos(target=True)

    def set_current_qpos(self, qpos: np.ndarray) -> None:
        """Write selected current positions through the EmbodiChain abstraction."""
        self._set_qpos(qpos, target=False)

    def set_target_qpos(self, qpos: np.ndarray) -> None:
        """Write selected drive targets through the EmbodiChain abstraction."""
        self._set_qpos(qpos, target=True)

    def get_actived_joint_names(self) -> list[str]:
        """Return selected active joint names using dexsim's API spelling."""
        return self.joint_names.copy()

    def get_world_pose(self) -> np.ndarray:
        """Return the selected robot instance's root pose as a matrix."""
        pose = self.robot.get_local_pose(to_matrix=True)[self.env_id]
        return pose.detach().cpu().numpy().astype(np.float32, copy=True)

    def get_link_names(self, include_fixed: bool = True) -> list[str]:
        """Return all runtime link names.

        Args:
            include_fixed: Kept for compatibility with the dexsim articulation
                API. EmbodiChain's link list already includes fixed links.
        """
        del include_fixed
        return list(self.robot.link_names)

    def get_link_pose(self, link_name: str) -> np.ndarray:
        """Return one runtime link pose as a world-space matrix."""
        pose = self.robot.get_link_pose(
            link_name,
            env_ids=[self.env_id],
            to_matrix=True,
        )[0]
        return pose.detach().cpu().numpy().astype(np.float32, copy=True)

    def _selected_qpos(self, target: bool) -> np.ndarray:
        qpos = self.robot.get_qpos(target=target)[self.env_id, self.joint_ids]
        return qpos.detach().cpu().numpy().astype(np.float32, copy=True)

    def _set_qpos(self, qpos: np.ndarray, target: bool) -> None:
        values = np.asarray(qpos, dtype=np.float32)
        if values.shape != (len(self.joint_ids),):
            raise ValueError(
                f"Expected qpos shape ({len(self.joint_ids)},), got {values.shape}."
            )
        self.robot.set_qpos(
            qpos=torch.as_tensor(
                values,
                dtype=torch.float32,
                device=self.robot.device,
            ).unsqueeze(0),
            joint_ids=self.joint_ids,
            env_ids=[self.env_id],
            target=target,
        )


class Gizmo:
    """Control one rigid object, robot end effector, or camera interactively.

    Robot targets use dexsim's :class:`IKGizmoController` and
    :class:`NewtonChainIK`. Rigid-object and camera behavior remains on the
    existing direct/proxy paths.

    .. attention::
        Gizmos currently expose only one environment instance. Create them only
        when ``num_envs=1``.
    """

    def __init__(
        self,
        target: BatchEntity,
        cfg: GizmoCfg | None = None,
        control_part: str | None = "arm",
    ) -> None:
        """Create and attach a gizmo.

        Args:
            target: Simulation element to control.
            cfg: Gizmo appearance and robot IK configuration.
            control_part: Robot control part. When omitted, the first configured
                part is selected.
        """
        world = dexsim.default_world()
        if world is None:
            raise RuntimeError("A dexsim world must exist before creating a gizmo.")

        self.cfg = cfg if cfg is not None else GizmoCfg()
        self._world = world
        self._env = world.get_env()
        self._control_part = control_part
        self._callback: Callable[..., Any] | None = None
        self._state = "active"
        self._is_visible = True
        self._gizmo: object | None = None
        self._proxy_cube: object | None = None
        self._pending_target_transform: torch.Tensor | None = None
        self._ik_model: object | None = None
        self._ik_solver: NewtonChainIK | None = None
        self._ik_controller: IKGizmoController | None = None
        self._robot_adapter: _RobotGizmoAdapter | None = None
        self.target: BatchEntity | None = None
        self._target_type = ""
        self._attach_target(target)

    def _attach_target(self, target: BatchEntity) -> None:
        num_instances = int(getattr(target, "num_instances", dexsim.get_world_num()))
        if num_instances > 1:
            raise RuntimeError(
                "Gizmo can only be used in single environment mode "
                f"(num_envs=1), but target has {num_instances} instances."
            )

        self.target = target
        self._target_type = self._detect_target_type(target)
        if self._target_type == "robot":
            self._setup_robot_gizmo()
            return

        self._gizmo = self._create_gizmo(self.cfg)
        if self._target_type == "rigidobject":
            self._setup_rigid_object_gizmo()
        else:
            self._setup_camera_gizmo()

    @staticmethod
    def _detect_target_type(target: BatchEntity) -> str:
        if isinstance(target, Robot):
            return "robot"
        if isinstance(target, Camera):
            return "camera"
        if isinstance(target, RigidObject):
            return "rigidobject"
        raise ValueError(
            f"Unsupported target type: {type(target)}. Only RigidObject, Robot, "
            "and Camera are supported."
        )

    def _create_gizmo(self, cfg: GizmoCfg) -> object:
        options = cfg.to_options_dict()
        return self._env.create_gizmo(options["axis"], options["rings"])

    def _setup_rigid_object_gizmo(self) -> None:
        target = self._require_target()
        target_node = target._entities[0].node
        self._require_gizmo().follow(target_node)
        self._require_gizmo().set_flush_localpose_callback(create_gizmo_callback())

    def _setup_robot_gizmo(self) -> None:
        try:
            from dexsim.kit.ik import (
                IKApplyMode,
                IKGizmoController,
                NewtonChainIK,
                build_newton_model_from_urdf,
            )
        except ImportError as error:
            raise RuntimeError(
                "Robot gizmo requires a dexsim build that exports "
                "IKGizmoController, NewtonChainIK, and "
                "build_newton_model_from_urdf."
            ) from error

        target = self._require_robot()
        control_parts = list(target.control_parts or {})
        if not control_parts:
            raise ValueError("Robot has no control parts defined.")
        if self._control_part is None:
            self._control_part = control_parts[0]
        if self._control_part not in control_parts:
            raise ValueError(
                f"Control part {self._control_part!r} is not defined. Available "
                f"parts: {control_parts}."
            )

        root_link, end_link, tcp_pose = self._resolve_robot_ik_chain(target)
        if self.cfg.ik_iterations <= 0:
            raise ValueError("ik_iterations must be greater than zero.")
        if not np.isfinite(self.cfg.ik_gizmo_scale) or self.cfg.ik_gizmo_scale <= 0:
            raise ValueError("ik_gizmo_scale must be positive and finite.")

        adapter = _RobotGizmoAdapter(target, self._control_part)
        ik_device = self.cfg.ik_device or str(target.device)
        with wp.ScopedDevice(ik_device):
            ik_model = build_newton_model_from_urdf(
                target.cfg.fpath,
                hide_visuals=True,
            )
            ik_solver = NewtonChainIK(
                ik_model,
                start_link=root_link,
                end_link=end_link,
                iterations=self.cfg.ik_iterations,
                tcp_pose=tcp_pose,
            )

        current_qpos = adapter.get_current_qpos()
        ik_solver.set_qpos_from_joint_names(
            adapter.get_actived_joint_names(),
            current_qpos,
        )
        base_pose = adapter.get_world_pose()
        ik_solver.sync_target_state_from_link(adapter, base_pose)

        target_name = getattr(target.cfg, "uid", "robot")
        ik_controller = IKGizmoController(
            self._world,
            adapter,
            ik_solver,
            base_state={"pose": base_pose},
            toggle_key=self.cfg.ik_toggle_key,
            follow_robot_base=True,
            apply_mode=IKApplyMode.DRIVE_TARGET,
            gizmo_scale=self.cfg.ik_gizmo_scale,
            name=f"{target_name}_{self._control_part}_ik",
        )

        self._robot_adapter = adapter
        self._ik_model = ik_model
        self._ik_solver = ik_solver
        self._ik_controller = ik_controller
        self._gizmo = ik_controller.target_gizmo.gizmo
        logger.log_info(
            f"Robot gizmo uses dexsim Newton IK for control part "
            f"{self._control_part!r} ({root_link} -> {end_link})."
        )

    def _resolve_robot_ik_chain(
        self,
        target: Robot,
    ) -> tuple[str, str, np.ndarray]:
        solver = (
            target.get_solver(self._control_part)
            if target.cfg.solver_cfg is not None
            else None
        )
        root_link = self.cfg.ik_root_link_name or getattr(
            solver,
            "root_link_name",
            None,
        )
        end_link = self.cfg.ik_end_link_name or getattr(
            solver,
            "end_link_name",
            None,
        )
        if not root_link or not end_link:
            raise ValueError(
                "Robot gizmo needs an IK chain. Set GizmoCfg.ik_root_link_name "
                "and ik_end_link_name, or configure a solver for the selected "
                "robot control part."
            )

        tcp_pose = self.cfg.ik_tcp_pose
        if tcp_pose is None and solver is not None:
            tcp_pose = solver.get_tcp()
        if tcp_pose is None:
            tcp_pose = np.eye(4, dtype=np.float32)
        if isinstance(tcp_pose, torch.Tensor):
            tcp_pose = tcp_pose.detach().cpu().numpy()
        return root_link, end_link, np.asarray(tcp_pose, dtype=np.float32)

    def _setup_camera_gizmo(self) -> None:
        target = self._require_target()
        camera_pose = target.get_local_pose(to_matrix=True)[0]
        camera_pos = camera_pose[:3, 3].detach().cpu().numpy()
        camera_rotation = camera_pose[:3, :3].detach().cpu().numpy()
        self._proxy_cube = self._create_proxy_cube(
            camera_pos,
            camera_rotation,
            "Camera",
        )
        self._require_gizmo().set_flush_localpose_callback(self._proxy_gizmo_callback)

    def _create_proxy_cube(
        self,
        position: np.ndarray,
        rotation_matrix: np.ndarray,
        name: str,
    ) -> object:
        euler = R.from_matrix(rotation_matrix).as_euler("xyz", degrees=False)
        proxy_cube = self._env.create_cube(0.02, 0.02, 0.02)
        proxy_cube.set_location(*position)
        proxy_cube.set_rotation_euler(*euler)
        self._require_gizmo().follow(proxy_cube.node)
        logger.log_info(f"{name} gizmo proxy created at position: {position}.")
        return proxy_cube

    def _proxy_gizmo_callback(self, *args: object) -> None:
        if len(args) != 3 or self._proxy_cube is None:
            return
        node, local_pose, flag = args
        if node is None:
            return

        if isinstance(local_pose, torch.Tensor):
            pose = local_pose.detach().cpu().numpy()
        else:
            pose = np.asarray(local_pose)
        if pose.shape != (4, 4):
            return

        node.set_transform(pose, flag)
        position = pose[:3, 3]
        euler = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)
        self._proxy_cube.set_location(*position)
        self._proxy_cube.set_rotation_euler(*euler)
        self._pending_target_transform = torch.as_tensor(
            pose,
            dtype=torch.float32,
        ).unsqueeze(0)

    def _update_camera_pose(self, target_transform: torch.Tensor) -> bool:
        try:
            self._require_target().set_local_pose(target_transform)
            return True
        except Exception as error:
            logger.log_error(f"Error updating camera pose: {error}")
            return False

    def attach(self, target: BatchEntity) -> None:
        """Attach this controller to another supported single-instance target."""
        self._release_resources()
        self._attach_target(target)

    def detach(self) -> None:
        """Detach the gizmo and release target-specific controller resources."""
        self._release_resources()
        self.target = None
        self._target_type = ""

    def set_transform_callback(self, callback: Callable[..., Any]) -> None:
        """Set an additional raw gizmo transform callback."""
        self._callback = callback
        self._require_gizmo().set_transform_flush_callback(callback)

    def set_world_pose(self, pose: np.ndarray) -> None:
        """Set the underlying gizmo's world pose."""
        self._require_gizmo().set_world_pose(pose)

    def set_local_pose(self, pose: np.ndarray) -> None:
        """Set the underlying gizmo's local pose."""
        self._require_gizmo().set_local_pose(pose)

    def set_line_width(self, width: float) -> None:
        """Set the underlying gizmo line width."""
        self._require_gizmo().set_line_width(width)

    def enable_collision(self, enabled: bool) -> None:
        """Enable or disable gizmo collision."""
        self._require_gizmo().enable_collision(enabled)

    def get_world_pose(self) -> np.ndarray:
        """Return the underlying gizmo's world pose."""
        return self._require_gizmo().get_world_pose()

    def get_local_pose(self) -> np.ndarray:
        """Return the underlying gizmo's local pose."""
        return self._require_gizmo().get_local_pose()

    def get_name(self) -> str:
        """Return the underlying gizmo name."""
        return self._require_gizmo().get_name()

    def get_parent(self) -> object:
        """Return the underlying gizmo parent."""
        return self._require_gizmo().get_parent()

    def toggle_visibility(self) -> bool:
        """Toggle gizmo visibility and return the new state."""
        visible = not self.is_visible()
        self.set_visible(visible)
        return visible

    def set_visible(self, visible: bool) -> None:
        """Set gizmo visibility."""
        self._is_visible = bool(visible)
        if self._ik_controller is not None:
            self._ik_controller.enabled = self._is_visible
        gizmo = self._gizmo
        if gizmo is not None:
            gizmo.set_visible(self._is_visible)

    def is_visible(self) -> bool:
        """Return whether the gizmo is visible."""
        if self._ik_controller is not None:
            return bool(self._ik_controller.enabled)
        return self._is_visible

    def update(self) -> None:
        """Synchronize the gizmo and apply pending target changes."""
        if self.target is None:
            return
        if self._target_type == "rigidobject":
            target_node = self.target._entities[0].node
            self._require_gizmo().follow(target_node)
        elif self._target_type == "robot":
            if self._ik_controller is not None:
                self._ik_controller.update(iterations=self.cfg.ik_iterations)
        elif self._target_type == "camera":
            if self._proxy_cube is not None:
                camera_pose = self.target.get_local_pose(to_matrix=True)[0]
                position = camera_pose[:3, 3].detach().cpu().numpy()
                self._proxy_cube.set_location(*position)
            if self._pending_target_transform is not None:
                self._update_camera_pose(self._pending_target_transform)
                self._pending_target_transform = None

    def apply_transform(
        self,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> None:
        """Apply a direct transform where the target path supports it."""
        if self.target is None:
            return
        if self._target_type == "rigidobject":
            self.target.set_location(*translation)
            self.target.set_rotation_euler(*rotation)
        elif self._target_type == "camera" and self._proxy_cube is not None:
            self._proxy_cube.set_location(*translation)
            self._proxy_cube.set_rotation_euler(*rotation)

    def destroy(self) -> None:
        """Release gizmo resources and target references."""
        self._release_resources()
        self.target = None
        self._target_type = ""

    def _release_resources(self) -> None:
        gizmo = self._gizmo
        if gizmo is not None:
            for method_name in (
                "set_flush_localpose_callback",
                "set_transform_flush_callback",
            ):
                method = getattr(gizmo, method_name, None)
                if callable(method):
                    try:
                        method(None)
                    except (TypeError, RuntimeError):
                        pass
            try:
                gizmo.set_visible(False)
            except (AttributeError, TypeError, RuntimeError):
                pass
            try:
                gizmo.detach_parent()
            except (AttributeError, TypeError, RuntimeError):
                pass

        if self._ik_controller is not None:
            target_node = self._ik_controller.target_gizmo.target_node
            try:
                target_node.detach_parent()
            except (AttributeError, TypeError, RuntimeError):
                pass

        if self._proxy_cube is not None:
            try:
                self._env.remove_actor(self._proxy_cube)
            except (AttributeError, TypeError, RuntimeError) as error:
                logger.log_warning(f"Failed to remove gizmo proxy cube: {error}")

        if gizmo is not None:
            remove_gizmo = getattr(self._env, "remove_gizmo", None)
            if callable(remove_gizmo):
                try:
                    remove_gizmo(gizmo)
                except (AttributeError, TypeError, RuntimeError) as error:
                    logger.log_warning(
                        f"Failed to remove gizmo from dexsim environment: {error}"
                    )

        self._pending_target_transform = None
        self._proxy_cube = None
        self._gizmo = None
        self._ik_controller = None
        self._ik_solver = None
        self._ik_model = None
        self._robot_adapter = None

    def _require_gizmo(self) -> object:
        if self._gizmo is None:
            raise RuntimeError("Gizmo is not attached.")
        return self._gizmo

    def _require_target(self) -> BatchEntity:
        if self.target is None:
            raise RuntimeError("Gizmo has no target.")
        return self.target

    def _require_robot(self) -> Robot:
        target = self._require_target()
        if not isinstance(target, Robot):
            raise TypeError(f"Expected Robot target, got {type(target)}.")
        return target
