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
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.objects.rigid_object import RigidObject
from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.sensors import Camera
from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    from dexsim.kit.ik import IKGizmoController, NewtonChainIK

__all__ = ["Gizmo", "GizmoCfg"]


@configclass
class GizmoCfg:
    """Configure Gizmo appearance and robot IK behavior."""

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

    ik_root_link_name: str | None = None
    """Robot IK chain root link.

    When omitted, the selected control part's EmbodiChain solver supplies it.
    """

    ik_end_link_name: str | None = None
    """Robot IK chain end link.

    When omitted, the selected control part's EmbodiChain solver supplies it.
    """

    ik_tcp_pose: torch.Tensor | np.ndarray | list[list[float]] | None = None
    """End-link-to-TCP transform used by native DexSim robot IK."""

    ik_iterations: int = 24
    """Number of Newton IK iterations per changed native target."""

    ik_device: str | None = None
    """Warp device for native Newton IK, or the robot device when omitted."""

    ik_gizmo_scale: float = 1.5
    """Isotropic scale of the native robot IK target Gizmo."""

    ik_toggle_key: InputKey = InputKey.SCANCODE_I
    """Native-window key used to toggle the robot IK Gizmo."""

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


class _RobotGizmoAdapter:
    """Expose one EmbodiChain robot control part to DexSim's IK controller."""

    def __init__(self, robot: Robot, control_part: str, env_id: int = 0) -> None:
        if not robot.control_parts or control_part not in robot.control_parts:
            raise ValueError(
                f"Control part {control_part!r} is not defined. Available parts: "
                f"{list(robot.control_parts or {})}."
            )
        if env_id < 0 or env_id >= robot.num_instances:
            raise ValueError(
                f"Robot Gizmo env_id={env_id} is outside [0, {robot.num_instances})."
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
        """Return current selected joint positions in DexSim joint-name order."""
        return self._selected_qpos(target=False)

    def get_target_qpos(self) -> np.ndarray:
        """Return target selected joint positions in DexSim joint-name order."""
        return self._selected_qpos(target=True)

    def set_current_qpos(self, qpos: np.ndarray) -> None:
        """Write selected current positions through the robot abstraction."""
        self._set_qpos(qpos, target=False)

    def set_target_qpos(self, qpos: np.ndarray) -> None:
        """Write selected drive targets through the robot abstraction."""
        self._set_qpos(qpos, target=True)

    def get_actived_joint_names(self) -> list[str]:
        """Return active joint names using DexSim's API spelling."""
        return self.joint_names.copy()

    def get_world_pose(self) -> np.ndarray:
        """Return the selected robot instance's root pose as a matrix."""
        pose = self.robot.get_local_pose(to_matrix=True)[self.env_id]
        return pose.detach().cpu().numpy().astype(np.float32, copy=True)

    def get_link_names(self, include_fixed: bool = True) -> list[str]:
        """Return all runtime link names."""
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
        enable_native: Whether to create a native DexSim Gizmo handle. Robot
            Gizmos solve IK with DexSim Newton IK in both native and headless
            (Viser) modes; ``enable_native`` only controls whether a native
            window Gizmo handle is created for direct interaction.
    """

    def __init__(
        self,
        target: BatchEntity,
        cfg: GizmoCfg | None = None,
        control_part: str | None = "arm",
        *,
        enable_native: bool = True,
    ) -> None:
        world = dexsim.default_world()
        if world is None:
            raise RuntimeError("A DexSim world must exist before creating a Gizmo.")

        num_envs = int(getattr(target, "num_instances", dexsim.get_world_num()))
        if num_envs > 1:
            raise RuntimeError(
                "Gizmo can only be used in single environment mode "
                f"(num_envs=1), but target has {num_envs} instances."
            )

        self.target: BatchEntity | None = target
        self.cfg = cfg or GizmoCfg()
        self._world = world
        self._control_part = control_part
        self._target_type = self._detect_target_type(target)
        self._env = world.get_env()
        self._enable_native = enable_native
        self._gizmo: object | None = None
        self._proxy_cube: object | None = None
        self._callback: Callable[..., Any] | None = None
        self._is_visible = True
        self._state_lock = threading.RLock()
        self._interaction_owner: str | None = None
        self._pending_target_transform: torch.Tensor | None = None
        self._desired_target_transform: torch.Tensor | None = None
        self._robot_arm_name: str | None = None
        self._ik_model: object | None = None
        self._ik_solver: NewtonChainIK | None = None
        self._ik_controller: IKGizmoController | None = None
        self._robot_adapter: _RobotGizmoAdapter | None = None
        self._native_robot_end_link: str | None = None
        self._native_robot_tcp_pose: np.ndarray | None = None

        if self._target_type == "robot":
            self._configure_robot()
            self._setup_robot_ik_solver()
            if enable_native:
                self._setup_native_robot_gizmo()
            self._desired_target_transform = self._read_native_robot_pose()
        else:
            self._desired_target_transform = self._read_target_pose()

        if enable_native and self._target_type != "robot":
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

    def _setup_robot_ik_solver(self) -> None:
        """Build the shared DexSim Newton IK solver and robot adapter.

        The solver is shared by native and headless (Viser) robot Gizmos so both
        paths solve IK with DexSim Newton IK instead of an EmbodiChain solver.
        Native Gizmos additionally create an :class:`IKGizmoController` in
        :meth:`_setup_native_robot_gizmo`.
        """
        try:
            from dexsim.kit.ik import NewtonChainIK, build_newton_model_from_urdf
        except ImportError as error:
            raise RuntimeError(
                "Robot Gizmo requires a DexSim build that exports "
                "NewtonChainIK and build_newton_model_from_urdf."
            ) from error

        if self.target is None or not isinstance(self.target, Robot):
            raise RuntimeError("Robot Gizmo has no attached Robot.")
        if self._robot_arm_name is None:
            raise RuntimeError("Robot Gizmo control part is not configured.")
        if self.cfg.ik_iterations <= 0:
            raise ValueError("ik_iterations must be greater than zero.")

        root_link, end_link, tcp_pose = self._resolve_robot_ik_chain(self.target)
        adapter = _RobotGizmoAdapter(self.target, self._robot_arm_name)
        ik_device = self.cfg.ik_device or str(self.target.device)
        with wp.ScopedDevice(ik_device):
            ik_model = build_newton_model_from_urdf(
                self.target.cfg.fpath,
                hide_visuals=True,
            )
            ik_solver = NewtonChainIK(
                ik_model,
                start_link=root_link,
                end_link=end_link,
                iterations=self.cfg.ik_iterations,
                tcp_pose=tcp_pose,
            )

        ik_solver.set_qpos_from_joint_names(
            adapter.get_actived_joint_names(),
            adapter.get_current_qpos(),
        )
        base_pose = adapter.get_world_pose()
        ik_solver.sync_target_state_from_link(adapter, base_pose)

        self._robot_adapter = adapter
        self._ik_model = ik_model
        self._ik_solver = ik_solver
        self._native_robot_end_link = end_link
        self._native_robot_tcp_pose = tcp_pose
        logger.log_info(
            f"Robot Gizmo uses DexSim Newton IK for control part "
            f"{self._robot_arm_name!r} ({root_link} -> {end_link})."
        )

    def _setup_native_robot_gizmo(self) -> None:
        """Create DexSim's native IK controller on top of the shared solver."""
        try:
            from dexsim.kit.ik import IKApplyMode, IKGizmoController
        except ImportError as error:
            raise RuntimeError(
                "Robot Gizmo requires a DexSim build that exports "
                "IKGizmoController and IKApplyMode."
            ) from error

        if self._ik_solver is None or self._robot_adapter is None:
            raise RuntimeError("Robot Gizmo IK solver is not configured.")
        if self.target is None or not isinstance(self.target, Robot):
            raise RuntimeError("Robot Gizmo has no attached Robot.")
        if not np.isfinite(self.cfg.ik_gizmo_scale) or self.cfg.ik_gizmo_scale <= 0:
            raise ValueError("ik_gizmo_scale must be positive and finite.")

        base_pose = self._robot_adapter.get_world_pose()
        target_name = getattr(self.target.cfg, "uid", "robot")
        ik_controller = IKGizmoController(
            self._world,
            self._robot_adapter,
            self._ik_solver,
            base_state={"pose": base_pose},
            toggle_key=self.cfg.ik_toggle_key,
            follow_robot_base=True,
            apply_mode=IKApplyMode.DRIVE_TARGET,
            gizmo_scale=self.cfg.ik_gizmo_scale,
            name=f"{target_name}_{self._robot_arm_name}_ik",
        )

        self._ik_controller = ik_controller
        self._gizmo = ik_controller.target_gizmo.gizmo

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
                "Robot Gizmo needs an IK chain. Set GizmoCfg.ik_root_link_name "
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
        tcp_matrix = np.asarray(tcp_pose, dtype=np.float32)
        if tcp_matrix.shape != (4, 4) or not np.isfinite(tcp_matrix).all():
            raise ValueError("ik_tcp_pose must be a finite 4x4 transform.")
        return root_link, end_link, tcp_matrix

    def _read_native_robot_pose(self) -> torch.Tensor:
        """Read the native robot TCP pose without an EmbodiChain solver."""
        if self._robot_adapter is None:
            raise RuntimeError("Native robot Gizmo adapter is not configured.")
        link_pose = self._robot_adapter.get_link_pose(self._native_robot_end_link)
        tcp_pose = link_pose @ self._native_robot_tcp_pose
        return self._as_pose_matrix(tcp_pose, self._target_device())

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

    def _read_target_pose(self) -> torch.Tensor:
        if self.target is None:
            raise RuntimeError("Gizmo is detached.")
        if self._target_type == "robot":
            return self._read_native_robot_pose()
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
        if self._ik_solver is None or self._robot_adapter is None:
            return False
        try:
            from dexsim.kit.ik.pose import (
                local_pose_from_world,
                rotation_matrix_to_quat_xyzw,
            )

            # The queued target is the TCP transform in the arena-local frame.
            # Newton IK tracks a base-local target, so convert it with the same
            # helper the native gizmo callback uses (inv(base_pose) @ target).
            base_pose = self._robot_adapter.get_world_pose()
            target_pose = target_transform[0].detach().cpu().numpy().astype(np.float32)
            base_local = local_pose_from_world(base_pose, target_pose)
            position = np.asarray(base_local[:3, 3], dtype=np.float32)
            rotation = rotation_matrix_to_quat_xyzw(base_local[:3, :3])

            joint_names = self._robot_adapter.get_actived_joint_names()
            current_qpos = self._robot_adapter.get_current_qpos()
            self._ik_solver.set_target_pose(position, rotation)
            self._ik_solver.solve(
                joint_names,
                current_qpos,
                iterations=self.cfg.ik_iterations,
            )
            solved_qpos = self._ik_solver.qpos_for_joint_names(
                joint_names, current_qpos
            )
            # Drive the joint targets (matching native IKApplyMode.DRIVE_TARGET)
            # so physics moves the robot instead of snapping its current pose.
            self._robot_adapter.set_target_qpos(solved_qpos)
            return True
        except Exception as error:
            logger.log_error(f"Error in Gizmo robot IK: {error}")
            return False

    def update(self) -> None:
        """Apply the latest queued target pose on the simulation thread."""
        if self._ik_controller is not None:
            self._ik_controller.update(iterations=self.cfg.ik_iterations)
            return

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
        num_envs = int(getattr(target, "num_instances", dexsim.get_world_num()))
        if num_envs > 1:
            raise RuntimeError(
                "Gizmo can only be used in single environment mode "
                f"(num_envs=1), but target has {num_envs} instances."
            )

        self._release_native_resources()
        self.target = target
        self._target_type = self._detect_target_type(target)
        self._robot_arm_name = None
        if self._target_type == "robot":
            self._configure_robot()
            self._setup_robot_ik_solver()
            if self._enable_native:
                self._setup_native_robot_gizmo()
            desired_pose = self._read_native_robot_pose()
        else:
            desired_pose = self._read_target_pose()
            if self._enable_native:
                self._gizmo = self._create_native_gizmo(self.cfg)
                self._setup_native_gizmo()
        with self._state_lock:
            self._interaction_owner = None
            self._pending_target_transform = None
            self._desired_target_transform = desired_pose

    def detach(self) -> None:
        """Detach this Gizmo from its current target."""
        self._release_native_resources()
        with self._state_lock:
            self._interaction_owner = None
            self._pending_target_transform = None
            self._desired_target_transform = None
        self.target = None
        self._target_type = ""

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
        if self._ik_controller is not None:
            self._ik_controller.enabled = self._is_visible
        if self._gizmo is not None:
            self._gizmo.set_visible(self._is_visible)

    def is_visible(self) -> bool:
        """Return whether this Gizmo should be visible."""
        if self._ik_controller is not None:
            return bool(self._ik_controller.enabled)
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

    def _release_native_resources(self) -> None:
        """Release DexSim Gizmo, proxy, and native IK resources."""
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
            try:
                self._ik_controller.target_gizmo.target_node.detach_parent()
            except (AttributeError, TypeError, RuntimeError):
                pass

        self._remove_proxy_cube()

        if gizmo is not None:
            remove_gizmo = getattr(self._env, "remove_gizmo", None)
            if callable(remove_gizmo):
                try:
                    remove_gizmo(gizmo)
                except (AttributeError, TypeError, RuntimeError) as error:
                    logger.log_warning(
                        f"Failed to remove Gizmo from DexSim environment: {error}"
                    )

        self._gizmo = None
        self._proxy_cube = None
        self._ik_controller = None
        self._ik_solver = None
        self._ik_model = None
        self._robot_adapter = None

    def destroy(self) -> None:
        """Release native resources and target references."""
        self._release_native_resources()
        with self._state_lock:
            self._interaction_owner = None
            self._pending_target_transform = None
            self._desired_target_transform = None
        self.target = None
        self._target_type = ""
