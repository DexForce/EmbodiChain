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

"""Viser Gizmo control and the EmbodiChain-to-DexSim robot IK adapter."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Literal

import dexsim
import numpy as np
import torch
from dexsim.kit.ik.pose import (
    local_pose_from_world,
    pose_from_position_rotation,
    rotation_matrix_to_quat_xyzw,
)
from dexsim.types import InputKey

from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.objects.rigid_object import RigidObject
from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.sensors import Camera
from embodichain.utils import configclass, logger

if TYPE_CHECKING:
    from dexsim.engine import GizmoController
    from dexsim.kit.ik import IKGizmoController, NewtonChainIK
    from embodichain.lab.sim.solvers import BaseSolver

__all__ = ["Gizmo", "GizmoCfg", "create_robot_ik_gizmo_controller"]


@configclass
class GizmoCfg:
    """Configure Viser Gizmo appearance and robot IK behavior."""

    axis_length_x: float = 0.2
    """Length of the X-axis arrow."""

    axis_length_y: float = 0.2
    """Length of the Y-axis arrow."""

    axis_length_z: float = 0.2
    """Length of the Z-axis arrow."""

    axis_size: float = 0.01
    """Thickness of the Viser axis lines."""

    rings_radius: float = 0.15
    """Radius of the rotation rings."""

    rings_size: float = 0.01
    """Thickness of the rotation rings."""

    ik_solver: Literal["dexsim", "embodichain"] = "dexsim"
    """Use native Newton IK, or the control part's configured EmbodiChain solver.

    The native window always uses DexSim's ``IKGizmoController``. Selecting
    ``embodichain`` reuses ``robot.get_solver(control_part)`` (for example,
    PinkSolver), including that solver's convergence and joint-limit settings.
    """

    ik_root_link_name: str | None = None
    """Robot IK chain root link, or the configured solver root when omitted."""

    ik_end_link_name: str | None = None
    """Robot IK chain end link, or the configured solver end when omitted."""

    ik_tcp_pose: torch.Tensor | np.ndarray | list[list[float]] | None = None
    """End-link-to-TCP transform."""

    ik_iterations: int = 24
    """Number of Newton IK iterations per changed target."""

    ik_device: str | None = None
    """Warp device for Newton IK, or the robot device when omitted."""

    ik_gizmo_scale: float = 1.5
    """Isotropic scale of a native DexSim robot IK target."""

    ik_toggle_key: InputKey = InputKey.SCANCODE_I
    """Native-window key used to toggle a DexSim robot IK target."""


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
        self.root_link_name: str | None = None
        self.model_root_inverse = np.eye(4, dtype=np.float32)

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
        """Map the IK model frame into the selected instance's arena frame."""
        if self.root_link_name is not None:
            return self.get_link_pose(self.root_link_name) @ self.model_root_inverse
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


def _resolve_control_part(robot: Robot, control_part: str | None) -> str:
    part_names = list(robot.control_parts or {})
    if not part_names:
        raise ValueError("Robot has no control parts defined.")
    if control_part is None:
        return part_names[0]
    if control_part not in part_names:
        raise ValueError(
            f"Control part {control_part!r} was not found; available parts are "
            f"{part_names}."
        )
    return control_part


def _resolve_robot_ik_chain(
    robot: Robot,
    control_part: str,
    cfg: GizmoCfg,
) -> tuple[str, str, np.ndarray]:
    solver = (
        robot.get_solver(control_part) if robot.cfg.solver_cfg is not None else None
    )
    root_link = cfg.ik_root_link_name or getattr(solver, "root_link_name", None)
    end_link = cfg.ik_end_link_name or getattr(solver, "end_link_name", None)
    if not root_link or not end_link:
        raise ValueError(
            "Robot Gizmo needs an IK chain. Set GizmoCfg.ik_root_link_name and "
            "GizmoCfg.ik_end_link_name, or configure a solver for the control part."
        )

    tcp_pose = cfg.ik_tcp_pose
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


class _EmbodiChainIK:
    """Adapt BaseSolver to the solver contract consumed by DexSim and Viser."""

    def __init__(
        self, adapter: _RobotGizmoAdapter, solver: BaseSolver, tcp_pose: np.ndarray
    ) -> None:
        self.solver = solver
        self.joint_names = list(solver.joint_names or adapter.joint_names)
        if set(self.joint_names) != set(adapter.joint_names):
            raise ValueError(
                "IK solver joints must match the control part's active joints."
            )
        self._tcp_inverse = np.linalg.inv(tcp_pose)
        self._solution: dict[str, float] = {}
        pose = (
            local_pose_from_world(
                adapter.get_world_pose(), adapter.get_link_pose(solver.end_link_name)
            )
            @ tcp_pose
        )
        self._target_state: dict[str, np.ndarray] = {}
        self.set_target_pose(pose[:3, 3], rotation_matrix_to_quat_xyzw(pose[:3, :3]))
        self.reset_target_state_changed()

    def target_state(self) -> dict[str, np.ndarray]:
        return self._target_state

    def set_target_pose(self, position: np.ndarray, rotation: np.ndarray) -> None:
        self._target_state.update(
            position=np.array(position), rotation=np.array(rotation)
        )

    def target_state_changed(self) -> bool:
        return any(
            not np.allclose(value, self._snapshot[key], atol=1e-5)
            for key, value in self._target_state.items()
        )

    def reset_target_state_changed(self) -> None:
        self._snapshot = {
            key: value.copy() for key, value in self._target_state.items()
        }

    def solve(
        self,
        joint_names: list[str],
        current_qpos: np.ndarray,
        *,
        iterations: int | None = None,
    ) -> None:
        # EmbodiChain solvers retain their own iteration/convergence configuration.
        del iterations
        self._solution.clear()
        seed = dict(zip(joint_names, current_qpos))
        target = pose_from_position_rotation(**self._target_state)
        # The gizmo TCP may differ from the shared solver's TCP. Adapt the
        # target without mutating the solver used by the rest of the robot.
        target = target @ self._tcp_inverse @ self.solver.get_tcp()
        success, qpos = self.solver.get_ik(
            torch.as_tensor(
                target, dtype=torch.float32, device=self.solver.device
            ).unsqueeze(0),
            qpos_seed=torch.tensor(
                [[seed[name] for name in self.joint_names]],
                dtype=torch.float32,
                device=self.solver.device,
            ),
            return_all_solutions=False,
        )
        if bool(success.all()) and bool(torch.isfinite(qpos).all()):
            values = qpos.detach().cpu().numpy().reshape(len(self.joint_names))
            self._solution = dict(zip(self.joint_names, values))

    def qpos_for_joint_names(
        self, joint_names: list[str], fallback_qpos: np.ndarray
    ) -> np.ndarray:
        return np.array(
            [
                self._solution.get(name, value)
                for name, value in zip(joint_names, fallback_qpos)
            ],
            dtype=np.float32,
        )


def _build_robot_ik(
    robot: Robot,
    control_part: str,
    cfg: GizmoCfg,
) -> tuple[_RobotGizmoAdapter, NewtonChainIK | _EmbodiChainIK, str, np.ndarray]:
    if robot.num_instances != 1:
        raise RuntimeError(
            "Robot Gizmo supports exactly one environment, "
            f"but the robot has {robot.num_instances} instances."
        )
    if cfg.ik_solver not in {"dexsim", "embodichain"}:
        raise ValueError("ik_solver must be 'dexsim' or 'embodichain'.")
    if cfg.ik_solver == "dexsim" and cfg.ik_iterations <= 0:
        raise ValueError("ik_iterations must be greater than zero.")

    root_link, end_link, tcp_pose = _resolve_robot_ik_chain(
        robot,
        control_part,
        cfg,
    )
    adapter = _RobotGizmoAdapter(robot, control_part)
    adapter.root_link_name = root_link
    if cfg.ik_solver == "embodichain":
        solver = (
            robot.get_solver(control_part) if robot.cfg.solver_cfg is not None else None
        )
        if solver is None:
            raise ValueError(
                f"Control part {control_part!r} needs a configured EmbodiChain solver."
            )
        if (root_link, end_link) != (solver.root_link_name, solver.end_link_name):
            raise ValueError(
                "Gizmo IK links must match the configured EmbodiChain solver."
            )
        return adapter, _EmbodiChainIK(adapter, solver, tcp_pose), end_link, tcp_pose

    import warp as wp
    from dexsim.kit.ik import NewtonChainIK, build_newton_model_from_urdf

    with wp.ScopedDevice(cfg.ik_device or str(robot.device)):
        model = build_newton_model_from_urdf(robot.cfg.fpath, hide_visuals=True)
        solver = NewtonChainIK(
            model,
            start_link=root_link,
            end_link=end_link,
            iterations=cfg.ik_iterations,
            tcp_pose=tcp_pose,
        )

    # Newton preserves the start link's URDF rest transform in its reduced
    # model. Follow the live chain root (including upstream joints) while
    # retaining that model frame, rather than assuming the robot root is it.
    root_index = list(solver.model.body_label).index(solver.info.start_link)
    root_pose = solver.state.body_q.numpy()[root_index]
    adapter.model_root_inverse = np.linalg.inv(
        pose_from_position_rotation(root_pose[:3], root_pose[3:])
    )
    solver.set_qpos_from_joint_names(
        adapter.get_actived_joint_names(),
        adapter.get_current_qpos(),
    )
    solver.sync_target_state_from_link(adapter, adapter.get_world_pose())
    logger.log_info(
        f"Robot Gizmo uses DexSim Newton IK for control part {control_part!r} "
        f"({root_link} -> {end_link})."
    )
    return adapter, solver, end_link, tcp_pose


def create_robot_ik_gizmo_controller(
    robot: Robot,
    control_part: str = "arm",
    cfg: GizmoCfg | None = None,
    *,
    world: dexsim.World | None = None,
) -> tuple[IKGizmoController, GizmoController]:
    """Create DexSim's native IK controller for one EmbodiChain control part.

    The caller owns the returned objects and must call ``controller.update()``
    once per frame. Retain the input controller while the native window exists.

    Args:
        robot: Single-instance EmbodiChain robot.
        control_part: Robot control part driven by IK.
        cfg: IK chain, solver selection, and target appearance settings.
        world: DexSim world, or the current default world when omitted.

    Returns:
        ``(ik_controller, input_controller)`` owned by the caller.
    """
    from dexsim.engine import GizmoController
    from dexsim.kit.ik import IKApplyMode, IKGizmoController

    cfg = cfg or GizmoCfg()
    if not np.isfinite(cfg.ik_gizmo_scale) or cfg.ik_gizmo_scale <= 0:
        raise ValueError("ik_gizmo_scale must be positive and finite.")
    if world is None:
        world = dexsim.default_world()
    if world is None:
        raise RuntimeError("A DexSim world must exist before creating an IK Gizmo.")
    window = world.get_windows()
    if window is None:
        raise RuntimeError("A native DexSim window is required for an IK Gizmo.")

    control_part = _resolve_control_part(robot, control_part)
    adapter, solver, _, _ = _build_robot_ik(robot, control_part, cfg)
    input_controller = GizmoController()
    controller = IKGizmoController(
        world,
        adapter,
        solver,
        base_state={"pose": adapter.get_world_pose()},
        toggle_key=cfg.ik_toggle_key,
        follow_robot_base=True,
        apply_mode=IKApplyMode.DRIVE_TARGET,
        gizmo_scale=cfg.ik_gizmo_scale,
        name=f"{robot.uid}_{control_part}_ik",
    )
    window.add_input_control(input_controller)
    return controller, input_controller


class Gizmo:
    """Apply Viser Gizmo commands to one simulation target.

    Native-window entity manipulation is owned by DexSim. Use
    :meth:`SimulationManager.enable_entity_gizmo` for entity roots and
    :func:`create_robot_ik_gizmo_controller` for a native robot TCP target.

    .. attention::
        Viser Gizmo control supports exactly one simulation environment.

    Args:
        target: Rigid object, robot, or camera controlled from Viser.
        cfg: Viser appearance and robot IK configuration.
        control_part: Robot control part used for FK and IK.
    """

    def __init__(
        self,
        target: BatchEntity,
        cfg: GizmoCfg | None = None,
        control_part: str | None = None,
    ) -> None:
        if target.num_instances != 1:
            raise RuntimeError(
                "Viser Gizmo supports exactly one environment, "
                f"but the target has {target.num_instances} instances."
            )

        self.target: BatchEntity | None = target
        self.cfg = cfg or GizmoCfg()
        self._target_type = self._detect_target_type(target)
        self._control_part = control_part
        self._is_visible = True
        self._state_lock = threading.RLock()
        self._interaction_owner: str | None = None
        self._pending_target_transform: torch.Tensor | None = None
        self._desired_target_transform: torch.Tensor | None = None
        self._ik_solver: NewtonChainIK | _EmbodiChainIK | None = None
        self._robot_adapter: _RobotGizmoAdapter | None = None
        self._robot_end_link: str | None = None
        self._robot_tcp_pose: np.ndarray | None = None

        if self._target_type == "robot":
            self._control_part = _resolve_control_part(target, control_part)
            self._setup_robot_ik_solver()
            self._desired_target_transform = self._read_robot_pose()
        else:
            self._desired_target_transform = self._read_target_pose()

    @property
    def target_type(self) -> str:
        """Return ``rigid_object``, ``robot``, or ``camera``."""
        return self._target_type

    @property
    def control_part(self) -> str | None:
        """Return the robot control part, if applicable."""
        return self._control_part

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

    def _setup_robot_ik_solver(self) -> None:
        if self.target is None or not isinstance(self.target, Robot):
            raise RuntimeError("Robot Gizmo has no attached Robot.")
        if self._control_part is None:
            raise RuntimeError("Robot Gizmo control part is not configured.")
        adapter, solver, end_link, tcp_pose = _build_robot_ik(
            self.target,
            self._control_part,
            self.cfg,
        )
        self._robot_adapter = adapter
        self._ik_solver = solver
        self._robot_end_link = end_link
        self._robot_tcp_pose = tcp_pose

    def _read_robot_pose(self) -> torch.Tensor:
        if (
            self._robot_adapter is None
            or self._robot_end_link is None
            or self._robot_tcp_pose is None
        ):
            raise RuntimeError("Robot Gizmo IK is not configured.")
        link_pose = self._robot_adapter.get_link_pose(self._robot_end_link)
        return self._as_pose_matrix(
            link_pose @ self._robot_tcp_pose,
            self._target_device(),
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

    def _read_target_pose(self) -> torch.Tensor:
        if self.target is None:
            raise RuntimeError("Gizmo is detached.")
        if self._target_type == "robot":
            return self._read_robot_pose()
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
        """Acquire this Gizmo for one Viser drag source."""
        if not source_id:
            raise ValueError("source_id must not be empty.")
        with self._state_lock:
            if self._interaction_owner not in {None, source_id}:
                return False
            self._interaction_owner = source_id
            return True

    def request_local_pose(self, pose: object, *, source_id: str) -> bool:
        """Queue a local target pose for application by :meth:`update`."""
        matrix = self._as_pose_matrix(pose, self._target_device())
        with self._state_lock:
            if self._interaction_owner not in {None, source_id}:
                return False
            self._pending_target_transform = matrix
            self._desired_target_transform = matrix.detach().clone()
            return True

    def end_interaction(self, source_id: str) -> bool:
        """Release a Viser drag source's ownership of this Gizmo."""
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

    def update(self) -> None:
        """Apply the latest queued Viser target pose on the simulation thread."""
        with self._state_lock:
            pending = self._pending_target_transform
            self._pending_target_transform = None

        if pending is None or self.target is None:
            return
        if self._robot_adapter is None:
            self.target.set_local_pose(pending, env_ids=[0])
            return
        adapter, solver = self._robot_adapter, self._ik_solver
        base_local = local_pose_from_world(
            adapter.get_world_pose(), pending[0].detach().cpu().numpy()
        )
        joint_names = adapter.get_actived_joint_names()
        current_qpos = adapter.get_current_qpos()
        solver.set_target_pose(
            base_local[:3, 3], rotation_matrix_to_quat_xyzw(base_local[:3, :3])
        )
        solver.solve(joint_names, current_qpos, iterations=self.cfg.ik_iterations)
        adapter.set_target_qpos(solver.qpos_for_joint_names(joint_names, current_qpos))

    def toggle_visibility(self) -> bool:
        """Toggle Viser visibility and return the new state."""
        self._is_visible = not self._is_visible
        return self._is_visible

    def set_visible(self, visible: bool) -> None:
        """Set Viser Gizmo visibility."""
        self._is_visible = bool(visible)

    def is_visible(self) -> bool:
        """Return whether the Viser Gizmo should be visible."""
        return self._is_visible

    def destroy(self) -> None:
        """Release target and IK references."""
        with self._state_lock:
            self._interaction_owner = None
            self._pending_target_transform = None
            self._desired_target_transform = None
        self._ik_solver = None
        self._robot_adapter = None
        self.target = None
        self._target_type = ""
