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

"""Reusable TCP marker visualization and calibration helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from embodichain.lab.sim.cfg import MarkerCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "TCPMarkerCalibrator",
    "adjust_tcp_transform",
    "relative_transform",
    "save_solver_tcp_overrides",
    "solver_tcp_overrides",
]


def _as_transform(transform: object, *, name: str) -> np.ndarray:
    """Convert and validate one homogeneous transformation matrix."""
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} must be a homogeneous transformation matrix.")
    return matrix.copy()


def _axis_rotation(axis: str, angle_degrees: float) -> np.ndarray:
    """Return a 3x3 active rotation about an EE-frame axis."""
    angle = math.radians(float(angle_degrees))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotations = {
        "x": np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]),
        "y": np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]),
        "z": np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]),
    }
    try:
        return rotations[axis.lower()]
    except KeyError as error:
        raise ValueError("rotation_axis must be one of 'x', 'y', or 'z'.") from error


def adjust_tcp_transform(
    tcp_transform: object,
    *,
    translation: Sequence[float] = (0.0, 0.0, 0.0),
    rotation_axis: str | None = None,
    rotation_degrees: float = 0.0,
) -> np.ndarray:
    """Apply incremental translation and rotation in the EE frame.

    Args:
        tcp_transform: Current ``T_ee_tcp`` homogeneous transform.
        translation: XYZ translation increment expressed in the EE frame.
        rotation_axis: EE-frame axis about which to rotate TCP orientation.
        rotation_degrees: Rotation increment in degrees.

    Returns:
        A new adjusted ``T_ee_tcp`` matrix.
    """
    result = _as_transform(tcp_transform, name="tcp_transform")
    translation_array = np.asarray(translation, dtype=float)
    if translation_array.shape != (3,) or not np.isfinite(translation_array).all():
        raise ValueError("translation must contain three finite XYZ values.")
    result[:3, 3] += translation_array

    if rotation_axis is None:
        if not math.isclose(float(rotation_degrees), 0.0):
            raise ValueError("rotation_axis is required for a non-zero rotation.")
    else:
        result[:3, :3] = (
            _axis_rotation(rotation_axis, rotation_degrees) @ result[:3, :3]
        )
    return result


def relative_transform(parent_pose: object, child_pose: object) -> np.ndarray:
    """Compute ``T_parent_child`` from two poses in the same reference frame."""
    parent = _as_transform(parent_pose, name="parent_pose")
    child = _as_transform(child_pose, name="child_pose")
    return np.linalg.inv(parent) @ child


def solver_tcp_overrides(
    tcp_by_control_part: Mapping[str, object],
) -> dict[str, dict[str, dict[str, list[list[float]]]]]:
    """Build a config fragment directly consumable by ``RobotCfg.from_dict``."""
    return {
        "solver_cfg": {
            control_part: {
                "tcp": _as_transform(tcp, name=f"tcp[{control_part!r}]").tolist()
            }
            for control_part, tcp in tcp_by_control_part.items()
        }
    }


def save_solver_tcp_overrides(
    path: str | Path,
    tcp_by_control_part: Mapping[str, object],
) -> Path:
    """Save calibrated TCP matrices as a robot-config JSON fragment."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(solver_tcp_overrides(tcp_by_control_part), indent=4) + "\n",
        encoding="utf-8",
    )
    return output_path


class TCPMarkerCalibrator:
    """Visualize and adjust an end-link-relative TCP transform.

    The EE and TCP markers are expressed in the selected arena's local frame.
    The value exposed by :attr:`tcp_transform` is always ``T_ee_tcp``, which is
    the matrix expected by ``SolverCfg.tcp``.

    Args:
        sim: Simulation manager used to create axis markers.
        robot: Robot containing the end link.
        control_part: Solver/control-part name, such as ``"left_arm"``.
        end_link_name: URDF end link to which the TCP is relative.
        tcp_transform: Initial ``T_ee_tcp`` transform.
        arena_index: Arena to visualize. Defaults to zero.
        marker_prefix: Unique marker-name prefix.
        ee_axis_len: Display length of the EE marker in meters.
        tcp_axis_len: Display length of the TCP marker in meters.
    """

    def __init__(
        self,
        sim: SimulationManager,
        robot: Robot,
        *,
        control_part: str,
        end_link_name: str,
        tcp_transform: object,
        arena_index: int = 0,
        marker_prefix: str | None = None,
        ee_axis_len: float = 0.06,
        tcp_axis_len: float = 0.12,
    ) -> None:
        if arena_index < 0 or arena_index >= robot.num_instances:
            raise ValueError(
                f"arena_index must be in [0, {robot.num_instances}), "
                f"got {arena_index}."
            )
        if end_link_name not in robot.link_names:
            raise ValueError(
                f"Unknown end link {end_link_name!r}; available links: "
                f"{robot.link_names}."
            )

        self.sim = sim
        self.robot = robot
        self.control_part = control_part
        self.end_link_name = end_link_name
        self.arena_index = arena_index
        self.marker_prefix = marker_prefix or f"tcp_calibration_{control_part}"
        self.ee_axis_len = float(ee_axis_len)
        self.tcp_axis_len = float(tcp_axis_len)
        self._initial_tcp = _as_transform(tcp_transform, name="tcp_transform")
        self._tcp = self._initial_tcp.copy()
        self._ee_marker: list[object] | None = None
        self._tcp_marker: list[object] | None = None

    @property
    def tcp_transform(self) -> np.ndarray:
        """Return a copy of the current ``T_ee_tcp`` transform."""
        return self._tcp.copy()

    def set_tcp_transform(self, transform: object) -> None:
        """Replace ``T_ee_tcp`` and refresh existing markers."""
        self._tcp = _as_transform(transform, name="tcp_transform")
        self.update()

    def translate(self, axis: str, distance: float) -> None:
        """Translate the TCP along an EE-frame axis, in meters."""
        axis_indices = {"x": 0, "y": 1, "z": 2}
        try:
            axis_index = axis_indices[axis.lower()]
        except KeyError as error:
            raise ValueError("axis must be one of 'x', 'y', or 'z'.") from error
        delta = np.zeros(3)
        delta[axis_index] = float(distance)
        self._tcp = adjust_tcp_transform(self._tcp, translation=delta)
        self.update()

    def rotate(self, axis: str, angle_degrees: float) -> None:
        """Rotate TCP orientation about an EE-frame axis, in degrees."""
        self._tcp = adjust_tcp_transform(
            self._tcp,
            rotation_axis=axis,
            rotation_degrees=angle_degrees,
        )
        self.update()

    def set_translation(self, xyz: Sequence[float]) -> None:
        """Set the TCP translation component in the EE frame."""
        values = np.asarray(xyz, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("xyz must contain three finite values.")
        self._tcp[:3, 3] = values
        self.update()

    def reset(self) -> None:
        """Restore the transform supplied when the calibrator was created."""
        self._tcp = self._initial_tcp.copy()
        self.update()

    def get_ee_pose(self) -> np.ndarray:
        """Return the current end-link pose in the local arena frame."""
        pose = self.robot.get_link_pose(
            link_name=self.end_link_name,
            env_ids=[self.arena_index],
            to_matrix=True,
        )[0]
        if hasattr(pose, "detach"):
            pose = pose.detach().cpu().numpy()
        return _as_transform(pose, name="ee_pose")

    def get_tcp_pose(self) -> np.ndarray:
        """Return the visualized TCP pose in the local arena frame."""
        return self.get_ee_pose() @ self._tcp

    def draw(self) -> None:
        """Create EE and TCP axis markers, replacing existing markers."""
        self.close()
        ee_pose = self.get_ee_pose()
        tcp_pose = ee_pose @ self._tcp
        self._ee_marker = self.sim.draw_marker(
            MarkerCfg(
                name=f"{self.marker_prefix}_ee",
                marker_type="axis",
                axis_xpos=ee_pose,
                axis_size=0.003,
                axis_len=self.ee_axis_len,
                arena_index=self.arena_index,
            )
        )
        self._tcp_marker = self.sim.draw_marker(
            MarkerCfg(
                name=f"{self.marker_prefix}_tcp",
                marker_type="axis",
                axis_xpos=tcp_pose,
                axis_size=0.006,
                axis_len=self.tcp_axis_len,
                arena_index=self.arena_index,
            )
        )

    def update(self) -> None:
        """Refresh marker poses so they follow the current robot state."""
        if not self._ee_marker or not self._tcp_marker:
            return
        ee_pose = self.get_ee_pose()
        self._ee_marker[0].set_local_pose(ee_pose)
        self._tcp_marker[0].set_local_pose(ee_pose @ self._tcp)

    def _registered_marker_name(self, suffix: str) -> str:
        """Return the internal name used by ``SimulationManager``."""
        name = f"{self.marker_prefix}_{suffix}"
        return f"{name}_{self.arena_index}"

    def close(self) -> None:
        """Remove calibration markers from the scene."""
        if self._ee_marker:
            self.sim.remove_marker(self._registered_marker_name("ee"))
        if self._tcp_marker:
            self.sim.remove_marker(self._registered_marker_name("tcp"))
        self._ee_marker = None
        self._tcp_marker = None
