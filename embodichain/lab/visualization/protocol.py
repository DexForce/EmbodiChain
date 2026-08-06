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

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol

import numpy as np

from ._utils import to_numpy_array as _array

__all__ = [
    "SCHEMA_VERSION",
    "CameraImage",
    "CameraImageFrame",
    "CameraSpec",
    "DynamicMeshUpdate",
    "FrameOverlay",
    "GizmoCommand",
    "GizmoSpec",
    "GizmoState",
    "JointControlCommand",
    "JointControlProvider",
    "JointControlSpec",
    "JointControlState",
    "MeshGeometry",
    "PickCommand",
    "PointCloudOverlay",
    "SceneFrame",
    "SceneManifest",
    "SceneNode",
    "SceneOverlays",
    "TargetOverlay",
    "TrajectoryOverlay",
    "estimate_camera_image_frame_bytes",
    "estimate_frame_bytes",
    "estimate_manifest_bytes",
    "pose_to_position_wxyz",
]

SCHEMA_VERSION = 5
Color = tuple[int, int, int]


def _validate_color(color: Color) -> None:
    if len(color) != 3 or any(component < 0 or component > 255 for component in color):
        raise ValueError(
            f"Expected an RGB color with values in [0, 255], received {color}."
        )


def _rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert one 3x3 rotation matrix to a normalized wxyz quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ],
            dtype=np.float32,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = (
                np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            )
            quat = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ],
                dtype=np.float32,
            )
        elif index == 1:
            scale = (
                np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            )
            quat = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ],
                dtype=np.float32,
            )
        else:
            scale = (
                np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            )
            quat = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float32,
            )
    norm = np.linalg.norm(quat)
    if norm <= np.finfo(np.float32).eps:
        raise ValueError("Pose contains a degenerate rotation.")
    return quat / norm


def pose_to_position_wxyz(pose: object) -> tuple[np.ndarray, np.ndarray]:
    """Split pose arrays into positions and normalized wxyz quaternions.

    The accepted layouts are ``(..., 7)`` in EmbodiChain's
    ``(x, y, z, qw, qx, qy, qz)`` convention or homogeneous ``(..., 4, 4)``
    matrices. This is the single conversion boundary used by scene exporters.

    Args:
        pose: Pose or batch of poses.

    Returns:
        A pair of float32 arrays containing positions and wxyz quaternions.

    Raises:
        ValueError: If the shape is unsupported or a quaternion is degenerate.
    """
    array = _array(pose, np.float32)
    if array.ndim >= 1 and array.shape[-1] == 7:
        position = array[..., :3].copy()
        wxyz = array[..., 3:7].copy()
        norms = np.linalg.norm(wxyz, axis=-1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("Pose contains a degenerate quaternion.")
        return position, wxyz / norms

    if array.ndim >= 2 and array.shape[-2:] == (4, 4):
        position = array[..., :3, 3].copy()
        flat_rotations = array[..., :3, :3].reshape((-1, 3, 3))
        wxyz = np.stack(
            [_rotation_matrix_to_wxyz(rotation) for rotation in flat_rotations], axis=0
        ).reshape(array.shape[:-2] + (4,))
        return position, wxyz

    raise ValueError(
        f"Expected pose shape (..., 7) or (..., 4, 4), received {array.shape}."
    )


@dataclass(frozen=True)
class MeshGeometry:
    """Backend-neutral triangle mesh stored in local coordinates."""

    geometry_id: str
    vertices: np.ndarray
    faces: np.ndarray
    color: Color = (90, 200, 255)

    def __post_init__(self) -> None:
        vertices = _array(self.vertices, np.float32)
        faces = _array(self.faces, np.uint32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                f"vertices must have shape (N, 3), received {vertices.shape}."
            )
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (N, 3), received {faces.shape}.")
        _validate_color(self.color)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)


@dataclass(frozen=True)
class CameraSpec:
    """Static pinhole-camera parameters for one environment instance."""

    camera_id: str
    sensor_uid: str
    env_id: int
    path: str
    fov_y: float
    aspect: float
    near: float
    far: float

    def __post_init__(self) -> None:
        if self.env_id < 0:
            raise ValueError("Camera env_id must be non-negative.")
        if not 0.0 < self.fov_y < np.pi:
            raise ValueError("Camera vertical field of view must be in (0, pi).")
        if self.aspect <= 0.0:
            raise ValueError("Camera aspect ratio must be greater than zero.")
        if self.near <= 0.0 or self.far <= self.near:
            raise ValueError("Camera clipping planes must satisfy 0 < near < far.")


@dataclass(frozen=True)
class CameraImage:
    """One detached RGB image associated with a manifest camera."""

    camera_id: str
    image: np.ndarray

    def __post_init__(self) -> None:
        image = _array(self.image, np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Camera RGB image must have shape (H, W, 3), received {image.shape}."
            )
        object.__setattr__(self, "image", image)


@dataclass(frozen=True)
class CameraImageFrame:
    """Low-frequency RGB images captured at one simulation timestamp."""

    run_id: str
    scene_revision: int
    sequence: int
    sim_step: int
    sim_time: float
    images: tuple[CameraImage, ...]
    wall_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        camera_ids = [image.camera_id for image in self.images]
        if len(camera_ids) != len(set(camera_ids)):
            raise ValueError("CameraImageFrame contains duplicate camera IDs.")


@dataclass(frozen=True)
class DynamicMeshUpdate:
    """Detached vertex positions for one deformable scene node."""

    node_id: str
    vertices: np.ndarray

    def __post_init__(self) -> None:
        vertices = _array(self.vertices, np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(
                "Dynamic mesh vertices must have shape (N, 3), "
                f"received {vertices.shape}."
            )
        object.__setattr__(self, "vertices", vertices)


@dataclass(frozen=True)
class GizmoSpec:
    """Static description of one simulation Gizmo exposed by a backend."""

    gizmo_id: str
    target_uid: str
    target_type: str
    control_part: str | None
    env_id: int
    path: str
    scale: float = 0.2
    line_width: float = 2.5
    visible: bool = True

    def __post_init__(self) -> None:
        if not self.gizmo_id:
            raise ValueError("Gizmo ID must not be empty.")
        if not self.target_uid:
            raise ValueError("Gizmo target UID must not be empty.")
        if self.target_type not in {"rigid_object", "robot", "camera"}:
            raise ValueError(
                "Gizmo target_type must be 'rigid_object', 'robot', or 'camera'."
            )
        if self.env_id < 0:
            raise ValueError("Gizmo env_id must be non-negative.")
        if not self.path.startswith("/"):
            raise ValueError("Gizmo path must be an absolute scene path.")
        if self.scale <= 0.0:
            raise ValueError("Gizmo scale must be greater than zero.")
        if self.line_width <= 0.0:
            raise ValueError("Gizmo line_width must be greater than zero.")


@dataclass(frozen=True)
class GizmoState:
    """Authoritative world pose and visibility of one simulation Gizmo."""

    gizmo_id: str
    position: np.ndarray
    wxyz: np.ndarray
    visible: bool = True

    def __post_init__(self) -> None:
        position, wxyz = pose_to_position_wxyz(
            np.concatenate(
                (_array(self.position, np.float32), _array(self.wxyz, np.float32))
            )
        )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "wxyz", wxyz)


@dataclass(frozen=True)
class GizmoCommand:
    """Immutable browser drag command consumed on the simulation thread."""

    run_id: str
    scene_revision: int
    sequence: int
    gizmo_id: str
    phase: Literal["start", "update", "end"]
    client_id: str
    position: np.ndarray
    wxyz: np.ndarray
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Gizmo command run_id must not be empty.")
        if self.scene_revision < 0:
            raise ValueError("Gizmo command scene_revision must be non-negative.")
        if self.sequence < 0:
            raise ValueError("Gizmo command sequence must be non-negative.")
        if not self.gizmo_id:
            raise ValueError("Gizmo command gizmo_id must not be empty.")
        if self.phase not in {"start", "update", "end"}:
            raise ValueError("Gizmo command phase must be 'start', 'update', or 'end'.")
        if not self.client_id:
            raise ValueError("Gizmo command client_id must not be empty.")
        position, wxyz = pose_to_position_wxyz(
            np.concatenate(
                (_array(self.position, np.float32), _array(self.wxyz, np.float32))
            )
        )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "wxyz", wxyz)


@dataclass(frozen=True)
class PickCommand:
    """Immutable browser click-pick command consumed on the simulation thread.

    A non-empty ``node_id`` requests a Gizmo on the clicked scene node; a
    ``None`` ``node_id`` (clicking empty space) clears the picker-owned Gizmo.
    """

    run_id: str
    scene_revision: int
    client_id: str
    node_id: str | None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Pick command run_id must not be empty.")
        if self.scene_revision < 0:
            raise ValueError("Pick command scene_revision must be non-negative.")
        if not self.client_id:
            raise ValueError("Pick command client_id must not be empty.")
        if self.node_id is not None and not self.node_id:
            raise ValueError("Pick command node_id must be None or non-empty.")


@dataclass(frozen=True)
class JointControlSpec:
    """Static description of one scalar articulation joint control.

    Values use simulation units: radians for rotational joints and meters for
    prismatic joints. Controls with two finite limits can be rendered as a
    slider; controls with one or both limits missing use a numeric input.
    """

    control_id: str
    articulation_uid: str
    env_id: int
    joint_id: int
    joint_name: str
    joint_type: Literal["revolute", "continuous", "prismatic"]
    lower: float | None
    upper: float | None
    step: float
    initial_value: float

    def __post_init__(self) -> None:
        if not self.control_id:
            raise ValueError("Joint control ID must not be empty.")
        if not self.articulation_uid:
            raise ValueError("Joint control articulation_uid must not be empty.")
        if self.env_id < 0:
            raise ValueError("Joint control env_id must be non-negative.")
        if self.joint_id < 0:
            raise ValueError("Joint control joint_id must be non-negative.")
        if not self.joint_name:
            raise ValueError("Joint control joint_name must not be empty.")
        if self.joint_type not in {"revolute", "continuous", "prismatic"}:
            raise ValueError(
                "Joint control joint_type must be 'revolute', 'continuous', "
                "or 'prismatic'."
            )
        if self.lower is not None and not np.isfinite(self.lower):
            raise ValueError("Joint control lower limit must be finite when set.")
        if self.upper is not None and not np.isfinite(self.upper):
            raise ValueError("Joint control upper limit must be finite when set.")
        if self.lower is not None and self.initial_value < self.lower:
            raise ValueError("Joint control initial value is below its lower limit.")
        if self.upper is not None and self.initial_value > self.upper:
            raise ValueError("Joint control initial value is above its upper limit.")
        if self.lower is not None and self.upper is not None:
            if self.lower >= self.upper:
                raise ValueError("Joint control lower limit must be less than upper.")
        if not np.isfinite(self.step) or self.step <= 0.0:
            raise ValueError("Joint control step must be finite and greater than zero.")
        if not np.isfinite(self.initial_value):
            raise ValueError("Joint control initial value must be finite.")


@dataclass(frozen=True)
class JointControlState:
    """Authoritative value and command acknowledgement for one joint control."""

    control_id: str
    value: float
    applied_sequence: int = 0

    def __post_init__(self) -> None:
        if not self.control_id:
            raise ValueError("Joint control state ID must not be empty.")
        if not np.isfinite(self.value):
            raise ValueError("Joint control state value must be finite.")
        if self.applied_sequence < 0:
            raise ValueError(
                "Joint control state applied_sequence must be non-negative."
            )


@dataclass(frozen=True)
class JointControlCommand:
    """Immutable browser joint command consumed on the simulation thread."""

    run_id: str
    scene_revision: int
    sequence: int
    client_id: str
    control_id: str
    value: float
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("Joint control command run_id must not be empty.")
        if self.scene_revision < 0:
            raise ValueError(
                "Joint control command scene_revision must be non-negative."
            )
        if self.sequence < 0:
            raise ValueError("Joint control command sequence must be non-negative.")
        if not self.client_id:
            raise ValueError("Joint control command client_id must not be empty.")
        if not self.control_id:
            raise ValueError("Joint control command control_id must not be empty.")
        if not np.isfinite(self.value):
            raise ValueError("Joint control command value must be finite.")


class JointControlProvider(Protocol):
    """Simulation-thread source of optional articulation joint controls."""

    def joint_control_specs(self) -> tuple[JointControlSpec, ...]:
        """Return the static controls to include in the next manifest."""
        ...

    def joint_control_states(self) -> tuple[JointControlState, ...]:
        """Return current values ordered independently of backend state."""
        ...


@dataclass(frozen=True)
class SceneNode:
    """One mesh-bearing logical node in a scene manifest."""

    node_id: str
    path: str
    parent_id: str | None
    env_id: int
    kind: str
    geometry_id: str
    dynamic_geometry: bool = False
    visible: bool = True


@dataclass(frozen=True)
class SceneManifest:
    """Static scene topology and geometry for one scene revision."""

    run_id: str
    scene_revision: int
    nodes: tuple[SceneNode, ...]
    geometries: tuple[MeshGeometry, ...]
    cameras: tuple[CameraSpec, ...] = field(default_factory=tuple)
    gizmos: tuple[GizmoSpec, ...] = field(default_factory=tuple)
    joint_controls: tuple[JointControlSpec, ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION
    up_direction: str = "+z"
    length_unit: str = "meter"

    def __post_init__(self) -> None:
        if self.scene_revision < 0:
            raise ValueError("scene_revision must be non-negative.")
        geometry_ids = {geometry.geometry_id for geometry in self.geometries}
        if len(geometry_ids) != len(self.geometries):
            raise ValueError("SceneManifest contains duplicate geometry IDs.")
        node_ids = {node.node_id for node in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("SceneManifest contains duplicate node IDs.")
        camera_ids = {camera.camera_id for camera in self.cameras}
        if len(camera_ids) != len(self.cameras):
            raise ValueError("SceneManifest contains duplicate camera IDs.")
        gizmo_ids = {gizmo.gizmo_id for gizmo in self.gizmos}
        if len(gizmo_ids) != len(self.gizmos):
            raise ValueError("SceneManifest contains duplicate Gizmo IDs.")
        joint_control_ids = {control.control_id for control in self.joint_controls}
        if len(joint_control_ids) != len(self.joint_controls):
            raise ValueError("SceneManifest contains duplicate joint control IDs.")
        missing = {node.geometry_id for node in self.nodes} - geometry_ids
        if missing:
            raise ValueError(
                f"Scene nodes reference missing geometry IDs: {sorted(missing)}."
            )


@dataclass(frozen=True)
class FrameOverlay:
    """Coordinate frame overlay."""

    overlay_id: str
    position: np.ndarray
    wxyz: np.ndarray
    axes_length: float = 0.15
    axes_radius: float = 0.006
    visible: bool = True

    def __post_init__(self) -> None:
        position, wxyz = pose_to_position_wxyz(
            np.concatenate(
                (_array(self.position, np.float32), _array(self.wxyz, np.float32))
            )
        )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "wxyz", wxyz)


@dataclass(frozen=True)
class TargetOverlay:
    """Target pose overlay rendered as a coordinate frame."""

    overlay_id: str
    position: np.ndarray
    wxyz: np.ndarray
    axes_length: float = 0.2
    visible: bool = True

    def __post_init__(self) -> None:
        position, wxyz = pose_to_position_wxyz(
            np.concatenate(
                (_array(self.position, np.float32), _array(self.wxyz, np.float32))
            )
        )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "wxyz", wxyz)


@dataclass(frozen=True)
class TrajectoryOverlay:
    """Polyline trajectory overlay."""

    overlay_id: str
    points: np.ndarray
    color: Color = (255, 170, 30)
    line_width: float = 3.0
    visible: bool = True

    def __post_init__(self) -> None:
        points = _array(self.points, np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Trajectory points must have shape (N, 3), received {points.shape}."
            )
        _validate_color(self.color)
        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class PointCloudOverlay:
    """Point cloud overlay with per-cloud or per-point RGB colors."""

    overlay_id: str
    points: np.ndarray
    colors: np.ndarray | Color = (90, 200, 255)
    point_size: float = 0.01
    visible: bool = True

    def __post_init__(self) -> None:
        points = _array(self.points, np.float32)
        colors = _array(self.colors, np.uint8)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"Point cloud points must have shape (N, 3), received {points.shape}."
            )
        if colors.shape not in {(3,), points.shape}:
            raise ValueError(
                "Point cloud colors must have shape (3,) or match the points shape; "
                f"received {colors.shape}."
            )
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors", colors)


@dataclass(frozen=True)
class SceneOverlays:
    """All optional overlays attached to a dynamic frame."""

    frames: tuple[FrameOverlay, ...] = field(default_factory=tuple)
    trajectories: tuple[TrajectoryOverlay, ...] = field(default_factory=tuple)
    targets: tuple[TargetOverlay, ...] = field(default_factory=tuple)
    point_clouds: tuple[PointCloudOverlay, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SceneFrame:
    """Dynamic poses and overlays for one simulation sample."""

    run_id: str
    scene_revision: int
    sequence: int
    sim_step: int
    sim_time: float
    node_ids: tuple[str, ...]
    positions: np.ndarray
    wxyz: np.ndarray
    visible: np.ndarray
    camera_ids: tuple[str, ...] = field(default_factory=tuple)
    camera_positions: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    camera_wxyz: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )
    dynamic_meshes: tuple[DynamicMeshUpdate, ...] = field(default_factory=tuple)
    gizmos: tuple[GizmoState, ...] = field(default_factory=tuple)
    joint_controls: tuple[JointControlState, ...] = field(default_factory=tuple)
    overlays: SceneOverlays = field(default_factory=SceneOverlays)
    wall_time: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        positions = _array(self.positions, np.float32)
        wxyz = _array(self.wxyz, np.float32)
        visible = _array(self.visible, np.bool_)
        camera_positions = _array(self.camera_positions, np.float32)
        camera_wxyz = _array(self.camera_wxyz, np.float32)
        dynamic_node_ids = [mesh.node_id for mesh in self.dynamic_meshes]
        if len(dynamic_node_ids) != len(set(dynamic_node_ids)):
            raise ValueError("SceneFrame contains duplicate dynamic mesh node IDs.")
        gizmo_ids = [gizmo.gizmo_id for gizmo in self.gizmos]
        if len(gizmo_ids) != len(set(gizmo_ids)):
            raise ValueError("SceneFrame contains duplicate Gizmo IDs.")
        joint_control_ids = [control.control_id for control in self.joint_controls]
        if len(joint_control_ids) != len(set(joint_control_ids)):
            raise ValueError("SceneFrame contains duplicate joint control IDs.")
        node_count = len(self.node_ids)
        if positions.shape != (node_count, 3):
            raise ValueError(
                f"positions must have shape ({node_count}, 3), received {positions.shape}."
            )
        if wxyz.shape != (node_count, 4):
            raise ValueError(
                f"wxyz must have shape ({node_count}, 4), received {wxyz.shape}."
            )
        if visible.shape != (node_count,):
            raise ValueError(
                f"visible must have shape ({node_count},), received {visible.shape}."
            )
        norms = np.linalg.norm(wxyz, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise ValueError("Frame contains a degenerate quaternion.")
        camera_count = len(self.camera_ids)
        if camera_positions.shape != (camera_count, 3):
            raise ValueError(
                "camera_positions must have shape "
                f"({camera_count}, 3), received {camera_positions.shape}."
            )
        if camera_wxyz.shape != (camera_count, 4):
            raise ValueError(
                "camera_wxyz must have shape "
                f"({camera_count}, 4), received {camera_wxyz.shape}."
            )
        camera_norms = np.linalg.norm(camera_wxyz, axis=1, keepdims=True)
        if np.any(camera_norms <= np.finfo(np.float32).eps):
            raise ValueError("Frame contains a degenerate camera quaternion.")
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "wxyz", wxyz / norms)
        object.__setattr__(self, "visible", visible)
        object.__setattr__(self, "camera_positions", camera_positions)
        object.__setattr__(self, "camera_wxyz", camera_wxyz / camera_norms)


def estimate_manifest_bytes(manifest: SceneManifest) -> int:
    """Estimate binary geometry bytes retained by a manifest."""
    return sum(
        geometry.vertices.nbytes + geometry.faces.nbytes
        for geometry in manifest.geometries
    )


def estimate_camera_image_frame_bytes(frame: CameraImageFrame) -> int:
    """Estimate NumPy image bytes retained by a camera image frame."""
    return sum(image.image.nbytes for image in frame.images)


def estimate_frame_bytes(frame: SceneFrame) -> int:
    """Estimate NumPy payload bytes retained by a frame."""
    total = (
        frame.positions.nbytes
        + frame.wxyz.nbytes
        + frame.visible.nbytes
        + frame.camera_positions.nbytes
        + frame.camera_wxyz.nbytes
    )
    total += sum(mesh.vertices.nbytes for mesh in frame.dynamic_meshes)
    for overlay in frame.overlays.trajectories:
        total += overlay.points.nbytes
    for overlay in frame.overlays.point_clouds:
        total += overlay.points.nbytes + overlay.colors.nbytes
    for overlay in (*frame.overlays.frames, *frame.overlays.targets):
        total += overlay.position.nbytes + overlay.wxyz.nbytes
    for gizmo in frame.gizmos:
        total += gizmo.position.nbytes + gizmo.wxyz.nbytes
    return total
