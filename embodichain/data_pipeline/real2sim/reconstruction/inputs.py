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
"""Dataset-neutral inputs for image-grounded robot trajectory preparation.

Dataset file layouts and deployment addresses belong to callers. Observed joints
remain the existing RobotDemonstration type; refinement returns a separate array.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
from ..demo_skill import RobotDemonstration

__all__ = [
    "CameraObservation",
    "RobotModel",
    "ReconstructionInput",
    "RegisteredScene",
    "RefinedTrajectory",
]


@dataclass(frozen=True)
class CameraObservation:
    """RGB video and its explicit trajectory-sample to decoded-frame mapping.

    Depth is optional, in metres, at the first video frame. Auxiliary depth-only
    views may omit first_image; the primary view must supply it. The optical
    camera pose is expressed in the robot base frame. No timestamp or extrinsic
    is inferred.
    """

    video: Path
    first_image: Path | None
    sample_frames: np.ndarray
    intrinsics: np.ndarray | None = None
    camera_to_base: np.ndarray | None = None
    depth_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        ids = np.asarray(self.sample_frames)
        if (
            ids.ndim != 1
            or len(ids) == 0
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids[1:] < ids[:-1])
        ):
            raise ValueError(
                "sample_frames must be ordered nonnegative integer frame IDs"
            )
        object.__setattr__(self, "sample_frames", ids.copy())
        if self.depth_m is not None:
            if self.intrinsics is None or self.camera_to_base is None:
                raise ValueError("Depth requires intrinsics and camera_to_base")
            depth = np.asarray(self.depth_m, dtype=float)
            if depth.ndim != 2 or not np.isfinite(depth).all() or np.any(depth < 0):
                raise ValueError("Depth must be a finite nonnegative image in metres")
            object.__setattr__(self, "depth_m", depth.copy())
        if self.intrinsics is not None:
            k = np.asarray(self.intrinsics, dtype=float)
            if (
                k.shape != (3, 3)
                or not np.isfinite(k).all()
                or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1])
            ):
                raise ValueError("Invalid calibrated camera intrinsics")
            object.__setattr__(self, "intrinsics", k.copy())
        if self.camera_to_base is not None:
            pose = np.asarray(self.camera_to_base, dtype=float)
            if (
                pose.shape != (4, 4)
                or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1)
            ):
                raise ValueError(
                    "camera_to_base must be a rigid optical-to-base transform"
                )
            object.__setattr__(self, "camera_to_base", pose.copy())


@dataclass(frozen=True)
class RobotModel:
    """Original URDF and explicit arm layout for the current source-near method.

    The default layout uses one or two six-axis arms followed by paired
    prismatic jaws. Explicit arm_joint_groups and gripper_calibrations support
    seven-axis arms and measured linkage tables. Each table contains joint_names,
    width_m, qpos, driver_index, pad_links and local outward_axes. It does not
    retarget between robots. Visual records contain link, absolute mesh path
    and local origin matrix.
    """

    urdf: Path
    joint_names: tuple[str, ...]
    side_names: tuple[str, ...]
    tcp_links: tuple[str, ...]
    flange_to_tcp: np.ndarray
    visuals: tuple[dict[str, Any], ...]
    pad_center_tcp_z_m: float
    max_half_open_m: float = 0.065
    arm_joint_groups: tuple[tuple[str, ...], ...] = ()
    gripper_calibrations: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ReconstructionInput:
    """One real demonstration, its unchanged robot, and camera observations."""

    demonstration: RobotDemonstration
    robot: RobotModel
    cameras: dict[str, CameraObservation]
    primary_camera: str
    instruction: str

    def __post_init__(self) -> None:
        if self.primary_camera not in self.cameras or not self.instruction.strip():
            raise ValueError("A primary camera and task instruction are required")
        if self.cameras[self.primary_camera].first_image is None:
            raise ValueError("The primary camera requires a first_image")
        if tuple(self.demonstration.joint_names) != tuple(self.robot.joint_names):
            raise ValueError("Robot and demonstration joint ordering must agree")
        if any(
            len(c.sample_frames) != len(self.demonstration.qpos)
            for c in self.cameras.values()
        ):
            raise ValueError(
                "Each camera needs one video-frame mapping per trajectory sample"
            )


@dataclass(frozen=True)
class RegisteredScene:
    """Generated meshes registered to the robot base, plus sampled world vertices."""

    description: dict[str, Any]
    vertices: dict[str, np.ndarray]


@dataclass(frozen=True)
class RefinedTrajectory:
    """Nominal joint reference and evidence; no physical success or object commands."""

    qpos: np.ndarray
    timestamps: np.ndarray
    tcp_poses: dict[str, np.ndarray]
    report: dict[str, Any]
    object_pose_hypotheses: dict[str, np.ndarray] = field(default_factory=dict)
