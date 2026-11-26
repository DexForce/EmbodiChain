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
"""Plan an ordered Cartesian track on a fixed control grid."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.compute.trajectory._cartesian_path import (
    _interpolate_pose_path,
    _quintic_progress,
    _sample_axis_arc,
)

from .solvers._trajectory_ik import _TrajectoryIK

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _ArcGeometry:
    axis: np.ndarray
    radius: float
    start_angle: float
    end_angle: float
    heights: tuple[float, float]
    offset: np.ndarray
    orientation: np.ndarray

    def positions(self, progress: np.ndarray) -> np.ndarray:
        start, end = self.heights
        return _sample_axis_arc(
            self.axis,
            self.radius,
            self.start_angle + (self.end_angle - self.start_angle) * progress,
            (1 - progress) * start + progress * end,
            self.offset,
        )

    def endpoint(
        self, *, end: bool, radial_offset: float, tangent_offset: float
    ) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.orientation
        pose[:3, 3] = self.positions(np.array([float(end)]))[0]
        angle = self.end_angle if end else self.start_angle
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        tangent = np.sign(self.end_angle - self.start_angle) * np.array(
            [-np.sin(angle), np.cos(angle), 0.0]
        )
        pose[:3, 3] += self.axis[:3, :3] @ (
            radial_offset * radial + tangent_offset * tangent
        )
        return pose


@dataclass(frozen=True, slots=True)
class _TrackKeyframe:
    step: int
    pose: np.ndarray
    grip: float
    joint_home: bool
    arc: _ArcGeometry | None = None


@dataclass(frozen=True, slots=True)
class _TrackLimits:
    joint_names: tuple[str, ...]
    position: np.ndarray
    velocity: np.ndarray
    step_dt: float

    def __post_init__(self) -> None:
        joints = len(self.joint_names)
        if (
            self.position.shape != (joints, 2)
            or np.isnan(self.position).any()
            or (self.position[:, 0] > self.position[:, 1]).any()
            or self.velocity.shape != (joints,)
            or not np.isfinite(self.velocity).all()
            or (self.velocity < 0).any()
        ):
            raise ValueError("Trajectory limits must match the ordered robot joints.")

    def validate(self, positions: np.ndarray) -> None:
        if not np.isfinite(positions).all():
            raise ValueError("Trajectory joint positions must be finite.")
        low, high = self.position.T
        position_bad = (positions < low - 1e-6) | (positions > high + 1e-6)
        if position_bad.any():
            frame, joint = np.argwhere(position_bad)[0]
            raise ValueError(
                f"Trajectory sample {frame} exceeds {self.joint_names[joint]!r} position limits."
            )
        speed_bad = (
            np.abs(np.diff(positions, axis=0)) > self.velocity * self.step_dt + 1e-5
        )
        if speed_bad.any():
            frame, joint = np.argwhere(speed_bad)[0]
            raise ValueError(
                f"Trajectory interval {frame} exceeds {self.joint_names[joint]!r} velocity limit."
            )


def _plan_cartesian_track(
    keyframes: tuple[_TrackKeyframe, ...],
    *,
    home: np.ndarray,
    arm_ids: list[int],
    gripper_ids: list[int],
    control_hz: float,
    kinematics: _TrajectoryIK,
    position_tolerance: float,
    rotation_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample geometry, then solve each pose with the preceding valid posture."""
    frames = keyframes[-1].step + 1
    positions = np.broadcast_to(home, (frames, len(home))).astype(np.float32, copy=True)
    poses = np.empty((frames, 4, 4))
    home_arm = home[arm_ids]
    q = home_arm.copy()
    preceding_pose: np.ndarray | None = None
    first = 0
    for start, end in pairwise(keyframes):
        # Include frame zero once. Following spans begin after the shared endpoint.
        last = end.step + 1
        start_time, end_time = start.step / control_hz, end.step / control_hz
        u = _quintic_progress(
            (np.arange(first, last) / control_hz - start_time) / (end_time - start_time)
        )
        if end.joint_home:
            arm = (1 - u[:, None]) * q + u[:, None] * home_arm
            targets = kinematics.forward(arm)
            q = arm[-1].copy()
            preceding_pose = None
        else:
            if end.arc is not None:
                targets = np.tile(start.pose, (len(u), 1, 1))
                targets[:, :3, 3] = end.arc.positions(u)
            elif np.array_equal(start.pose, end.pose):
                targets = np.broadcast_to(start.pose, (len(u), 4, 4))
            else:
                targets = _interpolate_pose_path(start.pose, end.pose, u)
            arm = np.empty((len(u), len(arm_ids)))
            for row, pose in enumerate(targets):
                if not np.array_equal(pose, preceding_pose):
                    q = kinematics.plan_sample(
                        pose,
                        q,
                        position_tolerance=position_tolerance,
                        rotation_tolerance=rotation_tolerance,
                    )
                    preceding_pose = pose
                arm[row] = q
        poses[first:last] = targets
        positions[first:last, arm_ids] = arm
        grip = (1 - u) * start.grip + u * end.grip
        positions[first:last, gripper_ids] = grip[:, None]
        first = last

    # Qualify the quantized positions that will actually be sent.
    actual = kinematics.forward(positions[:, arm_ids])
    linear = np.linalg.norm(actual[:, :3, 3] - poses[:, :3, 3], axis=1)
    angular = Rotation.from_matrix(
        poses[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()
    if not np.isfinite(linear).all() or not np.isfinite(angular).all():
        raise ValueError("Planned TCP tracking errors must be finite.")
    bad = np.flatnonzero((linear > position_tolerance) | (angular > rotation_tolerance))
    if bad.size:
        frame = bad[0]
        raise ValueError(
            f"TCP tracking failed at trajectory sample {frame}: "
            f"{linear[frame]:.6g} m, {angular[frame]:.6g} rad."
        )
    return positions, poses
