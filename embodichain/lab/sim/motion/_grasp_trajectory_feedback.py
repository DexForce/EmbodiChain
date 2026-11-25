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
"""Pose corrections for authored joint trajectories with an observed grasp."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .solvers._trajectory_ik import _TrajectoryIK

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _PoseOffset:
    reference: tuple[tuple[float, ...], ...] | str
    rise: tuple[float, float]
    fall: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _GraspCapture:
    first: int
    last: int
    weight: float


@dataclass(frozen=True, slots=True)
class _GraspTrack:
    first: int
    last: int
    tool_in_object: tuple[tuple[float, ...], ...]
    captures: tuple[_GraspCapture, ...]
    object_offset: _PoseOffset | None = None


def _smooth_offset(time: float, offset: _PoseOffset) -> float:
    a, b = offset.rise
    c, d = offset.fall
    u = float(np.clip(min((time - a) / (b - a), (d - time) / (d - c)), 0.0, 1.0))
    return u**3 * (10 - 15 * u + 6 * u * u)


def _interpolate_rotation(
    start: np.ndarray, end: np.ndarray, weight: float
) -> np.ndarray:
    return Slerp([0, 1], Rotation.from_matrix([start, end]))([weight]).as_matrix()[0]


class _GraspTrajectoryFeedback:
    """Update the next target using observed object-to-tool transforms."""

    def __init__(
        self,
        kinematics: _TrajectoryIK,
        joint_indices: tuple[int, ...],
        initial_object: np.ndarray,
        initial_offset: _PoseOffset | None,
        tracks: tuple[_GraspTrack, ...],
        hz: float,
    ) -> None:
        self._kinematics = kinematics
        self._joint_indices = list(joint_indices)
        self._initial_object = initial_object
        self._initial_offset = initial_offset
        self._tracks = tracks
        self._hz = hz
        self._grasps = [np.array(track.tool_in_object) for track in tracks]
        self._reference_objects: dict[int, np.ndarray] = {}

    def observation_steps(self) -> set[int]:
        return {
            track.first for track in self._tracks if track.object_offset is not None
        } | {
            step
            for track in self._tracks
            for capture in track.captures
            for step in range(capture.first, capture.last + 1)
        }

    @staticmethod
    def _apply_offset(
        target: np.ndarray, observed: np.ndarray, offset: _PoseOffset, time: float
    ) -> np.ndarray:
        ideal = np.asarray(offset.reference)
        weight = _smooth_offset(time, offset)
        delta = observed @ np.linalg.inv(ideal)
        turn = _interpolate_rotation(np.eye(3), delta[:3, :3], weight)
        result = target.copy()
        result[:3, 3] = (
            ideal[:3, 3]
            + turn @ (target[:3, 3] - ideal[:3, 3])
            + weight * (observed[:3, 3] - ideal[:3, 3])
        )
        result[:3, :3] = turn @ target[:3, :3]
        return result

    def correct(
        self,
        frame: int,
        nominal_q: np.ndarray,
        nominal_tool: np.ndarray,
        observe: Callable[[], tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        time = frame / self._hz
        target = nominal_tool
        offset = self._initial_offset
        if offset is not None and offset.rise[0] < time < offset.fall[1]:
            # Before grasping, register the path to the settled object pose.
            target = self._apply_offset(target, self._initial_object, offset, time)
        else:
            for i, track in enumerate(self._tracks):
                if not track.first <= frame <= track.last:
                    continue
                grasp = self._grasps[i]
                for capture in track.captures:
                    if not capture.first <= frame <= capture.last:
                        continue
                    hand, obj = observe()
                    measured = np.linalg.inv(obj) @ hand
                    if capture.weight == 1.0:
                        grasp[:] = measured
                    else:
                        w = capture.weight
                        grasp[:3, 3] = (1 - w) * grasp[:3, 3] + w * measured[:3, 3]
                        grasp[:3, :3] = _interpolate_rotation(
                            grasp[:3, :3], measured[:3, :3], w
                        )
                    if track.object_offset is not None and capture.weight == 1.0:
                        self._reference_objects[i] = obj.copy()
                    break
                if track.object_offset is not None and i not in self._reference_objects:
                    _, obj = observe()
                    self._reference_objects[i] = obj.copy()
                desired_object = nominal_tool @ np.linalg.inv(track.tool_in_object)
                if track.object_offset is not None:
                    desired_object = self._apply_offset(
                        desired_object,
                        self._reference_objects[i],
                        track.object_offset,
                        time,
                    )
                target = desired_object @ grasp
                break
            else:
                return nominal_q.copy()

        result = nominal_q.copy()
        result[self._joint_indices] = self._kinematics.feedback_sample(
            target, nominal_q[self._joint_indices]
        )
        return result
