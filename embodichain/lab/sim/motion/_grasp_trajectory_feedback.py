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
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _PoseOffset:
    reference: tuple[tuple[float, ...], ...]
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
        urdf: Path,
        tool_link: str,
        joint_indices: tuple[int, ...],
        joint_names: tuple[str, ...],
        root_position: tuple[float, ...],
        root_rotation: tuple[float, ...],
        initial_object: np.ndarray,
        initial_offset: _PoseOffset | None,
        tracks: tuple[_GraspTrack, ...],
        hz: float,
    ) -> None:
        import pytorch_kinematics as pk

        self._chain = pk.build_serial_chain_from_urdf(urdf.read_bytes(), tool_link).to(
            dtype=torch.float64, device="cpu"
        )
        if tuple(self._chain.get_joint_parameter_names()) != tuple(
            joint_names[index] for index in joint_indices
        ):
            raise ValueError(
                "The trajectory's feedback joints differ from its tool chain."
            )
        limits = np.array(self._chain.get_joint_limits()).T
        self._lower = limits[:, 0] + 1e-5
        self._upper = limits[:, 1] - 1e-5
        self._joint_indices = list(joint_indices)
        self._root_position = np.asarray(root_position, dtype=float)
        self._root_rotation = Rotation.from_euler(
            "xyz", root_rotation, degrees=True
        ).as_matrix()
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

    def _solve(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        q = seed.astype(float)
        for _ in range(20):
            joints = torch.as_tensor(q, dtype=torch.float64)
            current = self._chain.forward_kinematics(joints).get_matrix()[0].numpy()
            current[:3, :3] = self._root_rotation @ current[:3, :3]
            current[:3, 3] = self._root_rotation @ current[:3, 3] + self._root_position
            error = np.r_[
                target[:3, 3] - current[:3, 3],
                0.15
                * Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
            ]
            if np.linalg.norm(error) < 1e-7:
                break
            jacobian = self._chain.jacobian(joints)[0].numpy()
            jacobian[:3] = self._root_rotation @ jacobian[:3]
            jacobian[3:] = self._root_rotation @ jacobian[3:]
            jacobian[3:] *= 0.15
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + np.eye(6) * 1e-8, error
            )
            q = np.clip(q + delta, self._lower, self._upper)
        return q.astype(np.float32)

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
        result[self._joint_indices] = self._solve(
            target, nominal_q[self._joint_indices]
        )
        return result
