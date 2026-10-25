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

"""Fixed POE geometry extracted from the caller's actual PK serial chain."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__: list[str] = []

# PK stores URDF rotations in float32 before a chain is moved to float64.
# This allowance covers that representation error, not calibrated CAD drift.
_STRUCTURE_ATOL = 5e-7


@dataclass(frozen=True)
class _SSIKModel:
    names: tuple[str, ...]
    axes: np.ndarray
    offsets: np.ndarray
    home_rotation: np.ndarray
    shoulder: np.ndarray
    wrist: np.ndarray
    family: str
    fingerprint: str
    frames: np.ndarray
    local_axes: np.ndarray
    limits: np.ndarray

    @property
    def dof(self) -> int:
        return len(self.names)


def _concurrence(points: np.ndarray, axes: np.ndarray) -> np.ndarray | None:
    projections = np.eye(3)[None] - axes[:, :, None] * axes[:, None, :]
    matrix = projections.sum(axis=0)
    if np.linalg.matrix_rank(matrix, tol=1e-10) < 3:
        return None
    pivot = np.linalg.solve(matrix, np.einsum("nij,nj->i", projections, points))
    residual = np.linalg.norm(np.cross(points - pivot, axes), axis=-1)
    return pivot if residual.max() <= _STRUCTURE_ATOL else None


def _extract_model(chain: pk.SerialChain) -> _SSIKModel:
    """Flatten fixed frames without changing model coordinates or joint order."""
    names = tuple(chain.get_joint_parameter_names())
    if len(names) not in (6, 7):
        raise ValueError("SSIK requires a six- or seven-revolute-joint serial chain")
    home = np.eye(4)
    pending = np.eye(4)
    frames, local_axes = [], []
    points, axes, observed_names = [], [], []
    digest = hashlib.sha256(repr(names).encode())
    for frame in chain._serial_frames:
        for offset in (frame.link.offset, frame.joint.offset):
            if offset is not None:
                transform = (
                    offset.get_matrix()[0].detach().cpu().numpy().astype(np.float64)
                )
                digest.update(transform.tobytes())
                home = home @ transform
                pending = pending @ transform
        joint = frame.joint
        if joint.joint_type == "fixed":
            continue
        if joint.joint_type != "revolute":
            raise ValueError("SSIK supports revolute joints only")
        local_axis = joint.axis.detach().cpu().numpy().astype(np.float64)
        digest.update(local_axis.tobytes())
        frames.append(pending[:3].reshape(-1).copy())
        local_axes.append(local_axis.copy())
        pending = np.eye(4)
        axis = home[:3, :3] @ local_axis
        if not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError("SSIK received a degenerate joint axis")
        axes.append(axis / np.linalg.norm(axis))
        points.append(home[:3, 3].copy())
        observed_names.append(joint.name)
    if tuple(observed_names) != names:
        raise ValueError("SSIK chain traversal and joint parameter order disagree")
    h, p = np.zeros((7, 3)), np.zeros((8, 3))
    h[: len(names)] = axes
    points = np.array(points)
    p[: len(names)] = np.diff(np.vstack((np.zeros(3), points)), axis=0)
    p[len(names)] = home[:3, 3] - points[-1]
    shoulder, wrist = np.zeros(3), np.zeros(3)
    family = "unsupported"
    if len(names) == 6:
        if max(np.linalg.norm(np.cross(h[1], h[j])) for j in (2, 3)) <= _STRUCTURE_ATOL:
            family = "three_parallel"
    else:
        s = _concurrence(points[:3], h[:3])
        w = _concurrence(points[4:7], h[4:7])
        if s is not None and w is not None:
            family, shoulder, wrist = "srs", s, w
    frames.append(pending[:3].reshape(-1).copy())
    fixed = np.zeros((8, 12))
    fixed[: len(frames)] = frames
    local = np.zeros((7, 3))
    local[: len(names)] = local_axes
    limits = np.stack(
        (chain.low.detach().cpu().numpy(), chain.high.detach().cpu().numpy()), axis=-1
    ).astype(np.float64)
    digest.update(limits.tobytes())
    return _SSIKModel(
        names,
        h,
        p,
        home[:3, :3].copy(),
        shoulder,
        wrist,
        family,
        digest.hexdigest(),
        fixed,
        local,
        limits,
    )
