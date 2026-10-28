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

"""Sparse, bounded pose fitting with exact interaction anchors on a fixed grid."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

__all__ = ["PoseRefinementResult", "refine_pose_trajectory"]


def _poses(value: object, name: str) -> np.ndarray:
    """Return owned, proper rigid transforms without accepting object arrays."""
    result = np.array(value, dtype=np.float64, copy=True)
    if (
        result.ndim != 3
        or result.shape[1:] != (4, 4)
        or not len(result)
        or not np.isfinite(result).all()
        or not np.allclose(result[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0)
        or not np.allclose(
            result[:, :3, :3].transpose(0, 2, 1) @ result[:, :3, :3],
            np.eye(3),
            atol=2e-6,
            rtol=0,
        )
        or not np.allclose(np.linalg.det(result[:, :3, :3]), 1, atol=2e-6, rtol=0)
    ):
        raise ValueError(f"{name} must contain finite proper SE(3) transforms")
    return result


@dataclass(frozen=True)
class PoseRefinementResult:
    """An owned fixed-grid fit and measured corrections to its input.

    Args:
        poses: Refined world TCP transforms, shaped ``(T, 4, 4)``.
        max_translation_m: Largest Euclidean correction from the input.
        max_rotation_rad: Largest geodesic rotation correction from the input.
        iterations: Maximum QP iteration count across coordinate solves.
    """

    poses: np.ndarray
    max_translation_m: float
    max_rotation_rad: float
    iterations: int

    def __post_init__(self) -> None:
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
                for value in (self.max_translation_m, self.max_rotation_rad)
            )
            or type(self.iterations) is not int
            or self.iterations < 1
        ):
            raise ValueError("refinement diagnostics must be finite and nonnegative")
        poses = _poses(self.poses, "refined poses")
        poses.setflags(write=False)
        object.__setattr__(self, "poses", poses)


def refine_pose_trajectory(
    poses: np.ndarray,
    *,
    anchors: Mapping[int, np.ndarray],
    reference: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    max_translation_m: float = 0.02,
    max_rotation_rad: float = math.radians(10),
    smoothing_weight: float = 1.0,
    max_iterations: int = 4000,
) -> PoseRefinementResult:
    """Fit a pose reference while retaining exact, caller-owned hard anchors.

    Six sparse, banded quadratic programs fit translation and a local rotation
    chart. Memory is linear in sample count; no dense sample-pair matrices are
    formed. ``reference`` may contain a geometric anchor warp, but trust bounds
    are always measured against ``poses``. Rotations spanning the chart cut are
    rejected. Component trust boxes are followed by exact Euclidean/geodesic
    checks, so an approximate box never enlarges the declared correction budget.
    This operation neither changes timestamps nor certifies collisions or IK.

    Args:
        poses: Original proper world TCP poses, shaped ``(T, 4, 4)``.
        anchors: Exact target transforms keyed by source sample index.
        reference: Optional geometrically corrected initial reference.
        weights: Positive per-sample observation weights, shaped ``(T,)``.
        max_translation_m: Maximum total correction relative to original input.
        max_rotation_rad: Maximum total orientation correction in radians.
        smoothing_weight: Nonnegative penalty on second sample differences.
        max_iterations: Bounded iteration budget for each coordinate QP.

    Returns:
        Owned refined poses and measured correction diagnostics.

    Raises:
        ValueError: Inputs, chart, constraints, convergence or budgets are invalid.
    """
    import osqp

    original = _poses(poses, "poses")
    count = len(original)
    target = original.copy() if reference is None else _poses(reference, "reference")
    if count < 3 or target.shape != original.shape:
        raise ValueError("refinement needs at least three aligned samples")
    for value, name in (
        (max_translation_m, "max_translation_m"),
        (max_rotation_rad, "max_rotation_rad"),
        (smoothing_weight, "smoothing_weight"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (name != "smoothing_weight" and value == 0):
            raise ValueError(f"{name} has an invalid range")
    if max_rotation_rad >= math.pi / 2:
        raise ValueError("rotation correction must be smaller than pi/2")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    weight = (
        np.ones(count) if weights is None else np.asarray(weights, dtype=np.float64)
    )
    if weight.shape != (count,) or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("weights must be finite positive sample weights")
    if not isinstance(anchors, Mapping) or not anchors:
        raise ValueError("refinement requires explicit hard anchors")
    if any(type(index) is not int or not 0 <= index < count for index in anchors):
        raise ValueError("anchor indices must address source samples")
    indices = sorted(anchors)
    anchored = _poses(np.stack([anchors[index] for index in indices]), "anchors")
    base = original[0, :3, :3]

    def coordinates(values: np.ndarray) -> np.ndarray:
        angular = Rotation.from_matrix(base.T @ values[:, :3, :3]).as_rotvec()
        if np.any(np.linalg.norm(angular, axis=1) >= math.pi - 0.1):
            raise ValueError("trajectory crosses the local rotation chart boundary")
        return np.column_stack((values[:, :3, 3], angular))

    source_coordinates = coordinates(original)
    reference_coordinates = coordinates(target)
    anchor_coordinates = coordinates(anchored)
    # O(T) storage, with three diagonals before forming the five-diagonal Hessian.
    difference = sparse.diags(
        [np.ones(count - 2), -2 * np.ones(count - 2), np.ones(count - 2)],
        [0, 1, 2],
        shape=(count - 2, count),
        format="csc",
    )
    hessian = sparse.diags(weight, format="csc") + smoothing_weight * (
        difference.T @ difference
    )
    hessian = sparse.triu(hessian, format="csc")
    bounds = np.array([max_translation_m] * 3 + [max_rotation_rad] * 3)
    result = np.empty((count, 6))
    iterations = 0
    solver = osqp.OSQP()
    for dimension in range(6):
        lower = (source_coordinates[:, dimension] - bounds[dimension]).copy()
        upper = (source_coordinates[:, dimension] + bounds[dimension]).copy()
        lower[indices] = upper[indices] = anchor_coordinates[:, dimension]
        if np.any(
            np.abs(
                anchor_coordinates[:, dimension]
                - source_coordinates[indices, dimension]
            )
            > bounds[dimension] + 1e-8
        ):
            raise ValueError("interaction anchor exceeds the refinement budget")
        linear = -weight * reference_coordinates[:, dimension]
        if dimension == 0:
            solver.setup(
                P=hessian,
                q=linear,
                A=sparse.eye(count, format="csc"),
                l=lower,
                u=upper,
                verbose=False,
                eps_abs=1e-9,
                eps_rel=1e-9,
                max_iter=max_iterations,
                polishing=True,
                scaled_termination=True,
            )
        else:
            solver.update(q=linear, l=lower, u=upper)
            solver.warm_start(x=reference_coordinates[:, dimension], y=np.zeros(count))
        solved = solver.solve(raise_error=False)
        if (
            solved.info.status != "solved"
            or solved.x is None
            or not np.isfinite(solved.x).all()
        ):
            raise ValueError(f"pose refinement did not converge: {solved.info.status}")
        result[:, dimension] = solved.x
        iterations = max(iterations, solved.info.iter)
    output = original.copy()
    output[:, :3, 3] = result[:, :3]
    output[:, :3, :3] = base @ Rotation.from_rotvec(result[:, 3:]).as_matrix()
    output[indices] = anchored
    translation = float(
        np.linalg.norm(output[:, :3, 3] - original[:, :3, 3], axis=1).max()
    )
    rotation = float(
        Rotation.from_matrix(output[:, :3, :3] @ original[:, :3, :3].transpose(0, 2, 1))
        .magnitude()
        .max()
    )
    if translation > max_translation_m + 1e-7 or rotation > max_rotation_rad + 1e-7:
        raise ValueError("refined path exceeds the exact correction budget")
    return PoseRefinementResult(output, translation, rotation, iterations)
