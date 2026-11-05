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

"""Discrete FCL clearance of conservative input-mesh solids."""

from __future__ import annotations

import math
from typing import Any
import torch
from embodichain.compute.geometry.mesh_bounds import maximum_halfspace_violation
from ._collision_backends import _require_fcl

__all__ = ["convex_input_clearance"]


def convex_input_clearance(
    vertices: torch.Tensor,
    poses: torch.Tensor,
    other_vertices: torch.Tensor,
    other_pose: torch.Tensor,
    frames: torch.Tensor,
    *,
    clearance_m: float,
    numerical_allowance_m: float,
) -> dict[str, Any]:
    """Qualify selected discrete frames against conservative solids of input meshes.

    Args:
        vertices: Already-scaled moving local vertices, shape (1, V, 3).
        poses: Moving local-to-world rigid poses, shape (1, T, 4, 4).
        other_vertices: Already-scaled obstacle local vertices, shape (1, W, 3).
        other_pose: Same-observation obstacle-to-world pose, shape (1, 4, 4).
        frames: Ordered integer frame indices; order and repeats are retained.
        clearance_m: Required nonnegative gap in metres.
        numerical_allowance_m: Explicit nonnegative numerical allowance in metres.

    Returns:
        Clearance receipt with first rejection, minimum distance and actual
        checked-frame count. Convex solids reject full containment, unlike
        triangle-surface distance alone. This is an input-envelope query, not
        a native-cooked, continuous-collision or physical-contact certificate.
    """
    if (
        not math.isfinite(clearance_m)
        or clearance_m < 0
        or not math.isfinite(numerical_allowance_m)
        or numerical_allowance_m < 0
    ):
        raise ValueError(
            "Input convex clearance margins must be finite and nonnegative."
        )
    if (
        poses.ndim != 4
        or poses.shape[0] != 1
        or poses.shape[-2:] != (4, 4)
        or other_pose.shape != (1, 4, 4)
        or frames.ndim != 1
        or frames.dtype not in (torch.int32, torch.int64)
        or (frames < 0).any()
        or (frames >= poses.shape[1]).any()
    ):
        raise ValueError(
            "Input convex clearance requires B1 poses and ordered valid frame indices."
        )
    _require_fcl()
    import fcl
    import numpy as np
    import trimesh

    def solid(points: torch.Tensor) -> Any:
        if (
            points.ndim != 3
            or points.shape[0] != 1
            or points.shape[-1] != 3
            or not torch.isfinite(points).all()
        ):
            raise ValueError("Convex clearance requires complete finite vertices.")
        points = points[0].detach().cpu().double().numpy()
        if len(points) < 4 or np.linalg.matrix_rank(points - points[0]) != 3:
            raise ValueError("Convex clearance requires a nondegenerate solid.")
        hull = trimesh.convex.convex_hull(points)
        if not hull.is_volume or not hull.is_convex:
            raise ValueError("Convex clearance requires a nondegenerate solid.")
        offsets = np.einsum("ij,ij->i", hull.face_normals, hull.triangles[:, 0])
        if maximum_halfspace_violation(points, hull.face_normals, offsets) > 1e-8:
            raise ValueError("Convex cover does not contain every source vertex.")
        faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).ravel()
        return fcl.Convex(hull.vertices, len(hull.faces), faces)

    def transform(pose: torch.Tensor) -> Any:
        pose = pose.detach().cpu().double().numpy()
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-7, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=2e-5)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=2e-5)
        ):
            raise ValueError("Convex clearance requires rigid observed poses.")
        return fcl.Transform(pose[:3, :3], pose[:3, 3])

    payload = solid(vertices)
    obstacle = fcl.CollisionObject(solid(other_vertices), transform(other_pose[0]))
    result: dict[str, Any] = {
        "valid": True,
        "scope": "discrete_source_convex_solids_not_native_contact",
        "required_clearance_m": clearance_m,
        "numerical_allowance_m": numerical_allowance_m,
        "minimum_distance_m": None,
        "checked_frames": 0,
    }
    for frame in frames.tolist():
        moving = fcl.CollisionObject(payload, transform(poses[0, frame]))
        # Convex solids, unlike triangle soups, also reject complete containment.
        distance = (
            0.0
            if fcl.collide(moving, obstacle)
            else float(fcl.distance(moving, obstacle))
        )
        if not math.isfinite(distance):
            raise ValueError("Convex solid distance is unavailable.")
        previous = result["minimum_distance_m"]
        result["minimum_distance_m"] = (
            distance if previous is None else min(previous, distance)
        )
        result["checked_frames"] += 1
        if distance <= clearance_m + numerical_allowance_m:
            return {**result, "valid": False, "frame": frame}
    return result
