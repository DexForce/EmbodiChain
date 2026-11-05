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

"""Pure mesh geometry for bounded drawer placement and containment.

Targets are geometric proposals; oversize payloads remain finite proposals and
must be qualified by the caller's physical execution and evidence contracts.
"""

from __future__ import annotations

import numpy as np
import trimesh

__all__ = [
    "drawer_placement_region",
    "drawer_placement_pose",
    "drawer_placement_pose_candidates",
    "drawer_containment_report",
]

RELEASE_GAP = 0.003
DRAWER_PLACEMENT_MARGIN = 0.005
DRAWER_PLACEMENT_ROTATIONS = (0.0, np.pi / 2.0)
DRAWER_PLACEMENT_MAX_CANDIDATES = 3


def _rotation_matrix(
    angle: float, axis: np.ndarray | tuple[float, float, float]
) -> np.ndarray:
    """Return one world-frame axis rotation for release-pose candidates."""
    axis_array = np.asarray(axis, dtype=float)
    axis_array /= np.linalg.norm(axis_array)
    x, y, z = axis_array
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [
                cosine + x * x * (1 - cosine),
                x * y * (1 - cosine) - z * sine,
                x * z * (1 - cosine) + y * sine,
            ],
            [
                y * x * (1 - cosine) + z * sine,
                cosine + y * y * (1 - cosine),
                y * z * (1 - cosine) - x * sine,
            ],
            [
                z * x * (1 - cosine) - y * sine,
                z * y * (1 - cosine) + x * sine,
                cosine + z * z * (1 - cosine),
            ],
        ]
    )


def drawer_placement_region(
    meshes: list[trimesh.Trimesh],
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a floor and center; missing walls or obstructions do not reject a task."""
    vertices = np.concatenate([mesh.vertices for mesh in meshes])
    low, high = vertices.min(0), vertices.max(0)
    components = [part for mesh in meshes for part in mesh.split(only_watertight=False)]
    floors = [
        mesh for mesh in components if mesh.extents[2] < min(mesh.extents[:2]) / 3
    ]
    if floors:
        floor = max(floors, key=lambda mesh: float(np.prod(mesh.extents[:2])))
        low[:2], high[:2] = floor.bounds[:, :2]
        low[2] = floor.bounds[1, 2]
        high[2] = max(high[2], low[2])
    return low, high


def drawer_placement_pose(
    vertices: np.ndarray,
    rotation: np.ndarray,
    link_pose: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Aim over the floor center, even when the object exceeds the drawer bounds."""
    if not all(
        np.isfinite(v).all() for v in (vertices, rotation, link_pose, low, high)
    ):
        raise ValueError(
            "Drawer placement requires finite observed geometry and poses."
        )
    local_rotation = link_pose[:3, :3].T @ rotation
    points = vertices @ local_rotation.T
    minimum, maximum = points.min(0), points.max(0)
    local_position = (low + high - minimum - maximum) / 2
    local_position[2] = low[2] - minimum[2] + RELEASE_GAP
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = link_pose[:3, :3] @ local_position + link_pose[:3, 3]
    return result


def drawer_placement_pose_candidates(
    vertices: np.ndarray,
    rotation: np.ndarray,
    link_pose: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    margin: float = DRAWER_PLACEMENT_MARGIN,
    preserve_vertical: bool = False,
) -> tuple[np.ndarray, ...]:
    """Generate a bounded set of link-local floor targets for one payload.

    The returned poses are object poses, not end-effector poses.  Candidates
    are eroded by the payload footprint and vertical extent before being
    emitted.  This is a geometric preflight only; callers still must run IK,
    velocity, rim-clearance, and terminal containment checks.
    """
    vertices = np.asarray(vertices, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    link_pose = np.asarray(link_pose, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or rotation.shape != (3, 3)
        or link_pose.shape != (4, 4)
        or low.shape != (3,)
        or high.shape != (3,)
        or not all(
            np.isfinite(value).all() for value in (rotation, link_pose, low, high)
        )
        or (low > high).any()
        or not np.isfinite(margin)
        or margin < 0.0
    ):
        raise ValueError("Drawer placement candidate inputs are invalid.")
    if not np.allclose(link_pose[:3, :3].T @ link_pose[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Drawer link pose must be a rigid transform.")

    link_rotation = link_pose[:3, :3]
    candidates: list[np.ndarray] = []
    extents = np.ptp(vertices, axis=0)
    isotropic = float(extents.max()) <= max(float(extents.min()) * 1.25, 1e-6)
    if isotropic and not preserve_vertical:
        rotation_variants = (
            rotation,
            _rotation_matrix(np.pi / 2.0, link_rotation[:, 0]) @ rotation,
            _rotation_matrix(np.pi / 2.0, link_rotation[:, 1]) @ rotation,
        )
    else:
        rotation_variants = tuple(
            _rotation_matrix(angle, link_rotation[:, 2]) @ rotation
            for angle in (
                (*DRAWER_PLACEMENT_ROTATIONS, -np.pi / 2.0, np.pi)
                if preserve_vertical
                else DRAWER_PLACEMENT_ROTATIONS
            )
        )
    for variant_index, world_rotation in enumerate(rotation_variants):
        local_rotation = link_rotation.T @ world_rotation
        points = vertices @ local_rotation.T
        minimum, maximum = points.min(0), points.max(0)
        safe_low = low + margin
        safe_high = high - margin
        center_low = safe_low[:2] - minimum[:2]
        center_high = safe_high[:2] - maximum[:2]
        # Keep a finite center even when the footprint exceeds the drawer.
        # The runtime must attempt the placement and let the physical result
        # decide whether the object actually fits.
        center = (center_low + center_high) / 2.0
        xy_candidates = (center,)
        if not isotropic and variant_index == 0:
            xy_candidates += (center_low * 0.25 + center_high * 0.75,)
        z = low[2] + margin - minimum[2] + RELEASE_GAP
        for xy in xy_candidates:
            pose = np.eye(4)
            pose[:3, :3] = world_rotation
            local_position = np.array([xy[0], xy[1], z])
            pose[:3, 3] = link_rotation @ local_position + link_pose[:3, 3]
            if not any(
                np.allclose(pose, existing, atol=1e-8) for existing in candidates
            ):
                candidates.append(pose)
                if len(candidates) >= (
                    5 if preserve_vertical else DRAWER_PLACEMENT_MAX_CANDIDATES
                ):
                    return tuple(candidates)
    return tuple(candidates)


def drawer_containment_report(
    vertices: np.ndarray,
    object_pose: np.ndarray,
    link_pose: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    margin: float = DRAWER_PLACEMENT_MARGIN,
    lateral_margin: float | None = None,
) -> dict[str, object]:
    """Check a full payload mesh; closing may omit the planning-only XY inset."""
    inset = margin if lateral_margin is None else lateral_margin
    if not np.isfinite(inset) or inset < 0.0:
        raise ValueError("Drawer lateral margin must be finite and nonnegative.")
    vertices = np.asarray(vertices, dtype=float)
    object_pose = np.asarray(object_pose, dtype=float)
    link_pose = np.asarray(link_pose, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    relative = np.linalg.inv(link_pose) @ object_pose
    local = vertices @ relative[:3, :3].T + relative[:3, 3]
    safe_low = low + margin
    # Keep floor/top tolerances independent of a planning-only lateral inset.
    safe_low[2] = low[2] - margin
    safe_high = high - margin
    safe_low[:2] = low[:2] + inset
    safe_high[:2] = high[:2] - inset
    minimum, maximum = local.min(0), local.max(0)
    checks = {
        "finite": bool(np.isfinite(local).all()),
        "lower_bounds": bool((minimum >= safe_low).all()),
        "upper_bounds": bool((maximum <= safe_high).all()),
        "floor_gap": float(minimum[2] - low[2]),
    }
    accepted = bool(
        checks["finite"]
        and checks["lower_bounds"]
        and checks["upper_bounds"]
        and checks["floor_gap"] >= -margin
    )
    return {
        "accepted": accepted,
        "minimum": minimum.tolist(),
        "maximum": maximum.tolist(),
        "safe_lower": safe_low.tolist(),
        "safe_upper": safe_high.tolist(),
        "checks": checks,
        "margin": float(margin),
        "lateral_margin": float(inset),
    }
