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
"""Authored source geometry adaptation for explicit pose declarations."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
from embodichain.compute.geometry.placement_dimensions import vertical_placement_bounds
from embodichain.compute.geometry.source_targets import (
    upright_source_quaternion,
    upright_origin_height,
    constrained_pick_approach,
)
from .source_geometry import (
    source_local_bounds,
    source_longest_local_axis,
    source_rotation,
    read_source_vertices,
)

__all__ = [
    "source_axis_vertical_component",
    "source_vertical_bounds",
    "source_upright_target",
    "source_handover_staging_pose",
    "source_constrained_pick_geometry",
]


def source_constrained_pick_geometry(
    source: Mapping[str, Any],
    *,
    table_top_z: float,
    release_origin_z: float,
    previously_aligned: bool,
    force_vertical: bool,
    supported_upright: bool,
    ordinary_release_clearance: float,
    slender_release_clearance: float,
) -> tuple[list[float], float]:
    """Resolve original constrained Pick direction and release-plane geometry.

    Args:
        source: Explicit authored shape, scale and intrinsic XYZ source angles.
        table_top_z: Measured source tabletop height.
        release_origin_z: Declared object's release-origin height.
        previously_aligned: Earlier semantic alignment established world +Z.
        force_vertical: Explicit recipe uses downward entry independent of source tilt.
        supported_upright: Recipe releases on its selected upright support.
        ordinary_release_clearance: Original ordinary-extent drop offset.
        slender_release_clearance: Original offset at the 2.5 extent ratio.

    Returns:
        Approach direction and release-clearance plane. The original approach
        uses authored Euler angles even when a pose matrix is present; supported
        release bounds retain their separate shared geometry convention. This
        is a declaration calculation, without native state or planning.

    Raises:
        ValueError: The selected source geometry or upright extent is unavailable.
    """
    axis = source_longest_local_axis(source)
    world_axis = (
        np.array([0.0, 0.0, 1.0])
        if previously_aligned
        else Rotation.from_euler(
            "XYZ", source.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).apply(axis)
    )
    approach = constrained_pick_approach(world_axis)
    if force_vertical:
        approach = [0.0, 0.0, -1.0]
    plane = float(table_top_z)
    if supported_upright:
        bounds = source_local_bounds(source)
        if bounds is None:
            raise ValueError("Upright release requires finite object extents.")
        ordered = sorted(hi - lo for lo, hi in zip(*bounds, strict=True))
        clearance = (
            slender_release_clearance
            if ordered[-1] / max(ordered[-2], 1e-6) >= 2.5
            else ordinary_release_clearance
        )
        bottom, _ = source_vertical_bounds(source, axis_aligned=True)
        plane = float(release_origin_z) + bottom - clearance
    return approach, plane


def source_axis_vertical_component(source: Mapping[str, Any]) -> float:
    """Read the longest authored axis in world vertical coordinates.

    Args:
        source: Physical source pose and shape declaration.

    Returns:
        Signed vertical component, preserving matrix precedence and the original
        raw-bounds fallback followed by glTF basis conversion.
    """
    return float(
        (source_rotation(source) @ np.asarray(source_longest_local_axis(source)))[2]
    )


def source_vertical_bounds(
    source: Mapping[str, Any], *, axis_aligned: bool
) -> tuple[float, float]:
    """Adapt complete scaled source vertices to the selected vertical convention.

    Args:
        source: Rigid or articulation physical source and declared pose/scale.
        axis_aligned: Align the unique dominant mesh axis instead of the authored pose.

    Returns:
        Original minimum/maximum object-origin offsets. These are declaration
        geometry; no settled pose or native collision result is inferred.
    """
    return vertical_placement_bounds(
        read_source_vertices(source, articulation=source.get("role") == "articulation"),
        source_rotation(source),
        axis_aligned=axis_aligned,
    )


def source_upright_target(
    source: Mapping[str, Any],
    *,
    support_height: float,
    world_yaw: float,
    ordinary_release_clearance: float,
    slender_release_clearance: float,
    full_rotation_staging: bool = False,
    staging_clearance: float = 0.0,
    rotation_margin: float = 0.0,
) -> tuple[float, list[float]] | None:
    """Adapt raw bounds and scaled rotation radius into explicit upright targets.

    Args:
        source: Authored rigid shape, scale and three intrinsic Euler angles.
        support_height: Explicit selected world support top.
        world_yaw: Explicit final vertical heading in radians.
        ordinary_release_clearance: Release offset for the ordinary source extent ratio.
        slender_release_clearance: Release offset at the original 2.5 extent ratio.
        full_rotation_staging: Bound all rotated vertices above support.
        staging_clearance: Fixed extra staging clearance for other selected routes.
        rotation_margin: Reserve around a complete source rotation radius.

    Returns:
        Object-origin height and XYZW quaternion, or None for unavailable raw
        bounds. Raw-axis minima intentionally precede body-scale/Z-up conversion;
        radius uses the complete scaled imported mesh. Pose construction delegates
        to Compute and does not claim contact, reachability or stability.
    """
    bounds = source_local_bounds(source)
    if bounds is None:
        return None
    extents = [hi - lo for lo, hi in zip(*bounds, strict=True)]
    axis_index = max(range(3), key=extents.__getitem__)
    ordered = sorted(extents)
    clearance = (
        slender_release_clearance
        if ordered[-1] / max(ordered[-2], 1.0e-6) >= 2.5
        else ordinary_release_clearance
    )
    angles = source.get("init_rot")
    if not isinstance(angles, list) or len(angles) != 3:
        raise ValueError("Source upright target requires three-value init_rot.")
    radius = (
        float(np.linalg.norm(read_source_vertices(source), axis=1).max())
        if full_rotation_staging
        else None
    )
    return (
        upright_origin_height(
            bounds[0][axis_index],
            support_height,
            clearance,
            staging_radius=radius,
            staging_margin=rotation_margin,
            staging_clearance=staging_clearance,
        ),
        upright_source_quaternion(
            angles, source_longest_local_axis(source), world_yaw=world_yaw
        ),
    )


def source_handover_staging_pose(
    source: Mapping[str, Any], table: Mapping[str, Any], *, lift_height: float
) -> dict[str, list[float]]:
    """Propose the original half-way table-inward pose from authored source facts.

    Args:
        source: Physical object's authored position and intrinsic Euler angles.
        table: Explicit table Cube or scaled mesh source declaration.
        lift_height: Explicit vertical source lift.

    Returns:
        Existing position/XYZW quaternion pose fields. This proposal deliberately
        retains authored Euler/position conventions instead of a live observation;
        the caller owns first-use eligibility, graph identity and later planning.
    """
    if table.get("shape", {}).get("shape_type") == "Cube":
        center = np.asarray(table["init_pos"], dtype=float)[:2]
    else:
        vertices = np.asarray(read_source_vertices(table))
        world = vertices @ Rotation.from_euler(
            "XYZ", table.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
        ).as_matrix().T + np.asarray(table["init_pos"])
        center = (world[:, :2].min(0) + world[:, :2].max(0)) * 0.5
    position = np.asarray(source["init_pos"], dtype=float).copy()
    position[:2] = (position[:2] + center) * 0.5
    position[2] += lift_height
    quaternion = Rotation.from_euler(
        "XYZ", source.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
    ).as_quat()
    return dict(position=position.tolist(), quaternion_xyzw=quaternion.tolist())
