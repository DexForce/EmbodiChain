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
"""Static placement frames from authored geometry, never runtime contact proofs."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import math
import numpy as np
import trimesh
from embodichain.compute.geometry.landing import select_container_landing
from .source_geometry import (
    read_source_vertices,
    source_rotation,
    source_position,
    source_local_bounds,
    source_longest_local_axis,
)

__all__ = [
    "source_container_target_pose",
    "source_support_target_pose",
    "source_pour_target_pose",
    "source_horizontal_slender_object",
]


def _pose(source: Mapping[str, Any]) -> np.ndarray:
    if source.get("init_local_pose") is not None:
        return np.asarray(source["init_local_pose"], dtype=float)
    result = np.eye(4)
    result[:3, :3] = source_rotation(source)
    result[:3, 3] = source_position(source)
    return result


def source_container_target_pose(
    container: Mapping[str, Any],
    child: Mapping[str, Any],
    arm_root_world: np.ndarray,
    *,
    occupied: Sequence[tuple[Sequence[float], Mapping[str, Any]]] = (),
    child_rotation: np.ndarray | None = None,
    occupied_rotations: Sequence[np.ndarray | None] = (),
) -> list[float] | None:
    """Resolve an arm-side landing using the original full source footprint.

    Args:
        container: Authored rigid container source declaration.
        child: Authored rigid payload source declaration.
        arm_root_world: Explicit measured/authored arm-origin world coordinates.
        occupied: Earlier local target matrices and their source payloads.
        child_rotation: Declared world orientation override, if selected.
        occupied_rotations: Aligned prior release-orientation overrides. Empty
            means all reservations retain their authored initial rotations.

    Returns:
        Flattened local translation pose, or None for a floor outside the
        original five-degree level bound or lacking a measured planar region.
        No native success, reachability or future collision is inferred.
    """
    container_pose = _pose(container)
    rotation = container_pose[:3, :3]
    if rotation[2, 2] < math.cos(math.radians(5.0)):
        return None
    root_local = rotation.T @ (np.asarray(arm_root_world) - container_pose[:3, 3])
    mesh = trimesh.load(str(container["shape"]["fpath"]), force="scene").to_geometry()
    mesh.vertices = read_source_vertices(container)

    def vertices(item, override):
        current = _pose(item)[:3, :3] if override is None else override
        return read_source_vertices(item) @ (rotation.T @ current).T

    selected = (
        tuple(occupied_rotations) if occupied_rotations else (None,) * len(occupied)
    )
    if len(selected) != len(occupied):
        raise ValueError("Container reservation rotations must match every payload.")
    footprints = [
        (
            float(target[3]),
            float(target[7]),
            float(np.linalg.norm(vertices(previous, override)[:, :2], axis=1).max()),
        )
        for (target, previous), override in zip(occupied, selected, strict=True)
    ]
    translation = select_container_landing(
        mesh, vertices(child, child_rotation), root_local, occupied=footprints
    )
    if translation is None:
        return None
    result = np.eye(4)
    result[:3, 3] = translation
    return result.reshape(-1).tolist()


def source_support_target_pose(
    support: Mapping[str, Any],
    child: Mapping[str, Any],
    *,
    axis_aligned: bool,
    clearance: float,
    support_fallback_extents: Sequence[float],
    child_fallback_extents: Sequence[float],
    world_support_top: float | None = None,
) -> list[float]:
    """Retain source-bound object-origin placement along the selected support axis.

    Args:
        support: Authored support geometry declaration.
        child: Authored payload geometry declaration.
        axis_aligned: Select the support's longest source axis as its normal.
        clearance: Explicit source release offset in metres.
        support_fallback_extents: Explicit recipe dimensions if source unavailable.
        child_fallback_extents: Explicit recipe dimensions if source unavailable.
        world_support_top: Optional source-provided measured world top height.

    Returns:
        Flattened local object-origin translation pose. Raw bounds deliberately
        retain historical scaling/basis semantics; no collision proof is issued.
    """
    support_bounds, child_bounds = source_local_bounds(support), source_local_bounds(
        child
    )
    extents = (
        list(support_fallback_extents)
        if support_bounds is None
        else [hi - lo for lo, hi in zip(*support_bounds, strict=True)]
    )
    payload = (
        list(child_fallback_extents)
        if child_bounds is None
        else [hi - lo for lo, hi in zip(*child_bounds, strict=True)]
    )
    child_half_height = max(payload) / 2.0
    translation = [0.0, 0.0, 0.0]
    if axis_aligned:
        normal = source_longest_local_axis(support)
        # Source longest-axis fallback is Z before any glTF basis conversion.
        source_axis = (
            2 if support_bounds is None else max(range(3), key=extents.__getitem__)
        )
        top = (
            extents[source_axis] / 2.0
            if support_bounds is None
            else support_bounds[1][source_axis]
        )
        translation = [
            component * (top + child_half_height + clearance) for component in normal
        ]
    else:
        height = (
            extents[2] / 2.0
            if world_support_top is None
            else float(world_support_top)
            - float(support.get("init_pos", [0.0, 0.0, 0.0])[2])
        )
        translation[2] = height + child_half_height + clearance
    result = np.eye(4)
    result[:3, 3] = translation
    return result.reshape(-1).tolist()


def source_pour_target_pose(
    source: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    finger_radius: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt authored source geometry to the original static Pour staging frame.

    Args:
        source: Frozen body declaration owning the vessel asset and initial pose.
        reference: Frozen receiver declaration owning its mesh and orientation.
        finger_radius: Explicit model-owned nominal reserve, not native geometry.

    Returns:
        Existing scene-entity pose fields and vessel axis/pivot metadata. No
        simulator, IK or live grasp selection is involved.
    """
    from embodichain.compute.geometry.pouring import source_pour_staging_frame

    upright = np.asarray(source_longest_local_axis(source))
    top = float((read_source_vertices(reference) @ source_rotation(reference)[2]).max())
    relative, rotation, pivot = source_pour_staging_frame(
        read_source_vertices(source),
        upright,
        source_rotation(source),
        top,
        finger_radius,
    )
    return (
        {
            "relative_pose": relative.reshape(-1).tolist(),
            "world_orientation": rotation.reshape(-1).tolist(),
        },
        {"upright_axis": upright.tolist(), "pivot_local": pivot.tolist()},
    )


def source_horizontal_slender_object(source: Mapping[str, Any]) -> bool:
    """Inspect an authored rigid source through the original full mesh selector.

    Args:
        source: Physical rigid-object declaration, pose and scaled input mesh.

    Returns:
        Source horizontal/slender proposal permission. Non-rigid sources return
        False. Geometry reads do not construct a simulator or grasp generator.
    """
    from embodichain.compute.geometry.placement_dimensions import (
        horizontal_slender_axis,
    )

    return source.get("role") == "rigid_object" and horizontal_slender_axis(
        read_source_vertices(source),
        source_rotation(source),
    )
