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

"""Authored drawer topology and link-local collision region adaptation."""

from __future__ import annotations

from typing import Any
import numpy as np
from embodichain.compute.geometry.drawer import drawer_placement_region
from .articulation_inputs import PrismaticBinding, discover_prismatic_parts, _stage
from .source_collisions import read_source_collision_meshes

__all__ = ["drawer_moving_links", "read_drawer_region", "source_drawer_container_pose"]


def drawer_moving_links(config: dict[str, Any], binding: PrismaticBinding) -> set[str]:
    """Include fixed descendants by physical topology, not USD path ancestry."""
    from pxr import UsdPhysics

    stage = _stage(config["fpath"])
    candidates = discover_prismatic_parts(config)
    candidate = next(p for p in candidates if p.joint == binding.joint)
    paths = {candidate.link_path}
    edges = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        if joint.GetJointEnabledAttr().Get() is False:
            continue
        parent, child = (
            joint.GetBody0Rel().GetTargets(),
            joint.GetBody1Rel().GetTargets(),
        )
        if len(parent) == len(child) == 1:
            edges.append((str(parent[0]), str(child[0])))
    while True:
        updated = paths | {child for parent, child in edges if parent in paths}
        if updated == paths:
            return {stage.GetPrimAtPath(path).GetName() for path in paths}
        paths = updated


def read_drawer_region(
    config: dict[str, Any], binding: PrismaticBinding
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the selected part in the same link frame E6 uses for its handle."""
    from .articulation_geometry import read_articulation_geometry

    poses = read_articulation_geometry(config["fpath"]).link_poses
    selected = drawer_moving_links(config, binding)
    meshes = []
    for (link, path), mesh in read_source_collision_meshes(config).items():
        if link not in selected:
            continue
        relative = np.linalg.inv(poses[binding.link]) @ poses[link]
        relative[:3, 3] *= binding.scale
        mesh.apply_transform(relative)
        meshes.append((path, mesh))
    structural = [mesh for path, mesh in meshes if path != binding.handle_path]
    return drawer_placement_region(structural or [mesh for _, mesh in meshes])


def source_drawer_container_pose(
    config: dict[str, Any],
    binding: PrismaticBinding,
    source: dict[str, Any],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Declare the source drawer target in the exact authored physical link frame.

    Args:
        config: Frozen articulation declaration owning the inspected asset.
        binding: Exact qualified prismatic part and uniform source scale.
        source: Frozen payload body declaration and initial orientation.
        lower: Inspected drawer-local minimum region coordinates.
        upper: Inspected drawer-local maximum region coordinates.

    Returns:
        Link-local 4x4 payload target. This uses the original authored link pose
        and scale once; native runtime re-grounding remains independently owned.
    """
    from .articulation_geometry import read_articulation_geometry, _root_pose
    from .source_geometry import read_source_vertices
    from embodichain.compute.geometry.drawer import drawer_placement_pose

    local = read_articulation_geometry(config["fpath"]).link_poses[binding.link].copy()
    local[:3, 3] *= binding.scale
    link = _root_pose(config) @ local
    target = drawer_placement_pose(
        read_source_vertices(source),
        _root_pose(source)[:3, :3],
        link,
        np.asarray(lower),
        np.asarray(upper),
    )
    return np.linalg.inv(link) @ target
