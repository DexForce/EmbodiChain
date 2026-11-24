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
"""URDF collision surfaces for one hand, excluding upstream arm geometry."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from embodichain.toolkits.graspkit.pg_grasp.urdf_surfaces import (
    sample_urdf_collision_surfaces,
)

if TYPE_CHECKING:
    import trimesh
    from ._kinematics import Robot

__all__: list[str] = []


def hand_collision_surfaces(robot: Robot, side: int) -> dict[str, np.ndarray]:
    """Select the pad links' common hand subtree and sample authored collisions.

    Calibration identifies the pads; URDF topology identifies the palm and
    articulated links. Reject a subtree containing any non-hand moving joint,
    rather than silently including an arm or substituting visual geometry.
    Returned points are link-local and use the caller's world-aware FK.
    """
    document = ET.parse(robot.path).getroot()
    joints = document.findall("joint")
    parents = {
        j.find("child").get("link"): j.find("parent").get("link") for j in joints
    }
    pads = tuple(robot.finger_surfaces(side)[0])
    if len(pads) != 2 or pads[0] == pads[1]:
        raise ValueError(
            "Full-hand clearance requires two distinct calibrated pad links"
        )

    def ancestors(link):
        result = [link]
        while link in parents:
            link = parents[link]
            result.append(link)
        return result

    chains = [ancestors(link) for link in pads]
    root = next((link for link in chains[0] if link in chains[1]), None)
    if root is None:
        raise ValueError("Pad links do not share a URDF hand root")
    selected = {root}
    for _ in range(len(joints)):
        additions = {
            j.find("child").get("link")
            for j in joints
            if j.find("parent").get("link") in selected
        }
        if additions <= selected:
            break
        selected.update(additions)
    allowed = {robot.names[int(i)] for i in robot.jaw_columns[side]}
    for joint in joints:
        if (
            joint.find("child").get("link") in selected
            and joint.find("parent").get("link") in selected
        ):
            if joint.get("type") != "fixed" and joint.get("name") not in allowed:
                raise ValueError(
                    "Hand subtree contains a non-hand moving joint: "
                    + joint.get("name")
                )
    links = [
        link.get("name")
        for link in document.findall("link")
        if link.get("name") in selected and link.findall("collision")
    ]
    if not set(pads) <= set(links):
        raise ValueError("Calibrated pads require authored URDF collision geometry")
    clouds = sample_urdf_collision_surfaces(
        robot.path,
        link_names=links,
        max_edge_by_link={link: 0.004 if link in pads else 0.008 for link in links},
        max_subdivisions=8,
    )
    result = {}
    for link, tensor in clouds.items():
        points = tensor.numpy()
        # Bound repeated optimizer queries on CAD meshes with excessive seam
        # tessellation. Preserve axis extrema as well as distributed samples;
        # this remains an explicitly sampled, not continuous, collision check.
        if len(points) > 4096:
            ids = np.unique(
                np.r_[
                    np.linspace(0, len(points) - 1, 4096).astype(int),
                    points.argmin(0),
                    points.argmax(0),
                ]
            )
            points = points[ids]
        result[link] = points
    return result


class ClosedCollisionPart:
    """One closed authored component; overlapping parts keep independent SDFs."""

    def __init__(self, mesh: trimesh.Trimesh) -> None:
        import open3d as o3d

        self.bounds = np.asarray(mesh.bounds)
        self.bvh = o3d.t.geometry.RaycastingScene(nthreads=2)
        self.bvh.add_triangles(
            o3d.t.geometry.TriangleMesh(
                o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.int64)),
            )
        )


def closed_hand_collision_parts(
    robot: Robot, link_names: Iterable[str]
) -> dict[str, tuple[ClosedCollisionPart, ...]]:
    """Decode existing collisions without filling holes or changing assets.

    Components are deliberately separate: parity inside two overlapping closed
    solids must not cancel their penetration. Collision origins and mesh scale
    come from the same URDF decoder used by the forward surface sampler.
    """
    from pathlib import Path
    import trimesh
    from embodichain.toolkits.graspkit.pg_grasp._urdf_geometry import (
        _load_collision_geometry,
    )

    path = Path(robot.path)
    root = ET.parse(path).getroot()
    result = {}
    for name in link_names:
        link = root.find(f"link[@name='{name}']")
        if link is None:
            raise ValueError(f"Missing hand collision link: {name}")
        parts = []
        for collision in link.findall("collision"):
            vertices, faces, origin, _ = _load_collision_geometry(
                collision, path.parent
            )
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            mesh.apply_transform(origin)
            for component in mesh.split(only_watertight=False):
                if not component.is_watertight:
                    raise ValueError(
                        f"Reverse hand penetration requires closed authored collision components: {name}"
                    )
                parts.append(ClosedCollisionPart(component))
        if not parts:
            raise ValueError(f"Missing closed authored hand collision geometry: {name}")
        result[name] = tuple(parts)
    return result


def reverse_hand_penetration(
    parts: tuple[ClosedCollisionPart, ...],
    target_surface: np.ndarray,
    transforms: np.ndarray,
) -> float:
    """Maximum sampled target penetration into a union of closed hand parts.

    Only points inside each part's AABB can have negative SDF. Query each part
    separately and retain the deepest penetration, including overlapping parts.
    This augments forward clearance; a zero result is not a clearance proof.
    """
    import open3d as o3d

    depth = 0.0
    for transform in transforms:
        local = (target_surface - transform[:3, 3]) @ transform[:3, :3]
        for part in parts:
            inside = np.all(
                (local >= part.bounds[0]) & (local <= part.bounds[1]), axis=1
            )
            if not inside.any():
                continue
            distance = part.bvh.compute_signed_distance(
                o3d.core.Tensor(local[inside].astype(np.float32)),
                nthreads=2,
                nsamples=3,
            ).numpy()
            depth = max(depth, -float(distance.min()))
    return depth
