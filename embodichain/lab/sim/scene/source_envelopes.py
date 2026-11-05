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
"""Static physical source envelopes without semantic goals or live state."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .source_geometry import read_gltf_z_up_mesh
from .articulation_geometry import read_articulation_geometry, _root_pose

__all__ = ["source_world_vertices", "source_obstacle_bounds"]


def source_world_vertices(
    source: dict[str, Any],
    *,
    articulation: bool = False,
    local_vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Measure the declared finite envelope, preserving source frame/scale order.

    Explicit local_vertices are already scaled source geometry. Primitive cubes
    retain their authored size path; otherwise articulation and glTF readers own
    source geometry. The result is an envelope, not a native collision oracle.

    Args:
        source: Physical source declaration with pose, scale and shape or asset.
        articulation: Read an articulation asset when no local override exists.
        local_vertices: Optional already-scaled local source vertices.

    Returns:
        World-frame vertices with shape ``(V, 3)``.
    """
    position = source.get("init_pos")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("Source envelope requires a finite declared init_pos.")
    if not np.isfinite([float(value) for value in position]).all():
        raise ValueError("Source envelope requires a finite declared init_pos.")
    pose = _root_pose(source)
    scale = np.asarray(source.get("body_scale", [1.0, 1.0, 1.0]), dtype=float)
    if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Table landing geometry requires finite positive scale.")
    shape = source.get("shape", {})
    if shape.get("shape_type") == "Cube":
        import trimesh

        size = shape.get("size")
        if not isinstance(size, list) or len(size) != 3 or min(map(float, size)) <= 0:
            raise ValueError("Table landing requires measured primitive geometry.")
        low = [-float(value) / 2.0 for value in size]
        high = [float(value) / 2.0 for value in size]
        points = (
            np.asarray(trimesh.creation.box(extents=np.subtract(high, low)).vertices)
            * scale
        )
    elif local_vertices is not None:
        points = np.asarray(local_vertices)
    elif articulation:
        points = read_articulation_geometry(source["fpath"]).vertices * scale
    else:
        points = np.asarray(
            read_gltf_z_up_mesh(shape["fpath"], body_scale=scale).vertices,
            dtype=np.float64,
        )
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not len(points)
        or not np.isfinite(points).all()
    ):
        raise ValueError("Table landing requires finite mesh geometry.")
    return points @ pose[:3, :3].T + pose[:3, 3]


def source_obstacle_bounds(
    source: dict[str, Any],
    *,
    articulation: bool = False,
    local_vertices: np.ndarray | None = None,
) -> np.ndarray:
    """Bound fixed rigid geometry and the full travel of qualified sliders.

    Expanding the aggregate by every slider's maximum authored displacement
    deliberately overestimates fixed links and independent drawers. Revolute,
    floating and unqualified mechanisms are not assumed static. This envelope
    certifies neither the robot's approach path nor runtime contact.

    Args:
        source: Physical source declaration with pose, scale and shape or asset.
        articulation: Include the full authored travel of a fixed mechanism.
        local_vertices: Optional already-scaled local source vertices.

    Returns:
        World-frame minimum and maximum corners with shape ``(2, 3)``.
    """
    points = source_world_vertices(
        source, articulation=articulation, local_vertices=local_vertices
    )
    bounds = np.stack((points.min(0), points.max(0)))
    if not articulation:
        return bounds
    from pxr import Gf, UsdPhysics

    from .articulation_inputs import _fixed_base
    from embodichain.lab.sim.spawn.source_geometry import read_usd_source
    from .articulation_geometry import read_articulation_geometry

    if not _fixed_base(source):
        raise ValueError("Table landing requires a fixed articulation obstacle base.")
    snapshot = read_usd_source(source["fpath"])
    snapshot.require_geometry()
    stage = snapshot.stage
    joints = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
        and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
    ]
    if any(
        not prim.IsA(UsdPhysics.FixedJoint) and not prim.IsA(UsdPhysics.PrismaticJoint)
        for prim in joints
    ):
        raise ValueError(
            "Table landing cannot qualify non-prismatic articulation joints."
        )
    if not any(prim.IsA(UsdPhysics.PrismaticJoint) for prim in joints):
        return bounds
    geometry = read_articulation_geometry(source["fpath"])
    scale = np.asarray(source.get("body_scale", [1.0] * 3), dtype=float)
    if not np.allclose(scale, scale[0], atol=1e-6, rtol=1e-6):
        raise ValueError("Table landing slider geometry requires uniform scale.")
    expansion = np.zeros(3)
    for prim in joints:
        if not prim.IsA(UsdPhysics.PrismaticJoint):
            continue
        joint = UsdPhysics.PrismaticJoint(prim)
        parent, child = (
            geometry.link_poses[stage.GetPrimAtPath(targets[0]).GetName()]
            for targets in (
                joint.GetBody0Rel().GetTargets(),
                joint.GetBody1Rel().GetTargets(),
            )
        )
        anchor0 = parent[:3, 3] + parent[:3, :3] @ np.asarray(
            joint.GetLocalPos0Attr().Get(), dtype=float
        )
        anchor1 = child[:3, 3] + child[:3, :3] @ np.asarray(
            joint.GetLocalPos1Attr().Get(), dtype=float
        )
        axis_name = str(joint.GetAxisAttr().Get())
        if axis_name not in {"X", "Y", "Z"}:
            raise ValueError("Table landing slider requires an explicit XYZ axis.")
        quaternion = joint.GetLocalRot1Attr().Get()
        joint_rotation = np.asarray(
            Gf.Matrix3d(
                Gf.Quatd(quaternion.GetReal(), Gf.Vec3d(quaternion.GetImaginary()))
            )
        ).T
        axis = child[:3, :3] @ joint_rotation @ np.eye(3)["XYZ".index(axis_name)]
        limits = (
            np.asarray(
                [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()],
                dtype=float,
            )
            * scale[0]
        )
        if (
            not np.isfinite(axis).all()
            or not np.isclose(np.linalg.norm(axis), 1.0, atol=1e-6)
            or not np.isfinite(limits).all()
            or limits[0] >= limits[1]
        ):
            raise ValueError(
                "Table landing slider requires finite axis and joint limits."
            )
        authored_position = float(np.dot(anchor1 - anchor0, axis)) * scale[0]
        travel = float(np.max(np.abs(limits - authored_position)))
        expansion += (
            np.abs(
                Rotation.from_euler(
                    "XYZ", source.get("init_rot", [0.0, 0.0, 0.0]), degrees=True
                ).as_matrix()
                @ axis
            )
            * travel
        )
    if not np.isfinite(expansion).all():
        raise ValueError("Table landing articulation travel is not finite.")
    return bounds + np.stack((-expansion, expansion))
