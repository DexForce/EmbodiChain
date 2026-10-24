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

"""Translate pinned NeRD assets through public Newton import and Spawn data.

Only a temporary CPU-side ModelBuilder is used for source interpretation. No
live World, backend, scene or model is mutated. Scene materialization remains
entirely owned by DexSim Spawn.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dexsim.spawn import ArticulationDesc

__all__ = [
    "make_reference_articulation",
    "reference_asset_hash",
    "reference_joint_names",
    "reference_profile",
]

from embodichain.data_pipeline.dynamics._reference_assets import (
    _asset,
    _ant_reference_friction,
    _import_reference,
    reference_asset_hash,
    reference_joint_names,
    reference_profile,
)


def _pose(transform: Any) -> np.ndarray:
    import warp as wp

    value = np.asarray(transform, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(
        wp.quat_to_matrix(wp.quat(*value[3:])), dtype=np.float32
    ).reshape(3, 3)
    pose[:3, 3] = value[:3]
    return pose


def make_reference_articulation(
    robot: str, physics_profile: str = "native"
) -> ArticulationDesc:
    """Build an independent public descriptor with imported physical properties.

    ``warp_reference`` reconstructs measured old-Warp asset parameters. The
    host applies a solver-specific Ant friction conversion to fresh descriptors
    when using Newton Featherstone; other solvers retain these source material
    coefficients. This does not select the old simulator or assert identical
    contact algorithms.
    Body/joint names retain their asset spelling, with only the importer's path
    prefix removed. The free root is represented by ``fixed_base=False`` and is
    created by Spawn. Default nonzero joint positions are in ``reference_profile``
    and must be applied through the host's public state interface after prepare.
    """
    _asset(robot)
    import newton
    from dexsim.spawn import (
        ArticulationDesc,
        CollisionDesc,
        JointDesc,
        LinkDesc,
        NewtonJointDesc,
        RenderDesc,
        RigidBodyPhysicsDesc,
    )
    from dexsim.engine import JointType

    builder = _import_reference(robot, physics_profile)
    names = [str(name).rsplit("/", 1)[-1] for name in builder.body_label]
    if len(names) != len(set(names)):
        raise ValueError("Reference body names are not unique")
    parents = {
        int(child): int(parent)
        for parent, child in zip(builder.joint_parent, builder.joint_child)
    }
    roots = [body for body, parent in parents.items() if parent < 0]
    if len(roots) != 1:
        raise ValueError("Reference asset must have one articulation root")
    root = roots[0]
    links = [
        LinkDesc(
            name=name,
            parent_name="" if parents[index] < 0 else names[parents[index]],
            pose=_pose(builder.body_q[index]),
            rigid_body=RigidBodyPhysicsDesc.dynamic(
                mass=float(builder.body_mass[index]),
                inertia=np.asarray(builder.body_inertia[index], dtype=np.float32)
                .reshape(3, 3)
                .copy(),
                com_position=np.asarray(
                    builder.body_com[index], dtype=np.float32
                ).copy(),
            ),
        )
        for index, name in enumerate(names)
    ]
    if robot == "cartpole":
        # Spawn exposes fixed/free articulation roots. Preserve the URDF's welded
        # massless slideBar when Newton's collapse moved its prismatic child to
        # the world; this adds no generalized coordinate or dynamic inertia.
        for link in links:
            if not link.parent_name:
                link.parent_name = "slideBar"
        links.append(
            LinkDesc(
                name="slideBar",
                parent_name="",
                pose=_pose(builder.body_q[root]),
                rigid_body=RigidBodyPhysicsDesc.static(),
            )
        )
    snapshot_names = (
        "shape_body",
        "shape_flags",
        "shape_type",
        "shape_scale",
        "shape_transform",
        "shape_material_ke",
        "shape_material_kd",
        "shape_material_mu",
        "shape_margin",
        "shape_gap",
    )
    snapshot = {key: np.asarray(getattr(builder, key)) for key in snapshot_names}
    for index, body, collision in CollisionDesc.from_newton_snapshot(snapshot):
        if body < 0:
            raise ValueError(
                "Reference articulation unexpectedly includes world geometry"
            )
        collision.segment_name = str(builder.shape_label[index]).rsplit("/", 1)[-1]
        # Snapshot conversion supplies geometry/basic contact fields. Preserve the
        # rest explicitly to avoid Spawn's defaults changing source materials.
        for field in fields(collision.newton):
            source_name = "shape_material_" + field.name
            if hasattr(builder, source_name):
                setattr(
                    collision.newton,
                    field.name,
                    float(getattr(builder, source_name)[index]),
                )
        collision.newton.is_solid = bool(builder.shape_is_solid[index])
        collision.newton.collision_group = int(builder.shape_collision_group[index])
        attribute = builder.custom_attributes.get("mujoco:condim")
        if attribute is not None:
            collision.newton.condim = int(
                getattr(
                    attribute.values.get(index, attribute.default),
                    "value",
                    attribute.values.get(index, attribute.default),
                )
            )
        links[body].collisions.append(collision)
        links[body].visuals.append(RenderDesc.from_geometry(collision))
    joints = []
    for index, (parent, child) in enumerate(
        zip(builder.joint_parent, builder.joint_child)
    ):
        if parent < 0 and robot != "cartpole":
            continue
        kind = builder.joint_type[index]
        if kind not in (newton.JointType.PRISMATIC, newton.JointType.REVOLUTE):
            raise ValueError(f"Unsupported reference joint type: {kind}")
        offset = builder.joint_qd_start[index]
        name = str(builder.joint_label[index]).rsplit("/", 1)[-1]
        properties = NewtonJointDesc(
            target_mode=int(builder.joint_target_mode[offset]),
            **{
                key: float(getattr(builder, source)[offset])
                for key, source in (
                    ("target_ke", "joint_target_ke"),
                    ("target_kd", "joint_target_kd"),
                    ("damping", "joint_damping"),
                    ("effort_limit", "joint_effort_limit"),
                    ("velocity_limit", "joint_velocity_limit"),
                    ("friction", "joint_friction"),
                    ("armature", "joint_armature"),
                    ("limit_ke", "joint_limit_ke"),
                    ("limit_kd", "joint_limit_kd"),
                )
            },
        )
        joints.append(
            JointDesc(
                name=name,
                parent_link_name=names[parent] if parent >= 0 else "slideBar",
                child_link_name=names[child],
                joint_type=(
                    JointType.PRISMATIC
                    if kind == newton.JointType.PRISMATIC
                    else (
                        JointType.CONTINUOUS
                        if robot == "cartpole" and name == "cart_to_pole"
                        else JointType.REVOLUTE
                    )
                ),
                origin_pose=(
                    _pose(builder.joint_X_p[index])
                    if parent >= 0
                    else np.linalg.inv(_pose(builder.body_q[root]))
                    @ _pose(builder.joint_X_p[index])
                ),
                target_pose=_pose(builder.joint_X_c[index]),
                axis=np.asarray(builder.joint_axis[offset], dtype=np.float32).copy(),
                lower_limit=float(builder.joint_limit_lower[offset]),
                upper_limit=float(builder.joint_limit_upper[offset]),
                armature=properties.armature,
                effort_limit=properties.effort_limit,
                velocity_limit=properties.velocity_limit,
                newton=properties,
            )
        )
    if tuple(joint.name for joint in joints) != reference_joint_names(robot):
        raise ValueError("Newton importer changed reference joint ordering")
    pose = _pose(builder.body_q[root])
    if robot == "ant":
        pose[:3, 3] = reference_profile(robot)["default_q"][:3]
    return ArticulationDesc(
        name=robot,
        pose=pose,
        links=([links[-1], *links[:-1]] if robot == "cartpole" else links),
        joints=joints,
        fixed_base=robot == "cartpole",
        root_link_name=names[root] if robot == "ant" else "slideBar",
        enable_self_collision=False,
    )
