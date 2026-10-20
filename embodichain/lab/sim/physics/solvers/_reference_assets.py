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
import hashlib
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

_SOURCE_COMMIT = "476dc67fccc6a99be293bf611f897b1c06f13e30"
_ASSETS = {"cartpole": "cartpole_single.urdf", "ant": "ant.xml"}
_JOINTS = {
    "cartpole": ("slider_to_cart", "cart_to_pole"),
    "ant": (
        "hip_1",
        "ankle_1",
        "hip_2",
        "ankle_2",
        "hip_3",
        "ankle_3",
        "hip_4",
        "ankle_4",
    ),
}


def _asset(robot: str) -> Path:
    if robot not in _ASSETS:
        raise ValueError(f"Unsupported NeRD reference robot: {robot}")
    import embodichain

    return Path(embodichain.__file__).parent / "data" / "nerd" / _ASSETS[robot]


def reference_asset_hash(robot: str) -> str:
    """Return the SHA256 of the exact packaged reference robot source."""
    return hashlib.sha256(_asset(robot).read_bytes()).hexdigest()


def reference_joint_names(robot: str) -> tuple[str, ...]:
    """Return actuated/scalar joints in the official generalized-state order."""
    _asset(robot)
    return _JOINTS[robot]


def reference_profile(robot: str) -> dict[str, Any]:
    """Return source identity, state dimensions, controls and default reset state."""
    _asset(robot)
    return {
        "robot": robot,
        "dt": 1 / 60,
        "source_commit": _SOURCE_COMMIT,
        "asset_sha256": reference_asset_hash(robot),
        "joint_names": list(reference_joint_names(robot)),
        "q_dim": 2 if robot == "cartpole" else 15,
        "qd_dim": 2 if robot == "cartpole" else 14,
        "default_q": (
            [0.0, 0.0]
            if robot == "cartpole"
            else [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                -1.0,
                0.0,
                -1.0,
                0.0,
                1.0,
            ]
        ),
        "default_qd": [0.0] * (2 if robot == "cartpole" else 14),
        "effort_limits": [1500.0, 0.0] if robot == "cartpole" else [400.0] * 8,
        "codec": {
            "robot": robot,
            "base_com": [0.0, 0.0, 0.0],
            "continuous_q_indices": [1] if robot == "cartpole" else [],
        },
        "teacher_profile": "current_newton_import_generalized_effort",
        "source_actuators_imported": False,
        "source_solver_options_imported": False,
    }


def _import_reference(robot: str) -> Any:
    """Interpret the pinned asset with upstream environment import defaults."""
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    if robot == "cartpole":
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.01, limit_ke=1e4, limit_kd=10.0
        )
        builder.add_urdf(
            str(_asset(robot)),
            xform=wp.transform(
                (0.0, 0.0, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
            ),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
        )
    else:
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.05, limit_ke=1e3, limit_kd=1e2
        )
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=4e4, kd=1e2, kf=3e3, mu=0.75
        )
        builder.add_mjcf(
            str(_asset(robot)),
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_names=["floor", "ground"],
        )
    return builder


def _pose(transform: Any) -> np.ndarray:
    import warp as wp

    value = np.asarray(transform, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = np.asarray(
        wp.quat_to_matrix(wp.quat(*value[3:])), dtype=np.float32
    ).reshape(3, 3)
    pose[:3, 3] = value[:3]
    return pose


def make_reference_articulation(robot: str) -> ArticulationDesc:
    """Build an independent public descriptor with imported physical properties.

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

    builder = _import_reference(robot)
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
