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
"""Adapt neutral precision-assembly geometry to native simulation configs."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from embodichain.toolkits.assembly import AssemblyAsset, AssemblyBody, AssemblyGeometry

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.lab.sim.shapes import ShapePartCfg

__all__ = ["to_rigid_config", "to_rigid_configs"]


def _shape_part(geometry: AssemblyGeometry) -> ShapePartCfg:
    """Construct a native subshape without re-decomposing the body."""
    from embodichain.lab.sim.shapes import (
        CubeCfg,
        CylinderCfg,
        MeshCfg,
        MeshCollisionCfg,
        ShapePartCfg,
        SphereCfg,
    )
    from embodichain.lab.sim.material import VisualMaterialCfg

    if geometry.kind == "mesh":
        shape = MeshCfg(
            vertices=geometry.vertices,
            triangles=geometry.triangles,
            collision=MeshCollisionCfg(approximation="convex_hull"),
        )
    elif geometry.kind == "cylinder":
        shape = CylinderCfg(radius=geometry.size[0], half_height=geometry.size[1])
    elif geometry.kind == "box":
        shape = CubeCfg(size=[2 * x for x in geometry.size[:3]])
    elif geometry.kind == "sphere":
        shape = SphereCfg(radius=geometry.size[0])
    else:
        raise ValueError(f"Unsupported assembly geometry {geometry.kind!r}.")
    shape.visual_material = VisualMaterialCfg(base_color=list(geometry.rgba))
    return ShapePartCfg(
        shape=shape,
        position=tuple(geometry.position),
        quaternion=tuple(geometry.quaternion),
    )


def to_rigid_config(
    body: AssemblyBody,
    *,
    uid: str | None = None,
    source_contact: bool = True,
    backend: str = "newton",
) -> RigidObjectCfg:
    """Convert an authored body into one native compound configuration.

    Args:
        body: Source collision pieces, visuals, attachment and mass properties.
        uid: Optional simulation identity, defaulting to the source body name.
        source_contact: Preserve supported source contact parameters.
        backend: Explicit ``newton`` or ``default`` physics configuration.

    Returns:
        One rigid-body configuration retaining source-local collision pieces.

    Raises:
        ValueError: If geometry, attachment or contact overlays cannot be represented.
        NotImplementedError: If the selected backend cannot preserve source contacts.
    """
    from embodichain.lab.sim.cfg import (
        CollisionPropertiesCfg,
        MassPropertiesCfg,
        NewtonCollisionPropertiesCfg,
        NewtonRigidBodyMaterialCfg,
        RigidBodyMaterialCfg,
        RigidBodyPhysicsCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import CompoundShapeCfg
    from scipy.spatial.transform import Rotation

    if body.joint.kind == "hinge":
        raise ValueError(
            "Hinged assembly bodies require an articulation, not a free rigid body."
        )
    collisions = [g for g in body.geometries if g.collision]
    if not collisions:
        raise ValueError(f"Assembly body {body.name!r} has no collision geometry.")
    reference = collisions[0]
    for geom in collisions[1:]:
        if any(
            getattr(geom, key) != getattr(reference, key)
            for key in (
                "friction",
                "condim",
                "solref",
                "solimp",
                "priority",
                "solmix",
            )
        ):
            raise ValueError(
                "Per-piece contact properties cannot be represented by a body-wide physics overlay."
            )
    if backend not in {"newton", "default"}:
        raise ValueError("Assembly body backend must be newton or default.")
    if backend == "default" and source_contact:
        raise NotImplementedError(
            "Default cannot preserve MuJoCo solref/solimp or torsional/rolling "
            "friction. Set source_contact=False to explicitly measure its backend contact model."
        )
    collision_props = (
        NewtonCollisionPropertiesCfg(
            contact_offset=0.0,
            rest_offset=0.0,
            condim=reference.condim,
            priority=reference.priority,
            solref=tuple(reference.solref),
            solimp=tuple(reference.solimp),
            solmix=reference.solmix,
        )
        if backend == "newton" and source_contact
        else CollisionPropertiesCfg(contact_offset=0.0, rest_offset=0.0)
    )
    shape = CompoundShapeCfg(
        parts=[_shape_part(g) for g in collisions],
        visual_parts=[_shape_part(g) for g in body.geometries if g.visible],
    )
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(body.quaternion).as_matrix()
    transform[:3, 3] = body.position
    return RigidObjectCfg(
        uid=uid or body.name,
        shape=shape,
        init_local_pose=transform,
        body_type=body.body_type,
        attrs=RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(
                mass=body.mass if body.body_type == "dynamic" else None,
                inertia=body.inertia if body.body_type == "dynamic" else None,
                com_position=body.com if body.body_type == "dynamic" else None,
            ),
            collision_props=collision_props,
            material_props=(
                NewtonRigidBodyMaterialCfg(
                    dynamic_friction=reference.friction[0],
                    torsional_friction=reference.friction[1],
                    rolling_friction=reference.friction[2],
                )
                if backend == "newton"
                else RigidBodyMaterialCfg(
                    static_friction=reference.friction[0],
                    dynamic_friction=reference.friction[0],
                )
            ),
        ),
    )


def to_rigid_configs(
    asset: AssemblyAsset,
    *,
    prefix: str = "",
    source_contact: bool = True,
    backend: str = "newton",
) -> dict[str, RigidObjectCfg]:
    """Create native configurations for all free or fixed source bodies.

    Args:
        asset: Portable assembly with explicit body ownership.
        prefix: Prefix added to each simulation UID.
        source_contact: Preserve supported source contact parameters.
        backend: Explicit ``newton`` or ``default`` configuration.

    Returns:
        Source body names mapped to their native configurations.

    Raises:
        ValueError: If any body requires an articulation or unsupported properties.
    """
    return {
        body.name: to_rigid_config(
            body, uid=prefix + body.name, source_contact=source_contact, backend=backend
        )
        for body in asset.bodies
    }
