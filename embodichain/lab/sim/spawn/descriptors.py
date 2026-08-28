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
"""Translate EmbodiChain asset configurations into DexSim Spawn descriptors.

This module translates one EmbodiChain configuration into a canonical
descriptor carrying both the common physics values and the optional backend
extension blocks. The selected :mod:`dexsim.spawn` adapter remains the only
component that chooses between DexSim and Newton. When supplied, the active
Newton solver type only prevents common contact values from being authored to
a solver that cannot consume them.

Articulation source names come from the handles produced by normal backend
materialization. EmbodiChain owns regex/group selection, applies exact-name
typed properties, and explicitly rebuilds Newton once when those post-load
properties must be committed to its immutable model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, dataclass, field, fields
import math
import numbers
import os
from typing import TYPE_CHECKING

import numpy as np
from dexsim.spawn import (
    ArticulationDesc,
    ClothDesc,
    ClothPhysicsDesc,
    CollisionApproximation,
    CollisionDesc,
    DexsimCollisionDesc,
    DexsimJointDesc,
    DexsimPhysicsDesc,
    GeometryDesc,
    MaterialDesc,
    NewtonCollisionDesc,
    NewtonJointDesc,
    NewtonPhysicsDesc,
    ObjectDesc,
    RenderDesc,
    RigidBodyPhysicsDesc,
    SoftBodyDesc,
    SoftBodyMeshingDesc,
    SoftBodyPhysicsDesc,
)
from dexsim.spawn.descs import NEWTON_CONTACT_SOLVER_FIELDS
from dexsim.types import ActorType, DriveType, LoadOption as DexsimLoadOption

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ClothObjectCfg,
    CollisionPropertiesCfg,
    DexsimCollisionPropertiesCfg,
    DexsimRigidBodyMaterialCfg,
    DexsimRigidBodyPropertiesCfg,
    MassPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonRigidBodyMaterialCfg,
    NewtonRigidBodyPropertiesCfg,
    RigidBodyAttributesCfg,
    RigidBodyAttributesOverrideCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidBodyPropertiesCfg,
    RigidObjectCfg,
    SoftObjectCfg,
    SurfaceDeformableObjectCfg,
    VolumeDeformableObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, SphereCfg
from embodichain.utils import logger
from embodichain.utils.math import convert_quat
from embodichain.utils.string import (
    resolve_matching_names,
    resolve_matching_names_values,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.material import VisualMaterialCfg

__all__ = [
    "articulation_desc_from_cfg",
    "cloth_desc_from_cfg",
    "configure_articulation_desc",
    "rigid_desc_from_cfg",
    "soft_desc_from_cfg",
    "surface_deformable_desc_from_cfg",
    "volume_deformable_desc_from_cfg",
]


@dataclass
class _RigidPhysicsSpec:
    """Canonical, backend-partitioned rigid-physics values."""

    mass_props: dict[str, object] = field(default_factory=dict)
    dexsim_rigid_props: dict[str, object] = field(default_factory=dict)
    newton_rigid_props: dict[str, object] = field(default_factory=dict)
    collision_enabled: bool | None = None
    dexsim_collision_props: dict[str, object] = field(default_factory=dict)
    newton_collision_props: dict[str, object] = field(default_factory=dict)
    material_props: dict[str, object] = field(default_factory=dict)
    dexsim_material_props: dict[str, object] = field(default_factory=dict)
    newton_material_props: dict[str, object] = field(default_factory=dict)

    def merged(self, override: _RigidPhysicsSpec) -> _RigidPhysicsSpec:
        """Return ``override`` layered onto this spec using non-None values."""
        result = _RigidPhysicsSpec(
            mass_props=dict(self.mass_props),
            dexsim_rigid_props=dict(self.dexsim_rigid_props),
            newton_rigid_props=dict(self.newton_rigid_props),
            collision_enabled=self.collision_enabled,
            dexsim_collision_props=dict(self.dexsim_collision_props),
            newton_collision_props=dict(self.newton_collision_props),
            material_props=dict(self.material_props),
            dexsim_material_props=dict(self.dexsim_material_props),
            newton_material_props=dict(self.newton_material_props),
        )
        for name in (
            "mass_props",
            "dexsim_rigid_props",
            "newton_rigid_props",
            "dexsim_collision_props",
            "newton_collision_props",
            "material_props",
            "dexsim_material_props",
            "newton_material_props",
        ):
            getattr(result, name).update(getattr(override, name))
        if "mass" in override.mass_props:
            mass = float(override.mass_props["mass"])
            if mass > 0.0:
                result.mass_props.pop("density", None)
            elif mass == 0.0 and "density" in result.mass_props:
                result.mass_props.pop("mass", None)
        elif "density" in override.mass_props:
            result.mass_props.pop("mass", None)
        if override.collision_enabled is not None:
            result.collision_enabled = override.collision_enabled
        return result


def _configured_values(cfg: object | None) -> dict[str, object]:
    """Return non-None configclass fields without backend metadata."""
    if cfg is None:
        return {}
    return {
        item.name: value
        for item in fields(cfg)
        if (value := getattr(cfg, item.name)) is not None
    }


def _resolve_rigid_physics(
    cfg: RigidBodyAttributesCfg | RigidBodyAttributesOverrideCfg | RigidBodyPhysicsCfg,
    *,
    newton_solver_type: str | None = None,
) -> _RigidPhysicsSpec:
    """Normalize grouped and legacy rigid-body configs into one internal spec."""
    if isinstance(cfg, RigidBodyPhysicsCfg):
        spec = _RigidPhysicsSpec(
            mass_props=_configured_values(cfg.mass_props),
            collision_enabled=(
                None
                if cfg.collision_props is None
                else cfg.collision_props.collision_enabled
            ),
            material_props={
                name: getattr(cfg.material_props, name)
                for name in ("static_friction", "dynamic_friction", "restitution")
                if cfg.material_props is not None
                and getattr(cfg.material_props, name) is not None
            },
        )

        rigid_props = cfg.rigid_props
        if isinstance(rigid_props, DexsimRigidBodyPropertiesCfg):
            spec.dexsim_rigid_props = _configured_values(rigid_props)
        elif isinstance(rigid_props, NewtonRigidBodyPropertiesCfg):
            spec.newton_rigid_props = _configured_values(rigid_props)
        elif (
            rigid_props is not None and type(rigid_props) is not RigidBodyPropertiesCfg
        ):
            raise TypeError(
                f"Unsupported rigid_props type {type(rigid_props).__name__!r}."
            )

        collision_props = cfg.collision_props
        if isinstance(collision_props, DexsimCollisionPropertiesCfg):
            spec.dexsim_collision_props = _configured_values(collision_props)
            spec.dexsim_collision_props.pop("collision_enabled", None)
        elif isinstance(collision_props, NewtonCollisionPropertiesCfg):
            spec.newton_collision_props = _configured_values(collision_props)
            spec.newton_collision_props.pop("collision_enabled", None)
        elif (
            collision_props is not None
            and type(collision_props) is not CollisionPropertiesCfg
        ):
            raise TypeError(
                f"Unsupported collision_props type {type(collision_props).__name__!r}."
            )

        material_props = cfg.material_props
        if isinstance(material_props, DexsimRigidBodyMaterialCfg):
            spec.dexsim_material_props = _configured_values(material_props)
            for name in ("static_friction", "dynamic_friction", "restitution"):
                spec.dexsim_material_props.pop(name, None)
        elif isinstance(material_props, NewtonRigidBodyMaterialCfg):
            values = _configured_values(material_props)
            for name in ("static_friction", "dynamic_friction", "restitution"):
                values.pop(name, None)
            if "torsional_friction" in values:
                values["mu_torsional"] = values.pop("torsional_friction")
            if "rolling_friction" in values:
                values["mu_rolling"] = values.pop("rolling_friction")
            spec.newton_material_props = values
        elif (
            material_props is not None
            and type(material_props) is not RigidBodyMaterialCfg
        ):
            raise TypeError(
                f"Unsupported material_props type {type(material_props).__name__!r}."
            )
        return spec

    if not isinstance(cfg, (RigidBodyAttributesCfg, RigidBodyAttributesOverrideCfg)):
        raise TypeError(
            f"Unsupported rigid-body physics config {type(cfg).__name__!r}."
        )
    if newton_solver_type is not None:
        raise TypeError(
            f"{type(cfg).__name__} is a deprecated Default-backend-only "
            "configuration. Newton assets must use RigidBodyPhysicsCfg with "
            "grouped mass_props, rigid_props, collision_props, and "
            "material_props."
        )

    legacy_values = _configured_values(cfg)
    mass_names = {
        "mass",
        "density",
        "inertia",
        "com_position",
        "com_quaternion",
    }
    dexsim_rigid_names = {
        "angular_damping",
        "linear_damping",
        "max_depenetration_velocity",
        "sleep_threshold",
        "min_position_iters",
        "min_velocity_iters",
        "max_linear_velocity",
        "max_angular_velocity",
        "enable_ccd",
    }
    dexsim_collision_names = {"contact_offset", "rest_offset"}
    material_names = {"restitution", "dynamic_friction", "static_friction"}
    spec = _RigidPhysicsSpec(
        mass_props={
            name: legacy_values[name] for name in mass_names if name in legacy_values
        },
        dexsim_rigid_props={
            name: legacy_values[name]
            for name in dexsim_rigid_names
            if name in legacy_values
        },
        collision_enabled=legacy_values.get("enable_collision"),
        dexsim_collision_props={
            name: legacy_values[name]
            for name in dexsim_collision_names
            if name in legacy_values
        },
        material_props={
            name: legacy_values[name]
            for name in material_names
            if name in legacy_values
        },
    )
    return spec


def rigid_desc_from_cfg(
    cfg: RigidObjectCfg,
    *,
    per_env: bool = True,
    newton_solver_type: str | None = None,
) -> tuple[ObjectDesc, dict[str, MaterialDesc]]:
    """Translate a rigid-object config into a DexSim Spawn descriptor."""
    uid = _required_uid(cfg.uid, "Rigid object")
    if isinstance(cfg.shape, MeshCfg) and _is_usd_path(cfg.shape.fpath):
        raise NotImplementedError(
            "USD files describe typed scenes; use rigid_desc_from_usd() to "
            "select the sole rigid object."
        )

    physics = _resolve_rigid_physics(
        cfg.attrs,
        newton_solver_type=newton_solver_type,
    )
    geometry, approximation, max_hulls = _compile_geometry(cfg)
    material_ref, material_entry = _compile_visual_material(
        uid, cfg.shape.visual_material
    )
    collision = CollisionDesc.from_geometry(
        geometry,
        approximation=approximation,
    )
    collision.enable_collision = physics.collision_enabled
    collision.decomp_max_hulls = max_hulls
    collision.dexsim = _compile_dexsim_collision(physics)
    collision.newton = _compile_newton_collision(
        physics,
        newton_solver_type=newton_solver_type,
        author_shape_defaults=True,
        sdf_resolution=(
            _resolved_mesh_collision_settings(cfg)[2]
            if isinstance(cfg.shape, MeshCfg)
            else 0
        ),
    )
    collision.render_source_index = 0

    descriptor = ObjectDesc(
        name=uid,
        pose=_pose_from_cfg(cfg),
        renders=[
            RenderDesc.from_geometry(
                geometry,
                load_option=_compile_load_option(cfg.shape),
                material_ref=material_ref,
            )
        ],
        collisions=[collision],
        physics=_compile_rigid_physics(physics, cfg.body_type),
        per_env=per_env,
        body_scale=_vector3(cfg.body_scale, field_name="body_scale"),
    )
    materials = {} if material_entry is None else {material_entry[0]: material_entry[1]}
    return descriptor, materials


def volume_deformable_desc_from_cfg(
    cfg: VolumeDeformableObjectCfg,
    *,
    per_env: bool = True,
) -> tuple[SoftBodyDesc, dict[str, MaterialDesc]]:
    """Translate a volume deformable into a Newton particle-set descriptor."""
    uid = _required_uid(cfg.uid, "Volume deformable")
    if _is_missing(cfg.shape.fpath) or not str(cfg.shape.fpath).strip():
        raise ValueError(
            "VolumeDeformableObjectCfg.shape.fpath must be a non-empty path."
        )
    geometry = GeometryDesc.mesh(file_path=str(cfg.shape.fpath), segment_name=uid)
    material_ref, material_entry = _compile_visual_material(
        uid, cfg.shape.visual_material
    )
    physical_attr = cfg.physical_attr
    youngs = float(physical_attr.youngs)
    poissons = float(physical_attr.poissons)
    density = float(physical_attr.density)
    particle_radius = (
        None if cfg.particle_radius is None else float(cfg.particle_radius)
    )
    if not math.isfinite(youngs) or youngs < 0.0:
        raise ValueError("Soft-body youngs must be a finite non-negative value.")
    if not math.isfinite(poissons) or not -1.0 < poissons < 0.5:
        raise ValueError("Soft-body poissons must be finite and lie in (-1, 0.5).")
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("Soft-body density must be a finite positive value.")
    if particle_radius is not None and (
        not math.isfinite(particle_radius) or particle_radius <= 0.0
    ):
        raise ValueError(
            "Soft-body particle_radius must be finite and positive when set."
        )

    descriptor = SoftBodyDesc(
        name=uid,
        pose=_pose_from_cfg(cfg),
        mesh=RenderDesc.from_geometry(
            geometry,
            load_option=_compile_load_option(cfg.shape),
            material_ref=material_ref,
        ),
        physics=SoftBodyPhysicsDesc(
            volume_density=density,
            k_mu=youngs / (2.0 * (1.0 + poissons)),
            k_lambda=(youngs * poissons / ((1.0 + poissons) * (1.0 - 2.0 * poissons))),
            k_damp=float(physical_attr.elasticity_damping),
            surface_tri_ke=float(physical_attr.surface_tri_ke),
            surface_tri_ka=float(physical_attr.surface_tri_ka),
            surface_tri_kd=float(physical_attr.surface_tri_kd),
            surface_tri_drag=float(physical_attr.surface_tri_drag),
            surface_tri_lift=float(physical_attr.surface_tri_lift),
            add_surface_edges=bool(physical_attr.add_surface_edges),
            surface_edge_ke=float(physical_attr.surface_edge_ke),
            surface_edge_kd=float(physical_attr.surface_edge_kd),
        ),
        meshing=SoftBodyMeshingDesc(
            proxy_simplify_target=cfg.voxel_attr.triangle_simplify_target,
            proxy_remesh_resolution=cfg.voxel_attr.triangle_remesh_resolution,
            voxel_resolution=cfg.voxel_attr.simulation_mesh_resolution,
            voxel_num_relaxation_iters=cfg.voxel_attr.voxel_num_relaxation_iters,
            voxel_rel_min_tet_volume=cfg.voxel_attr.voxel_rel_min_tet_volume,
            voxel_surface_dist_ratio=cfg.voxel_attr.voxel_surface_dist_ratio,
            embedding_impl=cfg.voxel_attr.embedding_impl,
        ),
        particle_radius=particle_radius,
        validate_mesh=cfg.validate_mesh,
        per_env=per_env,
    )
    materials = {} if material_entry is None else {material_entry[0]: material_entry[1]}
    return descriptor, materials


def surface_deformable_desc_from_cfg(
    cfg: SurfaceDeformableObjectCfg,
    *,
    per_env: bool = True,
) -> tuple[ClothDesc, dict[str, MaterialDesc]]:
    """Translate a surface deformable into a Newton particle-set descriptor."""
    uid = _required_uid(cfg.uid, "Surface deformable")
    if _is_missing(cfg.shape.fpath) or not str(cfg.shape.fpath).strip():
        raise ValueError(
            "SurfaceDeformableObjectCfg.shape.fpath must be a non-empty path."
        )
    geometry = GeometryDesc.mesh(file_path=str(cfg.shape.fpath), segment_name=uid)
    material_ref, material_entry = _compile_visual_material(
        uid, cfg.shape.visual_material
    )
    physical_attr = cfg.physical_attr
    density = float(physical_attr.density)
    particle_radius = (
        None if cfg.particle_radius is None else float(cfg.particle_radius)
    )
    if not math.isfinite(density) or density <= 0.0:
        raise ValueError("Cloth density must be a finite positive value.")
    if particle_radius is not None and (
        not math.isfinite(particle_radius) or particle_radius <= 0.0
    ):
        raise ValueError("Cloth particle_radius must be finite and positive when set.")
    descriptor = ClothDesc(
        name=uid,
        pose=_pose_from_cfg(cfg),
        mesh=RenderDesc.from_geometry(
            geometry,
            load_option=_compile_load_option(cfg.shape),
            material_ref=material_ref,
        ),
        physics=ClothPhysicsDesc(
            surface_density=density,
            tri_ke=physical_attr.tri_ke,
            tri_ka=physical_attr.tri_ka,
            tri_kd=physical_attr.tri_kd,
            tri_drag=physical_attr.tri_drag,
            tri_lift=physical_attr.tri_lift,
            edge_ke=physical_attr.edge_ke,
            edge_kd=physical_attr.edge_kd,
            add_springs=bool(physical_attr.add_springs),
            spring_ke=physical_attr.spring_ke,
            spring_kd=physical_attr.spring_kd,
        ),
        particle_radius=particle_radius,
        validate_mesh=cfg.validate_mesh,
        per_env=per_env,
    )
    materials = {} if material_entry is None else {material_entry[0]: material_entry[1]}
    return descriptor, materials


def soft_desc_from_cfg(
    cfg: SoftObjectCfg,
    *,
    per_env: bool = True,
) -> tuple[SoftBodyDesc, dict[str, MaterialDesc]]:
    """Compatibility wrapper for :func:`volume_deformable_desc_from_cfg`."""
    return volume_deformable_desc_from_cfg(cfg, per_env=per_env)


def cloth_desc_from_cfg(
    cfg: ClothObjectCfg,
    *,
    per_env: bool = True,
) -> tuple[ClothDesc, dict[str, MaterialDesc]]:
    """Compatibility wrapper for :func:`surface_deformable_desc_from_cfg`."""
    return surface_deformable_desc_from_cfg(cfg, per_env=per_env)


def articulation_desc_from_cfg(
    cfg: ArticulationCfg,
    *,
    per_env: bool = True,
    source_path: str | None = None,
    newton_solver_type: str | None = None,
) -> ArticulationDesc:
    """Translate an articulation config into a DexSim Spawn descriptor."""
    path = source_path if source_path is not None else cfg.fpath
    if path is None or not str(path).strip():
        raise ValueError(
            "No articulation source path is available. Assemble the robot URDF "
            "before converting its configuration."
        )
    if _is_usd_path(path):
        raise NotImplementedError(
            "USD files describe typed scenes; use articulation_desc_from_usd() "
            "to select the sole articulation."
        )
    if cfg.resolve_asset_physics_mode() == "overlay":
        _validate_articulation_rigid_physics(
            cfg,
            newton_solver_type=newton_solver_type,
        )
    fixed_base, self_collision_enabled = _articulation_root_values(cfg)
    return ArticulationDesc(
        name=_articulation_uid(cfg.uid, str(path)),
        pose=_pose_from_cfg(cfg),
        path=str(path),
        urdf_path=str(path),
        fixed_base=fixed_base,
        enable_self_collision=self_collision_enabled,
        urdf_fix_root_link=fixed_base,
        # EmbodiChain's preserve/overlay policy starts from source-authored
        # inertia. Individual link groups can still request recomputation via
        # ``replace_inertial`` after exact source names are available.
        urdf_read_inertia=True,
        per_env=per_env,
        body_scale=_vector3(cfg.body_scale, field_name="body_scale"),
    )


def _validate_articulation_rigid_physics(
    cfg: ArticulationCfg,
    *,
    newton_solver_type: str | None,
) -> None:
    """Validate global and per-link physics before source materialization."""
    _resolve_rigid_physics(
        cfg.attrs,
        newton_solver_type=newton_solver_type,
    )
    for group in (cfg.link_attrs or {}).values():
        _resolve_rigid_physics(
            group.attrs,
            newton_solver_type=newton_solver_type,
        )


def _articulation_root_values(cfg: ArticulationCfg) -> tuple[bool, bool]:
    """Resolve grouped articulation-root values over legacy aliases."""
    props = cfg.articulation_props
    fixed_base = (
        bool(cfg.fix_base) if props.fixed_base is None else bool(props.fixed_base)
    )
    self_collision_enabled = (
        not bool(cfg.disable_self_collision)
        if props.self_collision_enabled is None
        else bool(props.self_collision_enabled)
    )
    return fixed_base, self_collision_enabled


def _compile_link_properties(
    physics: _RigidPhysicsSpec,
    *,
    newton_solver_type: str | None,
    author_newton_shape_defaults: bool,
) -> tuple[RigidBodyPhysicsDesc, CollisionDesc]:
    collision = CollisionDesc(
        enable_collision=physics.collision_enabled,
        dexsim=_compile_dexsim_collision(physics),
        newton=_compile_newton_collision(
            physics,
            newton_solver_type=newton_solver_type,
            author_shape_defaults=author_newton_shape_defaults,
        ),
    )
    return _compile_rigid_physics(physics, "dynamic"), collision


def configure_articulation_desc(
    desc: ArticulationDesc,
    cfg: ArticulationCfg,
    *,
    newton_solver_type: str | None = None,
) -> ArticulationDesc:
    """Apply one EmbodiChain config to exact source-resolved names.

    Regex/default/group semantics remain private to EmbodiChain. The DexSim
    descriptor receives only concrete link and joint properties.
    """
    if not desc.links:
        raise RuntimeError(
            f"Articulation source {desc.name!r} must be resolved before "
            "configuration."
        )
    if cfg.resolve_asset_physics_mode() == "preserve":
        return desc
    if (
        newton_solver_type is not None
        and cfg.drive_pros is not None
        and cfg.drive_pros.drive_type == "acceleration"
    ):
        raise NotImplementedError(
            "Newton Spawn does not have an exact acceleration-drive mode; "
            "use drive_type='force' or drive_type='none'."
        )

    default_physics = _resolve_rigid_physics(
        cfg.attrs,
        newton_solver_type=newton_solver_type,
    )
    author_newton_shape_defaults = not _is_usd_path(cfg.fpath)
    default_link_properties = _compile_link_properties(
        default_physics,
        newton_solver_type=newton_solver_type,
        author_newton_shape_defaults=author_newton_shape_defaults,
    )
    link_properties = {
        link.name: (*default_link_properties, False) for link in desc.links
    }

    claimed_links: dict[str, str] = {}
    link_names = [link.name for link in desc.links]
    for group_name, group in (cfg.link_attrs or {}).items():
        _, matched_names = resolve_matching_names(
            group.link_names_expr,
            link_names,
        )
        group_body, group_collision = _compile_link_properties(
            default_physics.merged(
                _resolve_rigid_physics(
                    group.attrs,
                    newton_solver_type=newton_solver_type,
                )
            ),
            newton_solver_type=newton_solver_type,
            author_newton_shape_defaults=author_newton_shape_defaults,
        )
        for link_name in matched_names:
            previous = claimed_links.get(link_name)
            if previous is not None:
                raise ValueError(
                    f"Link {link_name!r} matches both {previous!r} and "
                    f"{group_name!r}."
                )
            claimed_links[link_name] = group_name
            link_properties[link_name] = (
                group_body,
                group_collision,
                group.replace_inertial,
            )

    (
        joint_properties,
        joint_common,
        joint_limits,
        joint_target_modes,
    ) = _compile_joint_properties(desc, cfg)

    # Commit only after every regex, value, and limit has been validated. Each
    # source-resolved item receives one exact-name update.
    for link_name, (rigid_body, collision, replace_inertial) in link_properties.items():
        link = desc.get_link_desc(link_name)
        desc.set_link_properties(
            link_name,
            rigid_body=rigid_body,
            # The URDF resolver intentionally keeps source-owned collision
            # geometry outside LinkDesc.  An attribute-only CollisionDesc is
            # still required so the adapters can overlay properties onto the
            # native source shapes; it does not synthesize geometry.  Explicit
            # descriptors, including collisionless links, remain unchanged.
            collision=(
                collision if link.collisions or desc.urdf_path is not None else None
            ),
            replace_inertial=replace_inertial,
        )
    for joint_name, (dexsim, newton) in joint_properties.items():
        lower_limit, upper_limit = joint_limits.get(joint_name, (None, None))
        common = joint_common[joint_name]
        desc.set_joint_properties(
            joint_name,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            effort_limit=common.get("effort_limit"),
            velocity_limit=common.get("velocity_limit"),
            armature=common.get("armature"),
            dexsim=dexsim,
            newton=newton,
            newton_target_mode=joint_target_modes.get(joint_name),
        )
    return desc


def _compile_joint_properties(
    desc: ArticulationDesc,
    cfg: ArticulationCfg,
) -> tuple[
    dict[str, tuple[DexsimJointDesc, NewtonJointDesc]],
    dict[str, dict[str, float]],
    dict[str, tuple[float, float]],
    dict[str, int],
]:
    joint_names = [joint.name for joint in desc.joints]
    drive_type = None if cfg.drive_pros is None else cfg.drive_pros.drive_type
    if drive_type is None:
        dexsim_mode = None
        newton_mode = None
    else:
        try:
            dexsim_mode = {
                "force": DriveType.FORCE,
                "acceleration": DriveType.ACCELERATION,
                "none": DriveType.NONE,
            }[drive_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported joint drive type {drive_type!r}.") from exc
        newton_mode = {"force": 3, "none": 0}.get(drive_type)
    joint_properties = {
        joint_name: (
            DexsimJointDesc(drive_mode=dexsim_mode),
            NewtonJointDesc(),
        )
        for joint_name in joint_names
    }
    joint_target_modes = (
        {} if newton_mode is None else {name: newton_mode for name in joint_names}
    )
    joint_common: dict[str, dict[str, float]] = {
        joint_name: {} for joint_name in joint_names
    }
    property_fields = {
        "stiffness": ("stiffness", "target_ke"),
        "damping": ("damping", "target_kd"),
        "max_effort": ("max_force", "effort_limit"),
        "max_velocity": ("max_velocity", "velocity_limit"),
        "friction": ("joint_friction", "friction"),
    }
    control_parts = getattr(cfg, "control_parts", None)

    for property_name in (
        "stiffness",
        "damping",
        "max_effort",
        "max_velocity",
        "friction",
        "armature",
    ):
        if cfg.drive_pros is None:
            continue
        configured = getattr(cfg.drive_pros, property_name)
        if configured is None:
            continue
        matches = _joint_property_matches(
            configured,
            joint_names,
            property_name=property_name,
            control_parts=control_parts,
        )
        for joint_name, value in matches:
            if not isinstance(value, numbers.Number):
                raise TypeError(
                    f"Articulation drive rule for {joint_name!r} and "
                    f"{property_name!r} must contain a numeric value."
                )
            scalar = float(value)
            dexsim, newton = joint_properties[joint_name]
            if property_name == "armature":
                joint_common[joint_name]["armature"] = scalar
            elif property_name == "max_effort":
                dexsim.max_force = scalar
                joint_common[joint_name]["effort_limit"] = scalar
            elif property_name == "max_velocity":
                dexsim.max_velocity = scalar
                joint_common[joint_name]["velocity_limit"] = scalar
            else:
                dexsim_field, newton_field = property_fields[property_name]
                setattr(dexsim, dexsim_field, scalar)
                setattr(newton, newton_field, scalar)

    if isinstance(cfg.drive_pros, NewtonJointDrivePropertiesCfg):
        if cfg.drive_pros.target_mode is not None:
            matches = _joint_property_matches(
                cfg.drive_pros.target_mode,
                joint_names,
                property_name="target_mode",
                numeric_only=False,
                control_parts=control_parts,
            )
            for joint_name, value in matches:
                joint_target_modes[joint_name] = _normalize_newton_target_mode(value)

    joint_limits: dict[str, tuple[float, float]] = {}
    if isinstance(cfg.qpos_limits, dict):
        indices, _, values = resolve_matching_names_values(
            cfg.qpos_limits,
            joint_names,
        )
        for index, limits in zip(indices, values):
            limit_values = np.asarray(limits, dtype=np.float32).reshape(-1)
            if limit_values.size != 2:
                raise ValueError(
                    f"qpos_limits for {joint_names[index]!r} must contain "
                    "[lower, upper]."
                )
            lower_limit, upper_limit = map(float, limit_values)
            if not math.isfinite(lower_limit) or not math.isfinite(upper_limit):
                raise ValueError(
                    f"qpos_limits for {joint_names[index]!r} must be finite."
                )
            if lower_limit > upper_limit:
                raise ValueError(
                    f"qpos_limits for {joint_names[index]!r} has lower limit "
                    f"{lower_limit} greater than upper limit {upper_limit}."
                )
            joint_limits[joint_names[index]] = (lower_limit, upper_limit)

    return joint_properties, joint_common, joint_limits, joint_target_modes


def _joint_property_matches(
    configured: object,
    joint_names: list[str],
    *,
    property_name: str,
    numeric_only: bool = True,
    control_parts: dict[str, Sequence[str]] | None = None,
) -> list[tuple[str, object]]:
    """Resolve scalar, regex, and robot control-part drive rules."""
    scalar_types = (numbers.Number,) if numeric_only else (numbers.Number, str)
    if isinstance(configured, scalar_types):
        return [(name, configured) for name in joint_names]
    if isinstance(configured, dict):
        control_parts = control_parts or {}
        part_rules = {
            name: value for name, value in configured.items() if name in control_parts
        }
        direct_rules = {
            name: value
            for name, value in configured.items()
            if name not in control_parts
        }

        resolved: dict[str, object] = {}
        owners: dict[str, str] = {}
        for part_name, value in part_rules.items():
            expressions = list(control_parts[part_name])
            if not expressions:
                raise ValueError(f"Robot control part {part_name!r} has no joints.")
            indices, _, _ = resolve_matching_names_values(
                {expression: value for expression in expressions},
                joint_names,
            )
            for index in indices:
                joint_name = joint_names[index]
                previous = owners.get(joint_name)
                if previous is not None:
                    raise ValueError(
                        f"Joint {joint_name!r} is selected by both control "
                        f"parts {previous!r} and {part_name!r} for drive "
                        f"property {property_name!r}."
                    )
                resolved[joint_name] = value
                owners[joint_name] = part_name

        if direct_rules:
            indices, _, values = resolve_matching_names_values(
                direct_rules,
                joint_names,
            )
            # Exact/regex joint rules intentionally override a broader control
            # part rule, matching RobotCfg's public configuration contract.
            for index, value in zip(indices, values):
                resolved[joint_names[index]] = value
        return [(name, resolved[name]) for name in joint_names if name in resolved]
    expected = "number" if numeric_only else "string/integer"
    raise TypeError(
        f"Articulation drive property {property_name!r} must be a {expected} "
        f"or regex-to-{expected} mapping."
    )


def _normalize_newton_target_mode(value: object) -> int:
    """Normalize an EmbodiChain target-mode value to DexSim's integer enum."""
    if isinstance(value, str):
        normalized = value.replace("-", "_").lower()
        modes = {
            "none": 0,
            "position": 1,
            "velocity": 2,
            "position_velocity": 3,
        }
        if normalized not in modes:
            raise ValueError(
                f"Unsupported Newton joint target mode {value!r}; expected one "
                f"of {tuple(modes)}."
            )
        return modes[normalized]
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        mode = int(value)
        if 0 <= mode <= 3:
            return mode
        raise ValueError("Newton joint target-mode integers must be in [0, 3].")
    raise TypeError(
        "Newton joint target mode must be a string or an integer in [0, 3]."
    )


def _compile_rigid_physics(
    physics: _RigidPhysicsSpec,
    body_type: str,
) -> RigidBodyPhysicsDesc:
    actor_types = {
        "dynamic": ActorType.DYNAMIC,
        "kinematic": ActorType.KINEMATIC,
        "static": ActorType.STATIC,
    }
    try:
        actor_type = actor_types[body_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rigid body_type {body_type!r}; expected one of "
            f"{tuple(actor_types)}."
        ) from exc

    mass_value = physics.mass_props.get("mass")
    density_value = physics.mass_props.get("density")
    if mass_value is not None and float(mass_value) < 0:
        raise ValueError("Rigid-body mass cannot be negative.")
    if density_value is not None and float(density_value) <= 0:
        raise ValueError("Rigid-body density must be positive.")
    if mass_value == 0 and density_value is None:
        raise ValueError("Rigid-body density is required when mass is zero.")

    inertia = _rigid_array(
        physics.mass_props.get("inertia"),
        field_name="inertia",
        allowed_sizes=(3, 9),
    )
    com_position = _rigid_array(
        physics.mass_props.get("com_position"),
        field_name="com_position",
        allowed_sizes=(3,),
    )
    com_quaternion = _rigid_array(
        physics.mass_props.get("com_quaternion"),
        field_name="com_quaternion",
        allowed_sizes=(4,),
    )
    if inertia is not None:
        if mass_value is None or float(mass_value) <= 0:
            raise ValueError("Explicit rigid-body inertia requires a positive mass.")
        if inertia.size == 3 and (np.any(inertia <= 0.0) or np.allclose(inertia, 0.0)):
            raise ValueError(
                "Rigid-body inertia must contain positive principal moments."
            )
        if inertia.size == 9:
            inertia_matrix = inertia.reshape(3, 3)
            if not np.allclose(inertia_matrix, inertia_matrix.T, atol=1.0e-6):
                raise ValueError("Rigid-body inertia matrix must be symmetric.")
            if np.any(np.linalg.eigvalsh(inertia_matrix) <= 0.0):
                raise ValueError("Rigid-body inertia matrix must be positive definite.")
    if com_quaternion is not None:
        quaternion_norm = float(np.linalg.norm(com_quaternion))
        if quaternion_norm <= 1.0e-8:
            raise ValueError("Rigid-body com_quaternion cannot be zero.")
        com_quaternion = com_quaternion / quaternion_norm
        # DexSim descriptors use wxyz; EmbodiChain configuration uses xyzw.
        com_quaternion = convert_quat(com_quaternion, to="wxyz")

    if body_type != "static":
        mass = (
            float(mass_value)
            if mass_value is not None and float(mass_value) > 0
            else None
        )
        density = (
            float(density_value)
            if mass is None and density_value is not None and float(density_value) > 0
            else None
        )
    else:
        # Both backends ignore mass properties on static actors. Omitting them
        # also avoids a Newton build warning for the common default cfg.
        mass = None
        density = None
        inertia = None
        com_position = None
        com_quaternion = None

    if physics.dexsim_rigid_props:
        dexsim_values = {item.name: None for item in fields(DexsimPhysicsDesc)}
        dexsim_values.update(physics.dexsim_rigid_props)
        dexsim = DexsimPhysicsDesc(**dexsim_values)
    else:
        dexsim = None
    newton = (
        NewtonPhysicsDesc(**physics.newton_rigid_props)
        if physics.newton_rigid_props
        else None
    )
    return RigidBodyPhysicsDesc(
        actor_type=actor_type,
        mass=mass,
        density=density,
        inertia=inertia,
        com_position=com_position,
        com_quaternion=com_quaternion,
        dexsim=dexsim,
        newton=newton,
    )


def _rigid_array(
    value: object | None,
    *,
    field_name: str,
    allowed_sizes: tuple[int, ...],
) -> np.ndarray | None:
    """Validate and copy a rigid-body mass-property array."""
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size not in allowed_sizes or not np.all(np.isfinite(result)):
        expected = " or ".join(str(size) for size in allowed_sizes)
        raise ValueError(
            f"Rigid-body {field_name} must contain {expected} finite values."
        )
    return result.copy()


def _compile_dexsim_collision(
    physics: _RigidPhysicsSpec,
) -> DexsimCollisionDesc | None:
    values = dict(physics.material_props)
    values.update(physics.dexsim_collision_props)
    values.update(physics.dexsim_material_props)
    if not values:
        return None
    configured = {item.name: None for item in fields(DexsimCollisionDesc)}
    configured.update(values)
    return DexsimCollisionDesc(**configured)


def _compile_newton_collision(
    physics: _RigidPhysicsSpec,
    *,
    sdf_resolution: int = 0,
    newton_solver_type: str | None = None,
    author_shape_defaults: bool = False,
) -> NewtonCollisionDesc | None:
    # Keep partial descriptors sparse for source overlays. Once a newly authored
    # shape has a Newton override, fill the Spawn margin/gap defaults because a
    # non-None descriptor suppresses DexSim's descriptor factory defaults.
    values = {field.name: None for field in fields(NewtonCollisionDesc)}
    values.update(physics.newton_collision_props)
    values.update(physics.newton_material_props)
    dynamic_friction = physics.material_props.get("dynamic_friction")
    if dynamic_friction is not None:
        values["mu"] = float(dynamic_friction)
    solver_contact_fields = NEWTON_CONTACT_SOLVER_FIELDS.get(newton_solver_type)
    restitution = physics.material_props.get("restitution")
    if restitution is not None and (
        solver_contact_fields is None or "restitution" in solver_contact_fields
    ):
        values["restitution"] = float(restitution)
    if sdf_resolution > 0:
        if "force_sdf" in values:
            values["force_sdf"] = True
        if values["sdf_max_resolution"] is None:
            values["sdf_max_resolution"] = int(sdf_resolution)
    if all(value is None for value in values.values()):
        return None
    if author_shape_defaults:
        defaults = NewtonCollisionDesc()
        if values["margin"] is None:
            values["margin"] = defaults.margin
        if values["gap"] is None:
            values["gap"] = defaults.gap
    return NewtonCollisionDesc(**values)


def _compile_geometry(
    cfg: RigidObjectCfg,
) -> tuple[GeometryDesc, CollisionApproximation, int]:
    shape = cfg.shape
    if isinstance(shape, MeshCfg):
        if _is_missing(shape.fpath) or not str(shape.fpath).strip():
            raise ValueError("MeshCfg.fpath must be a non-empty path.")
        max_hulls, acd_method, sdf_resolution = _resolved_mesh_collision_settings(cfg)
        if sdf_resolution > 0:
            approximation = CollisionApproximation.SDF
        elif max_hulls > 1:
            approximation = CollisionApproximation.CONVEX_DECOMPOSITION
        else:
            approximation = CollisionApproximation.CONVEX_HULL

        if shape.compute_uv:
            logger.log_warning(
                "Mesh UV projection is not represented by GeometryDesc and was "
                "not applied."
            )
        if max_hulls > 1 and str(acd_method).lower() != "coacd":
            logger.log_warning(
                f"Spawn preserves max_convex_hull_num={max_hulls}, but does not "
                f"expose the requested ACD method {acd_method!r}."
            )
        if sdf_resolution > 0:
            logger.log_warning(
                "CollisionApproximation.SDF is preserved and Newton receives "
                "sdf_max_resolution, but the DexSim descriptor does not expose "
                "its cooking resolution."
            )
        return (
            GeometryDesc.mesh(
                file_path=str(shape.fpath), segment_name=cfg.uid or "mesh"
            ),
            approximation,
            max(1, max_hulls),
        )

    if isinstance(shape, CubeCfg):
        size = tuple(float(value) for value in shape.size)
        if len(size) != 3 or any(value <= 0 for value in size):
            raise ValueError("CubeCfg.size must contain three positive values.")
        return GeometryDesc.cube(size), CollisionApproximation.NONE, 1

    if isinstance(shape, SphereCfg):
        if shape.radius <= 0:
            raise ValueError("SphereCfg.radius must be positive.")
        return (
            GeometryDesc.sphere(float(shape.radius)),
            CollisionApproximation.NONE,
            1,
        )

    raise NotImplementedError(
        f"RigidObjectCfg shape {type(shape).__name__!r} is not supported by "
        "the Spawn converter; supported shapes are MeshCfg, CubeCfg, and SphereCfg."
    )


def _compile_load_option(shape: object) -> DexsimLoadOption | None:
    """Translate mesh import options without leaking EmbodiChain config types."""
    if not isinstance(shape, MeshCfg):
        return None
    source = shape.load_option
    option = DexsimLoadOption()
    option.rebuild_normals = bool(source.rebuild_normals)
    option.rebuild_tangent = bool(source.rebuild_tangent)
    option.rebuild_3rdnormal = bool(source.rebuild_3rdnormal)
    option.rebuild_3rdtangent = bool(source.rebuild_3rdtangent)
    option.smooth = float(source.smooth)
    return option


def _compile_visual_material(
    object_uid: str,
    cfg: VisualMaterialCfg | None,
) -> tuple[str | None, tuple[str, MaterialDesc] | None]:
    if cfg is None:
        return None, None
    key = str(cfg.uid or f"{object_uid}_material")
    base_color = tuple(float(value) for value in cfg.base_color)
    if len(base_color) != 4:
        raise ValueError("VisualMaterialCfg.base_color must be RGBA.")
    emissive_rgb = tuple(
        float(value) * float(cfg.emissive_intensity) for value in cfg.emissive
    )
    if len(emissive_rgb) != 3:
        raise ValueError("VisualMaterialCfg.emissive must be RGB.")
    desc = MaterialDesc(
        name=key,
        base_color=base_color,
        base_color_map=cfg.base_color_texture,
        normal_map=cfg.normal_texture,
        emissive=(*emissive_rgb, 1.0),
        roughness=float(cfg.roughness),
        roughness_map=cfg.roughness_texture,
        metallic=float(cfg.metallic),
        metallic_map=cfg.metallic_texture,
        ao_map=cfg.ao_texture,
        ior=float(cfg.ior),
    )
    return key, (key, desc)


def _resolved_mesh_collision_settings(
    cfg: RigidObjectCfg,
) -> tuple[int, str, int]:
    if not isinstance(cfg.shape, MeshCfg):
        return 1, "coacd", 0

    def first_value(values: Sequence[object], default: object) -> object:
        for value in values:
            if not _is_missing(value):
                return value
        return default

    max_hulls = int(
        first_value((cfg.max_convex_hull_num, cfg.shape.max_convex_hull_num), 1)
    )
    acd_method = str(first_value((cfg.acd_method, cfg.shape.acd_method), "coacd"))
    sdf_resolution = int(first_value((cfg.sdf_resolution, cfg.shape.sdf_resolution), 0))
    if max_hulls < 1:
        raise ValueError("max_convex_hull_num must be at least 1.")
    if sdf_resolution < 0:
        raise ValueError("sdf_resolution cannot be negative.")
    return max_hulls, acd_method, sdf_resolution


def _pose_from_cfg(cfg: object) -> np.ndarray:
    local_pose = getattr(cfg, "init_local_pose", None)
    if local_pose is not None:
        pose = np.asarray(local_pose, dtype=np.float32).reshape(4, 4).copy()
    else:
        position = _vector3(getattr(cfg, "init_pos"), field_name="init_pos")
        rotation_deg = _vector3(getattr(cfg, "init_rot"), field_name="init_rot")
        rx, ry, rz = np.deg2rad(rotation_deg)
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rot_x = np.array(
            ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)),
            dtype=np.float32,
        )
        rot_y = np.array(
            ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)),
            dtype=np.float32,
        )
        rot_z = np.array(
            ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )
        pose = np.eye(4, dtype=np.float32)
        # Match EmbodiChain's shared matrix_from_euler(..., "XYZ") contract
        # used by the legacy RigidObject reset path.
        pose[:3, :3] = rot_x @ rot_y @ rot_z
        pose[:3, 3] = position

    if not np.isfinite(pose).all():
        raise ValueError("init_local_pose must contain finite values.")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError("init_local_pose must be a homogeneous 4x4 transform.")
    return pose


def _vector3(value: object, *, field_name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size != 3 or not np.isfinite(result).all():
        raise ValueError(f"{field_name} must contain three finite values.")
    if field_name == "body_scale" and np.any(result <= 0):
        raise ValueError("body_scale values must be positive.")
    return result.copy()


def _required_uid(value: str | None, label: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{label} uid must be specified before Spawn conversion.")
    uid = str(value)
    if "/" in uid:
        raise ValueError(f"{label} uid cannot contain '/': {uid!r}.")
    return uid


def _articulation_uid(value: str | None, path: str | None) -> str:
    if value is not None and str(value).strip():
        return _required_uid(str(value), "Articulation")
    if path is None or not str(path).strip():
        raise ValueError(
            "Articulation uid is required when its source path is unresolved."
        )
    inferred = os.path.splitext(os.path.basename(str(path)))[0]
    return _required_uid(inferred, "Articulation")


def _is_usd_path(path: object) -> bool:
    return str(path).lower().endswith((".usd", ".usda", ".usdc"))


def _is_missing(value: object) -> bool:
    # ``@configclass`` deepcopy can create a distinct _MISSING_TYPE instance.
    return value is MISSING or isinstance(value, type(MISSING))
