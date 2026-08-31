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
component that chooses between the Default and Newton backends. When supplied, the active
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
import warnings
from typing import TYPE_CHECKING

import numpy as np
from dexsim.spawn import (
    ArticulationDesc,
    ClothDesc,
    ClothPhysicsDesc,
    CollisionApproximation,
    CollisionDesc,
    DexsimClothPhysicsDesc,
    DexsimCollisionDesc,
    DexsimJointDesc,
    DexsimPhysicsDesc,
    DexsimSoftBodyPhysicsDesc,
    GeometryDesc,
    MaterialDesc,
    NewtonCollisionDesc,
    NewtonJointDesc,
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
    _normalize_joint_target_mode,
    ArticulationCfg,
    ClothObjectCfg,
    CollisionPropertiesCfg,
    DefaultCollisionPropertiesCfg,
    DefaultRigidBodyPropertiesCfg,
    MassPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonRigidBodyMaterialCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    SoftObjectCfg,
    SurfaceDeformableObjectCfg,
    VolumeDeformableObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, MeshCollisionCfg, SphereCfg
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
    recompute_inertia: bool | None = None
    default_rigid_props: dict[str, object] = field(default_factory=dict)
    collision_enabled: bool | None = None
    contact_offset: float | None = None
    rest_offset: float | None = None
    default_collision_props: dict[str, object] = field(default_factory=dict)
    newton_collision_props: dict[str, object] = field(default_factory=dict)
    material_props: dict[str, object] = field(default_factory=dict)
    newton_material_props: dict[str, object] = field(default_factory=dict)

    def merged(self, override: _RigidPhysicsSpec) -> _RigidPhysicsSpec:
        """Return ``override`` layered onto this spec using non-None values."""
        result = _RigidPhysicsSpec(
            mass_props=dict(self.mass_props),
            recompute_inertia=self.recompute_inertia,
            default_rigid_props=dict(self.default_rigid_props),
            collision_enabled=self.collision_enabled,
            contact_offset=self.contact_offset,
            rest_offset=self.rest_offset,
            default_collision_props=dict(self.default_collision_props),
            newton_collision_props=dict(self.newton_collision_props),
            material_props=dict(self.material_props),
            newton_material_props=dict(self.newton_material_props),
        )
        for name in (
            "mass_props",
            "default_rigid_props",
            "default_collision_props",
            "newton_collision_props",
            "material_props",
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
        if override.recompute_inertia is not None:
            result.recompute_inertia = override.recompute_inertia
        if override.collision_enabled is not None:
            result.collision_enabled = override.collision_enabled
        if override.contact_offset is not None:
            result.contact_offset = override.contact_offset
        if override.rest_offset is not None:
            result.rest_offset = override.rest_offset
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
    cfg: RigidBodyPhysicsCfg,
    *,
    newton_solver_type: str | None = None,
) -> _RigidPhysicsSpec:
    """Normalize grouped rigid-body configuration into one internal spec."""
    if isinstance(cfg, RigidBodyPhysicsCfg):
        mass_props = _configured_values(cfg.mass_props)
        recompute_inertia = mass_props.pop("recompute_inertia", None)
        if recompute_inertia is not None and not isinstance(
            recompute_inertia, (bool, np.bool_)
        ):
            raise TypeError("recompute_inertia must be a boolean or None.")
        spec = _RigidPhysicsSpec(
            mass_props=mass_props,
            recompute_inertia=(
                None if recompute_inertia is None else bool(recompute_inertia)
            ),
            collision_enabled=(
                None
                if cfg.collision_props is None
                else cfg.collision_props.collision_enabled
            ),
            contact_offset=(
                None
                if cfg.collision_props is None
                else cfg.collision_props.contact_offset
            ),
            rest_offset=(
                None if cfg.collision_props is None else cfg.collision_props.rest_offset
            ),
            material_props={
                name: getattr(cfg.material_props, name)
                for name in ("static_friction", "dynamic_friction", "restitution")
                if cfg.material_props is not None
                and getattr(cfg.material_props, name) is not None
            },
        )

        rigid_props = cfg.rigid_props
        if isinstance(rigid_props, DefaultRigidBodyPropertiesCfg):
            spec.default_rigid_props = _configured_values(rigid_props)
        elif rigid_props is not None:
            raise TypeError(
                f"Unsupported rigid_props type {type(rigid_props).__name__!r}."
            )

        collision_props = cfg.collision_props
        if isinstance(collision_props, DefaultCollisionPropertiesCfg):
            spec.default_collision_props = _configured_values(collision_props)
            for name in ("collision_enabled", "contact_offset", "rest_offset"):
                spec.default_collision_props.pop(name, None)
        elif isinstance(collision_props, NewtonCollisionPropertiesCfg):
            values = _configured_values(collision_props)
            for name in ("collision_enabled", "contact_offset", "rest_offset"):
                values.pop(name, None)
            spec.newton_collision_props = values
        elif (
            collision_props is not None
            and type(collision_props) is not CollisionPropertiesCfg
        ):
            raise TypeError(
                f"Unsupported collision_props type {type(collision_props).__name__!r}."
            )

        material_props = cfg.material_props
        if isinstance(material_props, NewtonRigidBodyMaterialCfg):
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

    raise AssertionError("Unhandled grouped rigid-body physics configuration.")


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
    collision.dexsim = _compile_default_collision(physics)
    collision.newton = _compile_newton_collision(
        physics,
        newton_solver_type=newton_solver_type,
        author_shape_defaults=True,
        mesh_collision=(
            cfg.shape.collision if isinstance(cfg.shape, MeshCfg) else None
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
    """Translate a volume-deformable config into a DexSim descriptor."""
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
    descriptor = SoftBodyDesc(
        name=uid,
        pose=_pose_from_cfg(cfg),
        mesh=RenderDesc.from_geometry(
            geometry,
            load_option=_compile_load_option(cfg.shape),
            material_ref=material_ref,
        ),
        physics=SoftBodyPhysicsDesc(
            volume_density=float(physical_attr.density),
            k_mu=youngs / (2.0 * (1.0 + poissons)),
            k_lambda=(youngs * poissons / ((1.0 + poissons) * (1.0 - 2.0 * poissons))),
            dexsim=DexsimSoftBodyPhysicsDesc(**_configured_values(physical_attr)),
        ),
        # DexSim's typed meshing contract currently exposes these three
        # source-mesh controls; maximal_edge_length has no Spawn equivalent.
        meshing=SoftBodyMeshingDesc(
            proxy_simplify_target=cfg.voxel_attr.triangle_simplify_target,
            proxy_remesh_resolution=cfg.voxel_attr.triangle_remesh_resolution,
            voxel_resolution=cfg.voxel_attr.simulation_mesh_resolution,
        ),
        per_env=per_env,
    )
    materials = {} if material_entry is None else {material_entry[0]: material_entry[1]}
    return descriptor, materials


def surface_deformable_desc_from_cfg(
    cfg: SurfaceDeformableObjectCfg,
    *,
    per_env: bool = True,
) -> tuple[ClothDesc, dict[str, MaterialDesc]]:
    """Translate a surface-deformable config into a DexSim descriptor."""
    uid = _required_uid(cfg.uid, "Surface deformable")
    if _is_missing(cfg.shape.fpath) or not str(cfg.shape.fpath).strip():
        raise ValueError(
            "SurfaceDeformableObjectCfg.shape.fpath must be a non-empty path."
        )
    geometry = GeometryDesc.mesh(file_path=str(cfg.shape.fpath), segment_name=uid)
    material_ref, material_entry = _compile_visual_material(
        uid, cfg.shape.visual_material
    )
    descriptor = ClothDesc(
        name=uid,
        pose=_pose_from_cfg(cfg),
        mesh=RenderDesc.from_geometry(
            geometry,
            load_option=_compile_load_option(cfg.shape),
            material_ref=material_ref,
        ),
        physics=ClothPhysicsDesc(
            surface_density=float(cfg.physical_attr.density),
            dexsim=DexsimClothPhysicsDesc(**_configured_values(cfg.physical_attr)),
        ),
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
        # inertia. MassPropertiesCfg can request geometry-based recomputation
        # after exact source names are available.
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


def _articulation_root_values(
    cfg: ArticulationCfg,
    *,
    fixed_base_default: bool = True,
    self_collision_default: bool = False,
) -> tuple[bool, bool]:
    """Resolve articulation-root values over source/import defaults."""
    props = cfg.root_props
    fixed_base = (
        fixed_base_default if props.fixed_base is None else bool(props.fixed_base)
    )
    self_collision_enabled = (
        self_collision_default
        if props.self_collision_enabled is None
        else bool(props.self_collision_enabled)
    )
    return fixed_base, self_collision_enabled


def _configured_articulation_overlay_fields(cfg: ArticulationCfg) -> list[str]:
    """Return physics overlay fields that preserve mode would ignore."""
    configured: list[str] = []
    if any(
        _configured_values(group)
        for group in (
            cfg.attrs.mass_props,
            cfg.attrs.rigid_props,
            cfg.attrs.collision_props,
            cfg.attrs.material_props,
        )
    ):
        configured.append("attrs")
    if cfg.link_attrs:
        configured.append("link_attrs")
    if _configured_values(cfg.joint_drive_props):
        configured.append("joint_drive_props")
    if cfg.qpos_limits is not None:
        configured.append("qpos_limits")
    return configured


def _compile_link_properties(
    physics: _RigidPhysicsSpec,
    *,
    newton_solver_type: str | None,
    author_newton_shape_defaults: bool,
) -> tuple[RigidBodyPhysicsDesc, CollisionDesc, bool]:
    collision = CollisionDesc(
        enable_collision=physics.collision_enabled,
        dexsim=_compile_default_collision(physics),
        newton=_compile_newton_collision(
            physics,
            newton_solver_type=newton_solver_type,
            author_shape_defaults=author_newton_shape_defaults,
        ),
    )
    return (
        _compile_rigid_physics(physics, "dynamic"),
        collision,
        bool(physics.recompute_inertia),
    )


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
        configured_fields = _configured_articulation_overlay_fields(cfg)
        if configured_fields:
            warnings.warn(
                "asset_physics_mode='preserve' ignores configured articulation "
                f"physics overlays: {', '.join(configured_fields)}. Set "
                "asset_physics_mode='overlay' to apply them.",
                UserWarning,
                stacklevel=2,
            )
        return desc
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
    link_properties = {link.name: default_link_properties for link in desc.links}

    claimed_links: dict[str, str] = {}
    link_names = [link.name for link in desc.links]
    for group_name, group in (cfg.link_attrs or {}).items():
        _, matched_names = resolve_matching_names(
            group.link_names_expr,
            link_names,
        )
        group_properties = _compile_link_properties(
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
            link_properties[link_name] = group_properties

    (
        joint_properties,
        joint_common,
        joint_limits,
        joint_target_modes,
    ) = _compile_joint_properties(
        desc,
        cfg,
        newton_solver_type=newton_solver_type,
    )

    # Commit only after every regex, value, and limit has been validated. Each
    # source-resolved item receives one exact-name update.
    for link_name, (
        rigid_body,
        collision,
        recompute_inertia,
    ) in link_properties.items():
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
            replace_inertial=recompute_inertia,
        )
    for joint_name, (default_desc, newton_desc) in joint_properties.items():
        lower_limit, upper_limit = joint_limits.get(joint_name, (None, None))
        common = joint_common[joint_name]
        desc.set_joint_properties(
            joint_name,
            lower_limit=lower_limit,
            upper_limit=upper_limit,
            effort_limit=common.get("effort_limit"),
            velocity_limit=common.get("velocity_limit"),
            armature=common.get("armature"),
            dexsim=default_desc,
            newton=newton_desc,
            newton_target_mode=joint_target_modes.get(joint_name),
        )
    return desc


def _compile_joint_properties(
    desc: ArticulationDesc,
    cfg: ArticulationCfg,
    *,
    newton_solver_type: str | None,
) -> tuple[
    dict[str, tuple[DexsimJointDesc, NewtonJointDesc]],
    dict[str, dict[str, float]],
    dict[str, tuple[object, object]],
    dict[str, int],
]:
    joint_names = [joint.name for joint in desc.joints]
    control_parts = getattr(cfg, "control_parts", None)
    target_mode_cfg: object = None
    drive_type: str | None = None
    if cfg.joint_drive_props is not None:
        target_mode_cfg, drive_type = cfg.joint_drive_props._resolve_modes()

    joint_target_modes: dict[str, int] = {}
    if target_mode_cfg is not None:
        matches = _joint_property_matches(
            target_mode_cfg,
            joint_names,
            property_name="target_mode",
            numeric_only=False,
            control_parts=control_parts,
        )
        for joint_name, value in matches:
            joint_target_modes[joint_name] = _normalize_joint_target_mode(value)

    # A scalar drive type remains the fallback for joints not selected by an
    # explicit target-mode rule. The established force drive activates both
    # position and velocity targets.
    if drive_type is not None:
        fallback_target_mode = 0 if drive_type == "none" else 3
        for joint_name in joint_names:
            joint_target_modes.setdefault(joint_name, fallback_target_mode)

    active_joints = [
        name for name, mode in joint_target_modes.items() if mode in {1, 2, 3}
    ]
    if drive_type == "none" and active_joints:
        raise ValueError(
            "drive_type='none' conflicts with an active joint target_mode; "
            "use target_mode='none' or 'effort'."
        )
    if newton_solver_type is not None and drive_type == "acceleration":
        if active_joints:
            raise NotImplementedError(
                "Newton Spawn does not have an exact acceleration-drive "
                "equivalent; use drive_type='force' or disable the drive."
            )

    default_drive_mode = {
        None: None,
        "force": DriveType.FORCE,
        "acceleration": DriveType.ACCELERATION,
        "none": DriveType.NONE,
    }[drive_type]
    joint_properties = {
        joint_name: (
            DexsimJointDesc(
                drive_mode=(
                    DriveType.NONE
                    if joint_target_modes.get(joint_name) in {0, 4}
                    else (
                        (
                            default_drive_mode
                            if default_drive_mode is not None
                            else DriveType.FORCE
                        )
                        if joint_target_modes.get(joint_name) in {1, 2, 3}
                        else None
                    )
                )
            ),
            NewtonJointDesc(),
        )
        for joint_name in joint_names
    }
    joint_common: dict[str, dict[str, float]] = {
        joint_name: {} for joint_name in joint_names
    }
    property_fields = {
        "stiffness": ("stiffness", "target_ke"),
        "damping": ("damping", "target_kd"),
        "friction": ("joint_friction", "friction"),
    }
    for property_name in ("stiffness", "damping"):
        if cfg.joint_drive_props is None:
            continue
        configured = getattr(cfg.joint_drive_props, property_name)
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
            default_desc, newton_desc = joint_properties[joint_name]
            default_field, newton_field = property_fields[property_name]
            setattr(default_desc, default_field, scalar)
            setattr(newton_desc, newton_field, scalar)

    if cfg.joint_drive_props is not None:
        source = cfg.joint_drive_props
        for property_name in (
            "max_effort",
            "max_velocity",
            "friction",
            "armature",
        ):
            configured = getattr(source, property_name)
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
                        f"Articulation joint rule for {joint_name!r} and "
                        f"{property_name!r} must contain a numeric value."
                    )
                scalar = float(value)
                default_desc, newton_desc = joint_properties[joint_name]
                if property_name == "armature":
                    joint_common[joint_name]["armature"] = scalar
                elif property_name == "max_effort":
                    default_desc.max_force = scalar
                    joint_common[joint_name]["effort_limit"] = scalar
                elif property_name == "max_velocity":
                    default_desc.max_velocity = scalar
                    joint_common[joint_name]["velocity_limit"] = scalar
                else:
                    default_field, newton_field = property_fields[property_name]
                    setattr(default_desc, default_field, scalar)
                    setattr(newton_desc, newton_field, scalar)

    # Solvers that ignore Newton's target-mode enum still consume drive gains.
    # Masking inactive components makes NONE, EFFORT, and VELOCITY deterministic
    # across the currently supported solver set.
    for joint_name, target_mode in joint_target_modes.items():
        default_desc, newton_desc = joint_properties[joint_name]
        if target_mode in {0, 4}:
            default_desc.stiffness = 0.0
            default_desc.damping = 0.0
            newton_desc.target_ke = 0.0
            newton_desc.target_kd = 0.0
        elif target_mode == 2:
            default_desc.stiffness = 0.0
            newton_desc.target_ke = 0.0

    normalized_solver = (
        None
        if newton_solver_type is None
        else newton_solver_type.replace("-", "_").lower()
    )
    if normalized_solver not in {None, "mujoco_warp", "mjwarp"} and any(
        mode == 1 for mode in joint_target_modes.values()
    ):
        warnings.warn(
            f"Newton solver {newton_solver_type!r} does not consume "
            "joint_target_mode. POSITION is emulated with its configured "
            "gains and assumes the velocity target remains zero.",
            UserWarning,
            stacklevel=3,
        )

    joint_limits = _compile_joint_limits(desc, cfg)

    return joint_properties, joint_common, joint_limits, joint_target_modes


def _joint_limit_array(value: object) -> np.ndarray:
    """Convert a tensor/array/sequence limit value to a CPU NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _compile_joint_limits(
    desc: ArticulationDesc,
    cfg: ArticulationCfg,
) -> dict[str, tuple[object, object]]:
    """Compile regex or flattened-DOF joint limits before backend build."""
    joint_limits: dict[str, tuple[object, object]] = {}
    if cfg.qpos_limits is None:
        return joint_limits

    joint_names = [joint.name for joint in desc.joints]
    if isinstance(cfg.qpos_limits, dict):
        indices, _, values = resolve_matching_names_values(
            cfg.qpos_limits,
            joint_names,
        )
        for index, limits in zip(indices, values):
            limit_values = _joint_limit_array(limits).reshape(-1)
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
        return joint_limits

    dof_joints = [joint for joint in desc.joints if joint.dof_count > 0]
    dof_count = sum(joint.dof_count for joint in dof_joints)
    limit_values = _joint_limit_array(cfg.qpos_limits)
    expected_shape = (dof_count, 2)
    if tuple(limit_values.shape) != expected_shape:
        raise ValueError(
            "Array qpos_limits must have flattened source-resolved DOF shape "
            f"{expected_shape}, got {tuple(limit_values.shape)}."
        )
    if not np.isfinite(limit_values).all():
        raise ValueError("Array qpos_limits must contain only finite values.")
    if np.any(limit_values[:, 0] > limit_values[:, 1]):
        raise ValueError(
            "Array qpos_limits contains a lower limit greater than its upper limit."
        )

    dof_start = 0
    for joint in dof_joints:
        dof_stop = dof_start + joint.dof_count
        joint_values = limit_values[dof_start:dof_stop]
        if joint.dof_count == 1:
            lower_limit: object = float(joint_values[0, 0])
            upper_limit: object = float(joint_values[0, 1])
        else:
            lower_limit = joint_values[:, 0].copy()
            upper_limit = joint_values[:, 1].copy()
        joint_limits[joint.name] = (lower_limit, upper_limit)
        dof_start = dof_stop
    return joint_limits


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
    if inertia is not None and physics.recompute_inertia:
        raise ValueError(
            "Rigid-body inertia cannot be explicit when recompute_inertia is true."
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

    if physics.default_rigid_props:
        default_values = {item.name: None for item in fields(DexsimPhysicsDesc)}
        default_values.update(physics.default_rigid_props)
        default_desc = DexsimPhysicsDesc(**default_values)
    else:
        default_desc = None
    return RigidBodyPhysicsDesc(
        actor_type=actor_type,
        mass=mass,
        density=density,
        inertia=inertia,
        com_position=com_position,
        com_quaternion=com_quaternion,
        dexsim=default_desc,
        newton=None,
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


def _common_collision_envelope(
    physics: _RigidPhysicsSpec,
) -> tuple[float | None, float | None]:
    """Validate and return the portable contact/rest envelope."""

    def optional_float(value: object | None, field_name: str) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} must be a finite number.") from exc
        if not math.isfinite(result):
            raise ValueError(f"{field_name} must be finite.")
        return result

    contact_offset = optional_float(physics.contact_offset, "contact_offset")
    rest_offset = optional_float(physics.rest_offset, "rest_offset")
    if contact_offset is not None and contact_offset < 0.0:
        raise ValueError("contact_offset must be non-negative.")
    if (
        contact_offset is not None
        and rest_offset is not None
        and contact_offset < rest_offset
    ):
        raise ValueError("contact_offset must be no smaller than rest_offset.")
    return contact_offset, rest_offset


def _compile_default_collision(
    physics: _RigidPhysicsSpec,
) -> DexsimCollisionDesc | None:
    values = dict(physics.material_props)
    contact_offset, rest_offset = _common_collision_envelope(physics)
    if contact_offset is not None:
        values["contact_offset"] = contact_offset
    if rest_offset is not None:
        values["rest_offset"] = rest_offset
    values.update(physics.default_collision_props)
    if not values:
        return None
    configured = {item.name: None for item in fields(DexsimCollisionDesc)}
    configured.update(values)
    return DexsimCollisionDesc(**configured)


def _compile_newton_collision(
    physics: _RigidPhysicsSpec,
    *,
    mesh_collision: MeshCollisionCfg | None = None,
    newton_solver_type: str | None = None,
    author_shape_defaults: bool = False,
) -> NewtonCollisionDesc | None:
    # Keep partial descriptors sparse for source overlays. Once a newly authored
    # shape has a Newton override, fill the Spawn margin/gap defaults because a
    # non-None descriptor suppresses DexSim's descriptor factory defaults.
    values = {field.name: None for field in fields(NewtonCollisionDesc)}
    contact_offset, rest_offset = _common_collision_envelope(physics)
    native_margin = physics.newton_collision_props.get("margin")
    native_gap = physics.newton_collision_props.get("gap")
    if rest_offset is not None:
        values["margin"] = rest_offset
    if contact_offset is not None and native_gap is None:
        effective_margin = native_margin if native_margin is not None else rest_offset
        if effective_margin is None:
            if newton_solver_type is not None:
                raise ValueError(
                    "Newton requires rest_offset (or a native margin) when a "
                    "portable contact_offset is configured."
                )
        else:
            try:
                gap = contact_offset - float(effective_margin)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Newton collision margin must be a finite number."
                ) from exc
            if not math.isfinite(gap):
                raise ValueError("Newton collision margin must be finite.")
            if gap < 0.0:
                raise ValueError(
                    "Newton collision margin must be no larger than contact_offset."
                )
            values["gap"] = gap
    values.update(physics.newton_collision_props)
    if mesh_collision is not None and mesh_collision.approximation == "sdf":
        values["force_sdf"] = True
        for field_name in (
            "is_hydroelastic",
            "sdf_narrow_band_range",
            "sdf_target_voxel_size",
            "sdf_texture_format",
            "sdf_padding",
        ):
            value = getattr(mesh_collision, field_name)
            if value is not None:
                values[field_name] = value
        if mesh_collision.sdf_resolution is not None:
            values["sdf_max_resolution"] = int(mesh_collision.sdf_resolution)
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
        collision_cfg = shape.collision or MeshCollisionCfg()
        approximation = {
            "convex_hull": CollisionApproximation.CONVEX_HULL,
            "convex_decomposition": CollisionApproximation.CONVEX_DECOMPOSITION,
            "triangle_mesh": CollisionApproximation.NONE,
            "sdf": CollisionApproximation.SDF,
        }[collision_cfg.approximation]
        max_hulls = collision_cfg.max_hulls or 1
        acd_method = collision_cfg.acd_method or "coacd"

        if collision_cfg.approximation == "triangle_mesh" and cfg.body_type != "static":
            raise ValueError(
                "triangle_mesh collision is supported only for static rigid objects."
            )

        if shape.compute_uv:
            logger.log_warning(
                "Mesh UV projection is not represented by GeometryDesc and was "
                "not applied."
            )
        if (
            collision_cfg.approximation == "convex_decomposition"
            and acd_method != "coacd"
        ):
            raise ValueError(
                "Spawn supports only acd_method='coacd' for convex_decomposition."
            )
        if collision_cfg.sdf_resolution is not None:
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
