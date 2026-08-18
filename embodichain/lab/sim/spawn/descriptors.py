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

This module is deliberately independent of the active physics backend.  It
translates one EmbodiChain configuration into a canonical descriptor carrying
both the common physics values and the optional backend extension blocks.  The
selected :mod:`dexsim.spawn` adapter remains the only component that chooses
between PhysX and Newton.

Articulation joint and link names are resolved by the normal DexSim adapter
finalization, not by a second source parser in EmbodiChain. Configuration that
depends on those names is applied directly from the EmbodiChain config after
the facade binds to the finalized result.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING, fields
import math
import os
from typing import TYPE_CHECKING

import numpy as np
from dexsim.spawn import (
    ArticulationDesc,
    ClothObjectDesc,
    CollisionApproximation,
    CollisionDesc,
    DexsimCollisionDesc,
    DexsimPhysicsDesc,
    GeometryDesc,
    MaterialDesc,
    NewtonCollisionDesc,
    ObjectDesc,
    RenderDesc,
    RigidBodyPhysicsDesc,
    SoftObjectDesc,
)
from dexsim.types import ActorType

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ClothObjectCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
    SoftObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg, SphereCfg
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.material import VisualMaterialCfg

__all__ = [
    "articulation_desc_from_cfg",
    "cloth_desc_from_cfg",
    "rigid_desc_from_cfg",
    "soft_desc_from_cfg",
]


def rigid_desc_from_cfg(
    cfg: RigidObjectCfg,
    *,
    per_env: bool = True,
) -> tuple[ObjectDesc, dict[str, MaterialDesc]]:
    """Translate a rigid-object config into a DexSim Spawn descriptor."""
    uid = _required_uid(cfg.uid, "Rigid object")
    if isinstance(cfg.shape, MeshCfg) and _is_usd_path(cfg.shape.fpath):
        raise NotImplementedError(
            "USD files describe typed scenes; use rigid_desc_from_usd() to "
            "select the sole rigid object."
        )

    geometry, approximation, max_hulls = _compile_geometry(cfg)
    material_ref, material_entry = _compile_visual_material(
        uid, cfg.shape.visual_material
    )
    collision = CollisionDesc.from_geometry(
        geometry,
        approximation=approximation,
    )
    collision.enable_collision = bool(cfg.attrs.enable_collision)
    collision.decomp_max_hulls = max_hulls
    collision.dexsim = _compile_dexsim_collision(cfg.attrs)
    collision.newton = _compile_newton_collision(
        cfg.attrs,
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
        renders=[RenderDesc.from_geometry(geometry, material_ref=material_ref)],
        collisions=[collision],
        physics=_compile_rigid_physics(cfg.attrs, cfg.body_type),
        per_env=per_env,
        body_scale=_vector3(cfg.body_scale, field_name="body_scale"),
    )
    materials = {} if material_entry is None else {material_entry[0]: material_entry[1]}
    return descriptor, materials


def soft_desc_from_cfg(
    cfg: SoftObjectCfg,
    *,
    per_env: bool = True,
) -> tuple[SoftObjectDesc, dict[str, MaterialDesc]]:
    """Translate a soft-object config into a DexSim Spawn descriptor."""
    uid = _required_uid(cfg.uid, "Soft object")
    if _is_missing(cfg.shape.fpath) or not str(cfg.shape.fpath).strip():
        raise ValueError("SoftObjectCfg.shape.fpath must be a non-empty path.")
    geometry = GeometryDesc.mesh(file_path=str(cfg.shape.fpath), segment_name=uid)
    material_ref, material_entry = _compile_visual_material(
        uid, cfg.shape.visual_material
    )
    descriptor = SoftObjectDesc(
        name=uid,
        pose=_pose_from_cfg(cfg),
        renders=[RenderDesc.from_geometry(geometry, material_ref=material_ref)],
        voxel_config=cfg.voxel_attr.attr(),
        body_attr=cfg.physical_attr.attr(),
        per_env=per_env,
    )
    materials = {} if material_entry is None else {material_entry[0]: material_entry[1]}
    return descriptor, materials


def cloth_desc_from_cfg(
    cfg: ClothObjectCfg,
    *,
    per_env: bool = True,
) -> tuple[ClothObjectDesc, dict[str, MaterialDesc]]:
    """Translate a cloth-object config into a DexSim Spawn descriptor."""
    uid = _required_uid(cfg.uid, "Cloth object")
    if _is_missing(cfg.shape.fpath) or not str(cfg.shape.fpath).strip():
        raise ValueError("ClothObjectCfg.shape.fpath must be a non-empty path.")
    geometry = GeometryDesc.mesh(file_path=str(cfg.shape.fpath), segment_name=uid)
    material_ref, material_entry = _compile_visual_material(
        uid, cfg.shape.visual_material
    )
    descriptor = ClothObjectDesc(
        name=uid,
        pose=_pose_from_cfg(cfg),
        renders=[RenderDesc.from_geometry(geometry, material_ref=material_ref)],
        body_attr=cfg.physical_attr.attr(),
        per_env=per_env,
    )
    materials = {} if material_entry is None else {material_entry[0]: material_entry[1]}
    return descriptor, materials


def articulation_desc_from_cfg(
    cfg: ArticulationCfg,
    *,
    per_env: bool = True,
    source_path: str | None = None,
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
    if cfg.use_usd_properties:
        logger.log_warning(
            "ArticulationCfg.use_usd_properties only applies to USD sources and "
            "is ignored for URDF articulations."
        )
    if cfg.min_position_iters != 4 or cfg.min_velocity_iters != 1:
        logger.log_warning(
            "Per-articulation solver iteration counts are not exposed by the "
            "backend-neutral Spawn facade and were not applied."
        )

    return ArticulationDesc(
        name=_articulation_uid(cfg.uid, str(path)),
        pose=_pose_from_cfg(cfg),
        path=str(path),
        urdf_path=str(path),
        fixed_base=bool(cfg.fix_base),
        enable_self_collision=not bool(cfg.disable_self_collision),
        urdf_fix_root_link=bool(cfg.fix_base),
        per_env=per_env,
        body_scale=_vector3(cfg.body_scale, field_name="body_scale"),
        newton_collision=_compile_newton_collision(cfg.attrs),
    )


def _compile_rigid_physics(
    attrs: RigidBodyAttributesCfg,
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

    if attrs.mass is not None and attrs.mass < 0:
        raise ValueError("Rigid-body mass cannot be negative.")
    if attrs.mass == 0 and (attrs.density is None or attrs.density <= 0):
        raise ValueError("Rigid-body density must be positive when mass is zero.")

    mass = float(attrs.mass) if attrs.mass is not None and attrs.mass > 0 else None
    density = (
        float(attrs.density)
        if mass is None and attrs.density is not None and attrs.density > 0
        else None
    )
    return RigidBodyPhysicsDesc(
        actor_type=actor_type,
        mass=mass,
        density=density,
        dexsim=DexsimPhysicsDesc(
            linear_damping=float(attrs.linear_damping),
            angular_damping=float(attrs.angular_damping),
            max_linear_velocity=float(attrs.max_linear_velocity),
            max_angular_velocity=float(attrs.max_angular_velocity),
            max_depenetration_velocity=float(attrs.max_depenetration_velocity),
            enable_ccd=bool(attrs.enable_ccd),
            min_position_iters=int(attrs.min_position_iters),
            min_velocity_iters=int(attrs.min_velocity_iters),
            sleep_threshold=float(attrs.sleep_threshold),
        ),
    )


def _compile_dexsim_collision(
    attrs: RigidBodyAttributesCfg,
) -> DexsimCollisionDesc:
    return DexsimCollisionDesc(
        dynamic_friction=float(attrs.dynamic_friction),
        static_friction=float(attrs.static_friction),
        restitution=float(attrs.restitution),
        contact_offset=float(attrs.contact_offset),
        rest_offset=float(attrs.rest_offset),
    )


def _compile_newton_collision(
    attrs: RigidBodyAttributesCfg,
    *,
    sdf_resolution: int = 0,
) -> NewtonCollisionDesc:
    # ``None`` means "leave the backend default untouched". Initializing every
    # field avoids accidentally authoring NewtonCollisionDesc's convenience
    # defaults when the EmbodiChain Newton sub-config did not set them.
    values = {field.name: None for field in fields(NewtonCollisionDesc)}
    if attrs.newton is not None:
        for name in values:
            if hasattr(attrs.newton, name):
                values[name] = getattr(attrs.newton, name)
    if "mu" in values:
        values["mu"] = float(attrs.dynamic_friction)
    if "restitution" in values:
        values["restitution"] = float(attrs.restitution)
    if sdf_resolution > 0:
        if "force_sdf" in values:
            values["force_sdf"] = True
        if values["sdf_max_resolution"] is None:
            values["sdf_max_resolution"] = int(sdf_resolution)
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

        option = shape.load_option
        if any(
            (
                option.rebuild_normals,
                option.rebuild_tangent,
                option.rebuild_3rdnormal,
                option.rebuild_3rdtangent,
                option.smooth != -1.0,
            )
        ):
            logger.log_warning(
                "Mesh LoadOption is not represented by ObjectDesc; the Spawn "
                "adapter will use its default mesh loading policy."
            )
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
                "sdf_max_resolution, but the PhysX descriptor does not expose "
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
