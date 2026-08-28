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
"""Compatibility translation for EmbodiChain's singleton USD APIs."""

from __future__ import annotations

import os
from dataclasses import fields, replace
from typing import TypeVar

from dexsim.spawn import (
    ArticulationDesc,
    CollisionDesc,
    MaterialDesc,
    ObjectDesc,
    RenderDesc,
    RigidBodyPhysicsDesc,
)
from dexsim.types import ActorType

from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg
from embodichain.lab.sim.spawn.descriptors import (
    _compile_default_collision,
    _compile_newton_collision,
    _compile_rigid_physics,
    _compile_visual_material,
    _articulation_root_values,
    _pose_from_cfg,
    _required_uid,
    _resolve_rigid_physics,
    _validate_articulation_rigid_physics,
    _vector3,
)

__all__ = ["articulation_desc_from_usd", "rigid_desc_from_usd"]

_PropertyCfgT = TypeVar("_PropertyCfgT")


def _overlay_optional_properties(
    source: _PropertyCfgT | None,
    configured: _PropertyCfgT | None,
) -> _PropertyCfgT | None:
    """Overlay non-None dataclass fields without erasing source values."""
    if configured is None:
        return source
    if source is None:
        return configured
    for item in fields(configured):
        value = getattr(configured, item.name)
        if value is not None:
            setattr(source, item.name, value)
    return source


def _overlay_rigid_body_properties(
    source: RigidBodyPhysicsDesc | None,
    configured: RigidBodyPhysicsDesc,
) -> RigidBodyPhysicsDesc:
    """Merge a partial body config into properties parsed from USD."""
    if source is None:
        return configured
    source.actor_type = configured.actor_type
    source.dexsim = _overlay_optional_properties(source.dexsim, configured.dexsim)
    source.newton = _overlay_optional_properties(source.newton, configured.newton)
    if configured.mass is not None:
        source.mass = configured.mass
        source.density = None
    elif configured.density is not None:
        source.mass = None
        source.density = configured.density
    for name in ("inertia", "com_position", "com_quaternion"):
        value = getattr(configured, name)
        if value is not None:
            setattr(source, name, value)
    return source


def _overlay_collision_properties(
    source: CollisionDesc,
    configured: CollisionDesc,
) -> None:
    """Merge partial contact properties while retaining parsed geometry."""
    if configured.enable_collision is not None:
        source.enable_collision = configured.enable_collision
    source.dexsim = _overlay_optional_properties(source.dexsim, configured.dexsim)
    source.newton = _overlay_optional_properties(source.newton, configured.newton)


def rigid_desc_from_usd(
    cfg: RigidObjectCfg,
    *,
    per_env: bool = True,
    newton_solver_type: str | None = None,
) -> tuple[ObjectDesc, dict[str, MaterialDesc]]:
    """Select the sole rigid object in a USD stage."""
    uid = _required_uid(cfg.uid, "Rigid object")
    path = getattr(cfg.shape, "fpath", None)
    scene, desc = _parse_singleton(path, "mesh_objects", "rigid object")

    desc.name = uid
    desc.pose = _pose_from_cfg(cfg)
    desc.per_env = per_env
    materials = _namespace_materials(desc.renders, scene.materials, uid)

    if cfg.resolve_asset_physics_mode() == "preserve":
        if desc.physics is None:
            raise ValueError(f"USD rigid object {path!r} has no physics.")
        cfg.body_type = {
            ActorType.DYNAMIC: "dynamic",
            ActorType.KINEMATIC: "kinematic",
            ActorType.STATIC: "static",
        }[desc.physics.actor_type]
        cfg.body_scale = tuple(float(value) for value in desc.body_scale)
        return desc, materials

    physics = _resolve_rigid_physics(
        cfg.attrs,
        newton_solver_type=newton_solver_type,
    )
    configured_body = _compile_rigid_physics(physics, cfg.body_type)
    desc.physics = _overlay_rigid_body_properties(desc.physics, configured_body)
    desc.body_scale = _vector3(cfg.body_scale, field_name="body_scale")
    for collision in desc.collisions:
        _overlay_collision_properties(
            collision,
            CollisionDesc(
                enable_collision=physics.collision_enabled,
                dexsim=_compile_default_collision(physics),
                newton=_compile_newton_collision(
                    physics,
                    newton_solver_type=newton_solver_type,
                ),
            ),
        )

    material_ref, material_entry = _compile_visual_material(
        uid,
        cfg.shape.visual_material,
    )
    if material_entry is not None:
        materials = {material_entry[0]: material_entry[1]}
        for render in desc.renders:
            render.material = None
            render.material_ref = material_ref
    return desc, materials


def articulation_desc_from_usd(
    cfg: ArticulationCfg,
    *,
    per_env: bool = True,
    source_path: str | None = None,
    newton_solver_type: str | None = None,
) -> tuple[ArticulationDesc, dict[str, MaterialDesc]]:
    """Select the sole articulation in a USD stage."""
    preserve_asset_physics = cfg.resolve_asset_physics_mode() == "preserve"
    if not preserve_asset_physics:
        _validate_articulation_rigid_physics(
            cfg,
            newton_solver_type=newton_solver_type,
        )
    path = source_path or cfg.fpath
    scene, desc = _parse_singleton(path, "articulations", "articulation")
    uid = _required_uid(
        cfg.uid or os.path.splitext(os.path.basename(str(path)))[0],
        "Articulation",
    )
    cfg.uid = uid
    desc.name = uid
    desc.pose = _pose_from_cfg(cfg)
    desc.per_env = per_env
    renders = [visual for link in desc.links for visual in link.visuals]
    materials = _namespace_materials(renders, scene.materials, uid)

    if preserve_asset_physics:
        cfg.fix_base = bool(desc.fixed_base)
        cfg.disable_self_collision = not desc.enable_self_collision
        cfg.body_scale = tuple(float(value) for value in desc.body_scale)
    else:
        desc.fixed_base, desc.enable_self_collision = _articulation_root_values(cfg)
        desc.body_scale = _vector3(cfg.body_scale, field_name="body_scale")
    return desc, materials


def _parse_singleton(path: object, collection: str, label: str):
    if path is None:
        raise ValueError(f"A USD path is required for the {label}.")

    from dexsim.kit.usd import parse_usd

    scene = parse_usd(str(path))
    candidates = getattr(scene, collection)
    if len(candidates) != 1:
        found = [
            (item.name, None if item.usd is None else item.usd.prim_path)
            for item in candidates
        ]
        raise ValueError(
            f"Expected exactly one {label} in USD file {path!r}, found "
            f"{len(candidates)}: {found}."
        )
    return scene, candidates[0]


def _namespace_materials(
    renders: list[RenderDesc],
    materials: dict[str, MaterialDesc],
    uid: str,
) -> dict[str, MaterialDesc]:
    selected = {}
    for render in renders:
        if render.material_ref is None:
            continue
        source_ref = render.material_ref
        material = materials[source_ref]
        render.material_ref = f"{uid}::{source_ref}"
        selected[render.material_ref] = replace(
            material,
            name=f"{uid}::{material.name}",
        )
    return selected
