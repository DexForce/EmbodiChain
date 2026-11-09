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
import numpy as np

from dexsim.spawn import (
    ArticulationDesc,
    CollisionDesc,
    MaterialDesc,
    ObjectDesc,
    RenderDesc,
    RigidBodyPhysicsDesc,
)
from dexsim.types import ActorType

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ObjectBaseCfg,
    RigidObjectCfg,
    VisualObjectCfg,
    UsdSourceCfg,
    DeformableObjectCfg,
)
from embodichain.lab.sim.cfg.source import _asset_file_path, _source_prim_path
from embodichain.lab.sim.spawn.descriptors import (
    _apply_newton_gap_default,
    _compile_default_collision,
    _compile_newton_collision,
    _compile_rigid_physics,
    _compile_visual_material,
    _articulation_root_values,
    _pose_from_cfg,
    _particle_flags_from_cfg,
    _required_uid,
    _resolve_rigid_physics,
    _validate_articulation_rigid_physics,
    _vector3,
)

__all__ = ["articulation_desc_from_usd", "rigid_desc_from_usd", "visual_desc_from_usd"]

_PropertyCfgT = TypeVar("_PropertyCfgT")


def _require_usd_import_contract() -> None:
    """Reject DexSim wheels without source selection and scene constraints."""
    import inspect
    from dexsim.kit.usd import parse_stage
    from dexsim.spawn import SceneDesc

    if (
        "prim_path" not in inspect.signature(parse_stage).parameters
        or "constraints" not in SceneDesc.__dataclass_fields__
    ):
        raise RuntimeError(
            "USD loading requires a compatible DexSim build with source selection "
            "and standalone constraint support."
        )


def _source_pose(
    cfg: RigidObjectCfg | ArticulationCfg | DeformableObjectCfg | VisualObjectCfg,
    source_pose: np.ndarray,
) -> np.ndarray:
    source = cfg.source
    mode = cfg.source_pose_mode
    if source is not None and source.pose_mode != "auto":
        if mode != "auto" and mode != source.pose_mode:
            raise ValueError(
                "Conflicting source_pose_mode and legacy source.pose_mode."
            )
        mode = source.pose_mode
    if mode not in {"auto", "config", "preserve", "offset"}:
        raise ValueError(f"Unsupported source_pose_mode {mode!r}.")
    configured = _pose_from_cfg(cfg)
    if mode == "preserve":
        return source_pose.copy()
    if mode == "offset":
        return configured @ source_pose
    if mode == "auto" and cfg.init_local_pose is None:
        if not np.any(np.asarray(cfg.init_pos)) and not np.any(
            np.asarray(cfg.init_rot)
        ):
            return source_pose.copy()
    return configured


def _set_resolved_usd_pose(cfg: ObjectBaseCfg, pose: np.ndarray) -> None:
    """Keep a resolved runtime config reusable as an absolute placement.

    Import callers retain their original intent in their own config. The
    returned facade config represents the effective placement, including any
    scene transform, so reimporting it must not apply source placement again.
    """
    cfg.init_local_pose = pose.copy()
    cfg.source_pose_mode = "config"
    source = getattr(cfg, "source", None)
    if source is not None:
        source.pose_mode = "auto"


def _deformable_desc_from_usd(cfg: DeformableObjectCfg):
    if cfg.source is None:
        raise ValueError("USD deformables require a source configuration.")
    cfg.source.validate_source()
    scene, descriptor = _parse_singleton(
        cfg.source.file_path, "particle_sets", "deformable", source=cfg.source
    )
    _configure_deformable_usd(cfg, descriptor)
    return descriptor, scene.materials


def _configure_deformable_usd(cfg: DeformableObjectCfg, descriptor) -> None:
    from dexsim.spawn import SoftBodyDesc, ClothDesc

    expected = SoftBodyDesc if cfg.deformable_type == "volume" else ClothDesc
    if not isinstance(descriptor, expected):
        raise ValueError(f"USD source is not a {cfg.deformable_type} deformable.")
    if cfg.attrs.to_dict() != type(cfg.attrs)().to_dict():
        raise ValueError(
            "USD deformable physics is source-owned; physical config overrides are not supported."
        )
    meshing = getattr(cfg, "meshing", None)
    if meshing is not None and meshing.to_dict() != type(meshing)().to_dict():
        raise ValueError(
            "USD deformable simulation topology is source-owned; meshing overrides are not supported."
        )
    if getattr(cfg, "visual_shape", None) is not None:
        raise ValueError("USD deformable source already owns its visual mesh.")
    if (
        getattr(cfg.shape, "fpath", None) is not None
        or getattr(cfg.shape, "vertices", None) is not None
    ):
        raise ValueError(
            "USD deformable source cannot be combined with configured mesh geometry."
        )
    descriptor.name = _required_uid(cfg.uid, "Deformable")
    descriptor.pose = _source_pose(cfg, descriptor.pose)
    _set_resolved_usd_pose(cfg, descriptor.pose)
    if cfg.validate_mesh:
        descriptor.validate_mesh = True
    if cfg.particle_radius is not None:
        if not np.isfinite(cfg.particle_radius) or cfg.particle_radius <= 0:
            raise ValueError("particle_radius must be finite and positive.")
        descriptor.particle_radius = cfg.particle_radius
    if cfg.particle_flags is not None:
        descriptor.particle_flags = _particle_flags_from_cfg(cfg.particle_flags)


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
    *,
    recompute_inertia: bool = False,
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
    if recompute_inertia:
        source.inertia = None
    for name in ("inertia", "com_position"):
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


def visual_desc_from_usd(
    cfg: VisualObjectCfg, *, per_env: bool = True
) -> tuple[ObjectDesc, dict[str, MaterialDesc]]:
    """Select a USD object's appearance, explicitly omitting its physics.

    Args:
        cfg: Source asset, optional exact prim path, and visual transform.
        per_env: Replicate the visual in each environment.

    Returns:
        Render-only descriptor and referenced material templates.
    """
    cfg.validate()
    scene, desc = _parse_singleton(
        _asset_file_path(cfg),
        "mesh_objects",
        "visual object",
        source=cfg.source,
    )
    desc.name = _required_uid(cfg.uid, "Visual object")
    desc.pose = _source_pose(cfg, desc.pose)
    _set_resolved_usd_pose(cfg, desc.pose)
    desc.per_env = per_env
    desc.physics = None
    desc.collisions = []
    if cfg.body_scale is not None:
        desc.body_scale = _vector3(cfg.body_scale, field_name="body_scale")
    cfg.body_scale = tuple(float(value) for value in desc.body_scale)
    materials = _namespace_materials(desc.renders, scene.materials, desc.name)
    material_ref, material_entry = _compile_visual_material(
        desc.name, getattr(cfg.shape, "visual_material", None)
    )
    if material_entry is not None:
        materials = {material_entry[0]: material_entry[1]}
        for render in desc.renders:
            render.material = None
            render.material_ref = material_ref
    return desc, materials


def rigid_desc_from_usd(
    cfg: RigidObjectCfg,
    *,
    per_env: bool = True,
    newton_solver_type: str | None = None,
) -> tuple[ObjectDesc, dict[str, MaterialDesc]]:
    """Select a rigid object by exact prim path, or require a singleton stage."""
    uid = _required_uid(cfg.uid, "Rigid object")
    path = _asset_file_path(cfg)
    scene, desc = _parse_singleton(
        path, "mesh_objects", "rigid object", source=cfg.source
    )

    return _configure_rigid_usd(
        cfg, scene, desc, per_env=per_env, newton_solver_type=newton_solver_type
    )


def _configure_rigid_usd(
    cfg: RigidObjectCfg,
    scene,
    desc: ObjectDesc,
    *,
    per_env: bool = True,
    newton_solver_type: str | None = None,
) -> tuple[ObjectDesc, dict[str, MaterialDesc]]:
    """Apply a rigid configuration to an already parsed USD descriptor."""
    uid = _required_uid(cfg.uid, "Rigid object")
    path = _asset_file_path(cfg)

    desc.name = uid
    desc.pose = _source_pose(cfg, desc.pose)
    _set_resolved_usd_pose(cfg, desc.pose)
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
        _apply_newton_gap_default(desc.collisions, newton_solver_type)
        return desc, materials

    physics = _resolve_rigid_physics(
        cfg.attrs,
        newton_solver_type=newton_solver_type,
    )
    actor_type = cfg.body_type
    if actor_type is None:
        if desc.physics is None:
            actor_type = "static"
        else:
            actor_type = {
                ActorType.DYNAMIC: "dynamic",
                ActorType.KINEMATIC: "kinematic",
                ActorType.STATIC: "static",
            }[desc.physics.actor_type]
    configured_body = _compile_rigid_physics(physics, actor_type)
    desc.physics = _overlay_rigid_body_properties(
        desc.physics,
        configured_body,
        recompute_inertia=bool(physics.recompute_inertia),
    )
    if cfg.body_scale is not None:
        desc.body_scale = _vector3(cfg.body_scale, field_name="body_scale")
    cfg.body_type = actor_type
    cfg.body_scale = tuple(float(value) for value in desc.body_scale)
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

    _apply_newton_gap_default(desc.collisions, newton_solver_type)
    material_ref, material_entry = _compile_visual_material(
        uid,
        getattr(cfg.shape, "visual_material", None),
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
    """Select an articulation by exact prim path, or require a singleton stage."""
    preserve_asset_physics = cfg.resolve_asset_physics_mode() == "preserve"
    if not preserve_asset_physics:
        _validate_articulation_rigid_physics(
            cfg,
            newton_solver_type=newton_solver_type,
        )
    configured_path = _asset_file_path(cfg)
    path = source_path or configured_path
    scene, desc = _parse_singleton(
        path, "articulations", "articulation", source=cfg.source
    )
    return _configure_articulation_usd(
        cfg,
        scene,
        desc,
        per_env=per_env,
        source_path=path,
        newton_solver_type=newton_solver_type,
    )


def _configure_articulation_usd(
    cfg: ArticulationCfg,
    scene,
    desc: ArticulationDesc,
    *,
    per_env: bool = True,
    source_path: str | None = None,
    newton_solver_type: str | None = None,
) -> tuple[ArticulationDesc, dict[str, MaterialDesc]]:
    """Apply an articulation configuration to an already parsed USD descriptor."""
    preserve_asset_physics = cfg.resolve_asset_physics_mode() == "preserve"
    if not preserve_asset_physics:
        _validate_articulation_rigid_physics(cfg, newton_solver_type=newton_solver_type)
    path = source_path or _asset_file_path(cfg)
    uid = _required_uid(
        cfg.uid or os.path.splitext(os.path.basename(str(path)))[0],
        "Articulation",
    )
    cfg.uid = uid
    desc.name = uid
    desc.pose = _source_pose(cfg, desc.pose)
    _set_resolved_usd_pose(cfg, desc.pose)
    desc.per_env = per_env
    renders = [visual for link in desc.links for visual in link.visuals]
    materials = _namespace_materials(renders, scene.materials, uid)

    if not preserve_asset_physics and cfg.body_scale is not None:
        desc.body_scale = _vector3(cfg.body_scale, field_name="body_scale")
    cfg.body_scale = tuple(float(value) for value in desc.body_scale)
    desc.fixed_base, desc.enable_self_collision = _articulation_root_values(
        cfg,
        fixed_base_default=bool(desc.fixed_base),
        self_collision_default=desc.enable_self_collision,
    )
    return desc, materials


def _parse_singleton(path: object, collection: str, label: str, *, source=None):
    if path is None:
        raise ValueError(f"A USD path is required for the {label}.")

    scene = _parse_usd_source(
        source or UsdSourceCfg(file_path=str(path)), source_path=str(path)
    )
    candidates = getattr(scene, collection)
    prim_path = None if source is None else source.source_prim_path
    if prim_path is not None:
        candidates = [
            item
            for item in candidates
            if item.usd is not None and item.usd.prim_path == prim_path
        ]
    if len(candidates) != 1:
        found = [
            (item.name, None if item.usd is None else item.usd.prim_path)
            for item in candidates
        ]
        selection = "" if prim_path is None else f" at prim path {prim_path!r}"
        raise ValueError(
            f"Expected exactly one {label} in USD file {path!r}{selection}, found "
            f"{len(candidates)}: {found}."
        )
    selected = candidates[0]
    if label != "visual object":
        for constraint in scene.constraints:
            if any(
                endpoint is not None and endpoint.partition("/")[0] == selected.name
                for endpoint in (constraint.body0, constraint.body1)
            ):
                raise ValueError(
                    "A selected USD asset with standalone constraints requires SimulationManager.add_usd()."
                )
    return scene, selected


def _parse_usd_source(
    source: UsdSourceCfg,
    *,
    name: str | None = None,
    namespace: str | None = None,
    source_path: str | None = None,
):
    """Read one fresh DexSim scene for singleton and whole-scene imports."""
    from dexsim.kit.usd import parse_usd

    _require_usd_import_contract()
    source.validate_source()
    options = source.parse_kwargs()
    if name is not None:
        options["root_name"] = name
    if namespace is not None:
        options["namespace"] = namespace
    return parse_usd(source_path or source.file_path, **options)


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
