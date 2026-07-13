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

from __future__ import annotations

import os
import dexsim
import open3d as o3d

from dataclasses import MISSING
from typing import List, Union

from dexsim.types import (
    CloneStrategy,
    DriveType,
    ArticulationFlag,
    LoadOption,
    ObjectCloneOptions,
    RigidBodyShape,
    SDFConfig,
    ActorType,
)
from dexsim.engine import Articulation
from dexsim.environment import Env, Arena
from dexsim.models import MeshObject

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    LinkPhysicsOverrideCfg,
    RigidObjectCfg,
    SoftObjectCfg,
    ClothObjectCfg,
)
from embodichain.utils.string import resolve_matching_names
from embodichain.lab.sim.shapes import MeshCfg, CubeCfg, SphereCfg
from embodichain.utils import logger
from dexsim.kit.meshproc import get_mesh_auto_uv
import numpy as np


def _is_newton_backend_active() -> bool:
    """Return whether the current default world uses the Newton physics scene."""
    from embodichain.lab.sim.sim_manager import get_physics_scene
    from embodichain.lab.sim.objects.backends import is_newton_scene

    return is_newton_scene(get_physics_scene())


def _set_body_scale_after_rigidbody(obj: MeshObject, body_scale: tuple | list) -> None:
    """Set body scale after rigid body creation for Newton compatibility."""
    obj.set_body_scale(*body_scale)


def _newton_solver_type() -> str | None:
    """Return the active Newton solver type, or None if unavailable."""
    try:
        from embodichain.lab.sim.sim_manager import get_physics_scene

        mgr = getattr(get_physics_scene(), "manager", None)
        if mgr is None:
            return None
        return getattr(getattr(mgr, "cfg", None), "solver_cfg", None).solver_type
    except Exception:
        return None


def _attach_newton_rigidbody_desc(
    obj: MeshObject,
    cfg: RigidObjectCfg,
    body_type: ActorType,
    shape_type: RigidBodyShape,
) -> None:
    """Attach rigid-body physics via dexsim's Newton desc-native path.

    Used when ``cfg.attrs.newton`` is set on the Newton backend: builds the
    resolved Newton shape descriptor (common fields projected + Newton-native
    sub-config) and a ``RigidBodyPhysicsDesc`` body descriptor, populates the
    ``mgr.dexsim_meta`` scaffolding that dexsim's registration/rebuild reads
    (mirroring ``NewtonSpawnAdapter._attach_newton``), and registers via
    ``register_mesh_object_to_newton_patch`` — fully bypassing the legacy
    ``PhysicalAttr`` path so Newton-native contact/shape params reach the model.
    Emits per-solver / backend-mismatch warnings.
    """
    from embodichain.lab.sim.sim_manager import get_physics_scene
    from dexsim.engine.newton_physics.rigid_body.registration import (
        register_mesh_object_to_newton_patch,
    )
    from dexsim.engine.newton_physics.registry import _get_entity_native_handle
    from embodichain.lab.sim.physics_attrs import (
        resolve_newton_body,
        resolve_newton_shape,
        warn_ignored_contact_fields,
        warn_backend_mismatched_fields,
    )

    mgr = getattr(get_physics_scene(), "manager", None)
    if mgr is None:
        logger.log_error(
            "Newton manager is unavailable; cannot attach rigid body via the "
            "desc-native path."
        )
    shape = resolve_newton_shape(cfg.attrs)
    solver_type = _newton_solver_type()
    if solver_type is not None:
        warn_ignored_contact_fields(shape, solver_type)
    warn_backend_mismatched_fields(cfg.attrs, "newton")
    body = resolve_newton_body(cfg.attrs, body_type)

    # Populate the dexsim_meta scaffolding registration/rebuild read. This
    # mirrors dexsim's NewtonSpawnAdapter._attach_newton meta dict so the body
    # rebuilds correctly on the next finalize.
    entity_handle = _get_entity_native_handle(obj)
    arena = obj.get_arena() if hasattr(obj, "get_arena") else None
    arena_handle = arena.get_native_handle() if arena is not None else -1
    mgr.dexsim_meta[entity_handle] = {
        "actor_type": body_type,
        "shape_type": shape_type,
        "node_scale": np.asarray(obj.get_scale(), dtype=np.float32).reshape(-1)[:3],
        "body_scale": np.asarray(obj.get_body_scale(), dtype=np.float32).reshape(-1)[
            :3
        ],
        "arena_native_handle": arena_handle,
        "newton_world_index": -1,
        "newton_shape": shape,
        "newton_body": body,
    }

    register_mesh_object_to_newton_patch(
        mgr,
        obj,
        body_type,
        shape_type,
        attr=None,
        mesh_source_obj=obj,
        newton_shape=shape,
        newton_body=body,
    )
    # Newton requires body scale after rigid-body creation.
    _set_body_scale_after_rigidbody(obj, cfg.body_scale)


def _use_newton_desc_path(cfg: RigidObjectCfg) -> bool:
    """Whether to route rigid-body spawn through the Newton desc-native path."""
    return _is_newton_backend_active() and cfg.attrs.newton is not None


def _newton_subcfg_has_fields(newton_cfg) -> bool:
    """Return True if a Newton sub-config sets any field."""
    if newton_cfg is None:
        return False
    return any(
        getattr(newton_cfg, f.name, None) is not None
        for f in newton_cfg.__dataclass_fields__
        if f.name != "newton"
    )


def _warn_newton_articulation_native_attrs(cfg: "ArticulationCfg") -> None:
    """Warn that Newton-native per-link contact params are not applied to articulations.

    dexsim's ``NewtonArticulation`` exposes no per-link contact-material setter
    (ke/kd/margin/...), so the ``attrs.newton`` sub-config on an articulation is
    accepted for config symmetry but cannot be applied per-link on Newton today.
    Common fields (mass/friction/restitution/contact_offset) are still applied
    via the legacy ``set_physical_attr`` path.
    """
    sources = []
    if _newton_subcfg_has_fields(getattr(cfg.attrs, "newton", None)):
        sources.append("attrs.newton")
    for group_name, group_cfg in (cfg.link_attrs or {}).items():
        if _newton_subcfg_has_fields(getattr(group_cfg.attrs, "newton", None)):
            sources.append(f"link_attrs['{group_name}'].attrs.newton")
    if sources:
        logger.log_warning(
            "Newton-native per-link contact/shape params (" + ", ".join(sources) + ") "
            "are not yet applied to articulation links on the Newton backend "
            "(no dexsim per-link contact-material API). Common fields are applied."
        )


def get_dexsim_arenas() -> List[dexsim.environment.Arena]:
    """Get all arenas in the default dexsim world.

    Returns:
        List[dexsim.environment.Arena]: A list of arenas in the default world, or an empty list if no world is found.
    """
    world = dexsim.default_world()
    if world is None:
        logger.log_warning(f"No default world found. Returning empty arena list.")
        return []

    env = world.get_env()
    arenas = env.get_all_arenas()
    if len(arenas) == 0:
        return [env]
    return arenas


def get_dexsim_arena_num() -> int:
    """Get the number of arenas in the default dexsim world.

    Returns:
        int: The number of arenas in the default world, or 0 if no world is found.
    """
    arenas = get_dexsim_arenas()
    return len(arenas)


def _resolve_mesh_collision_params(
    cfg: RigidObjectCfg,
) -> tuple[int, str, int]:
    """Resolve legacy and shape-level mesh collision parameters."""

    def is_missing(value) -> bool:
        # deepcopy() can produce a distinct instance of dataclasses.MISSING.
        return value is MISSING or isinstance(value, type(MISSING))

    max_convex_hull_num = next(
        value
        for value in (
            cfg.max_convex_hull_num,
            cfg.shape.max_convex_hull_num,
            1,
        )
        if not is_missing(value)
    )
    acd_method = next(
        value
        for value in (cfg.acd_method, cfg.shape.acd_method, "coacd")
        if not is_missing(value)
    )
    sdf_resolution = next(
        value
        for value in (cfg.sdf_resolution, cfg.shape.sdf_resolution, 0)
        if not is_missing(value)
    )
    return max_convex_hull_num, acd_method, sdf_resolution


def get_dexsim_drive_type(drive_type: str) -> DriveType:
    """Get the dexsim drive type from a string.

    Args:
        drive_type (str): The drive type as a string.

    Returns:
        DriveType: The corresponding DriveType enum.
    """
    if drive_type == "force":
        return DriveType.FORCE
    elif drive_type == "acceleration":
        return DriveType.ACCELERATION
    elif drive_type == "none":
        return DriveType.NONE
    else:
        logger.error(f"Invalid dexsim drive type: {drive_type}")


def _resolve_link_physics_groups(
    link_names: list[str], link_attrs: dict[str, LinkPhysicsOverrideCfg]
) -> dict[str, LinkPhysicsOverrideCfg]:
    """Map each link name to exactly one override group.

    Raises:
        ValueError: If a link matches zero groups (not required) or multiple groups.
    """
    link_to_group: dict[str, LinkPhysicsOverrideCfg] = {}
    for group_cfg in link_attrs.values():
        _, matched_names = resolve_matching_names(
            keys=group_cfg.link_names_expr, list_of_strings=link_names
        )
        for name in matched_names:
            if name in link_to_group:
                raise ValueError(
                    f"Link '{name}' matched multiple link_attrs groups. Each link must "
                    "match at most one group."
                )
            link_to_group[name] = group_cfg
    return link_to_group


def _apply_link_physics_overrides(
    art: Articulation, cfg: ArticulationCfg, link_names: list[str]
) -> None:
    """Apply per-link physics overrides on top of global articulation attrs."""
    if not cfg.link_attrs:
        return

    link_to_group = _resolve_link_physics_groups(link_names, cfg.link_attrs)
    for name in link_names:
        group_cfg = link_to_group.get(name)
        if group_cfg is None:
            continue
        physical_attr = group_cfg.attrs.merge_with(cfg.attrs)
        replace_inertial = group_cfg.replace_inertial or (
            group_cfg.attrs.mass is not None
        )
        art.set_physical_attr(physical_attr, name, is_replace_inertial=replace_inertial)


def default_articulation_clone_options() -> ObjectCloneOptions:
    """Return clone options used when duplicating articulations across arenas."""
    options = ObjectCloneOptions()
    options.render.material = CloneStrategy.DEEP_COPY
    return options


def default_rigid_object_clone_options() -> ObjectCloneOptions:
    """Return clone options used when duplicating rigid actors across arenas."""
    options = ObjectCloneOptions()
    options.render.material = CloneStrategy.DEEP_COPY
    return options


def _clone_actor_between_arenas(
    source_arena: Arena | Env,
    source_name: str,
    target_arena: Arena | Env,
    target_name: str,
    clone_options: ObjectCloneOptions,
) -> MeshObject:
    """Clone a mesh actor from one arena/env to another."""
    return source_arena.clone_actor_to(
        source_name, target_arena, target_name, clone_options
    )


def _clone_articulation_between_arenas(
    source_arena: Arena | Env,
    source_name: str,
    target_arena: Arena | Env,
    target_name: str,
    clone_options: ObjectCloneOptions,
) -> Articulation:
    """Clone an articulation from one arena/env to another."""
    if _is_newton_backend_active():
        return source_arena.clone_skeleton_to(
            source_name, target_arena, target_name, clone_options
        )
    return source_arena.clone_articulation_to(
        source_name, target_arena, target_name, clone_options
    )


def spawn_articulation_entities(
    cfg: ArticulationCfg,
    env_list: list[Arena | Env],
    *,
    clone_options: ObjectCloneOptions | None = None,
) -> list[Articulation]:
    """Load one articulation prototype and clone it into additional arenas.

    DexSim configuration is applied once on the prototype before cloning.
    """
    if cfg.uid is None:
        logger.log_error("Articulation uid must be set before spawning entities.")

    if clone_options is None:
        clone_options = default_articulation_clone_options()

    source_env = env_list[0]
    prototype_name = f"{cfg.uid}_0"
    prototype = source_env.load_urdf(cfg.fpath)
    prototype.set_name(prototype_name)

    if not cfg.use_usd_properties:
        set_dexsim_articulation_cfg(prototype, cfg)

    entities = [prototype]
    for env_idx in range(1, len(env_list)):
        target_name = f"{cfg.uid}_{env_idx}"
        clone = _clone_articulation_between_arenas(
            source_env,
            prototype_name,
            env_list[env_idx],
            target_name,
            clone_options,
        )
        if clone is None:
            logger.log_error(
                f"Failed to clone articulation '{prototype_name}' into env {env_idx}."
            )
        entities.append(clone)
    return entities


def _find_single_articulation_in_usd_import(results: dict, fpath: str) -> Articulation:
    """Return the sole articulation imported from a USD file."""
    articulations_found = [
        value for value in results.values() if isinstance(value, Articulation)
    ]
    if len(articulations_found) == 0:
        logger.log_error(f"No articulation found in USD file {fpath}.")
    if len(articulations_found) > 1:
        logger.log_error(f"Multiple articulations found in USD file {fpath}.")
    return articulations_found[0]


def spawn_usd_articulation_entities(
    cfg: ArticulationCfg,
    env_list: list[Arena | Env],
    *,
    cache_dir: str | None = None,
    clone_options: ObjectCloneOptions | None = None,
) -> list[Articulation]:
    """Import one USD articulation prototype and clone it into additional arenas."""
    if cfg.uid is None:
        logger.log_error("Articulation uid must be set before spawning entities.")
    if len(env_list) == 0:
        return []

    if clone_options is None:
        clone_options = default_articulation_clone_options()

    source_env = env_list[0]
    prototype_name = f"{cfg.uid}_0"
    results = source_env.import_from_usd_file(
        cfg.fpath, return_object=True, cache_dir=cache_dir
    )
    prototype = _find_single_articulation_in_usd_import(results, cfg.fpath)
    prototype.set_name(prototype_name)

    if not cfg.use_usd_properties:
        set_dexsim_articulation_cfg(prototype, cfg)

    entities = [prototype]
    for env_idx in range(1, len(env_list)):
        target_name = f"{cfg.uid}_{env_idx}"
        clone = _clone_articulation_between_arenas(
            source_env,
            prototype_name,
            env_list[env_idx],
            target_name,
            clone_options,
        )
        if clone is None:
            logger.log_error(
                f"Failed to clone articulation '{prototype_name}' into env {env_idx}."
            )
        entities.append(clone)
    return entities


def set_dexsim_articulation_cfg(art: Articulation, cfg: ArticulationCfg) -> None:
    """Apply EmbodiChain articulation cfg to a single DexSim articulation entity.

    Args:
        art: DexSim articulation (or Newton skeleton carrier) to configure.
        cfg: EmbodiChain articulation configuration.
    """

    def get_drive_type(drive_pros):
        if isinstance(drive_pros, dict):
            return drive_pros.get("drive_type", None)
        return getattr(drive_pros, "drive_type", None)

    drive_pros = getattr(cfg, "drive_pros", None)
    drive_type = get_drive_type(drive_pros) if drive_pros is not None else None

    if drive_type == "force":
        drive_type = DriveType.FORCE
    elif drive_type == "acceleration":
        drive_type = DriveType.ACCELERATION
    elif drive_type == "none":
        drive_type = DriveType.NONE
    else:
        logger.log_error(f"Unknow drive type {drive_type}")

    is_newton_art = hasattr(art, "dexsim_meta_links")
    lifecycle_state = getattr(getattr(art, "_mgr", None), "_lifecycle_state", None)
    lifecycle_name = getattr(lifecycle_state, "name", "")
    if not is_newton_art or lifecycle_name == "BUILDER":
        art.set_body_scale(cfg.body_scale)

    link_names = art.get_link_names()
    if is_newton_art:
        for name in link_names:
            art.set_physical_attr(cfg.attrs.attr(), name)
        _warn_newton_articulation_native_attrs(cfg)
    else:
        art.set_physical_attr(cfg.attrs.attr())
    _apply_link_physics_overrides(art, cfg, link_names)
    art.set_articulation_flag(ArticulationFlag.FIX_BASE, cfg.fix_base)
    art.set_articulation_flag(
        ArticulationFlag.DISABLE_SELF_COLLISION, cfg.disable_self_collision
    )
    if hasattr(art, "set_solver_iteration_counts"):
        art.set_solver_iteration_counts(
            min_position_iters=cfg.min_position_iters,
            min_velocity_iters=cfg.min_velocity_iters,
        )

    for name in link_names:
        if not hasattr(art, "get_physical_body"):
            continue
        physical_body = art.get_physical_body(name)
        inertia = physical_body.get_mass_space_inertia_tensor()
        inertia = np.maximum(inertia, 1e-4)
        physical_body.set_mass_space_inertia_tensor(inertia)

        if cfg.compute_uv:
            render_body = art.get_render_body(name)
            if render_body:
                render_body.set_projective_uv()

            # TODO: will crash when exit if not explicitly delete.
            # This may due to the destruction of render body order when exiting.
            del render_body


def is_rt_enabled() -> bool:
    """Check if Ray Tracing rendering backend is enabled in the default dexsim world.

    Returns:
        bool: True if Ray Tracing rendering is enabled, False otherwise.
    """
    config = dexsim.get_world_config()

    return (
        config.renderer == dexsim.types.Renderer.FASTRT
        or config.renderer == dexsim.types.Renderer.HYBRID
        or config.renderer == dexsim.types.Renderer.OFFLINERT
    )


def create_cube(
    envs: List[Union[Env, Arena]], size: List[float], uid: str = "cube"
) -> List[MeshObject]:
    """Create cube objects in the specified environments or arenas.

    Args:
        envs (List[Union[Env, Arena]]): List of environments or arenas to create cubes in.
        size (List[float]): Size of the cube as [length, width, height] in meters.
        uid (str, optional): Unique identifier for the cube objects. Defaults to "cube".

    Returns:
        List[MeshObject]: List of created cube mesh objects.
    """
    cubes = []
    for i, env in enumerate(envs):
        cube = env.create_cube(size[0], size[1], size[2])
        cube.set_name(f"{uid}_{i}")
        cubes.append(cube)
    return cubes


def create_sphere(
    envs: List[Union[Env, Arena]],
    radius: float,
    resolution: int = 20,
    uid: str = "sphere",
) -> List[MeshObject]:
    """Create sphere objects in the specified environments or arenas.

    Args:
        envs (List[Union[Env, Arena]]): List of environments or arenas to create spheres in.
        radius (float): Radius of the sphere in meters.
        resolution (int, optional): Resolution of the sphere mesh. Defaults to 20.
        uid (str, optional): Unique identifier for the sphere objects. Defaults to "sphere".

    Returns:
        List[MeshObject]: List of created sphere mesh objects.
    """
    spheres = []
    for i, env in enumerate(envs):
        sphere = env.create_sphere(radius, resolution)
        sphere.set_name(f"{uid}_{i}")
        spheres.append(sphere)
    return spheres


def _mesh_load_option_from_cfg(cfg: RigidObjectCfg) -> LoadOption:
    """Build DexSim mesh load options from a rigid-object configuration."""
    option = LoadOption()
    option.rebuild_normals = cfg.shape.load_option.rebuild_normals
    option.rebuild_tangent = cfg.shape.load_option.rebuild_tangent
    option.rebuild_3rdnormal = cfg.shape.load_option.rebuild_3rdnormal
    option.rebuild_3rdtangent = cfg.shape.load_option.rebuild_3rdtangent
    option.smooth = cfg.shape.load_option.smooth
    return option


def _apply_mesh_uv_mapping(obj: MeshObject, cfg: RigidObjectCfg) -> None:
    """Compute and apply UV mapping for a mesh rigid-object prototype."""
    if not cfg.shape.compute_uv:
        return

    vertices = obj.get_vertices()
    triangles = obj.get_triangles()
    o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
    _, uvs = get_mesh_auto_uv(o3d_mesh, np.array(cfg.shape.project_direction))
    obj.set_uv_mapping(uvs)


def _configure_primitive_rigidbody(
    obj: MeshObject,
    cfg: RigidObjectCfg,
    body_type,
    *,
    is_newton_backend: bool,
    shape_type: RigidBodyShape,
) -> None:
    """Attach primitive rigid-body physics to a cube or sphere prototype."""
    if is_newton_backend and cfg.attrs.newton is not None:
        _attach_newton_rigidbody_desc(obj, cfg, body_type, shape_type)
        return
    if not is_newton_backend:
        obj.set_body_scale(*cfg.body_scale)
    obj.add_rigidbody(body_type, shape_type, cfg.attrs.attr())
    if is_newton_backend:
        _set_body_scale_after_rigidbody(obj, cfg.body_scale)


def _import_usd_rigid_prototype(
    env: Arena | Env,
    fpath: str,
    prototype_name: str,
) -> MeshObject:
    """Import a single rigid mesh actor from USD as the spawn prototype."""
    results = env.import_from_usd_file(fpath, return_object=True)
    rigidbodys_found = [
        value for value in results.values() if isinstance(value, MeshObject)
    ]
    if len(rigidbodys_found) == 0:
        logger.log_error(f"No rigid body found in USD file: {fpath}")
    if len(rigidbodys_found) > 1:
        logger.log_error(f"Multiple rigid bodies found in USD file: {fpath}.")
    prototype = rigidbodys_found[0]
    prototype.set_name(prototype_name)
    return prototype


def _load_rigid_mesh_prototype(
    env: Arena | Env,
    cfg: RigidObjectCfg,
    *,
    cache_dir: str | None,
    body_type,
    is_newton_backend: bool,
) -> MeshObject:
    """Load and configure one mesh rigid-object prototype in the source arena."""
    option = _mesh_load_option_from_cfg(cfg)
    fpath = cfg.shape.fpath
    max_convex_hull_num, acd_method, sdf_resolution = _resolve_mesh_collision_params(
        cfg
    )

    if max_convex_hull_num > 1:
        obj = env.load_actor_with_acd(
            fpath,
            duplicate=True,
            attach_scene=True,
            option=option,
            cache_path=cache_dir,
            actor_type=body_type,
            max_convex_hull_num=max_convex_hull_num,
            method=acd_method,
        )
    elif sdf_resolution > 0:
        if not is_newton_backend and cfg.body_scale not in [
            (1.0, 1.0, 1.0),
            [1.0, 1.0, 1.0],
        ]:
            logger.log_error(
                f"Non-unit body scale {cfg.body_scale} is not supported for SDF "
                "collision yet. Please set body_scale to (1.0, 1.0, 1.0) for SDF "
                "collision."
            )
        obj = env.load_actor(fpath, duplicate=True, attach_scene=True, option=option)
        sdf_cfg = SDFConfig(resolution=sdf_resolution)
        obj.add_physical_body(
            body_type,
            RigidBodyShape.SDF,
            config=sdf_cfg,
            attr=cfg.attrs.attr(),
        )
    else:
        obj = env.load_actor(fpath, duplicate=True, attach_scene=True, option=option)
        if is_newton_backend and cfg.attrs.newton is not None:
            _attach_newton_rigidbody_desc(obj, cfg, body_type, RigidBodyShape.CONVEX)
        else:
            obj.add_rigidbody(body_type, RigidBodyShape.CONVEX, cfg.attrs.attr())

    _apply_mesh_uv_mapping(obj, cfg)
    return obj


def _spawn_clones_from_prototype(
    source_env: Arena | Env,
    prototype_name: str,
    env_list: list[Arena | Env],
    uid: str,
    clone_options: ObjectCloneOptions,
) -> list[MeshObject]:
    """Return the prototype plus clones for all remaining arenas."""
    prototype = source_env.get_actor(prototype_name)
    if prototype is None:
        logger.log_error(
            f"Rigid object prototype '{prototype_name}' was not found in the source arena."
        )

    entities = [prototype]
    for env_idx in range(1, len(env_list)):
        target_name = f"{uid}_{env_idx}"
        clone = _clone_actor_between_arenas(
            source_env,
            prototype_name,
            env_list[env_idx],
            target_name,
            clone_options,
        )
        if clone is None:
            logger.log_error(
                f"Failed to clone rigid object '{prototype_name}' into env {env_idx}."
            )
        entities.append(clone)
    return entities


def spawn_rigid_object_entities(
    cfg: RigidObjectCfg,
    env_list: list[Arena | Env],
    *,
    cache_dir: str | None = None,
    clone_options: ObjectCloneOptions | None = None,
) -> list[MeshObject]:
    """Load one rigid-object prototype and clone it into additional arenas.

    Mesh loading, convex decomposition, and physics setup run once on the
    prototype in ``env_list[0]`` before cloning.
    """
    if cfg.uid is None:
        logger.log_error("Rigid object uid must be set before spawning entities.")
    if len(env_list) == 0:
        return []

    if clone_options is None:
        clone_options = default_rigid_object_clone_options()

    body_type = cfg.to_dexsim_body_type()
    is_newton_backend = _is_newton_backend_active()
    source_env = env_list[0]
    prototype_name = f"{cfg.uid}_0"

    if isinstance(cfg.shape, MeshCfg):
        fpath = cfg.shape.fpath
        is_usd = fpath.endswith((".usd", ".usda", ".usdc"))
        if is_usd:
            prototype = _import_usd_rigid_prototype(source_env, fpath, prototype_name)
        else:
            cfg.use_usd_properties = False
            prototype = _load_rigid_mesh_prototype(
                source_env,
                cfg,
                cache_dir=cache_dir,
                body_type=body_type,
                is_newton_backend=is_newton_backend,
            )
            prototype.set_name(prototype_name)
    elif isinstance(cfg.shape, CubeCfg):
        prototype = source_env.create_cube(
            cfg.shape.size[0], cfg.shape.size[1], cfg.shape.size[2]
        )
        prototype.set_name(prototype_name)
        _configure_primitive_rigidbody(
            prototype,
            cfg,
            body_type,
            is_newton_backend=is_newton_backend,
            shape_type=RigidBodyShape.BOX,
        )
    elif isinstance(cfg.shape, SphereCfg):
        prototype = source_env.create_sphere(cfg.shape.radius, cfg.shape.resolution)
        prototype.set_name(prototype_name)
        _configure_primitive_rigidbody(
            prototype,
            cfg,
            body_type,
            is_newton_backend=is_newton_backend,
            shape_type=RigidBodyShape.SPHERE,
        )
    else:
        logger.log_error(
            f"Unsupported rigid object shape type: {type(cfg.shape)}. "
            "Supported types: MeshCfg, CubeCfg, SphereCfg."
        )
        return []

    if len(env_list) == 1:
        return [prototype]
    return _spawn_clones_from_prototype(
        source_env, prototype_name, env_list, cfg.uid, clone_options
    )


def load_mesh_objects_from_cfg(
    cfg: RigidObjectCfg, env_list: List[Arena], cache_dir: str | None = None
) -> List[MeshObject]:
    """Load mesh objects from configuration.

    Args:
        cfg (RigidObjectCfg): Configuration for the rigid object.
        env_list (List[Arena]): List of arenas to load the objects into.

    cache_dir (str | None, optional): Directory for caching convex decomposition files. Defaults to None
    Returns:
        List[MeshObject]: List of loaded mesh objects.
    """
    return spawn_rigid_object_entities(cfg, env_list, cache_dir=cache_dir)


def load_soft_object_from_cfg(
    cfg: SoftObjectCfg, env_list: List[Arena]
) -> List[MeshObject]:
    obj_list = []

    option = LoadOption()
    option.rebuild_normals = cfg.shape.load_option.rebuild_normals
    option.rebuild_tangent = cfg.shape.load_option.rebuild_tangent
    option.rebuild_3rdnormal = cfg.shape.load_option.rebuild_3rdnormal
    option.rebuild_3rdtangent = cfg.shape.load_option.rebuild_3rdtangent
    option.smooth = cfg.shape.load_option.smooth
    option.share_mesh = False

    for i, env in enumerate(env_list):
        obj = env.load_actor(
            fpath=cfg.shape.fpath, duplicate=True, attach_scene=True, option=option
        )
        obj.add_softbody(cfg.voxel_attr.attr(), cfg.physical_attr.attr())
        if cfg.shape.compute_uv:
            vertices = obj.get_vertices()
            triangles = obj.get_triangles()

            o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
            _, uvs = get_mesh_auto_uv(o3d_mesh, cfg.shape.project_direction)
            obj.set_uv_mapping(uvs)
        obj.set_name(f"{cfg.uid}_{i}")
        obj_list.append(obj)
    return obj_list


def load_cloth_object_from_cfg(
    cfg: ClothObjectCfg, env_list: List[Arena]
) -> List[MeshObject]:
    obj_list = []

    option = LoadOption()
    option.rebuild_normals = cfg.shape.load_option.rebuild_normals
    option.rebuild_tangent = cfg.shape.load_option.rebuild_tangent
    option.rebuild_3rdnormal = cfg.shape.load_option.rebuild_3rdnormal
    option.rebuild_3rdtangent = cfg.shape.load_option.rebuild_3rdtangent
    option.smooth = cfg.shape.load_option.smooth
    option.share_mesh = False

    for i, env in enumerate(env_list):
        obj = env.load_actor(
            fpath=cfg.shape.fpath, duplicate=True, attach_scene=True, option=option
        )
        obj.add_clothbody(cfg.physical_attr.attr())
        if cfg.shape.compute_uv:
            vertices = obj.get_vertices()
            triangles = obj.get_triangles()

            o3d_mesh = o3d.t.geometry.TriangleMesh(vertices, triangles)
            _, uvs = get_mesh_auto_uv(o3d_mesh, cfg.shape.project_direction)
            obj.set_uv_mapping(uvs)
        obj.set_name(f"{cfg.uid}_{i}")
        obj_list.append(obj)
    return obj_list
