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

"""Compile a USD import completely before native scene declaration."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from dexsim.spawn import (
    ArticulationDesc,
    ConstraintDesc,
    MeshObjectDesc,
    SoftBodyDesc,
    ClothDesc,
)

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    RigidObjectCfg,
    RobotCfg,
    VisualObjectCfg,
    UsdSourceCfg,
    UsdImportCfg,
    VolumeDeformableObjectCfg,
    SurfaceDeformableObjectCfg,
)
from embodichain.lab.sim.objects import (
    Articulation,
    RigidObject,
    Robot,
    VisualObject,
    VolumeDeformableObject,
    SurfaceDeformableObject,
)
from embodichain.lab.sim.sensors import Camera, CameraCfg, StereoCamera, StereoCameraCfg
from .descriptors import _pose_from_cfg, configure_articulation_desc
from .usd import (
    _configure_articulation_usd,
    _configure_rigid_usd,
    _parse_usd_source,
    _source_pose,
    _set_resolved_usd_pose,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__: list[str] = []


@dataclass
class _UsdImportPlan:
    source: Any
    declarations: list[tuple[str, str, Any, Any]]
    mounted_sensors: list[Any]
    resource_uids: dict[str, str]
    resource_types: dict[str, str]
    constraint_uids: dict[str, str]
    resource_source_paths: dict[str, str]


def _source_path(descriptor: Any) -> str:
    usd = getattr(descriptor, "usd", None)
    return usd.prim_path if usd is not None and usd.prim_path else descriptor.name


def _compile_usd_import(
    manager: SimulationManager, recipe: UsdImportCfg
) -> _UsdImportPlan:
    recipe.validate_import()
    file_path = recipe.source.file_path
    source = _parse_usd_source(
        recipe.source, name=recipe.name, namespace=recipe.namespace
    )
    if source.gaussian_splats:
        raise NotImplementedError(
            "USD Gaussian-splat imports are not supported by the EmbodiChain USD entry point."
        )
    unsupported_particles = [
        type(item).__name__
        for item in source.particle_sets
        if not isinstance(item, (SoftBodyDesc, ClothDesc))
    ]
    if unsupported_particles:
        raise NotImplementedError(
            f"Unsupported USD particle-set types: {unsupported_particles}."
        )
    manager._spawn_scene._require_compatible_particles(
        cables=bool(source.cables), particles=bool(source.particle_sets)
    )
    resources = list(source.iter_entities())
    source_paths_by_id = {id(item): _source_path(item) for item in resources}
    counts = Counter(source_paths_by_id.values())
    occurrences: Counter[str] = Counter()
    resource_keys = {}
    for item in resources:
        path = source_paths_by_id[id(item)]
        resource_keys[id(item)] = (
            path if counts[path] == 1 else f"{path}#{occurrences[path]}"
        )
        occurrences[path] += 1
    descriptors = [item for item in resources if not isinstance(item, ConstraintDesc)]
    robot_cfgs = recipe.robot_cfgs
    asset_cfgs = recipe.asset_cfgs
    source_paths = {
        (
            item.usd.prim_path
            if item.usd is not None and item.usd.prim_path
            else item.name
        )
        for item in (*source.mesh_objects, *source.articulations, *source.particle_sets)
    }
    overlap = robot_cfgs.keys() & asset_cfgs.keys()
    missing = (robot_cfgs.keys() | asset_cfgs.keys()) - source_paths
    if overlap or missing:
        raise ValueError(
            f"Invalid USD configuration targets; duplicate={sorted(overlap)}, missing={sorted(missing)}."
        )
    root_pose = (
        None if recipe.pose is None else np.asarray(recipe.pose, dtype=np.float32)
    )
    if root_pose is not None and (
        root_pose.shape != (4, 4) or not np.isfinite(root_pose).all()
    ):
        raise ValueError("USD scene pose must be a finite (4, 4) transform.")

    occupied = set(manager.asset_uids)
    for collection in (
        "mesh_objects",
        "articulations",
        "particle_sets",
        "constraints",
        "cameras",
        "lights",
        "records",
        "cables",
        "scene_entities",
    ):
        for item in getattr(manager._spawn_scene.builder, collection, ()):
            occupied.add(item.name)
    declarations = []
    mounted_sensors = []
    sensor_uids = set(manager._sensors)
    renamed = {}
    rigid_renames = {}
    for descriptor in descriptors:
        original_name = descriptor.name
        source_path = (
            descriptor.usd.prim_path
            if descriptor.usd is not None and descriptor.usd.prim_path
            else descriptor.name
        )
        configured = asset_cfgs.get(source_path)
        legacy_robot = robot_cfgs.get(source_path)
        is_articulation = isinstance(descriptor, ArticulationDesc)
        is_soft = isinstance(descriptor, SoftBodyDesc)
        is_cloth = isinstance(descriptor, ClothDesc)
        is_deformable = is_soft or is_cloth
        if is_deformable and (
            manager.physics.name != "newton" or manager.device.type != "cuda"
        ):
            raise NotImplementedError(
                "USD soft bodies and cloth require the Newton backend on CUDA."
            )
        if is_deformable and manager._active_newton_solver_type not in (
            None,
            "dexuni",
            "xpbd",
            "semi_implicit",
            "vbd",
        ):
            raise NotImplementedError(
                "The active Newton solver does not support USD soft bodies or cloth."
            )
        if is_deformable and manager.sim_config.physics_cfg.requires_grad:
            raise NotImplementedError(
                "USD deformable facades do not support Newton gradient mode."
            )
        if legacy_robot is not None and not is_articulation:
            raise ValueError(
                f"USD robot target must be an articulation: {source_path}."
            )
        selected_path = (
            descriptor.usd.prim_path
            if descriptor.usd is not None and descriptor.usd.prim_path
            else None
        )
        is_visual = (
            type(descriptor) is MeshObjectDesc
            and descriptor.physics is None
            and not descriptor.collisions
        )
        if configured is not None and not isinstance(
            descriptor, (MeshObjectDesc, ArticulationDesc, SoftBodyDesc, ClothDesc)
        ):
            raise ValueError(
                f"USD configuration type does not match source entity {source_path}."
            )
        if isinstance(configured, dict):
            data = deepcopy(configured)
            source_options = data.pop("source", None)
            if source_options is not None:
                if not isinstance(source_options, dict) or set(source_options) != {
                    "pose_mode"
                }:
                    raise ValueError(
                        f"Per-asset source selection is not supported: {source_path}."
                    )
            if "shape" in data or "fpath" in data:
                raise ValueError(
                    f"USD source selection belongs to add_usd(), not asset_cfgs: {source_path}."
                )
            config_type = (
                VolumeDeformableObjectCfg
                if is_soft
                else (
                    SurfaceDeformableObjectCfg
                    if is_cloth
                    else (
                        ArticulationCfg
                        if is_articulation
                        else VisualObjectCfg if is_visual else RigidObjectCfg
                    )
                )
            )
            if is_deformable and set(data) - {
                "uid",
                "init_pos",
                "init_rot",
                "init_local_pose",
                "source_pose_mode",
                "particle_flags",
                "particle_radius",
            }:
                raise ValueError(
                    "USD deformable overrides currently support identity, placement and particle flags/radius; physical properties stay authored by the source."
                )
            unknown = data.keys() - {item.name for item in fields(config_type)}
            if unknown:
                raise ValueError(
                    f"Unknown USD override fields at {source_path}: {sorted(unknown)}."
                )
            cfg = config_type.from_dict(data)
            if data.get("init_local_pose") is None:
                cfg.init_local_pose = None
                if {"init_pos", "init_rot"}.intersection(data):
                    cfg.init_local_pose = _pose_from_cfg(cfg)
            if not is_deformable and "body_scale" not in data:
                cfg.body_scale = None
            if (
                not is_articulation
                and not is_deformable
                and not is_visual
                and "body_type" not in data
            ):
                cfg.body_type = None
            pose_mode = (
                "auto" if source_options is None else source_options["pose_mode"]
            )
        elif configured is not None:
            expected_type = (
                VolumeDeformableObjectCfg
                if is_soft
                else (
                    SurfaceDeformableObjectCfg
                    if is_cloth
                    else (
                        ArticulationCfg
                        if is_articulation
                        else VisualObjectCfg if is_visual else RigidObjectCfg
                    )
                )
            )
            if not isinstance(configured, expected_type):
                raise ValueError(
                    f"USD configuration type does not match source entity {source_path}."
                )
            cfg = deepcopy(configured)
            if (is_articulation and cfg.fpath is not None) or (
                not is_articulation and not is_deformable and cfg.shape is not None
            ):
                raise ValueError(
                    f"Per-asset source selection is not supported: {source_path}."
                )
            if cfg.source is not None and (
                isinstance(cfg.source.file_path, str)
                or cfg.source.source_prim_path is not None
                or cfg.source.prim_path is not None
                or cfg.source.root_prim_path is not None
                or cfg.source.variants is not None
                or cfg.source.namespace is not None
            ):
                raise ValueError(
                    f"Per-asset source selection is not supported: {source_path}."
                )
            pose_mode = "auto" if cfg.source is None else cfg.source.pose_mode
        elif legacy_robot is not None:
            cfg = deepcopy(legacy_robot)
            cfg.uid = descriptor.name
            cfg.asset_physics_mode = "preserve"
            cfg.init_local_pose = None
            cfg.root_props.fixed_base = descriptor.fixed_base
            cfg.root_props.self_collision_enabled = descriptor.enable_self_collision
            pose_mode = "preserve"
        elif is_articulation:
            cfg = ArticulationCfg(build_pk_chain=False)
            cfg.init_local_pose = None
            cfg.body_scale = None
            pose_mode = "preserve"
        elif is_deformable:
            cfg = (
                VolumeDeformableObjectCfg() if is_soft else SurfaceDeformableObjectCfg()
            )
            pose_mode = "preserve"
        elif is_visual:
            cfg = VisualObjectCfg()
            cfg.body_scale = tuple(float(value) for value in descriptor.body_scale)
            pose_mode = "preserve"
        elif type(descriptor) is MeshObjectDesc:
            cfg = RigidObjectCfg()
            cfg.init_local_pose = None
            cfg.body_type = None
            cfg.body_scale = None
            pose_mode = "preserve"
        else:
            if descriptor.name in occupied:
                raise ValueError(f"USD asset UID already exists: {descriptor.name!r}.")
            occupied.add(descriptor.name)
            if root_pose is not None and hasattr(descriptor, "pose"):
                descriptor.pose = root_pose @ descriptor.pose
            continue

        uid = cfg.uid or descriptor.name
        if is_visual and uid in occupied:
            uid = f"{recipe.name}_{uid}"
        if uid in occupied:
            raise ValueError(f"USD asset UID already exists: {uid!r}.")
        occupied.add(uid)
        cfg.uid = uid
        cfg.source = deepcopy(recipe.source)
        cfg.source.source_prim_path = selected_path
        if cfg.source_pose_mode == "auto":
            cfg.source_pose_mode = pose_mode
        if is_deformable:
            from .usd import _configure_deformable_usd

            _configure_deformable_usd(cfg, descriptor)
            kind = "soft_object" if is_soft else "cloth_object"
            facade_type = VolumeDeformableObject if is_soft else SurfaceDeformableObject
            materials = {}
        elif is_articulation:
            cfg.fpath = None
            cfg.source.parse_kwargs()
            if configured is None and legacy_robot is None:
                cfg.root_props = ArticulationRootPropertiesCfg()
            descriptor, materials = _configure_articulation_usd(
                cfg,
                source,
                descriptor,
                source_path=file_path,
                newton_solver_type=manager._active_newton_solver_type,
            )
            configure_articulation_desc(
                descriptor, cfg, newton_solver_type=manager._active_newton_solver_type
            )
            kind = "articulation"
            facade_type = Robot if isinstance(cfg, RobotCfg) else Articulation
            cfg.root_props.fixed_base = bool(descriptor.fixed_base)
            cfg.root_props.self_collision_enabled = descriptor.enable_self_collision
        elif is_visual:
            descriptor.name = uid
            descriptor.pose = _source_pose(cfg, descriptor.pose)
            if cfg.body_scale is not None:
                from .descriptors import _vector3

                descriptor.body_scale = _vector3(
                    cfg.body_scale, field_name="body_scale"
                )
            cfg.body_scale = tuple(float(value) for value in descriptor.body_scale)
            kind = "visual_object"
            facade_type = VisualObject
            materials = {}
        else:
            cfg.shape = None
            cfg.source.parse_kwargs()
            if (
                descriptor.physics is None
                and cfg.resolve_asset_physics_mode() == "preserve"
            ):
                descriptor.name = uid
                descriptor.pose = _source_pose(cfg, descriptor.pose)
                cfg.body_type = "static"
                cfg.body_scale = tuple(float(value) for value in descriptor.body_scale)
                materials = {}
            else:
                descriptor, materials = _configure_rigid_usd(
                    cfg,
                    source,
                    descriptor,
                    newton_solver_type=manager._active_newton_solver_type,
                )
            kind = "rigid_object"
            facade_type = RigidObject
        if root_pose is not None:
            descriptor.pose = root_pose @ descriptor.pose
        _set_resolved_usd_pose(cfg, descriptor.pose)
        source.materials.update(materials)
        if isinstance(cfg, RobotCfg) and cfg.enable_sensor:
            for sensor_cfg in cfg.sensors:
                if not isinstance(sensor_cfg, CameraCfg):
                    raise TypeError("RobotCfg.sensors supports mounted cameras only.")
                sensor_factory = manager.SUPPORTED_SENSOR_TYPES.get(
                    sensor_cfg.sensor_type
                )
                if not isinstance(sensor_factory, type) or not issubclass(
                    sensor_factory, Camera
                ):
                    raise ValueError(
                        f"Unsupported mounted camera type {sensor_cfg.sensor_type!r}."
                    )
                if issubclass(sensor_factory, StereoCamera) and not isinstance(
                    sensor_cfg, StereoCameraCfg
                ):
                    raise TypeError(
                        "StereoCamera requires a StereoCameraCfg configuration."
                    )
                manager._require_native_renderer("Mounted cameras")
                if not isinstance(sensor_cfg.uid, str) or not sensor_cfg.uid.strip():
                    raise ValueError("Mounted sensors require a non-empty uid.")
                if sensor_cfg.uid in sensor_uids:
                    raise ValueError(f"Sensor {sensor_cfg.uid!r} already exists.")
                sensor_uids.add(sensor_cfg.uid)
                parent = sensor_cfg.extrinsics.parent
                if not isinstance(parent, str) or not parent.strip():
                    raise ValueError("Mounted sensors require a parent link.")
                link_name = parent.removeprefix(f"{uid}/")
                if not link_name or "/" in link_name:
                    raise ValueError(
                        f"Mounted sensor {sensor_cfg.uid!r} must refer to a local link on robot {uid!r}."
                    )
                if link_name not in {link.name for link in descriptor.links}:
                    raise ValueError(
                        f"Mounted sensor {sensor_cfg.uid!r} refers to unknown link {link_name!r} on robot {uid!r}."
                    )
                bound_cfg = deepcopy(sensor_cfg)
                bound_cfg.extrinsics.parent = f"{uid}/{link_name}"
                mounted_sensors.append(bound_cfg)
        facade = facade_type(cfg=cfg, device=manager.device)
        declarations.append((resource_keys[id(descriptor)], kind, descriptor, facade))
        renamed[original_name] = descriptor.name
        if isinstance(descriptor, MeshObjectDesc) and descriptor.physics is not None:
            rigid_renames[original_name] = descriptor.name

    for sensor_cfg in mounted_sensors:
        if sensor_cfg.uid in occupied:
            raise ValueError(
                f"Mounted sensor UID {sensor_cfg.uid!r} already exists in the scene."
            )

    def renamed_endpoint(endpoint: str) -> str:
        owner, separator, suffix = endpoint.partition("/")
        return renamed.get(owner, owner) + separator + suffix

    for constraint in source.constraints:
        if constraint.body0 is not None:
            constraint.body0 = renamed_endpoint(constraint.body0)
        elif root_pose is not None:
            constraint.origin_pose = root_pose @ constraint.origin_pose
        if constraint.body1 is not None:
            constraint.body1 = renamed_endpoint(constraint.body1)
        elif root_pose is not None:
            constraint.target_pose = root_pose @ constraint.target_pose
    if source.newton is not None:
        if source.newton.body_order is not None:
            source.newton.body_order = [
                rigid_renames.get(name, name) for name in source.newton.body_order
            ]
        for equality in source.newton.equalities:
            if equality.equality_type == 1:
                equality.object1 = rigid_renames.get(equality.object1, equality.object1)
                equality.object2 = rigid_renames.get(equality.object2, equality.object2)
    source.rigid_collision_filter_pairs = [
        (renamed_endpoint(first), renamed_endpoint(second))
        for first, second in source.rigid_collision_filter_pairs
    ]

    resource_uids = {resource_keys[id(item)]: item.name for item in resources}
    resource_source_paths = {
        resource_keys[id(item)]: source_paths_by_id[id(item)] for item in resources
    }
    public_kinds = {
        "soft_object": "volume_deformable",
        "cloth_object": "surface_deformable",
    }
    resource_types = {
        path: "robot" if isinstance(facade, Robot) else public_kinds.get(kind, kind)
        for path, kind, _, facade in declarations
    }
    for collection, kind in (
        ("cameras", "camera"),
        ("lights", "light"),
        ("records", "record"),
        ("cables", "cable"),
        ("constraints", "constraint"),
    ):
        resource_types.update(
            {resource_keys[id(item)]: kind for item in getattr(source, collection)}
        )
    constraint_uids = {
        resource_keys[id(item)]: item.name for item in source.constraints
    }
    return _UsdImportPlan(
        source,
        declarations,
        mounted_sensors,
        resource_uids,
        resource_types,
        constraint_uids,
        resource_source_paths,
    )
