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

"""Build route-specific runtime, prompt, and action-graph bundles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math
from pathlib import Path
from typing import Any

from embodichain.utils import logger
from embodichain.gen_sim.action_agent_pipeline.protocol.artifacts import (
    ACTION_AGENT_ENV_ID,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_light_config as _make_light_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_intent import (
    _arrangement_order_is_constrained,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_spec import (
    _with_arrangement_generated_pose_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.bundle_support import (
    _make_sensor_config_factory_for_robot,
    _runtime_object_registry,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _container_rigid_object_max_convex_hull_num,
    _make_arrangement_dataset_config,
    _make_arrangement_events_config,
    _make_background_config,
    _make_observations_config,
    _make_relative_dataset_config,
    _make_relative_rigid_object_config,
    _moved_rigid_object_max_convex_hull_num,
    _relative_rigid_object_max_convex_hull_num,
    _source_body_scale,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    ArrangementLineSpec,
    RelativePlacementSpec,
    StackingSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.glb_geometry_baking import (
    GlbGeometryNormalizer,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _mesh_config_world_zmax,
)
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_agent_config,
    make_arrangement_atom_actions_prompt,
    make_arrangement_basic_background,
    make_arrangement_task_prompt,
    make_relative_atom_actions_prompt,
    make_relative_basic_background,
    make_relative_task_prompt,
    make_stacking_atom_actions_prompt,
    make_stacking_basic_background,
    make_stacking_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_geometry import (
    _make_relative_summary,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_intent import (
    _relative_relation_phrase,
    _relative_scene_runtime_uid_mapping,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_offsets import (
    _side_relation_xy_offsets,
    _with_final_auto_arm_sides,
    _with_self_relative_absolute_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_slot_geometry import (
    _with_inside_container_slot_offsets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_surface_geometry import (
    _with_on_surface_release_offsets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_transport_geometry import (
    _with_coordinated_side_release_height_offsets,
    _with_coordinated_transport_geometry,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    RobotProfile,
)
from embodichain.gen_sim.action_agent_pipeline.generation.seed_task_graph import (
    make_arrangement_seed_task_graph,
    make_relative_seed_task_graph,
    make_stacking_seed_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _collect_scene_objects,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_transforms import (
    _apply_scene_z_rotation,
    _apply_source_scene_transforms,
    _maybe_apply_source_scene_body_scale,
    _maybe_apply_source_scene_xy_scale,
    _maybe_apply_tabletop_z_placement,
    _maybe_preserve_source_scene_vertical_contacts,
    _relative_generated_object_body_scale,
    _source_objects_by_runtime_uid,
    _source_scene_body_scale_override,
)
from embodichain.gen_sim.action_agent_pipeline.generation.stacking_spec import (
    _make_stacking_summary,
    _with_stacking_generated_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_graph_builders import (
    compile_arrangement_task_graph,
    compile_relative_task_graph,
    compile_stacking_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _make_arrangement_extensions_config,
    _make_relative_extensions_config,
    _make_stacking_extensions_config,
)

__all__ = [
    "_build_arrangement_line_bundle",
    "_build_relative_placement_bundle",
    "_build_stacking_bundle",
    "_make_stacking_dataset_config",
]


def _build_arrangement_line_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: ArrangementLineSpec,
    project_name: str,
    task_name: str,
    robot_profile: RobotProfile,
    target_body_scale: float | list[float] | tuple[float, float, float],
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: GlbGeometryNormalizer,
    source_scene_body_scale_mode: str | None,
    preserve_source_scene_geometry: bool,
    source_scene_z_rotation_degrees: float,
    arrangement_debug_visualization: bool,
    load_template_material: bool,
) -> dict[str, Any]:
    seed_task_graph = make_arrangement_seed_task_graph(task_name, spec)
    scene_objects = _collect_scene_objects(source_config)
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_scene_runtime_uid_mapping(
        scene_objects,
        table_source_uid=spec.table_source_uid,
    )
    moved_source_uids = {step.source_uid for step in spec.steps}
    for step in spec.steps:
        runtime_uids[step.source_uid] = step.runtime_uid

    dynamic_rigid_objects = [
        obj for obj in scene_objects if obj.source_uid != spec.table_source_uid
    ]
    table_obj = by_uid[spec.table_source_uid]
    table_config = _make_background_config(
        scene_dir,
        table_obj,
        mesh_normalizer,
    )
    _maybe_apply_source_scene_body_scale(
        table_config,
        table_obj,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    )
    table_top_z = _mesh_config_world_zmax(table_config)
    robot_config = robot_profile.make_robot_config(table_top_z)
    sensor_config_factory = _make_sensor_config_factory_for_robot(robot_config)

    gym_config = {
        "id": ACTION_AGENT_ENV_ID,
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": {},
            "events": _make_arrangement_events_config(
                [step.runtime_uid for step in spec.steps],
                sensor_config_factory=sensor_config_factory,
                task_name=task_name,
                load_template_material=load_template_material,
            ),
            "observations": _make_observations_config(robot_config),
            "dataset": {},
        },
        "robot": robot_config,
        "sensor": sensor_config_factory(),
        "light": _make_light_config(),
        "background": [table_config],
        "rigid_object": [
            _make_relative_rigid_object_config(
                scene_dir=scene_dir,
                obj=obj,
                runtime_uid=runtime_uids[obj.source_uid],
                body_scale=(
                    _source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    )
                    or _source_body_scale(obj)
                ),
                max_convex_hull_num=(
                    _moved_rigid_object_max_convex_hull_num(obj)
                    if obj.source_uid in moved_source_uids
                    else 1
                ),
                mesh_normalizer=mesh_normalizer,
            )
            for obj in dynamic_rigid_objects
        ],
    }
    _apply_source_scene_transforms(
        gym_config,
        runtime_uids=runtime_uids,
        by_uid=by_uid,
        table_top_z=table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
        source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
        robot_profile=robot_profile,
    )
    spec = _with_arrangement_generated_pose_targets(spec, gym_config)
    gym_config["env"]["extensions"] = _make_arrangement_extensions_config(
        spec,
        robot_profile=robot_profile,
    )
    if arrangement_debug_visualization:
        gym_config["env"]["extensions"]["arrangement_debug"] = (
            _make_arrangement_debug_config(spec)
        )
        for step in spec.steps:
            logger.log_info(
                "Arrangement debug slot "
                f"{step.slot_index}: object={step.runtime_uid}, "
                f"category={step.category}, arm={step.active_side}_arm, "
                f"target={step.release_position}, high={step.high_position}."
            )
    gym_config["env"]["dataset"] = _make_arrangement_dataset_config(
        project_name,
        spec,
        robot_profile=robot_profile,
    )
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_arrangement_task_prompt(
            task_name,
            project_name,
            spec,
            robot_profile=robot_profile,
        ),
        "seed_task_graph": seed_task_graph,
        "task_graph": compile_arrangement_task_graph(
            task_name,
            seed_task_graph,
            spec,
        ),
        "basic_background": make_arrangement_basic_background(
            project_name,
            spec,
            robot_profile=robot_profile,
            object_registry=_runtime_object_registry(runtime_uids, by_uid=by_uid),
        ),
        "atom_actions": make_arrangement_atom_actions_prompt(
            spec,
            robot_profile=robot_profile,
        ),
        "summary": {
            "robot_profile": robot_profile.summary(),
            **_make_arrangement_summary(spec),
        },
    }


def _make_arrangement_summary(spec: ArrangementLineSpec) -> dict[str, Any]:
    return {
        "mode": "arrangement_line",
        "order_constraint": (
            "ordered"
            if _arrangement_order_is_constrained(
                spec.order_by,
                task_description=spec.task_description,
            )
            else "free"
        ),
        "axis": spec.axis,
        "anchor": spec.anchor,
        "order_by": spec.order_by,
        "order_direction": spec.order_direction,
        "line_origin_xy": [
            float(spec.line_origin_xy[0]),
            float(spec.line_origin_xy[1]),
        ],
        "spacing": float(spec.spacing),
        "layout_clearance": float(spec.layout_clearance),
        "category_order": list(spec.category_order),
        "spatial_direction": spec.spatial_direction,
        "placements": [
            {
                "object": step.runtime_uid,
                "source_uid": step.source_uid,
                "slot_index": step.slot_index,
                "active_arm": f"{step.active_side}_arm",
                "target_xy": [float(step.target_xy[0]), float(step.target_xy[1])],
                "orientation_goal": step.orientation_goal,
                "orientation_axis": step.orientation_axis,
                "category": step.category,
                "cross_side": step.cross_side,
                "execution_index": step.execution_index,
                "blocked_by": list(step.blocked_by),
            }
            for step in spec.steps
        ],
    }


def _make_arrangement_debug_config(spec: ArrangementLineSpec) -> dict[str, Any]:
    return {
        "slots": [
            {
                "object": step.runtime_uid,
                "category": step.category,
                "arm": f"{step.active_side}_arm",
                "target": [float(value) for value in step.release_position],
                "high": [float(value) for value in step.high_position],
                "slot_index": step.slot_index,
                "cross_side": step.cross_side,
                "execution_index": step.execution_index,
                "blocked_by": list(step.blocked_by),
            }
            for step in spec.steps
        ]
    }


def _build_stacking_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: StackingSpec,
    project_name: str,
    task_name: str,
    robot_profile: RobotProfile,
    target_body_scale: float | list[float] | tuple[float, float, float],
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: GlbGeometryNormalizer,
    source_scene_body_scale_mode: str | None,
    preserve_source_scene_geometry: bool,
    source_scene_z_rotation_degrees: float,
    load_template_material: bool,
) -> dict[str, Any]:
    seed_task_graph = make_stacking_seed_task_graph(task_name, spec)
    scene_objects = _collect_scene_objects(source_config)
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_scene_runtime_uid_mapping(
        scene_objects,
        table_source_uid=spec.table_source_uid,
    )
    moved_source_uids = {step.source_uid for step in spec.steps}
    for step in spec.steps:
        runtime_uids[step.source_uid] = step.runtime_uid
    if spec.anchor_source_uid is not None and spec.anchor_runtime_uid is not None:
        runtime_uids[spec.anchor_source_uid] = spec.anchor_runtime_uid

    dynamic_rigid_objects = [
        obj for obj in scene_objects if obj.source_uid != spec.table_source_uid
    ]
    table_obj = by_uid[spec.table_source_uid]
    table_config = _make_background_config(
        scene_dir,
        table_obj,
        mesh_normalizer,
    )
    _maybe_apply_source_scene_body_scale(
        table_config,
        table_obj,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    )
    table_top_z = _mesh_config_world_zmax(table_config)
    robot_config = robot_profile.make_robot_config(table_top_z)
    sensor_config_factory = _make_sensor_config_factory_for_robot(robot_config)

    gym_config = {
        "id": ACTION_AGENT_ENV_ID,
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": {},
            "events": _make_arrangement_events_config(
                [step.runtime_uid for step in spec.steps]
                + (
                    [spec.anchor_runtime_uid]
                    if spec.anchor_runtime_uid is not None
                    else []
                ),
                sensor_config_factory=sensor_config_factory,
                task_name=task_name,
                load_template_material=load_template_material,
            ),
            "observations": _make_observations_config(robot_config),
            "dataset": {},
        },
        "robot": robot_config,
        "sensor": sensor_config_factory(),
        "light": _make_light_config(),
        "background": [table_config],
        "rigid_object": [
            _make_relative_rigid_object_config(
                scene_dir=scene_dir,
                obj=obj,
                runtime_uid=runtime_uids[obj.source_uid],
                body_scale=(
                    _source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    )
                    or _source_body_scale(obj)
                ),
                max_convex_hull_num=(
                    _moved_rigid_object_max_convex_hull_num(obj)
                    if obj.source_uid in moved_source_uids
                    else (
                        _container_rigid_object_max_convex_hull_num(obj)
                        if obj.source_uid == spec.anchor_source_uid
                        else 1
                    )
                ),
                mesh_normalizer=mesh_normalizer,
            )
            for obj in dynamic_rigid_objects
        ],
    }
    _apply_source_scene_transforms(
        gym_config,
        runtime_uids=runtime_uids,
        by_uid=by_uid,
        table_top_z=table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
        source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
        robot_profile=robot_profile,
    )
    spec = _with_stacking_generated_targets(spec, gym_config)
    gym_config["env"]["extensions"] = _make_stacking_extensions_config(
        spec,
        robot_profile=robot_profile,
    )
    gym_config["env"]["dataset"] = _make_stacking_dataset_config(
        project_name,
        spec,
        robot_profile=robot_profile,
    )
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_stacking_task_prompt(
            task_name,
            project_name,
            spec,
            robot_profile=robot_profile,
        ),
        "seed_task_graph": seed_task_graph,
        "task_graph": compile_stacking_task_graph(
            task_name,
            seed_task_graph,
            spec,
        ),
        "basic_background": make_stacking_basic_background(
            project_name,
            spec,
            robot_profile=robot_profile,
            object_registry=_runtime_object_registry(runtime_uids, by_uid=by_uid),
        ),
        "atom_actions": make_stacking_atom_actions_prompt(
            spec,
            robot_profile=robot_profile,
        ),
        "summary": {
            "robot_profile": robot_profile.summary(),
            **_make_stacking_summary(spec),
        },
    }


def _make_stacking_dataset_config(
    project_name: str,
    spec: StackingSpec,
    *,
    robot_profile: RobotProfile,
) -> dict[str, Any]:
    ordered = ", ".join(step.runtime_uid for step in spec.steps)
    anchor_text = (
        f"the object {spec.anchor_runtime_uid}"
        if spec.anchor_runtime_uid is not None
        else "the selected free table anchor"
    )
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "save_failed_episodes": True,
            "params": {
                "robot_meta": {
                    "robot_type": robot_profile.robot_meta_type,
                    "control_freq": 25,
                },
                "instruction": {
                    "lang": (
                        f"Stack the selected objects on {anchor_text} "
                        f"bottom-to-top as: {ordered}."
                    ),
                },
                "extra": {
                    "scene_type": project_name,
                    "task_description": spec.task_description,
                    "data_type": "sim",
                },
                "use_videos": True,
            },
        }
    }


def _validate_surface_release_clearance(surface_release_clearance: float) -> float:
    if isinstance(surface_release_clearance, bool):
        raise ValueError(
            "surface_release_clearance must be a finite non-negative number."
        )
    try:
        clearance = float(surface_release_clearance)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "surface_release_clearance must be a finite non-negative number."
        ) from exc
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError(
            "surface_release_clearance must be a finite non-negative number."
        )
    return clearance


def _with_relative_surface_release_clearance(
    spec: RelativePlacementSpec,
    surface_release_clearance: float,
) -> RelativePlacementSpec:
    placements = tuple(
        replace(placement, surface_clearance=surface_release_clearance)
        for placement in spec.placements
    )
    return replace(
        spec,
        placements=placements,
        surface_clearance=surface_release_clearance,
    )


def _build_relative_placement_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: RelativePlacementSpec,
    project_name: str,
    task_name: str,
    robot_profile: RobotProfile,
    target_body_scale: float | list[float] | tuple[float, float, float],
    preserve_source_target_body_scale: bool,
    source_target_body_scale_multiplier: float | None,
    source_scene_body_scale_mode: str | None,
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: GlbGeometryNormalizer,
    preserve_source_scene_geometry: bool,
    source_scene_z_rotation_degrees: float,
    inside_container_slot_distance_scale: float,
    surface_release_clearance: float,
    load_template_material: bool,
) -> dict[str, Any]:
    seed_task_graph = make_relative_seed_task_graph(task_name, spec)
    spec = _with_relative_surface_release_clearance(
        spec,
        _validate_surface_release_clearance(surface_release_clearance),
    )
    scene_objects = _collect_scene_objects(source_config)
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_scene_runtime_uid_mapping(
        scene_objects,
        table_source_uid=spec.table_source_uid,
    )
    moved_source_uids = {placement.moved_source_uid for placement in spec.placements}
    reference_runtime_uids = {
        placement.reference_runtime_uid
        for placement in spec.placements
        if placement.intent in {"place_relative", "coordinated_pickment"}
    }
    moved_runtime_uids = {placement.moved_runtime_uid for placement in spec.placements}
    registered_runtime_uids = sorted(moved_runtime_uids | reference_runtime_uids)
    dynamic_rigid_objects = [
        obj for obj in scene_objects if obj.source_uid != spec.table_source_uid
    ]
    table_obj = by_uid[spec.table_source_uid]
    table_config = _make_background_config(
        scene_dir,
        table_obj,
        mesh_normalizer,
    )
    _maybe_apply_source_scene_body_scale(
        table_config,
        table_obj,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    )
    table_top_z = _mesh_config_world_zmax(table_config)
    robot_config = robot_profile.make_robot_config(table_top_z)
    sensor_config_factory = _make_sensor_config_factory_for_robot(robot_config)

    gym_config = {
        "id": ACTION_AGENT_ENV_ID,
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": {},
            "events": _make_arrangement_events_config(
                registered_runtime_uids,
                sensor_config_factory=sensor_config_factory,
                task_name=task_name,
                load_template_material=load_template_material,
            ),
            "observations": _make_observations_config(robot_config),
            "dataset": {},
        },
        "robot": robot_config,
        "sensor": sensor_config_factory(),
        "light": _make_light_config(),
        "background": [table_config],
        "rigid_object": [
            _make_relative_rigid_object_config(
                scene_dir=scene_dir,
                obj=obj,
                runtime_uid=runtime_uids[obj.source_uid],
                body_scale=_relative_generated_object_body_scale(
                    obj,
                    moved_source_uids=moved_source_uids,
                    target_body_scale=target_body_scale,
                    preserve_source_target_body_scale=preserve_source_target_body_scale,
                    source_target_body_scale_multiplier=(
                        source_target_body_scale_multiplier
                    ),
                    source_scene_body_scale_mode=source_scene_body_scale_mode,
                ),
                max_convex_hull_num=_relative_rigid_object_max_convex_hull_num(
                    runtime_uids[obj.source_uid],
                    spec,
                ),
                mesh_normalizer=mesh_normalizer,
            )
            for obj in dynamic_rigid_objects
        ],
    }
    source_objects_by_runtime_uid = _source_objects_by_runtime_uid(
        runtime_uids, by_uid=by_uid
    )
    # Relative planning observes intermediate geometry, so this order is part of
    # the generated action contract rather than an interchangeable transform list.
    _maybe_apply_source_scene_xy_scale(
        gym_config,
        source_objects_by_runtime_uid,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    )
    _maybe_preserve_source_scene_vertical_contacts(
        gym_config,
        source_objects_by_runtime_uid,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
        robot_profile=robot_profile,
    )
    if spec.intent in {"place_relative", "coordinated_pickment"}:
        spec = _with_coordinated_side_release_height_offsets(
            spec,
            gym_config,
            table_reference_mode="skip",
        )
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)
    spec = _with_final_auto_arm_sides(spec, gym_config)
    if spec.intent in {"place_relative", "coordinated_pickment"}:
        spec = _with_coordinated_side_release_height_offsets(
            spec,
            gym_config,
            table_reference_mode="only",
        )
        spec = _with_self_relative_absolute_targets(spec, gym_config)
        spec = _with_inside_container_slot_offsets(
            spec,
            gym_config,
            slot_distance_scale=inside_container_slot_distance_scale,
        )
        spec = _with_on_surface_release_offsets(spec, gym_config)
        spec = _with_coordinated_transport_geometry(spec, gym_config)
    gym_config["env"]["extensions"] = _make_relative_extensions_config(
        spec,
        robot_profile=robot_profile,
        side_relation_xy_offsets=_side_relation_xy_offsets,
    )
    gym_config["env"]["dataset"] = _make_relative_dataset_config(
        project_name,
        spec,
        robot_profile=robot_profile,
        relation_phrase=_relative_relation_phrase,
    )
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_relative_task_prompt(
            task_name,
            project_name,
            spec,
            robot_profile=robot_profile,
        ),
        "seed_task_graph": seed_task_graph,
        "task_graph": compile_relative_task_graph(
            task_name,
            seed_task_graph,
            spec,
        ),
        "basic_background": make_relative_basic_background(
            project_name,
            spec,
            robot_profile=robot_profile,
            object_registry=_runtime_object_registry(runtime_uids, by_uid=by_uid),
        ),
        "atom_actions": make_relative_atom_actions_prompt(
            spec,
            robot_profile=robot_profile,
        ),
        "summary": {
            "robot_profile": robot_profile.summary(),
            **_make_relative_summary(spec),
        },
    }
