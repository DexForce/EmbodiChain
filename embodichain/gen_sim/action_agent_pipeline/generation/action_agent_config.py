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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from embodichain.gen_sim.action_agent_pipeline.generation.config_io import (
    read_json as _read_json,
    raise_if_generated_files_exist as _raise_if_generated_files_exist,
    write_config_bundle as _write_config_bundle,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    _ArrangementLineSpec,
    GeneratedActionAgentConfigPaths,
    TargetReplacementSpec,
    _BasketTaskRoles,
    _RelativePlacementSpec,
    _ResolvedTargetReplacement,
    _SceneObject,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _collect_scene_objects,
    _infer_basket_task_roles,
    _infer_project_name,
    _resolve_gym_config_path,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_frame_normalization import (
    MeshFrameNormalizer,
)
from embodichain.gen_sim.action_agent_pipeline.generation.glb_io import read_glb
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_agent_config,
    make_arrangement_atom_actions_prompt,
    make_arrangement_basic_background,
    make_arrangement_task_prompt,
    make_basket_atom_actions_prompt,
    make_basket_basic_background,
    make_basket_task_prompt,
    make_relative_atom_actions_prompt,
    make_relative_basic_background,
    make_relative_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_spec import (
    _build_arrangement_line_spec_with_llm,
    _call_arrangement_task_llm,
    _is_arrangement_task_description,
    _with_arrangement_generated_z_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_dual_ur5_robot_config as _make_dual_ur5_robot_config,
    make_light_config as _make_light_config,
    make_sensor_config as _make_sensor_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _make_background_config,
    _make_arrangement_dataset_config,
    _make_arrangement_events_config,
    _make_container_background_config,
    _make_dataset_config,
    _make_events_config,
    _make_extra_background_config,
    _make_extra_rigid_object_config,
    _make_observations_config,
    _make_relative_background_object_config,
    _make_relative_dataset_config,
    _make_relative_events_config,
    _make_relative_rigid_object_config,
    _make_target_object_config,
    _relative_rigid_object_max_convex_hull_num,
    _relative_static_background_max_convex_hull_num,
    _source_body_scale,
    _target_body_scale_vector,
)
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _DUAL_UR5_ARM_COMPONENT_Z,
    _DUAL_UR5_TABLETOP_CLEARANCE,
    _TABLETOP_OBJECT_CLEARANCE,
    _apply_tabletop_z_placement,
    _dual_ur5_init_z_from_table_top,
    _mesh_config_world_z_bounds,
    _mesh_config_world_zmax,
    _resolve_table_mesh_world_zmax,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_geometry import (
    _POSE_SENSITIVE_STAGING_Z_DELTA,
    _STAGING_Z_DELTA,
    _inside_container_axis_offsets,
    _inside_container_slot_axis_and_distance,
    _make_relative_summary,
    _offset_position,
    _relative_release_offset,
    _side_relation_xy_offsets,
    _with_inside_container_slot_offsets,
    _with_on_surface_release_offsets,
    _with_self_relative_absolute_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_spec import (
    _build_relative_placement_spec_with_llm,
    _call_relative_task_llm,
    _normalize_relative_relation,
    _relative_relation_phrase,
    _relative_scene_runtime_uid_mapping,
)
from embodichain.gen_sim.action_agent_pipeline.generation.replacement_generation import (
    _apply_replacement_names,
    _normalize_target_replacements,
    _run_prompt2geometry_replacement,
    _run_target_replacements,
    _validate_target_replacement_sources,
)
from embodichain.gen_sim.action_agent_pipeline.generation.role_refinement import (
    _refine_roles_with_llm,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _make_arrangement_extensions_config,
    _make_extensions_config,
    _make_relative_extensions_config,
    _object_in_container_success,
    _validate_arrangement_bundle,
    _validate_bundle,
    _validate_relative_bundle,
    _validate_success_uids,
)

__all__ = [
    "GeneratedActionAgentConfigPaths",
    "TargetReplacementSpec",
    "generate_action_agent_config_from_project",
]


def generate_action_agent_config_from_project(
    gym_project: str | Path,
    output_dir: str | Path,
    *,
    task_name: str = "UR5BreadBasket",
    task_description: str | None = None,
    use_llm_roles: bool = False,
    llm_model: str | None = None,
    target_body_scale: float | list[float] | tuple[float, float, float] = 0.7,
    target_replacements: Sequence[TargetReplacementSpec] | None = None,
    sync_replacement_names: bool = False,
    reuse_target_replacements: bool = True,
    prewarm_coacd_cache: bool = True,
    overwrite: bool = False,
    max_episodes: int = 1,
    max_episode_steps: int = 1000,
) -> GeneratedActionAgentConfigPaths:
    """Generate action-agent configs from an exported gym project.

    This first-stage generator intentionally keeps the UR5BreadBasket task
    structure fixed: the left arm grasps the left target object, the right arm
    grasps the right target object, and both objects are placed into one
    basket-like container.

    Args:
        gym_project: Project root, formatted scene folder, ``gym_config.json``,
            or ``gym_config_merged.json``.
        output_dir: Destination config directory.
        task_name: Name passed to ``run_agent``.
        task_description: Optional natural-language relative-placement task.
            When provided, the generator asks the shared LLM for a constrained
            config-level task spec and generates prompts from that spec.
        use_llm_roles: If true, use an LLM only to refine object role mapping.
        llm_model: Optional model override for role refinement.
        target_body_scale: Uniform or xyz scale applied to generated target
            objects. Basket-like containers keep their source ``body_scale``.
        target_replacements: Optional prompt-generated GLB replacements for
            selected default basket target objects. Each replacement writes to
            ``<gym_project>/mesh_assets/<output_dir_name>`` and only affects the
            generated config, not the original source mesh file.
        sync_replacement_names: If true, update runtime target UIDs and prompts
            from the replacement prompts. If false, only mesh paths are replaced.
        reuse_target_replacements: If true, reuse an existing replacement GLB
            at the expected output path when it matches the requested prompt.
        prewarm_coacd_cache: If true, precompute environment-side CoACD cache
            files referenced by the generated gym config before writing it.
        overwrite: If false, fail when generated files already exist.
        max_episodes: Value written to ``fast_gym_config.json``.
        max_episode_steps: Value written to ``fast_gym_config.json``.

    Returns:
        Paths of generated config files.
    """

    output_dir_path = Path(output_dir).expanduser().resolve()
    _raise_if_generated_files_exist(output_dir_path, overwrite)

    input_path = Path(gym_project).expanduser().resolve()
    gym_config_path = _resolve_gym_config_path(input_path)
    scene_dir = gym_config_path.parent
    source_config = _read_json(gym_config_path)
    project_name = _infer_project_name(input_path, scene_dir)
    replacement_specs = _normalize_target_replacements(target_replacements)
    mesh_normalizer = MeshFrameNormalizer(
        output_dir=output_dir_path / "mesh_assets" / "normalized"
    )

    scene_objects = _collect_scene_objects(source_config)
    if task_description:
        if replacement_specs:
            raise ValueError(
                "target_replacements are only supported by the default basket "
                "template. Do not combine them with task_description."
            )
        if _is_arrangement_task_description(task_description):
            spec = _build_arrangement_line_spec_with_llm(
                scene_objects=scene_objects,
                project_name=project_name,
                scene_dir=scene_dir,
                task_description=task_description,
                model=llm_model,
                task_llm_caller=_call_arrangement_task_llm,
            )
            bundle = _build_arrangement_line_bundle(
                scene_dir=scene_dir,
                source_config=source_config,
                spec=spec,
                project_name=project_name,
                task_name=task_name,
                max_episodes=max_episodes,
                max_episode_steps=max_episode_steps,
                mesh_normalizer=mesh_normalizer,
            )
            _validate_arrangement_bundle(bundle, spec)
            _attach_mesh_normalization_summary(bundle, mesh_normalizer)
            if prewarm_coacd_cache:
                _attach_coacd_cache_summary(bundle)
            return _write_config_bundle(
                output_dir=output_dir_path,
                bundle=bundle,
                overwrite=overwrite,
            )
        spec = _build_relative_placement_spec_with_llm(
            scene_objects=scene_objects,
            project_name=project_name,
            task_description=task_description,
            model=llm_model,
            release_offset_fn=_relative_release_offset,
            staging_z_delta=_STAGING_Z_DELTA,
            pose_sensitive_staging_z_delta=_POSE_SENSITIVE_STAGING_Z_DELTA,
            task_llm_caller=_call_relative_task_llm,
        )
        bundle = _build_relative_placement_bundle(
            scene_dir=scene_dir,
            source_config=source_config,
            spec=spec,
            project_name=project_name,
            task_name=task_name,
            target_body_scale=target_body_scale,
            max_episodes=max_episodes,
            max_episode_steps=max_episode_steps,
            mesh_normalizer=mesh_normalizer,
        )
        _validate_relative_bundle(bundle, spec)
        _attach_mesh_normalization_summary(bundle, mesh_normalizer)
        if prewarm_coacd_cache:
            _attach_coacd_cache_summary(bundle)
        return _write_config_bundle(
            output_dir=output_dir_path,
            bundle=bundle,
            overwrite=overwrite,
        )

    roles = _infer_basket_task_roles(scene_objects)
    if use_llm_roles:
        roles = _refine_roles_with_llm(
            roles=roles,
            scene_objects=scene_objects,
            project_name=project_name,
            model=llm_model,
        )

    _validate_target_replacement_sources(roles, replacement_specs)
    resolved_replacements = _run_target_replacements(
        scene_dir=scene_dir,
        replacement_specs=replacement_specs,
        reuse_target_replacements=reuse_target_replacements,
        prompt2geometry_runner=_run_prompt2geometry_replacement,
    )
    if sync_replacement_names:
        roles = _apply_replacement_names(
            roles,
            resolved_replacements,
        )

    bundle = _build_ur5_basket_bundle(
        scene_dir=scene_dir,
        source_config=source_config,
        roles=roles,
        project_name=project_name,
        task_name=task_name,
        target_body_scale=target_body_scale,
        target_replacements=resolved_replacements,
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
        mesh_normalizer=mesh_normalizer,
    )
    _validate_bundle(bundle, roles)
    _attach_mesh_normalization_summary(bundle, mesh_normalizer)
    if prewarm_coacd_cache:
        _attach_coacd_cache_summary(bundle)
    return _write_config_bundle(
        output_dir=output_dir_path,
        bundle=bundle,
        overwrite=overwrite,
    )


def _build_ur5_basket_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    roles: _BasketTaskRoles,
    project_name: str,
    task_name: str,
    target_body_scale: float | list[float] | tuple[float, float, float],
    target_replacements: Sequence[_ResolvedTargetReplacement],
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: MeshFrameNormalizer,
) -> dict[str, Any]:
    scene_objects = _collect_scene_objects(source_config)
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    replacement_by_source_uid = {
        replacement.source_uid: replacement for replacement in target_replacements
    }
    object_scale = _target_body_scale_vector(target_body_scale)
    container_scale = _source_body_scale(by_uid[roles.container_source_uid])
    task_source_uids = {
        roles.container_source_uid,
        roles.left_target_source_uid,
        roles.right_target_source_uid,
    }
    extra_rigid_objects = [
        obj
        for obj in scene_objects
        if obj.source_role == "rigid_object" and obj.source_uid not in task_source_uids
    ]
    extra_background_objects = [
        obj
        for obj in scene_objects
        if obj.source_role == "background" and obj.source_uid != roles.table_source_uid
    ]
    table_config = _make_background_config(
        scene_dir,
        by_uid[roles.table_source_uid],
        mesh_normalizer,
    )
    table_top_z = _mesh_config_world_zmax(table_config)
    robot_init_z = _dual_ur5_init_z_from_table_top(table_top_z)

    gym_config = {
        "id": "AtomicActionsAgent-v3",
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": _make_extensions_config(roles),
            "events": _make_events_config(
                roles,
                sensor_config_factory=_make_sensor_config,
            ),
            "observations": _make_observations_config(),
            "dataset": _make_dataset_config(project_name, roles),
        },
        "robot": _make_dual_ur5_robot_config(robot_init_z=robot_init_z),
        "sensor": _make_sensor_config(),
        "light": _make_light_config(),
        "background": [
            table_config,
            _make_container_background_config(
                scene_dir,
                by_uid[roles.container_source_uid],
                roles.container_runtime_uid,
                container_scale,
                mesh_normalizer,
            ),
            *[
                _make_extra_background_config(scene_dir, obj, mesh_normalizer)
                for obj in extra_background_objects
            ],
        ],
        "rigid_object": [
            _make_target_object_config(
                scene_dir,
                by_uid[roles.right_target_source_uid],
                roles.right_target_runtime_uid,
                object_scale,
                mesh_normalizer,
                replacement_by_source_uid.get(roles.right_target_source_uid),
            ),
            _make_target_object_config(
                scene_dir,
                by_uid[roles.left_target_source_uid],
                roles.left_target_runtime_uid,
                object_scale,
                mesh_normalizer,
                replacement_by_source_uid.get(roles.left_target_source_uid),
            ),
            *[
                _make_extra_rigid_object_config(
                    scene_dir,
                    obj,
                    _source_body_scale(obj),
                    mesh_normalizer,
                )
                for obj in extra_rigid_objects
            ],
        ],
    }
    _apply_tabletop_z_placement(gym_config, table_top_z)
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_basket_task_prompt(task_name, project_name, roles),
        "basic_background": make_basket_basic_background(project_name, roles),
        "atom_actions": make_basket_atom_actions_prompt(roles),
        "summary": {
            "mode": "basket_template",
            "left_target": roles.left_target_runtime_uid,
            "right_target": roles.right_target_runtime_uid,
            "container": roles.container_runtime_uid,
            "target_replacements": [
                {
                    "source_uid": replacement.source_uid,
                    "prompt": replacement.prompt,
                    "output_dir_name": replacement.output_dir_name,
                    "mesh_path": replacement.mesh_path.as_posix(),
                    "runtime_noun": replacement.runtime_noun,
                    "reused": replacement.reused,
                }
                for replacement in target_replacements
            ],
        },
    }


def _build_arrangement_line_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: _ArrangementLineSpec,
    project_name: str,
    task_name: str,
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: MeshFrameNormalizer,
) -> dict[str, Any]:
    scene_objects = _collect_scene_objects(source_config)
    background_objects = [
        obj for obj in scene_objects if obj.source_role == "background"
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_scene_runtime_uid_mapping(
        scene_objects,
        table_source_uid=spec.table_source_uid,
    )
    moved_source_uids = {step.source_uid for step in spec.steps}
    for step in spec.steps:
        runtime_uids[step.source_uid] = step.runtime_uid

    dynamic_rigid_objects = [
        obj for obj in rigid_objects if obj.source_uid in moved_source_uids
    ]
    static_scene_objects = [
        obj for obj in rigid_objects if obj.source_uid not in moved_source_uids
    ]
    table_config = _make_background_config(
        scene_dir,
        by_uid[spec.table_source_uid],
        mesh_normalizer,
    )
    table_top_z = _mesh_config_world_zmax(table_config)
    robot_init_z = _dual_ur5_init_z_from_table_top(table_top_z)

    gym_config = {
        "id": "AtomicActionsAgent-v3",
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": {},
            "events": _make_arrangement_events_config(
                [step.runtime_uid for step in spec.steps],
                sensor_config_factory=_make_sensor_config,
            ),
            "observations": _make_observations_config(),
            "dataset": {},
        },
        "robot": _make_dual_ur5_robot_config(robot_init_z=robot_init_z),
        "sensor": _make_sensor_config(),
        "light": _make_light_config(),
        "background": [
            table_config,
            *[
                _make_relative_background_object_config(
                    scene_dir,
                    obj,
                    runtime_uids[obj.source_uid],
                    max_convex_hull_num=1,
                    mesh_normalizer=mesh_normalizer,
                )
                for obj in static_scene_objects
            ],
            *[
                _make_extra_background_config(
                    scene_dir,
                    obj,
                    mesh_normalizer,
                    runtime_uid=runtime_uids[obj.source_uid],
                )
                for obj in background_objects
                if obj.source_uid != spec.table_source_uid
            ],
        ],
        "rigid_object": [
            _make_relative_rigid_object_config(
                scene_dir=scene_dir,
                obj=obj,
                runtime_uid=runtime_uids[obj.source_uid],
                body_scale=_source_body_scale(obj),
                max_convex_hull_num=16,
                mesh_normalizer=mesh_normalizer,
            )
            for obj in dynamic_rigid_objects
        ],
    }
    _apply_tabletop_z_placement(gym_config, table_top_z)
    spec = _with_arrangement_generated_z_targets(spec, gym_config)
    gym_config["env"]["extensions"] = _make_arrangement_extensions_config(spec)
    gym_config["env"]["dataset"] = _make_arrangement_dataset_config(
        project_name,
        spec,
    )
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_arrangement_task_prompt(task_name, project_name, spec),
        "basic_background": make_arrangement_basic_background(project_name, spec),
        "atom_actions": make_arrangement_atom_actions_prompt(spec),
        "summary": {
            **_make_arrangement_summary(spec),
        },
    }


def _make_arrangement_summary(spec: _ArrangementLineSpec) -> dict[str, Any]:
    return {
        "mode": "arrangement_line",
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
        "placements": [
            {
                "object": step.runtime_uid,
                "source_uid": step.source_uid,
                "slot_index": step.slot_index,
                "active_arm": f"{step.active_side}_arm",
                "target_xy": [float(step.target_xy[0]), float(step.target_xy[1])],
                "orientation_goal": step.orientation_goal,
                "orientation_axis": step.orientation_axis,
            }
            for step in spec.steps
        ],
    }


def _attach_coacd_cache_summary(bundle: dict[str, Any]) -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation.coacd_cache import (
        prewarm_coacd_cache_for_gym_config,
    )

    bundle.setdefault("summary", {})["coacd_cache"] = (
        prewarm_coacd_cache_for_gym_config(bundle["gym_config"])
    )


def _attach_mesh_normalization_summary(
    bundle: dict[str, Any],
    mesh_normalizer: MeshFrameNormalizer,
) -> None:
    reports = mesh_normalizer.reports
    if reports:
        bundle.setdefault("summary", {})["normalized_meshes"] = reports


def _build_relative_placement_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: _RelativePlacementSpec,
    project_name: str,
    task_name: str,
    target_body_scale: float | list[float] | tuple[float, float, float],
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: MeshFrameNormalizer,
) -> dict[str, Any]:
    scene_objects = _collect_scene_objects(source_config)
    background_objects = [
        obj for obj in scene_objects if obj.source_role == "background"
    ]
    rigid_objects = [obj for obj in scene_objects if obj.source_role == "rigid_object"]
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    runtime_uids = _relative_scene_runtime_uid_mapping(
        scene_objects,
        table_source_uid=spec.table_source_uid,
    )
    moved_source_uids = {placement.moved_source_uid for placement in spec.placements}
    reference_runtime_uids = {
        placement.reference_runtime_uid for placement in spec.placements
    }
    registered_runtime_uids = sorted(
        {runtime_uids[obj.source_uid] for obj in rigid_objects} | reference_runtime_uids
    )
    dynamic_rigid_objects = [
        obj for obj in rigid_objects if obj.source_uid in moved_source_uids
    ]
    static_scene_objects = [
        obj for obj in rigid_objects if obj.source_uid not in moved_source_uids
    ]
    object_scale = _target_body_scale_vector(target_body_scale)
    table_config = _make_background_config(
        scene_dir,
        by_uid[spec.table_source_uid],
        mesh_normalizer,
    )
    table_top_z = _mesh_config_world_zmax(table_config)
    robot_init_z = _dual_ur5_init_z_from_table_top(table_top_z)

    gym_config = {
        "id": "AtomicActionsAgent-v3",
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": {},
            "events": _make_relative_events_config(
                spec,
                registered_runtime_uids,
                sensor_config_factory=_make_sensor_config,
            ),
            "observations": _make_observations_config(),
            "dataset": {},
        },
        "robot": _make_dual_ur5_robot_config(robot_init_z=robot_init_z),
        "sensor": _make_sensor_config(),
        "light": _make_light_config(),
        "background": [
            table_config,
            *[
                _make_relative_background_object_config(
                    scene_dir,
                    obj,
                    runtime_uids[obj.source_uid],
                    max_convex_hull_num=_relative_static_background_max_convex_hull_num(
                        runtime_uids[obj.source_uid],
                        spec,
                    ),
                    mesh_normalizer=mesh_normalizer,
                )
                for obj in static_scene_objects
            ],
            *[
                _make_extra_background_config(
                    scene_dir,
                    obj,
                    mesh_normalizer,
                    runtime_uid=runtime_uids[obj.source_uid],
                )
                for obj in background_objects
                if obj.source_uid != spec.table_source_uid
            ],
        ],
        "rigid_object": [
            _make_relative_rigid_object_config(
                scene_dir=scene_dir,
                obj=obj,
                runtime_uid=runtime_uids[obj.source_uid],
                body_scale=object_scale,
                max_convex_hull_num=_relative_rigid_object_max_convex_hull_num(
                    runtime_uids[obj.source_uid],
                    spec,
                ),
                mesh_normalizer=mesh_normalizer,
            )
            for obj in dynamic_rigid_objects
        ],
    }
    _apply_tabletop_z_placement(gym_config, table_top_z)
    spec = _with_self_relative_absolute_targets(spec, gym_config)
    spec = _with_inside_container_slot_offsets(spec, gym_config)
    spec = _with_on_surface_release_offsets(spec, gym_config)
    gym_config["env"]["extensions"] = _make_relative_extensions_config(
        spec,
        side_relation_xy_offsets=_side_relation_xy_offsets,
    )
    gym_config["env"]["dataset"] = _make_relative_dataset_config(
        project_name,
        spec,
        relation_phrase=_relative_relation_phrase,
    )
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_relative_task_prompt(task_name, project_name, spec),
        "basic_background": make_relative_basic_background(project_name, spec),
        "atom_actions": make_relative_atom_actions_prompt(spec),
        "summary": _make_relative_summary(spec),
    }
