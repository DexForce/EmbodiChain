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

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
import math
import warnings

from embodichain.gen_sim.action_agent_pipeline.defaults import (
    DEFAULT_MAX_EPISODES,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
    DEFAULT_TARGET_BODY_SCALE,
    DEFAULT_TASK_NAME,
)
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
    _StackingSpec,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _collect_scene_objects,
    _infer_basket_task_roles,
    _infer_project_name,
    _resolve_gym_config_path,
)
from embodichain.gen_sim.action_agent_pipeline.generation.naming import (
    _normalize_runtime_uid,
)
from embodichain.gen_sim.action_agent_pipeline.generation.glb_geometry_baking import (
    GlbGeometryNormalizer,
    bake_body_scale_into_glbs,
)
from embodichain.gen_sim.action_agent_pipeline.generation.glb_io import read_glb
from embodichain.gen_sim.action_agent_pipeline.generation.prompt_builders import (
    make_agent_config,
    make_arrangement_atom_actions_prompt,
    make_arrangement_basic_background,
    make_arrangement_task_graph,
    make_arrangement_task_prompt,
    make_basket_atom_actions_prompt,
    make_basket_basic_background,
    make_basket_task_graph,
    make_basket_task_prompt,
    make_relative_atom_actions_prompt,
    make_relative_basic_background,
    make_relative_task_graph,
    make_relative_task_prompt,
    make_stacking_atom_actions_prompt,
    make_stacking_basic_background,
    make_stacking_task_graph,
    make_stacking_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_spec import (
    _build_arrangement_line_spec_with_llm,
    _call_arrangement_task_llm,
    _with_arrangement_generated_pose_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.stacking_spec import (
    _build_stacking_spec_with_llm,
    _call_stacking_task_llm,
    _make_stacking_summary,
    _with_stacking_generated_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_light_config as _make_light_config,
    make_sensor_config as _make_sensor_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _clean_vector3,
    _container_rigid_object_max_convex_hull_num,
    _make_background_config,
    _make_arrangement_dataset_config,
    _make_arrangement_events_config,
    _make_container_rigid_object_config,
    _make_dataset_config,
    _make_events_config,
    _make_extra_rigid_object_config,
    _moved_rigid_object_max_convex_hull_num,
    _make_observations_config,
    _make_relative_dataset_config,
    _make_relative_events_config,
    _make_relative_rigid_object_config,
    _make_target_object_config,
    _relative_rigid_object_max_convex_hull_num,
    _source_body_scale,
    _target_body_scale_vector,
)
from embodichain.utils import logger
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_bounds import (
    _DUAL_UR5_ARM_COMPONENT_Z,
    _DUAL_UR5_TABLETOP_CLEARANCE,
    _TABLETOP_OBJECT_CLEARANCE,
    _apply_tabletop_z_placement,
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
    _with_coordinated_side_release_height_offsets,
    _with_final_auto_arm_sides,
    _with_inside_container_slot_offsets,
    _with_on_surface_release_offsets,
    _with_self_relative_absolute_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_spec import (
    _normalize_relative_relation,
    _relative_relation_phrase,
    _relative_scene_runtime_uid_mapping,
)
from embodichain.gen_sim.action_agent_pipeline.generation.object_manipulation_spec import (
    _build_object_manipulation_spec_with_llm,
    _call_object_manipulation_task_llm,
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
from embodichain.gen_sim.action_agent_pipeline.generation.task_router import (
    _TASK_ROUTE_ARRANGEMENT_LINE,
    _TASK_ROUTE_OBJECT_MANIPULATION,
    _TASK_ROUTE_STACKING,
    _TASK_ROUTE_UNSUPPORTED,
    _TaskRouteSpec,
    _call_task_router_llm,
    _route_task_with_llm,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _make_arrangement_extensions_config,
    _make_extensions_config,
    _make_relative_extensions_config,
    _make_stacking_extensions_config,
    _object_in_container_success,
    _validate_arrangement_bundle,
    _validate_bundle,
    _validate_relative_bundle,
    _validate_stacking_bundle,
)

_call_relative_task_llm = _call_object_manipulation_task_llm
_SOURCE_SCENE_BODY_SCALE_MODES = {"preserve", "multiply", "absolute"}

__all__ = [
    "GeneratedActionAgentConfigPaths",
    "TargetReplacementSpec",
    "generate_action_agent_config_from_project",
]


def generate_action_agent_config_from_project(
    gym_project: str | Path,
    output_dir: str | Path,
    *,
    task_name: str = DEFAULT_TASK_NAME,
    task_description: str | None = None,
    use_llm_roles: bool = False,
    llm_model: str | None = None,
    robot_profile: str | RobotProfile | None = DEFAULT_ROBOT_PROFILE_ID,
    target_body_scale: float | list[float] | tuple[float, float, float] = (
        DEFAULT_TARGET_BODY_SCALE
    ),
    preserve_source_target_body_scale: bool = False,
    source_target_body_scale_multiplier: float | None = None,
    source_scene_body_scale_mode: str | None = None,
    preserve_source_scene_geometry: bool = False,
    load_source_meshes_directly: bool = False,
    source_scene_z_rotation_degrees: float = 0.0,
    source_mesh_x_rotation_degrees: float = 0.0,
    load_template_material: bool = False,
    inside_container_slot_distance_scale: float = 1.0,
    surface_release_clearance: float = DEFAULT_SURFACE_RELEASE_CLEARANCE,
    target_replacements: Sequence[TargetReplacementSpec] | None = None,
    sync_replacement_names: bool = False,
    reuse_target_replacements: bool = True,
    acd_method: str = "vhacd",
    arrangement_debug_visualization: bool = False,
    overwrite: bool = False,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
) -> GeneratedActionAgentConfigPaths:
    """Generate action-agent configs from an exported gym project.

    For the default basket template, this first-stage generator keeps the task
    structure fixed: the left arm grasps the left target object, the right arm
    grasps the right target object, and both objects are placed into one
    basket-like container. The generated robot can be switched with
    ``robot_profile``.

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
        robot_profile: Robot profile ID or profile instance used to generate the
            robot config, runtime arm-slot mapping, prompts, and dataset robot
            metadata. Defaults to ``dual_ur10``.
        target_body_scale: Uniform or xyz scale applied to generated target
            objects. Basket-like containers keep their source ``body_scale``.
        preserve_source_target_body_scale: If true, moved target objects keep
            their source ``body_scale`` instead of using ``target_body_scale``.
            This is intended for metric-scaled prompt2scene exports.
        source_target_body_scale_multiplier: Optional multiplier applied to
            moved target objects' source ``body_scale``. When set, it takes
            precedence over ``preserve_source_target_body_scale`` and
            ``target_body_scale`` for relative-placement targets.
        source_scene_body_scale_mode: Optional source-scene scale policy for
            prompt2scene-style metric exports. ``preserve`` keeps source
            ``body_scale`` for every source-scene object, ``multiply`` applies
            ``target_body_scale`` as a multiplier to every source-scene
            ``body_scale``, and ``absolute`` sets every source-scene object to
            ``target_body_scale``. When unset, legacy target-only scale
            behavior is preserved.
        preserve_source_scene_geometry: If true, generated scene objects keep
            source z placement instead of re-snapping objects to the tabletop.
        load_source_meshes_directly: Deprecated compatibility option. Generated
            runtime assets are always normalized and baked GLB files.
        source_scene_z_rotation_degrees: World-frame Z rotation applied to
            generated scene object poses after config generation. Mesh paths and
            scales are unchanged.
        source_mesh_x_rotation_degrees: Deprecated compatibility option. GLB
            frame conversion is handled by the GLB geometry baker.
        load_template_material: If true, add a startup event that randomly
            selects a table texture from the packaged action-agent texture
            set. If false, preserve the source scene's table appearance.
        inside_container_slot_distance_scale: Multiplier for automatically
            generated inside-container slot offsets when multiple moved objects
            share one container. Values below ``1`` place release points closer
            to the container center.
        surface_release_clearance: Final object-bottom clearance above support
            surfaces for ``object_on_surface`` release moves.
        target_replacements: Optional prompt-generated GLB replacements for
            selected default basket target objects. Each replacement writes to
            ``<gym_project>/mesh_assets/<output_dir_name>`` and only affects the
            generated config, not the original source mesh file.
        sync_replacement_names: If true, update runtime target UIDs and prompts
            from the replacement prompts. If false, only mesh paths are replaced.
        reuse_target_replacements: If true, reuse an existing replacement GLB
            at the expected output path when it matches the requested prompt.
        acd_method: Convex decomposition backend written to generated mesh
            objects. Only ``"vhacd"`` is supported.
        arrangement_debug_visualization: If true, write target-slot and
            high-transport-point markers into the generated environment config.
        overwrite: If false, fail when generated files already exist.
        max_episodes: Value written to ``fast_gym_config.json``.
        max_episode_steps: Value written to ``fast_gym_config.json``.

    Returns:
        Paths of generated config files.
    """

    output_dir_path = Path(output_dir).expanduser().resolve()
    _raise_if_generated_files_exist(output_dir_path, overwrite)
    robot_profile = resolve_robot_profile(robot_profile)

    input_path = Path(gym_project).expanduser().resolve()
    gym_config_path = _resolve_gym_config_path(input_path)
    scene_dir = gym_config_path.parent
    source_config = _read_json(gym_config_path)
    project_name = _infer_project_name(input_path, scene_dir)
    replacement_specs = _normalize_target_replacements(target_replacements)
    source_scene_body_scale_mode = _validate_source_scene_body_scale_mode(
        source_scene_body_scale_mode
    )
    acd_method = _validate_acd_method(acd_method)
    mesh_normalizer = GlbGeometryNormalizer(
        output_dir=output_dir_path / "mesh_assets" / "normalized_glb",
    )

    scene_objects = _collect_scene_objects(source_config)
    if task_description:
        if replacement_specs:
            raise ValueError(
                "target_replacements are only supported by the default basket "
                "template. Do not combine them with task_description."
            )
        task_route = _route_task_with_llm(
            scene_objects=scene_objects,
            project_name=project_name,
            task_description=task_description,
            model=llm_model,
            task_router_llm_caller=_call_task_router_llm,
        )
        if task_route.route == _TASK_ROUTE_STACKING:
            spec = _build_stacking_spec_with_llm(
                scene_objects=scene_objects,
                project_name=project_name,
                scene_dir=scene_dir,
                task_description=task_description,
                model=llm_model,
                task_llm_caller=_call_stacking_task_llm,
            )
            bundle = _build_stacking_bundle(
                scene_dir=scene_dir,
                source_config=source_config,
                spec=spec,
                project_name=project_name,
                task_name=task_name,
                robot_profile=robot_profile,
                target_body_scale=target_body_scale,
                max_episodes=max_episodes,
                max_episode_steps=max_episode_steps,
                mesh_normalizer=mesh_normalizer,
                source_scene_body_scale_mode=source_scene_body_scale_mode,
                preserve_source_scene_geometry=preserve_source_scene_geometry,
                source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
                load_template_material=load_template_material,
            )
            _validate_stacking_bundle(bundle, spec)
            return _finalize_and_write_bundle(
                _with_task_route_summary(bundle, task_route),
                output_dir=output_dir_path,
                mesh_normalizer=mesh_normalizer,
                load_source_meshes_directly=load_source_meshes_directly,
                acd_method=acd_method,
                overwrite=overwrite,
            )
        if task_route.route == _TASK_ROUTE_ARRANGEMENT_LINE:
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
                robot_profile=robot_profile,
                target_body_scale=target_body_scale,
                max_episodes=max_episodes,
                max_episode_steps=max_episode_steps,
                mesh_normalizer=mesh_normalizer,
                source_scene_body_scale_mode=source_scene_body_scale_mode,
                preserve_source_scene_geometry=preserve_source_scene_geometry,
                source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
                arrangement_debug_visualization=arrangement_debug_visualization,
                load_template_material=load_template_material,
            )
            _validate_arrangement_bundle(bundle, spec)
            return _finalize_and_write_bundle(
                _with_task_route_summary(bundle, task_route),
                output_dir=output_dir_path,
                mesh_normalizer=mesh_normalizer,
                load_source_meshes_directly=load_source_meshes_directly,
                acd_method=acd_method,
                overwrite=overwrite,
            )
        if task_route.route == _TASK_ROUTE_UNSUPPORTED:
            raise ValueError(
                "Task router classified the task as unsupported: "
                f"{task_route.reason}"
            )
        if task_route.route != _TASK_ROUTE_OBJECT_MANIPULATION:
            raise ValueError(f"Unsupported task route: {task_route.route!r}.")
        spec = _build_object_manipulation_spec_with_llm(
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
            robot_profile=robot_profile,
            target_body_scale=target_body_scale,
            preserve_source_target_body_scale=preserve_source_target_body_scale,
            source_target_body_scale_multiplier=source_target_body_scale_multiplier,
            source_scene_body_scale_mode=source_scene_body_scale_mode,
            max_episodes=max_episodes,
            max_episode_steps=max_episode_steps,
            mesh_normalizer=mesh_normalizer,
            preserve_source_scene_geometry=preserve_source_scene_geometry,
            source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
            inside_container_slot_distance_scale=inside_container_slot_distance_scale,
            surface_release_clearance=surface_release_clearance,
            load_template_material=load_template_material,
        )
        _validate_relative_bundle(bundle, spec)
        return _finalize_and_write_bundle(
            _with_task_route_summary(bundle, task_route),
            output_dir=output_dir_path,
            mesh_normalizer=mesh_normalizer,
            load_source_meshes_directly=load_source_meshes_directly,
            acd_method=acd_method,
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

    bundle = _build_basket_bundle(
        scene_dir=scene_dir,
        source_config=source_config,
        roles=roles,
        project_name=project_name,
        task_name=task_name,
        robot_profile=robot_profile,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
        target_replacements=resolved_replacements,
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
        mesh_normalizer=mesh_normalizer,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
        source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
        load_template_material=load_template_material,
    )
    _validate_bundle(bundle, roles)
    return _finalize_and_write_bundle(
        bundle,
        output_dir=output_dir_path,
        mesh_normalizer=mesh_normalizer,
        load_source_meshes_directly=load_source_meshes_directly,
        acd_method=acd_method,
        overwrite=overwrite,
    )


def _make_sensor_config_for_robot(
    robot_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    sensors = _make_sensor_config()
    wrist_parent_by_uid = {
        "cam_wrist_left": _robot_solver_end_link(robot_config, "left_arm"),
        "cam_wrist_right": _robot_solver_end_link(robot_config, "right_arm"),
    }
    for sensor in sensors:
        parent = wrist_parent_by_uid.get(str(sensor.get("uid", "")))
        if not parent:
            continue
        extrinsics = sensor.get("extrinsics")
        if isinstance(extrinsics, dict):
            extrinsics["parent"] = parent
    return sensors


def _make_sensor_config_factory_for_robot(
    robot_config: Mapping[str, Any],
) -> Callable[[], list[dict[str, Any]]]:
    def sensor_config_factory() -> list[dict[str, Any]]:
        return _make_sensor_config_for_robot(robot_config)

    return sensor_config_factory


def _robot_solver_end_link(
    robot_config: Mapping[str, Any], arm_name: str
) -> str | None:
    solver_cfg = robot_config.get("solver_cfg", {})
    if not isinstance(solver_cfg, Mapping):
        return None
    arm_solver_cfg = solver_cfg.get(arm_name, {})
    if not isinstance(arm_solver_cfg, Mapping):
        return None
    end_link_name = arm_solver_cfg.get("end_link_name")
    if end_link_name is None:
        return None
    return str(end_link_name)


def _build_basket_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    roles: _BasketTaskRoles,
    project_name: str,
    task_name: str,
    robot_profile: RobotProfile,
    target_body_scale: float | list[float] | tuple[float, float, float],
    source_scene_body_scale_mode: str | None,
    target_replacements: Sequence[_ResolvedTargetReplacement],
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: GlbGeometryNormalizer,
    preserve_source_scene_geometry: bool,
    source_scene_z_rotation_degrees: float,
    load_template_material: bool,
) -> dict[str, Any]:
    scene_objects = _collect_scene_objects(source_config)
    by_uid = {obj.source_uid: obj for obj in scene_objects}
    replacement_by_source_uid = {
        replacement.source_uid: replacement for replacement in target_replacements
    }
    object_scale = _target_body_scale_vector(target_body_scale)
    container_obj = by_uid[roles.container_source_uid]
    container_scale = _source_scene_body_scale_override(
        container_obj,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    ) or _source_body_scale(container_obj)
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
    runtime_uids = {
        roles.table_source_uid: "table",
        roles.container_source_uid: roles.container_runtime_uid,
        roles.left_target_source_uid: roles.left_target_runtime_uid,
        roles.right_target_source_uid: roles.right_target_runtime_uid,
        **{
            obj.source_uid: _normalize_runtime_uid(obj.source_uid)
            for obj in [*extra_background_objects, *extra_rigid_objects]
        },
    }
    table_obj = by_uid[roles.table_source_uid]
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
        "id": "AtomicActionsAgent-v3",
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": _make_extensions_config(roles, robot_profile=robot_profile),
            "events": _make_events_config(
                roles,
                sensor_config_factory=sensor_config_factory,
                task_name=task_name,
                load_template_material=load_template_material,
            ),
            "observations": _make_observations_config(robot_config),
            "dataset": _make_dataset_config(
                project_name,
                roles,
                robot_profile=robot_profile,
            ),
        },
        "robot": robot_config,
        "sensor": sensor_config_factory(),
        "light": _make_light_config(),
        "background": [table_config],
        "rigid_object": [
            _make_container_rigid_object_config(
                scene_dir,
                by_uid[roles.container_source_uid],
                roles.container_runtime_uid,
                container_scale,
                mesh_normalizer,
            ),
            _make_target_object_config(
                scene_dir,
                by_uid[roles.right_target_source_uid],
                roles.right_target_runtime_uid,
                _source_scene_body_scale_override(
                    by_uid[roles.right_target_source_uid],
                    target_body_scale=target_body_scale,
                    source_scene_body_scale_mode=source_scene_body_scale_mode,
                )
                or object_scale,
                mesh_normalizer,
                replacement_by_source_uid.get(roles.right_target_source_uid),
            ),
            _make_target_object_config(
                scene_dir,
                by_uid[roles.left_target_source_uid],
                roles.left_target_runtime_uid,
                _source_scene_body_scale_override(
                    by_uid[roles.left_target_source_uid],
                    target_body_scale=target_body_scale,
                    source_scene_body_scale_mode=source_scene_body_scale_mode,
                )
                or object_scale,
                mesh_normalizer,
                replacement_by_source_uid.get(roles.left_target_source_uid),
            ),
            *[
                _make_extra_rigid_object_config(
                    scene_dir,
                    obj,
                    _source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    )
                    or _source_body_scale(obj),
                    mesh_normalizer,
                )
                for obj in extra_rigid_objects
            ],
            *[
                _make_extra_rigid_object_config(
                    scene_dir,
                    obj,
                    _source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    )
                    or _source_body_scale(obj),
                    mesh_normalizer,
                    runtime_uid=runtime_uids[obj.source_uid],
                )
                for obj in extra_background_objects
            ],
        ],
    }
    source_objects_by_runtime_uid = _source_objects_by_runtime_uid(
        runtime_uids, by_uid=by_uid
    )
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
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_basket_task_prompt(
            task_name,
            project_name,
            roles,
            robot_profile=robot_profile,
        ),
        "task_graph": make_basket_task_graph(task_name, roles),
        "basic_background": make_basket_basic_background(
            project_name,
            roles,
            robot_profile=robot_profile,
            object_registry=_runtime_object_registry(runtime_uids, by_uid=by_uid),
        ),
        "atom_actions": make_basket_atom_actions_prompt(
            roles,
            robot_profile=robot_profile,
        ),
        "summary": {
            "mode": "basket_template",
            "robot_profile": robot_profile.summary(),
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
        "id": "AtomicActionsAgent-v3",
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
    source_objects_by_runtime_uid = _source_objects_by_runtime_uid(
        runtime_uids, by_uid=by_uid
    )
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
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)
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
        "task_graph": make_arrangement_task_graph(task_name, spec),
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


def _make_arrangement_debug_config(spec: _ArrangementLineSpec) -> dict[str, Any]:
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


def _with_task_route_summary(
    bundle: Mapping[str, Any],
    route: _TaskRouteSpec,
) -> dict[str, Any]:
    routed_bundle = dict(bundle)
    routed_summary = dict(routed_bundle.get("summary", {}))
    routed_summary["task_route"] = route.to_summary()
    routed_bundle["summary"] = routed_summary
    return routed_bundle


def _build_stacking_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: _StackingSpec,
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
        "id": "AtomicActionsAgent-v3",
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
    source_objects_by_runtime_uid = _source_objects_by_runtime_uid(
        runtime_uids, by_uid=by_uid
    )
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
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)
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
        "task_graph": make_stacking_task_graph(task_name, spec),
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
    spec: _StackingSpec,
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


def _finalize_and_write_bundle(
    bundle: dict[str, Any],
    *,
    output_dir: Path,
    mesh_normalizer: GlbGeometryNormalizer,
    load_source_meshes_directly: bool,
    acd_method: str,
    overwrite: bool,
) -> GeneratedActionAgentConfigPaths:
    acd_method = _validate_acd_method(acd_method)
    _apply_acd_method(
        bundle["gym_config"],
        method=acd_method,
    )
    _attach_mesh_normalization_summary(bundle, mesh_normalizer)
    _attach_body_scale_bake_summary(bundle, output_dir)
    summary = bundle.setdefault("summary", {})
    summary["mesh_loading_mode"] = "baked_glb"
    summary["acd_method"] = acd_method
    summary.pop("convex_decomposition_method", None)
    return _write_config_bundle(
        output_dir=output_dir,
        bundle=bundle,
        overwrite=overwrite,
    )


def _validate_acd_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized != "vhacd":
        raise ValueError("acd_method must be 'vhacd'")
    return normalized


def _apply_acd_method(
    gym_config: dict[str, Any],
    *,
    method: str,
) -> None:
    for obj in _iter_generated_mesh_objects(gym_config):
        obj.pop("convex_decomposition_method", None)
        obj.pop("acd_method", None)
        shape = obj.get("shape")
        if isinstance(shape, MutableMapping):
            shape.pop("convex_decomposition_method", None)
            shape.pop("acd_method", None)

        max_convex_hull_num = int(obj.get("max_convex_hull_num", 1))
        if max_convex_hull_num > 1:
            obj["acd_method"] = method
            if isinstance(shape, MutableMapping):
                shape["acd_method"] = method


def _iter_generated_mesh_objects(
    gym_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    objects = []
    for section in ("background", "rigid_object"):
        value = gym_config.get(section, [])
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            continue
        for obj in value:
            if not isinstance(obj, dict):
                continue
            shape = obj.get("shape", {})
            if isinstance(shape, Mapping) and shape.get("shape_type") == "Mesh":
                objects.append(obj)
    return objects


def _attach_body_scale_bake_summary(
    bundle: dict[str, Any],
    output_dir: Path,
) -> None:
    reports = bake_body_scale_into_glbs(
        bundle["gym_config"],
        output_dir=output_dir / "mesh_assets" / "baked_glb",
    )
    if reports:
        bundle.setdefault("summary", {})["body_scaled_meshes"] = reports


def _attach_mesh_normalization_summary(
    bundle: dict[str, Any],
    mesh_normalizer: GlbGeometryNormalizer,
) -> None:
    if mesh_normalizer is None:
        return
    reports = mesh_normalizer.reports
    if reports:
        bundle.setdefault("summary", {})["normalized_meshes"] = reports


def _maybe_apply_tabletop_z_placement(
    gym_config: dict[str, Any],
    table_top_z: float | None,
    *,
    preserve_source_scene_geometry: bool,
) -> None:
    if preserve_source_scene_geometry:
        return
    _apply_tabletop_z_placement(gym_config, table_top_z)


def _source_objects_by_runtime_uid(
    runtime_uids_by_source_uid: Mapping[str, str],
    *,
    by_uid: Mapping[str, _SceneObject],
) -> dict[str, _SceneObject]:
    return {
        runtime_uid: by_uid[source_uid]
        for source_uid, runtime_uid in runtime_uids_by_source_uid.items()
        if source_uid in by_uid
    }


def _maybe_apply_source_scene_xy_scale(
    gym_config: dict[str, Any],
    source_objects_by_runtime_uid: Mapping[str, _SceneObject],
    *,
    source_scene_body_scale_mode: str | None,
) -> None:
    if source_scene_body_scale_mode in {None, "preserve"}:
        return

    anchor_xy = _scene_xy_scale_anchor(gym_config)
    for obj_config in _iter_scene_pose_configs(gym_config):
        runtime_uid = str(obj_config.get("uid", ""))
        source_obj = source_objects_by_runtime_uid.get(runtime_uid)
        if source_obj is None:
            continue
        _scale_scene_init_pos_xy_about_anchor(obj_config, source_obj, anchor_xy)


def _scene_xy_scale_anchor(gym_config: Mapping[str, Any]) -> list[float]:
    table_config = next(
        (
            obj_config
            for obj_config in _iter_scene_pose_configs(gym_config)
            if obj_config.get("uid") == "table"
        ),
        None,
    )
    if table_config is None:
        return [0.0, 0.0]
    init_pos = _clean_vector3(table_config.get("init_pos", [0.0, 0.0, 0.0]))
    return [init_pos[0], init_pos[1]]


def _scale_scene_init_pos_xy_about_anchor(
    obj_config: dict[str, Any],
    source_obj: _SceneObject,
    anchor_xy: Sequence[float],
) -> None:
    source_scale = _source_body_scale(source_obj)
    current_scale = _clean_vector3(obj_config.get("body_scale", [1.0, 1.0, 1.0]))
    ratio_x = _scale_ratio(current_scale[0], source_scale[0])
    ratio_y = _scale_ratio(current_scale[1], source_scale[1])
    if math.isclose(ratio_x, 1.0, rel_tol=0.0, abs_tol=1e-12) and math.isclose(
        ratio_y, 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        return

    init_pos = _clean_vector3(obj_config.get("init_pos", [0.0, 0.0, 0.0]))
    new_x = float(anchor_xy[0]) + (init_pos[0] - anchor_xy[0]) * ratio_x
    new_y = float(anchor_xy[1]) + (init_pos[1] - anchor_xy[1]) * ratio_y
    obj_config["init_pos"] = [
        _round_pose_value(new_x),
        _round_pose_value(new_y),
        _round_pose_value(init_pos[2]),
    ]


def _scale_ratio(current: float, source: float) -> float:
    if math.isclose(float(source), 0.0, rel_tol=0.0, abs_tol=1e-12):
        return 1.0
    return float(current) / float(source)


def _maybe_preserve_source_scene_vertical_contacts(
    gym_config: dict[str, Any],
    source_objects_by_runtime_uid: Mapping[str, _SceneObject],
    *,
    preserve_source_scene_geometry: bool,
    source_scene_body_scale_mode: str | None,
    robot_profile: RobotProfile | str | None = None,
) -> None:
    if not preserve_source_scene_geometry:
        return
    if source_scene_body_scale_mode in {None, "preserve"}:
        return

    for obj_config in _iter_scene_pose_configs(gym_config):
        runtime_uid = str(obj_config.get("uid", ""))
        source_obj = source_objects_by_runtime_uid.get(runtime_uid)
        if source_obj is None:
            continue
        _preserve_source_scene_vertical_boundary(obj_config, source_obj)
    _sync_robot_init_z_to_current_tabletop(gym_config, robot_profile=robot_profile)


def _preserve_source_scene_vertical_boundary(
    obj_config: dict[str, Any],
    source_obj: _SceneObject,
) -> None:
    source_scale = _source_body_scale(source_obj)
    current_scale = _clean_vector3(obj_config.get("body_scale", [1.0, 1.0, 1.0]))
    if all(
        math.isclose(source, current, rel_tol=0.0, abs_tol=1e-12)
        for source, current in zip(source_scale, current_scale)
    ):
        return

    source_config = dict(obj_config)
    source_config["body_scale"] = source_scale
    source_bounds = _mesh_config_world_z_bounds(source_config)
    current_bounds = _mesh_config_world_z_bounds(obj_config)
    if source_bounds is None or current_bounds is None:
        return

    boundary_index = 1 if obj_config.get("uid") == "table" else 0
    delta_z = source_bounds[boundary_index] - current_bounds[boundary_index]
    if math.isclose(delta_z, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return

    init_pos = _clean_vector3(obj_config.get("init_pos", [0.0, 0.0, 0.0]))
    init_pos[2] = _round_pose_value(init_pos[2] + delta_z)
    obj_config["init_pos"] = init_pos


def _sync_robot_init_z_to_current_tabletop(
    gym_config: dict[str, Any],
    *,
    robot_profile: RobotProfile | str | None = None,
) -> None:
    robot_config = gym_config.get("robot")
    if not isinstance(robot_config, dict):
        return

    table_config = next(
        (
            obj_config
            for obj_config in _iter_scene_pose_configs(gym_config)
            if obj_config.get("uid") == "table"
        ),
        None,
    )
    if table_config is None:
        return

    table_top_z = _mesh_config_world_zmax(table_config)
    if table_top_z is None:
        return

    profile = resolve_robot_profile(
        robot_profile
        or gym_config.get("env", {}).get("extensions", {}).get("agent_robot_profile")
    )
    init_pos = _clean_vector3(robot_config.get("init_pos", [0.0, 0.0, 0.0]))
    init_pos[2] = profile.robot_init_z_from_table_top(table_top_z)
    robot_config["init_pos"] = init_pos


def _apply_scene_z_rotation(
    gym_config: dict[str, Any],
    rotation_degrees: float,
) -> None:
    if not rotation_degrees:
        return
    for obj in _iter_scene_pose_configs(gym_config):
        _rotate_pose_about_world_z(obj, rotation_degrees)


def _iter_scene_pose_configs(gym_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for section in ("background", "rigid_object"):
        value = gym_config.get(section, [])
        if isinstance(value, Mapping):
            value = [value]
        if not isinstance(value, list):
            continue
        objects.extend(obj for obj in value if isinstance(obj, dict))
    return objects


def _rotate_pose_about_world_z(
    obj_config: dict[str, Any],
    rotation_degrees: float,
) -> None:
    position = _clean_vector3(obj_config.get("init_pos", [0.0, 0.0, 0.0]))
    theta = math.radians(float(rotation_degrees))
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    obj_config["init_pos"] = [
        _round_pose_value(position[0] * cos_theta - position[1] * sin_theta),
        _round_pose_value(position[0] * sin_theta + position[1] * cos_theta),
        _round_pose_value(position[2]),
    ]

    from scipy.spatial.transform import Rotation

    # Prompt2Scene exports and RigidObject.reset both use intrinsic XYZ Euler angles.
    rotation = _clean_vector3(obj_config.get("init_rot", [0.0, 0.0, 0.0]))
    original = Rotation.from_euler("XYZ", rotation, degrees=True)
    world_z = Rotation.from_rotvec([0.0, 0.0, theta])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected")
        rotated_euler = (world_z * original).as_euler("XYZ", degrees=True)
    obj_config["init_rot"] = [_round_pose_value(value) for value in rotated_euler]


def _round_pose_value(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 1e-12 else rounded


def _runtime_object_registry(
    runtime_uids_by_source_uid: Mapping[str, str],
    *,
    by_uid: Mapping[str, _SceneObject],
) -> list[dict[str, str]]:
    entries = []
    for source_uid, runtime_uid in sorted(
        runtime_uids_by_source_uid.items(),
        key=lambda item: item[1],
    ):
        obj = by_uid.get(source_uid)
        if obj is None:
            continue
        entries.append(
            {
                "runtime_uid": str(runtime_uid),
                "source_uid": str(source_uid),
                "source_role": obj.source_role,
                "description": str(obj.config.get("description", "")).strip(),
            }
        )
    return entries


def _validate_source_scene_body_scale_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized not in _SOURCE_SCENE_BODY_SCALE_MODES:
        expected = ", ".join(sorted(_SOURCE_SCENE_BODY_SCALE_MODES))
        raise ValueError(f"source_scene_body_scale_mode must be one of: {expected}")
    return normalized


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
    spec: _RelativePlacementSpec,
    surface_release_clearance: float,
) -> _RelativePlacementSpec:
    placements = tuple(
        replace(placement, surface_clearance=surface_release_clearance)
        for placement in spec.placements
    )
    return replace(
        spec,
        placements=placements,
        surface_clearance=surface_release_clearance,
    )


def _source_scene_body_scale(
    obj: _SceneObject,
    *,
    target_body_scale: float | list[float] | tuple[float, float, float],
    mode: str,
) -> list[float]:
    if mode == "preserve":
        return _source_body_scale(obj)
    if mode == "multiply":
        target_scale = _target_body_scale_vector(target_body_scale)
        return [
            _round_pose_value(source * multiplier)
            for source, multiplier in zip(_source_body_scale(obj), target_scale)
        ]
    if mode == "absolute":
        return _target_body_scale_vector(target_body_scale)
    raise AssertionError(f"Unhandled source scene body_scale mode: {mode}")


def _source_scene_body_scale_override(
    obj: _SceneObject,
    *,
    target_body_scale: float | list[float] | tuple[float, float, float],
    source_scene_body_scale_mode: str | None,
) -> list[float] | None:
    if source_scene_body_scale_mode is None:
        return None
    return _source_scene_body_scale(
        obj,
        target_body_scale=target_body_scale,
        mode=source_scene_body_scale_mode,
    )


def _maybe_apply_source_scene_body_scale(
    obj_config: dict[str, Any],
    obj: _SceneObject,
    *,
    target_body_scale: float | list[float] | tuple[float, float, float],
    source_scene_body_scale_mode: str | None,
) -> None:
    body_scale = _source_scene_body_scale_override(
        obj,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    )
    if body_scale is not None:
        obj_config["body_scale"] = body_scale


def _relative_target_body_scale(
    obj: _SceneObject,
    *,
    target_body_scale: float | list[float] | tuple[float, float, float],
    preserve_source_target_body_scale: bool,
    source_target_body_scale_multiplier: float | None,
    source_scene_body_scale_mode: str | None,
) -> list[float]:
    if source_scene_body_scale_mode is not None:
        return _source_scene_body_scale(
            obj,
            target_body_scale=target_body_scale,
            mode=source_scene_body_scale_mode,
        )
    if source_target_body_scale_multiplier is not None:
        multiplier = float(source_target_body_scale_multiplier)
        return [
            _round_pose_value(value * multiplier) for value in _source_body_scale(obj)
        ]
    if preserve_source_target_body_scale:
        return _source_body_scale(obj)
    return _target_body_scale_vector(target_body_scale)


def _relative_generated_object_body_scale(
    obj: _SceneObject,
    *,
    moved_source_uids: set[str],
    target_body_scale: float | list[float] | tuple[float, float, float],
    preserve_source_target_body_scale: bool,
    source_target_body_scale_multiplier: float | None,
    source_scene_body_scale_mode: str | None,
) -> list[float]:
    if obj.source_uid in moved_source_uids:
        return _relative_target_body_scale(
            obj,
            target_body_scale=target_body_scale,
            preserve_source_target_body_scale=preserve_source_target_body_scale,
            source_target_body_scale_multiplier=source_target_body_scale_multiplier,
            source_scene_body_scale_mode=source_scene_body_scale_mode,
        )
    return _source_scene_body_scale_override(
        obj,
        target_body_scale=target_body_scale,
        source_scene_body_scale_mode=source_scene_body_scale_mode,
    ) or _source_body_scale(obj)


def _build_relative_placement_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: _RelativePlacementSpec,
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
        "id": "AtomicActionsAgent-v3",
        "max_episodes": int(max_episodes),
        "max_episode_steps": int(max_episode_steps),
        "env": {
            "extensions": {},
            "events": _make_relative_events_config(
                spec,
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
        "task_graph": make_relative_task_graph(task_name, spec),
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
