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
import math

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
from embodichain.gen_sim.action_agent_pipeline.generation.mesh_frame_normalization import (
    MeshFrameNormalizer,
)
from embodichain.gen_sim.action_agent_pipeline.generation.body_scale_baking import (
    bake_body_scale_into_meshes,
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
    make_stacking_atom_actions_prompt,
    make_stacking_basic_background,
    make_stacking_task_prompt,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_spec import (
    _build_arrangement_line_spec_with_llm,
    _call_arrangement_task_llm,
    _is_arrangement_task_description,
    _with_arrangement_generated_z_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.stacking_spec import (
    _build_stacking_spec_with_llm,
    _call_stacking_task_llm,
    _is_stacking_task_description,
    _make_stacking_summary,
    _with_stacking_generated_targets,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_dual_ur5_robot_config as _make_dual_ur5_robot_config,
    make_light_config as _make_light_config,
    make_sensor_config as _make_sensor_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _clean_vector3,
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
    task_name: str = "UR5BreadBasket",
    task_description: str | None = None,
    use_llm_roles: bool = False,
    llm_model: str | None = None,
    target_body_scale: float | list[float] | tuple[float, float, float] = 0.7,
    preserve_source_target_body_scale: bool = False,
    source_target_body_scale_multiplier: float | None = None,
    source_scene_body_scale_mode: str | None = None,
    preserve_source_scene_geometry: bool = False,
    source_scene_z_rotation_degrees: float = 0.0,
    source_mesh_x_rotation_degrees: float = 0.0,
    target_replacements: Sequence[TargetReplacementSpec] | None = None,
    sync_replacement_names: bool = False,
    reuse_target_replacements: bool = True,
    convex_decomposition_method: str = "vhacd",
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
            GLB/GLTF meshes are still normalized to OBJ for consistent
            action-agent runtime loading.
        source_scene_z_rotation_degrees: World-frame Z rotation applied to
            generated scene object poses after config generation. Mesh paths and
            scales are unchanged.
        source_mesh_x_rotation_degrees: Local X-axis rotation baked into
            normalized GLB/GLTF meshes. Keep this at ``0`` for the legacy
            image2scene path; prompt2scene uses ``90`` to match the action-agent
            runtime mesh frame.
        target_replacements: Optional prompt-generated GLB replacements for
            selected default basket target objects. Each replacement writes to
            ``<gym_project>/mesh_assets/<output_dir_name>`` and only affects the
            generated config, not the original source mesh file.
        sync_replacement_names: If true, update runtime target UIDs and prompts
            from the replacement prompts. If false, only mesh paths are replaced.
        reuse_target_replacements: If true, reuse an existing replacement GLB
            at the expected output path when it matches the requested prompt.
        convex_decomposition_method: Convex decomposition backend written to
            generated mesh objects whose ``max_convex_hull_num`` is larger than
            one. ``"vhacd"`` is the action-agent default; ``"visacd"`` is
            accepted as an alias for ``"vhacd"``.
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
    repo_root = _repo_root_from_gym_config_path(gym_config_path)
    source_config = _read_json(gym_config_path)
    project_name = _infer_project_name(input_path, scene_dir)
    replacement_specs = _normalize_target_replacements(target_replacements)
    source_scene_body_scale_mode = _validate_source_scene_body_scale_mode(
        source_scene_body_scale_mode
    )
    convex_decomposition_method = _normalize_convex_decomposition_method(
        convex_decomposition_method
    )
    mesh_normalizer = MeshFrameNormalizer(
        output_dir=output_dir_path / "mesh_assets" / "normalized",
        local_x_correction_degrees=source_mesh_x_rotation_degrees,
    )

    scene_objects = _collect_scene_objects(source_config)
    if task_description:
        if replacement_specs:
            raise ValueError(
                "target_replacements are only supported by the default basket "
                "template. Do not combine them with task_description."
            )
        if _is_stacking_task_description(task_description):
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
                target_body_scale=target_body_scale,
                max_episodes=max_episodes,
                max_episode_steps=max_episode_steps,
                mesh_normalizer=mesh_normalizer,
                source_scene_body_scale_mode=source_scene_body_scale_mode,
                preserve_source_scene_geometry=preserve_source_scene_geometry,
                source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
            )
            _validate_stacking_bundle(bundle, spec)
            return _finalize_and_write_bundle(
                bundle,
                output_dir=output_dir_path,
                mesh_normalizer=mesh_normalizer,
                repo_root=repo_root,
                convex_decomposition_method=convex_decomposition_method,
                prewarm_coacd_cache=prewarm_coacd_cache,
                overwrite=overwrite,
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
                target_body_scale=target_body_scale,
                max_episodes=max_episodes,
                max_episode_steps=max_episode_steps,
                mesh_normalizer=mesh_normalizer,
                source_scene_body_scale_mode=source_scene_body_scale_mode,
                preserve_source_scene_geometry=preserve_source_scene_geometry,
                source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
            )
            _validate_arrangement_bundle(bundle, spec)
            return _finalize_and_write_bundle(
                bundle,
                output_dir=output_dir_path,
                mesh_normalizer=mesh_normalizer,
                repo_root=repo_root,
                convex_decomposition_method=convex_decomposition_method,
                prewarm_coacd_cache=prewarm_coacd_cache,
                overwrite=overwrite,
            )
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
            target_body_scale=target_body_scale,
            preserve_source_target_body_scale=preserve_source_target_body_scale,
            source_target_body_scale_multiplier=source_target_body_scale_multiplier,
            source_scene_body_scale_mode=source_scene_body_scale_mode,
            max_episodes=max_episodes,
            max_episode_steps=max_episode_steps,
            mesh_normalizer=mesh_normalizer,
            preserve_source_scene_geometry=preserve_source_scene_geometry,
            source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
        )
        _validate_relative_bundle(bundle, spec)
        return _finalize_and_write_bundle(
            bundle,
            output_dir=output_dir_path,
            mesh_normalizer=mesh_normalizer,
            repo_root=repo_root,
            convex_decomposition_method=convex_decomposition_method,
            prewarm_coacd_cache=prewarm_coacd_cache,
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
        source_scene_body_scale_mode=source_scene_body_scale_mode,
        target_replacements=resolved_replacements,
        max_episodes=max_episodes,
        max_episode_steps=max_episode_steps,
        mesh_normalizer=mesh_normalizer,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
        source_scene_z_rotation_degrees=source_scene_z_rotation_degrees,
    )
    _validate_bundle(bundle, roles)
    return _finalize_and_write_bundle(
        bundle,
        output_dir=output_dir_path,
        mesh_normalizer=mesh_normalizer,
        repo_root=repo_root,
        convex_decomposition_method=convex_decomposition_method,
        prewarm_coacd_cache=prewarm_coacd_cache,
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
    source_scene_body_scale_mode: str | None,
    target_replacements: Sequence[_ResolvedTargetReplacement],
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: MeshFrameNormalizer | None,
    preserve_source_scene_geometry: bool,
    source_scene_z_rotation_degrees: float,
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
    robot_init_z = _dual_ur5_init_z_from_table_top(table_top_z)
    robot_config = _make_dual_ur5_robot_config(robot_init_z=robot_init_z)

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
            "observations": _make_observations_config(robot_config),
            "dataset": _make_dataset_config(project_name, roles),
        },
        "robot": robot_config,
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
                _make_extra_background_config(
                    scene_dir,
                    obj,
                    mesh_normalizer,
                    body_scale=_source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    ),
                )
                for obj in extra_background_objects
            ],
        ],
        "rigid_object": [
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
        ],
    }
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)
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
    target_body_scale: float | list[float] | tuple[float, float, float],
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: MeshFrameNormalizer | None,
    source_scene_body_scale_mode: str | None,
    preserve_source_scene_geometry: bool,
    source_scene_z_rotation_degrees: float,
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
    robot_init_z = _dual_ur5_init_z_from_table_top(table_top_z)
    robot_config = _make_dual_ur5_robot_config(robot_init_z=robot_init_z)

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
            "observations": _make_observations_config(robot_config),
            "dataset": {},
        },
        "robot": robot_config,
        "sensor": _make_sensor_config(),
        "light": _make_light_config(),
        "background": [
            table_config,
            *[
                _make_relative_background_object_config(
                    scene_dir,
                    obj,
                    runtime_uids[obj.source_uid],
                    body_scale=_source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    ),
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
                    body_scale=_source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    ),
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
                body_scale=(
                    _source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    )
                    or _source_body_scale(obj)
                ),
                max_convex_hull_num=16,
                mesh_normalizer=mesh_normalizer,
            )
            for obj in dynamic_rigid_objects
        ],
    }
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)
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


def _build_stacking_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: _StackingSpec,
    project_name: str,
    task_name: str,
    target_body_scale: float | list[float] | tuple[float, float, float],
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: MeshFrameNormalizer | None,
    source_scene_body_scale_mode: str | None,
    preserve_source_scene_geometry: bool,
    source_scene_z_rotation_degrees: float,
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
    robot_init_z = _dual_ur5_init_z_from_table_top(table_top_z)
    robot_config = _make_dual_ur5_robot_config(robot_init_z=robot_init_z)

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
            "observations": _make_observations_config(robot_config),
            "dataset": {},
        },
        "robot": robot_config,
        "sensor": _make_sensor_config(),
        "light": _make_light_config(),
        "background": [
            table_config,
            *[
                _make_relative_background_object_config(
                    scene_dir,
                    obj,
                    runtime_uids[obj.source_uid],
                    body_scale=_source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    ),
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
                    body_scale=_source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    ),
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
                body_scale=(
                    _source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    )
                    or _source_body_scale(obj)
                ),
                max_convex_hull_num=16,
                mesh_normalizer=mesh_normalizer,
            )
            for obj in dynamic_rigid_objects
        ],
    }
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)
    spec = _with_stacking_generated_targets(spec, gym_config)
    gym_config["env"]["extensions"] = _make_stacking_extensions_config(spec)
    gym_config["env"]["dataset"] = _make_stacking_dataset_config(project_name, spec)
    return {
        "gym_config": gym_config,
        "agent_config": make_agent_config(),
        "task_prompt": make_stacking_task_prompt(task_name, project_name, spec),
        "basic_background": make_stacking_basic_background(project_name, spec),
        "atom_actions": make_stacking_atom_actions_prompt(spec),
        "summary": _make_stacking_summary(spec),
    }


def _make_stacking_dataset_config(
    project_name: str,
    spec: _StackingSpec,
) -> dict[str, Any]:
    ordered = ", ".join(step.runtime_uid for step in spec.steps)
    return {
        "lerobot": {
            "func": "LeRobotRecorder",
            "mode": "save",
            "params": {
                "robot_meta": {
                    "robot_type": "DualUR5",
                    "control_freq": 25,
                },
                "instruction": {
                    "lang": (
                        "Move the selected objects to the table center and stack "
                        f"them bottom-to-top as: {ordered}."
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
    mesh_normalizer: MeshFrameNormalizer | None,
    repo_root: Path | None = None,
    convex_decomposition_method: str,
    prewarm_coacd_cache: bool,
    overwrite: bool,
) -> GeneratedActionAgentConfigPaths:
    convex_decomposition_method = _normalize_convex_decomposition_method(
        convex_decomposition_method
    )
    _apply_convex_decomposition_method(
        bundle["gym_config"],
        method=convex_decomposition_method,
    )
    _attach_mesh_normalization_summary(bundle, mesh_normalizer)
    _attach_body_scale_bake_summary(bundle, output_dir)
    bundle.setdefault("summary", {})[
        "convex_decomposition_method"
    ] = convex_decomposition_method
    if prewarm_coacd_cache and convex_decomposition_method == "coacd":
        _attach_coacd_cache_summary(bundle, repo_root=repo_root)
    elif prewarm_coacd_cache:
        _attach_skipped_coacd_cache_summary(
            bundle,
            convex_decomposition_method=convex_decomposition_method,
        )
    return _write_config_bundle(
        output_dir=output_dir,
        bundle=bundle,
        overwrite=overwrite,
    )


def _normalize_convex_decomposition_method(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized == "visacd":
        normalized = "vhacd"
    if normalized not in {"coacd", "vhacd"}:
        raise ValueError(
            "convex_decomposition_method must be one of: 'vhacd', 'visacd', 'coacd'"
        )
    return normalized


def _apply_convex_decomposition_method(
    gym_config: dict[str, Any],
    *,
    method: str,
) -> None:
    for obj in _iter_generated_mesh_objects(gym_config):
        max_convex_hull_num = int(obj.get("max_convex_hull_num", 1))
        if max_convex_hull_num > 1:
            obj["convex_decomposition_method"] = method


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


def _attach_skipped_coacd_cache_summary(
    bundle: dict[str, Any],
    *,
    convex_decomposition_method: str,
) -> None:
    needs_decomposition = any(
        int(obj.get("max_convex_hull_num", 1)) > 1
        for obj in _iter_generated_mesh_objects(bundle["gym_config"])
    )
    bundle.setdefault("summary", {})["coacd_cache"] = (
        [
            {
                "status": "skipped",
                "reason": (
                    "convex_decomposition_method="
                    f"{convex_decomposition_method}; environment loading uses ACD "
                    "without CoACD prewarm"
                ),
            }
        ]
        if needs_decomposition
        else []
    )


def _attach_coacd_cache_summary(
    bundle: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> None:
    from embodichain.gen_sim.action_agent_pipeline.generation.coacd_cache import (
        prewarm_coacd_cache_for_gym_config,
    )

    bundle.setdefault("summary", {})["coacd_cache"] = (
        prewarm_coacd_cache_for_gym_config(bundle["gym_config"], repo_root=repo_root)
    )


def _repo_root_from_gym_config_path(gym_config_path: Path) -> Path:
    for parent in gym_config_path.resolve().parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent
    return gym_config_path.resolve().parent


def _attach_body_scale_bake_summary(
    bundle: dict[str, Any],
    output_dir: Path,
) -> None:
    reports = bake_body_scale_into_meshes(
        bundle["gym_config"],
        output_dir=output_dir / "mesh_assets" / "body_scaled",
    )
    if reports:
        bundle.setdefault("summary", {})["body_scaled_meshes"] = reports


def _attach_mesh_normalization_summary(
    bundle: dict[str, Any],
    mesh_normalizer: MeshFrameNormalizer | None,
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

    rotation = _clean_vector3(obj_config.get("init_rot", [0.0, 0.0, 0.0]))
    if abs(rotation[0]) < 1e-12 and abs(rotation[1]) < 1e-12:
        obj_config["init_rot"] = [
            0.0,
            0.0,
            _normalize_degrees(rotation[2] + float(rotation_degrees)),
        ]
        return

    from scipy.spatial.transform import Rotation

    original = Rotation.from_euler("xyz", rotation, degrees=True)
    world_z = Rotation.from_euler("z", float(rotation_degrees), degrees=True)
    obj_config["init_rot"] = [
        _round_pose_value(value)
        for value in (world_z * original).as_euler("xyz", degrees=True)
    ]


def _normalize_degrees(value: float) -> float:
    normalized = (float(value) + 180.0) % 360.0 - 180.0
    return _round_pose_value(180.0 if normalized == -180.0 else normalized)


def _round_pose_value(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 1e-12 else rounded


def _validate_source_scene_body_scale_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    normalized = str(mode).strip().lower()
    if normalized not in _SOURCE_SCENE_BODY_SCALE_MODES:
        expected = ", ".join(sorted(_SOURCE_SCENE_BODY_SCALE_MODES))
        raise ValueError(f"source_scene_body_scale_mode must be one of: {expected}")
    return normalized


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


def _build_relative_placement_bundle(
    *,
    scene_dir: Path,
    source_config: Mapping[str, Any],
    spec: _RelativePlacementSpec,
    project_name: str,
    task_name: str,
    target_body_scale: float | list[float] | tuple[float, float, float],
    preserve_source_target_body_scale: bool,
    source_target_body_scale_multiplier: float | None,
    source_scene_body_scale_mode: str | None,
    max_episodes: int,
    max_episode_steps: int,
    mesh_normalizer: MeshFrameNormalizer | None,
    preserve_source_scene_geometry: bool,
    source_scene_z_rotation_degrees: float,
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
        placement.reference_runtime_uid
        for placement in spec.placements
        if placement.intent == "place_relative"
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
    robot_init_z = _dual_ur5_init_z_from_table_top(table_top_z)
    robot_config = _make_dual_ur5_robot_config(robot_init_z=robot_init_z)

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
            "observations": _make_observations_config(robot_config),
            "dataset": {},
        },
        "robot": robot_config,
        "sensor": _make_sensor_config(),
        "light": _make_light_config(),
        "background": [
            table_config,
            *[
                _make_relative_background_object_config(
                    scene_dir,
                    obj,
                    runtime_uids[obj.source_uid],
                    body_scale=_source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    ),
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
                    body_scale=_source_scene_body_scale_override(
                        obj,
                        target_body_scale=target_body_scale,
                        source_scene_body_scale_mode=source_scene_body_scale_mode,
                    ),
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
                body_scale=_relative_target_body_scale(
                    obj,
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
    _maybe_apply_tabletop_z_placement(
        gym_config,
        table_top_z,
        preserve_source_scene_geometry=preserve_source_scene_geometry,
    )
    _apply_scene_z_rotation(gym_config, source_scene_z_rotation_degrees)
    if spec.intent == "place_relative":
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
