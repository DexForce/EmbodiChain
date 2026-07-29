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

from pathlib import Path

from embodichain.gen_sim.action_agent_pipeline.config.defaults import (
    DEFAULT_GENERATED_CONFIG_TASK_NAME,
    DEFAULT_MAX_EPISODES,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
    DEFAULT_TARGET_BODY_SCALE,
)
from embodichain.gen_sim.action_agent_pipeline.generation.arrangement_spec import (
    _build_arrangement_line_spec_from_response,
)
from embodichain.gen_sim.action_agent_pipeline.generation.bundle_finalization import (
    _finalize_and_write_bundle,
    _validate_acd_method,
)
from embodichain.gen_sim.action_agent_pipeline.generation.bundle_support import (
    _with_task_route_summary,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_bundle_builders import (
    _build_arrangement_line_bundle,
    _build_relative_placement_bundle,
    _build_stacking_bundle,
    _make_stacking_dataset_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_io import (
    read_json as _read_json,
    raise_if_generated_files_exist as _raise_if_generated_files_exist,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_types import (
    GeneratedActionAgentConfigPaths,
)
from embodichain.gen_sim.action_agent_pipeline.generation.glb_geometry_baking import (
    GlbGeometryNormalizer,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_offsets import (
    _POSE_SENSITIVE_STAGING_Z_DELTA,
    _STAGING_Z_DELTA,
    _relative_release_offset,
)
from embodichain.gen_sim.action_agent_pipeline.generation.relative_spec import (
    _build_object_manipulation_spec_from_response,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    RobotProfile,
    resolve_robot_profile,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_objects import (
    _collect_scene_objects,
    _infer_project_name,
    _resolve_gym_config_path,
)
from embodichain.gen_sim.action_agent_pipeline.generation.scene_transforms import (
    _validate_source_scene_body_scale_mode,
)
from embodichain.gen_sim.action_agent_pipeline.generation.stacking_spec import (
    _build_stacking_spec_from_response,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_interpretation import (
    _call_task_interpretation_llm,
    _interpret_task_with_llm,
)
from embodichain.gen_sim.action_agent_pipeline.generation.task_router import (
    _TASK_ROUTE_ARRANGEMENT_LINE,
    _TASK_ROUTE_OBJECT_MANIPULATION,
    _TASK_ROUTE_STACKING,
    _TASK_ROUTE_UNSUPPORTED,
)
from embodichain.gen_sim.action_agent_pipeline.generation.success_specs import (
    _validate_arrangement_bundle,
    _validate_relative_bundle,
    _validate_stacking_bundle,
)

__all__ = [
    "GeneratedActionAgentConfigPaths",
    "generate_action_agent_config_from_project",
]


def generate_action_agent_config_from_project(
    gym_project: str | Path,
    output_dir: str | Path,
    *,
    task_name: str = DEFAULT_GENERATED_CONFIG_TASK_NAME,
    task_description: str | None = None,
    llm_model: str | None = None,
    robot_profile: str | RobotProfile | None = DEFAULT_ROBOT_PROFILE_ID,
    target_body_scale: float | list[float] | tuple[float, float, float] = (
        DEFAULT_TARGET_BODY_SCALE
    ),
    preserve_source_target_body_scale: bool = False,
    source_target_body_scale_multiplier: float | None = None,
    source_scene_body_scale_mode: str | None = None,
    preserve_source_scene_geometry: bool = False,
    source_scene_z_rotation_degrees: float = 0.0,
    load_template_material: bool = False,
    inside_container_slot_distance_scale: float = 1.0,
    surface_release_clearance: float = DEFAULT_SURFACE_RELEASE_CLEARANCE,
    acd_method: str = "vhacd",
    arrangement_debug_visualization: bool = False,
    overwrite: bool = False,
    max_episodes: int = DEFAULT_MAX_EPISODES,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
) -> GeneratedActionAgentConfigPaths:
    """Generate action-agent configs from an exported gym project.

    ``task_description`` is required: one LLM interpretation selects a supported
    route (stacking, arrangement line, object manipulation) and its semantic
    intent. The matching deterministic generator then derives every pose, slot,
    and graph edge from scene geometry.

    Args:
        gym_project: Project root, formatted scene folder, ``gym_config.json``,
            or ``gym_config_merged.json``.
        output_dir: Destination config directory.
        task_name: Name passed to ``run_agent``.
        task_description: Natural-language task goal. Required; an empty value
            raises because there is no default task template to fall back to.
        llm_model: Optional model override for the combined task interpretation.
        robot_profile: Robot profile ID or profile instance used to generate the
            robot config, runtime arm-slot mapping, prompts, and dataset robot
            metadata. Defaults to ``dual_ur10``.
        target_body_scale: Uniform or xyz scale applied to generated target
            objects. Container-like objects keep their source ``body_scale``.
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
        source_scene_z_rotation_degrees: World-frame Z rotation applied to
            generated scene object poses after config generation. Mesh paths and
            scales are unchanged.
        load_template_material: If true, add a startup event that randomly
            selects a table texture from the packaged action-agent texture
            set. If false, preserve the source scene's table appearance.
        inside_container_slot_distance_scale: Multiplier for automatically
            generated inside-container slot offsets when multiple moved objects
            share one container. Values below ``1`` place release points closer
            to the container center.
        surface_release_clearance: Final object-bottom clearance above support
            surfaces for ``object_on_surface`` release moves.
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

    task_description = str(task_description or "").strip()
    if not task_description:
        raise ValueError(
            "task_description is required. Provide the natural-language task "
            "goal so the task router can select a supported route."
        )

    output_dir_path = Path(output_dir).expanduser().resolve()
    _raise_if_generated_files_exist(output_dir_path, overwrite)
    robot_profile = resolve_robot_profile(robot_profile)

    input_path = Path(gym_project).expanduser().resolve()
    gym_config_path = _resolve_gym_config_path(input_path)
    scene_dir = gym_config_path.parent
    source_config = _read_json(gym_config_path)
    project_name = _infer_project_name(input_path, scene_dir)
    source_scene_body_scale_mode = _validate_source_scene_body_scale_mode(
        source_scene_body_scale_mode
    )
    acd_method = _validate_acd_method(acd_method)
    mesh_normalizer = GlbGeometryNormalizer(
        output_dir=output_dir_path / "mesh_assets" / "normalized_glb",
    )

    scene_objects = _collect_scene_objects(source_config)
    interpretation = _interpret_task_with_llm(
        scene_objects=scene_objects,
        project_name=project_name,
        task_description=task_description,
        model=llm_model,
        task_llm_caller=_call_task_interpretation_llm,
    )
    task_route = interpretation.task_route
    if task_route.route == _TASK_ROUTE_STACKING:
        spec = _build_stacking_spec_from_response(
            response=interpretation.spec,
            scene_objects=scene_objects,
            scene_dir=scene_dir,
            task_description=task_description,
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
            acd_method=acd_method,
            overwrite=overwrite,
        )
    if task_route.route == _TASK_ROUTE_ARRANGEMENT_LINE:
        spec = _build_arrangement_line_spec_from_response(
            response=interpretation.spec,
            scene_objects=scene_objects,
            scene_dir=scene_dir,
            task_description=task_description,
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
            acd_method=acd_method,
            overwrite=overwrite,
        )
    if task_route.route == _TASK_ROUTE_UNSUPPORTED:
        raise ValueError(
            "Task router classified the task as unsupported: " f"{task_route.reason}"
        )
    if task_route.route != _TASK_ROUTE_OBJECT_MANIPULATION:
        raise ValueError(f"Unsupported task route: {task_route.route!r}.")
    spec = _build_object_manipulation_spec_from_response(
        response=interpretation.spec,
        scene_objects=scene_objects,
        task_description=task_description,
        release_offset_fn=_relative_release_offset,
        staging_z_delta=_STAGING_Z_DELTA,
        pose_sensitive_staging_z_delta=_POSE_SENSITIVE_STAGING_Z_DELTA,
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
        acd_method=acd_method,
        overwrite=overwrite,
    )
