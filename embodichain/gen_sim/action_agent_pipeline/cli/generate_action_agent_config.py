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

import argparse
from pathlib import Path

from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config import (
    TargetReplacementSpec,
    generate_action_agent_config_from_project,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    available_robot_profile_choices,
)
from embodichain.gen_sim.action_agent_pipeline.cli.target_replacements import (
    resolve_target_replacements,
)

__all__ = ["cli"]


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an action-agent config from an " "exported tabletop gym project."
        )
    )
    parser.add_argument(
        "--gym_project",
        type=str,
        required=True,
        help=(
            "Path to a project root, formatted tabletop scene folder, or "
            "gym_config.json/gym_config_merged.json. Directory inputs prefer "
            "gym_config_merged.json when available."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory for generated agent configs.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="UR5BreadBasket",
        help="Task name passed to run_agent.",
    )
    parser.add_argument(
        "--task_description",
        type=str,
        default=None,
        help=(
            "Simple natural-language relative-placement task. Providing this "
            "uses the LLM to generate a constrained config-level prompt/spec."
        ),
    )
    parser.add_argument(
        "--task_file",
        type=str,
        default=None,
        help="Optional text file containing --task_description.",
    )
    parser.add_argument(
        "--use_llm_roles",
        action="store_true",
        default=False,
        help=(
            "Use the shared LLM only to refine object role mapping. The task "
            "template and prompts remain deterministic."
        ),
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="Optional LLM model override for --use_llm_roles.",
    )
    parser.add_argument(
        "--robot-profile",
        "--robot_profile",
        choices=available_robot_profile_choices(),
        default=DEFAULT_ROBOT_PROFILE_ID,
        help=(
            "Robot profile used by action-agent config generation. Defaults to "
            f"{DEFAULT_ROBOT_PROFILE_ID}."
        ),
    )
    parser.add_argument(
        "--target_body_scale",
        type=float,
        default=0.7,
        help=(
            "Uniform body_scale for generated target objects. Basket-like "
            "containers keep their source body_scale."
        ),
    )
    parser.add_argument(
        "--preserve_source_target_body_scale",
        "--preserve-source-target-body-scale",
        action="store_true",
        default=False,
        help=(
            "Keep moved target objects at their source body_scale instead of "
            "using --target_body_scale."
        ),
    )
    parser.add_argument(
        "--source_scene_body_scale_mode",
        "--source-scene-body-scale-mode",
        "--target_body_scale_mode",
        "--target-body-scale-mode",
        choices=("preserve", "multiply", "absolute"),
        default=None,
        help=(
            "Optional source-scene body_scale policy for prompt2scene-style "
            "exports. preserve keeps source scales, multiply uses "
            "source_scale * --target_body_scale, and absolute sets every "
            "source-scene object to --target_body_scale."
        ),
    )
    parser.add_argument(
        "--preserve_source_scene_geometry",
        "--preserve-source-scene-geometry",
        action="store_true",
        default=False,
        help=(
            "Keep source z placement instead of re-snapping objects to the "
            "tabletop. GLBs are still normalized to OBJ."
        ),
    )
    parser.add_argument(
        "--source_scene_z_rotation_degrees",
        "--source-scene-z-rotation-degrees",
        type=float,
        default=0.0,
        help=(
            "World-frame Z rotation applied to generated scene object poses. "
            "Use -90 for prompt2scene exports that need action-agent alignment."
        ),
    )
    parser.add_argument(
        "--source_mesh_x_rotation_degrees",
        "--source-mesh-x-rotation-degrees",
        type=float,
        default=0.0,
        help=(
            "Local X-axis rotation baked into normalized GLB/GLTF meshes. "
            "Use 90 for prompt2scene exports that need mesh-frame alignment."
        ),
    )
    parser.add_argument(
        "--inside_container_slot_distance_scale",
        "--inside-container-slot-distance-scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for generated inside-container release slot offsets "
            "when multiple objects are placed into one container. Values below "
            "1.0 move release points closer to the container center."
        ),
    )
    parser.add_argument(
        "--target_replacement",
        "--target-replacement",
        dest="target_replacement",
        action="append",
        nargs="+",
        metavar="SOURCE_OR_PROMPT",
        default=[],
        help=(
            "Generate one replacement foreground interactive object. Repeat for "
            "0-N replacements. Accepts either PROMPT for auto-selection from "
            "numbered rigid_object targets, or SOURCE_UID PROMPT for explicit "
            "selection."
        ),
    )
    parser.add_argument(
        "--target_replacement1",
        "--target-replacement1",
        nargs="+",
        metavar="SOURCE_OR_PROMPT",
        default=None,
        help=(
            "Generate <gym_project>/mesh_assets/new1 from PROMPT and use it "
            "to replace SOURCE_UID in the generated config. PROMPT alone "
            "auto-selects the first numbered foreground rigid object."
        ),
    )
    parser.add_argument(
        "--target_replacement2",
        "--target-replacement2",
        nargs="+",
        metavar="SOURCE_OR_PROMPT",
        default=None,
        help=(
            "Generate <gym_project>/mesh_assets/new2 from PROMPT and use it "
            "to replace SOURCE_UID in the generated config. PROMPT alone "
            "auto-selects the second numbered foreground rigid object."
        ),
    )
    parser.add_argument(
        "--sync_replacement_names",
        "--sync-replacement-names",
        action="store_true",
        default=False,
        help=(
            "Also update replacement target runtime UIDs and generated prompts "
            "from the replacement prompts."
        ),
    )
    parser.add_argument(
        "--reuse_target_replacements",
        "--reuse-target-replacements",
        dest="reuse_target_replacements",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing prompt-generated replacement GLBs when the prompt "
            "and expected output name match. Defaults to true."
        ),
    )
    parser.add_argument(
        "--prewarm_coacd_cache",
        "--prewarm-coacd-cache",
        dest="prewarm_coacd_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Precompute environment CoACD cache files during config generation. "
            "Defaults to true."
        ),
    )
    parser.add_argument(
        "--convex_decomposition_method",
        "--convex-decomposition-method",
        choices=("vhacd", "visacd", "coacd"),
        default="vhacd",
        help=(
            "Convex decomposition backend written to generated mesh objects. "
            "'visacd' is accepted as an alias for 'vhacd'. Defaults to vhacd."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite generated files if they already exist.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=1,
        help="max_episodes value written to fast_gym_config.json.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="max_episode_steps value written to fast_gym_config.json.",
    )
    args = parser.parse_args()
    task_description = _resolve_task_description(args)
    target_replacements = _resolve_target_replacements(args)

    paths = generate_action_agent_config_from_project(
        gym_project=args.gym_project,
        output_dir=args.output_dir,
        task_name=args.task_name,
        task_description=task_description,
        use_llm_roles=args.use_llm_roles,
        llm_model=args.llm_model,
        robot_profile=args.robot_profile,
        target_body_scale=args.target_body_scale,
        preserve_source_target_body_scale=args.preserve_source_target_body_scale,
        source_scene_body_scale_mode=args.source_scene_body_scale_mode,
        preserve_source_scene_geometry=args.preserve_source_scene_geometry,
        source_scene_z_rotation_degrees=args.source_scene_z_rotation_degrees,
        source_mesh_x_rotation_degrees=args.source_mesh_x_rotation_degrees,
        inside_container_slot_distance_scale=args.inside_container_slot_distance_scale,
        target_replacements=target_replacements,
        sync_replacement_names=args.sync_replacement_names,
        reuse_target_replacements=args.reuse_target_replacements,
        convex_decomposition_method=args.convex_decomposition_method,
        prewarm_coacd_cache=args.prewarm_coacd_cache,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    print(f"Generated gym config: {paths.gym_config}")
    print(f"Generated agent config: {paths.agent_config}")
    print(f"Generated task prompt: {paths.task_prompt}")
    print(f"Generated task graph: {paths.task_graph}")
    print(f"Generated basic background: {paths.basic_background}")
    print(f"Generated atom actions: {paths.atom_actions}")
    if paths.summary:
        print("Generation summary:")
        for key, value in paths.summary.items():
            print(f"  {key}: {value}")
    print(
        "Run with:\n"
        "python -m embodichain.gen_sim.action_agent_pipeline.cli.run_agent "
        f"--task_name {args.task_name} "
        f'--gym_config "{paths.gym_config}" '
        f'--agent_config "{paths.agent_config}" '
        "--regenerate"
    )


def _resolve_task_description(args: argparse.Namespace) -> str | None:
    if args.task_description and args.task_file:
        raise ValueError("Use either --task_description or --task_file, not both.")
    if args.task_file:
        return Path(args.task_file).expanduser().read_text(encoding="utf-8").strip()
    if args.task_description:
        return args.task_description.strip()
    return None


def _resolve_target_replacements(
    args: argparse.Namespace,
) -> list[TargetReplacementSpec]:
    return resolve_target_replacements(
        args, TargetReplacementSpec, Path(args.gym_project)
    )


if __name__ == "__main__":
    cli()
