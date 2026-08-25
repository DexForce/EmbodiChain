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

"""CLI for generating Action Engine configs from a Prompt2Scene gym export."""

from __future__ import annotations

import argparse
from pathlib import Path

from embodichain.gen_sim.action_engine.config import generation_defaults
from embodichain.gen_sim.action_engine.generation import (
    generate_action_engine_config,
)

__all__ = ["build_parser", "cli"]

_GENERATION_DEFAULTS = generation_defaults()
_TASK_DEFAULTS = _GENERATION_DEFAULTS["task"]
_SCENE_DEFAULTS = _GENERATION_DEFAULTS["scene"]

_ROBOT_PROFILE_CHOICES = (
    "ur5",
    "ur10",
    "dual_ur5",
    "dual_ur10",
    "franka",
    "dual_franka",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone config-generation argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Plan and compile an Action Engine task from an exported tabletop "
            "gym project."
        )
    )
    parser.add_argument(
        "--gym_project",
        "--gym-project",
        required=True,
        help=(
            "Prompt2Scene task/export directory or gym_config.json/"
            "scene_config.json path."
        ),
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        required=True,
        help="Directory receiving canonical JSON artifacts and the Seed PNG.",
    )
    parser.add_argument(
        "--task_name",
        "--task-name",
        required=True,
        help="Stable task identifier stored in both programs.",
    )
    parser.add_argument(
        "--task_description",
        "--task-description",
        help="Natural-language goal passed to structured LLM interpretation.",
    )
    parser.add_argument(
        "--task_file",
        "--task-file",
        help="Optional UTF-8 file containing the natural-language goal.",
    )
    parser.add_argument(
        "--task-spec",
        "--task_spec",
        dest="task_spec",
        help=(
            "Optional existing Action Engine v2 TaskSpec JSON; bypasses text "
            "LLM interpretation and uses its role_bindings hand-off."
        ),
    )
    parser.add_argument(
        "--robot-profile",
        "--robot_profile",
        choices=_ROBOT_PROFILE_CHOICES,
        default=str(_TASK_DEFAULTS["default_robot_profile"]),
        help="Robot template used in fast_gym_config.json.",
    )
    parser.add_argument(
        "--gripper-model",
        "--gripper_model",
        choices=("pgi", "robotiq"),
        default=str(_TASK_DEFAULTS["default_gripper_model"]),
        help="Gripper asset, control, TCP, and grasp profile used by both arms.",
    )
    parser.add_argument(
        "--llm_model",
        "--llm-model",
        default=None,
        help="Optional planner model override.",
    )
    parser.add_argument(
        "--vlm_model",
        "--vlm-model",
        default=None,
        help="Optional online visual/planner model override stored for A/B runs.",
    )
    parser.add_argument(
        "--planning-mode",
        "--planning_mode",
        choices=("offline", "ab"),
        default="offline",
        help="Generate one offline bundle or an offline/online A/B bundle.",
    )
    parser.add_argument(
        "--source_scene_z_rotation_degrees",
        "--source-scene-z-rotation-degrees",
        type=float,
        default=None,
        help=(
            "World-frame scene rotation. Prompt2Scene exports default to -90 "
            "degrees; other inputs default to zero."
        ),
    )
    parser.add_argument(
        "--body-scale-policy",
        choices=("preserve", "multiply", "absolute"),
        default=str(_SCENE_DEFAULTS["body_scale_policy"]),
        help="How the requested xyz scale combines with source body_scale.",
    )
    parser.add_argument(
        "--body-scale",
        type=float,
        nargs=3,
        default=tuple(float(value) for value in _SCENE_DEFAULTS["body_scale"]),
        metavar=("X", "Y", "Z"),
        help="Positive xyz scale used by multiply or absolute policy.",
    )
    parser.add_argument(
        "--max_episodes",
        "--max-episodes",
        type=int,
        default=int(_TASK_DEFAULTS["max_episodes"]),
        help="Episode count written to fast_gym_config.json.",
    )
    parser.add_argument(
        "--max_episode_steps",
        "--max-episode-steps",
        type=int,
        default=int(_TASK_DEFAULTS["max_episode_steps"]),
        help="Per-episode step limit written to fast_gym_config.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing canonical artifacts in the output directory.",
    )
    parser.add_argument(
        "--randomize-scene",
        action="store_true",
        help="Randomize rigid-object poses and table height on every reset.",
    )
    parser.add_argument(
        "--randomize-table-material",
        action="store_true",
        help="Randomize the table material independently on every reset.",
    )
    return parser


def cli() -> None:
    """Generate and report the canonical Action Engine artifact bundle."""
    args = build_parser().parse_args()
    task_description = _resolve_task_description(args)
    paths = generate_action_engine_config(
        args.gym_project,
        args.output_dir,
        task_name=args.task_name,
        task_description=task_description,
        task_spec=args.task_spec,
        robot_profile=args.robot_profile,
        gripper_model=args.gripper_model,
        llm_model=args.llm_model,
        source_scene_z_rotation_degrees=args.source_scene_z_rotation_degrees,
        body_scale_policy=args.body_scale_policy,
        body_scale=args.body_scale,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        max_episode_steps=args.max_episode_steps,
        randomize_scene=args.randomize_scene,
        randomize_table_material=args.randomize_table_material,
        planning_mode=args.planning_mode,
        vlm_model=args.vlm_model,
    )

    print(f"Generated gym config: {paths.gym_config}")
    print(f"Generated agent config: {paths.agent_config}")
    print(f"Generated TaskSpec: {paths.task_spec}")
    print(f"Generated SceneRequirements: {paths.scene_requirements}")
    print(f"Generated SeedGraph: {paths.seed_task_graph}")
    print(f"Generated Seed graph PNG: {paths.seed_task_graph_png}")
    print(
        "Run with:\n"
        "python -m embodichain.gen_sim.action_engine.cli.run_agent "
        f"--task_name {args.task_name} "
        f'--gym_config "{paths.gym_config}" '
        f'--agent_config "{paths.agent_config}" '
        "--regenerate"
    )


def _resolve_task_description(args: argparse.Namespace) -> str:
    task_spec = getattr(args, "task_spec", None)
    if task_spec:
        if args.task_description or args.task_file:
            raise ValueError(
                "--task-spec cannot be combined with --task_description or "
                "--task_file."
            )
        return ""
    if args.task_description and args.task_file:
        raise ValueError("Use either --task_description or --task_file, not both.")
    if args.task_file:
        description = (
            Path(args.task_file).expanduser().read_text(encoding="utf-8").strip()
        )
    else:
        description = str(args.task_description or "").strip()
    if not description:
        raise ValueError(
            "--task_description (or --task_file) must provide a non-empty goal."
        )
    return description


if __name__ == "__main__":
    cli()
