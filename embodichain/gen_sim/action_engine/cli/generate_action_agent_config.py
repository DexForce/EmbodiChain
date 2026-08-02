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

from embodichain.gen_sim.action_engine.generation import (
    generate_action_engine_config,
)

__all__ = ["build_parser", "cli"]

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
        help="Gym export directory or gym_config.json path.",
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
        help="Natural-language goal passed to the Task Agent planner.",
    )
    parser.add_argument(
        "--task_file",
        "--task-file",
        help="Optional UTF-8 file containing the natural-language goal.",
    )
    parser.add_argument(
        "--task-agent",
        "--task_agent",
        dest="task_agent",
        help="Optional Task Agent v1 JSON; bypasses natural-language planning.",
    )
    parser.add_argument(
        "--robot-profile",
        "--robot_profile",
        choices=_ROBOT_PROFILE_CHOICES,
        default="ur10",
        help="Robot template used in fast_gym_config.json.",
    )
    parser.add_argument(
        "--llm_model",
        "--llm-model",
        default=None,
        help="Optional planner model override.",
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
        default="preserve",
        help="How the requested xyz scale combines with source body_scale.",
    )
    parser.add_argument(
        "--body-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Positive xyz scale used by multiply or absolute policy.",
    )
    parser.add_argument(
        "--max_episodes",
        "--max-episodes",
        type=int,
        default=1,
        help="Episode count written to fast_gym_config.json.",
    )
    parser.add_argument(
        "--max_episode_steps",
        "--max-episode-steps",
        type=int,
        default=2000,
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
        task_agent=args.task_agent,
        robot_profile=args.robot_profile,
        llm_model=args.llm_model,
        source_scene_z_rotation_degrees=args.source_scene_z_rotation_degrees,
        body_scale_policy=args.body_scale_policy,
        body_scale=args.body_scale,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        max_episode_steps=args.max_episode_steps,
        randomize_scene=args.randomize_scene,
        randomize_table_material=args.randomize_table_material,
    )

    print(f"Generated gym config: {paths.gym_config}")
    print(f"Generated agent config: {paths.agent_config}")
    print(f"Generated Task Agent: {paths.task_agent}")
    print(f"Generated Execution Program: {paths.execution_program}")
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
    if args.task_agent:
        if args.task_description or args.task_file:
            raise ValueError(
                "--task-agent cannot be combined with a natural-language task."
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
