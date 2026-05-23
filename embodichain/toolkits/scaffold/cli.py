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
import sys
from pathlib import Path

from embodichain.toolkits.scaffold.generator import generate_task, print_summary
from embodichain.toolkits.scaffold.naming import default_gym_id
from embodichain.toolkits.scaffold.spec import INREPO_CATEGORIES, TaskSpec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embodichain-new-task",
        description="Scaffold EmbodiChain task environments (in-repo or external extension).",
    )
    parser.add_argument(
        "--target",
        choices=("inrepo", "extension"),
        default="inrepo",
        help="Generate inside EmbodiChain repo or a new extension project.",
    )
    parser.add_argument(
        "--workflow",
        choices=("demo", "rl", "config-only"),
        required=False,
        help="Task workflow: expert demo, RL, or config-only env class.",
    )
    parser.add_argument(
        "--name",
        "--task-name",
        dest="task_snake",
        help="Task name in snake_case (e.g. pick_place).",
    )
    parser.add_argument("--gym-id", help="Gym registration id (e.g. PickPlace-v1).")
    parser.add_argument(
        "--category",
        choices=INREPO_CATEGORIES,
        default="tableware",
        help="In-repo task category (ignored for RL workflow).",
    )
    parser.add_argument(
        "--robot-preset",
        choices=("cobot_magic", "ur5_minimal"),
        default="cobot_magic",
        help="Robot/sensor/light preset for gym JSON.",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=300, help="Max steps per episode."
    )
    parser.add_argument(
        "--max-episodes", type=int, default=5, help="Episodes in gym JSON metadata."
    )
    parser.add_argument(
        "--reward-style",
        choices=("json", "python"),
        default="json",
        help="RL rewards in gym JSON or Python get_reward().",
    )
    parser.add_argument(
        "--project-name",
        help="Extension project name (pyproject name).",
    )
    parser.add_argument(
        "--package-name",
        help="Extension Python package name (snake_case).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Extension output directory (default: ./<project-name>).",
    )
    parser.add_argument(
        "--no-test", action="store_true", help="Skip generating test stub."
    )
    parser.add_argument(
        "--no-black", action="store_true", help="Skip running black on generated files."
    )
    parser.add_argument(
        "--init-git",
        action="store_true",
        help="Run git init in extension output (extension target only).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print paths only; do not write files.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Prompt for missing options.",
    )
    return parser


def _prompt_choice(label: str, options: list[str], default: str) -> str:
    print(f"{label} [{'/'.join(options)}] (default: {default}): ", end="")
    value = input().strip()
    return value if value in options else default


def _interactive_fill(args: argparse.Namespace) -> None:
    if args.target is None:
        args.target = _prompt_choice("Target", ["inrepo", "extension"], "inrepo")
    if args.workflow is None:
        args.workflow = _prompt_choice(
            "Workflow", ["demo", "rl", "config-only"], "demo"
        )
    if args.task_snake is None:
        args.task_snake = input("Task name (snake_case): ").strip()
    if args.gym_id is None and args.task_snake:
        args.gym_id = default_gym_id(args.task_snake)
        print(f"Gym id (default: {args.gym_id}): ", end="")
        custom = input().strip()
        if custom:
            args.gym_id = custom
    if args.target == "inrepo" and args.workflow != "rl":
        if args.category == "tableware":
            args.category = _prompt_choice(
                "Category", list(INREPO_CATEGORIES), "tableware"
            )
    if args.target == "extension":
        if args.package_name is None:
            args.package_name = args.task_snake
        if args.project_name is None:
            args.project_name = args.package_name.replace("_", "-")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.interactive or args.workflow is None or args.task_snake is None:
        _interactive_fill(args)

    if args.workflow is None:
        parser.error("--workflow is required (or use --interactive).")
    if args.task_snake is None:
        parser.error("--name is required (or use --interactive).")

    try:
        spec = TaskSpec(
            task_snake=args.task_snake,
            workflow=args.workflow,
            target=args.target,
            gym_id=args.gym_id,
            category=args.category,
            robot_preset=args.robot_preset,
            max_episode_steps=args.max_episode_steps,
            max_episodes=args.max_episodes,
            reward_style=args.reward_style,
            project_name=args.project_name,
            package_name=args.package_name,
            output_dir=args.output_dir,
            include_test=not args.no_test,
            dry_run=args.dry_run,
            force=args.force,
            run_black=not args.no_black,
            init_git=args.init_git,
        )
        paths = generate_task(spec)
        print_summary(spec, paths)
    except (ValueError, FileExistsError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
