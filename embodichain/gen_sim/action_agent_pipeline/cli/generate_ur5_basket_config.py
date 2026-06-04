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

from embodichain.gen_sim.action_agent_pipeline.ur5_basket_config_generation import (
    generate_ur5_basket_config_from_project,
)

__all__ = ["cli"]


def cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Dual-UR5 basket-placement action-agent config from an "
            "exported tabletop gym project."
        )
    )
    parser.add_argument(
        "--gym_project",
        type=str,
        required=True,
        help=(
            "Path to a project root, formatted tabletop scene folder, or "
            "gym_config.json."
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
        "--target_body_scale",
        type=float,
        default=0.7,
        help=(
            "Uniform body_scale for the two target objects, e.g. 0.5, 0.6, " "or 1.0."
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

    paths = generate_ur5_basket_config_from_project(
        gym_project=args.gym_project,
        output_dir=args.output_dir,
        task_name=args.task_name,
        use_llm_roles=args.use_llm_roles,
        llm_model=args.llm_model,
        target_body_scale=args.target_body_scale,
        overwrite=args.overwrite,
        max_episodes=args.max_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    print(f"Generated gym config: {paths.gym_config}")
    print(f"Generated agent config: {paths.agent_config}")
    print(f"Generated task prompt: {paths.task_prompt}")
    print(f"Generated basic background: {paths.basic_background}")
    print(f"Generated atom actions: {paths.atom_actions}")
    print(
        "Run with:\n"
        "python -m embodichain.gen_sim.action_agent_pipeline.cli.run_agent "
        f"--task_name {args.task_name} "
        f'--gym_config "{paths.gym_config}" '
        f'--agent_config "{paths.agent_config}" '
        "--regenerate"
    )


if __name__ == "__main__":
    cli()
