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

import gymnasium
import numpy as np
import torch

from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.lab.scripts.run_env import main
from embodichain.utils.logger import log_error
from embodichain.utils.utility import load_config

__all__ = ["cli"]


def cli() -> None:
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--task_name",
        type=str,
        help="Name of the task.",
        required=True,
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        help="Path to the agent configuration file.",
        required=True,
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Whether to regenerate code if already existed.",
        default=False,
    )
    parser.add_argument(
        "--recovery",
        action="store_true",
        help="Whether to generate recovery actions.",
        default=False,
    )
    parser.add_argument(
        "--interactive_error_injection",
        action="store_true",
        help="Whether to enable terminal-triggered interactive error injection during drive execution.",
        default=False,
    )
    parser.add_argument(
        "--use_public_atomic_actions",
        action=argparse.BooleanOptionalAction,
        help="Whether to use public AtomicActionEngine-backed atom actions.",
        default=True,
    )
    parser.add_argument(
        "--require_public_atomic_actions",
        action=argparse.BooleanOptionalAction,
        help="Whether to raise instead of falling back to legacy when public atomic actions fail.",
        default=False,
    )
    parser.add_argument(
        "--use_public_grasp_semantics",
        action=argparse.BooleanOptionalAction,
        help="Whether to use mesh semantics and AntipodalAffordance for grasp.",
        default=True,
    )
    parser.add_argument(
        "--use_public_grasp_action",
        action=argparse.BooleanOptionalAction,
        help="Whether to use public PickUpAction with legacy grasp_pose_obj targets.",
        default=False,
    )
    parser.add_argument(
        "--use_public_place_action",
        action=argparse.BooleanOptionalAction,
        help="Whether to use public PlaceAction for place_on_table.",
        default=True,
    )
    parser.add_argument(
        "--allow_public_grasp_annotation",
        action=argparse.BooleanOptionalAction,
        help="Whether to allow public grasp annotation when cache is missing.",
        default=True,
    )
    parser.add_argument(
        "--force_public_grasp_reannotate",
        action=argparse.BooleanOptionalAction,
        help="Whether to force re-annotating public grasp regions even if cache exists.",
        default=False,
    )

    args = parser.parse_args()

    if args.num_envs != 1:
        log_error(f"Currently only support num_envs=1, but got {args.num_envs}.")
        raise SystemExit(1)
    if args.require_public_atomic_actions and not args.use_public_atomic_actions:
        log_error(
            "--require_public_atomic_actions requires --use_public_atomic_actions."
        )

    env_cfg, gym_config, _ = build_env_cfg_from_args(args)
    agent_config = load_config(args.agent_config)

    env = gymnasium.make(
        id=gym_config["id"],
        cfg=env_cfg,
        agent_config=agent_config,
        agent_config_path=args.agent_config,
        task_name=args.task_name,
    )

    main(args, env, gym_config)

    if args.headless:
        env.reset(options={"final": True})


if __name__ == "__main__":
    cli()
