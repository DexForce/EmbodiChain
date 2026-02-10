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

import gymnasium
import numpy as np
import argparse
import torch

from embodichain.utils.utility import load_json
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.utils.logger import log_error
from .run_env import main


if __name__ == "__main__":
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

    args = parser.parse_args()

    # Validate arguments
    if args.num_envs != 1:
        log_error(f"Currently only support num_envs=1, but got {args.num_envs}.")
        exit(1)

    # Load configurations
    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)
    agent_config = load_json(args.agent_config)

    # Create environment
    env = gymnasium.make(
        id=gym_config["id"],
        cfg=env_cfg,
        agent_config=agent_config,
        agent_config_path=args.agent_config,
        task_name=args.task_name,
    )

    # Run main function
    main(args, env, gym_config)

    if args.headless:
        env.reset(options={"final": True})
