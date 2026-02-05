# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
import gymnasium
import argparse
import numpy as np

from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import (
    config_to_cfg,
    DEFAULT_MANAGER_MODULES,
)
from embodichain.utils.utility import load_json
from embodichain.utils import logger

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_envs",
        help="The number of environments to run in parallel.",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run the environment on, e.g., 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "--headless",
        help="Whether to perform the simulation in headless mode.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--enable_rt",
        help="Whether to use RTX rendering backend for the simulation.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--gpu_id",
        help="The GPU ID to use for the simulation.",
        default=0,
        type=int,
    )
    parser.add_argument("--gym_config", type=str, help="gym_config", default="")
    parser.add_argument(
        "--action_config",
        type=str,
        help="Path to the action configuration file.",
        default=None,
    )
    parser.add_argument(
        "--filter_visual_rand",
        help="Whether to filter out visual randomization.",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    """
    TODO: Currently, this file is only used to preview the template.json config file.
    We may add more features to support more general case parsing from config files.
    """

    ##############################################################################################
    # load gym config
    gym_config = load_json(args.gym_config)
    cfg: EmbodiedEnvCfg = config_to_cfg(
        gym_config, manager_modules=DEFAULT_MANAGER_MODULES
    )
    cfg.filter_visual_rand = args.filter_visual_rand

    action_config = {}
    if args.action_config is not None:
        action_config = load_json(args.action_config)
        action_config["action_config"] = action_config

    cfg.num_envs = args.num_envs
    cfg.sim_cfg = SimulationManagerCfg(
        headless=args.headless,
        sim_device=args.device,
        enable_rt=args.enable_rt,
        gpu_id=args.gpu_id,
    )

    env = gymnasium.make(id=gym_config["id"], cfg=cfg, **action_config)

    obs, info = env.reset()

    """
    Run the following code to create a demonstration and perform env steps.
    
    ```
    # Demo version of environment rollout
    for i in range(10):
        qpos = env.robot.get_qpos()

        obs, reward, done, truncated, info = env.step(qpos)

    # reset the environment
    env.reset()
    ```

    Run the following code to preview the sensor observations.

    ```
    env.preview_sensor_data("camera")
    ```
    """

    end = False
    while end is False:
        print("Press `p` to into embed mode to interact with the environment.")
        print("Press `q` to quit the simulation.")
        txt = input()
        if txt == "p":
            from IPython import embed

            embed()
        elif txt == "q":
            end = True
