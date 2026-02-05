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

import gymnasium
import numpy as np
import argparse
import os
import torch
import tqdm

from threading import Thread

from embodichain.utils.utility import load_json
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import (
    config_to_cfg,
    DEFAULT_MANAGER_MODULES,
)
from embodichain.utils.logger import log_warning, log_info, log_error


def generate_and_execute_action_list(env, idx, debug_mode, **kwargs):

    action_list = env.get_wrapper_attr("create_demo_action_list")(
        action_sentence=idx, **kwargs
    )

    if action_list is None or len(action_list) == 0:
        log_warning("Action is invalid. Skip to next generation.")
        return False

    for action in tqdm.tqdm(
        action_list, desc=f"Executing action list #{idx}", unit="step"
    ):
        # Step the environment with the current action
        # The environment will automatically detect truncation based on action_length
        obs, reward, terminated, truncated, info = env.step(action)

    # TODO: We may assume in export demonstration rollout, there is no truncation from the env.
    # but truncation is useful to improve the generation efficiency.

    return True


def generate_function(
    env,
    num_traj,
    time_id: int = 0,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    **kwargs,
):
    """Generate and execute a sequence of actions in the environment.

    This function resets the environment, generates and executes action trajectories,
    collects data, and optionally saves videos of the episodes. It supports both online
    and offline data generation modes.

    Args:
        env: The environment instance.
        num_traj (int): Number of trajectories to generate per episode.
        time_id (int, optional): Identifier for the current time step or episode.
        save_path (str, optional): Path to save generated videos.
        save_video (bool, optional): Whether to save episode videos.
        debug_mode (bool, optional): Enable debug mode for visualization and logging.
        **kwargs: Additional keyword arguments for data generation.

    Returns:
        bool: True if data generation is successful, False otherwise.
    """

    valid = True
    _, _ = env.reset()
    while True:
        ret = []
        for trajectory_idx in range(num_traj):
            valid = generate_and_execute_action_list(
                env, trajectory_idx, debug_mode, **kwargs
            )

            if not valid:
                # Failed execution: reset without saving invalid data
                _, _ = env.reset(options={"save_data": False})
                break

            # Successful execution: reset and save data
            _, _ = env.reset()

        if valid:
            break
        else:
            log_warning("Reset valid flag to True.")
            valid = True

    return True


def main(args, env, gym_config):

    log_info("Start offline data generation.", color="green")
    # TODO: Support multiple trajectories per episode generation.
    num_traj = 1
    for i in range(gym_config.get("max_episodes", 1)):
        generate_function(
            env,
            num_traj,
            i,
            save_path=getattr(args, "save_path", ""),
            save_video=getattr(args, "save_video", False),
            debug_mode=args.debug_mode,
            regenerate=getattr(args, "regenerate", False),
        )


if __name__ == "__main__":
    np.set_printoptions(5, suppress=True)
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
    parser.add_argument(
        "--save_path", help="path", default="./outputs/thirdviewvideo", type=str
    )
    parser.add_argument(
        "--save_video",
        help="Whether to save the video of the episode.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--debug_mode",
        help="Enable debug mode.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--filter_visual_rand",
        help="Whether to filter out visual randomization.",
        default=False,
        action="store_true",
    )
    parser.add_argument("--gym_config", type=str, help="gym_config", default="")
    parser.add_argument("--action_config", type=str, help="action_config", default=None)

    args = parser.parse_args()

    # if args.num_envs != 1:
    #     log_error(f"Currently only support num_envs=1, but got {args.num_envs}.")

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
    main(args, env, gym_config)
