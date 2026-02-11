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
import os
import torch
import tqdm

from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
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
    if getattr(args, "preview", False):
        log_info(
            "Preview mode enabled. Launching environment preview...", color="green"
        )
        preview(env)

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
            debug_mode=getattr(args, "debug_mode", False),
            regenerate=getattr(args, "regenerate", False),
        )


def preview(env: gymnasium.Env) -> None:
    """
    Run the following code to create a demonstration and perform env steps.

    ```
    # Demo version of environment rollout
    for i in range(10):
        qpos = env.robot.get_qpos()

        obs, reward, terminated, truncated, info = env.step(qpos)

    # reset the environment
    env.reset()
    ```

    Run the following code to preview the sensor observations.

    ```
    env.preview_sensor_data("camera")
    ```
    """
    _, _ = env.reset()

    end = False
    while end is False:
        print("Press `p` to enter embed mode to interact with the environment.")
        print("Press `q` to quit the simulation.")
        txt = input()
        if txt == "p":
            try:
                from IPython import embed
            except ImportError:
                log_error(
                    "IPython is not installed. Preview mode requires IPython to be "
                    "available. Please install it with `pip install ipython` and try again."
                )
                continue

            embed()
        elif txt == "q":
            end = True

    exit(0)


if __name__ == "__main__":
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()

    add_env_launcher_args_to_parser(parser)

    args = parser.parse_args()

    env_cfg, gym_config, action_config = build_env_cfg_from_args(args)

    env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, **action_config)

    main(args, env, gym_config)
