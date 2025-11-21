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

from threading import Thread

from embodichain.utils.utility import load_json
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.gym.envs import EmbodiedEnvCfg
from embodichain.lab.gym.utils.gym_utils import (
    config_to_cfg,
)
from embodichain.lab.scripts.generate_video import visualize_data_dict
from embodichain.data.data_engine.online.online_generator import (
    OnlineGenerator,
)
from embodichain.utils.logger import log_warning, log_info, log_error
from embodichain.lab.sim.cfg import MarkerCfg


def generate_and_execute_action_list(env, idx, debug_mode):

    action_list = env.create_demo_action_list(action_sentence=idx)

    # TODO: To be modified.
    # if debug_mode:
    #     env.visual_action(action_list)

    if action_list is None or len(action_list) == 0:
        log_warning("Action is invalid. Skip to next generation.")
        return False

    for action in action_list:
        # Step the environment with the current action
        obs, reward, terminated, truncated, info = env.step(action)

        # TODO: To be modified.
        # if debug_mode:
        #     xpos_dict = env.agent.get_debug_xpos_dict()

        #     for key, val in xpos_dict.items():
        # env.scene.draw_marker(cfg=MarkerCfg(
        #     marker_type="axis",
        #     axis_xpos=val,
        #     axis_size=0.002,
        #     axis_len=0.005
        # ))

        #     for key, val in xpos_dict.items():
        #         env.scene.remove_fixed_actor(key)

    # TODO: we may assume in export demonstration rollout, there is no truncation from the env.
    # but truncation is useful to improve the generation efficiency.

    return True


def generate_function(
    env,
    obj_num,
    time_id: int = 0,
    online_training: bool = False,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    **kwargs,
):
    """
    Generate and execute a sequence of actions in the environment.

    This function resets the environment, generates and executes action trajectories,
    collects data, and optionally saves videos of the episodes. It supports both online
    and offline data generation modes.

    Args:
        env: The environment instance.
        obj_num (int): Number of trajectories to generate per episode.
        time_id (int, optional): Identifier for the current time step or episode.
        online_training (bool, optional): Whether to use online data generation.
        save_path (str, optional): Path to save generated videos.
        save_video (bool, optional): Whether to save episode videos.
        debug_mode (bool, optional): Enable debug mode for visualization and logging.
        **kwargs: Additional keyword arguments for data generation.

    Returns:
        list or bool: Returns a list of data dicts if online_training is True,
                      otherwise returns True if generation is successful.
    """

    def wait_for_threads(threads):
        for t in threads:
            t.join()

    vis_threads = []
    valid = True
    while True:
        _, _ = env.reset()

        ret = []
        for trajectory_idx in range(obj_num):
            valid = generate_and_execute_action_list(env, trajectory_idx, debug_mode)

            if not valid:
                log_warning("Invalid action, skipping trajectory.")
                break

            if not debug_mode and env.is_task_success().item():
                # Create a unique identifier for the dataset entry
                dataset_id = f"time_{time_id}_trajectory_{trajectory_idx}"
                if online_training:
                    dataset_id += "_online_generated"
                    num_samples = kwargs.get("num_samples", 0)
                    is_save_dataset = time_id < num_samples

                    data_dict = env.to_dataset(
                        id=dataset_id if is_save_dataset else None,
                    )

                    ret.append(data_dict)
                else:
                    data_dict = env.to_dataset(
                        id=dataset_id,
                    )

                episode = getattr(env, "get_current_episode", lambda: time_id)()

                if save_video:
                    video_path = os.path.join(save_path, f"episode_{episode}")
                    if online_training:
                        vis_thread = Thread(
                            target=visualize_data_dict,
                            args=(data_dict["data"], video_path),
                            daemon=True,
                        )
                        vis_thread.start()
                        vis_threads.append(vis_thread)
                    else:
                        visualize_data_dict(data_dict["data"], video_path)

            else:
                log_warning(f"Task fail, Skip to next generation.")
                valid = False
                break

        if valid:
            break
        else:
            log_warning("Reset valid flag to True.")
            valid = True

    wait_for_threads(vis_threads)
    return ret if online_training else True


def main(args, env, gym_config):
    is_online_training = os.path.exists(args.online_config)
    if is_online_training:

        log_info("Start online data generation.", color="green")
        assert os.path.exists(args.online_config), "{} does not exist.".format(
            args.online_config
        )

        online_config = load_json(args.online_config)
        online_callback = OnlineGenerator(**online_config)

        obj_num = 1
        generator_func = lambda time_id, **kwargs: generate_function(
            env,
            obj_num,
            time_id,
            online_training=is_online_training,
            save_path=args.save_path,
            save_video=args.save_video,
            headless=args.headless,
            **kwargs,
        )
        online_callback.generator(generator_func, **online_config)
    else:
        log_info("Start offline data generation.", color="green")
        obj_num = 1
        for i in range(gym_config["max_episodes"]):
            generate_function(
                env,
                obj_num,
                i,
                online_training=is_online_training,
                save_path=args.save_path,
                save_video=args.save_video,
                debug_mode=args.debug_mode,
            )

    if args.headless:
        env.reset(options={"final": True})


if __name__ == "__main__":
    np.set_printoptions(5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)

    parser = argparse.ArgumentParser()
    # parser.add_argument("--task_type", help="Type of task to perform.")
    # parser.add_argument("--robot_name", help="Name of the robot.")
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
        "--save_video",
        help="Whether to save data as video.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--save_path", help="path", default="./outputs/thirdviewvideo", type=str
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

    parser.add_argument("--online_config", type=str, help="online_config", default="")
    parser.add_argument("--gym_config", type=str, help="gym_config", default="")
    parser.add_argument("--action_config", type=str, help="action_config", default=None)

    args = parser.parse_args()

    if args.num_envs != 1:
        log_error(f"Currently only support num_envs=1, but got {args.num_envs}.")

    gym_config = load_json(args.gym_config)
    cfg: EmbodiedEnvCfg = config_to_cfg(gym_config)
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
