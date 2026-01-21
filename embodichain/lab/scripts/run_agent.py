# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import gymnasium
import numpy as np
import argparse
import os
import torch

from threading import Thread
from tqdm import tqdm
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


def generate_function(
    env,
    time_id: int = 0,
    online_training: bool = False,
    save_path: str = "",
    save_video: bool = False,
    debug_mode: bool = False,
    regenerate: bool = True,
    **kwargs,
):
    """
    Generate and execute a sequence of actions in the environment.

    This function resets the environment, generates and executes action trajectories,
    collects data, and optionally saves videos of the episodes. It supports both online
    and offline data generation modes.

    Args:
        env: The environment instance.
        time_id (int, optional): Identifier for the current time step or episode.
        online_training (bool, optional): Whether to use online data generation.
        save_path (str, optional): Path to save generated videos.
        save_video (bool, optional): Whether to save episode videos.
        debug_mode (bool, optional): Enable debug mode for visualization and logging.
        regenerate (bool, optional): Whether enable regenerating if existed.
        **kwargs: Additional keyword arguments for data generation.

    Returns:
        list or bool: Returns a list of data dicts if online_training is True,
                      otherwise returns True if generation is successful.
    """

    def wait_for_threads(threads):
        for t in threads:
            t.join()

    vis_threads = []

    while True:  # repeat until success
        env.reset()

        ret = []
        trajectory_idx = 0

        # Access the wrapped environment's method
        env.get_wrapper_attr("create_demo_action_list")(regenerate=regenerate)

        # ---------------------------------------------------------
        # SUCCESS CASE
        # ---------------------------------------------------------
        if not debug_mode and env.get_wrapper_attr("is_task_success")().item():

            dataset_id = f"time_{time_id}_trajectory_{trajectory_idx}"

            # online training: dataset may not be saved every iteration
            if online_training:
                dataset_id += "_online_generated"
                num_samples = kwargs.get("num_samples", 0)
                is_save_dataset = time_id < num_samples

                data_dict = env.get_wrapper_attr("to_dataset")(
                    id=dataset_id if is_save_dataset else None
                )
                ret.append(data_dict)
            else:
                data_dict = env.get_wrapper_attr("to_dataset")(id=dataset_id)

            # episode id
            try:
                episode = env.get_wrapper_attr("get_current_episode")()
            except AttributeError:
                episode = time_id

            # video saving
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

            break  # success

        # ---------------------------------------------------------
        # FAILURE CASE
        # ---------------------------------------------------------
        else:
            log_warning("Task fail, Skip to next generation and retry.")
            continue  # retry until success

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

        generator_func = lambda time_id, **kwargs: generate_function(
            env,
            time_id,
            online_training=is_online_training,
            save_path=args.save_path,
            save_video=args.save_video,
            regenerate=args.regenerate,
            **kwargs,
        )
        online_callback.generator(generator_func, **online_config)
    else:
        log_info("Start offline data generation.", color="green")
        for i in range(gym_config["max_episodes"]):
            generate_function(
                env,
                i,
                online_training=is_online_training,
                save_path=args.save_path,
                save_video=args.save_video,
                debug_mode=args.debug_mode,
                regenerate=args.regenerate,
            )

    if args.headless:
        env.reset(options={"final": True})


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
        "--render_backend",
        help="The rendering backend to use for the simulation.",
        default="egl",
        type=str,
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
    parser.add_argument(
        "--task_name", type=str, help="Name of the task.", required=True
    )

    # Agent related configs
    parser.add_argument(
        "--agent_config", type=str, help="agent_config", default=None, required=True
    )
    parser.add_argument(
        "--regenerate",
        type=bool,
        help="Whether regenerate code if already existed.",
        default=False,
    )

    args = parser.parse_args()

    if args.num_envs != 1:
        log_error(f"Currently only support num_envs=1, but got {args.num_envs}.")

    gym_config = load_json(args.gym_config)
    cfg: EmbodiedEnvCfg = config_to_cfg(gym_config)
    cfg.filter_visual_rand = args.filter_visual_rand

    agent_config = load_json(args.agent_config)

    cfg.num_envs = args.num_envs
    cfg.sim_cfg = SimulationManagerCfg(
        headless=args.headless,
        sim_device=args.device,
        enable_rt=args.enable_rt,
        gpu_id=args.gpu_id,
    )

    env = gymnasium.make(
        id=gym_config["id"],
        cfg=cfg,
        agent_config=agent_config,
        task_name=args.task_name,
    )
    main(args, env, gym_config)
