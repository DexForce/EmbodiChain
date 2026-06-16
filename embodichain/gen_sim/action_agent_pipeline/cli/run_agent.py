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
import tqdm

from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.agent_env import (  # noqa: F401
    AtomicActionsAgentEnv,
)
from embodichain.lab.gym.utils.gym_utils import (
    add_env_launcher_args_to_parser,
    build_env_cfg_from_args,
)
from embodichain.utils.logger import log_error, log_info, log_warning
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

    args = parser.parse_args()

    if args.num_envs != 1:
        log_error(f"Currently only support num_envs=1, but got {args.num_envs}.")
        raise SystemExit(1)

    env_cfg, gym_config, _ = build_env_cfg_from_args(args)
    agent_config = load_config(args.agent_config)

    env = gymnasium.make(
        id=gym_config["id"],
        cfg=env_cfg,
        agent_config=agent_config,
        agent_config_path=args.agent_config,
        task_name=args.task_name,
    )

    _run_action_agent(args, env, gym_config)

    if args.headless:
        env.reset(options={"final": True})


def _run_action_agent(args: argparse.Namespace, env: gymnasium.Env, gym_config: dict):
    """Run action-agent graphs without relying on the shared run_env runner."""
    if getattr(args, "preview", False):
        log_warning("Preview mode is handled by the shared runner and is skipped here.")

    log_info("Start action-agent data generation.", color="green")
    for trajectory_idx in range(gym_config.get("max_episodes", 1)):
        _generate_action_agent_trajectory(
            args,
            env,
            trajectory_idx,
        )
    _, _ = env.reset()


def _generate_action_agent_trajectory(
    args: argparse.Namespace,
    env: gymnasium.Env,
    trajectory_idx: int,
) -> bool:
    _, _ = env.reset()
    action_list = env.get_wrapper_attr("create_demo_action_list")(
        action_sentence=trajectory_idx,
        save_path=getattr(args, "save_path", ""),
        save_video=getattr(args, "save_video", False),
        debug_mode=getattr(args, "debug_mode", False),
        regenerate=getattr(args, "regenerate", False),
        recovery=getattr(args, "recovery", False),
    )
    if action_list is None or len(action_list) == 0:
        log_warning("Action is invalid. Skip to next generation.")
        return False

    if getattr(action_list, "already_executed", False):
        log_info("Action list was already executed by the action-agent runtime.")
        _log_task_success(env)
        return True

    for action in tqdm.tqdm(
        action_list,
        desc=f"Executing action list #{trajectory_idx}",
        unit="step",
    ):
        env.step(action)
    _log_task_success(env)
    return True


def _log_task_success(env: gymnasium.Env) -> bool | None:
    try:
        success_fn = (
            env.get_wrapper_attr("is_task_success")
            if hasattr(env, "get_wrapper_attr")
            else env.is_task_success
        )
        success = success_fn()
    except Exception as exc:
        log_warning(f"Failed to evaluate task success after execution: {exc}")
        return None

    if isinstance(success, torch.Tensor):
        success_value = bool(success.detach().cpu().flatten().all().item())
    else:
        success_value = bool(np.asarray(success).flatten().all())
    log_info(f"Task success after execution: {success_value}", color="green")
    return success_value


if __name__ == "__main__":
    cli()
