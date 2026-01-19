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

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb
import json
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from embodichain.agents.rl.models import build_policy, get_registered_policy_names
from embodichain.agents.rl.models import build_mlp_from_cfg
from embodichain.agents.rl.algo import build_algo, get_registered_algo_names
from embodichain.agents.rl.utils.trainer import Trainer
from embodichain.utils import logger
from embodichain.lab.gym.envs.tasks.rl import build_env
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.utils.utility import load_json
from embodichain.utils.module_utils import find_function_from_modules
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.gym.envs.managers.cfg import EventCfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg_json = json.load(f)

    trainer_cfg = cfg_json["trainer"]
    policy_block = cfg_json["policy"]
    algo_block = cfg_json["algorithm"]

    # Runtime
    exp_name = trainer_cfg.get("exp_name", "generic_exp")
    seed = int(trainer_cfg.get("seed", 1))
    device_str = trainer_cfg.get("device", "cpu")
    iterations = int(trainer_cfg.get("iterations", 250))
    rollout_steps = int(trainer_cfg.get("rollout_steps", 2048))
    eval_freq = int(trainer_cfg.get("eval_freq", 10000))
    save_freq = int(trainer_cfg.get("save_freq", 50000))
    headless = bool(trainer_cfg.get("headless", True))
    enable_rt = bool(trainer_cfg.get("enable_rt", False))
    gpu_id = int(trainer_cfg.get("gpu_id", 0))
    num_envs = trainer_cfg.get("num_envs", None)
    wandb_project_name = trainer_cfg.get("wandb_project_name", "embodychain-generic")

    # Device
    if not isinstance(device_str, str):
        raise ValueError(
            f"runtime.device must be a string such as 'cpu' or 'cuda:0'. Got: {device_str!r}"
        )
    try:
        device = torch.device(device_str)
    except RuntimeError as exc:
        raise ValueError(
            f"Failed to parse runtime.device='{device_str}': {exc}"
        ) from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA device requested but torch.cuda.is_available() is False."
            )
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        device_count = torch.cuda.device_count()
        if index < 0 or index >= device_count:
            raise ValueError(
                f"CUDA device index {index} is out of range (available devices: {device_count})."
            )
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    elif device.type != "cpu":
        raise ValueError(f"Unsupported device type: {device}")
    logger.log_info(f"Device: {device}")

    # Seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Outputs
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_base = os.path.join("outputs", f"{exp_name}_{run_stamp}")
    log_dir = os.path.join(run_base, "logs")
    checkpoint_dir = os.path.join(run_base, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(f"{log_dir}/{exp_name}")

    # Initialize Weights & Biases (optional)
    use_wandb = trainer_cfg.get("use_wandb", False)

    # Initialize Weights & Biases (optional)
    if use_wandb:
        wandb.init(project=wandb_project_name, name=exp_name, config=cfg_json)

    gym_config_path = Path(trainer_cfg["gym_config"])
    logger.log_info(f"Current working directory: {Path.cwd()}")

    gym_config_data = load_json(str(gym_config_path))
    gym_env_cfg = config_to_cfg(gym_config_data)

    # Override num_envs from train config if provided
    if num_envs is not None:
        gym_env_cfg.num_envs = num_envs

    # Ensure sim configuration mirrors runtime overrides
    if gym_env_cfg.sim_cfg is None:
        gym_env_cfg.sim_cfg = SimulationManagerCfg()
    if device.type == "cuda":
        gpu_index = device.index
        if gpu_index is None:
            gpu_index = torch.cuda.current_device()
        gym_env_cfg.sim_cfg.sim_device = torch.device(f"cuda:{gpu_index}")
        if hasattr(gym_env_cfg.sim_cfg, "gpu_id"):
            gym_env_cfg.sim_cfg.gpu_id = gpu_index
    else:
        gym_env_cfg.sim_cfg.sim_device = torch.device("cpu")
    gym_env_cfg.sim_cfg.headless = headless
    gym_env_cfg.sim_cfg.enable_rt = enable_rt
    gym_env_cfg.sim_cfg.gpu_id = gpu_id

    logger.log_info(
        f"Loaded gym_config from {gym_config_path} (env_id={gym_config_data['id']}, num_envs={gym_env_cfg.num_envs}, headless={gym_env_cfg.sim_cfg.headless}, enable_rt={gym_env_cfg.sim_cfg.enable_rt}, sim_device={gym_env_cfg.sim_cfg.sim_device})"
    )

    env = build_env(gym_config_data["id"], base_env_cfg=gym_env_cfg)

    eval_gym_env_cfg = deepcopy(gym_env_cfg)
    eval_gym_env_cfg.num_envs = 4
    eval_gym_env_cfg.sim_cfg.headless = True

    eval_env = build_env(gym_config_data["id"], base_env_cfg=eval_gym_env_cfg)

    # Build Policy via registry
    policy_name = policy_block["name"]
    # Build Policy via registry (actor/critic must be explicitly defined in JSON when using actor_critic)
    if policy_name.lower() == "actor_critic":
        # Get observation dimension from flattened observation space
        # flattened_observation_space returns Box space for RL training
        obs_dim = env.flattened_observation_space.shape[-1]
        action_dim = env.action_space.shape[-1]

        actor_cfg = policy_block.get("actor")
        critic_cfg = policy_block.get("critic")
        if actor_cfg is None or critic_cfg is None:
            raise ValueError(
                "ActorCritic requires 'actor' and 'critic' definitions in JSON (policy.actor / policy.critic)."
            )

        actor = build_mlp_from_cfg(actor_cfg, obs_dim, action_dim)
        critic = build_mlp_from_cfg(critic_cfg, obs_dim, 1)

        policy = build_policy(
            policy_block,
            env.flattened_observation_space,
            env.action_space,
            device,
            actor=actor,
            critic=critic,
        )
    else:
        policy = build_policy(
            policy_block, env.flattened_observation_space, env.action_space, device
        )

    # Build Algorithm via factory
    algo_name = algo_block["name"].lower()
    algo_cfg = algo_block["cfg"]
    algo = build_algo(algo_name, algo_cfg, policy, device)

    # Build Trainer
    event_modules = [
        "embodichain.lab.gym.envs.managers.randomization",
        "embodichain.lab.gym.envs.managers.record",
        "embodichain.lab.gym.envs.managers.events",
    ]
    events_dict = trainer_cfg.get("events", {})
    train_event_cfg = {}
    eval_event_cfg = {}
    # Parse train events
    for event_name, event_info in events_dict.get("train", {}).items():
        event_func_str = event_info.get("func")
        mode = event_info.get("mode", "interval")
        params = event_info.get("params", {})
        interval_step = event_info.get("interval_step", 1)
        event_func = find_function_from_modules(
            event_func_str, event_modules, raise_if_not_found=True
        )
        train_event_cfg[event_name] = EventCfg(
            func=event_func,
            mode=mode,
            params=params,
            interval_step=interval_step,
        )
    # Parse eval events
    for event_name, event_info in events_dict.get("eval", {}).items():
        event_func_str = event_info.get("func")
        mode = event_info.get("mode", "interval")
        params = event_info.get("params", {})
        interval_step = event_info.get("interval_step", 1)
        event_func = find_function_from_modules(
            event_func_str, event_modules, raise_if_not_found=True
        )
        eval_event_cfg[event_name] = EventCfg(
            func=event_func,
            mode=mode,
            params=params,
            interval_step=interval_step,
        )
    trainer = Trainer(
        policy=policy,
        env=env,
        algorithm=algo,
        num_steps=rollout_steps,
        batch_size=algo_cfg["batch_size"],
        writer=writer,
        eval_freq=eval_freq,
        save_freq=save_freq,
        checkpoint_dir=checkpoint_dir,
        exp_name=exp_name,
        use_wandb=use_wandb,
        eval_env=eval_env,
        event_cfg=train_event_cfg,
        eval_event_cfg=eval_event_cfg,
    )

    logger.log_info("Generic training initialized")
    logger.log_info(f"Task: {type(env).__name__}")
    logger.log_info(
        f"Policy: {policy_name} (available: {get_registered_policy_names()})"
    )
    logger.log_info(
        f"Algorithm: {algo_name} (available: {get_registered_algo_names()})"
    )

    total_steps = int(iterations * rollout_steps * env.num_envs)
    logger.log_info(f"Total steps: {total_steps} (iterations≈{iterations})")

    try:
        trainer.train(total_steps)
    except KeyboardInterrupt:
        logger.log_info("Training interrupted by user")
    finally:
        trainer.save_checkpoint()
        writer.close()
        if use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
        logger.log_info("Training finished")


if __name__ == "__main__":
    main()
