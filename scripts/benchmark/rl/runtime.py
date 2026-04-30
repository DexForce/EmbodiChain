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

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from embodichain.agents.rl.algo import build_algo
from embodichain.agents.rl.models import build_mlp_from_cfg, build_policy
from embodichain.agents.rl.utils import dict_to_tensordict, flatten_dict_observation
from embodichain.agents.rl.utils.trainer import Trainer
from embodichain.lab.gym.envs.managers.cfg import EventCfg
from embodichain.lab.gym.envs.tasks.rl import build_env
from embodichain.lab.gym.utils.gym_utils import DEFAULT_MANAGER_MODULES, config_to_cfg
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.utils.module_utils import find_function_from_modules
from embodichain.utils.utility import load_json

EVENT_MODULES = [
    "embodichain.lab.gym.envs.managers.randomization",
    "embodichain.lab.gym.envs.managers.record",
    "embodichain.lab.gym.envs.managers.events",
]


def resolve_device(device_str: str) -> torch.device:
    """Resolve a runtime device string into a validated torch device."""
    device = torch.device(device_str)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but no CUDA device is available.")
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is out of range.")
        torch.cuda.set_device(index)
        return torch.device(f"cuda:{index}")
    if device.type != "cpu":
        raise ValueError(f"Unsupported device type: {device.type}")
    return device


def set_random_seed(seed: int, device: torch.device) -> None:
    """Set deterministic random seeds for numpy and torch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)


def _parse_event_cfg(events_dict: dict[str, Any]) -> dict[str, EventCfg]:
    parsed: dict[str, EventCfg] = {}
    for event_name, event_info in events_dict.items():
        event_func = find_function_from_modules(
            event_info["func"], EVENT_MODULES, raise_if_not_found=True
        )
        parsed[event_name] = EventCfg(
            func=event_func,
            mode=event_info.get("mode", "interval"),
            params=event_info.get("params", {}),
            interval_step=event_info.get("interval_step", 1),
        )
    return parsed


def _build_env_cfg(
    gym_config_path: str,
    num_envs: int | None,
    headless: bool,
    device: torch.device,
    gpu_id: int,
):
    gym_config_data = load_json(gym_config_path)
    gym_env_cfg = config_to_cfg(
        gym_config_data, manager_modules=DEFAULT_MANAGER_MODULES
    )
    if num_envs is not None:
        gym_env_cfg.num_envs = int(num_envs)
    if gym_env_cfg.sim_cfg is None:
        gym_env_cfg.sim_cfg = SimulationManagerCfg()
    gym_env_cfg.seed = getattr(gym_env_cfg, "seed", None)
    gym_env_cfg.sim_cfg.headless = headless
    gym_env_cfg.sim_cfg.gpu_id = gpu_id
    gym_env_cfg.sim_cfg.sim_device = device
    return gym_config_data, gym_env_cfg


def _allocate_eval_rollout_buffer(env, policy, device: torch.device) -> TensorDict:
    """Allocate a small RL-style rollout buffer for evaluation-only environments."""
    rollout_len = 2
    return TensorDict(
        {
            "obs": torch.zeros(
                env.num_envs,
                rollout_len + 1,
                policy.obs_dim,
                dtype=torch.float32,
                device=device,
            ),
            "action": torch.zeros(
                env.num_envs,
                rollout_len + 1,
                policy.action_dim,
                dtype=torch.float32,
                device=device,
            ),
            "sample_log_prob": torch.zeros(
                env.num_envs,
                rollout_len + 1,
                dtype=torch.float32,
                device=device,
            ),
            "value": torch.zeros(
                env.num_envs,
                rollout_len + 1,
                dtype=torch.float32,
                device=device,
            ),
            "reward": torch.zeros(
                env.num_envs,
                rollout_len + 1,
                dtype=torch.float32,
                device=device,
            ),
            "done": torch.zeros(
                env.num_envs,
                rollout_len + 1,
                dtype=torch.bool,
                device=device,
            ),
            "terminated": torch.zeros(
                env.num_envs,
                rollout_len + 1,
                dtype=torch.bool,
                device=device,
            ),
            "truncated": torch.zeros(
                env.num_envs,
                rollout_len + 1,
                dtype=torch.bool,
                device=device,
            ),
        },
        batch_size=[env.num_envs, rollout_len + 1],
        device=device,
    )


def _compact_eval_rollout_buffer(env, rollout_buffer: TensorDict) -> None:
    """Keep only the previous transition needed by rollout-dependent eval rewards."""
    if getattr(env, "current_rollout_step", 0) < 2:
        return
    for key in ("action", "reward", "done", "terminated", "truncated"):
        rollout_buffer[key][:, 0].copy_(rollout_buffer[key][:, 1])
        rollout_buffer[key][:, 1:].zero_()
    env.current_rollout_step = 1


def build_policy_from_env(policy_block: dict[str, Any], env, device: torch.device):
    """Build a policy using the current environment spaces."""
    sample_obs, _ = env.reset()
    sample_obs_td = dict_to_tensordict(sample_obs, device)
    obs_dim = flatten_dict_observation(sample_obs_td).shape[-1]
    flat_obs_space = env.flattened_observation_space
    env_action_dim = env.action_space.shape[-1]

    policy_name = policy_block["name"].lower()
    if policy_name == "actor_critic":
        actor = build_mlp_from_cfg(policy_block["actor"], obs_dim, env_action_dim)
        critic = build_mlp_from_cfg(policy_block["critic"], obs_dim, 1)
        return build_policy(
            policy_block,
            flat_obs_space,
            env.action_space,
            device,
            actor=actor,
            critic=critic,
        )
    if policy_name == "actor_only":
        actor = build_mlp_from_cfg(policy_block["actor"], obs_dim, env_action_dim)
        return build_policy(
            policy_block,
            flat_obs_space,
            env.action_space,
            device,
            actor=actor,
        )
    return build_policy(policy_block, flat_obs_space, env.action_space, device)


def train_with_config(
    cfg_json: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train an RL configuration and return a structured summary."""
    trainer_cfg = deepcopy(cfg_json["trainer"])
    policy_block = deepcopy(cfg_json["policy"])
    algo_block = deepcopy(cfg_json["algorithm"])

    device = resolve_device(trainer_cfg.get("device", "cpu"))
    seed = int(trainer_cfg.get("seed", 1))
    set_random_seed(seed, device)

    output_root = Path(output_dir)
    log_dir = output_root / "logs"
    checkpoint_dir = output_root / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    gym_config_data, gym_env_cfg = _build_env_cfg(
        gym_config_path=trainer_cfg["gym_config"],
        num_envs=trainer_cfg.get("num_envs"),
        headless=bool(trainer_cfg.get("headless", True)),
        device=device,
        gpu_id=int(trainer_cfg.get("gpu_id", 0)),
    )
    env = None
    eval_env = None
    writer = SummaryWriter(str(log_dir))
    try:
        env = build_env(gym_config_data["id"], base_env_cfg=gym_env_cfg)

        enable_eval = bool(trainer_cfg.get("enable_eval", True))
        if enable_eval:
            eval_gym_env_cfg = deepcopy(gym_env_cfg)
            eval_gym_env_cfg.num_envs = int(
                trainer_cfg.get("num_eval_envs", min(4, gym_env_cfg.num_envs))
            )
            eval_gym_env_cfg.sim_cfg.headless = True
            eval_env = build_env(gym_config_data["id"], base_env_cfg=eval_gym_env_cfg)

        policy = build_policy_from_env(policy_block, env, device)
        algo = build_algo(algo_block["name"], algo_block["cfg"], policy, device)

        events_dict = trainer_cfg.get("events", {})
        trainer = Trainer(
            policy=policy,
            env=env,
            algorithm=algo,
            buffer_size=int(trainer_cfg.get("buffer_size", 2048)),
            batch_size=int(algo_block["cfg"]["batch_size"]),
            writer=writer,
            eval_freq=int(trainer_cfg.get("eval_freq", 0)) if enable_eval else 0,
            save_freq=int(trainer_cfg.get("save_freq", 0)) or 10**18,
            checkpoint_dir=str(checkpoint_dir),
            exp_name=str(trainer_cfg.get("exp_name", "benchmark_run")),
            use_wandb=False,
            eval_env=eval_env,
            event_cfg=_parse_event_cfg(events_dict.get("train", {})),
            eval_event_cfg=(
                _parse_event_cfg(events_dict.get("eval", {})) if enable_eval else {}
            ),
            num_eval_episodes=int(trainer_cfg.get("num_eval_episodes", 5)),
        )

        total_steps = (
            int(trainer_cfg.get("iterations", 1))
            * int(trainer_cfg.get("buffer_size", 2048))
            * int(env.num_envs)
        )
        start_time = time.perf_counter()
        summary = trainer.train(total_steps)
        wall_time = time.perf_counter() - start_time
        checkpoint_path = trainer.save_checkpoint()
    finally:
        writer.close()
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()

    peak_gpu_memory_mb = 0.0
    if device.type == "cuda":
        peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device=device) / (
            1024.0 * 1024.0
        )

    summary.update(
        {
            "checkpoint_path": checkpoint_path,
            "output_dir": str(output_root),
            "wall_time_sec": float(wall_time),
            "training_fps": float(total_steps / max(wall_time, 1e-6)),
            "peak_gpu_memory_mb": float(peak_gpu_memory_mb),
        }
    )
    return summary


def evaluate_checkpoint(
    cfg_json: dict[str, Any],
    checkpoint_path: str | Path,
    num_episodes: int,
    num_envs: int | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint deterministically and collect task metrics."""
    trainer_cfg = deepcopy(cfg_json["trainer"])
    policy_block = deepcopy(cfg_json["policy"])

    device = resolve_device(trainer_cfg.get("device", "cpu"))
    gym_config_data, gym_env_cfg = _build_env_cfg(
        gym_config_path=trainer_cfg["gym_config"],
        num_envs=num_envs if num_envs is not None else trainer_cfg.get("num_eval_envs"),
        headless=True,
        device=device,
        gpu_id=int(trainer_cfg.get("gpu_id", 0)),
    )
    env = None
    try:
        env = build_env(gym_config_data["id"], base_env_cfg=gym_env_cfg)
        policy = build_policy_from_env(policy_block, env, device)
        eval_rollout_buffer = None
        if hasattr(env, "set_rollout_buffer"):
            eval_rollout_buffer = _allocate_eval_rollout_buffer(env, policy, device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["policy"])
        policy.eval()

        target_episodes = int(num_episodes)
        completed = 0
        cumulative_reward = torch.zeros(
            env.num_envs, dtype=torch.float32, device=device
        )
        step_count = torch.zeros(env.num_envs, dtype=torch.int32, device=device)

        returns: list[float] = []
        lengths: list[int] = []
        successes: list[float] = []
        metric_values: dict[str, list[float]] = {}
        env_step_count = 0
        env_step_time = 0.0

        if eval_rollout_buffer is not None:
            env.set_rollout_buffer(eval_rollout_buffer)
        obs, _ = env.reset()
        while completed < target_episodes:
            flat_obs = flatten_dict_observation(obs)
            action_td = TensorDict(
                {"obs": flat_obs},
                batch_size=[env.num_envs],
                device=device,
            )
            action_td = policy.get_action(action_td, deterministic=True)
            action_manager = getattr(env, "action_manager", None)
            if action_manager is None:
                action_in = action_td["action"]
            else:
                action_in = action_manager.convert_policy_action_to_env_action(
                    action_td["action"]
                )

            if eval_rollout_buffer is not None:
                _compact_eval_rollout_buffer(env, eval_rollout_buffer)
                eval_rollout_buffer["action"][:, env.current_rollout_step].copy_(
                    action_td["action"]
                )
            step_start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action_in)
            env_step_time += time.perf_counter() - step_start
            env_step_count += env.num_envs

            done = terminated | truncated
            cumulative_reward += reward.float()
            step_count += 1

            newly_done = done.nonzero(as_tuple=False).squeeze(-1)
            for env_id in newly_done.tolist():
                if completed >= target_episodes:
                    break
                returns.append(float(cumulative_reward[env_id].item()))
                lengths.append(int(step_count[env_id].item()))
                if "success" in info:
                    successes.append(float(info["success"][env_id].item()))
                if "metrics" in info:
                    for key, value in info["metrics"].items():
                        metric_values.setdefault(key, []).append(
                            float(value[env_id].item())
                        )
                cumulative_reward[env_id] = 0.0
                step_count[env_id] = 0
                completed += 1
    finally:
        if env is not None:
            env.close()

    return {
        "num_episodes": completed,
        "avg_reward": float(np.mean(returns)) if returns else float("nan"),
        "avg_episode_length": float(np.mean(lengths)) if lengths else float("nan"),
        "success_rate": float(np.mean(successes)) if successes else float("nan"),
        "environment_fps": float(env_step_count / max(env_step_time, 1e-6)),
        "metrics": {
            key: float(np.mean(values))
            for key, values in metric_values.items()
            if values
        },
    }


def dump_json(data: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON artifact to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output


__all__ = [
    "build_policy_from_env",
    "dump_json",
    "evaluate_checkpoint",
    "resolve_device",
    "set_random_seed",
    "train_with_config",
]
