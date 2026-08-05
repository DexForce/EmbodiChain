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

from embodichain.learning.rl.algo import build_algo
from embodichain.learning.rl.differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from embodichain.learning.rl.env import build_learning_env
from embodichain.learning.rl.evaluation import evaluate_episodes
from embodichain.learning.rl.models import build_mlp_from_cfg, build_policy
from embodichain.learning.rl.routing import get_trainer_class
from embodichain.learning.rl.utils import dict_to_tensordict, flatten_dict_observation
from embodichain.learning.rl.utils.trainer import Trainer
from embodichain.lab.gym.envs.managers.cfg import EventCfg
from embodichain.lab.gym.utils.registration import build_env, discover_task_packages
from embodichain.lab.gym.utils.gym_utils import get_manager_modules, config_to_cfg
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.utils.module_utils import find_function_from_modules
from embodichain.utils.utility import load_config

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
    gym_config_data = load_config(gym_config_path)
    gym_env_cfg = config_to_cfg(gym_config_data, manager_modules=get_manager_modules())
    if num_envs is not None:
        gym_env_cfg.num_envs = int(num_envs)
    if gym_env_cfg.sim_cfg is None:
        gym_env_cfg.sim_cfg = SimulationManagerCfg()
    gym_env_cfg.seed = getattr(gym_env_cfg, "seed", None)
    gym_env_cfg.sim_cfg.headless = headless
    gym_env_cfg.sim_cfg.gpu_id = gpu_id
    gym_env_cfg.sim_cfg.device = device
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
    flat_obs_space = getattr(env, "flattened_observation_space", None)
    if flat_obs_space is None:
        flat_obs_space = env.single_observation_space
    action_space = getattr(env, "action_space", None)
    if action_space is None:
        action_space = env.single_action_space
    env_action_dim = action_space.shape[-1]

    policy_name = policy_block["name"].lower()
    if policy_name == "actor_critic":
        actor = build_mlp_from_cfg(policy_block["actor"], obs_dim, env_action_dim)
        critic = build_mlp_from_cfg(policy_block["critic"], obs_dim, 1)
        return build_policy(
            policy_block,
            flat_obs_space,
            action_space,
            device,
            actor=actor,
            critic=critic,
        )
    if policy_name == "actor_only":
        actor = build_mlp_from_cfg(policy_block["actor"], obs_dim, env_action_dim)
        return build_policy(
            policy_block,
            flat_obs_space,
            action_space,
            device,
            actor=actor,
        )
    return build_policy(policy_block, flat_obs_space, action_space, device)


def _train_learning_config(
    cfg_json: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    trainer_cfg = deepcopy(cfg_json["trainer"])
    policy_block = deepcopy(cfg_json["policy"])
    algo_block = deepcopy(cfg_json["algorithm"])
    device = resolve_device(trainer_cfg.get("device", "cpu"))
    set_random_seed(int(trainer_cfg.get("seed", 1)), device)
    discover_task_packages()

    env_block = trainer_cfg["learning_env"]
    env_name = env_block if isinstance(env_block, str) else env_block["name"]
    env_kwargs = {} if isinstance(env_block, str) else dict(env_block.get("cfg", {}))
    num_envs = int(trainer_cfg.get("num_envs", 64))
    env = build_learning_env(env_name, num_envs=num_envs, device=device, **env_kwargs)
    eval_env = None
    if trainer_cfg.get("enable_eval", True):
        eval_env = build_learning_env(
            env_name,
            num_envs=int(trainer_cfg.get("num_eval_envs", 8)),
            device=device,
            **env_kwargs,
        )

    output_root = Path(output_dir)
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output_root / "logs"))
    try:
        policy = build_policy_from_env(policy_block, env, device)
        if "initial_log_std" in policy_block and hasattr(policy, "log_std"):
            with torch.no_grad():
                policy.log_std.fill_(float(policy_block["initial_log_std"]))
        algorithm = build_algo(algo_block["name"], algo_block["cfg"], policy, device)
        iterations = int(trainer_cfg.get("iterations", 1))
        eval_freq = int(trainer_cfg.get("eval_freq", 0))
        trainer_class = get_trainer_class(algorithm)
        best_eval_metric = trainer_cfg.get("best_eval_metric", "eval/avg_reward")
        best_eval_mode = trainer_cfg.get("best_eval_mode", "max")
        if trainer_class is DifferentiableTrainer:
            segment_length = int(trainer_cfg.get("segment_length", 25))
            update_horizon = int(trainer_cfg.get("update_horizon", segment_length))
            trainer = DifferentiableTrainer(
                cfg=DifferentiableTrainerCfg(
                    segment_length=segment_length,
                    update_horizon=update_horizon,
                    deterministic_actions=bool(
                        trainer_cfg.get("deterministic_actions", False)
                    ),
                    checkpoint_dir=str(checkpoint_dir),
                    experiment_name=str(trainer_cfg.get("exp_name", "benchmark_run")),
                    save_frequency_updates=int(
                        trainer_cfg.get("save_frequency_updates", 0)
                    ),
                    eval_frequency_steps=eval_freq,
                    num_eval_episodes=int(trainer_cfg.get("num_eval_episodes", 5)),
                    eval_seed=trainer_cfg.get("eval_seed"),
                    best_eval_metric=best_eval_metric,
                    best_eval_mode=best_eval_mode,
                ),
                env=env,
                policy=policy,
                algorithm=algorithm,
                writer=writer,
                eval_env=eval_env,
            )
            total_steps = iterations * update_horizon * num_envs
        else:
            buffer_size = int(trainer_cfg.get("buffer_size", 256))
            trainer = Trainer(
                policy=policy,
                env=env,
                algorithm=algorithm,
                buffer_size=buffer_size,
                batch_size=int(algorithm.cfg.batch_size),
                writer=writer,
                eval_freq=eval_freq,
                save_freq=int(trainer_cfg.get("save_freq", 0)),
                checkpoint_dir=str(checkpoint_dir),
                exp_name=str(trainer_cfg.get("exp_name", "benchmark_run")),
                use_wandb=False,
                eval_env=eval_env,
                num_eval_episodes=int(trainer_cfg.get("num_eval_episodes", 5)),
                eval_seed=trainer_cfg.get("eval_seed"),
                best_eval_metric=best_eval_metric,
                best_eval_mode=best_eval_mode,
            )
            total_steps = iterations * buffer_size * num_envs
        start_time = time.perf_counter()
        summary = trainer.train(total_steps)
        wall_time = time.perf_counter() - start_time
        checkpoint_path = trainer.save_checkpoint()
    finally:
        writer.close()
        if eval_env is not None:
            eval_env.close()
        env.close()

    peak_gpu_memory_mb = (
        torch.cuda.max_memory_allocated(device=device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
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


def train_with_config(
    cfg_json: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Train an RL configuration and return a structured summary."""
    if "learning_env" in cfg_json["trainer"]:
        return _train_learning_config(cfg_json, output_dir)
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
    is_learning_env = "learning_env" in trainer_cfg
    if is_learning_env:
        discover_task_packages()
        env_block = trainer_cfg["learning_env"]
        env_name = env_block if isinstance(env_block, str) else env_block["name"]
        env_kwargs = (
            {} if isinstance(env_block, str) else dict(env_block.get("cfg", {}))
        )
        eval_num_envs = int(
            num_envs if num_envs is not None else trainer_cfg.get("num_eval_envs", 4)
        )
    else:
        gym_config_data, gym_env_cfg = _build_env_cfg(
            gym_config_path=trainer_cfg["gym_config"],
            num_envs=(
                num_envs if num_envs is not None else trainer_cfg.get("num_eval_envs")
            ),
            headless=True,
            device=device,
            gpu_id=int(trainer_cfg.get("gpu_id", 0)),
        )
    env = None
    try:
        if is_learning_env:
            env = build_learning_env(
                env_name,
                num_envs=eval_num_envs,
                device=device,
                **env_kwargs,
            )
        else:
            env = build_env(gym_config_data["id"], base_env_cfg=gym_env_cfg)
        policy = build_policy_from_env(policy_block, env, device)
        eval_rollout_buffer = None
        if hasattr(env, "set_rollout_buffer"):
            eval_rollout_buffer = _allocate_eval_rollout_buffer(env, policy, device)
            env.set_rollout_buffer(eval_rollout_buffer)

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["policy"])
        policy.eval()

        def on_step(_: dict[str, Any]) -> None:
            if eval_rollout_buffer is not None:
                _compact_eval_rollout_buffer(env, eval_rollout_buffer)

        start_time = time.perf_counter()
        unified_metrics = evaluate_episodes(
            policy=policy,
            env=env,
            num_episodes=int(num_episodes),
            device=device,
            seed=trainer_cfg.get("eval_seed"),
            on_step=on_step,
        )
        elapsed = time.perf_counter() - start_time
        avg_length = unified_metrics["eval/avg_length"]
        return {
            "num_episodes": int(num_episodes),
            "avg_reward": unified_metrics["eval/avg_reward"],
            "avg_episode_length": avg_length,
            "success_rate": unified_metrics["eval/success_rate"],
            "environment_fps": float(num_episodes * avg_length / max(elapsed, 1e-6)),
            "metrics": {
                key.removeprefix("eval/metrics/"): value
                for key, value in unified_metrics.items()
                if key.startswith("eval/metrics/")
            },
        }
    finally:
        if env is not None:
            env.close()


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
