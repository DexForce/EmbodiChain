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
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

from embodichain.learning.rl.models import build_policy, get_registered_policy_names
from embodichain.learning.rl.models import build_mlp_from_cfg
from embodichain.learning.rl.algo import (
    RolloutKind,
    build_algo,
    get_registered_algo_names,
)
from embodichain.learning.rl.differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from embodichain.learning.rl.env import build_learning_env
from embodichain.learning.rl.routing import get_trainer_class
from embodichain.learning.rl.utils import dict_to_tensordict, flatten_dict_observation
from embodichain.learning.rl.utils.trainer import Trainer
from embodichain.utils import logger
from embodichain.lab.gym.utils.registration import (
    build_env,
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
from embodichain.lab.gym.utils.profiler import EnvProfilerCfg
from embodichain.utils.utility import load_config
from embodichain.utils.module_utils import find_function_from_modules
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.gym.envs.managers.cfg import EventCfg


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.

    Returns:
        Parsed training arguments.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain train-rl",
        description="Train an RL agent from a JSON or YAML config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config file (.json, .yaml, or .yml).",
    )
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable multi-GPU distributed training",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help=(
            "Enable per-section time profiling of gym env reset/step "
            "(report on env.close()). Requires trainer.gym_config."
        ),
    )
    parser.add_argument(
        "--profile_output",
        type=str,
        default=None,
        help="Dump the profiling report as JSON on env.close() (requires --profile).",
    )
    return parser.parse_args(argv)


def _resolve_profile_output(
    path: str | None,
    *,
    rank: int,
    world_size: int,
) -> str | None:
    if path is None or world_size <= 1:
        return path
    output = Path(path)
    return str(output.with_name(f"{output.stem}_rank{rank}{output.suffix}"))


def _build_learning_policy(
    policy_block: dict,
    env,
    device: torch.device,
):
    obs_dim = int(env.single_observation_space.shape[-1])
    action_dim = int(env.single_action_space.shape[-1])
    policy_name = policy_block["name"].lower()
    actor_cfg = policy_block.get("actor")
    critic_cfg = policy_block.get("critic")
    actor = (
        build_mlp_from_cfg(actor_cfg, obs_dim, action_dim)
        if actor_cfg is not None
        else None
    )
    critic = (
        build_mlp_from_cfg(critic_cfg, obs_dim, 1) if critic_cfg is not None else None
    )
    policy = build_policy(
        policy_block,
        env.single_observation_space,
        env.single_action_space,
        device,
        actor=actor,
        critic=critic,
    )
    if "initial_log_std" in policy_block and hasattr(policy, "log_std"):
        with torch.no_grad():
            policy.log_std.fill_(float(policy_block["initial_log_std"]))
    return policy


def _train_learning_env(
    cfg_data: dict,
    *,
    distributed: bool | None,
    profile: bool = False,
):
    """Train a lightweight registered environment through the unified CLI."""
    if profile:
        raise ValueError(
            "--profile requires trainer.gym_config; learning_env is unsupported."
        )
    trainer_cfg = cfg_data["trainer"]
    policy_block = cfg_data["policy"]
    algorithm_block = cfg_data["algorithm"]
    distributed = (
        bool(trainer_cfg.get("distributed", False))
        if distributed is None
        else distributed
    )
    if distributed:
        raise ValueError(
            "Learning environments do not yet support distributed training."
        )

    discover_task_packages()
    execute_init_hooks()
    seed = int(trainer_cfg.get("seed", 1))
    device = torch.device(trainer_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_block = trainer_cfg["learning_env"]
    if isinstance(env_block, str):
        env_name = env_block
        env_cfg = {}
    else:
        env_name = env_block["name"]
        env_cfg = dict(env_block.get("cfg", {}))
    num_envs = int(trainer_cfg.get("num_envs", 64))
    env = build_learning_env(
        env_name,
        num_envs=num_envs,
        device=device,
        **env_cfg,
    )

    enable_eval = bool(trainer_cfg.get("enable_eval", False))
    eval_env = None
    if enable_eval:
        eval_env = build_learning_env(
            env_name,
            num_envs=int(trainer_cfg.get("num_eval_envs", 16)),
            device=device,
            **env_cfg,
        )

    policy = _build_learning_policy(policy_block, env, device)
    algorithm = build_algo(
        algorithm_block["name"],
        dict(algorithm_block.get("cfg", {})),
        policy,
        device,
    )
    trainer_class = get_trainer_class(algorithm)

    exp_name = trainer_cfg.get("exp_name", f"{env_name}_{algorithm_block['name']}")
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_base = Path("outputs") / f"{exp_name}_{run_stamp}"
    log_dir = run_base / "logs" / exp_name
    checkpoint_dir = run_base / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    use_wandb = bool(trainer_cfg.get("use_wandb", False))
    if use_wandb:
        wandb.init(
            project=trainer_cfg.get("wandb_project_name", "embodichain-generic"),
            name=exp_name,
            config=cfg_data,
        )

    eval_freq = int(trainer_cfg.get("eval_freq", 0)) if enable_eval else 0
    eval_seed = int(trainer_cfg.get("eval_seed", seed + 10_000))
    iterations = int(trainer_cfg.get("iterations", 250))
    try:
        if trainer_class is DifferentiableTrainer:
            segment_length = int(trainer_cfg.get("segment_length", 16))
            update_horizon = int(trainer_cfg.get("update_horizon", segment_length))
            diff_cfg = DifferentiableTrainerCfg(
                segment_length=segment_length,
                update_horizon=update_horizon,
                deterministic_actions=bool(
                    trainer_cfg.get("deterministic_actions", False)
                ),
                checkpoint_dir=str(checkpoint_dir),
                experiment_name=exp_name,
                save_frequency_updates=int(
                    trainer_cfg.get("save_frequency_updates", 0)
                ),
                eval_frequency_steps=eval_freq,
                num_eval_episodes=int(trainer_cfg.get("num_eval_episodes", 5)),
                eval_seed=eval_seed,
                use_wandb=use_wandb,
                best_eval_metric=trainer_cfg.get("best_eval_metric", "eval/avg_reward"),
                best_eval_mode=trainer_cfg.get("best_eval_mode", "max"),
            )
            trainer = DifferentiableTrainer(
                cfg=diff_cfg,
                env=env,
                policy=policy,
                algorithm=algorithm,
                writer=writer,
                eval_env=eval_env,
            )
            default_steps = iterations * update_horizon * num_envs
        else:
            buffer_size = int(
                trainer_cfg.get("buffer_size", trainer_cfg.get("rollout_steps", 256))
            )
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
                exp_name=exp_name,
                use_wandb=use_wandb,
                eval_env=eval_env,
                num_eval_episodes=int(trainer_cfg.get("num_eval_episodes", 5)),
                eval_seed=eval_seed,
                best_eval_metric=trainer_cfg.get("best_eval_metric", "eval/avg_reward"),
                best_eval_mode=trainer_cfg.get("best_eval_mode", "max"),
            )
            default_steps = iterations * buffer_size * num_envs
        total_timesteps = int(trainer_cfg.get("total_timesteps", default_steps))
        trainer.train(total_timesteps)
        trainer.save_checkpoint()
        return trainer.get_summary()
    finally:
        writer.close()
        if use_wandb:
            wandb.finish()
        env.close()
        if eval_env is not None:
            eval_env.close()


def train_from_config(
    config_path: str,
    distributed: bool | None = None,
    *,
    profile: bool = False,
    profile_output: str | None = None,
):
    """Run training from a config file path.

    Args:
        config_path: Path to the training config file (.json, .yaml, or .yml).
        distributed: If True, run multi-GPU distributed training.
            If None, use trainer.distributed from config.
        profile: Enable gym ``EnvProfiler`` on the training environment.
        profile_output: Optional JSON dump path for the profiling report.
    """
    if profile_output is not None and not profile:
        raise ValueError("--profile_output requires --profile.")

    cfg_data = load_config(config_path)

    trainer_cfg = cfg_data["trainer"]
    if "learning_env" in trainer_cfg:
        return _train_learning_env(
            cfg_data,
            distributed=distributed,
            profile=profile,
        )
    policy_block = cfg_data["policy"]
    algo_block = cfg_data["algorithm"]

    if distributed is None:
        distributed = bool(trainer_cfg.get("distributed", False))

    rank = 0
    world_size = 1
    local_rank = 0
    if distributed:
        if not torch.distributed.is_available():
            raise RuntimeError(
                "Distributed training requested but torch.distributed is not available."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Distributed training with NCCL backend requires CUDA, "
                "but torch.cuda.is_available() is False."
            )
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK {local_rank} is out of range "
                f"(available GPUs: {torch.cuda.device_count()})."
            )
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

    exp_name = trainer_cfg.get("exp_name", "generic_exp")
    seed = int(trainer_cfg.get("seed", 1))
    device_str = trainer_cfg.get("device", "cpu")
    if distributed:
        device_str = f"cuda:{local_rank}"
    iterations = int(trainer_cfg.get("iterations", 250))
    buffer_size = int(
        trainer_cfg.get("buffer_size", trainer_cfg.get("rollout_steps", 2048))
    )
    enable_eval = bool(trainer_cfg.get("enable_eval", False))
    eval_freq = int(trainer_cfg.get("eval_freq", 10000))
    save_freq = int(trainer_cfg.get("save_freq", 50000))
    num_eval_episodes = int(trainer_cfg.get("num_eval_episodes", 5))
    headless = bool(trainer_cfg.get("headless", True))
    renderer = trainer_cfg.get("renderer", "hybrid")
    gpu_id = int(trainer_cfg.get("gpu_id", 0))
    num_envs = trainer_cfg.get("num_envs", None)
    wandb_project_name = trainer_cfg.get("wandb_project_name", "embodichain-generic")

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
    if rank == 0:
        logger.log_info(f"Device: {device}")
    if distributed and rank == 0:
        logger.log_info(f"Distributed training: world_size={world_size}")

    # Seeds
    effective_seed = seed + rank
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.backends.cudnn.deterministic = True
    if device.type == "cuda":
        torch.cuda.manual_seed_all(effective_seed)

    # Outputs
    if distributed:
        run_stamp = time.strftime("%Y%m%d_%H%M%S") if rank == 0 else None
        run_stamp_list = [run_stamp]
        torch.distributed.broadcast_object_list(run_stamp_list, src=0)
        run_stamp = run_stamp_list[0]
    else:
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_base = os.path.join("outputs", f"{exp_name}_{run_stamp}")
    log_dir = os.path.join(run_base, "logs")
    checkpoint_dir = os.path.join(run_base, "checkpoints")
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
    writer = SummaryWriter(f"{log_dir}/{exp_name}") if rank == 0 else None

    # Initialize Weights & Biases (optional)
    use_wandb = trainer_cfg.get("use_wandb", False)
    if use_wandb and rank == 0:
        wandb.init(project=wandb_project_name, name=exp_name, config=cfg_data)

    gym_config_path = Path(trainer_cfg["gym_config"])
    if rank == 0:
        logger.log_info(f"Current working directory: {Path.cwd()}")

    gym_config_data = load_config(str(gym_config_path))
    gym_env_cfg = config_to_cfg(gym_config_data, manager_modules=get_manager_modules())
    if num_envs is not None:
        gym_env_cfg.num_envs = int(num_envs)

    # Ensure sim configuration mirrors runtime overrides
    if gym_env_cfg.sim_cfg is None:
        gym_env_cfg.sim_cfg = SimulationManagerCfg()
    if device.type == "cuda":
        gpu_index = device.index
        if gpu_index is None:
            gpu_index = torch.cuda.current_device()
        gym_env_cfg.sim_cfg.device = torch.device(f"cuda:{gpu_index}")
        if hasattr(gym_env_cfg.sim_cfg, "gpu_id"):
            gym_env_cfg.sim_cfg.gpu_id = gpu_index
    else:
        gym_env_cfg.sim_cfg.device = torch.device("cpu")
    gym_env_cfg.sim_cfg.headless = headless
    gym_env_cfg.sim_cfg.render_cfg = RenderCfg(renderer=renderer)
    gym_env_cfg.sim_cfg.gpu_id = gpu_id
    if profile:
        gym_env_cfg.profiler = EnvProfilerCfg(
            enable_time=True,
            output_path=_resolve_profile_output(
                profile_output,
                rank=rank,
                world_size=world_size,
            ),
        )
    if rank == 0:
        logger.log_info(
            f"Loaded gym_config from {gym_config_path} (env_id={gym_config_data['id']}, num_envs={gym_env_cfg.num_envs}, headless={gym_env_cfg.sim_cfg.headless}, renderer={gym_env_cfg.sim_cfg.render_cfg.renderer}, device={gym_env_cfg.sim_cfg.device})"
        )

    env = build_env(gym_config_data["id"], base_env_cfg=gym_env_cfg)
    sample_obs, _ = env.reset()
    sample_obs_td = dict_to_tensordict(sample_obs, device)
    obs_dim = flatten_dict_observation(sample_obs_td).shape[-1]
    flat_obs_space = env.flattened_observation_space

    # Create evaluation environment only if enabled
    eval_env = None
    num_eval_envs = trainer_cfg.get("num_eval_envs", 4)
    if enable_eval and rank == 0:
        eval_gym_env_cfg = deepcopy(gym_env_cfg)
        eval_gym_env_cfg.num_envs = num_eval_envs
        eval_gym_env_cfg.sim_cfg.headless = True
        eval_gym_env_cfg.profiler = None
        eval_env = build_env(gym_config_data["id"], base_env_cfg=eval_gym_env_cfg)
        logger.log_info(
            f"Evaluation environment created (num_envs={num_eval_envs}, headless=True)"
        )

    # Build Policy via registry
    policy_name = policy_block["name"]
    env_action_dim = (
        env.get_wrapper_attr("action_manager").total_action_dim
        if env.get_wrapper_attr("action_manager") is not None
        else len(env.get_wrapper_attr("active_joint_ids"))
    )
    action_dim = policy_block.get("action_dim", env_action_dim)
    action_dim = int(action_dim)
    if action_dim != env_action_dim:
        raise ValueError(
            f"Configured policy.action_dim={action_dim} does not match env action dim {env_action_dim}."
        )
    # Build Policy via registry (actor/critic must be explicitly defined in JSON when using actor_critic/actor_only)
    if policy_name.lower() == "actor_critic":
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
            flat_obs_space,
            env.action_space,
            device,
            actor=actor,
            critic=critic,
        )
    elif policy_name.lower() == "actor_only":
        actor_cfg = policy_block.get("actor")
        if actor_cfg is None:
            raise ValueError(
                "ActorOnly requires 'actor' definition in JSON (policy.actor)."
            )

        actor = build_mlp_from_cfg(actor_cfg, obs_dim, action_dim)

        policy = build_policy(
            policy_block,
            flat_obs_space,
            env.action_space,
            device,
            actor=actor,
        )
    else:
        policy = build_policy(
            policy_block, env.observation_space, env.action_space, device
        )

    # Build Algorithm via factory
    algo_name = algo_block["name"].lower()
    algo_cfg = algo_block["cfg"]
    algo = build_algo(
        algo_name,
        algo_cfg,
        policy,
        device,
        distributed=distributed,
    )
    if algo.rollout_kind is RolloutKind.DIFFERENTIABLE:
        raise ValueError(
            "Differentiable algorithms require trainer.learning_env; "
            "simulator gym_config environments use standard rollouts."
        )

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
    # Parse eval events (only if evaluation is enabled)
    if enable_eval:
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
        buffer_size=buffer_size,
        batch_size=algo_cfg["batch_size"],
        writer=writer,
        eval_freq=eval_freq if enable_eval else 0,  # Disable eval if not enabled
        save_freq=save_freq,
        checkpoint_dir=checkpoint_dir,
        exp_name=exp_name,
        use_wandb=use_wandb,
        eval_env=eval_env,  # None if enable_eval=False
        event_cfg=train_event_cfg,
        eval_event_cfg=eval_event_cfg if (enable_eval and rank == 0) else {},
        num_eval_episodes=num_eval_episodes,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        eval_seed=int(trainer_cfg.get("eval_seed", seed + 10_000)),
        best_eval_metric=trainer_cfg.get("best_eval_metric", "eval/avg_reward"),
        best_eval_mode=trainer_cfg.get("best_eval_mode", "max"),
    )

    if rank == 0:
        logger.log_info("Generic training initialized")
        logger.log_info(f"Task: {type(env).__name__}")
        logger.log_info(
            f"Policy: {policy_name} (available: {get_registered_policy_names()})"
        )
        logger.log_info(
            f"Algorithm: {algo_name} (available: {get_registered_algo_names()})"
        )

    total_steps = int(iterations * buffer_size * env.num_envs * world_size)
    if rank == 0:
        logger.log_info(
            f"Total steps: {total_steps} (iterations≈{iterations}, world_size={world_size})"
        )

    try:
        trainer.train(total_steps)
    except KeyboardInterrupt:
        if rank == 0:
            logger.log_info("Training interrupted by user")
    finally:
        trainer.save_checkpoint()
        if writer is not None:
            writer.close()
        if use_wandb and rank == 0:
            try:
                wandb.finish()
            except Exception:
                pass

        # Clean up environments to prevent resource leaks
        try:
            if env is not None:
                env.close()
        except Exception as e:
            if rank == 0:
                logger.log_warning(f"Failed to close training environment: {e}")

        try:
            if eval_env is not None:
                eval_env.close()
        except Exception as e:
            if rank == 0:
                logger.log_warning(f"Failed to close evaluation environment: {e}")

        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        if rank == 0:
            logger.log_info("Training finished")


def cli(argv: Sequence[str] | None = None) -> None:
    """Command-line interface for RL training.

    Parses CLI arguments and launches training from a config file.

    Task packages are discovered (and init hooks executed) before training so
    that task environments registered in separate packages (e.g.
    ``embodichain_tasks``) are available to ``build_env``. This mirrors the
    ``run_env`` CLI.
    """
    args = parse_args(argv)

    # Discover all installed task packages and run init hooks (register custom
    # manager modules / asset resolvers) before building any environment.
    discover_task_packages()
    execute_init_hooks()

    train_from_config(
        args.config,
        distributed=args.distributed,
        profile=args.profile,
        profile_output=args.profile_output,
    )


if __name__ == "__main__":
    cli()


__all__ = ["cli", "parse_args", "train_from_config"]
