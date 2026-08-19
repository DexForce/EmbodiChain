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

"""Shared environment and Policy construction for RL training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
from embodichain.lab.gym.utils.profiler import EnvProfilerCfg
from embodichain.lab.gym.utils.registration import build_env
from embodichain.lab.sim import SimulationManagerCfg
from embodichain.lab.sim.cfg import RenderCfg
from embodichain.learning.rl.env import build_learning_env
from embodichain.learning.rl.models import build_mlp_from_cfg, build_policy
from embodichain.learning.rl.utils import dict_to_tensordict, flatten_dict_observation
from embodichain.utils.utility import load_config

__all__ = [
    "PolicyRuntime",
    "build_gym_policy_runtime",
    "build_learning_policy_runtime",
]


@dataclass(frozen=True)
class _GymEnvironmentRuntime:
    """A simulator task reconstructed from one training configuration."""

    env: Any
    env_id: str
    env_cfg: Any
    gym_config: dict[str, Any]
    gym_config_path: Path


@dataclass(frozen=True)
class PolicyRuntime:
    """An Environment and Policy reconstructed from one training configuration."""

    env: Any
    policy: torch.nn.Module
    device: torch.device
    env_id: str
    env_cfg: Any | None = None
    gym_config: dict[str, Any] | None = None
    gym_config_path: Path | None = None

    def close(self) -> None:
        """Close the Environment without terminating the current process."""
        _close_environment(self.env)


def _resolve_config_reference(
    value: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    """Resolve a referenced config relative to its containing config file."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is not None:
        candidate = Path(base_dir).expanduser().resolve() / path
        if candidate.exists():
            return candidate
    return path


def _build_learning_environment(
    config: dict[str, Any],
    *,
    device: torch.device,
    num_envs: int,
) -> tuple[str, Any]:
    """Build the lightweight Environment declared by a training config."""
    env_block = config["trainer"]["learning_env"]
    if isinstance(env_block, str):
        env_name = env_block
        env_config: dict[str, Any] = {}
    else:
        env_name = env_block["name"]
        env_config = dict(env_block.get("cfg", {}))
    return str(env_name), build_learning_env(
        str(env_name),
        num_envs=num_envs,
        device=device,
        **env_config,
    )


def build_learning_policy_runtime(
    config: dict[str, Any],
    *,
    device: torch.device,
    num_envs: int,
) -> PolicyRuntime:
    """Build a lightweight Environment and its configured Policy."""
    env_name, env = _build_learning_environment(
        config,
        device=device,
        num_envs=num_envs,
    )
    try:
        policy = _build_learning_policy(config["policy"], env, device)
    except Exception:
        env.close()
        raise
    return PolicyRuntime(env, policy, device, env_name)


def _build_gym_environment(
    config: dict[str, Any],
    *,
    simulation_device: torch.device,
    num_envs: int | None,
    headless: bool,
    renderer: str,
    gpu_id: int,
    config_dir: str | Path | None = None,
    profiler: EnvProfilerCfg | None = None,
) -> _GymEnvironmentRuntime:
    """Build the simulator Environment declared by a training config."""
    trainer_cfg = config["trainer"]
    gym_config_path = _resolve_config_reference(
        trainer_cfg["gym_config"],
        base_dir=config_dir,
    )
    gym_config = load_config(gym_config_path)
    env_cfg = config_to_cfg(gym_config, manager_modules=get_manager_modules())
    if num_envs is not None:
        env_cfg.num_envs = int(num_envs)
    if env_cfg.sim_cfg is None:
        env_cfg.sim_cfg = SimulationManagerCfg()
    env_cfg.sim_cfg.sim_device = simulation_device
    env_cfg.sim_cfg.headless = headless
    env_cfg.sim_cfg.render_cfg = RenderCfg(renderer=renderer)
    env_cfg.sim_cfg.gpu_id = (
        simulation_device.index
        if simulation_device.type == "cuda" and simulation_device.index is not None
        else gpu_id
    )
    env_cfg.profiler = profiler
    env = build_env(gym_config["id"], base_env_cfg=env_cfg)
    return _GymEnvironmentRuntime(
        env=env,
        env_id=str(gym_config["id"]),
        env_cfg=env_cfg,
        gym_config=gym_config,
        gym_config_path=gym_config_path.resolve(),
    )


def build_gym_policy_runtime(
    config: dict[str, Any],
    *,
    device: torch.device,
    num_envs: int | None,
    headless: bool,
    renderer: str,
    gpu_id: int,
    config_dir: str | Path | None = None,
    profiler: EnvProfilerCfg | None = None,
    simulation_device: torch.device | None = None,
) -> PolicyRuntime:
    """Build a simulator task and the Policy declared by its training config."""
    task = _build_gym_environment(
        config,
        simulation_device=simulation_device or device,
        num_envs=num_envs,
        headless=headless,
        renderer=renderer,
        gpu_id=gpu_id,
        config_dir=config_dir,
        profiler=profiler,
    )
    env = task.env
    try:
        sample_observation, _ = env.reset()
        sample_observation_td = dict_to_tensordict(sample_observation, device)
        observation_dim = int(flatten_dict_observation(sample_observation_td).shape[-1])
        action_manager = env.get_wrapper_attr("action_manager")
        environment_action_dim = (
            action_manager.total_action_dim
            if action_manager is not None
            else len(env.get_wrapper_attr("active_joint_ids"))
        )
        policy = _build_gym_policy(
            config["policy"],
            env=env,
            device=device,
            observation_dim=observation_dim,
            action_dim=environment_action_dim,
        )
    except Exception:
        _close_environment(env)
        raise
    return PolicyRuntime(
        env=env,
        policy=policy,
        device=device,
        env_id=task.env_id,
        env_cfg=task.env_cfg,
        gym_config=task.gym_config,
        gym_config_path=task.gym_config_path,
    )


def _build_gym_policy(
    policy_block: dict[str, Any],
    *,
    env: Any,
    device: torch.device,
    observation_dim: int,
    action_dim: int,
) -> torch.nn.Module:
    configured_action_dim = int(policy_block.get("action_dim", action_dim))
    if configured_action_dim != action_dim:
        raise ValueError(
            f"Configured policy.action_dim={configured_action_dim} does not match "
            f"env action dim {action_dim}."
        )
    policy_name = str(policy_block["name"]).lower()
    if policy_name == "actor_critic":
        actor_cfg = policy_block.get("actor")
        critic_cfg = policy_block.get("critic")
        if actor_cfg is None or critic_cfg is None:
            raise ValueError(
                "ActorCritic requires policy.actor and policy.critic definitions."
            )
        return build_policy(
            policy_block,
            env.flattened_observation_space,
            env.action_space,
            device,
            actor=build_mlp_from_cfg(actor_cfg, observation_dim, action_dim),
            critic=build_mlp_from_cfg(critic_cfg, observation_dim, 1),
        )
    if policy_name == "actor_only":
        actor_cfg = policy_block.get("actor")
        if actor_cfg is None:
            raise ValueError("ActorOnly requires a policy.actor definition.")
        return build_policy(
            policy_block,
            env.flattened_observation_space,
            env.action_space,
            device,
            actor=build_mlp_from_cfg(actor_cfg, observation_dim, action_dim),
        )
    return build_policy(
        policy_block,
        env.observation_space,
        env.action_space,
        device,
    )


def _build_learning_policy(
    policy_block: dict[str, Any],
    env: Any,
    device: torch.device,
) -> torch.nn.Module:
    observation_dim = int(env.single_observation_space.shape[-1])
    action_dim = int(env.single_action_space.shape[-1])
    actor_cfg = policy_block.get("actor")
    critic_cfg = policy_block.get("critic")
    policy = build_policy(
        policy_block,
        env.single_observation_space,
        env.single_action_space,
        device,
        actor=(
            build_mlp_from_cfg(actor_cfg, observation_dim, action_dim)
            if actor_cfg is not None
            else None
        ),
        critic=(
            build_mlp_from_cfg(critic_cfg, observation_dim, 1)
            if critic_cfg is not None
            else None
        ),
    )
    if "initial_log_std" in policy_block and hasattr(policy, "log_std"):
        with torch.no_grad():
            policy.log_std.fill_(float(policy_block["initial_log_std"]))
    return policy


def _close_environment(env: Any) -> None:
    target = getattr(env, "unwrapped", env)
    if getattr(target, "sim", None) is not None:
        target.close(exit_process=False)
    else:
        env.close()
