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

"""Shared deterministic episode evaluation for all RL trainers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from embodichain.learning.rl.utils import (
    dict_to_tensordict,
    flatten_dict_observation,
)

__all__ = ["evaluate_episodes"]


def _flat_observation(observation: Any, device: torch.device) -> torch.Tensor:
    tensor_dict = dict_to_tensordict(observation, device)
    return flatten_dict_observation(tensor_dict)


def _action_for_env(env: Any, action: torch.Tensor) -> Any:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None and hasattr(env, "get_wrapper_attr"):
        try:
            action_manager = env.get_wrapper_attr("action_manager")
        except AttributeError:
            action_manager = None
    if action_manager is None:
        return action
    return action_manager.convert_policy_action_to_env_action(action)


def _selected_values(value: Any, indices: torch.Tensor) -> list[float]:
    if isinstance(value, torch.Tensor):
        selected = value.detach().reshape(-1)[indices.to(value.device)]
        return [float(item) for item in selected.cpu().tolist()]
    array = np.asarray(value).reshape(-1)
    return [float(array[index]) for index in indices.cpu().tolist()]


def _is_per_env_scalar_metric(value: Any, num_envs: int) -> bool:
    """Return True for metrics with one scalar per env (shape ``[N]`` / ``[N, 1]``)."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return False
        if value.ndim == 0:
            return num_envs == 1
        if value.shape[0] != num_envs:
            return False
        return value.ndim == 1 or value.shape[1:] in {(1,)}
    array = np.asarray(value)
    if array.size == 0:
        return False
    if array.ndim == 0:
        return num_envs == 1
    if array.shape[0] != num_envs:
        return False
    return array.ndim == 1 or array.shape[1:] in {(1,)}


@torch.no_grad()
def evaluate_episodes(
    *,
    policy: torch.nn.Module,
    env: Any,
    num_episodes: int,
    device: torch.device | str,
    seed: int | None = None,
    on_step: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    """Evaluate exactly ``num_episodes`` completed asynchronous episodes."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    device = torch.device(device)
    previous_training = policy.training
    policy.eval()

    returns: list[float] = []
    lengths: list[float] = []
    successes: list[float] = []
    metric_values: dict[str, list[float]] = {}
    num_envs = int(env.num_envs)
    current_return = torch.zeros(num_envs, dtype=torch.float32, device=device)
    current_length = torch.zeros(num_envs, dtype=torch.long, device=device)

    try:
        observation, _ = env.reset(seed=seed)
        while len(returns) < num_episodes:
            flat_observation = _flat_observation(observation, device)
            policy_input = TensorDict(
                {"obs": flat_observation},
                batch_size=[num_envs],
                device=device,
            )
            policy_output = policy.get_action(policy_input, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(
                _action_for_env(env, policy_output["action"])
            )
            reward = torch.as_tensor(reward, device=device).reshape(num_envs)
            done = (
                torch.as_tensor(terminated, device=device, dtype=torch.bool)
                | torch.as_tensor(truncated, device=device, dtype=torch.bool)
            ).reshape(num_envs)
            current_return += reward.float()
            current_length += 1

            if on_step is not None:
                on_step(info)

            done_indices = torch.nonzero(done, as_tuple=False).squeeze(-1)
            if done_indices.numel() == 0:
                continue
            remaining = num_episodes - len(returns)
            selected = done_indices[:remaining]
            returns.extend(_selected_values(current_return, selected))
            lengths.extend(_selected_values(current_length, selected))

            if isinstance(info, Mapping):
                if "success" in info:
                    successes.extend(_selected_values(info["success"], selected))
                metrics = info.get("metrics", {})
                if isinstance(metrics, Mapping):
                    for name, value in metrics.items():
                        # Skip vector/matrix metrics: reshape(-1) would break env indexing.
                        if not _is_per_env_scalar_metric(value, num_envs):
                            continue
                        metric_values.setdefault(str(name), []).extend(
                            _selected_values(value, selected)
                        )

            current_return[done_indices] = 0.0
            current_length[done_indices] = 0
    finally:
        policy.train(previous_training)

    result = {
        "eval/avg_reward": float(np.mean(returns)),
        "eval/avg_length": float(np.mean(lengths)),
        "eval/success_rate": (float(np.mean(successes)) if successes else float("nan")),
    }
    for name, values in metric_values.items():
        if values:
            result[f"eval/metrics/{name}"] = float(np.mean(values))
    return result
