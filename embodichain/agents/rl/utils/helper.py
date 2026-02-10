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

"""Helper utilities for RL training.

This module provides utility functions for RL algorithms.
"""

import torch
import numpy as np
from tensordict import TensorDict


def dict_to_tensordict(obs_dict: dict, device: torch.device) -> TensorDict:
    """Convert nested dict observation to TensorDict recursively.

    Args:
        obs_dict: Nested observation dictionary
        device: Device to place tensors on

    Returns:
        TensorDict with nested structure preserved and "observation" key
    """

    def _recursive_convert(d):
        """Recursively convert dict to TensorDict-compatible structure."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively convert nested dicts
                result[key] = _recursive_convert(value)
            elif isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            else:
                result[key] = torch.tensor(value, device=device)
        return result

    # Convert the observation dict structure
    converted = _recursive_convert(obs_dict)

    # Infer batch_size from first tensor we find
    def _get_first_tensor_batch_size(d):
        """Find first tensor and get its batch dimension."""
        for value in d.values():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            elif isinstance(value, dict):
                bs = _get_first_tensor_batch_size(value)
                if bs is not None:
                    return bs
        return None

    batch_size = _get_first_tensor_batch_size(converted)
    if batch_size is None:
        batch_size = 1  # Default if no tensors found

    # Wrap in TensorDict with explicit batch_size
    obs_td = TensorDict(converted, batch_size=[batch_size], device=device)

    # Wrap observation in outer TensorDict with "observation" key
    return TensorDict({"observation": obs_td}, batch_size=[batch_size], device=device)


def mean_scalar(x) -> float:
    """Convert tensor or array to scalar float (mean if needed).

    Args:
        x: Tensor, array, or scalar value

    Returns:
        Float scalar value
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return float(np.mean(x))


def pack_log_dict(prefix: str, data: dict) -> dict:
    """Pack data dict into logging dict with prefix.

    Args:
        prefix: Prefix for keys (e.g., "train", "eval")
        data: Dictionary of values to pack

    Returns:
        Dictionary with prefixed keys and scalar values
    """
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        try:
            out[f"{prefix}/{k}"] = mean_scalar(v)
        except Exception:
            continue
    return out


def compute_gae(
    rollout: TensorDict,
    gamma: float,
    gae_lambda: float,
) -> TensorDict:
    """Compute Generalized Advantage Estimation (GAE) on rollout TensorDict.

    This follows the TorchRL convention where rollout has shape [T, N, ...].
    Computes advantage and value_target in-place and returns the modified TensorDict.

    Args:
        rollout: TensorDict with batch_size=[T, N] containing:
            - "value": Tensor[T, N, 1] - state values
            - "next": TensorDict with:
                - "reward": Tensor[T, N, 1]
                - "done": Tensor[T, N, 1]
                - "value": Tensor[T, N, 1] - next state values (bootstrapped)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        TensorDict with added keys:
            - "advantage": Tensor[T, N, 1]
            - "value_target": Tensor[T, N, 1]
    """
    T, N = rollout.batch_size[:2]
    device = rollout.device

    # Extract tensors - shape [T, N, 1]
    values = rollout["value"]
    rewards = rollout["next"]["reward"]
    dones = rollout["next"]["done"].float()

    # Bootstrap values: use next state value from rollout["next"]["value"]
    # This is computed during collection by evaluating policy on next_obs
    if "value" in rollout["next"]:
        bootstrap_values = rollout["next"]["value"]
    else:
        # If not provided, assume 0 (terminal state)
        bootstrap_values = torch.zeros_like(values)

    # Compute GAE advantages using backward iteration
    # advantage[t] = delta[t] + (gamma * gae_lambda) * (1 - done[t]) * advantage[t+1]
    # where delta[t] = reward[t] + gamma * (1 - done[t]) * V(s_{t+1}) - V(s_t)
    # V(s_{t+1}) comes from bootstrap_values[t] which was computed on next_obs[t]

    advantages = torch.zeros_like(values)
    gae = torch.zeros(N, 1, device=device)

    # Iterate backwards through time
    for t in reversed(range(T)):
        # Compute TD error (delta)
        # bootstrap_values[t] is V(s_{t+1}), the value of the next state after action at t
        delta = rewards[t] + gamma * bootstrap_values[t] * (1.0 - dones[t]) - values[t]

        # Compute GAE recursively
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae

    # Compute value targets (for value function loss)
    value_targets = advantages + values

    # Add to rollout TensorDict (in-place)
    rollout["advantage"] = advantages
    rollout["value_target"] = value_targets

    return rollout
