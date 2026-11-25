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

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from tensordict import TensorDict

__all__ = [
    "dict_to_tensordict",
    "flatten_dict_observation",
    "flatten_observation_groups",
]


def _flatten_observation_leaf(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(-1) if value.ndim == 1 else value.flatten(start_dim=1)


def _observation_leaves(obs: TensorDict) -> list[torch.Tensor]:
    obs_list: list[torch.Tensor] = []

    def _collect_tensors(data: TensorDict) -> None:
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, TensorDict):
                _collect_tensors(value)
            elif isinstance(value, torch.Tensor):
                obs_list.append(_flatten_observation_leaf(value))

    _collect_tensors(obs)

    if not obs_list:
        raise ValueError("No tensors found in observation TensorDict.")

    return obs_list


def _pack_observation_groups(
    obs: TensorDict,
    groups: Sequence[str] | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve ordered leaves, optionally packing into owned collector storage."""
    if groups is None:
        grouped = [_observation_leaves(obs)]
    else:
        if len(groups) == 0:
            raise ValueError("Observation groups cannot be empty.")
        grouped = []
        for group in groups:
            value: TensorDict | torch.Tensor = obs
            for key in group.split("."):
                if not isinstance(value, TensorDict) or key not in value.keys():
                    raise KeyError(
                        f"Observation group '{group}' was not found. "
                        f"Available top-level groups: {list(obs.keys())}."
                    )
                value = value[key]
            if isinstance(value, TensorDict):
                grouped.append(_observation_leaves(value))
            elif isinstance(value, torch.Tensor):
                grouped.append([_flatten_observation_leaf(value)])
            else:
                raise TypeError(
                    f"Observation group '{group}' must contain tensors, got "
                    f"{type(value)!r}."
                )

    # Preserve group-by-group dtype promotion. Concatenate multiple leaves
    # before copying into storage that may alias an observation producer.
    flattened = [
        leaves[0] if len(leaves) == 1 else torch.cat(leaves, dim=-1)
        for leaves in grouped
    ]
    result = flattened[0] if len(flattened) == 1 else torch.cat(flattened, dim=-1)
    if out is not None:
        out.copy_(result)
        return out
    return result


def flatten_dict_observation(obs: TensorDict) -> torch.Tensor:
    """Flatten a hierarchical observation TensorDict into a 2D tensor.

    Args:
        obs: Observation TensorDict with batch dimension `[num_envs]`.

    Returns:
        Flattened observation tensor of shape `[num_envs, obs_dim]`.
        A single leaf may share storage with the input; clone it when retaining
        a snapshot across environment steps that reuse observation buffers.
    """
    return _pack_observation_groups(obs, None)


def flatten_observation_groups(
    obs: TensorDict,
    groups: Sequence[str] | None,
) -> torch.Tensor:
    """Flatten selected observation groups in the configured order.

    A group name selects one top-level TensorDict key. Dot-separated names may
    select nested keys, for example ``"robot.qpos"``. When ``groups`` is
    ``None``, every tensor is flattened using
    :func:`flatten_dict_observation` for compatibility with existing tasks.

    Args:
        obs: Observation TensorDict with batch dimension ``[num_envs]``.
        groups: Observation group names consumed by one model.

    Returns:
        Concatenated observation tensor with shape ``[num_envs, obs_dim]``.
        A single leaf may share storage with the input; clone it when retaining
        a snapshot across environment steps that reuse observation buffers.

    Raises:
        ValueError: If ``groups`` is empty or a selected group has no tensors.
        KeyError: If a configured group is absent from ``obs``.
    """
    return _pack_observation_groups(obs, groups)


def dict_to_tensordict(
    obs_dict: torch.Tensor | TensorDict | Mapping[str, Any],
    device: torch.device | str,
) -> TensorDict:
    """Convert an environment observation mapping into a TensorDict.

    Args:
        obs_dict: Tensor or mapping returned by ``reset()`` or ``step()``.
        device: Target device for the resulting TensorDict.

    Returns:
        Observation TensorDict moved onto the target device.
    """
    if isinstance(obs_dict, TensorDict):
        return obs_dict.to(device)
    if isinstance(obs_dict, torch.Tensor):
        tensor = obs_dict.to(device)
        batch_size = [tensor.shape[0]] if tensor.ndim > 1 else []
        return TensorDict({"obs": tensor}, batch_size=batch_size, device=device)
    if not isinstance(obs_dict, Mapping):
        raise TypeError(
            f"Expected tensor, observation mapping, or TensorDict, "
            f"got {type(obs_dict)!r}."
        )
    return TensorDict.from_dict(dict(obs_dict), device=device)
