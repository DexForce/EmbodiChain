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

from collections.abc import Mapping
from typing import Any

import torch
from tensordict import TensorDict

__all__ = [
    "dict_to_tensordict",
    "flatten_dict_observation",
]


def flatten_dict_observation(obs: TensorDict) -> torch.Tensor:
    """Flatten a hierarchical observation TensorDict into a 2D tensor.

    Args:
        obs: Observation TensorDict with batch dimension `[num_envs]`.

    Returns:
        Flattened observation tensor of shape `[num_envs, obs_dim]`.
    """
    obs_list: list[torch.Tensor] = []

    def _collect_tensors(data: TensorDict) -> None:
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, TensorDict):
                _collect_tensors(value)
            elif isinstance(value, torch.Tensor):
                obs_list.append(value.flatten(start_dim=1))

    _collect_tensors(obs)

    if not obs_list:
        raise ValueError("No tensors found in observation TensorDict.")

    return torch.cat(obs_list, dim=-1)


def dict_to_tensordict(
    obs_dict: TensorDict | Mapping[str, Any], device: torch.device | str
) -> TensorDict:
    """Convert an environment observation mapping into a TensorDict.

    Args:
        obs_dict: Environment observation returned by `reset()` or `step()`.
        device: Target device for the resulting TensorDict.

    Returns:
        Observation TensorDict moved onto the target device.
    """
    if isinstance(obs_dict, TensorDict):
        return obs_dict.to(device)
    if not isinstance(obs_dict, Mapping):
        raise TypeError(
            f"Expected observation mapping or TensorDict, got {type(obs_dict)!r}."
        )
    return TensorDict.from_dict(dict(obs_dict), device=device)
