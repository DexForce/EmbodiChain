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

import torch
from tensordict import TensorDict


def flatten_dict_observation(obs: TensorDict) -> torch.Tensor:
    """
    Flatten hierarchical TensorDict observations from ObservationManager.

    Recursively traverse nested TensorDicts, collect all tensor values,
    flatten each to (num_envs, -1), and concatenate in sorted key order.

    Args:
        obs: Nested TensorDict structure, e.g. TensorDict(robot=TensorDict(qpos=..., qvel=...), ...)

    Returns:
        Concatenated flat tensor of shape (num_envs, total_dim)
    """
    obs_list = []

    def _collect_tensors(d, prefix=""):
        """Recursively collect tensors from nested TensorDicts in sorted order."""
        for key in sorted(d.keys()):
            full_key = f"{prefix}/{key}" if prefix else key
            value = d[key]
            if isinstance(value, TensorDict):
                _collect_tensors(value, full_key)
            elif isinstance(value, torch.Tensor):
                # Flatten tensor to (num_envs, -1) shape
                obs_list.append(value.flatten(start_dim=1))

    _collect_tensors(obs)

    if not obs_list:
        raise ValueError("No tensors found in observation TensorDict")

    result = torch.cat(obs_list, dim=-1)
    return result
