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

import torch


def flatten_dict_observation(input_dict: dict) -> torch.Tensor:
    """
    Flatten hierarchical dict observations from ObservationManager.

    Recursively traverse nested dicts, collect all tensor values,
    flatten each to (num_envs, -1), and concatenate in sorted key order.

    Args:
        input_dict: Nested dict structure, e.g. {"robot": {"qpos": tensor, "ee_pos": tensor}, "object": {...}}

    Returns:
        Concatenated flat tensor of shape (num_envs, total_dim)
    """
    obs_list = []

    def _collect_tensors(d, prefix=""):
        """Recursively collect tensors from nested dicts in sorted order."""
        for key in sorted(d.keys()):
            full_key = f"{prefix}/{key}" if prefix else key
            value = d[key]
            if isinstance(value, dict):
                _collect_tensors(value, full_key)
            elif isinstance(value, torch.Tensor):
                # Flatten tensor to (num_envs, -1) shape
                obs_list.append(value.flatten(start_dim=1))

    _collect_tensors(input_dict)

    if not obs_list:
        raise ValueError("No tensors found in observation dict")

    result = torch.cat(obs_list, dim=-1)
    return result
