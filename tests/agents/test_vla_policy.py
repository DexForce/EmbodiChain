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

import torch

from embodichain.agents.rl.models.vla_policy import VLAPolicy


def test_vla_policy_slice_obs_item_with_mapping_batch():
    policy = VLAPolicy(
        device=torch.device("cpu"),
        policy_cfg={"vla": {"model_path": "dummy_model_path"}},
        action_space=2,
    )
    obs = {
        "robot": {
            "qpos": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
            "qvel": torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32),
        },
        "image": torch.tensor([[9.0], [10.0]], dtype=torch.float32),
    }

    assert policy._infer_batch_size(obs) == 2
    obs_1 = policy._slice_obs_item(obs, 1)
    assert torch.allclose(obs_1["robot"]["qpos"], torch.tensor([3.0, 4.0]))
    assert torch.allclose(obs_1["robot"]["qvel"], torch.tensor([7.0, 8.0]))
    assert torch.allclose(obs_1["image"], torch.tensor([10.0]))
