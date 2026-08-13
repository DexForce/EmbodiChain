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

from embodichain.learning.rl.motion_policy_evaluation import load_policy_state_dict


def test_load_policy_state_dict_returns_embodichain_policy_weights(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    torch.save({"policy": {"actor.weight": torch.ones(2, 3)}}, checkpoint)

    state = load_policy_state_dict(checkpoint)

    assert torch.equal(state["actor.weight"], torch.ones(2, 3))
