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

import gymnasium as gym
import numpy as np
import torch

from embodichain.lab.gym.envs import EmbodiedEnv


def test_demo_action_uses_single_action_space_for_vectorized_env() -> None:
    env = EmbodiedEnv.__new__(EmbodiedEnv)
    env.single_action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(8,),
        dtype=np.float32,
    )
    env.action_space = gym.vector.utils.batch_space(env.single_action_space, n=2)

    action = torch.zeros(8)

    normalized = env._normalize_demo_action_list([action])

    assert len(normalized) == 1
    assert normalized[0] is action
