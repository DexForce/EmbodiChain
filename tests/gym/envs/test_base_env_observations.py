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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import warnings

import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs import BaseEnv

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("num_envs", [1, 4])
def test_proprioception_selects_joints_without_adding_batch_dims(num_envs: int) -> None:
    """Joint selection changes features, preserving one batch entry per environment."""
    joint_values = torch.arange(6, dtype=torch.float32).repeat(num_envs, 1)
    env_offsets = torch.arange(num_envs, dtype=torch.float32).unsqueeze(1) * 10
    field_offsets = {"qpos": 0, "qvel": 100, "qf": 200}
    proprioception = TensorDict(
        {
            key: joint_values + env_offsets + offset
            for key, offset in field_offsets.items()
        },
        batch_size=[num_envs],
        device="cpu",
    )
    env = BaseEnv.__new__(BaseEnv)
    env._num_envs = num_envs
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._profiler = SimpleNamespace(section=lambda name: nullcontext())
    env.robot = SimpleNamespace(get_proprioception=Mock(return_value=proprioception))
    env._get_sensor_obs = Mock(return_value={})

    # Repeat a selection, then change its order/content without changing its width.
    for joint_ids in ([5, 0, 2], [5, 0, 2], [1, 4, 3]):
        env.active_joint_ids = joint_ids
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            obs = env.get_obs()

        assert obs.batch_size == torch.Size([num_envs])
        assert obs["robot"].batch_size == torch.Size([num_envs])
        for key, offset in field_offsets.items():
            expected = (
                torch.tensor(joint_ids, dtype=torch.float32) + env_offsets + offset
            )
            torch.testing.assert_close(obs["robot", key], expected)
            torch.testing.assert_close(
                proprioception[key], joint_values + env_offsets + offset
            )
        assert proprioception.batch_size == torch.Size([num_envs])
