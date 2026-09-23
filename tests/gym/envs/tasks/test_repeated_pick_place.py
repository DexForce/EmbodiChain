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

"""Tests for the RLinf repeated pick-and-place task state reset contract."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain_tasks.manipulation.repeated_pick_place import (
    RepeatedPickPlaceRlinfEnv,
)


def test_repeated_pick_place_partial_reset_preserves_other_hold_counters(
    monkeypatch,
) -> None:
    """Resetting one vector row must preserve another row's hold progress."""
    env = RepeatedPickPlaceRlinfEnv.__new__(RepeatedPickPlaceRlinfEnv)
    env._num_envs = 2
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._success_hold_steps = torch.tensor([2, 1], dtype=torch.int32)

    monkeypatch.setattr(EmbodiedEnv, "reset", lambda self, *args, **kwargs: (None, {}))

    observation, info = env.reset(options={"reset_ids": [0]})

    assert observation is None
    assert info == {}
    torch.testing.assert_close(
        env._success_hold_steps, torch.tensor([0, 1], dtype=torch.int32)
    )
