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

"""Tests for task-environment seed reset behavior."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import embodichain.lab.gym.envs.base_env as base_env_module
from embodichain.lab.gym.envs import BaseEnv


class _ProfilerStub:
    @contextmanager
    def section(self, *args, **kwargs):
        del args, kwargs
        yield


class _ResetEnv(BaseEnv):
    """BaseEnv reset surface without constructing a simulator."""

    def __init__(self) -> None:
        self.cfg = SimpleNamespace(seed=None)
        self._num_envs = 1
        self.sim = SimpleNamespace(
            device=torch.device("cpu"),
            reset_objects_state=MagicMock(),
            capture_visualization_safely=MagicMock(),
        )
        self._profiler = _ProfilerStub()
        self._task_success = torch.zeros(1, dtype=torch.bool)
        self._detached_uids_for_reset: list[str] = []
        self._elapsed_steps = torch.zeros(1, dtype=torch.int32)
        self.event_manager = MagicMock()
        self.initial_random_value = 0.0

    def is_task_success(self, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.zeros(1, dtype=torch.bool)

    def _initialize_episode(self, env_ids: torch.Tensor, **kwargs) -> None:
        del env_ids, kwargs
        self.initial_random_value = float(self.np_random.random())

    def get_obs(self, **kwargs) -> dict[str, float]:
        del kwargs
        return {"random": self.initial_random_value}

    def get_info(self, **kwargs) -> dict:
        del kwargs
        return {}


def test_reset_seed_replays_gym_rng_and_reseeds_event_manager(monkeypatch) -> None:
    """A Gym reset seed becomes the effective task and event-manager seed."""
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)

    def _set_seed(seed: int) -> int:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return seed + 1

    monkeypatch.setattr(base_env_module, "set_seed", _set_seed)
    env = _ResetEnv()

    first_obs, _ = env.reset(seed=2026)
    second_obs, _ = env.reset(seed=2026)

    assert first_obs == second_obs
    assert env.cfg.seed == 2027
    assert env.event_manager.set_seed.call_count == 2
    env.event_manager.set_seed.assert_called_with(2027)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
