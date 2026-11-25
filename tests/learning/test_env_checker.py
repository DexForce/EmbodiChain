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

import random
from copy import deepcopy

from gymnasium import spaces
import numpy as np
import pytest
import torch
from tensordict import TensorDict

from embodichain.learning.rl import check_learning_env


class _Environment:
    def __init__(self, *, observation="tensor", terminal="full", failure=None):
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.single_action_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.single_observation_space = spaces.Box(
            -np.inf, np.inf, (2,), dtype=np.float32
        )
        if observation != "tensor":
            self.single_observation_space = spaces.Dict(
                {"state": spaces.Dict({"position": self.single_observation_space})}
            )
        self.observation_kind = observation
        self.terminal_kind = terminal
        self.failure = failure
        self.closed = False
        self.actions = []
        self._draw()

    @staticmethod
    def _draw():
        random.random()
        np.random.random()
        torch.rand(())

    def _observation(self, count=2):
        observation = torch.zeros(count, 2)
        if self.observation_kind == "tensor":
            return observation
        observation = {"state": {"position": observation}}
        return (
            TensorDict(observation, batch_size=[count])
            if self.observation_kind == "tensordict"
            else observation
        )

    def reset(self, *, seed=None):
        self.seed = seed
        self._draw()
        if self.failure == "reset":
            raise RuntimeError("reset failure")
        return self._observation(), {}

    def step(self, action):
        self._draw()
        if self.failure == "step":
            raise RuntimeError("step failure")
        assert action.shape == (2, 1)
        assert action.dtype == torch.float32
        assert bool((action.abs() <= 1).all())
        self.actions.append(action.clone())
        info = {}
        if self.terminal_kind != "missing":
            compact = self.terminal_kind == "compact"
            info["terminal_observation"] = self._observation(1 if compact else 2)
            info["terminal_observation_mask"] = torch.tensor([False, True])
            if compact:
                info["terminal_env_ids"] = torch.tensor([1])
        return (
            self._observation(),
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            torch.tensor([False, True]),
            info,
        )

    def close(self):
        self.closed = True
        self._draw()
        if self.failure == "close":
            raise RuntimeError("close failure")


@pytest.mark.parametrize(
    ("observation", "terminal"),
    [
        ("tensor", "full"),
        ("mapping", "compact"),
        ("tensordict", "full"),
        ("tensor", "missing"),
    ],
)
def test_checks_disposable_environment_and_reproducible_actions(observation, terminal):
    instances = []

    def factory():
        env = _Environment(observation=observation, terminal=terminal)
        if terminal == "missing":
            env.device = torch.device("cpu:0")
        instances.append(env)
        return env

    check_learning_env(factory, num_steps=2, seed=17)
    check_learning_env(factory, num_steps=2, seed=17)

    assert len(instances) == 2 and instances[0] is not instances[1]
    assert all(env.closed and env.seed == 17 for env in instances)
    assert all(len(env.actions) == 2 for env in instances)
    for first, second in zip(instances[0].actions, instances[1].actions):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.parametrize("failure", [None, "factory", "reset", "step", "close"])
def test_restores_rng_on_success_and_failure_without_initializing_accelerators(
    monkeypatch, failure
):
    def unexpected_cuda_access(*args, **kwargs):
        pytest.fail("CPU environment checking must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    for name in (
        "get_rng_state_all",
        "set_rng_state_all",
        "device_count",
        "manual_seed_all",
    ):
        monkeypatch.setattr(torch.cuda, name, unexpected_cuda_access)
    instances = []

    def factory():
        _Environment._draw()
        if failure == "factory":
            raise RuntimeError("factory failure")
        env = _Environment(failure=failure)
        instances.append(env)
        return env

    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    if failure is None:
        check_learning_env(factory)
    else:
        with pytest.raises(RuntimeError, match=f"{failure} failure"):
            check_learning_env(factory)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert all(env.closed for env in instances)


def test_preserves_preinitialized_accelerator_rng_on_failure(monkeypatch):
    states = [torch.tensor([17], dtype=torch.uint8)]
    seeds = []
    restored = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: states)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", seeds.append)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored.append)

    with pytest.raises(RuntimeError, match="step failure"):
        check_learning_env(lambda: _Environment(failure="step"), seed=29)

    assert seeds == [29]
    assert restored == [states]


@pytest.mark.parametrize(
    ("field", "replacement", "error", "message"),
    [
        (
            "observation",
            lambda: torch.zeros(2, 3),
            ValueError,
            "step.observation.*shape",
        ),
        (
            "observation",
            lambda: torch.zeros(2, 2, dtype=torch.float64),
            TypeError,
            "step.observation.*dtype",
        ),
        (
            "observation",
            lambda: torch.empty(2, 2, device="meta"),
            ValueError,
            "step.observation.*cpu",
        ),
        ("reward", lambda: torch.zeros(2, 1), ValueError, "reward.*shape"),
        ("terminated", lambda: torch.zeros(2), TypeError, "terminated.*dtype"),
        (
            "truncated",
            lambda: torch.zeros(1, dtype=torch.bool),
            ValueError,
            "truncated.*shape",
        ),
        (
            "terminal_observation",
            lambda: torch.zeros(1, 2),
            ValueError,
            "terminal_observation.*shape",
        ),
        (
            "terminal_observation_mask",
            lambda: torch.ones(2),
            TypeError,
            "terminal_observation_mask.*dtype",
        ),
        (
            "terminal_env_ids",
            lambda: torch.tensor([2]),
            ValueError,
            "terminal_env_ids.*valid",
        ),
        (
            "terminal_env_ids",
            lambda: torch.tensor([1, 1]),
            ValueError,
            "terminal_env_ids.*distinct",
        ),
    ],
)
def test_rejects_malformed_sample_and_closes_environment(
    field, replacement, error, message
):
    env = _Environment()
    original_step = env.step

    def step(action):
        result = list(original_step(action))
        fields = {"observation": 0, "reward": 1, "terminated": 2, "truncated": 3}
        if field in fields:
            result[fields[field]] = replacement()
        else:
            result[4][field] = replacement()
        return tuple(result)

    env.step = step
    with pytest.raises(error, match=message):
        check_learning_env(lambda: env)
    assert env.closed


def test_cleanup_failure_preserves_original_check_error():
    env = _Environment(failure="close")
    env.single_observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    with pytest.raises(ValueError, match="reset.observation.*shape") as caught:
        check_learning_env(lambda: env)

    assert env.closed
    assert "close failure" in caught.value.__notes__[0]


def test_checks_registered_point_mass_with_timeouts():
    from embodichain.extensions.tasks.classic_control.point_mass import PointMassEnv

    check_learning_env(
        lambda: PointMassEnv(num_envs=2, max_episode_steps=1), num_steps=2
    )


def test_action_sampling_preserves_shared_space_random_state():
    shared_space = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32, seed=42)
    original_state = deepcopy(shared_space.np_random.bit_generator.state)

    def factory():
        env = _Environment()
        env.single_action_space = shared_space
        return env

    check_learning_env(factory, seed=17)

    assert shared_space.np_random.bit_generator.state == original_state
