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

"""Tests for deterministic event-functor randomization streams."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.gym.envs.managers import EventCfg, Functor
from embodichain.lab.gym.envs.managers.event_manager import EventManager


class _StubEnv:
    """Minimal environment surface consumed by :class:`EventManager`."""

    def __init__(self, seed: int | None, device: torch.device | str = "cpu") -> None:
        self.cfg = SimpleNamespace(seed=seed)
        self.num_envs = 2
        self.device = torch.device(device)
        self.sim = SimpleNamespace()
        self._profiler = None
        self.samples: dict[str, list[tuple[float, float, list[float]]]] = {}
        self.initialization_samples: list[tuple[float, float, float]] = []
        self.reset_samples: list[tuple[float, float, float]] = []


def _mixed_rng_event(
    env: _StubEnv,
    env_ids: torch.Tensor | None,
    *,
    label: str,
) -> None:
    """Record values from every RNG family used by built-in event functors."""
    del env_ids
    env.samples.setdefault(label, []).append(
        (
            random.random(),
            float(np.random.random()),
            torch.rand(3, device=env.device).cpu().tolist(),
        )
    )


class _RandomClassFunctor(Functor):
    """Class-style event whose constructor and call both consume randomness."""

    def __init__(self, cfg: EventCfg, env: _StubEnv) -> None:
        super().__init__(cfg, env)
        env.initialization_samples.append(
            (random.random(), float(np.random.random()), torch.rand(()).item())
        )

    def __call__(
        self,
        env: _StubEnv,
        env_ids: torch.Tensor | None,
        *,
        label: str,
    ) -> None:
        _mixed_rng_event(env, env_ids, label=label)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        self._env.reset_samples.append(
            (random.random(), float(np.random.random()), torch.rand(()).item())
        )


def _manager(
    seed: int | None,
    *,
    include_noise: bool = False,
    class_style: bool = False,
    device: torch.device | str = "cpu",
) -> tuple[_StubEnv, EventManager]:
    env = _StubEnv(seed, device=device)
    target_func = _RandomClassFunctor if class_style else _mixed_rng_event
    cfg: dict[str, EventCfg] = {}
    if include_noise:
        cfg["noise"] = EventCfg(
            func=_mixed_rng_event,
            mode="reset",
            params={"label": "noise"},
        )
    cfg["target"] = EventCfg(
        func=target_func,
        mode="reset",
        params={"label": "target"},
    )
    return env, EventManager(cfg, env)


def test_same_seed_replays_all_event_rng_families() -> None:
    """Python, NumPy, and Torch samples replay for an identical event schedule."""
    env_a, manager_a = _manager(1234)
    env_b, manager_b = _manager(1234)
    env_ids = torch.tensor([0, 1])

    for _ in range(2):
        manager_a.apply("reset", env_ids)
        manager_b.apply("reset", env_ids)

    assert env_a.samples["target"] == env_b.samples["target"]


def test_different_seed_changes_event_randomization() -> None:
    """Changing the task seed changes its domain-randomization sequence."""
    env_a, manager_a = _manager(1234)
    env_b, manager_b = _manager(1235)

    manager_a.apply("reset", torch.tensor([0, 1]))
    manager_b.apply("reset", torch.tensor([0, 1]))

    assert env_a.samples["target"] != env_b.samples["target"]


def test_event_stream_is_independent_of_global_rng_and_other_functors() -> None:
    """Policy-side draws and unrelated events do not perturb a named event stream."""
    env_a, manager_a = _manager(41)
    env_b, manager_b = _manager(41, include_noise=True)

    manager_a.apply("reset", torch.tensor([0, 1]))
    for _ in range(20):
        random.random()
        np.random.random()
        torch.rand(8)
    manager_b.apply("reset", torch.tensor([0, 1]))

    assert env_a.samples["target"] == env_b.samples["target"]


def test_event_call_restores_process_rng_states() -> None:
    """Event randomization does not consume the caller's process RNG streams."""
    _, manager = _manager(99)

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), float(np.random.random()), torch.rand(3))

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    manager.apply("reset", torch.tensor([0, 1]))
    actual = (random.random(), float(np.random.random()), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_reseed_rewinds_event_and_interval_sequences() -> None:
    """Reseeding rewinds invocation streams and interval scheduling."""
    env = _StubEnv(55)
    manager = EventManager(
        {
            "target": EventCfg(
                func=_mixed_rng_event,
                mode="interval",
                interval_step=2,
                is_global=True,
                params={"label": "target"},
            )
        },
        env,
    )

    manager.apply("interval")
    assert "target" not in env.samples
    manager.apply("interval")
    first = env.samples["target"][0]

    manager.set_seed(55)
    manager.apply("interval")
    assert len(env.samples["target"]) == 1
    manager.apply("interval")

    assert env.samples["target"][1] == first


def test_class_functor_initialization_and_calls_are_seeded() -> None:
    """Class-style functors use deterministic streams during init and invocation."""
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    env_a, manager_a = _manager(808, class_style=True)

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    env_b, manager_b = _manager(808, class_style=True)

    manager_a.apply("reset", torch.tensor([0, 1]))
    manager_b.apply("reset", torch.tensor([0, 1]))

    manager_a.reset(torch.tensor([0, 1]))
    random.random()
    np.random.random()
    torch.rand(10)
    manager_b.reset(torch.tensor([0, 1]))

    assert env_a.initialization_samples == env_b.initialization_samples
    assert env_a.samples["target"] == env_b.samples["target"]
    assert env_a.reset_samples == env_b.reset_samples


def test_string_class_functor_is_tracked_for_seeded_reset() -> None:
    """String-referenced class functors retain deterministic reset handling."""
    env = _StubEnv(808)
    manager = EventManager(
        {
            "target": EventCfg(
                func="embodichain.lab.gym.envs.managers.events:prepare_extra_attr",
                mode="reset",
                params={"attrs": []},
            )
        },
        env,
    )

    assert len(manager._mode_class_functor_cfgs["reset"]) == 1


@pytest.mark.gpu
def test_cuda_event_stream_replays_and_restores_device_rng() -> None:
    """The simulation-device generator is scoped and restored on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    device = torch.device("cuda:0")
    env_a, manager_a = _manager(314, device=device)
    env_b, manager_b = _manager(314, device=device)

    torch.cuda.manual_seed(17)
    expected_after_event = torch.rand(3, device=device)
    torch.cuda.manual_seed(17)
    manager_a.apply("reset", torch.tensor([0, 1], device=device))
    actual_after_event = torch.rand(3, device=device)
    manager_b.apply("reset", torch.tensor([0, 1], device=device))

    torch.testing.assert_close(actual_after_event, expected_after_event)
    assert env_a.samples["target"] == env_b.samples["target"]
