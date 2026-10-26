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

"""Tests for simulation-derived environment timing."""

from __future__ import annotations

import math
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.sim import SimulationManagerCfg

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("with_history", [False, True])
def test_contact_sampling_precedes_reset_without_extra_physics(
    with_history: bool,
) -> None:
    """Ordinary contacts sample once per control; reset observations use buffers."""
    world = SimpleNamespace(ready=False, steps=0)

    class Contact:
        requires_substep_update = False
        requires_control_update = True

        def __init__(self) -> None:
            self.samples = 0
            self.values = torch.zeros(2)

        def update(self, **kwargs) -> None:
            assert world.ready, "SDK contact snapshot is unavailable after reset"
            self.samples += 1
            self.values.fill_(1)

        def get_data(self):
            return {"touch": self.values}

    contact = Contact()
    history_calls = []
    sensors = {"contact": contact}
    if with_history:
        sensors["history"] = SimpleNamespace(
            requires_substep_update=True,
            requires_control_update=False,
            begin_control_step=lambda: None,
            update_physics_step=lambda dt: history_calls.append(dt),
            get_data=lambda: {"touch": torch.zeros(2)},
        )

    def step(dt, count, after_substep=None, **kwargs):
        for _ in range(count):
            world.steps += 1
            world.ready = True
            if after_substep is not None:
                after_substep(dt)

    env = SimpleNamespace(
        sensors=sensors,
        sim=SimpleNamespace(update=step, render_camera_group=lambda ids: None),
        physics_dt=0.001,
        cfg=SimpleNamespace(sim_steps_per_control=20),
        num_envs=2,
        device=torch.device("cpu"),
        _camera_group_ids=[],
        _profiler=SimpleNamespace(section=lambda *args: nullcontext()),
    )
    BaseEnv._get_sensor_obs(env)
    assert contact.samples == 0
    BaseEnv._advance_physics(env)
    assert world.steps == 20 and contact.samples == 1
    assert len(history_calls) == (20 if with_history else 0)
    world.ready = False
    contact.values[0] = 0  # selective reset invalidates the SDK snapshot
    obs = BaseEnv._get_sensor_obs(env)
    torch.testing.assert_close(obs["contact"]["touch"], torch.tensor([0.0, 1.0]))
    assert contact.samples == 1 and world.steps == 20


def _configure_timing(
    physics_dt: float,
    sim_steps_per_control: int | float = 4,
    target_control_frequency: float | None = None,
) -> BaseEnv:
    """Build only the timing portion of a BaseEnv without starting simulation."""
    env = BaseEnv.__new__(BaseEnv)
    env.cfg = EnvCfg(
        sim_cfg=SimulationManagerCfg(physics_dt=physics_dt),
        sim_steps_per_control=sim_steps_per_control,
        target_control_frequency=target_control_frequency,
    )
    env.sim_cfg = env.cfg.sim_cfg
    env._configure_timing()
    return env


def test_timing_is_derived_without_integer_truncation() -> None:
    """All public timing values preserve the actual simulated cadence."""
    env = _configure_timing(physics_dt=0.006, sim_steps_per_control=7)

    assert env.physics_dt == pytest.approx(0.006)
    assert env.step_dt == pytest.approx(0.042)
    assert env.physics_frequency == pytest.approx(1.0 / 0.006)
    assert env.control_frequency == pytest.approx(1.0 / 0.042)
    assert env.sim_freq == pytest.approx(env.physics_frequency)
    assert env.control_freq == pytest.approx(env.control_frequency)
    assert env.metadata["render_fps"] == pytest.approx(env.control_frequency)
    assert "render_fps" not in BaseEnv.metadata


def test_exact_target_frequency_resolves_integer_sim_steps() -> None:
    """An exactly representable requested rate resolves the decimation value."""
    env = _configure_timing(
        physics_dt=0.01,
        sim_steps_per_control=2,
        target_control_frequency=20.0,
    )

    assert env.cfg.sim_steps_per_control == 5
    assert env.step_dt == pytest.approx(0.05)
    assert env.control_frequency == pytest.approx(20.0)


def test_unrepresentable_target_frequency_is_rejected() -> None:
    """A requested rate cannot silently change physics dt or be approximated."""
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        _configure_timing(
            physics_dt=0.01,
            target_control_frequency=30.0,
        )


@pytest.mark.parametrize("physics_dt", [0.0, -0.01, math.inf, math.nan])
def test_invalid_physics_dt_is_rejected(physics_dt: float) -> None:
    """Physics timing must be finite and positive."""
    with pytest.raises(ValueError, match="physics_dt must be"):
        _configure_timing(physics_dt=physics_dt)


@pytest.mark.parametrize("sim_steps_per_control", [0, -1, 1.5, True])
def test_invalid_sim_steps_per_control_is_rejected(
    sim_steps_per_control: int | float,
) -> None:
    """Simulation decimation must be a positive integer."""
    with pytest.raises(ValueError, match="sim_steps_per_control must be"):
        _configure_timing(
            physics_dt=0.01,
            sim_steps_per_control=sim_steps_per_control,
        )


@pytest.mark.parametrize("target_frequency", [0.0, -10.0, math.inf, math.nan, True])
def test_invalid_target_frequency_is_rejected(target_frequency: float) -> None:
    """Requested control timing must be finite and positive."""
    with pytest.raises(ValueError, match="target_control_frequency must be"):
        _configure_timing(
            physics_dt=0.01,
            target_control_frequency=target_frequency,
        )
