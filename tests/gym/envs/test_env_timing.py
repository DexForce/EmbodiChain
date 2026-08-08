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

import pytest

from embodichain.lab.gym.envs import BaseEnv, EnvCfg
from embodichain.lab.sim import SimulationManagerCfg

pytestmark = pytest.mark.no_sim


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
    assert env.physics_frequency_hz == pytest.approx(1.0 / 0.006)
    assert env.control_frequency_hz == pytest.approx(1.0 / 0.042)
    assert env.sim_freq == pytest.approx(env.physics_frequency_hz)
    assert env.control_freq == pytest.approx(env.control_frequency_hz)
    assert env.metadata["render_fps"] == pytest.approx(env.control_frequency_hz)
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
    assert env.control_frequency_hz == pytest.approx(20.0)


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
