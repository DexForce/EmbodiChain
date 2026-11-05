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

from types import SimpleNamespace
import pytest
from embodichain.lab.gym.envs import BaseEnv, EmbodiedEnv


@pytest.mark.parametrize("env_type", (BaseEnv, EmbodiedEnv))
@pytest.mark.parametrize("enabled", (False, True))
def test_standard_step_schedules_sensor_after_each_physics_substep(env_type, enabled):
    events = []
    env = env_type.__new__(env_type)
    env.cfg = SimpleNamespace(sim_steps_per_control=3)
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)

    def update(dt, n, *, after_substep=None, render_final_step=True):
        assert render_final_step is False
        events.append(("manager", dt, n))
        for _ in range(n):
            events.append(("physics", dt))
            if after_substep is not None:
                after_substep(dt)

    env.sim = SimpleNamespace(update=update)
    sensor = SimpleNamespace(
        requires_substep_update=enabled,
        begin_control_step=lambda: events.append("begin"),
        update_physics_step=lambda dt: events.append(("sample", dt)),
    )
    env.sensors = {"contact": sensor}
    env._advance_physics()
    expected = (
        ["begin", ("manager", 0.01, 3)] + [("physics", 0.01), ("sample", 0.01)] * 3
        if enabled
        else [("manager", 0.01, 3)] + [("physics", 0.01)] * 3
    )
    assert events == expected


@pytest.mark.parametrize(
    "env_type,controller_ready,substep_actions,refreshes",
    [
        (EmbodiedEnv, False, True, True),
        (EmbodiedEnv, True, True, False),
        (EmbodiedEnv, False, False, False),
        (BaseEnv, False, True, False),
    ],
)
def test_substep_action_updates_interleave_with_physics_and_sensor_observation(
    env_type,
    controller_ready,
    substep_actions,
    refreshes,
):
    """Only opted-in EmbodiedEnv policy controllers refresh between steps."""
    calls = []
    updates = []
    env = env_type.__new__(env_type)
    env.cfg = SimpleNamespace(sim_steps_per_control=3)
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)
    env._step_uses_action_manager = not controller_ready
    env.action_manager = SimpleNamespace(
        requires_substep_update=substep_actions,
        apply_substep_actions=lambda: calls.append("control"),
    )

    def update(dt, n, *, after_substep=None, render_final_step=True):
        assert render_final_step is False
        updates.append(n)
        for _ in range(n):
            calls.append("physics")
            if after_substep is not None:
                after_substep(dt)

    env.sim = SimpleNamespace(update=update)
    env.sensors = {
        "contact": SimpleNamespace(
            requires_substep_update=True,
            begin_control_step=lambda: None,
            update_physics_step=lambda dt: calls.append("sensor"),
        )
    }
    env._advance_physics()
    expected = (
        ["physics", "sensor"] * 3
        if not refreshes
        else [
            "physics",
            "sensor",
            "control",
            "physics",
            "sensor",
            "control",
            "physics",
            "sensor",
        ]
    )
    assert calls == expected
    assert updates == ([1, 1, 1] if refreshes else [3])
