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
from embodichain.lab.gym.envs import BaseEnv


@pytest.mark.parametrize("enabled", (False, True))
def test_standard_step_schedules_sensor_after_each_physics_substep(enabled):
    events = []
    env = BaseEnv.__new__(BaseEnv)
    env.cfg = SimpleNamespace(sim_steps_per_control=3)
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)
    env.sim = SimpleNamespace(update=lambda dt, n: events.append(("physics", dt, n)))
    sensor = SimpleNamespace(
        requires_substep_update=enabled,
        begin_control_step=lambda: events.append("begin"),
        update_physics_step=lambda dt: events.append(("sample", dt)),
    )
    env.sensors = {"contact": sensor}
    env._advance_physics()
    expected = (
        ["begin"] + [("physics", 0.01, 1), ("sample", 0.01)] * 3
        if enabled
        else [("physics", 0.01, 3)]
    )
    assert events == expected
