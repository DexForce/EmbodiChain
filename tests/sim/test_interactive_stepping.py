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

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize(
    "module_name",
    [
        "examples.sim.gizmo.gizmo_robot",
        "examples.sim.gizmo.gizmo_w1",
        "examples.sim.gizmo.gizmo_scene",
        "examples.sim.gizmo.gizmo_camera",
        "scripts.tutorials.sim.gizmo_robot",
    ],
)
@pytest.mark.parametrize("step_cost", [0.003, 0.02])
def test_gizmo_loops_step_before_pacing_and_close(
    module_name: str, step_cost: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module(module_name)
    physics_dt = 0.01
    loop_count = 3
    elapsed = 0.0
    events: list[str] = []
    sleep_durations: list[float] = []

    def update(*, step: int) -> None:
        nonlocal elapsed
        assert step == 1
        events.append("physics")
        elapsed += step_cost

    def sleep(duration: float) -> None:
        nonlocal elapsed
        events.append("pace")
        sleep_durations.append(duration)
        elapsed += duration
        if len(sleep_durations) == loop_count:
            raise KeyboardInterrupt

    sim = SimpleNamespace(
        sim_config=SimpleNamespace(physics_dt=physics_dt),
        num_envs=1,
        update=update,
        get_sensor=lambda _uid: None,
        has_gizmo=lambda _uid: False,
        destroy=Mock(),
    )
    monkeypatch.setattr(
        module,
        "time",
        SimpleNamespace(
            time=lambda: elapsed, perf_counter=lambda: elapsed, sleep=sleep
        ),
    )
    monkeypatch.setattr(module, "logger", Mock())

    if module_name.endswith("gizmo_camera"):
        module.run_simulation(sim, Mock(), show_camera_window=False)
    elif module_name.endswith("gizmo_scene"):
        module.run_simulation(sim, show_camera_window=False)
    else:
        module.run_simulation(sim)

    assert events == ["physics", "pace"] * loop_count
    assert sleep_durations == pytest.approx(
        [max(0.0, physics_dt - step_cost)] * loop_count
    )
    sim.destroy.assert_called_once_with()
