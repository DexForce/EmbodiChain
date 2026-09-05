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
from unittest.mock import Mock, call

import pytest

from scripts.tutorials.sim import gizmo_robot

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("work_duration", [0.002, 0.050])
def test_gizmo_loop_advances_manual_physics_and_paces_frames(
    monkeypatch: pytest.MonkeyPatch,
    native: bool,
    work_duration: float,
) -> None:
    """Apply IK before stepping, avoid duplicate updates, and clean up on exit."""
    physics_dt = 0.01
    calls = Mock()
    sim = calls.sim
    sim.sim_config = SimpleNamespace(physics_dt=physics_dt)
    native_control = (calls.ik, object()) if native else None
    clock = Mock(side_effect=[0.0, 0.0, work_duration])
    sleep = Mock(side_effect=KeyboardInterrupt)
    monkeypatch.setattr(gizmo_robot.time, "perf_counter", clock)
    monkeypatch.setattr(gizmo_robot.time, "sleep", sleep)

    gizmo_robot.run_simulation(sim, native_control)

    expected_calls = [call.ik.update()] if native else []
    expected_calls.extend([call.sim.update(step=1), call.sim.destroy()])
    assert calls.mock_calls == expected_calls
    sleep.assert_called_once_with(pytest.approx(max(0.0, physics_dt - work_duration)))
