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

import numpy as np
import pytest

from embodichain.lab.sim.planners.toppra_planner import _toppra_solve_one_env
from embodichain.lab.sim.planners.utils import TrajectorySampleMethod


class TestToppraWorker:
    def test_solve_one_env_quantity(self):
        # 2-waypoint, 6-DOF
        wp = np.array([[0.0] * 6, [0.5] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=20,
        )
        assert out["success"] is True
        assert out["positions"].shape == (20, 6)
        assert out["velocities"].shape == (20, 6)
        assert out["dt"].shape == (20,)

    def test_solve_one_env_infeasible(self):
        # Single waypoint -> SplineInterpolator raises -> caught, returns failure
        wp = np.array([[0.0] * 6])
        out = _toppra_solve_one_env(
            waypoints=wp,
            vel_constraint=1.0,
            acc_constraint=2.0,
            sample_method=TrajectorySampleMethod.QUANTITY,
            sample_interval=10,
        )
        assert out["success"] is False
