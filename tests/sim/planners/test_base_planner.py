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

import pytest

from embodichain.lab.sim.planners.base_planner import BasePlanner, PlanOptions
from embodichain.lab.sim.planners.curobo.curobo_planner import CuroboPlanner
from embodichain.lab.sim.planners.utils import PlanResult, PlanState


class _PlannerWithoutCollisionAvoidance(BasePlanner):
    def plan(
        self,
        target_states: list[PlanState],
        options: PlanOptions = PlanOptions(),
    ) -> PlanResult:
        raise NotImplementedError


def test_collision_model_visualization_is_unsupported_by_default():
    planner = _PlannerWithoutCollisionAvoidance.__new__(
        _PlannerWithoutCollisionAvoidance
    )

    with pytest.raises(
        NotImplementedError, match="does not support collision avoidance"
    ):
        planner.visualize_robot_collision_models("arm")


def test_curobo_overrides_robot_collision_model_visualization():
    assert (
        CuroboPlanner.visualize_robot_collision_models
        is not BasePlanner.visualize_robot_collision_models
    )
