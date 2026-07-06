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

"""Tests for ActionResult.success tensor and ActionCfg.motion_source/planner_type."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.core import ActionCfg, ActionResult, WorldState


class TestActionResultSuccess:
    def test_success_all_tensor(self):
        r = ActionResult(
            success=torch.tensor([True, False]),
            trajectory=torch.zeros(2, 0, 3),
            next_state=WorldState(last_qpos=torch.zeros(2, 3)),
        )
        assert r.success_all is False

    def test_bool_deprecation(self):
        r = ActionResult(
            success=torch.tensor([True, True]),
            trajectory=torch.zeros(2, 0, 3),
            next_state=WorldState(last_qpos=torch.zeros(2, 3)),
        )
        with pytest.warns(DeprecationWarning):
            assert bool(r) is True


class TestActionCfgMotionSource:
    def test_defaults(self):
        cfg = ActionCfg()
        assert cfg.motion_source == "ik_interp"
        assert cfg.planner_type is None
