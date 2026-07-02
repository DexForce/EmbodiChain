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

import torch

from embodichain.lab.sim.planners.utils import PlanState, PlanResult, MoveType, MovePart


class TestPlanStateBatched:
    def test_from_qpos_batched(self):
        qpos = torch.zeros(4, 7)
        ps = PlanState.from_qpos(qpos, move_type=MoveType.JOINT_MOVE, move_part=MovePart.LEFT)
        assert ps.qpos.shape == (4, 7)
        assert ps.move_type == MoveType.JOINT_MOVE

    def test_from_xpos_batched(self):
        xpos = torch.eye(4).unsqueeze(0).repeat(3, 1, 1)
        ps = PlanState.from_xpos(xpos, move_type=MoveType.EEF_MOVE)
        assert ps.xpos.shape == (3, 4, 4)

    def test_single_ctor_unsqueezes(self):
        ps = PlanState.single(qpos=torch.zeros(7), move_type=MoveType.JOINT_MOVE)
        assert ps.qpos.shape == (1, 7)


class TestPlanResultBatched:
    def test_is_all_success_tensor(self):
        r = PlanResult(success=torch.tensor([True, True, False]))
        assert r.is_all_success() is False

    def test_is_all_success_scalar(self):
        r = PlanResult(success=True)
        assert r.is_all_success() is True

    def test_batched_shapes(self):
        r = PlanResult(
            success=torch.tensor([True, False]),
            positions=torch.zeros(2, 10, 7),
            velocities=torch.zeros(2, 10, 7),
            accelerations=torch.zeros(2, 10, 7),
            dt=torch.zeros(2, 10),
            duration=torch.tensor([1.0, 0.0]),
        )
        assert r.positions.shape == (2, 10, 7)
        assert r.dt.shape == (2, 10)
        assert r.duration.shape == (2,)
