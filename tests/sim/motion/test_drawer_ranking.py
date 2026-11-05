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
"""Endpoint screening preserves seed order and bounded complete-plan fallback."""

from __future__ import annotations

import pytest
import torch
from embodichain.lab.sim.motion.drawer_ranking import prefilter_drawer_entry_paths


@pytest.mark.parametrize("collisions", [False, True])
def test_ranking_keeps_sequential_seed_and_kinematic_priority(collisions):
    candidates = []
    for value in (3.0, 1.0, 2.0):
        poses = torch.eye(4).repeat(1, 3, 1, 1)
        poses[:, :, 0, 3] = value
        poses[:, 1, 1, 3] = 1
        candidates.append(poses)
    calls = []

    def solve(pose, seed):
        value = float(pose[0, 0, 3])
        calls.append((value, float(seed[0, 0])))
        ok = not (value == 2.0 and pose[0, 1, 3] == 1)
        return torch.tensor([ok]), torch.tensor([[value]])

    def collision(stage, qpos):
        return [{"stage": stage, "valid": float(qpos[0, 0]) != 1.0}]

    selected, records = prefilter_drawer_entry_paths(
        tuple(candidates),
        torch.zeros(1, 1),
        solve,
        endpoint_collision=collision if collisions else None,
    )
    assert selected == ((0, 1) if collisions else (1, 0))
    assert calls == [
        (3.0, 0.0),
        (3.0, 3.0),
        (3.0, 3.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (1.0, 1.0),
        (2.0, 0.0),
        (2.0, 2.0),
    ]
    assert records[2]["solved_stages"] == 1
    assert records[2]["failure_stage"] == "rim"
    assert records[2]["failure_reason"] == "ik_failed"


def test_no_survivor_still_selects_one_full_plan_candidate():
    poses = torch.eye(4).repeat(1, 3, 1, 1)
    selected, records = prefilter_drawer_entry_paths(
        (poses, poses.clone()),
        torch.zeros(1, 2),
        lambda pose, seed: (torch.tensor([False]), seed),
    )
    assert selected == (0,)
    assert not any(record["accepted"] for record in records)
