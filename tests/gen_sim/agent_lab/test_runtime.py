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
from embodichain.gen_sim.agent_lab.runtime import Lab, LabCfg, _scene_entities


def test_scene_construction_matches_gensim_gym_order() -> None:
    physical = {
        "background": [{"uid": "table"}],
        "rigid_object": [{"uid": "can"}],
        "articulation": [{"uid": "cabinet"}, {"uid": "bell"}],
    }
    order = list(_scene_entities(physical))
    assert [item["uid"] for _, item in order] == ["table", "cabinet", "bell", "can"]
    assert order[-1][1] is physical["rigid_object"][0]


def test_config_instances_do_not_share_experiment_overrides() -> None:
    first, second = LabCfg(), LabCfg()
    first.robot_overrides["init_pos"] = [1, 2, 3]
    assert second.robot_overrides == {}


def test_profiler_counts_failed_and_successful_operations(monkeypatch) -> None:
    lab = Lab.__new__(Lab)
    lab.profile = {}
    clock = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(
        "embodichain.gen_sim.agent_lab.runtime.time.perf_counter", lambda: next(clock)
    )
    with lab._measure("physics_update"):
        pass
    with pytest.raises(ValueError), lab._measure("physics_update"):
        raise ValueError("failed update")
    assert lab.profile == {"physics_update": {"calls": 2, "wall_seconds": 0.75}}
