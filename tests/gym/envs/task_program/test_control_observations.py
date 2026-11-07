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


"""Completed control facts retain exact physical and compiled occurrence identity."""

from __future__ import annotations

from copy import deepcopy
import pytest
from .test_simulation_policies import _raw_press_observation


def test_completed_control_collector_copies_owned_facts_without_live_acquisition():
    system, collector, _, _, selectors, raw = _raw_press_observation()
    assert raw["episode_ids"] == [0]
    assert raw["control_identity"]["episode_id"] == "0"
    assert len(raw["policies"]) == 3
    assert raw["window"]["records"]
    assert raw["boundary_frames"] == [system.clock.step_index]
    first = deepcopy(raw)
    raw["control_identity"]["contact_actors"]["hand"].clear()
    raw["window"]["records"].clear()
    assert collector.capture(selectors, terminal_policy=selectors[-1]) == first
    # The helper already closed native hooks and forbids live-later qpos reads.
    assert not system.hooks


@pytest.mark.parametrize("damage", ["phase", "source_path", "index"])
def test_control_collector_rejects_foreign_compiled_selector(damage):
    _, collector, _, _, selectors, _ = _raw_press_observation()
    bad = deepcopy(selectors[-1])
    if damage == "phase":
        bad["phase"] = "ready"
    elif damage == "source_path":
        bad["source_path"] = ["program", "items", 0]
    else:
        bad["policy_index"] = -1
    with pytest.raises(ValueError):
        collector.capture(selectors, terminal_policy=bad)
