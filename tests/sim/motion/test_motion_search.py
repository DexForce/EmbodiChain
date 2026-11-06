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

from contextvars import Context

import pytest

from embodichain.lab.sim.motion import search


def test_search_ledger_nests_isolates_and_records_failed_operations(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(search, "perf_counter", lambda: clock[0])
    assert search.current_search_cost() is None
    with search.search_cost_scope() as cost:
        assert Context().run(search.current_search_cost) is None
        with search.search_cost_scope(wall_seconds=0.1) as nested:
            assert nested is cost and nested.wall_seconds is None
            with pytest.raises(RuntimeError, match="solver failed"):
                with search.measure_search("solve"):
                    clock[0] = 0.25
                    search.record_search("seeds", count=4)
                    raise RuntimeError("solver failed")
        assert cost.counters == {"solve": 1, "seeds": 4}
        assert cost.inclusive_wall_seconds == {"solve": 0.25}
        snapshot = cost.snapshot()
        snapshot["counters"]["seeds"] = 100
        assert cost.counters["seeds"] == 4
    assert search.current_search_cost() is None


def test_search_deadline_stops_only_at_cooperative_boundaries(monkeypatch):
    clock = [5.0]
    monkeypatch.setattr(search, "perf_counter", lambda: clock[0])
    with pytest.raises(search.PlanningBudgetExceeded) as failure:
        with search.search_cost_scope(wall_seconds=1.0) as cost:
            search.record_search("started")
            clock[0] = 6.0
            search.record_search(
                "additional_measurement", count=4, budget_checkpoint=False
            )
            search.record_search("next")
    assert failure.value.operation == "next"
    assert cost.counters == {"started": 1, "additional_measurement": 4}
    assert search.current_search_cost() is None
    with pytest.raises(ValueError, match="finite and positive"):
        with search.search_cost_scope(wall_seconds=0):
            pass


def test_additional_timing_retains_cost_without_adding_a_deadline_boundary(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(search, "perf_counter", lambda: clock[0])
    with pytest.raises(search.PlanningBudgetExceeded) as failure:
        with search.search_cost_scope(wall_seconds=1.0) as cost:
            with search.measure_search(
                "additional_measurement", budget_checkpoint=False
            ):
                clock[0] = 1.25
            # Existing solver work remains the cooperative stop boundary.
            search.record_search("ik_solve")
    assert failure.value.operation == "ik_solve"
    assert cost.counters == {"additional_measurement": 1}
    assert cost.inclusive_wall_seconds == {"additional_measurement": 1.25}
    assert search.current_search_cost() is None
