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

from embodichain.gen_sim.task_engine._task_program import _search_cost as costs


def test_nested_cost_scopes_share_ledger_but_independent_calls_do_not() -> None:
    with costs.search_cost_scope() as outer:
        costs.record_search("grasp_candidate", count=2)
        with costs.search_cost_scope() as inner:
            assert inner is outer
            costs.record_search("grasp_candidate")
        assert outer.snapshot()["counters"] == {"grasp_candidate": 3}
    with costs.search_cost_scope() as following:
        assert following is not outer
        assert following.snapshot()["counters"] == {}


def test_measurements_are_inclusive_and_snapshots_are_independent(monkeypatch) -> None:
    # Nested intervals are 3 s and 1 s; their sum is not the total wall time.
    monkeypatch.setattr(costs, "perf_counter", iter((0.0, 1.0, 2.0, 3.0)).__next__)
    with costs.search_cost_scope() as cost:
        with costs.measure_search("pick_preview"):
            with costs.measure_search("place_preview"):
                pass
        snapshot = cost.snapshot()
        assert snapshot["inclusive_wall_seconds"] == {
            "place_preview": 1.0,
            "pick_preview": 3.0,
        }
        assert snapshot["timings_overlap"] is True
        snapshot["counters"]["pick_preview"] = 100
        assert cost.snapshot()["counters"]["pick_preview"] == 1


def test_exception_records_cost_and_restores_scope() -> None:
    with pytest.raises(ValueError, match="failed"):
        with costs.search_cost_scope() as cost:
            with costs.measure_search("place_preview"):
                raise ValueError("failed")
    assert cost.snapshot()["counters"] == {"place_preview": 1}
    assert costs._COST.get() is None


def test_absent_or_independent_context_does_not_touch_active_ledger() -> None:
    costs.record_search("ignored")
    with costs.measure_search("ignored"):
        pass
    with costs.search_cost_scope() as cost:
        Context().run(costs.record_search, "other_context")
        assert cost.snapshot()["counters"] == {}
