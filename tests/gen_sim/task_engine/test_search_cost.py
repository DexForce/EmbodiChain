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


def test_nested_planning_cannot_replenish_the_parent_wall_budget(monkeypatch) -> None:
    now = [0.0]
    monkeypatch.setattr(costs, "perf_counter", lambda: now[0])
    with costs.search_cost_scope(wall_seconds=2.0) as outer:
        costs.record_search("initial_ik")
        now[0] = 1.5
        with costs.search_cost_scope(wall_seconds=1000.0) as inner:
            assert inner is outer
            now[0] = 2.0
            with pytest.raises(costs.PlanningBudgetExceeded, match="next_ik"):
                costs.record_search("next_ik")
        assert outer.snapshot()["counters"] == {"initial_ik": 1}
        assert outer.snapshot()["wall_budget_seconds"] == 2.0


def test_expired_measurement_fails_before_the_operation_and_restores_scope(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(costs, "perf_counter", lambda: now[0])
    with pytest.raises(costs.PlanningBudgetExceeded):
        with costs.search_cost_scope(wall_seconds=2.0):
            now[0] = 3.0
            with costs.measure_search("place_preview"):
                pytest.fail("Expired planning operation ran")
    assert costs._COST.get() is None
    with costs.search_cost_scope() as following:
        assert following.snapshot()["budget_enforced"] is False


def test_native_operation_overrun_is_rejected_before_returning_a_plan(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(costs, "perf_counter", lambda: now[0])
    with pytest.raises(costs.PlanningBudgetExceeded):
        with costs.search_cost_scope(wall_seconds=2.0) as cost:
            with costs.measure_search("opaque_ik"):
                now[0] = 3.0
    assert cost.snapshot()["inclusive_wall_seconds"] == {"opaque_ik": 3.0}
    assert (
        cost.snapshot()["budget_mode"]
        == "cooperative_operation_boundaries_not_native_preemption"
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_planning_wall_budget_is_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        with costs.search_cost_scope(wall_seconds=value):
            pass


@pytest.mark.parametrize("omit_context", [False, True])
def test_nested_preview_budget_failure_returns_one_nonretryable_outer_failure(
    monkeypatch, omit_context
):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from embodichain.gen_sim.task_engine._task_program.invocation_policy import (
        GenSimActionEngine,
    )
    from embodichain.lab.sim.atomic_actions import PlannerDiagnostics
    from embodichain.lab.sim.atomic_actions.engine import AtomicActionEngine

    @dataclass(frozen=True)
    class Plan:
        diagnostics: PlannerDiagnostics

    now = [0.0]
    monkeypatch.setattr(costs, "perf_counter", lambda: now[0])
    failures = []

    def failed(request, context, **kwargs):
        failures.append((request.invocation_id, kwargs))
        return Plan(PlannerDiagnostics(backend="test"))

    preview = SimpleNamespace(route=SimpleNamespace(place_id="place-1"))
    fake = SimpleNamespace(
        actions={
            "pick_up": SimpleNamespace(
                continuation_previews={"pick-1": preview}, failed_plan=failed
            ),
            "place": SimpleNamespace(failed_plan=failed),
        }
    )
    context = SimpleNamespace(batch_size=1)
    captured = []
    fake.initial_context = lambda: captured.append(context) or context

    def implementation(request, context, **_):
        if request.skill_id == "pick_up":
            now[0] = 40.0
            return GenSimActionEngine._plan_request(
                fake,
                SimpleNamespace(skill_id="place", invocation_id="place-1"),
                context,
            )
        now[0] = 89.0
        costs.record_search("within_extended_budget")
        now[0] = 90.0
        costs.record_search("next_preview_ik")
        pytest.fail("Expired child planning returned")

    fake._plan_request_impl = implementation
    fake._plan_request = (
        lambda request, current=None, **kwargs: GenSimActionEngine._plan_request(
            fake, request, current, **kwargs
        )
    )
    result = AtomicActionEngine.plan_request(
        fake,
        SimpleNamespace(skill_id="pick_up", invocation_id="pick-1"),
        None if omit_context else context,
    )
    assert len(failures) == 1
    assert failures[0][0] == "pick-1"
    assert failures[0][1]["failure_code"] == "gen_sim_planning_wall_budget_exhausted"
    assert failures[0][1]["retryable"] is False
    assert (
        result.diagnostics.metadata["gen_sim_search_cost"]["wall_budget_seconds"]
        == 90.0
    )
    assert (
        result.diagnostics.metadata["gen_sim_search_cost"]["counters"][
            "within_extended_budget"
        ]
        == 1
    )
    assert costs._COST.get() is None
    assert len(captured) == int(omit_context)
