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

"""Source released-hand route search preserves complete plans and random streams."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json, random
import numpy as np
import pytest
import torch
from ._released_clearance_fixture import fixture, record
from embodichain.lab.sim.atomic_actions.primitives._released_clearance import recover
from embodichain.lab.sim.motion import search
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport


@pytest.mark.parametrize(
    "mode",
    [
        "nominal",
        "retreat",
        "height",
        "exhausted",
        "unavailable",
        "initial",
        "secondpark",
    ],
)
def test_complete_released_search_matches_independent_source_ast(tmp_path, mode):
    engine, invocation, context, session, query, calls = fixture(
        tmp_path, mode, two_parks=mode == "secondpark"
    )
    request = engine.resolve(invocation)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    original = engine.plan_request(request, context)
    result = recover(
        request,
        context,
        original,
        build=lambda selected, current: engine.plan_request(selected, current),
        session=session,
        robot=engine.robot,
        query=query,
    )
    expected = json.loads(
        (Path(__file__).parent / f"data/released_clearance_{mode}_b0.json").read_text()
    )
    assert record(result, query, calls) == expected
    assert not context.is_preview


@pytest.mark.parametrize("mode", ["retreat", "exhausted"])
def test_plain_atomic_entry_uses_released_route_provider_without_registry_patches(
    tmp_path,
    mode,
):
    engine, invocation, context, session, query, calls = fixture(
        tmp_path, mode, enabled=True
    )
    result = engine.plan(invocation, context)
    assert result.plan_success.tolist() == [mode != "exhausted"]
    if mode == "exhausted":
        assert result.diagnostics.failure.code == "cleanup_clearance_failed"
        return
    assert (
        result.diagnostics.metadata["gen_sim_clearance_recovery"]["selected_route"]
        == "retreat_0.05_extra_height_0.00"
    )
    assert result.expected_effects.is_empty


def test_cancelled_cleanup_search_restores_random_state(tmp_path):
    engine, invocation, context, session, query, calls = fixture(tmp_path, "cancel")
    request = engine.resolve(invocation)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    original = engine.plan_request(request, context)
    before = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    with pytest.raises(KeyboardInterrupt):
        recover(
            request,
            context,
            original,
            build=lambda selected, current: engine.plan_request(selected, current),
            session=session,
            robot=engine.robot,
            query=query,
        )
    assert torch.equal(before[0], torch.get_rng_state())
    assert np.array_equal(before[1][1], np.random.get_state()[1])
    assert before[2] == random.getstate()


def test_cleanup_preview_keeps_original_transform_scoped_to_each_following_call(
    tmp_path,
):
    engine, invocation, context, session, query, calls = fixture(
        tmp_path, "nominal", enabled=True
    )
    transformed = []

    def transform(request, current, plan):
        transformed.append(
            (request.invocation_id, current.is_preview, current.observed_robot)
        )
        return plan

    result = engine.plan(invocation, context, plan_transform=transform)
    assert result.plan_success.tolist() == [True]
    assert [(i, preview) for i, preview, _ in transformed] == [
        ("cleanup", False),
        ("park0", True),
    ]
    assert transformed[-1][2] is context.robot
    engine.plan(invocation, context)
    assert len(transformed) == 2


def test_cancelled_transformed_park_preview_restores_original_cleanup_streams(tmp_path):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    engine, invocation, context, *_ = fixture(baseline, "nominal")
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    engine.plan(invocation, context)
    expected = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    engine, invocation, context, *_ = fixture(tmp_path, "nominal", enabled=True)

    def cancelled(request, current, plan):
        if request.invocation_id == "park0":
            raise KeyboardInterrupt()
        return plan

    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    with pytest.raises(KeyboardInterrupt):
        engine.plan(invocation, context, plan_transform=cancelled)
    assert torch.equal(expected[0], torch.get_rng_state())
    assert np.array_equal(expected[1][1], np.random.get_state()[1])
    assert expected[2] == random.getstate()


def test_suffix_preview_rejects_a_request_owned_by_another_engine(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    foreign, _, _, foreign_session, _, foreign_calls = fixture(other, "nominal")
    engine, invocation, context, session, _, _ = fixture(
        tmp_path, "nominal", enabled=True
    )
    foreign_request = foreign_session.resolve_park(0, context)
    session.resolve_park = lambda index, current: foreign_request
    result = engine.plan(invocation, context)
    assert result.plan_success.tolist() == [False] and foreign_calls == []
    attempt = result.diagnostics.metadata["gen_sim_clearance_recovery"]["attempts"][0]
    assert attempt["reason"] == "park_planning_error" and "identity" in attempt["error"]


def _released_park_with_collision(tmp_path, *, collision, clock=None, expires_at=None):
    engine, _, context, session, query, calls = fixture(
        tmp_path, "nominal", enabled=True
    )
    request = session.resolve_park(0, context)
    inspect = query.inspect

    def observed(positions, *, control_part):
        report = inspect(positions, control_part=control_part)
        if clock is not None and len(query.reads) == expires_at:
            clock[0] = 1.0
        if collision and len(query.reads) == 1:
            return MotionCollisionReport(
                False, frame=3, kind="scene", pairs=(("arm", "fixture"),)
            )
        return report

    query.inspect = observed
    return engine, request, context, query, calls


@pytest.mark.parametrize("collision", [False, True])
def test_released_park_counts_complete_candidates_without_counting_nominal_twice(
    tmp_path, collision
):
    engine, request, context, query, calls = _released_park_with_collision(
        tmp_path, collision=collision
    )
    with search.search_cost_scope() as parent:
        with search.search_cost_scope(wall_seconds=0.1) as nested:
            assert nested is parent and nested.wall_seconds is None
            result = engine.plan_request(request, context)
        assert parent.counters["complete_plan_request"] == (2 if collision else 1)
    assert result.plan_success.tolist() == [True]
    assert len(calls) == (2 if collision else 1)
    assert len(query.reads) == (3 if collision else 1)
    assert search.current_search_cost() is None


@pytest.mark.parametrize("expires_at", [1, 2])
def test_released_park_checks_inherited_deadline_before_and_after_rebuild(
    tmp_path, monkeypatch, expires_at
):
    clock = [0.0]
    monkeypatch.setattr(search, "perf_counter", lambda: clock[0])
    engine, request, context, query, calls = _released_park_with_collision(
        tmp_path, collision=True, clock=clock, expires_at=expires_at
    )
    with pytest.raises(search.PlanningBudgetExceeded) as failure:
        with search.search_cost_scope(wall_seconds=1.0) as parent:
            engine.plan_request(request, context)
    assert failure.value.operation == "complete_plan_request"
    assert parent.counters["complete_plan_request"] == expires_at
    # Expiry before rebuilding permits no second complete plan. Expiry after
    # recursive qualification permits no outer revalidation query.
    assert len(calls) == expires_at
    assert len(query.reads) == expires_at
    assert search.current_search_cost() is None
