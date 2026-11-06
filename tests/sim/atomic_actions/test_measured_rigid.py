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

"""Ordinary Atomic calls expose measured transformation without task-engine patches."""

from __future__ import annotations
from dataclasses import replace
import pytest
import torch
from ._measured_rigid_fixture import fixture


@pytest.mark.parametrize(
    "mode", ["accepted", "base_failure", "blocked", "closure_rejected", "unavailable"]
)
def test_direct_atomic_pick_measured_preshape_is_bounded_and_closes_queries(
    tmp_path, monkeypatch, mode
):
    engine, invocation, context, provider, _, _ = fixture(tmp_path, monkeypatch, mode)
    torch.manual_seed(147)
    result = engine.plan(invocation, context)
    session = provider.sessions[0]
    assert session.closed
    assert all(value.closed for value in session.queries + session.scenes)
    if mode == "accepted":
        assert result.plan_success.tolist() == [True], result.diagnostics
        report = result.diagnostics.metadata["measured_rigid"]
        assert report["accepted"] and report["extra_side_reserve_m"] == 0.002
        assert report["unchanged_arm_and_timing"] is True
        assert report["preshape_fraction"] > 0
        assert session.events == ["closing", "scene", "closing"]
    else:
        assert result.plan_success.tolist() == [False]
        if mode == "base_failure":
            assert session.events == []
        else:
            assert result.diagnostics.failure.code == "measured_rigid_unqualified"
    assert context.task.get_held_object("arm") is None


def test_measured_preshape_preview_uses_ordinary_planner_without_native_capture(
    tmp_path, monkeypatch
):
    engine, invocation, context, provider, _, _ = fixture(tmp_path, monkeypatch)
    result = engine.plan(invocation, replace(context, is_preview=True))
    assert result.plan_success.all() and provider.sessions == []


@pytest.mark.parametrize(
    "mode", ["accepted", "base_failure", "blocked", "closure_rejected", "unavailable"]
)
def test_complete_measured_pick_transform_matches_frozen_b0(
    tmp_path, monkeypatch, mode
):
    from pathlib import Path
    import json, hashlib, random
    import numpy as np
    from ._ordinary_pick_fixture import stable_closing_report

    engine, invocation, context, provider, calls, ik = fixture(
        tmp_path, monkeypatch, mode
    )
    random.seed(147)
    np.random.seed(147)
    torch.manual_seed(147)
    result = engine.plan(invocation, context)
    digest = lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest()
    metadata = stable_closing_report(dict(result.diagnostics.metadata))
    metadata.pop("search_cost", None)
    actual = dict(
        mode=mode,
        success=result.plan_success.tolist(),
        positions=digest(result.joint_trajectory.positions),
        dt=digest(result.joint_trajectory.dt),
        velocities=(
            None
            if result.joint_trajectory.velocities is None
            else digest(result.joint_trajectory.velocities)
        ),
        segments=[[s.name, s.start, s.stop] for s in result.segments],
        metadata=metadata,
        failure=(
            None
            if result.diagnostics.failure is None
            else dict(
                code=result.diagnostics.failure.code,
                retryable=result.diagnostics.failure.retryable,
            )
        ),
        events=provider.sessions[0].events,
        calls=calls,
        ik=ik,
        after_rng=[
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    )
    actual = json.loads(
        json.dumps(
            actual,
            default=lambda value: (
                dict(value) if hasattr(value, "items") else str(value)
            ),
        )
    )
    for key in ("calls", "ik"):
        actual[key + "_sha256"] = hashlib.sha256(
            json.dumps(actual.pop(key), sort_keys=True).encode()
        ).hexdigest()
    expected = next(
        row
        for row in json.loads(
            (Path(__file__).parent / "data/measured_pick_b0.json").read_text()
        )["cases"]
        if row["mode"] == mode
    )
    assert actual == expected


@pytest.mark.parametrize("mode", ["accepted", "occupied", "blocked", "unavailable"])
def test_direct_atomic_inside_measured_policy_preserves_bounded_candidate_scope(
    tmp_path, monkeypatch, mode
):
    from ._measured_rigid_fixture import place_fixture

    engine, invocation, context, provider, _, _ = place_fixture(
        tmp_path, monkeypatch, mode
    )
    torch.manual_seed(147)
    result = engine.plan(invocation, context)
    session = provider.sessions[0]
    assert session.closed and all(
        query.closed for query in session.queries + session.scenes
    )
    if mode in {"accepted", "occupied"}:
        assert result.plan_success.all(), result.diagnostics
        report = result.diagnostics.metadata["measured_rigid"]
        assert report["release_clearance_m_unchanged"] == 0.12
        assert len(report["candidates"]) == 1
        assert report["landing"]["observed_contents"] == (
            ["content"] if mode == "occupied" else []
        )
    else:
        assert not result.plan_success.any()
        assert result.diagnostics.failure.code == "measured_rigid_unqualified"
    assert context.get_held_object("arm") is not None


@pytest.mark.parametrize("mode", ["accepted", "occupied", "blocked", "unavailable"])
def test_complete_measured_inside_transform_matches_frozen_b0(
    tmp_path, monkeypatch, mode
):
    from pathlib import Path
    import json, hashlib, random
    import numpy as np
    from ._ordinary_pick_fixture import stable_closing_report
    from ._measured_rigid_fixture import place_fixture

    engine, invocation, context, provider, calls, ik = place_fixture(
        tmp_path, monkeypatch, mode
    )
    random.seed(147)
    np.random.seed(147)
    torch.manual_seed(147)
    result = engine.plan(invocation, context)
    digest = lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest()
    metadata = stable_closing_report(dict(result.diagnostics.metadata))
    metadata.pop("search_cost", None)
    actual = dict(
        mode=mode,
        success=result.plan_success.tolist(),
        positions=digest(result.joint_trajectory.positions),
        dt=digest(result.joint_trajectory.dt),
        velocities=(
            None
            if result.joint_trajectory.velocities is None
            else digest(result.joint_trajectory.velocities)
        ),
        segments=[[s.name, s.start, s.stop] for s in result.segments],
        metadata=metadata,
        failure=(
            None
            if result.diagnostics.failure is None
            else dict(
                code=result.diagnostics.failure.code,
                retryable=result.diagnostics.failure.retryable,
            )
        ),
        events=provider.sessions[0].events,
        calls=calls,
        ik=ik,
        after_rng=[
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    )
    actual = json.loads(
        json.dumps(
            actual,
            default=lambda value: (
                dict(value) if hasattr(value, "items") else str(value)
            ),
        )
    )
    for key in ("calls", "ik"):
        actual[key + "_sha256"] = hashlib.sha256(
            json.dumps(actual.pop(key), sort_keys=True).encode()
        ).hexdigest()
    expected = next(
        row
        for row in json.loads(
            (Path(__file__).parent / "data/measured_place_b0.json").read_text()
        )["cases"]
        if row["mode"] == mode
    )
    assert actual == expected
