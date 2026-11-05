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

"""Complete alignment-to-Place search preserves source candidate and RNG order."""

from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import json, random
import numpy as np
import pytest
import torch
from embodichain.lab.sim.motion import mesh_collision
from ._alignment_continuation_fixture import continuation_fixture, trace_record


@pytest.mark.parametrize(
    "mode", ["success", "accepted", "exhausted", "exception", "release_mismatch"]
)
def test_complete_alignment_continuation_matches_frozen_source(
    tmp_path, monkeypatch, mode
):
    engine, invocation, context, session, check, calls, checks, ik, trace = (
        continuation_fixture(tmp_path, mode, enabled=True)
    )
    monkeypatch.setattr(mesh_collision, "_make_checker", lambda *a, **k: check)
    original = session.plan

    def future_plan(request, predicted):
        assert predicted.is_preview and predicted.observed_robot is context.robot
        assert predicted.world_source_stamp == context.world_source_stamp
        return original(request, predicted)

    session.plan = future_plan
    before = context.robot.qpos.clone()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    result = engine.plan(invocation, context)
    actual = json.loads(
        json.dumps(
            trace_record(result, calls, checks, ik, trace),
            default=lambda value: (
                dict(value) if hasattr(value, "items") else str(value)
            ),
        )
    )
    expected = json.loads(
        (
            Path(__file__).parent / "data" / f"alignment_continuation_{mode}_b0.json"
        ).read_text()
    )
    assert actual == expected
    assert torch.equal(context.robot.qpos, before)
    assert not context.is_preview


def test_complete_alignment_continuation_preserves_transform_per_candidate(
    tmp_path, monkeypatch
):
    engine, invocation, context, session, check, calls, checks, ik, trace = (
        continuation_fixture(tmp_path, "accepted", enabled=True)
    )
    monkeypatch.setattr(mesh_collision, "_make_checker", lambda *a, **k: check)
    transforms = []

    def transform(request, current, plan):
        transforms.append(request.goal.world_yaw_free)
        return plan

    result = engine.plan(invocation, context, plan_transform=transform)
    assert result.plan_success.all()
    assert transforms == [True, False, False]
    assert result.diagnostics.metadata["gen_sim_alignment_continuation"][
        "yaw_offset_rad"
    ] == pytest.approx(np.pi * 0.25)


def test_declared_continuation_rejects_missing_retention_authority(
    tmp_path, monkeypatch
):
    engine, invocation, context, _, check, *_ = continuation_fixture(
        tmp_path, "authority_missing", enabled=True
    )
    monkeypatch.setattr(mesh_collision, "_make_checker", lambda *a, **k: check)
    with pytest.raises(ValueError, match="ownership is unavailable"):
        engine.plan(invocation, context)


def test_cancelled_continuation_restores_the_post_original_random_stream(
    tmp_path, monkeypatch
):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    engine, invocation, context, _, check, *_ = continuation_fixture(
        baseline, "success"
    )
    monkeypatch.setattr(mesh_collision, "_make_checker", lambda *a, **k: check)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    engine.plan(invocation, context)
    expected = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
    engine, invocation, context, _, check, *_ = continuation_fixture(
        tmp_path, "cancel", enabled=True
    )
    monkeypatch.setattr(mesh_collision, "_make_checker", lambda *a, **k: check)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    with pytest.raises(KeyboardInterrupt):
        engine.plan(invocation, context)
    assert torch.equal(expected[0], torch.get_rng_state())
    assert np.array_equal(expected[1][1], np.random.get_state()[1])
    assert expected[2] == random.getstate()
