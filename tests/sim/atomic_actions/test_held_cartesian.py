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

"""Declared held Cartesian grids preserve source per-heading recovery order."""

from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib, json, random
import numpy as np
import pytest
import torch
from .test_held_alignment import _inputs


@pytest.mark.parametrize(
    "mode", ["safe", "denser", "fallback", "lift_fallback", "raw_failed"]
)
def test_held_cartesian_grid_and_density_retries_match_source_scope(
    tmp_path, monkeypatch, mode
):
    engine, invocation, context, calls, checks, ik, captures = _inputs(
        tmp_path, monkeypatch, mode, linear=True
    )
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda t: hashlib.sha256(t.numpy().tobytes()).hexdigest()
    hashed = lambda value: hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    actual = {
        "mode": mode,
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "metadata": dict(plan.diagnostics.metadata),
        "failure": (
            None if plan.diagnostics.failure is None else plan.diagnostics.failure.code
        ),
        "calls_sha256": hashed(calls),
        "checks_sha256": hashed(checks),
        "ik_sha256": hashed(ik),
        "held_keys": list(plan.expected_effects.held_object_updates),
        "rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    expected = json.loads(
        (
            Path(__file__).parent / f"data/held_cartesian_{mode}_reference.json"
        ).read_text()
    )
    assert actual == expected
    if mode == "raw_failed":
        assert [call["count"] for call in calls] == [45, 260, 320] * 8
    elif mode == "denser":
        assert [call["count"] for call in calls] == [45, 260]
        assert plan.joint_trajectory.waypoint_count == 260


def test_held_segment_continuity_cannot_replace_source_final_heading_fallback(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls, *_ = _inputs(
        tmp_path, monkeypatch, "safe", linear=True
    )
    invocation = replace(
        invocation,
        motion_policy=replace(
            invocation.motion_policy, segment_continuous_ik_recovery=True
        ),
    )
    with pytest.raises(ValueError, match="continuity belongs to final-heading"):
        engine.plan(invocation, context)
    assert calls == []
