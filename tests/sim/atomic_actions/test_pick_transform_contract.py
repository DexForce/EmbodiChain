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

"""A changed emitted path must not reuse an earlier Pick clearance result."""

from __future__ import annotations
import hashlib, json
from pathlib import Path
import pytest
import torch
from embodichain.lab.sim.motion.search import search_cost_scope
from ._pick_transform_fixture import transform_inputs, transform, record

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/pick_transform_b0.json").read_text()
)
_CASES = _REFERENCE["cases"]
_CORRECTED = {
    ("ordinary", "arm_jump"),
    ("ordinary", "scene_bulge"),
    ("ordinary", "velocity_spike"),
    ("native", "scene_bulge"),
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c["mode"] + "-" + c["variant"])
def test_transformed_pick_rechecks_actual_path_and_preserves_original_candidate(
    tmp_path, monkeypatch, case
):
    pytest.importorskip(
        "fcl", reason="Robot-query wrapper coverage requires planning-geometry."
    )
    mode, variant = case["mode"], case["variant"]
    engine, invocation, context, action, calls, ik, events, state, provider = (
        transform_inputs(tmp_path, mode, monkeypatch)
    )
    torch.manual_seed(17)
    with search_cost_scope():
        plan = engine.plan(
            invocation,
            context,
            plan_transform=transform(action, events, state, variant),
        )
    actual = record(plan, calls, ik, events, state)
    for key in ("candidate_positions", "dt", "shape", "rng", "transform_count"):
        assert actual[key] == case[key], key
    assert _digest(calls) == case["calls_sha256"]
    assert _digest(ik) == case["ik_sha256"]
    before = lambda rows: [value for value in rows if value.get("stage") == "before"]
    assert before(events) == before(case["events"])
    if (mode, variant) in _CORRECTED:
        assert case["success"] == [
            True
        ]  # Frozen B0 red case: stale initial certificate.
        assert actual["success"] == [False]
    else:
        assert actual["success"] == case["success"]
    if not plan.plan_success.any():
        assert actual["commands_active"] == 0
        assert plan.rejected_candidate is not None
        assert torch.equal(
            plan.joint_trajectory.positions,
            context.robot.qpos[:, None].expand_as(plan.joint_trajectory.positions),
        )
    else:
        assert actual["positions"] == case["positions"]
        assert actual["commands_active"] == case["commands_active"]
    after = [value for value in events if value.get("stage") == "after"]
    if mode in {"ordinary", "native"}:
        # Invalid commands stop before scene/contact acquisition. A valid but
        # colliding changed path gets exactly one final scene query, no recovery.
        expected = (
            [] if variant in {"arm_jump", "velocity_spike"} else ["robot", "scene"]
        )
        assert [value["kind"] for value in after] == expected
        if variant == "scene_bulge":
            assert after[-1]["valid"] is False
    else:
        assert after == [
            value for value in case["events"] if value.get("stage") == "after"
        ]
    if provider is not None and mode == "native":
        before_captures = sum(value["kind"] == "plan" for value in events)
        assert len(provider.queries) == before_captures + (
            0 if variant in {"arm_jump", "scene_bulge", "velocity_spike"} else 1
        )
        assert all(query.closed for query in provider.queries)
