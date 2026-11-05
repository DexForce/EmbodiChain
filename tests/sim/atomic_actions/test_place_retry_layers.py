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

"""Independent inner segment and outer action budgets through shared Place."""

from __future__ import annotations
from dataclasses import replace
import hashlib, json, random
from pathlib import Path
import numpy as np
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    PlaceOptions,
)
from embodichain.lab.sim.motion import _cartesian as cartesian
from ._place_retry_fixture import retry_fixture

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/place_retry_layers_reference.json").read_text()
)["cases"]


def inputs(tmp_path, mode, *, enforce_velocity=False):
    generator, context, goal, calls = retry_fixture(tmp_path, mode)
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.12), grasp=torch.zeros(2)
            )
        },
    )
    policy = MotionPolicy(
        sample_count=140,
        dynamic_collision_mode="off",
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        acceleration_limit=None,
        sample_count_retries=() if mode == "inner" else (260, 320),
        segment_sample_count_retries=() if mode == "outer" else (260, 320),
        segment_continuous_ik_recovery=mode != "outer",
        require_final_velocity=enforce_velocity,
    )
    options = PlaceOptions(
        hand_interp_steps=2,
        release_settle_steps=0,
        lift_height=0.1,
        cartesian_waypoint_count=2,
        preserve_current_object_orientation=True,
        linear_segments=mode != "outer",
    )
    invocation = engine.make_invocation(
        "place",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=policy,
        skill_options=options,
        invocation_id="place",
    )
    return engine, invocation, context, calls


@pytest.mark.parametrize("reference", _REFERENCE, ids=lambda item: item["mode"])
def test_nested_sampling_budgets_match_source_attempt_order_and_rng(
    tmp_path, monkeypatch, reference
):
    engine, invocation, context, calls = inputs(tmp_path, reference["mode"])
    monkeypatch.setattr(cartesian, "resolve_continuous_ik_path", lambda *a, **k: None)
    torch.manual_seed(17)
    np.random.seed(17)
    random.seed(17)
    plan = engine.plan(invocation, context)
    digest = lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest()
    actual = {
        "mode": reference["mode"],
        "calls": calls,
        "success": plan.plan_success.tolist(),
        "positions_sha256": digest(plan.joint_trajectory.positions),
        "dt_sha256": digest(plan.joint_trajectory.dt),
        "segments": [[s.name, s.start, s.stop] for s in plan.segments],
        "after_rng": [
            digest(torch.get_rng_state()),
            float(np.random.rand()),
            random.random(),
        ],
    }
    # The source's best-effort action exhaustion may retain an overspeed plan.
    # These plans are not executed; final velocity qualification is separate.
    assert actual == reference
    if reference["mode"] == "both":
        assert plan.diagnostics.metadata["segment_sampling"][
            "attempted_sample_counts"
        ] == [138, 260]
        outer = plan.diagnostics.metadata["motion_sample_retries"]
        assert outer["velocity_retime_attempts"] == [260, 320]
        assert [
            item["segment_sampling"]["attempted_sample_counts"]
            for item in outer["segment_sampling_attempts"]
        ] == [[258, 260], [318]]


def test_nested_retry_exhaustion_keeps_original_candidate_and_cannot_dispatch(
    tmp_path, monkeypatch
):
    engine, invocation, context, calls = inputs(tmp_path, "both", enforce_velocity=True)
    monkeypatch.setattr(cartesian, "resolve_continuous_ik_path", lambda *a, **k: None)
    plan = engine.plan(invocation, context)
    assert [call["count"] for call in calls] == [138, 260, 258, 260, 318]
    assert not plan.plan_success.any()
    assert all(not frame.active_mask.any() for frame in plan.commands.frames)
    digest = lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest()
    assert (
        digest(plan.rejected_candidate.positions) == _REFERENCE[0]["positions_sha256"]
    )
    assert digest(plan.rejected_candidate.dt) == _REFERENCE[0]["dt_sha256"]
