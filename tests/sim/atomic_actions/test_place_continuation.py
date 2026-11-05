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

"""Private forecasts preserve source results without promoting predicted evidence."""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import replace
import pytest
import torch
from embodichain.lab.sim.atomic_actions.place_continuation import PlaceContinuationRoute
from embodichain.lab.sim.atomic_actions.primitives._place_continuation import (
    preview_place,
)
from ._place_continuation_fixture import preview_fixture

_CASES = json.loads(
    (Path(__file__).parent / "data/place_continuation_b0.json").read_text()
)


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=lambda c: f"{c['mode']}-{'transport' if c['transport'] else 'direct'}",
)
def test_forecast_preserves_source_guards_order_rng_and_observation(tmp_path, case):
    action, request, context, plan, session, trace = preview_fixture(
        tmp_path, case["mode"], case["transport"]
    )
    observed = context.robot
    qpos = observed.qpos.clone()
    object_pose = context.scene.entities[request.goal.semantics.entity_id].pose.clone()
    old_plan = session.plan

    def checked_plan(selected, future):
        assert future.is_preview
        assert future.observed_robot is observed
        assert future.world_source_stamp == context.world_source_stamp
        assert future.scene is not context.scene
        assert future.task is not context.task
        return old_plan(selected, future)

    session.plan = checked_plan
    torch.manual_seed(81)
    before = torch.random.get_rng_state().clone()
    try:
        result = preview_place(action, request, context, plan, session)
    except ValueError as error:
        result = {"raised": str(error)}
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(context.robot.qpos, qpos)
    assert torch.equal(
        context.scene.entities[request.goal.semantics.entity_id].pose, object_pose
    )
    assert not context.is_preview and not context.task.held_objects
    assert (
        json.loads(
            json.dumps(
                result, default=lambda v: dict(v) if hasattr(v, "items") else str(v)
            )
        )
        == case["result"]
    )
    assert trace == case["trace"]


def test_route_rejects_reused_occurrences_and_invalid_flags():
    with pytest.raises(ValueError, match="distinct"):
        PlaceContinuationRoute("call", "call", "object")
    with pytest.raises(ValueError, match="distinct"):
        PlaceContinuationRoute("pick", "place", "object", "place")
    with pytest.raises(TypeError, match="bool"):
        PlaceContinuationRoute("pick", "place", "object", requires_full_preview=1)


_RECOVERY_CASES = json.loads(
    (Path(__file__).parent / "data/pick_continuation_recovery_b0.json").read_text()
)


@pytest.mark.parametrize("case", _RECOVERY_CASES, ids=lambda c: c["mode"])
def test_bounded_pick_continuation_preserves_source_candidates_and_failures(
    tmp_path, monkeypatch, case
):
    from ._place_continuation_fixture import recovery_fixture
    from ._ordinary_pick_fixture import stable_closing_report
    from embodichain.lab.sim.atomic_actions.primitives import (
        _place_continuation as owner,
    )

    action, request, context, plan, selection, session, trace, forecast, inspect = (
        recovery_fixture(tmp_path, case["mode"])
    )
    monkeypatch.setattr(owner, "preview_place", forecast)
    monkeypatch.setattr(owner, "inspect_manipulation_plan", inspect)
    torch.manual_seed(91)
    before = torch.random.get_rng_state().clone()
    result = owner.recover(action, request, context, plan, selection, session)
    actual = dict(
        mode=case["mode"],
        success=result.plan_success.tolist(),
        metadata=stable_closing_report(dict(result.diagnostics.metadata)),
        failure=(
            None
            if result.diagnostics.failure is None
            else result.diagnostics.failure.code
        ),
        trace=trace,
    )
    actual = json.loads(
        json.dumps(actual, default=lambda x: dict(x) if hasattr(x, "items") else str(x))
    )
    assert actual == case
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("full_preview", [False, True])
def test_normal_atomic_engine_uses_explicit_occurrence_provider(tmp_path, full_preview):
    from types import SimpleNamespace as NS
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        ControlPartCommandProfile,
        RecoveryPolicy,
        PickPlanningRecovery,
    )
    from ._ordinary_pick_fixture import OrdinaryCandidates

    action, request, context, plan, session, trace = preview_fixture(tmp_path)
    session.route = replace(session.route, requires_full_preview=full_preview)
    queried = []

    def prepare(selected, current):
        queried.append((selected.invocation_id, current.is_preview))
        return session if selected.invocation_id == "pick" else None

    provider = NS(robot=action.robot, prepare=prepare)
    engine = AtomicActionEngine(
        action.motion_generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        grasp_pose_generators={"hand": OrdinaryCandidates()},
        place_continuation_provider=provider,
    )
    request = engine.resolve(
        engine.make_invocation(
            "pick_up",
            request.goal,
            control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
            motion_policy=replace(request.motion_policy, sample_count=140),
            skill_options=replace(request.skill_options, adaptive_grasp="ordinary"),
            invocation_id="pick",
            recovery_policy=RecoveryPolicy(
                planning_recovery=PickPlanningRecovery(place_continuation=True)
            ),
        )
    )
    result = engine.plan_request(request, context)
    assert result.plan_success.all()
    assert queried == [("pick", False)]
    if full_preview:
        assert [item["step"] for item in trace] == ["source", "place", "plan_place"]
    else:
        assert not trace
        assert (
            result.diagnostics.metadata["gen_sim_pick_continuation_deferred"][
                "physical_evidence"
            ]
            is False
        )


def test_explicit_pick_downstream_deferral_skips_the_entire_unresolved_tuple(tmp_path):
    from embodichain.lab.sim.atomic_actions import SceneEntityPose

    action, request, context, _, _, _ = preview_fixture(tmp_path)
    declared = replace(
        request.skill_options,
        downstream_object_target_poses=(
            SceneEntityPose("first_missing"),
            SceneEntityPose("second_missing"),
        ),
    )
    with pytest.raises((KeyError, ValueError)):
        action.plan(replace(request, skill_options=declared), context)
    deferred = replace(declared, qualify_downstream_targets=False)
    assert action.plan(
        replace(request, skill_options=deferred), context
    ).plan_success.all()
    assert len(declared.downstream_object_target_poses) == 2
