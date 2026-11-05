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

"""Complete drawer candidate budgets, reverse entry and selected transforms."""

from __future__ import annotations

from dataclasses import replace
import torch
import pytest
from ._drawer_fixture import drawer_fixture
from embodichain.lab.sim.atomic_actions import PlanningFailure


@pytest.mark.parametrize(
    "mode,count,success",
    [("nominal", 1, True), ("second_candidate", 2, True), ("exhausted", 2, False)],
)
def test_canonical_place_owns_bounded_candidates_and_reverse_entry(
    tmp_path, mode, count, success
):
    engine, _, context, invocation, calls, _, _ = drawer_fixture(tmp_path, mode)
    transforms = []

    def transform(request, current, plan):
        transforms.append(request.goal.xpos.clone())
        return plan

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert bool(plan.plan_success.all()) is success
    assert len(calls) == count
    assert len(transforms) == 1
    for call in calls:
        targets = torch.tensor(call["targets"])
        assert targets.shape[1] == 5
        assert torch.equal(targets[:, 0], targets[:, 4])
        assert torch.equal(targets[:, 1], targets[:, 3])
    records = plan.diagnostics.metadata["drawer_placement_candidates"]
    assert len(records) == count
    assert all(record["release_lift_m"] == 0.0 for record in records)


def test_upright_collision_defers_intermediate_lifts_but_keeps_final_attempt(tmp_path):
    engine, _, context, invocation, calls, queries, _ = drawer_fixture(
        tmp_path, "articulation_blocked", upright=True
    )
    plan = engine.plan(invocation, context)
    assert not plan.plan_success.any()
    assert len(calls) == 2
    attempts = plan.diagnostics.metadata["drawer_release_clearance_attempts"]
    assert [row["release_lift_m"] for row in attempts] == [0.0, 0.025, 0.05, 0.12]
    assert all(
        row["full_plan_deferred"] == "all_release_endpoints_collide"
        for row in attempts[:-1]
    )
    assert queries == ["open", "close"]


def test_selected_transform_failure_does_not_restart_candidate_search(tmp_path):
    engine, _, context, invocation, calls, _, _ = drawer_fixture(tmp_path)
    plan = engine.plan(
        invocation,
        context,
        plan_transform=lambda request, current, plan: replace(
            plan,
            plan_success=torch.zeros_like(plan.plan_success),
            diagnostics=replace(
                plan.diagnostics,
                failure=PlanningFailure("transform_rejected", retryable=False),
            ),
        ),
    )
    assert not plan.plan_success.any()
    assert len(calls) == 1


@pytest.mark.parametrize("upright", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_drawer_transport_uses_payload_query_and_preserves_held_relation(
    tmp_path, upright, fail
):
    from ._drawer_fixture import drawer_transport_fixture

    engine, _, context, invocation, calls, queries, observation, lifecycle, payloads = (
        drawer_transport_fixture(tmp_path, upright=upright, fail=fail)
    )
    plan = engine.plan(invocation, context)
    assert bool(plan.plan_success.all()) is not fail
    assert lifecycle == ["open", "close"]
    assert len(payloads) == 1
    payload, full = payloads[0]
    assert payload.object_id == "object" and payload.spheres.shape == (27, 4)
    assert torch.equal(full, context.robot.qpos)
    assert len(calls) == 1 and len(calls[0]["targets"][0]) == 1
    assert "cabinet__drawer" in plan.scene_dependencies
    assert not plan.expected_effects.held_object_updates
    assert plan.diagnostics.metadata["drawer_transport"]["world_ids"] == [
        "table",
        "cabinet__drawer",
    ]
    if upright:
        assert queries == ["open", "close"]
        assert (
            plan.diagnostics.metadata["drawer_transport_heading"]["scope"]
            == "endpoint_ranking_not_full_place_preview"
        )


def test_close_containment_rejects_before_any_motion_or_grasp_planning(tmp_path):
    from embodichain.lab.sim.atomic_actions import (
        Slide,
        SlideGoal,
        SlideOptions,
        SlideJointTarget,
        SlideAffordance,
        ObjectSemantics,
        SceneEntityPose,
    )

    engine, _, context, _, calls, _, observation = drawer_fixture(tmp_path)
    pose = torch.eye(4)[None]
    pose[:, 0, 3] = 1.0
    observation._obj.get_local_pose = lambda **kw: pose
    old = context.task.held_objects["arm"].semantics.affordance
    goal = SlideGoal(
        ObjectSemantics(
            SlideAffordance(
                mesh_vertices=old.mesh_vertices,
                mesh_triangles=old.mesh_triangles,
                translation_axis=torch.tensor([1.0, 0.0, 0.0]),
                joint_name="rail",
                joint_limits=(0.0, 0.4),
            ),
            {},
            entity_id="target",
        ),
        SceneEntityPose("target"),
        joint_target=SlideJointTarget("cabinet", "rail", 0.0, axis_sign=-1),
    )
    engine.register(Slide())
    invocation = engine.make_invocation(
        "slide",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        skill_options=SlideOptions(drawer_containment=True, direction="push"),
        invocation_id="drawer-close",
    )
    with pytest.raises(ValueError, match="payload containment failed"):
        engine.plan(invocation, context)
    assert calls == []
