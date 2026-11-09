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

"""Ordinary Place geometry and command boundaries against the frozen source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    PlaceOptions,
)
from ._ordinary_pick_fixture import place_fixture


@pytest.mark.parametrize("ik_failure", [False, True])
def test_ordinary_place_preserves_source_targets_commands_and_rng(tmp_path, ik_failure):
    generator, context, goal, calls, ik = place_fixture(tmp_path)
    if ik_failure:
        generator.robot.compute_batch_ik.side_effect = lambda pose, joint_seed, **kw: (
            torch.zeros(pose.shape[:2], dtype=torch.bool),
            joint_seed.clone(),
        )
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
    )
    invocation = engine.make_invocation(
        "place",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
        ),
        skill_options=PlaceOptions(
            lift_height=0.1,
            hand_interp_steps=12,
            release_settle_steps=60,
            cartesian_waypoint_count=2,
            preserve_current_object_orientation=True,
            linear_segments=True,
            qualify_required_waypoints=True,
            monitor_scene_until_approach=True,
        ),
        invocation_id="place",
    )
    torch.manual_seed(147)
    plan = engine.plan(invocation, context)
    digest = lambda tensor: hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    filename = (
        "ordinary_place_failure_source_reference.json"
        if ik_failure
        else "ordinary_place_source_reference.json"
    )
    reference = json.loads((Path(__file__).parent / "data" / filename).read_text())
    actual = dict(
        calls=calls,
        ik=ik,
        success=plan.plan_success.tolist(),
        positions_sha256=digest(plan.joint_trajectory.positions),
        dt_sha256=digest(plan.joint_trajectory.dt),
        segments=[list((s.name, s.start, s.stop)) for s in plan.segments],
        metadata=dict(plan.diagnostics.metadata),
        rng=digest(torch.get_rng_state()),
        monitor_until=dict(plan.scene_dependency_monitor_until),
        released=[
            key
            for key, value in plan.expected_effects.held_object_updates.items()
            if value is None
        ],
    )
    assert actual == {key: reference[key] for key in actual}
    if ik_failure:
        assert calls == []
        assert not plan.commands.frames
    else:
        assert plan.scene_dependency_monitor_until["target"] == plan.segments[0].stop
        assert plan.expected_effects.held_object_updates["arm"] is None


@pytest.mark.parametrize(
    "option",
    ["linear_segments", "qualify_required_waypoints", "monitor_scene_until_approach"],
)
def test_new_place_options_reject_external_nominal_before_source(tmp_path, option):
    generator, context, goal, calls, _ = place_fixture(tmp_path)
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
    )
    invocation = engine.make_invocation(
        "place",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        skill_options=PlaceOptions(**{option: True}),
    )

    class Source:
        def instantiate(self, request, current):
            raise AssertionError("Unsupported Place must not call the nominal source.")

    with pytest.raises(ValueError, match="External Place"):
        engine.plan(invocation, context, nominal_source=Source())
    assert calls == []


@pytest.mark.parametrize("skill,linear", [("pick_up", False), ("place", False)])
def test_unsupported_segment_policy_rejects_before_skill_planning(
    tmp_path, skill, linear
):
    from embodichain.lab.sim.atomic_actions import GraspGoal, PickUpOptions

    generator, context, goal, calls, _ = place_fixture(tmp_path)
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
    )
    if skill == "pick_up":
        goal = GraspGoal(semantics=context.task.held_objects["arm"].semantics)
        options = PickUpOptions()
    else:
        options = PlaceOptions(linear_segments=linear)
    invocation = engine.make_invocation(
        skill,
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        skill_options=options,
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            segment_sample_count_retries=(260, 320),
        ),
    )
    with pytest.raises(ValueError, match="recovery requires"):
        engine.plan(invocation, context)
    assert not calls
