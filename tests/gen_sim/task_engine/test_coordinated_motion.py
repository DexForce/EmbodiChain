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

"""Continuation plans preserve the two verified grasps until intentional release."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import runpy
from unittest.mock import Mock

import pytest
import torch

from embodichain.gen_sim.task_engine._task_program.coordinated_motion import (
    GenSimCoordinatedPickment,
)
from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AntipodalAffordance,
    CoordinatedPickGoal,
    CoordinatedPickment,
    CoordinatedPickmentOptions,
    EntityState,
    HeldObjectState,
    MotionPolicy,
    ObjectSemantics,
    SceneSnapshot,
    TaskState,
)

# Reuse the shared CPU engine fixtures without turning tests/ into a package.
_shared = runpy.run_path(
    str(Path(__file__).parents[2] / "sim/atomic_actions/test_actions.py")
)
_bind_action = _shared["_bind_action"]
_dual_binding = _shared["_dual_binding"]
_dual_context = _shared["_dual_context"]
_dual_motion_generator = _shared["_dual_motion_generator"]
_stub_dual_arm_grasp_poses = _shared["_stub_dual_arm_grasp_poses"]


def setup_case(*, release=True, held=True, right_entity="tray", masks=None):
    generator = _dual_motion_generator()
    action = _bind_action(generator, GenSimCoordinatedPickment())
    semantics = ObjectSemantics(AntipodalAffordance(), {}, entity_id="tray")
    pose = torch.eye(4).repeat(2, 1, 1)
    pose[:, 2, 3] = 0.8
    target = pose.clone()
    target[:, 2, 3] -= 0.14
    states = {}
    if held:
        for i, resource in enumerate(("left_arm", "right_arm")):
            offset = torch.eye(4).repeat(2, 1, 1)
            offset[:, 1, 3] = -0.2 if i == 0 else 0.2
            actual = (
                semantics
                if i == 0 or right_entity == "tray"
                else ObjectSemantics(AntipodalAffordance(), {}, entity_id=right_entity)
            )
            states[resource] = HeldObjectState(
                actual,
                offset,
                pose @ offset,
                env_mask=None if masks is None else masks[i],
            )
    task = TaskState(batch_size=2, device="cpu", held_objects=states)
    context = _dual_context(
        task,
        scene=SceneSnapshot(
            timestamp=0.0, version=0, entities={"tray": EntityState(pose)}
        ),
    )
    qpos = context.robot.qpos.clone()
    for name in ("left_hand", "right_hand"):
        qpos[:, generator.robot.get_joint_ids(name=name)] = 1.0
    context = replace(context, robot=replace(context.robot, qpos=qpos))
    invocation = ActionInvocation(
        skill_id=action.skill_id,
        goal=CoordinatedPickGoal(semantics=semantics, object_target_pose=target),
        binding=_dual_binding(action, "left", "right"),
        motion_policy=MotionPolicy(sample_count=40),
        skill_options=CoordinatedPickmentOptions(
            release=release,
            hold_steps=2,
            release_steps=4,
            retreat_steps=6,
            object_motion_keyframes=3,
        ),
    )
    return action, context, action.resolve_request(invocation), generator


@pytest.mark.parametrize("release", [False, True])
def test_held_continuation_never_opens_or_resamples_before_release(release):
    action, context, request, generator = setup_case(release=release)
    action._pickup.plan = Mock(
        side_effect=AssertionError("must not re-pick held object")
    )
    plan = action.plan(request, context)
    assert plan.plan_success.tolist() == [True, True]
    names = [s.name for s in plan.segments]
    assert names == (
        ["move", "hold", "release", "retreat"] if release else ["move", "hold"]
    )
    stop = next(
        (s.start for s in plan.segments if s.name == "release"),
        plan.commands.frame_count,
    )
    trajectory = plan.joint_trajectory.positions
    for hand in ("left_hand", "right_hand"):
        ids = generator.robot.get_joint_ids(name=hand)
        assert torch.all(trajectory[:, :stop, ids] == 1.0)
        if release:
            assert torch.all(trajectory[:, -1, ids] == 0.0)
    assert plan.scene_dependency_monitor_until["tray"] == 0
    for key in ("left_arm", "right_arm"):
        assert context.task.get_held_object(key) is not None
        projected = plan.expected_effects.held_object_updates[key]
        if release:
            assert projected is None
        else:
            torch.testing.assert_close(
                projected.object_to_eef, context.task.get_held_object(key).object_to_eef
            )


def test_no_grasp_uses_unchanged_pickup_pipeline():
    action, context, request, _ = setup_case(held=False)
    _stub_dual_arm_grasp_poses(action)
    plan = action.plan(request, context)
    assert plan.plan_success.all()
    assert [s.name for s in plan.segments][:3] == ["approach", "close", "lift"]
    assert GenSimCoordinatedPickment.descriptor() == CoordinatedPickment.descriptor()


def test_wrong_object_does_not_fall_back_to_opening_hands():
    action, context, request, _ = setup_case(right_entity="other")
    action._pickup.plan = Mock(side_effect=AssertionError("unsafe pickup fallback"))
    plan = action.plan(request, context)
    assert not plan.plan_success.any()


def test_partial_attachment_fails_only_the_unverified_row():
    action, context, request, _ = setup_case(
        masks=(torch.tensor([True, True]), torch.tensor([True, False]))
    )
    plan = action.plan(request, context)
    assert plan.plan_success.tolist() == [True, False]
    torch.testing.assert_close(
        plan.joint_trajectory.positions[1],
        context.robot.qpos[1].expand(plan.commands.frame_count, -1),
    )
