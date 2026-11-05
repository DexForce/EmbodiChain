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


"""Source Idle commands and qualification through normal Atomic entry points."""

from __future__ import annotations

import json
import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.sim.motion.mesh_collision import SimulationFreeMotionQuery
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    JointPositionGoal,
    JointOffsetTarget,
    MotionPolicy,
    MoveJointsOptions,
    AxisAlignAffordance,
    HeldObjectState,
    ObjectSemantics,
    TaskState,
)
from motion._mesh_collision_fixture import mesh_query_fixture

_requires_fcl = pytest.mark.skipif(
    importlib.util.find_spec("fcl") is None,
    reason="actual FCL comparisons require planning-geometry; skipped coverage is unqualified",
)
_IDLE_REFERENCE = json.loads(
    (Path(__file__).parent / "data/idle_motion_source_reference.json").read_text()
)


def _idle_engine(*, install_query=True):
    robot, simulation = mesh_query_fixture()
    generator = object.__new__(MotionGenerator)
    generator.robot, generator.device = robot, torch.device("cpu")
    generator.planner = SimpleNamespace(
        cfg=SimpleNamespace(planner_type="toppra"), collision_world_info=None
    )
    engine = AtomicActionEngine(
        generator,
        free_motion_query=(
            SimulationFreeMotionQuery(robot, simulation) if install_query else None
        ),
    )
    return engine


def _idle_invocation(engine, target, *, qualify=True):
    return engine.make_invocation(
        "move_joints",
        JointPositionGoal(torch.tensor(target)),
        control_parts={"primary": {"motion": "arm"}},
        motion_policy=MotionPolicy(
            sample_count=3,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            acceleration_limit=None,
        ),
        skill_options=MoveJointsOptions(require_free_motion=qualify),
        invocation_id="idle",
    )


@pytest.mark.parametrize(
    "case", _IDLE_REFERENCE["cases"], ids=lambda case: case["name"]
)
@_requires_fcl
def test_builtin_move_joints_matches_source_engine_and_real_motion_query(case):
    engine = _idle_engine()
    context = engine.initial_context(control_dt=0.04)
    torch.manual_seed(491)
    before = torch.get_rng_state().clone()
    plan = engine.plan(_idle_invocation(engine, case["target"]), context)
    actual = {
        "name": case["name"],
        "target": case["target"],
        "positions": plan.joint_trajectory.positions.tolist(),
        "dt": plan.joint_trajectory.dt.tolist(),
        "success": plan.plan_success.tolist(),
        "active_masks": [frame.active_mask.tolist() for frame in plan.commands.frames],
        "rng_unchanged": torch.equal(before, torch.get_rng_state()),
    }
    assert actual == case


def test_free_motion_requires_service_only_when_explicitly_enabled():
    engine = _idle_engine(install_query=False)
    context = engine.initial_context(control_dt=0.04)
    assert engine.plan(
        _idle_invocation(engine, [0.1, 0], qualify=False), context
    ).plan_success.all()
    failed = engine.plan(_idle_invocation(engine, [0.1, 0]), context)
    assert not failed.plan_success.any()
    assert "query is unavailable" in " ".join(failed.diagnostics.messages)
    assert all(not frame.active_mask.any() for frame in failed.commands.frames)


def test_idle_free_motion_rejects_an_already_held_resource():
    engine = _idle_engine(install_query=False)
    context = engine.initial_context(control_dt=0.04)
    held = HeldObjectState(
        ObjectSemantics(
            AxisAlignAffordance(internal_axis=torch.tensor([0.0, 0, 1.0])), {}, "item"
        ),
        object_to_eef=torch.eye(4)[None],
        grasp_xpos=torch.eye(4)[None],
    )
    context = replace(context, task=TaskState(1, "cpu", held_objects={"arm": held}))
    failed = engine.plan(_idle_invocation(engine, [0.1, 0]), context)
    assert not failed.plan_success.any()
    assert "cannot move a held object" in " ".join(failed.diagnostics.messages)
    assert context.task.held_object_mask("arm").all()
    assert all(not frame.active_mask.any() for frame in failed.commands.frames)


@_requires_fcl
def test_transformed_idle_path_is_rechecked_and_masked_before_dispatch():
    engine = _idle_engine()
    context = engine.initial_context(control_dt=0.04)
    invocation = _idle_invocation(engine, [0.1, 0])

    def transform(request, current, plan):
        positions = plan.joint_trajectory.positions.clone()
        positions[:, 1, 0] = (
            0.25  # Original path is clear; this emitted sample hits the scene.
        )
        return engine.rebuild_plan_from_trajectory(
            request, current, plan, replace(plan.joint_trajectory, positions=positions)
        )

    plan = engine.plan(invocation, context, plan_transform=transform)
    assert not plan.plan_success.any()
    assert plan.diagnostics.failure.code == "idle_final_plan_unqualified"
    assert "'kind': 'scene'" in " ".join(plan.diagnostics.messages)
    torch.testing.assert_close(
        plan.joint_trajectory.positions, context.robot.qpos[:, None].expand(1, 3, 5)
    )
    torch.testing.assert_close(
        plan.joint_trajectory.dt, torch.tensor([[0.0, 0.04, 0.04]])
    )
    assert all(not frame.active_mask.any() for frame in plan.commands.frames)


@_requires_fcl
def test_relative_idle_goal_keeps_bound_joints_and_authored_limits():
    engine = _idle_engine()
    context = engine.initial_context(control_dt=0.04)
    base = _idle_invocation(engine, [0.0, 0.0])
    invocation = replace(
        base, goal=JointPositionGoal(JointOffsetTarget("arm_slide", 0.05))
    )
    plan = engine.plan(invocation, context)
    assert plan.plan_success.tolist() == [True]
    torch.testing.assert_close(
        plan.joint_trajectory.positions[:, -1],
        torch.tensor([[0.05, 0.0, 0.0, 0.0, 0.0]]),
    )
    unsafe = replace(
        invocation, goal=JointPositionGoal(JointOffsetTarget("arm_slide", 2.0))
    )
    assert engine.plan(unsafe, context).plan_success.tolist() == [False]
    with pytest.raises(ValueError, match="qualification"):
        engine.plan(replace(invocation, skill_options=MoveJointsOptions()), context)
    with pytest.raises(ValueError, match="outside the bound"):
        engine.plan(
            replace(
                invocation,
                goal=JointPositionGoal(JointOffsetTarget("passive_slide", 0.05)),
            ),
            context,
        )
