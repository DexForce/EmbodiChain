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

"""Direct shared Pour variants against frozen B0-F1 complete plan traces."""

from __future__ import annotations

import json
import importlib.util
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import numpy as np

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ActionControlOverrides,
    ControlPartCommandProfile,
    JawPourGoal,
    JointPositionCommand,
    MotionPolicy,
    PourOptions,
    PourGoal,
    RetainedPourGoal,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners.utils import (
    PlanResult,
    RejectedMotionCandidate,
)
from ._pour_fixture import hand_cfg, pour_fixture

_REFERENCE = json.loads(
    (Path(__file__).parent / "data/pour_plans_source_reference.json").read_text()
)


def _inputs(tmp_path, case, *, hand_calibration=True):
    robot, context, vertices, receiving, returned = pour_fixture(tmp_path)
    calls = []

    def generate(targets, *, options):
        calls.append(
            {
                "targets": [state.xpos.tolist() for state in targets],
                "sample_count": options.sample_count,
                "preserve_cartesian_samples": options.preserve_cartesian_samples,
            }
        )
        positions = (
            torch.linspace(0.0, 0.006, options.sample_count)
            .reshape(1, -1, 1)
            .repeat(1, 1, 2)
        )
        dt = torch.full((1, options.sample_count), 0.04)
        dt[:, 0] = 0
        return PlanResult(success=torch.tensor([True]), positions=positions, dt=dt)

    generator = object.__new__(MotionGenerator)
    generator.robot, generator.device = robot, robot.device
    generator.planner = NS(cfg=NS(planner_type="oracle"), collision_world_info=None)
    generator._generate_once = generate
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(2), grasp=torch.full((2,), 0.1)
            )
        },
        hand_geometry_cfgs={"hand": hand_cfg()} if hand_calibration else None,
    )
    goal = (
        RetainedPourGoal(
            source_entity_id="object",
            receiver_entity_id="receiver",
            source_vertices=vertices,
            receiver_vertices=receiving,
            return_pose=returned,
            gravity_world=(0.0, 0.0, -9.81),
            proxy_position_tolerance_m=0.06,
        )
        if case["retained"]
        else JawPourGoal(receiver_entity_id="receiver" if case["receiver"] else None)
    )
    invocation = engine.make_invocation(
        "pour",
        goal,
        control_parts={"primary": {"motion": "left_arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=case["count"], dynamic_collision_mode="off"
        ),
        skill_options=PourOptions(rotate_angle=case["angle"]),
        control_overrides=ActionControlOverrides(
            endpoints=(
                {
                    "primary": {
                        "grasp": {"grasp": JointPositionCommand(torch.full((2,), 0.2))}
                    }
                }
                if case["override"]
                else {}
            )
        ),
        invocation_id="pour",
    )
    return engine, invocation, context, calls


@pytest.mark.parametrize("case", _REFERENCE["cases"], ids=lambda case: case["name"])
def test_shared_pour_variant_matches_frozen_source_engine_plan(tmp_path, case):
    engine, invocation, context, calls = _inputs(tmp_path, case)
    torch.manual_seed(317)
    before = torch.get_rng_state().clone()
    plan = engine.plan(invocation, context)
    actual = {
        name: case[name]
        for name in ("name", "retained", "count", "receiver", "override", "angle")
    }
    actual.update(
        motion_calls=calls,
        positions=plan.joint_trajectory.positions.tolist(),
        dt=plan.joint_trajectory.dt.tolist(),
        success=plan.plan_success.tolist(),
        segments=[
            {"name": s.name, "start": s.start, "stop": s.stop} for s in plan.segments
        ],
        active_masks=[frame.active_mask.tolist() for frame in plan.commands.frames],
        scene_dependencies=list(plan.scene_dependencies),
        effect_resources=list(plan.expected_effects.held_object_updates),
        effect_transform={
            key: value.object_to_eef.tolist()
            for key, value in plan.expected_effects.held_object_updates.items()
        },
        metadata=dict(plan.diagnostics.metadata),
        rng_unchanged=torch.equal(before, torch.get_rng_state()),
    )
    assert actual == case


def test_pour_receiver_is_resolved_per_context_without_adding_monitors(tmp_path):
    case = _REFERENCE["cases"][0]
    engine, invocation, context, calls = _inputs(tmp_path, case)
    first = engine.plan(invocation, context)
    receiver = context.scene.entities["receiver"]
    pose = receiver.pose.clone()
    pose[:, 0, 3] = 0.1
    latest = replace(
        context,
        robot=replace(context.robot, timestamp=4.0),
        scene=replace(
            context.scene,
            timestamp=4.0,
            version=5,
            entities={
                **context.scene.entities,
                "receiver": replace(receiver, pose=pose),
            },
        ),
    )
    second = engine.plan(invocation, latest)
    assert first.scene_dependencies == second.scene_dependencies == ()
    torch.testing.assert_close(
        torch.tensor(first.diagnostics.metadata["tilt_axis_local"]),
        -torch.tensor(second.diagnostics.metadata["tilt_axis_local"]),
    )
    assert calls[0]["targets"] != calls[1]["targets"]


def test_ordered_pad_pour_requires_explicit_calibration(tmp_path):
    engine, invocation, context, calls = _inputs(
        tmp_path, _REFERENCE["cases"][0], hand_calibration=False
    )
    with pytest.raises(KeyError, match="No hand geometry calibration"):
        engine.plan(invocation, context)
    assert calls == []


@pytest.mark.parametrize("kind", ["jaw", "retained", "default"])
def test_pour_nominal_entry_rejects_before_external_or_native_planning(tmp_path, kind):
    case = _REFERENCE["cases"][-1 if kind == "retained" else 0]
    engine, invocation, context, calls = _inputs(tmp_path, case)
    if kind == "default":
        invocation = replace(invocation, goal=PourGoal())

    class Source:
        def instantiate(self, request, current):
            raise AssertionError("unsupported Pour must not invoke its nominal source")

    with pytest.raises(ValueError, match="does not support external nominal"):
        engine.plan(invocation, context, nominal_source=Source())
    assert calls == []


def test_retained_goal_owns_geometry_and_return_target_before_invocation(tmp_path):
    _, _, vertices, receiver, returned = pour_fixture(tmp_path)
    goal = RetainedPourGoal(
        source_entity_id="object",
        receiver_entity_id="receiver",
        source_vertices=vertices,
        receiver_vertices=receiver,
        return_pose=returned,
        gravity_world=(0.0, 0.0, -9.81),
    )
    expected = (vertices.clone(), receiver.clone(), returned.clone())
    vertices.fill_(9)
    receiver.fill_(9)
    returned.fill_(9)
    for actual, value in zip(
        (goal.source_vertices, goal.receiver_vertices, goal.return_pose),
        expected,
        strict=True,
    ):
        torch.testing.assert_close(actual, value)


@pytest.mark.parametrize("outcome", ["failed", "accepted", "unavailable"])
def test_retained_pour_retries_rejected_candidate_not_the_executable_hold(
    tmp_path, monkeypatch, outcome
):
    if outcome == "accepted" and importlib.util.find_spec("newton") is None:
        pytest.skip("actual robot-proxy qualification requires Newton")
    from embodichain.lab.sim.atomic_actions.primitives import _pour_arc

    case = _REFERENCE["cases"][-1]
    engine, invocation, context, _ = _inputs(tmp_path, case)
    original_generate = engine.motion_generator._generate_once
    results, recovery_calls, accepted = [], [], []

    def failed_generate(targets, *, options):
        good = original_generate(targets, options=options)
        accepted.append(good)
        raw = good.positions.clone()
        raw[:, 1, 0] = 0.9  # .9 / .04 exceeds the authored/runtime 20-unit speed.
        result = PlanResult(
            success=torch.tensor([False]),
            positions=torch.zeros_like(raw),
            dt=good.dt,
            rejected_candidate=RejectedMotionCandidate(
                raw, good.dt, torch.tensor([True])
            ),
        )
        results.append(result)
        return result

    def recover(generator, targets, options, **kwargs):
        recovery_calls.append((len(targets), options.start_qpos.clone()))
        return None if outcome == "failed" else accepted[0]

    def unavailable(*args, **kwargs):
        random.random()
        np.random.random()
        torch.rand(1)
        raise ImportError("Newton collision queries are unavailable")

    engine.motion_generator._generate_once = failed_generate
    monkeypatch.setattr(_pour_arc, "resolve_continuous_ik_path", recover)
    if outcome == "unavailable":
        monkeypatch.setattr(_pour_arc, "RobotMeshCollisionSnapshot", unavailable)
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    plan = engine.plan(invocation, context)
    assert len(recovery_calls) == 1 and recovery_calls[0][0] == case["count"] - 1
    assert results[0].positions.count_nonzero() == 0
    assert results[0].rejected_candidate.positions[0, 1, 0].item() == pytest.approx(0.9)
    torch.testing.assert_close(
        results[0].rejected_candidate.dt, torch.tensor(case["dt"])
    )
    if outcome == "accepted":
        assert plan.plan_success.all()
        assert (
            plan.diagnostics.metadata["pour_continuous_ik"]["robot_proxy"]["scope"]
            == "discrete_robot_meshes"
        )
        torch.testing.assert_close(plan.joint_trajectory.dt, torch.tensor(case["dt"]))
    else:
        assert not plan.plan_success.any()
        assert plan.joint_trajectory.waypoint_count == 0
        assert plan.commands.frame_count == 0
        assert all(not frame.active_mask.any() for frame in plan.commands.frames)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
