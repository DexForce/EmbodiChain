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

"""Shared planning ports and validation without the migrated Pick/Place code."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    EntityState,
    SceneSnapshot,
    ObjectSemantics,
    PressAffordance,
    PressGoal,
    PressOptions,
    inspect_manipulation_plan,
)
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
from embodichain.lab.sim.atomic_actions.primitives import _manipulation_checks
from test_actions import _context, _motion_generator, _torch_interpolation
import test_actions


@pytest.mark.parametrize("foreign", [False, True])
def test_engine_borrows_same_robot_queries_without_acquisition(foreign):
    generator = _motion_generator()
    calls = []
    query = SimpleNamespace(
        robot=object() if foreign else generator.robot,
        inspect=lambda *args, **kwargs: calls.append("inspect"),
    )
    if foreign:
        with pytest.raises(ValueError, match="exact robot"):
            AtomicActionEngine(generator, free_motion_query=query)
    else:
        engine = AtomicActionEngine(generator, free_motion_query=query)
        action = engine.actions["press"]
        assert action.planning_services.free_motion_query is query
        assert action.planning_services.joint_prefix_planner is None
        assert action.planning_services.place_continuation_provider is None
    assert calls == []


@pytest.mark.parametrize("damage", [None, "velocity", "command"])
def test_public_validation_checks_real_unmodified_press_commands(
    monkeypatch, tmp_path, damage
):
    # Reuse the ordinary main action fixture, not a Pick/Place migration fixture.
    monkeypatch.setattr(test_actions, "NUM_ENVS", 1)
    generator = _motion_generator()
    robot = generator.robot
    path = tmp_path / "limits.urdf"
    path.write_text(
        '<robot name="limits">'
        + "".join(
            f'<joint name="{name}" type="prismatic"><limit velocity="1"/></joint>'
            for name in robot.joint_names
        )
        + "</robot>"
    )
    robot.cfg.fpath = str(path)
    robot.get_qpos_limits.side_effect = (
        lambda name=None, **kwargs: torch.tensor([[[-2.0, 2.0]]])
        .expand(1, len(robot.get_joint_ids(name=name)), 2)
        .clone()
    )
    engine = AtomicActionEngine(
        generator,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(2), grasp=torch.zeros(2)
            )
        },
    )
    invocation = engine.make_invocation(
        "press",
        PressGoal(
            ObjectSemantics(
                PressAffordance(
                    press_axis=torch.tensor([0.0, 0.0, -1.0]),
                    press_position=(0.0, 0.0, 0.0),
                ),
                {},
                "object",
            ),
            torch.eye(4)[None],
        ),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        skill_options=PressOptions(),
    )
    context = replace(
        _context(),
        scene=SceneSnapshot(0.0, 0, {"object": EntityState(torch.eye(4)[None])}),
    )
    plan = engine.plan(invocation, context=context)
    assert plan.plan_success.tolist() == [True]
    seen = []

    def robot_collision(owner, positions, *, control_part):
        assert owner is robot
        seen.append(positions.clone())
        random.random()
        np.random.rand()
        torch.rand(())
        return MotionCollisionReport(
            True, frames=len(positions), scope="discrete_robot_meshes"
        )

    monkeypatch.setattr(
        _manipulation_checks, "inspect_robot_collision", robot_collision
    )
    if damage == "velocity":
        plan.joint_trajectory.positions[0, 1, 0] = 1e6
    elif damage == "command":
        plan.commands.frames[-1].commands[0].payload.positions[0, 0] += 0.125
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = inspect_manipulation_plan(
        robot, engine.actions["press"].resolve_request(invocation), context, plan
    )
    assert result["valid"] is (damage is None), result
    if damage == "velocity":
        assert result["reason"] == "final_velocity_rejected"
    elif damage == "command":
        assert "command positions" in result["unavailable"]
    assert len(seen) == (1 if damage is None else 0)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.get_rng_state(), before[2])


def test_shared_grasp_selection_keeps_first_feasible_strategy_per_row():
    from embodichain.compute.geometry.gripper import GraspApproach
    from embodichain.lab.sim.atomic_actions import (
        AntipodalAffordance,
        AffordancePoseCandidates,
        AffordanceSamplingContext,
    )
    from embodichain.lab.sim.atomic_actions.primitives._grasp_sampling import (
        select_grasp,
    )

    affordance = AntipodalAffordance()
    calls = []
    poses = torch.eye(4).repeat(2, 1, 1)
    first = torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)
    second = torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1)

    def candidates(generator, pose, direction, **kwargs):
        index = len(calls)
        calls.append(direction.clone())
        value = pose[:, None].clone()
        value[:, :, 0, 3] = index
        return AffordancePoseCandidates(
            value,
            torch.zeros(2, 1),
            torch.tensor([[True], [index == 1]]),
        )

    affordance.get_grasp_candidates = candidates
    sample, directions = select_grasp(
        affordance,
        SimpleNamespace(),
        poses,
        (
            GraspApproach(first, None, torch.zeros(2, dtype=torch.bool)),
            GraspApproach(second, None, torch.zeros(2, dtype=torch.bool)),
        ),
        SimpleNamespace(
            affordance_sampling=AffordanceSamplingContext(),
            env_ids=torch.tensor([3, 7]),
        ),
        "shared-handover-grasp",
    )
    assert len(calls) == 2
    assert sample.success.tolist() == [True, True]
    assert sample.metadata["adaptive_grasp"]["strategy_indices"] == [0, 1]
    assert torch.equal(directions, torch.stack((first[0], second[1])))


def test_shared_observed_axis_uses_current_pose_without_mutating_local_axis():
    from embodichain.lab.sim.atomic_actions.primitives._grasp_sampling import (
        observed_axis,
    )

    local = torch.tensor([1.0, 0.0, 0.0])
    pose = torch.eye(4)[None]
    pose[0, :2, :2] = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    assert torch.equal(
        observed_axis(SimpleNamespace(internal_axis=local), pose),
        torch.tensor([[0.0, 1.0, 0.0]]),
    )
    assert local.tolist() == [1.0, 0.0, 0.0]
