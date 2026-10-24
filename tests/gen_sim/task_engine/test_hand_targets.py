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

from __future__ import annotations

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    ActionControlOverrides,
    AtomicActionEngine,
    AxisAlignAffordance,
    ControlPartCommandProfile,
    HeldObjectPoseGoal,
    HeldObjectState,
    JointPositionCommand,
    MotionPolicy,
    MoveHeldObject,
    MoveHeldObjectOptions,
    ObjectSemantics,
    Place,
    PlaceGoal,
    PlaceOptions,
    PlanningContext,
    Pour,
    PourGoal,
    PourOptions,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.gen_sim.task_engine._task_program.actions import (
    GenSimHandOver,
    GenSimMoveHeldObject,
)
from embodichain.gen_sim.task_engine._task_program.held_command import (
    bind_observed_alignment_calls,
)


@pytest.mark.parametrize(
    "kind,source",
    [
        ("legacy_alignment", "observed"),
        ("transport", "grasp_command"),
        ("gen_transport", "grasp_command"),
        ("place", "grasp_command"),
        ("pour", "grasp_command"),
    ],
)
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_source_selection_reaches_real_command_payloads_without_mutating_observation(
    source, kind, partial, override
):
    robot = Mock()
    robot.device, robot.dof, robot.num_instances = torch.device("cpu"), 4, 2
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.joint_names = ["a0", "a1", "hand", "passive"]
    robot.get_joint_ids.side_effect = lambda name=None: {
        None: [0, 1, 2, 3],
        "arm": [0, 1],
        "hand": [2],
    }[name]
    robot.get_qpos.return_value = torch.zeros(2, 4)
    robot.compute_fk.return_value = torch.eye(4).repeat(2, 1, 1)
    robot.compute_batch_fk.side_effect = (
        lambda qpos, **kw: torch.eye(4)
        .expand(qpos.shape[0], qpos.shape[1], 4, 4)
        .clone()
    )
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = [], [], []
    generator = object.__new__(MotionGenerator)
    generator.robot, generator.device = robot, robot.device
    generator.planner = NS(cfg=NS(planner_type="stub"), collision_world_info=None)
    generator.direct_transport_planning = True
    mask = torch.tensor([True, not partial])
    positions = torch.zeros(2, 4, 2)
    positions[:, :, 0] = torch.tensor([0.0, 0.01, 0.02, 0.0])
    dt = torch.full((2, 4), 0.04)
    dt[:, 0] = 0
    generator.generate = Mock(
        return_value=PlanResult(success=mask, positions=positions, dt=dt)
    )
    action_type, options, goal = {
        "transport": (
            MoveHeldObject,
            MoveHeldObjectOptions(),
            HeldObjectPoseGoal(torch.eye(4)),
        ),
        "place": (
            Place,
            PlaceOptions(hand_interp_steps=3),
            PlaceGoal(torch.eye(4)),
        ),
        "pour": (Pour, PourOptions(dwell_duration=0), PourGoal()),
        "legacy_alignment": (
            GenSimMoveHeldObject,
            MoveHeldObjectOptions(),
            HeldObjectPoseGoal(torch.eye(4)),
        ),
        "gen_transport": (
            GenSimMoveHeldObject,
            MoveHeldObjectOptions(),
            HeldObjectPoseGoal(torch.eye(4)),
        ),
    }[kind]
    action = action_type()
    if isinstance(action, GenSimMoveHeldObject):
        action.observed_hand_invocations = frozenset({"legacy_alignment"})
    engine = AtomicActionEngine(
        generator,
        load_builtins=False,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.zeros(1), grasp=torch.ones(1)
            )
        },
    )
    engine.register(action)
    identity = torch.eye(4).repeat(2, 1, 1)
    semantics = ObjectSemantics(
        affordance=AxisAlignAffordance(internal_axis=torch.tensor([0.0, 1.0, 0.0])),
        geometry={},
        entity_id="item",
    )
    held = HeldObjectState(
        semantics=semantics, object_to_eef=identity, grasp_xpos=identity
    )
    observed = torch.tensor([[0.0, 0.0, 0.4, 0.7]]).repeat(2, 1)
    context = PlanningContext(
        robot=RobotObservation(
            timestamp=0.0, qpos=observed, qvel=torch.zeros_like(observed)
        ),
        task=TaskState(2, "cpu", held_objects={"arm": held}),
        scene=SceneSnapshot.empty(),
        env_ids=torch.tensor([0, 1]),
        control_dt=0.04,
    )
    before = context.robot.qpos.clone()
    binding = engine.bind_control_parts(
        action.skill_id, {"primary": {"motion": "arm", "grasp": "hand"}}
    )
    request = action.resolve_request(
        ActionInvocation(
            skill_id=action.skill_id,
            invocation_id=kind,
            goal=goal,
            binding=binding,
            motion_policy=MotionPolicy(sample_count=8),
            skill_options=options,
            control_overrides=ActionControlOverrides(
                endpoints=(
                    {
                        "primary": {
                            "grasp": {
                                "grasp": JointPositionCommand(torch.tensor([0.8]))
                            }
                        }
                    }
                    if override
                    else {}
                )
            ),
        )
    )
    plan = action._plan(request, context)
    path = plan.joint_trajectory.positions
    assert torch.equal(context.robot.qpos, before)
    assert torch.equal(plan.plan_success, mask)
    expected = (0.8 if override else 1.0) if source == "grasp_command" else 0.4
    hold_stop = (
        next(s.stop for s in plan.segments if s.name == "approach")
        if kind == "place"
        else path.shape[1]
    )
    torch.testing.assert_close(
        path[0, :hold_stop, 2], torch.full((hold_stop,), expected)
    )
    if kind == "place":
        assert path[0, -1, 2].item() == 0.0
        release = next(s for s in plan.segments if s.name == "release")
        assert path[0, release.start, 2].item() == pytest.approx(expected)
    for i, frame in enumerate(plan.commands.frames):
        payload = next(
            c.payload for c in frame.commands if c.target.control_part == "hand"
        )
        torch.testing.assert_close(payload.positions, path[:, i, 2:3])
        torch.testing.assert_close(
            payload.velocities, plan.joint_trajectory.velocities[:, i, 2:3]
        )
        if i < hold_stop:
            assert payload.velocities[0].item() == 0.0
        assert torch.equal(frame.active_mask, mask)
    if partial:
        torch.testing.assert_close(path[1], before[1, None].expand(path.shape[1], -1))
    torch.testing.assert_close(path[:, :, 3], torch.full(path.shape[:2], 0.7))


@pytest.mark.parametrize(
    "contract,expected",
    [("lab.motion_limits/v1", ("p/s0:0",)), ("gen_sim.sampled_velocity/v1", ())],
)
def test_only_legacy_alignment_occurrences_keep_observed_hands(contract, expected):
    from embodichain.lab.task_program.semantics import RegisteredSemanticCall

    segments = [
        NS(
            name=f"segment{i}",
            segment_id=f"s{i}",
            calls=[
                NS(segment_call_index=0, call=RegisteredSemanticCall(call_id=call_id))
            ],
        )
        for i, call_id in enumerate(
            ("gen_sim.align_held", "simulation.move_held_object")
        )
    ]
    program = NS(program_id="p", iter_segments=lambda: iter(segments))
    assert bind_observed_alignment_calls(program, contract) == expected


def test_gensim_handover_retains_resolved_grasp_command():
    command = torch.tensor([[0.8]])
    resources = NS(first=NS(hand_grasp_qpos=command))
    assert GenSimHandOver()._existing_hold_hand_qpos(None, None, resources) is command
