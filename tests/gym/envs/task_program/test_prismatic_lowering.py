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
"""Prismatic lowerers preserve physical source intent and resource-local handoff."""

from __future__ import annotations

from types import SimpleNamespace as NS
import pytest
import torch

from embodichain.lab.sim.atomic_actions import (
    JointPositionTarget,
    JointPositionCommand,
    SlideOptions,
    MoveEndEffectorOptions,
    MoveJointsOptions,
    SlideAffordance,
    ObjectSemantics,
    TaskState,
)
from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding
from embodichain.lab.task_program.integrations.simulation.prismatic_lowering import (
    PrismaticControlBinding,
    _SlideLowerer,
    _WithdrawLowerer,
    _ParkLowerer,
)


def _lowering_case():
    source = PrismaticBinding(
        "cabinet",
        "rail",
        "slider",
        "base",
        "/slider/grip",
        "a" * 64,
        1.0,
        (1.0, 0.0, 0.0),
        (0.0, 0.2),
    )
    declaration = PrismaticControlBinding("cabinet__slider", source)
    semantics = ObjectSemantics(
        SlideAffordance(
            mesh_vertices=torch.zeros(3, 3),
            mesh_triangles=torch.tensor([[0, 1, 2]]),
            translation_axis=torch.tensor([-1.0, 0.0, 0.0]),
        ),
        {},
        entity_id=declaration.entity_id,
    )
    link = torch.eye(4)[None]
    link[:, :3, :3] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    current = torch.eye(4)[None]
    current[:, :3, 3] = torch.tensor([0.2, 0.1, 0.5])
    art = NS(
        joint_names=["rail"],
        get_qpos=lambda: torch.tensor([[0.2]]),
        get_link_pose=lambda *args, **kwargs: link.clone(),
    )
    robot = NS(compute_fk=lambda **kwargs: current.clone())
    motion = NS(
        task_state_key="manipulator",
        require_target=lambda _: JointPositionTarget("custom_motion", (0, 1)),
    )
    grasp = NS(task_state_key="manipulator")
    hand = NS(
        commands={"open": JointPositionCommand(torch.zeros(2))},
        runtime_target=JointPositionTarget("custom_hand", (2, 3)),
    )
    bound = NS(
        binding=NS(
            action_binding=NS(
                endpoint=lambda role, kind: motion if kind == "motion" else grasp
            ),
            resources={"primary": NS(endpoints={"grasp": hand})},
        )
    )
    task = TaskState(1, "cpu")
    context = NS(
        robot=NS(qpos=torch.zeros(1, 4)),
        task=task,
        get_held_object=task.get_held_object,
        env_ids=torch.tensor([0]),
        batch_size=1,
    )
    bindings = {declaration.selection_key: (declaration, semantics, art)}
    call = NS(arguments={"object": "cabinet", "state": "open"})
    return bindings, robot, art, bound, context, call, grasp


def test_slide_lowers_absolute_joint_intent_without_geometry_payload_or_options():
    bindings, robot, art, bound, context, call, grasp = _lowering_case()
    result = _SlideLowerer(bindings, robot).lower(
        call, context=context, bound=bound, option_template=SlideOptions()
    )
    assert result.skill_options is None
    assert result.goal.joint_target.position == 0.2
    assert result.goal.joint_target.axis_sign == -1
    assert result.goal.joint_target.articulation_id == "cabinet"
    assert result.goal.target_pose.entity_id == "cabinet__slider"
    assert result.goal.semantics.geometry == {}
    grasp.task_state_key = "other"
    with pytest.raises(ValueError, match="share one state resource"):
        _SlideLowerer(bindings, robot).lower(
            call, context=context, bound=bound, option_template=SlideOptions()
        )


def test_withdrawal_uses_observed_joint_open_hand_and_world_axis():
    bindings, robot, art, bound, context, call, _ = _lowering_case()
    lowerer = _WithdrawLowerer(bindings, robot)
    result = lowerer.lower(
        call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
    )
    assert result.skill_options is None
    torch.testing.assert_close(
        result.goal.xpos[0, :, :3, 3],
        torch.tensor([[0.2, 0.1, 0.5], [0.2, 0.1, 0.62], [0.2, 0.2, 0.62]]),
    )
    context.robot.qpos[:, 2:] = 0.1
    with pytest.raises(ValueError, match="open posture"):
        lowerer.lower(
            call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
        )
    context.robot.qpos[:, 2:] = 0.0
    art.get_qpos = lambda: torch.tensor([[0.16]])
    with pytest.raises(ValueError, match="target was lost"):
        lowerer.lower(
            call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
        )


def test_articulation_park_keeps_named_command_and_no_implicit_free_motion():
    result = _ParkLowerer().lower(
        NS(arguments={}), context=None, bound=None, option_template=MoveJointsOptions()
    )
    assert result.goal.target == "park" and result.skill_options is None
    assert not MoveJointsOptions().require_free_motion
