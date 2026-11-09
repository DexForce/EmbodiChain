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

"""Configured control declarations produce typed goals without a World."""

from __future__ import annotations
from types import SimpleNamespace
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    ObjectSemantics,
    PressGoal,
    PressOptions,
    TwistGoal,
    TwistOptions,
    OpenDoorGoal,
    OpenDoorOptions,
    SceneEntityPose,
)
from embodichain.lab.sim.atomic_actions.affordance import (
    PressAffordance,
    TwistAffordance,
    OpenDoorAffordance,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.lab.task_program.integrations.configured import (
    _decode_registered_lowerer,
    _decode_action_options,
)


def _affordance(kind):
    if kind == "press":
        return PressAffordance()
    if kind == "twist":
        return TwistAffordance(grasp_position=(0.0, 0.0, 0.0))
    return OpenDoorAffordance(
        mesh_vertices=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        rotation_axis=torch.tensor([0.0, 0.0, 1.0]),
        axis_origin=(0.0, 0.0, 0.0),
        joint_name="hinge",
    )


@pytest.mark.parametrize(
    "kind,goal_type,options_type",
    [
        ("press", PressGoal, PressOptions),
        ("twist", TwistGoal, TwistOptions),
        ("open_door", OpenDoorGoal, OpenDoorOptions),
    ],
)
def test_configured_controls_have_unique_declarations_and_fresh_typed_lowerers(
    kind, goal_type, options_type
):
    factory = _decode_registered_lowerer(
        dict(kind=kind, object_affordances=[dict(object="control", affordance="aff")]),
        path="lowerer",
    )
    semantics = ObjectSemantics(
        affordance=_affordance(kind), geometry={}, entity_id="control"
    )
    registry = SimpleNamespace(object_semantics=lambda obj, affordance: semantics)
    robot = object()
    engine = SimpleNamespace(robot=robot)
    first = factory.create(
        simulation=None, robot=robot, scene_registry=registry, engine=engine
    )
    second = factory.create(
        simulation=None, robot=robot, scene_registry=registry, engine=engine
    )
    assert first is not second
    assert factory.call_id == first.call_id == f"simulation.{kind}"
    assert first.target_descriptor == factory.target_descriptor
    args = dict(object="control")
    if kind == "open_door":
        args["state"] = "open"
    call = RegisteredSemanticCall(call_id=factory.call_id, arguments=args)
    bound = SimpleNamespace(
        binding=SimpleNamespace(
            resources={
                "primary": SimpleNamespace(
                    endpoints={"motion": SimpleNamespace(task_state_key="arm")}
                )
            }
        )
    )
    context = SimpleNamespace(task=SimpleNamespace(get_held_object=lambda _: None))
    options = _decode_action_options(dict(kind=kind), path="options")
    assert type(options) is options_type
    result = first.lower(call, context=context, bound=bound, option_template=options)
    assert type(result.goal) is goal_type
    assert isinstance(result.goal.target_pose, SceneEntityPose)
    assert result.goal.semantics is semantics
    if kind == "open_door":
        assert result.goal.open_fraction == 1.0
    context.task.get_held_object = lambda _: object()
    with pytest.raises(ValueError, match="free"):
        first.lower(call, context=context, bound=bound, option_template=options)


@pytest.mark.parametrize("kind", ["press", "twist", "open_door"])
def test_control_bindings_reject_unknown_fields_and_duplicate_objects(kind):
    with pytest.raises(ValueError):
        _decode_registered_lowerer(
            dict(kind=kind, object_affordances=[], class_type="unsafe"), path="lowerer"
        )
    with pytest.raises(ValueError):
        _decode_registered_lowerer(
            dict(
                kind=kind,
                object_affordances=[
                    dict(object="a", affordance="x"),
                    dict(object="a", affordance="y"),
                ],
            ),
            path="lowerer",
        )
