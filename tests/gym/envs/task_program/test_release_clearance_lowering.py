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

"""Released-hand nominal targets preserve exact source frames and free-hand gate."""

from __future__ import annotations
from types import SimpleNamespace as NS
from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from embodichain.lab.sim.atomic_actions import (
    MoveEndEffectorOptions,
    RobotObservation,
    TaskState,
    SceneSnapshot,
    EntityState,
    PlanningContext,
    StateDelta,
    AntipodalAffordance,
    ObjectSemantics,
    HeldObjectState,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.lab.task_program.integrations.configured import (
    _decode_registered_lowerer,
)
from embodichain.lab.task_program.integrations.simulation.release_clearance import (
    _ClearReleasedLowerer,
    ReleasedHandRoute,
)
from embodichain.toolkits.graspkit.hand_envelope import NominalHandEnvelope
from embodichain.utils.math import axis_angle_to_rotation_matrix


def inputs(case):
    current = torch.eye(4)[None]
    current[:, :3, 3] = torch.tensor([[0.2, 0.1, 0.5]])
    current[:, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor(case["tcp_rotation"])
    )[None]
    root = torch.eye(4)[None]
    root[:, :2, 3] = torch.tensor([case["root_xy"]])
    obj = torch.eye(4)[None]
    obj[:, :3, 3] = torch.tensor([[0.2, 0.1, case["object_z"]]])
    vertices = torch.tensor([[-0.1, -0.05, -0.02], [0.1, 0.05, 0.2]])
    reads = []

    def fk(**kw):
        reads.append(("fk", kw["name"]))
        return current.clone()

    def origin(**kw):
        reads.append(("root", kw["link_name"]))
        return root.clone()

    robot = NS(
        cfg=NS(solver_cfg={"custom_arm": NS(root_link_name="root_link")}),
        compute_fk=fk,
        get_link_pose=origin,
    )
    endpoint = NS(
        task_state_key="state",
        runtime_target=NS(control_part="custom_arm", joint_ids=(0, 1)),
    )
    bound = NS(
        binding=NS(
            resources={
                "primary": NS(
                    endpoints={
                        "motion": endpoint,
                        "grasp": NS(runtime_target=NS(target_id="custom_hand")),
                    }
                )
            }
        )
    )
    context = PlanningContext(
        robot=RobotObservation(
            timestamp=0.0, qpos=torch.zeros(1, 2), qvel=torch.zeros(1, 2)
        ),
        task=TaskState.empty(1, "cpu"),
        scene=SceneSnapshot(
            timestamp=0.0, version=0, entities={"item": EntityState(obj)}
        ),
        env_ids=torch.tensor([0]),
    )
    envelope = NominalHandEnvelope(0.13, 0.03, 0.01, 0.08, 0.003)
    model = NS(model_id="robotiq_arg2f_140", max_opening_width=0.14)
    call = RegisteredSemanticCall(
        call_id="gen_sim.clear_released",
        arguments={"object": "item", "target": "retreat"},
    )
    return robot, context, bound, vertices, envelope, model, call, reads


_REFERENCE = json.loads(
    (Path(__file__).parent / "data/released_hand_lowering_b0.json").read_text()
)


@pytest.mark.parametrize("case", _REFERENCE["cases"])
def test_released_hand_lowering_matches_source_targets_and_native_read_order(case):
    robot, context, bound, vertices, envelope, model, call, reads = inputs(
        case["inputs"]
    )
    lowerer = _ClearReleasedLowerer(
        (ReleasedHandRoute("item", "retreat", case["inputs"]["minimum_height"]),),
        robot,
        retreat_distance=0.15,
        object_vertices={"item": vertices},
        gripper_models={"custom_hand": model},
        hand_envelopes={"custom_hand": envelope},
    )
    result = lowerer.lower(
        call, context=context, bound=bound, option_template=MoveEndEffectorOptions()
    )
    assert result.goal.xpos.tolist() == case["targets"]
    assert [list(row) for row in reads] == case["reads"]
    assert result.registered_effect is None and lowerer.preserves_symbolic_state


def test_clearance_decode_owns_explicit_model_reserves_and_rejects_held_resource():
    envelope = dict(
        finger_length=0.13,
        finger_width=0.03,
        finger_thickness=0.01,
        palm_depth=0.08,
        opening_margin=0.003,
    )
    cfg = dict(
        kind="clear_released",
        routes=[dict(object_id="item", target_id="retreat", minimum_height=0.8)],
        hands=[
            dict(
                grasp_target_id="custom_hand",
                model_id="robotiq_arg2f_140",
                envelope=envelope,
            )
        ],
    )
    factory = _decode_registered_lowerer(cfg, path="lowerer")
    envelope["finger_length"] = 99
    assert (
        factory.hands[0].envelope.finger_length == 0.13
        and factory.retreat_distance == 0.15
    )
    with pytest.raises(ValueError, match="unsupported fields"):
        _decode_registered_lowerer({**cfg, "planner": "anything"}, path="lowerer")
    robot, context, bound, vertices, reserve, model, call, reads = inputs(
        _REFERENCE["cases"][0]["inputs"]
    )
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(), geometry={}, label="item", entity_id="item"
    )
    held = HeldObjectState(
        semantics=semantics,
        object_to_eef=torch.eye(4)[None],
        grasp_xpos=torch.eye(4)[None],
    )
    task = StateDelta(held_object_updates={"state": held}).apply(
        context.task, torch.tensor([True])
    )
    lowerer = _ClearReleasedLowerer(
        factory.routes,
        robot,
        retreat_distance=0.15,
        object_vertices={"item": vertices},
        gripper_models={"custom_hand": model},
        hand_envelopes={"custom_hand": reserve},
    )
    # B0 used removed HeldObjectState.active_mask here. The public task mask
    # preserves intended rejection instead of raising AttributeError (source fix).
    with pytest.raises(ValueError, match="held object"):
        lowerer.lower(
            call,
            context=replace(context, task=task),
            bound=bound,
            option_template=MoveEndEffectorOptions(),
        )
    assert reads == []
