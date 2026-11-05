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

"""Deterministic observed frames for retained alignment lowerer source comparison."""

from __future__ import annotations
from types import SimpleNamespace as NS
from dataclasses import asdict
import torch
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    ObjectSemantics,
    HeldObjectState,
    TaskState,
    StateDelta,
    RobotObservation,
    SceneSnapshot,
    EntityState,
    PlanningContext,
    MoveHeldObjectOptions,
)
from embodichain.lab.task_program.semantics import RegisteredSemanticCall
from embodichain.lab.task_program.integrations.simulation.held_alignment import (
    HeldAlignmentRoute,
)
from embodichain.utils.math import axis_angle_to_rotation_matrix


def alignment_inputs(case):
    pose = torch.eye(4).repeat(1, 1, 1)
    pose[:, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor(case["rotation"], dtype=torch.float32)
    )[None]
    pose[:, :3, 3] = torch.tensor([[0.2, 0.1, 0.6]])
    eef = torch.eye(4)[None]
    eef[:, :3, :3] = axis_angle_to_rotation_matrix(
        torch.tensor([0.0, 1.5707963267948966, 0.0])
    )[None]
    eef[:, 2, 3] = 0.05
    affordance = AntipodalAffordance()
    affordance.mesh_vertices = torch.tensor([[-0.1, -0.1, -0.05], [0.1, 0.1, 0.3]])
    semantics = ObjectSemantics(
        affordance=affordance, geometry={}, label="item", entity_id="item"
    )
    held = HeldObjectState(
        semantics=semantics, object_to_eef=eef, grasp_xpos=pose @ eef
    )
    task = StateDelta(held_object_updates={"custom_state": held}).apply(
        TaskState.empty(1, "cpu"), torch.tensor([True])
    )
    context = PlanningContext(
        robot=RobotObservation(
            timestamp=0.0, qpos=torch.zeros(1, 2), qvel=torch.zeros(1, 2)
        ),
        task=task,
        scene=SceneSnapshot(
            timestamp=0.0, version=1, entities={"item": EntityState(pose)}
        ),
        env_ids=torch.tensor([0]),
    )
    endpoint = NS(
        task_state_key="custom_state", runtime_target=NS(control_part="custom_arm")
    )
    bound = NS(
        binding=NS(
            action_binding=NS(endpoint=lambda *args: endpoint),
            resources={
                "primary": NS(endpoints={"motion": endpoint, "grasp": endpoint})
            },
        )
    )
    root = torch.eye(4)[None]
    root[:, :2, 3] = torch.tensor([case["root_xy"]])
    reads = []

    def get_root(**kwargs):
        reads.append(kwargs)
        return root.clone()

    robot = NS(
        cfg=NS(solver_cfg={"custom_arm": NS(root_link_name="arm_origin")}),
        get_link_pose=get_root,
    )
    target = "current_object_pose" if case["position"] is None else "destination"
    route = HeldAlignmentRoute(
        "item",
        target,
        case["preserve"],
        case["direction_free"],
        (0.0, 0.0, 1.0),
        None if case["position"] is None else tuple(case["position"]),
    )
    call = RegisteredSemanticCall(
        call_id="gen_sim.align_held",
        arguments={
            "object": "item",
            "target": target,
            "preserve_yaw": case["preserve"],
            "axis_direction_free": case["direction_free"],
        },
        resources={"primary": "resource"},
    )
    return route, call, context, bound, robot, reads


def lowering_record(result, reads):
    effect = result.registered_effect
    return dict(
        targets=result.goal.object_target_pose.tolist(),
        world_yaw_free=result.goal.world_yaw_free,
        effect=(
            None
            if effect is None
            else dict(
                kind=effect.effect_kind.value,
                held=[
                    dict(
                        expectation_id=x.expectation_id,
                        relation=x.relation.value,
                        object_id=x.object_id,
                        slot_id=x.slot_id,
                    )
                    for x in effect.held_objects
                ],
            )
        ),
        root_reads=reads,
    )
