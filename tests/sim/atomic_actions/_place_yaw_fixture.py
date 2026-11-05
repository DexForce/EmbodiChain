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

"""Real native-mesh ports and controlled yaw build/inspection boundaries."""

from __future__ import annotations
from dataclasses import dataclass, field, replace
from types import SimpleNamespace as NS
import random
import numpy as np
import torch
import trimesh
from embodichain.lab.sim.atomic_actions import (
    TaskState,
    PlaceSceneInputs,
    PlaceInsideLanding,
    PlaceOptions,
    MotionPolicy,
    EntityState,
    SceneSnapshot,
    PlaceGoal,
    SceneEntityPose,
)
from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.scene.planar_observation import NativePlanarPlacement
from embodichain.lab.sim.scene.mesh_snapshot import SceneMeshSnapshot
from ._ordinary_pick_fixture import place_fixture


@dataclass
class YawDiagnostics:
    metadata: dict = field(default_factory=dict)
    failure: object = None


@dataclass
class YawPlan:
    plan_success: torch.Tensor
    diagnostics: YawDiagnostics = field(default_factory=YawDiagnostics)
    endpoint: torch.Tensor | None = None


@dataclass
class YawRequest:
    goal: PlaceGoal
    skill_options: PlaceOptions
    motion_policy: MotionPolicy
    binding: object
    invocation_id: str = "release"


def yaw_fixture(directory, mode):
    gen, context, _, _, _ = place_fixture(directory)
    meshes = {
        "object": trimesh.creation.box((0.4, 0.1, 0.1)),
        "parent": trimesh.creation.box((0.25, 0.6, 0.02)),
    }
    meshes["parent"].apply_translation([0.0, 0.0, -0.01])
    objects = {
        key: NS(
            get_vertices=lambda scale, mesh=mesh: torch.tensor(
                mesh.vertices, dtype=torch.float32
            )[None],
            get_triangles=lambda mesh=mesh: torch.tensor(mesh.faces, dtype=torch.long)[
                None
            ],
        )
        for key, mesh in meshes.items()
    }
    sim = NS(get_rigid_object=lambda uid: objects[uid])
    native = NativePlanarPlacement(sim, "object", "parent", batch_size=1)
    target = torch.eye(4)[None]
    target[:, 2, 3] = 0.17
    held = torch.eye(4)[None]
    held[:, :3, 3] = torch.tensor([0.03, -0.02, 0.06])
    held[:, :3, :3] = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.93969262, -0.34202014], [0.0, 0.34202014, 0.93969262]]
    )
    attachment = replace(
        context.get_held_object("arm"), object_to_eef=held, grasp_xpos=target @ held
    )
    context = replace(
        context,
        task=TaskState(1, "cpu", held_objects={"arm": attachment}),
        scene=SceneSnapshot(
            0.0,
            0,
            {"target": EntityState(target), "parent": EntityState(torch.eye(4)[None])},
        ),
        is_preview=mode == "preview",
    )
    endpoint = NS(task_state_key="arm")
    request = YawRequest(
        PlaceGoal(SceneEntityPose("target", relative_pose=held)),
        PlaceOptions(preserve_current_object_orientation=True, lift_height=0.05),
        MotionPolicy(dynamic_collision_mode="off"),
        NS(endpoint=lambda *args: endpoint),
    )
    inputs = PlaceSceneInputs(
        geometry=SceneMeshSnapshot(
            {key: obj.get_vertices(True) for key, obj in objects.items()}
        ),
        floor_entity_id="parent",
        inside_landing=PlaceInsideLanding(
            geometry=native,
            subject_id="object",
            container_id="target",
            parent_id="parent",
            release_clearance=0.12,
        ),
    )
    action = NS(
        robot=gen.robot,
        motion_generator=gen,
        planning_services=NS(place_scene_provider=NS(prepare=lambda *args: inputs)),
    )
    first = YawPlan(torch.tensor([mode == "nominal"]))
    calls = []

    def build(selected, current):
        pose = resolve_pose_goal(selected.goal.xpos, current, name="yaw_test")
        calls.append(
            {
                "target": pose.tolist(),
                "draws": [
                    float(torch.rand(())),
                    float(np.random.rand()),
                    random.random(),
                ],
            }
        )
        if mode == "exception":
            raise ValueError("controlled planner unavailable")
        return YawPlan(torch.tensor([mode != "rejected"]), endpoint=pose)

    def inspect(action, selected, current, plan, *args):
        valid = bool(plan.plan_success.all()) and (mode != "late" or len(calls) >= 2)
        return {
            "valid": valid,
            "physical_evidence": False,
            "scope": "controlled_emitted_oracle",
        }

    return action, request, context, first, build, inspect, calls, sim, native
