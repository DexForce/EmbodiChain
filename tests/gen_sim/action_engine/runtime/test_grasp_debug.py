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

from types import SimpleNamespace

import torch

from embodichain.gen_sim.action_engine.runtime.grasp_debug import (
    selected_coordinated_grasp_scene,
)
from embodichain.gen_sim.action_engine.runtime.models import (
    ActionOutcome,
    GroundedAction,
)
from embodichain.gen_sim.action_engine.runtime.state import ExecutionState
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    HeldObjectState,
    ObjectSemantics,
)


def _pose(x: float, y: float, z: float) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[:, :3, 3] = torch.tensor([x, y, z], dtype=torch.float32)
    return pose


def _outcome(*, success: bool) -> ActionOutcome:
    vertices = torch.tensor(
        [
            [-0.1, -0.05, -0.02],
            [0.1, -0.05, -0.02],
            [0.0, 0.05, -0.02],
            [0.0, 0.0, 0.02],
        ],
        dtype=torch.float32,
    )
    triangles = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        dtype=torch.int64,
    )
    semantics = ObjectSemantics(
        affordance=AntipodalAffordance(
            object_label="tray",
            mesh_vertices=vertices,
            mesh_triangles=triangles,
        ),
        geometry={"mesh_vertices": vertices, "mesh_triangles": triangles},
        entity_id="tray",
        label="tray",
    )
    live_object_pose = _pose(0.4, -0.2, 0.75)
    left_object_to_eef = _pose(0.0, -0.12, 0.01)
    right_object_to_eef = _pose(0.0, 0.13, 0.02)
    state = ExecutionState(
        last_qpos=torch.zeros(1, 8),
        held_objects={
            "physical_left_arm": HeldObjectState(
                semantics=semantics,
                object_to_eef=left_object_to_eef,
                grasp_xpos=torch.bmm(live_object_pose, left_object_to_eef),
            ),
            "physical_right_arm": HeldObjectState(
                semantics=semantics,
                object_to_eef=right_object_to_eef,
                grasp_xpos=torch.bmm(live_object_pose, right_object_to_eef),
            ),
        },
    )
    grounded = GroundedAction(
        action_class="CoordinatedPickment",
        arm="coordinated",
        control="coordinated",
        target=SimpleNamespace(),
        cfg={},
        object_pose=live_object_pose,
        object_uid="tray",
    )
    return ActionOutcome(
        trajectory=torch.zeros(1, 1, 8),
        success=torch.tensor([success]),
        next_state=state,
        grounded=grounded,
    )


def test_selected_scene_uses_final_e5_grasps_at_live_object_pose() -> None:
    outcome = _outcome(success=True)

    scene = selected_coordinated_grasp_scene(
        outcome,
        left_control_part="physical_left_arm",
        right_control_part="physical_right_arm",
    )

    assert scene is not None
    assert scene.object_label == "tray"
    assert torch.allclose(scene.object_pose, _pose(0.4, -0.2, 0.75)[0])
    assert torch.allclose(
        scene.left_grasp_pose[:3, 3],
        torch.tensor([0.4, -0.32, 0.76]),
    )
    assert torch.allclose(
        scene.right_grasp_pose[:3, 3],
        torch.tensor([0.4, -0.07, 0.77]),
    )


def test_selected_scene_skips_e5_without_valid_env_zero_grasps() -> None:
    assert (
        selected_coordinated_grasp_scene(
            _outcome(success=False),
            left_control_part="physical_left_arm",
            right_control_part="physical_right_arm",
        )
        is None
    )
