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

"""CPU geometry checks for intentional grasp contacts and placed-object safety."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation


def _scene(link_name: str, contact_steps: tuple[int, ...]):
    pytest.importorskip("fcl")
    import torch
    from scripts.tools.grasp_assemble._geometry import TCP_Z
    from scripts.tools.grasp_assemble._robot import RobotSession

    initial, placed = np.eye(4), np.eye(4)
    initial[2, 3] = placed[2, 3] = 0.05
    placed[0, 3] = 0.5
    mug_poses = np.repeat(initial[None], 6, axis=0)
    mug_poses[2, 2, 3] = 0.25
    mug_poses[3:] = placed
    palm_poses = mug_poses.copy()
    palm_poses[:, 2, 3] -= TCP_Z
    link_poses = np.repeat(np.eye(4)[None], 6, axis=0)
    link_poses[:, :3, 3] = [2.0, 0.0, 0.25]
    for step in contact_steps:
        link_poses[step] = mug_poses[step]
    link_mesh = trimesh.creation.box([0.05, 0.05, 0.05])
    mug_mesh = trimesh.creation.box([0.04, 0.04, 0.06])
    rack_mesh = trimesh.creation.box([0.1, 0.1, 0.1])
    rack_mesh.apply_translation([5, 0, 0.05])
    transforms = {
        "gripper_base_link_1": SimpleNamespace(
            get_matrix=lambda: torch.tensor(palm_poses)
        ),
        link_name: SimpleNamespace(get_matrix=lambda: torch.tensor(link_poses)),
    }
    session = RobotSession.__new__(RobotSession)
    session.robot = SimpleNamespace(
        joint_names=["joint"],
        link_names=[link_name],
        pk_chain=SimpleNamespace(forward_kinematics=lambda _: transforms),
        get_local_pose=lambda **_: torch.eye(4)[None],
        get_link_vert_face=lambda _: (
            torch.tensor(link_mesh.vertices.copy()),
            torch.tensor(link_mesh.faces.copy()),
        ),
    )
    session.trajectory = {
        "positions": torch.zeros(1, 6, 1),
        "phases": {
            "close": {"start": 1, "stop": 3},
            "lift": {"start": 2, "stop": 4},
            "release": {"start": 3, "stop": 5},
            "retract": {"start": 4, "stop": 6},
        },
    }
    candidate = {
        "T_assemble_tcp": np.eye(4).tolist(),
        "T_world_assemble_target": placed.tolist(),
        "gripper_clear_qpos": 0.02,
    }
    geometry = SimpleNamespace(
        assemble_mesh=mug_mesh,
        base_world=rack_mesh,
        initial=initial,
        object_check=object(),
        _world_clearance=lambda *args: None,
        _overlap=lambda *args: False,
    )
    return session, candidate, geometry


@pytest.mark.parametrize(
    "step,state", [(1, "initial"), (2, "carried"), (3, "placed"), (5, "placed")]
)
def test_arm_contact_with_cup_rejected_in_each_stage(step: int, state: str) -> None:
    session, candidate, geometry = _scene("wrist_link", (step,))
    report = session._check_trajectory(candidate, geometry)
    assert not report["accepted"]
    assert (
        f"wrist_link hits {state} assemble object at joint sample {step}"
        in report["reason"]
    )


def test_finger_contact_exempt_only_while_holding_or_releasing() -> None:
    session, candidate, geometry = _scene("gripper_finger1_link_1", (1, 2, 3))
    assert session._check_trajectory(candidate, geometry)["accepted"]


@pytest.mark.parametrize("step", [4, 5])
def test_fully_open_fingers_cannot_intersect_placed_cup(step: int) -> None:
    session, candidate, geometry = _scene("gripper_finger1_link_1", (step,))
    report = session._check_trajectory(candidate, geometry)
    assert not report["accepted"]
    assert (
        f"gripper_finger1_link_1 hits placed assemble object at joint sample {step}"
        in report["reason"]
    )


@pytest.mark.parametrize("height,accepted", [(0.022, False), (0.024, True)])
def test_ground_contact_tolerance_preserved(height: float, accepted: bool) -> None:
    import torch

    session, candidate, geometry = _scene("wrist_link", ())
    transforms = session.robot.pk_chain.forward_kinematics({})
    matrices = np.repeat(np.eye(4)[None], 6, axis=0)
    matrices[:, :3, 3] = [2.0, 0.0, height]
    transforms["wrist_link"].get_matrix = lambda: torch.tensor(matrices)
    report = session._check_trajectory(candidate, geometry)
    assert report["accepted"] == accepted
    if not accepted:
        assert "wrist_link hits ground" in report["reason"]


def test_rotated_bounding_box_below_ground_requires_exact_vertex_check() -> None:
    import torch

    session, candidate, geometry = _scene("wrist_link", ())
    # This tetrahedron lies entirely above the plane after rotation, although
    # unused corners of its transformed local AABB extend below the plane.
    vertices = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1], [0, 0, 1]]) * 0.05
    mesh = trimesh.Trimesh(vertices=vertices).convex_hull
    matrices = np.repeat(np.eye(4)[None], 6, axis=0)
    matrices[:, :3, :3] = Rotation.from_euler("y", 45, degrees=True).as_matrix()
    matrices[:, :3, 3] = [2.0, 0.0, 0.01]
    transforms = session.robot.pk_chain.forward_kinematics({})
    transforms["wrist_link"].get_matrix = lambda: torch.tensor(matrices)
    session.robot.get_link_vert_face = lambda _: (
        torch.tensor(mesh.vertices.copy()),
        torch.tensor(mesh.faces.copy()),
    )
    assert session._check_trajectory(candidate, geometry)["accepted"]
