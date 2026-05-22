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

import torch
import pytest

from embodichain.lab.sim.agent.monitor_functions import monitor_object_held
from embodichain.lab.sim.agent.monitor_utils import get_arm_object_distance


def _pose(x: float, y: float, z: float) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 3] = torch.tensor([x, y, z], dtype=torch.float32)
    return pose


class _Object:
    def __init__(self, pose: torch.Tensor) -> None:
        self._pose = pose

    def get_local_pose(self, to_matrix: bool = True) -> torch.Tensor:
        return self._pose.unsqueeze(0)


class _Sim:
    def __init__(self, pose: torch.Tensor) -> None:
        self._pose = pose

    def get_rigid_object_uid_list(self) -> list[str]:
        return ["fork"]

    def get_rigid_object(self, obj_name: str) -> _Object:
        assert obj_name == "fork"
        return _Object(self._pose)


class _Robot:
    device = torch.device("cpu")

    def __init__(self, eef_pose: torch.Tensor, gripper_qpos: float = 0.0) -> None:
        self._eef_pose = eef_pose
        self._gripper_qpos = float(gripper_qpos)
        self._qpos = torch.zeros(1, 8, dtype=torch.float32)

    def get_qpos(self, name=None) -> torch.Tensor:
        if name == "left_eef":
            return torch.tensor([[self._gripper_qpos]], dtype=torch.float32)
        if name == "right_eef":
            return torch.tensor([[self._gripper_qpos]], dtype=torch.float32)
        return self._qpos.clone()

    def compute_fk(self, qpos, name: str, to_matrix: bool = True) -> torch.Tensor:
        return self._eef_pose.unsqueeze(0)


class _Env:
    def __init__(
        self,
        *,
        object_pose: torch.Tensor,
        grasp_pose_obj: torch.Tensor | None,
        eef_pose: torch.Tensor,
        gripper_qpos: float = 0.0,
    ) -> None:
        self.sim = _Sim(object_pose)
        self.robot = _Robot(eef_pose, gripper_qpos=gripper_qpos)
        self.left_arm_joints = [0, 1, 2, 3]
        self.right_arm_joints = [4, 5, 6, 7]
        self.open_state = torch.tensor([0.05], dtype=torch.float32)
        self.close_state = torch.tensor([0.0], dtype=torch.float32)
        self.obj_info = {
            "fork": {
                "pose": object_pose,
                "grasp_pose_obj": grasp_pose_obj,
            }
        }


def test_monitor_object_held_uses_grasp_reference_pose() -> None:
    env = _Env(
        object_pose=_pose(0.30, 0.00, 0.00),
        grasp_pose_obj=_pose(-0.15, 0.00, 0.00),
        eef_pose=_pose(0.09, 0.00, 0.00),
    )

    distance = get_arm_object_distance(env, "left_arm", "fork")

    assert distance == pytest.approx(0.06, abs=1e-6)
    assert monitor_object_held(env, "left_arm", "fork", threshold=0.05) is False


def test_monitor_object_held_falls_back_to_object_center_without_grasp_reference() -> (
    None
):
    env = _Env(
        object_pose=_pose(0.30, 0.00, 0.00),
        grasp_pose_obj=None,
        eef_pose=_pose(0.10, 0.00, 0.00),
    )

    distance = get_arm_object_distance(env, "left_arm", "fork")

    assert distance == pytest.approx(0.2, abs=1e-6)
    assert monitor_object_held(env, "left_arm", "fork", threshold=0.05) is True


def test_monitor_object_held_allows_object_width_gripper_opening() -> None:
    env = _Env(
        object_pose=_pose(0.30, 0.00, 0.00),
        grasp_pose_obj=_pose(-0.15, 0.00, 0.00),
        eef_pose=_pose(0.15, 0.00, 0.00),
        gripper_qpos=0.03,
    )

    assert monitor_object_held(env, "left_arm", "fork", threshold=0.05) is False


def test_monitor_object_held_triggers_when_gripper_is_effectively_open() -> None:
    env = _Env(
        object_pose=_pose(0.30, 0.00, 0.00),
        grasp_pose_obj=_pose(-0.15, 0.00, 0.00),
        eef_pose=_pose(0.15, 0.00, 0.00),
        gripper_qpos=0.05,
    )

    assert monitor_object_held(env, "left_arm", "fork", threshold=0.05) is True
