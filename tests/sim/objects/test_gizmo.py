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

import embodichain.lab.sim.objects.gizmo as gizmo_module
from embodichain.lab.sim.objects.gizmo import Gizmo


class _RigidObject:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(uid="cube")
        self.pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        self.set_calls: list[tuple[torch.Tensor, list[int]]] = []

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        assert to_matrix
        return self.pose.clone()

    def set_local_pose(
        self,
        pose: torch.Tensor,
        env_ids: list[int],
    ) -> None:
        self.pose = pose.clone()
        self.set_calls.append((pose.clone(), env_ids))


class _Camera(_RigidObject):
    pass


class _Robot:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(uid="robot", solver_cfg={"arm": object()})
        self.control_parts = {"arm": ["joint"]}
        self.set_calls: list[tuple[torch.Tensor, list[int], list[int]]] = []

    def get_proprioception(self) -> dict[str, torch.Tensor]:
        return {"qpos": torch.zeros((1, 2), dtype=torch.float32)}

    def get_joint_ids(self, name: str) -> list[int]:
        assert name == "arm"
        return [0, 1]

    def compute_fk(self, *args: object, **kwargs: object) -> torch.Tensor:
        return torch.eye(4, dtype=torch.float32).unsqueeze(0)

    def compute_ik(
        self,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([True]), torch.tensor([[0.4, -0.2]])

    def set_qpos(
        self,
        qpos: torch.Tensor,
        joint_ids: list[int],
        env_ids: list[int],
    ) -> None:
        self.set_calls.append((qpos.clone(), joint_ids, env_ids))


def _patch_headless_dexsim(monkeypatch) -> None:
    monkeypatch.setattr(gizmo_module.dexsim, "get_world_num", lambda: 1)
    monkeypatch.setattr(
        gizmo_module.dexsim,
        "default_world",
        lambda: SimpleNamespace(get_env=lambda: object()),
    )


def test_headless_gizmo_applies_shared_pose_and_arbitrates_sources(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gizmo_module, "RigidObject", _RigidObject)
    _patch_headless_dexsim(monkeypatch)
    target = _RigidObject()
    gizmo = Gizmo(target, enable_native=False)
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[0, :3, 3] = torch.tensor([0.2, 0.3, 0.4])

    assert gizmo.begin_interaction("viser:client-a")
    assert not gizmo.request_local_pose(pose, source_id="viser:client-b")
    assert gizmo.request_local_pose(pose, source_id="viser:client-a")
    gizmo.update()
    assert gizmo.end_interaction("viser:client-a")

    assert not gizmo.native_enabled
    assert target.set_calls[-1][1] == [0]
    torch.testing.assert_close(target.pose, pose)


def test_headless_camera_gizmo_uses_shared_pose_path(monkeypatch) -> None:
    monkeypatch.setattr(gizmo_module, "Camera", _Camera)
    _patch_headless_dexsim(monkeypatch)
    target = _Camera()
    gizmo = Gizmo(target, enable_native=False)
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[0, 2, 3] = 1.2

    assert gizmo.request_local_pose(pose, source_id="viser:client-a")
    gizmo.update()

    torch.testing.assert_close(target.pose, pose)


def test_headless_robot_gizmo_preserves_native_fk_ik_behavior(monkeypatch) -> None:
    monkeypatch.setattr(gizmo_module, "Robot", _Robot)
    _patch_headless_dexsim(monkeypatch)
    target = _Robot()
    gizmo = Gizmo(target, control_part="arm", enable_native=False)
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[0, 0, 3] = 0.5

    assert gizmo.request_local_pose(pose, source_id="viser:client-a")
    gizmo.update()

    assert target.set_calls[-1][1:] == ([0, 1], [0])
    torch.testing.assert_close(
        target.set_calls[-1][0],
        torch.tensor([[0.4, -0.2]]),
    )
