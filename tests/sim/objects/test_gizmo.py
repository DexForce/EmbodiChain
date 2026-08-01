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

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import embodichain.lab.sim.objects.gizmo as gizmo_module
from embodichain.lab.sim.objects.gizmo import Gizmo, GizmoCfg, _RobotGizmoAdapter


class _FakeAdapterRobot:
    """Small Robot-compatible state holder for native adapter tests."""

    def __init__(self) -> None:
        self.control_parts = {"arm": ["joint_a", "joint_mimic", "joint_b"]}
        self.num_instances = 1
        self.joint_names = ["joint_a", "joint_mimic", "joint_b"]
        self.link_names = ["base_link", "tool_link"]
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(solver_cfg=None)
        self.current_qpos = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        self.target_qpos = torch.tensor([[0.4, 0.5, 0.6]], dtype=torch.float32)
        self.write_calls: list[dict[str, object]] = []

    def get_joint_ids(
        self,
        name: str,
        remove_mimic: bool = False,
    ) -> list[int]:
        assert name == "arm"
        return [0, 2] if remove_mimic else [0, 1, 2]

    def get_qpos(self, target: bool = False) -> torch.Tensor:
        return self.target_qpos if target else self.current_qpos

    def set_qpos(self, **kwargs: object) -> None:
        self.write_calls.append(kwargs)

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        assert to_matrix
        return torch.eye(4, dtype=torch.float32).unsqueeze(0)

    def get_link_pose(
        self,
        link_name: str,
        env_ids: list[int],
        to_matrix: bool = False,
    ) -> torch.Tensor:
        assert link_name in self.link_names
        assert env_ids == [0]
        assert to_matrix
        pose = torch.eye(4, dtype=torch.float32)
        pose[2, 3] = 0.8
        return pose.unsqueeze(0)


def test_robot_adapter_synchronizes_selected_joint_state() -> None:
    robot = _FakeAdapterRobot()
    adapter = _RobotGizmoAdapter(robot, "arm")

    assert adapter.get_actived_joint_names() == ["joint_a", "joint_b"]
    np.testing.assert_allclose(adapter.get_current_qpos(), [0.1, 0.3])
    np.testing.assert_allclose(adapter.get_target_qpos(), [0.4, 0.6])

    adapter.set_target_qpos(np.array([0.7, 0.9], dtype=np.float32))

    write = robot.write_calls[-1]
    assert write["joint_ids"] == [0, 2]
    assert write["env_ids"] == [0]
    assert write["target"] is True
    torch.testing.assert_close(
        write["qpos"],
        torch.tensor([[0.7, 0.9]], dtype=torch.float32),
    )


def test_robot_adapter_reads_root_and_link_pose_through_robot() -> None:
    adapter = _RobotGizmoAdapter(_FakeAdapterRobot(), "arm")

    np.testing.assert_allclose(adapter.get_world_pose(), np.eye(4))
    link_pose = adapter.get_link_pose("tool_link")
    assert link_pose[2, 3] == pytest.approx(0.8)
    assert adapter.get_link_names(True) == ["base_link", "tool_link"]


def test_robot_adapter_rejects_wrong_qpos_shape() -> None:
    adapter = _RobotGizmoAdapter(_FakeAdapterRobot(), "arm")

    with pytest.raises(ValueError, match="Expected qpos shape"):
        adapter.set_target_qpos(np.zeros(3, dtype=np.float32))


def test_robot_native_ik_chain_can_be_configured_without_solver() -> None:
    gizmo = object.__new__(Gizmo)
    gizmo.cfg = GizmoCfg(
        ik_root_link_name="base_link",
        ik_end_link_name="tool_link",
    )
    gizmo._control_part = "arm"

    root_link, end_link, tcp_pose = gizmo._resolve_robot_ik_chain(_FakeAdapterRobot())

    assert (root_link, end_link) == ("base_link", "tool_link")
    np.testing.assert_allclose(tcp_pose, np.eye(4))


def test_robot_update_delegates_to_dexsim_ik_controller() -> None:
    calls: list[int] = []

    class _Controller:
        def update(self, *, iterations: int) -> None:
            calls.append(iterations)

    gizmo = object.__new__(Gizmo)
    gizmo.target = object()
    gizmo._ik_controller = _Controller()
    gizmo.cfg = GizmoCfg(ik_iterations=12)

    gizmo.update()

    assert calls == [12]


def test_destroy_removes_gizmo_from_dexsim_environment() -> None:
    class _DexsimGizmo:
        def __init__(self) -> None:
            self.detached = False

        def set_flush_localpose_callback(self, callback: object | None) -> None:
            pass

        def set_transform_flush_callback(self, callback: object | None) -> None:
            pass

        def set_visible(self, visible: bool) -> None:
            pass

        def detach_parent(self) -> None:
            self.detached = True

    class _Environment:
        def __init__(self) -> None:
            self.removed: object | None = None

        def remove_gizmo(self, gizmo: object) -> None:
            self.removed = gizmo

    native_gizmo = _DexsimGizmo()
    environment = _Environment()
    gizmo = object.__new__(Gizmo)
    gizmo._env = environment
    gizmo._gizmo = native_gizmo
    gizmo._proxy_cube = None
    gizmo._ik_controller = None
    gizmo._ik_solver = None
    gizmo._ik_model = None
    gizmo._robot_adapter = None
    gizmo._state_lock = threading.RLock()
    gizmo._interaction_owner = None
    gizmo._pending_target_transform = None
    gizmo._desired_target_transform = None
    gizmo.target = object()
    gizmo._target_type = "rigid_object"

    gizmo.destroy()

    assert environment.removed is native_gizmo
    assert native_gizmo.detached is True
    assert gizmo._gizmo is None


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
