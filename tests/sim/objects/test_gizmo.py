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

import numpy as np
import pytest
import torch

import embodichain.lab.sim.objects.gizmo as gizmo_module
from embodichain.lab.sim.objects.gizmo import (
    Gizmo,
    GizmoCfg,
    _RobotGizmoAdapter,
    create_robot_ik_gizmo_controller,
)


class _FakeAdapterRobot:
    """Small Robot-compatible state holder for native adapter tests."""

    def __init__(self) -> None:
        self.control_parts = {"arm": ["joint_a", "joint_mimic", "joint_b"]}
        self.num_instances = 1
        self.joint_names = ["joint_a", "joint_mimic", "joint_b"]
        self.link_names = ["base_link", "tool_link"]
        self.device = torch.device("cpu")
        self.uid = "robot"
        self.cfg = SimpleNamespace(solver_cfg=None, fpath="robot.urdf")
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
    cfg = GizmoCfg(
        ik_root_link_name="base_link",
        ik_end_link_name="tool_link",
    )

    root_link, end_link, tcp_pose = gizmo_module._resolve_robot_ik_chain(
        _FakeAdapterRobot(),
        "arm",
        cfg,
    )

    assert (root_link, end_link) == ("base_link", "tool_link")
    np.testing.assert_allclose(tcp_pose, np.eye(4))


def test_native_robot_factory_returns_dexsim_owned_controllers(monkeypatch) -> None:
    robot = _FakeAdapterRobot()
    adapter = _RobotGizmoAdapter(robot, "arm")
    solver = object()
    monkeypatch.setattr(
        gizmo_module,
        "_build_robot_ik",
        lambda robot, control_part, cfg: (
            adapter,
            solver,
            "tool_link",
            np.eye(4, dtype=np.float32),
        ),
    )

    class _InputController:
        pass

    class _IKController:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    import dexsim.engine
    import dexsim.kit.ik

    monkeypatch.setattr(dexsim.engine, "GizmoController", _InputController)
    monkeypatch.setattr(dexsim.kit.ik, "IKGizmoController", _IKController)
    window = SimpleNamespace(controls=[])
    window.add_input_control = window.controls.append
    world = SimpleNamespace(get_windows=lambda: window)

    controller, input_controller = create_robot_ik_gizmo_controller(
        robot,
        world=world,
    )

    assert controller.args[:3] == (world, adapter, solver)
    assert controller.kwargs["follow_robot_base"] is True
    assert isinstance(input_controller, _InputController)
    assert window.controls == [input_controller]


class _RigidObject:
    def __init__(self) -> None:
        self.num_instances = 1
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


def test_headless_gizmo_applies_shared_pose_and_arbitrates_sources(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gizmo_module, "RigidObject", _RigidObject)
    target = _RigidObject()
    gizmo = Gizmo(target)
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[0, :3, 3] = torch.tensor([0.2, 0.3, 0.4])

    assert gizmo.begin_interaction("viser:client-a")
    assert not gizmo.request_local_pose(pose, source_id="viser:client-b")
    assert gizmo.request_local_pose(pose, source_id="viser:client-a")
    gizmo.update()
    assert gizmo.end_interaction("viser:client-a")

    assert target.set_calls[-1][1] == [0]
    torch.testing.assert_close(target.pose, pose)


def test_headless_camera_gizmo_uses_shared_pose_path(monkeypatch) -> None:
    monkeypatch.setattr(gizmo_module, "Camera", _Camera)
    target = _Camera()
    gizmo = Gizmo(target)
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[0, 2, 3] = 1.2

    assert gizmo.request_local_pose(pose, source_id="viser:client-a")
    gizmo.update()

    torch.testing.assert_close(target.pose, pose)


def test_headless_robot_gizmo_uses_dexsim_newton_ik(monkeypatch) -> None:
    """Headless (Viser) robot Gizmo solves IK with DexSim Newton IK.

    The queued Viser target is converted to the robot base-local frame and
    driven through the Newton solver; the solved qpos is written back as a
    joint drive target, mirroring the native ``IKApplyMode.DRIVE_TARGET`` path
    instead of calling the EmbodiChain ``compute_ik`` solver.
    """
    monkeypatch.setattr(gizmo_module, "Robot", _FakeAdapterRobot)
    target = _FakeAdapterRobot()

    solved_qpos = np.array([0.4, -0.2], dtype=np.float32)

    class _FakeNewtonSolver:
        def __init__(self) -> None:
            self.set_target_calls: list[tuple[np.ndarray, np.ndarray]] = []
            self.solve_iterations: list[int | None] = []

        def set_target_pose(self, position, rotation) -> None:
            self.set_target_calls.append(
                (np.array(position, copy=True), np.array(rotation, copy=True))
            )

        def solve(self, joint_names, current_qpos, iterations=None) -> None:
            self.solve_iterations.append(iterations)

        def qpos_for_joint_names(self, joint_names, fallback_qpos):
            return solved_qpos

    fake_solver = _FakeNewtonSolver()

    def _inject_solver(self) -> None:
        self._robot_adapter = _RobotGizmoAdapter(target, "arm")
        self._ik_solver = fake_solver
        self._robot_end_link = "tool_link"
        self._robot_tcp_pose = np.eye(4, dtype=np.float32)

    monkeypatch.setattr(Gizmo, "_setup_robot_ik_solver", _inject_solver)

    gizmo = Gizmo(target, control_part="arm")
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose[0, 0, 3] = 0.5

    assert gizmo.request_local_pose(pose, source_id="viser:client-a")
    gizmo.update()

    # The Newton solver received a base-local target and was asked to solve
    # with the configured iteration count.
    assert fake_solver.set_target_calls
    assert fake_solver.solve_iterations == [gizmo.cfg.ik_iterations]
    # The solved qpos is written back as a drive target through the adapter.
    write = target.write_calls[-1]
    assert write["target"] is True
    assert write["joint_ids"] == [0, 2]
    torch.testing.assert_close(
        write["qpos"],
        torch.tensor([[0.4, -0.2]], dtype=torch.float32),
    )
