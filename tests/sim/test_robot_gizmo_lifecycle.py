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
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from dexsim.kit.ik.interactive import KeyPressTracker

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import GizmoCfg
from embodichain.lab.sim.objects import gizmo as gizmo_module
from embodichain.lab.visualization import VisualizationCfg

pytestmark = pytest.mark.no_sim


class _Robot:
    __hash__ = None  # Robot inherits dataclass equality and is not hashable.
    num_instances = 1
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.control_parts = {
            "left": ["left_joint"],
            "right": ["right_joint"],
            "hand": ["finger"],
        }
        self.cfg = SimpleNamespace(
            uid="robot",
            solver_cfg={
                part: SimpleNamespace(
                    root_link_name="base", end_link_name=f"{part}_tool"
                )
                for part in ("left", "right")
            },
        )
        self.pose = torch.eye(4).unsqueeze(0)
        self.destroy = Mock()

    def get_solver(self, part: str) -> SimpleNamespace:
        cfg = self.cfg.solver_cfg[part]
        return SimpleNamespace(**vars(cfg), get_tcp=lambda: np.eye(4))

    def get_link_pose(
        self, link: str, *, env_ids: list[int], to_matrix: bool
    ) -> torch.Tensor:
        return self.pose.clone()


class _Window:
    def __init__(self) -> None:
        self.controls: list[object] = []

    def add_input_control(self, control: object) -> None:
        self.controls.append(control)

    def remove_input_control(self, control: object) -> None:
        self.controls.remove(control)


class _World:
    def __init__(self) -> None:
        self.window = _Window()
        self.key_down = False
        self.env = SimpleNamespace(remove_gizmo=Mock(), remove_dummy_node=Mock())

    def get_windows(self) -> _Window | None:
        return self.window

    def get_env(self) -> SimpleNamespace:
        return self.env

    def key_state(self, key: object) -> bool:
        return self.key_down

    def close_window(self) -> None:
        self.window = None

    def open_window(self) -> None:
        self.window = _Window()


class _Controller:
    def __init__(self, world: _World, cfg: GizmoCfg) -> None:
        self.world = world
        self.toggle = KeyPressTracker(cfg.ik_toggle_key)
        self.enabled = True
        self.target_gizmo = SimpleNamespace(gizmo=object(), target_node=object())

    def update(self) -> None:
        if self.toggle.pressed(self.world):
            self.enabled = not self.enabled

    def _set_visible(self, visible: bool) -> None:
        self.enabled = visible


@pytest.fixture
def managed_robot(monkeypatch: pytest.MonkeyPatch):
    """Use the real manager/Gizmo lifecycle with rendering and IK construction stubbed."""
    monkeypatch.setattr(gizmo_module, "Robot", _Robot)
    robot = _Robot()
    sim = object.__new__(SimulationManager)
    sim.sim_config = SimulationManagerCfg(headless=True)
    sim.num_envs = 1
    sim._world = _World()
    sim._window = sim._world.window
    sim.is_window_opened = True
    sim._window_record_state = None
    sim._window_record_hotkey_cfg = None
    sim._window_camera_pose_hotkey_cfg = None
    sim._auto_entity_gizmo_pending = False
    sim._visualization_runtime = None
    sim._visualization_topology_revision = 0
    sim._gizmos = {}
    sim._disabled_robot_gizmos = set()
    sim._picker_gizmo = None
    sim._robots = {robot.cfg.uid: robot}
    for registry in (
        "_rigid_objects",
        "_rigid_object_groups",
        "_articulations",
        "_soft_objects",
        "_cloth_objects",
        "_sensors",
    ):
        setattr(sim, registry, {})
    sim.process_pick_commands = Mock(return_value=0)
    sim.process_visualization_commands = Mock(return_value=0)

    def create(robot, part, cfg, *, world):
        controller = _Controller(world, cfg)
        control = object()
        world.get_windows().add_input_control(control)
        return controller, control

    factory = Mock(side_effect=create)
    monkeypatch.setattr(gizmo_module, "create_robot_ik_gizmo_controller", factory)
    yield sim, robot, factory
    for _, gizmo in sim.get_gizmo_items():
        gizmo.destroy()


@pytest.mark.parametrize(
    "frontend", ["native", "viser", "read_only", "headless", "batch", "disabled"]
)
def test_automatic_registration_respects_interaction_boundaries(
    managed_robot, frontend: str
) -> None:
    sim, robot, factory = managed_robot
    if frontend in {"viser", "read_only", "headless"}:
        sim._window = None
    if frontend in {"viser", "read_only"}:
        sim.sim_config.visualization = VisualizationCfg(
            backend="viser", allow_commands=frontend == "viser"
        )
    if frontend == "batch":
        sim.num_envs = 2
    if frontend == "disabled":
        sim.sim_config.robot_ik_gizmo = None
    sim.update_gizmos()
    sim.update_gizmos()
    expected = (
        {"robot:left", "robot:right"} if frontend in {"native", "viser"} else set()
    )
    assert set(sim.list_gizmos()) == expected
    factory.assert_not_called()
    for _, gizmo in sim.get_gizmo_items():
        assert gizmo._ik_solver is None
        robot.pose[0, 0, 3] = 0.5
        assert gizmo.get_control_pose()[0, 0, 3] == pytest.approx(0.5)


def test_native_activation_and_reopen_preserve_controller_and_visibility(
    managed_robot,
) -> None:
    sim, _, factory = managed_robot
    sim.update_gizmos()
    sim._world.key_down = True
    sim.update_gizmos()
    controls = [gizmo._native_controller for _, gizmo in sim.get_gizmo_items()]
    assert factory.call_count == 2
    sim.update_gizmos()  # Holding I does not toggle a second time.
    assert all(control.enabled for control in controls)
    sim._world.key_down = False
    sim.update_gizmos()
    sim._world.key_down = True
    sim.update_gizmos()
    assert not any(control.enabled for control in controls)

    old_window = sim._window
    sim.close_window()
    assert old_window.controls == []
    assert sim.open_window()
    sim.update_gizmos()
    assert factory.call_count == 2
    assert len(sim._window.controls) == 2
    assert [g._native_controller for _, g in sim.get_gizmo_items()] == controls
    assert not any(control.enabled for control in controls)


def test_explicit_disable_and_solver_override_take_precedence(managed_robot) -> None:
    sim, _, _ = managed_robot
    sim.disable_gizmo("robot", "left")
    sim.update_gizmos()
    assert set(sim.list_gizmos()) == {"robot:right"}
    sim.enable_gizmo("robot", "right", GizmoCfg(ik_solver="embodichain"))
    right = sim.get_gizmo("robot", "right")
    sim.update_gizmos()
    assert sim.get_gizmo("robot", "right") is right
    assert right.cfg.ik_solver == "embodichain"
    sim.enable_gizmo("robot", "left")
    assert set(sim.list_gizmos()) == {"robot:left", "robot:right"}


def test_start_enabled_waits_for_window_and_preserves_later_visibility(
    managed_robot,
) -> None:
    sim, _, factory = managed_robot
    sim.sim_config.robot_ik_gizmo.ik_start_enabled = True
    sim.close_window()
    sim.update_gizmos()
    factory.assert_not_called()
    sim.open_window()
    sim.update_gizmos()  # No I press is needed.
    controls = [g._native_controller for _, g in sim.get_gizmo_items()]
    assert factory.call_count == 2
    assert all(control.enabled for control in controls)
    sim._world.key_down = True
    sim.update_gizmos()
    sim.update_gizmos()
    assert not any(control.enabled for control in controls)
    sim.close_window()
    sim.open_window()
    sim.update_gizmos()
    assert [g._native_controller for _, g in sim.get_gizmo_items()] == controls
    assert not any(control.enabled for control in controls)
    assert factory.call_count == 2


def test_start_enabled_failure_waits_for_a_key_before_retry(managed_robot) -> None:
    sim, _, factory = managed_robot
    sim.sim_config.robot_ik_gizmo.ik_start_enabled = True
    create = factory.side_effect
    factory.side_effect = ValueError("invalid chain")
    sim.update_gizmos()
    sim.update_gizmos()
    assert factory.call_count == 2
    assert not sim._window.controls
    factory.side_effect = create
    sim._world.key_down = True
    sim.update_gizmos()
    assert factory.call_count == 4
    assert all(g._native_controller.enabled for _, g in sim.get_gizmo_items())


def test_whole_robot_disable_cleans_activated_controls(managed_robot) -> None:
    sim, _, factory = managed_robot
    sim._world.key_down = True
    sim.update_gizmos()
    assert not sim.toggle_gizmo_visibility("robot", "left")
    assert not sim.get_gizmo("robot", "left")._native_controller.enabled
    sim.set_gizmo_visibility("robot", True, "left")
    assert sim.get_gizmo("robot", "left")._native_controller.enabled
    sim.disable_gizmo("robot")
    sim.update_gizmos()
    assert not sim.list_gizmos()
    assert not sim._window.controls
    assert factory.call_count == 2


def test_native_activation_failure_does_not_interrupt_other_gizmos(
    managed_robot,
) -> None:
    sim, _, factory = managed_robot
    create = factory.side_effect

    def fail_left(robot, part, cfg, *, world):
        if part == "left":
            raise ValueError("invalid chain")
        return create(robot, part, cfg, world=world)

    factory.side_effect = fail_left
    sim._world.key_down = True
    sim.update_gizmos()
    sim.update_gizmos()
    assert factory.call_count == 2  # A held key does not repeatedly retry a failure.
    assert sim.get_gizmo("robot", "left")._native_controller is None
    assert sim.get_gizmo("robot", "right")._native_controller is not None


def test_robot_gizmo_setting_rejects_boolean() -> None:
    with pytest.raises(TypeError, match="robot_ik_gizmo"):
        SimulationManagerCfg(robot_ik_gizmo=False)


def test_robot_removal_cleans_native_controls_and_allows_same_uid_again(
    managed_robot,
) -> None:
    sim, robot, _ = managed_robot
    sim._world.key_down = True
    sim.update_gizmos()
    assert sim.remove_asset("robot")
    assert not sim.list_gizmos()
    assert not sim._window.controls
    assert sim._world.env.remove_gizmo.call_count == 2
    assert sim._world.env.remove_dummy_node.call_count == 2
    robot.destroy.assert_called_once()
    sim._robots["robot"] = _Robot()
    sim._world.key_down = False
    sim.update_gizmos()
    assert set(sim.list_gizmos()) == {"robot:left", "robot:right"}


@pytest.mark.parametrize("start_enabled", [False, True])
def test_explicit_native_factory_controller_is_not_duplicated(
    managed_robot, monkeypatch: pytest.MonkeyPatch, start_enabled: bool
) -> None:
    sim, robot, factory = managed_robot
    sim.sim_config.robot_ik_gizmo.ik_start_enabled = start_enabled
    explicit = Mock()
    monkeypatch.setitem(
        gizmo_module._NATIVE_IK_CONTROLLERS, (id(robot), "left"), explicit
    )
    sim._world.key_down = True
    sim.update_gizmos()
    assert factory.call_count == 1
    assert factory.call_args.args[1] == "right"
    explicit.update.assert_not_called()
