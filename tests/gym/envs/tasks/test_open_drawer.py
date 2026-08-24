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

"""Tests for the declarative drawer-opening task."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import embodichain.data as data_module
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.expert_program import (
    ArticulationJointPositionValidatorCfg,
    RegisteredSemanticCallCfg,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)
from embodichain.lab.sim.atomic_actions import SlideAffordance

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.tableware.open_drawer import (  # noqa: E402
    DRAWER_ROBOT_PROFILE_ID,
    DRAWER_SCENE_REGISTRY_ID,
    OpenDrawerEnv,
    create_open_drawer_robot_profile_binding,
    create_open_drawer_scene_binding,
)


def _gym_config_path() -> Path:
    """Return the installed-source drawer Gym config path."""
    return (
        Path(__file__).parents[4]
        / "embodichain_tasks/configs/gym/open_drawer/cobot_magic_3cam.json"
    )


def _gym_payload() -> dict[str, object]:
    """Load the drawer Gym config as inert JSON data."""
    payload = json.loads(_gym_config_path().read_text(encoding="utf-8"))
    assert type(payload) is dict
    return payload


def test_registered_drawer_task_uses_shared_expert_program_runtime() -> None:
    """The environment delegates all demo generation to the shared runtime."""
    spec = REGISTERED_ENVS["OpenDrawer-v1"]

    assert spec.cls is OpenDrawerEnv
    assert issubclass(OpenDrawerEnv, EmbodiedEnv)
    assert "create_demo_action_list" not in OpenDrawerEnv.__dict__
    assert "create_demo_segments" not in OpenDrawerEnv.__dict__


def test_drawer_gym_config_loads_registered_slide_and_joint_validator(
    monkeypatch,
) -> None:
    """The runnable task combines semantic motion with physical acceptance."""
    payload = _gym_payload()

    assert payload["id"] == "OpenDrawer-v1"
    assert payload["expert_program_path"] == (
        "../../expert_program/tableware/open_drawer.json"
    )
    monkeypatch.setattr(data_module, "get_data_path", lambda value: value)
    cfg = config_to_cfg(payload, source_path=_gym_config_path())
    assert cfg.expert_program is not None
    assert cfg.expert_program.program_id == "open_drawer"
    call = cfg.expert_program.program.steps.call
    assert type(call) is RegisteredSemanticCallCfg
    assert call.call_id == "embodichain_tasks.open_drawer"
    assert call.arguments["direction"] == "pull"
    assert call.arguments["hand_interp_steps"] == 20
    assert call.arguments["translation_distance"] == 0.12
    assert call.resources == {"primary": "right_manipulator"}
    validator = cfg.expert_program.program.validators[0]
    assert type(validator) is ArticulationJointPositionValidatorCfg
    assert validator.articulation == "drawer"
    assert validator.joint == "slide_rails"
    assert validator.minimum_position == 0.09


def test_drawer_gym_config_preserves_physical_scene(monkeypatch) -> None:
    """Parsing still creates the CobotMagic robot and native drawer entity."""
    monkeypatch.setattr(data_module, "get_data_path", lambda value: value)
    cfg = config_to_cfg(_gym_payload(), source_path=_gym_config_path())

    assert cfg.robot.uid == "CobotMagic"
    assert cfg.robot.control_parts["right_arm"] == [
        "right_joint1",
        "right_joint2",
        "right_joint3",
        "right_joint4",
        "right_joint5",
        "right_joint6",
    ]
    assert cfg.robot.control_parts["right_eef"] == [
        "right_joint7",
        "right_joint8",
    ]
    assert cfg.articulation[0].uid == "drawer"


def test_drawer_bindings_select_native_handle_and_slide_resource() -> None:
    """Canonical identities and CobotMagic resource wiring remain explicit."""
    scene = create_open_drawer_scene_binding()
    profile = create_open_drawer_robot_profile_binding()

    assert scene.registry_id == DRAWER_SCENE_REGISTRY_ID
    assert scene.articulations[0].simulation_uid == "drawer"
    assert scene.links[0].entity_id == "drawer_handle"
    assert scene.links[0].native_link_name == "handle_xpos"
    assert profile.profile_id == DRAWER_ROBOT_PROFILE_ID
    assert dict(profile.defaults) == {"slide": {"primary": "right_manipulator"}}
    assert profile.command_presets[0].commands == {
        "open": (0.05, 0.05),
        "grasp": (0.0, 0.0),
    }


def test_task_initialization_builds_live_slide_semantics(monkeypatch) -> None:
    """The task passes handle mesh semantics to the registered Slide lowerer."""
    adapter = object()
    captured: dict[str, object] = {}

    class FakeDrawer:
        link_names = ("outer_box", "inner_box", "handle_xpos")

        @staticmethod
        def get_link_vert_face(name: str):
            assert name == "inner_box"
            return (
                torch.tensor(
                    [
                        [0.107, -0.01, 0.09],
                        [0.117, 0.01, 0.10],
                        [0.107, 0.01, 0.11],
                        [0.09, -0.08, 0.01],
                        [0.09, 0.08, 0.01],
                        [0.09, 0.08, 0.19],
                    ]
                ),
                torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long),
            )

    drawer = FakeDrawer()

    class FakeSimulation:
        device = torch.device("cpu")

        @staticmethod
        def get_articulation(uid: str):
            return drawer if uid == "drawer" else None

    class FakeFactory:
        @staticmethod
        def create_adapter(**kwargs):
            captured.update(kwargs)
            return adapter

    def fake_base_init(self, cfg, **kwargs) -> None:
        del kwargs
        self.cfg = cfg
        self.sim = FakeSimulation()
        self.robot = SimpleNamespace(uid="CobotMagic")

    def fake_from_environment(environment, **kwargs):
        captured["environment"] = environment
        captured.update(kwargs)
        return FakeFactory()

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_base_init)
    task_module = importlib.import_module(OpenDrawerEnv.__module__)
    monkeypatch.setattr(
        task_module.SimulationExpertProgramFactory,
        "from_environment",
        fake_from_environment,
    )

    env = OpenDrawerEnv(cfg=SimpleNamespace(expert_program=object()))

    assert env.expert_program_adapter is adapter
    assert captured["environment"] is env
    assert captured["scene_binding"].registry_id == DRAWER_SCENE_REGISTRY_ID
    assert captured["robot_profile_binding"].profile_id == DRAWER_ROBOT_PROFILE_ID
    assert set(captured["grasp_pose_generators"]) == {"right_eef"}
    grasp_generator = captured["grasp_pose_generators"]["right_eef"]
    success, grasp_poses, opening_widths = grasp_generator.get_best_grasp_poses(
        mesh_vertices=torch.zeros((3, 3)),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        obj_poses=torch.eye(4).unsqueeze(0),
        approach_direction=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    assert success.tolist() == [True]
    assert grasp_poses[0, :3, 3].tolist() == pytest.approx(
        [0.00049425, -0.00441209, 0.01492312]
    )
    assert opening_widths.tolist() == pytest.approx([0.01])
    lowerer = captured["registered_lowerers"][0]
    affordance = lowerer._semantics.affordance
    assert isinstance(affordance, SlideAffordance)
    assert affordance.joint_name == "slide_rails"
    assert affordance.translation_axis.tolist() == [0.0, 0.0, 1.0]
    assert affordance.mesh_triangles.tolist() == [[0, 1, 2]]
    assert torch.allclose(
        affordance.mesh_vertices,
        torch.tensor(
            [
                [-0.01, -0.01, -0.002],
                [0.0, 0.01, -0.012],
                [0.01, 0.01, -0.002],
            ]
        ),
    )


__all__: list[str] = []
