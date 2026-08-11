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

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.envs.expert_program import ExpertProgramEnvironmentMixin
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.tableware.open_drawer import (  # noqa: E402
    DRAWER_NATIVE_SLIDE_JOINT,
    DRAWER_OPEN_POSITION,
    DRAWER_ROBOT_PROFILE_ID,
    DRAWER_UID,
    OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION,
    OpenDrawerEnv,
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


def test_registered_drawer_task_uses_shared_expert_program_mixin() -> None:
    """The environment delegates all demo generation to the shared runtime."""
    spec = REGISTERED_ENVS["OpenDrawer-v1"]

    assert spec.cls is OpenDrawerEnv
    assert spec.expert_program_registration is OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION
    assert "expert_program_registration" not in spec.default_kwargs
    assert issubclass(OpenDrawerEnv, ExpertProgramEnvironmentMixin)
    assert issubclass(OpenDrawerEnv, EmbodiedEnv)
    assert "create_demo_action_list" not in OpenDrawerEnv.__dict__


def test_drawer_gym_config_selects_packaged_semantic_program() -> None:
    """The runnable task config points at the named-target Expert Program."""
    payload = _gym_payload()

    assert payload["id"] == "OpenDrawer-v1"
    assert payload["expert_program_path"] == (
        "../../expert_program/tableware/open_drawer.json"
    )
    assert payload["env"]["extensions"] == {}


def test_drawer_gym_config_preserves_physical_scene() -> None:
    """Parsing still creates the CobotMagic robot and native drawer entity."""
    path = _gym_config_path()
    cfg = config_to_cfg(_gym_payload(), source_path=path)

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
    assert cfg.expert_program is not None
    assert cfg.expert_program.program_id == "open_drawer"


def test_drawer_affordance_uses_reachable_post_release_retract() -> None:
    """The opened drawer retract remains clear of the handle and IK-reachable."""
    operation = create_open_drawer_scene_binding().articulation_operations[0]
    contact_z = operation.contact_offset[11]
    retract_z = operation.retract_offset[11]

    assert retract_z < contact_z
    assert contact_z - retract_z == pytest.approx(0.01)


def test_task_initialization_delegates_to_shared_simulation_factory(
    monkeypatch,
) -> None:
    """Drawer setup contributes declarations but no planner implementation."""
    adapter = object()
    captured: dict[str, object] = {}

    def fake_base_init(self, cfg, **kwargs) -> None:
        del self, cfg, kwargs

    def fake_create_adapter(environment, **kwargs):
        captured["environment"] = environment
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_base_init)
    task_module = importlib.import_module(OpenDrawerEnv.__module__)
    monkeypatch.setattr(
        task_module,
        "create_simulation_expert_program_adapter",
        fake_create_adapter,
    )

    env = OpenDrawerEnv(cfg=object())

    assert env.expert_program_adapter is adapter
    assert captured["environment"] is env
    registration = captured["registration"]
    assert registration is OPEN_DRAWER_EXPERT_PROGRAM_REGISTRATION
    assert registration.scene_binding.links[0].native_link_name == "handle_xpos"
    assert registration.robot_profile_binding.profile_id == DRAWER_ROBOT_PROFILE_ID


def test_task_config_compiles_through_real_simulation_factory(
    monkeypatch,
) -> None:
    """Packaged drawer config reaches the real adapter with explicit mocks."""

    class FakeRobot:
        uid = "CobotMagic"

        @staticmethod
        def get_qpos() -> torch.Tensor:
            return torch.zeros((1, 16), dtype=torch.float32)

    class FakeDrawer:
        link_names = ("outer_box", "inner_box", "handle_xpos")
        joint_names = ("slide_rails",)

        @staticmethod
        def get_local_pose(*, to_matrix) -> torch.Tensor:
            assert to_matrix is True
            return torch.eye(4, dtype=torch.float32).unsqueeze(0)

        @staticmethod
        def get_link_pose(name: str, *, env_ids, to_matrix) -> torch.Tensor:
            assert name == "handle_xpos"
            assert env_ids == [0]
            assert to_matrix is True
            return torch.eye(4, dtype=torch.float32).unsqueeze(0)

    robot = FakeRobot()
    drawer = FakeDrawer()

    class FakeSimulation:
        @staticmethod
        def get_robot(uid: str):
            return robot if uid == "CobotMagic" else None

        @staticmethod
        def get_articulation(uid: str):
            return drawer if uid == "drawer" else None

    def fake_base_init(self, cfg, **kwargs) -> None:
        del kwargs
        self.cfg = cfg
        self.sim_cfg = SimpleNamespace(physics_dt=0.01)
        self.sim = FakeSimulation()
        self.robot = robot

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_base_init)
    path = _gym_config_path()
    cfg = config_to_cfg(_gym_payload(), source_path=path)

    env = OpenDrawerEnv(cfg=cfg)
    segments = tuple(env.compile_expert_program(cfg.expert_program))

    assert len(segments) == 1
    assert segments[0].name == "open_drawer"
    assert env.expert_program_adapter.scene_registry_id == "open_drawer_v1"
    assert env.expert_program_adapter.robot_profile_id == DRAWER_ROBOT_PROFILE_ID


@pytest.mark.requires_sim
@pytest.mark.slow
def test_real_sim_expert_episode_opens_drawer_with_joint_effect_trace() -> None:
    """The packaged program completes against live drawer physics and evidence."""
    import gc

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

    path = _gym_config_path()
    cfg = config_to_cfg(_gym_payload(), source_path=path)
    cfg.num_envs = 1
    cfg.sim_cfg = SimulationManagerCfg(
        headless=True,
        sim_device="cpu",
        num_envs=1,
    )
    cfg.sensor = []
    cfg.events = None
    cfg.observations = None
    cfg.dataset = None
    cfg.init_rollout_buffer = False
    cfg.record_trajectory = False
    cfg.filter_dataset_saving = True

    env: OpenDrawerEnv | None = None
    try:
        env = OpenDrawerEnv(cfg=cfg)
        env.reset(seed=0)

        result = execute_demo_episode(env)

        assert result.completed
        assert result.all_success
        assert result.terminal_reason == "success"
        assert len(result.segments) == 1
        segment = result.segments[0]
        assert segment.name == "open_drawer"
        assert segment.success

        metadata = segment.metadata
        runtime = metadata["runtime"]
        assert runtime["kind"] == "skill_result"
        assert runtime["status"] == "completed"
        assert runtime["masks"]["success"] == [True]
        assert len(runtime["calls"]) == 1
        call = runtime["calls"][0]
        assert call["semantic_id"] == "operate_articulation"
        assert call["status"] == "completed"
        assert call["masks"] == {
            "entered": [True],
            "completed": [True],
            "failed": [False],
        }
        assert call["plan_attempts"]
        assert call["plan_attempts"][-1]["plan_success_mask"] == [True]

        effects = call["effects"]
        assert effects
        for effect in effects:
            assert effect["effect_spec"]["semantic_id"] == "operate_articulation"
            evidence = effect["evidence"]["joint.position"]
            assert evidence["valid_mask"] == [True]
            assert evidence["acquisition_errors"] == [None]
            assert evidence["env_ids"] == [0]
        final_effect = effects[-1]
        assert final_effect["decision"] == {
            "success_mask": [True],
            "failure_mask": [False],
        }

        assert metadata["post_policies"] == []
        assert metadata["validation"] == {
            "env_ids": [0],
            "runtime_success_mask": [True],
            "eligible_mask_before_validation": [True],
            "post_policy_success_mask": None,
            "validators": [],
            "accepted_mask": [True],
        }

        drawer = env.sim.get_articulation(DRAWER_UID)
        assert drawer is not None
        joint_index = drawer.joint_names.index(DRAWER_NATIVE_SLIDE_JOINT)
        final_position = float(drawer.get_qpos()[0, joint_index].item())
        joint_tolerance = float(
            final_effect["monitor"]["resolved_params"]["joint_success_tolerance"]
        )
        assert abs(final_position - DRAWER_OPEN_POSITION) <= joint_tolerance
    finally:
        if env is not None:
            env.close()
        SimulationManager.flush_cleanup_queue()
        gc.collect()


__all__: list[str] = []
