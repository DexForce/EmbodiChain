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

"""Tests for the declarative multi-segment cube task."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.expert_program import ExpertProgramEnvironmentMixin
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
)
from embodichain.lab.sim.robots import URRobotCfg

# Trigger official task auto-registration (idempotent).
discover_task_packages()

from embodichain_tasks.multi_segments.cube_pick_place import (  # noqa: E402
    CUBE_ROBOT_PROFILE_ID,
    CUBE_SCENE_REGISTRY_ID,
    MultiSegmentsCubePickPlaceEnv,
    _create_default_env_cfg,
    create_cube_robot_profile_binding,
)


def _gym_config_path() -> Path:
    """Return the installed-source cube Gym config path."""
    return (
        Path(__file__).parents[4]
        / "embodichain_tasks/configs/gym/multi_segments/cube_pick_place.json"
    )


def _gym_payload() -> dict[str, object]:
    """Load the runnable Gym configuration as inert JSON data."""
    path = _gym_config_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert type(payload) is dict
    return payload


def test_registered_task_uses_shared_expert_program_mixin() -> None:
    """The task is registered and delegates semantic execution to the mixin."""
    from embodichain_tasks.multi_segments import __all__

    assert "MultiSegmentsCubePickPlaceEnv" in __all__
    spec = REGISTERED_ENVS["MultiSegmentsCubePickPlace-v1"]
    assert spec.cls is MultiSegmentsCubePickPlaceEnv
    assert spec.max_episode_steps == 1200
    assert issubclass(MultiSegmentsCubePickPlaceEnv, ExpertProgramEnvironmentMixin)
    assert issubclass(MultiSegmentsCubePickPlaceEnv, EmbodiedEnv)


def test_gym_config_selects_packaged_expert_program() -> None:
    """Normal Gym startup selects the semantic program by a relative path."""
    payload = _gym_payload()

    assert payload["id"] == "MultiSegmentsCubePickPlace-v1"
    assert payload["expert_program_path"] == (
        "../../expert_program/multi_segments/repeated_cube_pick_place.yaml"
    )
    extensions = payload["env"]["extensions"]
    assert extensions == {
        "grasp_samples": 10000,
        "force_reannotate": False,
    }
    settle = payload["env"]["events"]["settle_cube_on_reset"]
    assert settle["func"] == "wait_for_dynamic_objects_to_settle"
    assert settle["mode"] == "reset"
    assert settle["params"]["entity_cfgs"] == [{"uid": "cube"}]


def test_gym_config_keeps_scene_and_robot_configuration() -> None:
    """The migration changes the expert layer, not the physical environment."""
    payload = _gym_payload()
    cfg = config_to_cfg(payload, source_path=_gym_config_path())

    assert isinstance(cfg.robot, URRobotCfg)
    assert cfg.robot.robot_type == "ur5"
    assert cfg.robot.control_parts["arm"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
    ]
    assert cfg.robot.control_parts["hand"] == ["gripper_finger1_joint_1"]
    assert cfg.rigid_object[0].uid == "cube"


def test_direct_default_cfg_loads_the_same_typed_program() -> None:
    """Direct Python construction and Gym startup share one packaged program."""
    cfg = _create_default_env_cfg()

    assert cfg.expert_program is not None
    assert cfg.expert_program.integration.scene_registry == CUBE_SCENE_REGISTRY_ID
    assert cfg.expert_program.integration.robot_profile == CUBE_ROBOT_PROFILE_ID
    assert cfg.expert_program.program_id == "repeated_cube_pick_place"
    settle = cfg.events["settle_cube_on_reset"]
    assert settle.func is not None
    assert settle.params["entity_cfgs"][0].uid == "cube"


def test_robot_profile_calibrates_physical_tracking_tolerance() -> None:
    """The UR5 preset tolerates its measured drive lag without disabling feedback."""
    binding = create_cube_robot_profile_binding()

    assert binding.presets[0].preset_id == "safe"
    assert binding.presets[0].recovery_policy.tracking_error_threshold == 0.08


def test_task_initialization_delegates_to_shared_simulation_factory(
    monkeypatch,
) -> None:
    """Task setup contributes bindings but no task-local motion generator."""
    adapter = object()
    captured: dict[str, object] = {}

    def fake_base_init(self, cfg, **kwargs) -> None:
        del cfg, kwargs
        self.grasp_samples = 48
        self.force_reannotate = True

    def fake_create_adapter(environment, **kwargs):
        captured["environment"] = environment
        captured.update(kwargs)
        return adapter

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_base_init)
    task_module = importlib.import_module(MultiSegmentsCubePickPlaceEnv.__module__)
    monkeypatch.setattr(
        task_module,
        "create_simulation_expert_program_adapter",
        fake_create_adapter,
    )

    env = MultiSegmentsCubePickPlaceEnv(cfg=object())

    assert env.expert_program_adapter is adapter
    assert captured["environment"] is env
    assert (
        captured["scene_binding"]
        .antipodal_grasps[0]
        .generator_cfg.antipodal_sampler_cfg.n_sample
        == 48
    )
    assert captured["scene_binding"].antipodal_grasps[0].force_reannotate is True
    assert captured["robot_profile_binding"].profile_id == CUBE_ROBOT_PROFILE_ID


def test_task_config_compiles_through_real_simulation_factory(
    monkeypatch,
) -> None:
    """Packaged config reaches the real adapter with explicitly bound mocks."""

    class FakeRobot:
        uid = "UR5"

        @staticmethod
        def get_qpos() -> torch.Tensor:
            return torch.zeros((1, 8), dtype=torch.float32)

    class FakeCube:
        is_non_dynamic = False

        @staticmethod
        def get_vertices(*, env_ids, scale) -> torch.Tensor:
            assert env_ids == [0]
            assert scale is True
            return torch.tensor(
                [[[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]],
                dtype=torch.float32,
            )

        @staticmethod
        def get_triangles(*, env_ids) -> torch.Tensor:
            assert env_ids == [0]
            return torch.tensor([[[0, 1, 2]]], dtype=torch.int64)

        @staticmethod
        def get_local_pose(*, to_matrix) -> torch.Tensor:
            assert to_matrix is True
            return torch.eye(4, dtype=torch.float32).unsqueeze(0)

    robot = FakeRobot()
    cube = FakeCube()

    class FakeSimulation:
        @staticmethod
        def get_robot(uid: str):
            return robot if uid == "UR5" else None

        @staticmethod
        def get_rigid_object(uid: str):
            return cube if uid == "cube" else None

    def fake_base_init(self, cfg, **kwargs) -> None:
        del kwargs
        self.cfg = cfg
        self.sim_cfg = SimpleNamespace(physics_dt=0.01)
        self.sim = FakeSimulation()
        self.robot = robot
        for name, value in cfg.extensions.items():
            setattr(self, name, value)

    monkeypatch.setattr(EmbodiedEnv, "__init__", fake_base_init)
    cfg = _create_default_env_cfg()

    env = MultiSegmentsCubePickPlaceEnv(cfg=cfg)
    segments = tuple(env.compile_expert_program(cfg.expert_program))

    assert len(segments) == 3
    assert [segment.name for segment in segments] == ["move_cube"] * 3
    assert env.expert_program_adapter.scene_registry_id == CUBE_SCENE_REGISTRY_ID
    assert env.expert_program_adapter.robot_profile_id == CUBE_ROBOT_PROFILE_ID


__all__: list[str] = []
