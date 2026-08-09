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

import os
from types import SimpleNamespace

import torch
import pytest
import numpy as np
import gymnasium as gym

from embodichain.lab.sim.cfg import RenderCfg
from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.managers import ActionManager
from embodichain.lab.gym.envs.managers.actions import QfTerm, QvelTerm
from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg, EventCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.gym.envs.managers.randomization.visual import (
    randomize_visual_material,
    set_rigid_object_visual_material,
)
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, DEFAULT_MANAGER_MODULES
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.data import get_data_path

NUM_ENVS = 2

pytestmark = pytest.mark.requires_sim

urdf_path = get_data_path("UniversalRobots/UR5/UR5.urdf")
METADATA = {
    "id": "EmbodiedEnv-v1",
    "max_episodes": 1,
    "env": {
        "events": {
            "random_light": {
                "func": "randomize_light",
                "mode": "interval",
                "interval_step": 10,
                "params": {
                    "entity_cfg": {"uid": "light_1"},
                    "position_range": [[-0.5, -0.5, 2], [0.5, 0.5, 2]],
                    "color_range": [[0.6, 0.6, 0.6], [1, 1, 1]],
                    "intensity_range": [500000.0, 1500000.0],
                },
            }
        }
    },
    "sensor": [
        {
            "sensor_type": "Camera",
            "width": 640,
            "height": 480,
            "enable_mask": True,
            "enable_depth": True,
            "extrinsics": {
                "eye": [0.0, 0.0, 1.0],
                "target": [0.0, 0.0, 0.0],
            },
        }
    ],
    "robot": {
        "fpath": urdf_path,
        "drive_pros": {"stiffness": {"joint[1-6]": 200.0}},
        "solver_cfg": {
            "class_type": "PytorchSolver",
            "end_link_name": "ee_link",
            "root_link_name": "base_link",
        },
        "init_pos": [0.0, 0.3, 1.0],
    },
    "light": {
        "direct": [
            {
                "uid": "light_1",
                "light_type": "point",
                "color": [1.0, 1.0, 1.0],
                "intensity": 1000000.0,
                "init_pos": [0, 0, 2],
                "radius": 10.0,
            }
        ]
    },
    "background": [
        {
            "uid": "shop_table",
            "shape": {
                "shape_type": "Mesh",
                "fpath": "ShopTableSimple/shop_table_simple.ply",
            },
            "max_convex_hull_num": 2,
            "attrs": {"mass": 10.0},
            "body_scale": (2, 1.6, 1),
        }
    ],
    "rigid_object": [
        {
            "uid": "duck",
            "shape": {
                "shape_type": "Mesh",
                "fpath": "ToyDuck/toy_duck.glb",
            },
            "body_scale": (0.75, 0.75, 1.0),
            "init_pos": (0.0, 0.0, 1.0),
        }
    ],
    "articulation": [
        {
            "uid": "sliding_box_drawer",
            "fpath": "SlidingBoxDrawer/SlidingBoxDrawer.urdf",
            "init_pos": (0.5, 0.0, 0.5),
        }
    ],
}


def test_visual_randomization_filter_keeps_deterministic_material_events():
    events = SimpleNamespace(
        random_material=EventCfg(func=randomize_visual_material),
        set_material=EventCfg(func=set_rigid_object_visual_material),
    )
    env = EmbodiedEnv.__new__(EmbodiedEnv)
    env.cfg = SimpleNamespace(filter_visual_rand=True, events=events)

    env._apply_functor_filter()

    assert events.random_material is None
    assert events.set_material is not None


class _CommandRobot:
    """Small robot stub for direct control-command routing tests."""

    def __init__(self) -> None:
        self.body_data = SimpleNamespace(
            qpos_limits=torch.tensor([[[-1.0, 1.0], [-2.0, 2.0]]]),
            qvel_limits=torch.tensor([[3.0, 4.0]]),
            qf_limits=torch.tensor([[5.0, 6.0]]),
        )
        self.calls: list[tuple[str, torch.Tensor, list[int]]] = []

    def set_qpos(self, qpos: torch.Tensor, joint_ids: list[int]) -> None:
        self.calls.append(("qpos", qpos.clone(), list(joint_ids)))

    def set_qvel(self, qvel: torch.Tensor, joint_ids: list[int]) -> None:
        self.calls.append(("qvel", qvel.clone(), list(joint_ids)))

    def set_qf(self, qf: torch.Tensor, joint_ids: list[int]) -> None:
        self.calls.append(("qf", qf.clone(), list(joint_ids)))


def _make_direct_control_env() -> EmbodiedEnv:
    """Build an uninitialized env with only direct command dependencies."""
    env = EmbodiedEnv.__new__(EmbodiedEnv)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._num_envs = 1
    env.active_joint_ids = [0, 1]
    env.robot = _CommandRobot()
    env.action_manager = None
    env._pending_direct_qf = None
    return env


def test_step_action_routes_direct_velocity_and_effort_commands() -> None:
    """Direct mappings retain their command type and enforce robot limits."""
    env = _make_direct_control_env()

    processed = env._step_action({"qvel": [[8.0, -8.0]], "qf": np.array([[9.0, -9.0]])})

    torch.testing.assert_close(processed["qvel"], torch.tensor([[3.0, -4.0]]))
    torch.testing.assert_close(processed["qf"], torch.tensor([[5.0, -6.0]]))
    assert [(kind, ids) for kind, _, ids in env.robot.calls] == [
        ("qvel", [0, 1]),
        ("qf", [0, 1]),
    ]


def test_effort_command_is_reapplied_before_physics_substep() -> None:
    """The latest direct effort is held throughout control decimation."""
    env = _make_direct_control_env()
    assert env._get_before_sim_step_callback() is None
    env._step_action({"qf": [[1.0, 2.0]]})

    callback = env._get_before_sim_step_callback()
    assert callback is not None
    callback(0)
    callback(1)

    effort_calls = [call for call in env.robot.calls if call[0] == "qf"]
    assert len(effort_calls) == 3
    for _, command, joint_ids in effort_calls:
        torch.testing.assert_close(command, torch.tensor([[1.0, 2.0]]))
        assert joint_ids == [0, 1]


def test_step_action_preserves_bare_tensor_qpos_compatibility() -> None:
    """A bare tensor remains the legacy joint-position command."""
    env = _make_direct_control_env()

    processed = env._step_action(torch.tensor([[2.0, -3.0]]))

    torch.testing.assert_close(processed, torch.tensor([[1.0, -2.0]]))
    assert env.robot.calls[0][0] == "qpos"


def test_single_velocity_action_term_is_not_misrouted_as_qpos() -> None:
    """The complete manager/env path preserves a single term's qvel type."""
    env = _make_direct_control_env()
    env._traj_buffer = None
    env.action_manager = ActionManager(
        {"velocity": ActionTermCfg(func=QvelTerm, params={"scale": 2.0})},
        env,
    )

    processed = env._preprocess_action(torch.tensor([[0.5, -0.5]]))
    env._step_action(processed)

    assert [kind for kind, _, _ in env.robot.calls] == ["qvel"]
    torch.testing.assert_close(env.robot.calls[0][1], torch.tensor([[1.0, -1.0]]))


def test_managed_effort_action_is_applied_at_substep_rate() -> None:
    """The complete manager/env path holds a qf term across decimation."""
    env = _make_direct_control_env()
    env._traj_buffer = None
    env.action_manager = ActionManager(
        {"effort": ActionTermCfg(func=QfTerm, params={"scale": 4.0})},
        env,
    )

    processed = env._preprocess_action(torch.tensor([[0.5, -0.5]]))
    env._step_action(processed)
    callback = env._get_before_sim_step_callback()
    assert callback is not None
    callback(0)
    callback(1)

    assert [kind for kind, _, _ in env.robot.calls] == ["qf", "qf"]
    for _, command, _ in env.robot.calls:
        torch.testing.assert_close(command, torch.tensor([[2.0, -2.0]]))


class EmbodiedEnvTest:
    """Shared test logic for CPU and CUDA."""

    def setup_simulation(self, sim_device):
        cfg: EmbodiedEnvCfg = config_to_cfg(
            METADATA, manager_modules=DEFAULT_MANAGER_MODULES
        )
        cfg.num_envs = NUM_ENVS
        cfg.sim_cfg = SimulationManagerCfg(
            headless=True,
            sim_device=sim_device,
        )

        self.env = gym.make(id=METADATA["id"], cfg=cfg)

    def test_env_rollout(self):
        """Test environment rollout."""
        for episode in range(2):
            print("Episode:", episode)
            obs, info = self.env.reset()

            for i in range(2):
                action = self.env.action_space.sample()
                action = torch.as_tensor(
                    action,
                    dtype=torch.float32,
                    device=self.env.get_wrapper_attr("device"),
                )

                obs, reward, done, truncated, info = self.env.step(action)

        assert reward.shape == (
            self.env.get_wrapper_attr("num_envs"),
        ), f"Expected reward shape ({self.env.get_wrapper_attr('num_envs')},), got {reward.shape}"
        assert done.shape == (
            self.env.get_wrapper_attr("num_envs"),
        ), f"Expected done shape ({self.env.get_wrapper_attr('num_envs')},), got {done.shape}"
        assert truncated.shape == (
            self.env.get_wrapper_attr("num_envs"),
        ), f"Expected truncated shape ({self.env.get_wrapper_attr('num_envs')},), got {truncated.shape}"
        assert obs.get("robot") is not None, "Expected 'robot' info in the info dict"

    def test_typed_velocity_and_effort_rollout(self):
        """Typed qvel and qf commands execute through a real simulation step."""
        self.env.reset()
        num_envs = self.env.get_wrapper_attr("num_envs")
        action_dim = len(self.env.get_wrapper_attr("active_joint_ids"))
        device = self.env.get_wrapper_attr("device")
        command = torch.zeros(num_envs, action_dim, device=device)

        _, velocity_reward, _, _, _ = self.env.step({"qvel": command})
        _, effort_reward, _, _, _ = self.env.step({"qf": command})

        assert velocity_reward.shape == (num_envs,)
        assert effort_reward.shape == (num_envs,)

    def teardown_method(self):
        """Clean up resources after each test method."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        import gc

        gc.collect()


# @pytest.mark.skip(reason="Skipping tests temporarily")
class TestCPU(EmbodiedEnvTest):
    def setup_method(self):
        self.setup_simulation("cpu")


# @pytest.mark.skip(reason="Skipping tests temporarily")
class TestCUDA(EmbodiedEnvTest):
    def setup_method(self):
        self.setup_simulation("cuda")
