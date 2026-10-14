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

"""Fluid Gym configuration, measurement and replay acceptance contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.utils.utility import load_config
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.observations import (
    get_fluid_summary,
    get_fluid_volume_in_region,
)
from embodichain.lab.sim.cfg import FluidObjectCfg

pytestmark = pytest.mark.no_sim
ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"
)


def test_fluid_deployment_resolves_sources_and_observation_entities():
    cfg = config_to_cfg(load_config(TASK), source_path=TASK)
    assert cfg.sim_cfg.render_cfg.renderer == "hybrid"
    assert cfg.events.record_camera.params["per_env"] is True
    assert cfg.events.record_camera.params["fps"] is None
    assert cfg.events.record_camera.interval_step == 1
    assert len(cfg.fluid_object) == 1
    fluid = cfg.fluid_object[0]
    assert isinstance(fluid, FluidObjectCfg)
    assert Path(fluid.spawn.file_path).is_absolute()
    assert fluid.spawn.sample(fluid.particle_spacing).shape == (76538, 3)
    assert cfg.observations.received_volume.extra["shape"] == [1]
    assert all(
        isinstance(item, SceneEntityCfg)
        for item in cfg.observations.received_volume.params["entity_cfgs"]
    )
    cup = next(item for item in cfg.background if item.uid == "cup")
    assert Path(cup.visual_shape.fpath).is_absolute()
    assert cup.shape.collision.force_sdf is False


def test_fluid_component_rejects_duplicate_inline_ownership():
    config = load_config(TASK)
    config["fluid_object"] = []
    with pytest.raises(ValueError, match="owns|duplicate|must not"):
        config_to_cfg(config, source_path=TASK)


def test_region_volume_uses_reference_pose_and_keeps_state_unchanged():
    state = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [1.5, 2.0, 3.0, 0.0, 0.0, 0.0]],
            [[10.0, 2.0, 3.0, 0.0, 0.0, 0.0], [10.5, 2.0, 3.0, 0.0, 0.0, 0.0]],
        ]
    )
    poses = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]
    )
    fluid = SimpleNamespace(
        data=SimpleNamespace(particle_state_w=state, particle_pos_w=state[..., :3]),
        cfg=SimpleNamespace(particle_spacing=0.1),
        particle_mass=0.001,
        num_particles=2,
        nominal_volume=0.002,
    )
    rigid = SimpleNamespace(get_local_pose=lambda: poses)
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        sim=SimpleNamespace(
            get_fluid_object=lambda _: fluid,
            get_rigid_object=lambda _: rigid,
            arena_offsets=torch.tensor([[0.0, 0, 0], [9.0, 0, 0]]),
        ),
    )
    before = state.clone()
    volume = get_fluid_volume_in_region(
        env,
        {},
        [SceneEntityCfg(uid="water"), SceneEntityCfg(uid="receiver")],
        [-0.1] * 3,
        [0.1] * 3,
    )
    torch.testing.assert_close(volume, torch.full((2, 1), 0.001))
    torch.testing.assert_close(
        get_fluid_summary(env, {}, SceneEntityCfg(uid="water"))[:, 4], torch.ones(2)
    )
    torch.testing.assert_close(state, before)


def test_fluid_replay_rejects_unqualified_recording_before_world_creation():
    from embodichain.lab.gym.envs.fluid_replay import FluidReplayEnv
    from tensordict import TensorDict

    cfg = config_to_cfg(load_config(TASK), source_path=TASK)
    source = {
        "meta": {"num_envs": 1, "num_steps": 2, "lengths": [2]},
        "states": TensorDict({}, batch_size=[1, 2]),
        "actions": torch.zeros(1, 2, 9),
    }
    with pytest.raises(ValueError, match="Task Program"):
        FluidReplayEnv(cfg, trajectory=source)


def _qualified_source(root_position=(0.0, 0.0, 0.0)):
    from tensordict import TensorDict

    root = torch.tensor([*root_position, 0.0, 0.0, 0.0, 1.0]).repeat(1, 2, 1)
    cup = torch.tensor([-0.75, 0.0, 0.552, 0.0, 0.0, 0.0, 1.0]).repeat(1, 2, 1)
    return {
        "meta": {
            "num_envs": 1,
            "num_steps": 2,
            "lengths": [2],
            "dt": 0.05,
            "rigid_object_uids": ["cup"],
            "fluid_source": {
                "kind": "task_program",
                "accepted": True,
                "program_hash": "test-program",
            },
        },
        "states": TensorDict(
            {
                "robot": TensorDict(
                    {"qpos": torch.zeros(1, 2, 9), "root_pose": root}, batch_size=[1, 2]
                ),
                "rigid_objects": TensorDict(
                    {"cup": TensorDict({"pose": cup}, batch_size=[1, 2])},
                    batch_size=[1, 2],
                ),
            },
            batch_size=[1, 2],
        ),
        "actions": torch.zeros(1, 2, 9),
    }


def test_replay_adopts_source_root_before_scene_creation(monkeypatch):
    from embodichain.lab.gym.envs.fluid_replay import FluidReplayEnv
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    cfg = config_to_cfg(load_config(TASK), source_path=TASK)
    cfg.dataset = {}  # Empty mapping is a supported Python configuration.
    captured = []

    def initialize(self, cfg, **kwargs):
        captured.append(cfg)
        self.robot = SimpleNamespace(dof=9)
        self.sim = SimpleNamespace(physics=SimpleNamespace(solver_type="sph"))

    monkeypatch.setattr(EmbodiedEnv, "__init__", initialize)
    env = FluidReplayEnv(cfg, trajectory=_qualified_source((1.0, 2.0, 3.0)))
    assert len(env._replay_time) == cfg.num_envs == 2
    np.testing.assert_array_equal(captured[0].robot.init_local_pose[:3, 3], [1, 2, 3])


@pytest.mark.parametrize("kind", ["trajectory", "actions", "dataset"])
def test_replay_rejects_misleading_policy_recording(kind):
    from embodichain.lab.gym.envs.fluid_replay import FluidReplayEnv

    cfg = config_to_cfg(load_config(TASK), source_path=TASK)
    if kind == "trajectory":
        cfg.record_trajectory = True
    elif kind == "actions":
        cfg.actions = SimpleNamespace(control="unused")
    else:
        cfg.dataset = SimpleNamespace(recorder="unused")
    with pytest.raises(ValueError, match="recording"):
        FluidReplayEnv(cfg, trajectory=_qualified_source())


def test_replay_rejects_moving_root_before_scene_creation():
    from embodichain.lab.gym.envs.fluid_replay import FluidReplayEnv

    cfg = config_to_cfg(load_config(TASK), source_path=TASK)
    source = _qualified_source()
    source["states"]["robot", "root_pose"][0, 1, 0] = 0.2
    with pytest.raises(ValueError, match="fixed-root"):
        FluidReplayEnv(cfg, trajectory=source)


def test_replay_exhaustion_requires_reset():
    from embodichain.lab.gym.envs.fluid_replay import FluidReplayEnv

    env = object.__new__(FluidReplayEnv)
    env._replay_time = np.array([2.0, 2.1])
    env.replay_duration = 2.0
    with pytest.raises(RuntimeError, match="reset"):
        env.step(None)


def test_refill_transforms_selected_container_pose_into_world_frame():
    from embodichain.lab.gym.envs.managers.events import reset_fluid_in_container

    recorded = []
    fluid = SimpleNamespace(
        cfg=FluidObjectCfg(uid="water", init_pos=(0, 0, 0.1)),
        reset_to_pose=lambda pose, ids: recorded.append((pose, ids)),
    )
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    env = SimpleNamespace(
        device=torch.device("cpu"),
        sim=SimpleNamespace(
            get_fluid_object=lambda _: fluid,
            get_rigid_object=lambda _: SimpleNamespace(
                get_local_pose=lambda **_: poses
            ),
            arena_offsets=torch.tensor([[0.0, 0, 0], [9.0, 0, 0]]),
        ),
    )
    reset_fluid_in_container(
        env, [1], [SceneEntityCfg(uid="water"), SceneEntityCfg(uid="cup")]
    )
    pose, ids = recorded[0]
    torch.testing.assert_close(pose[:, :3], torch.tensor([[10.0, 2.0, 3.1]]))
    assert ids.tolist() == [1]
