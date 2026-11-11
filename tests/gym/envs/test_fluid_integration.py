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
    / "extensions/embodichain_tasks/configs/tasks/manipulation/tableware/pour_cup/task.franka.fluid.yaml"
)


@pytest.fixture(autouse=True)
def task_mesh_cache(tmp_path, monkeypatch):
    """Resolve task meshes locally so configuration tests never download them."""
    from types import SimpleNamespace
    from embodichain.data import dataset

    directory = tmp_path / "PourCupAssets"
    directory.mkdir()
    for name in (
        "cup.glb",
        "mug_wall.obj",
        "receiver_wall.obj",
        "fluid_table.obj",
        "fluid_floor.obj",
    ):
        (directory / name).touch()
    original = dataset.get_data_class

    def resolve(name, extra_modules=None):
        if name == "PourCupAssets":
            return lambda: SimpleNamespace(extract_dir=str(directory))
        return original(name, extra_modules)

    monkeypatch.setattr(dataset, "get_data_class", resolve)
    return directory


def test_fluid_deployment_resolves_sources_and_observation_entities():
    cfg = config_to_cfg(load_config(TASK), source_path=TASK)
    assert cfg.sim_cfg.render_cfg.renderer == "fast-rt"
    assert cfg.events.record_camera.params["fps"] is None
    assert cfg.events.record_camera.interval_step == 1
    assert len(cfg.fluid_object) == 1
    fluid = cfg.fluid_object[0]
    assert isinstance(fluid, FluidObjectCfg)
    assert fluid.spawn.kind == "revolved_volume"
    assert fluid.spawn.file_path is None
    assert fluid.spawn.sample(fluid.particle_spacing).shape == (76607, 3)
    assert cfg.observations.received_volume.extra["shape"] == [1]
    assert all(
        isinstance(item, SceneEntityCfg)
        for item in cfg.observations.received_volume.params["entity_cfgs"]
    )
    cup = next(item for item in cfg.rigid_object if item.uid == "cup")
    assert Path(cup.particle_collision_shape.fpath).is_absolute()
    assert cup.body_type == "dynamic"
    assert cfg.task_program is not None


@pytest.mark.parametrize(
    "deployment",
    ("task.franka.fluid.yaml", "task.franka.newton.yaml"),
)
def test_pour_cup_embodiment_mounts_franka_hand(deployment):
    path = TASK.parent / deployment
    robot = config_to_cfg(load_config(path), source_path=path).robot
    assert "hand" in robot.urdf_cfg.components
    assert robot.control_parts["eef"] == [
        "fr3_finger_joint1",
        "fr3_finger_joint2",
    ]
    assert len(robot.init_qpos) == 9


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
