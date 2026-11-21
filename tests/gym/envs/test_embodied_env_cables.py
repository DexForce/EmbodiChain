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

"""Cable scene composition through the public Gym configuration loader."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim.cfg import CableObjectCfg, CablePhysicsCfg

pytestmark = pytest.mark.no_sim


@pytest.mark.parametrize("component", [False, True], ids=["inline", "component"])
def test_cable_config_is_declared_before_environment_preparation(
    tmp_path: Path, component: bool
) -> None:
    cable = {
        "uid": "wire",
        "points": [[0, 0, 1], [0.2, 0, 1]],
        "radius": 0.01,
        "fixed_segments": [0],
        "attrs": {"bend_stiffness": 0.1},
    }
    config = {
        "id": "EmbodiedEnv-v1",
        "physics": "newton",
        "env": {},
        "robot": {"uid": "robot"},
        "cable_object": [cable],
    }
    if component:
        (tmp_path / "env.yaml").write_text(
            yaml.safe_dump(
                {
                    "environment_id": "EmbodiedEnv-v1",
                    "physics": "newton",
                    "env": {},
                    "simulation": {"cable_object": [cable]},
                }
            )
        )
        config = {
            "id": "EmbodiedEnv-v1",
            "environment": {"component": "env.yaml"},
            "robot": {"uid": "robot"},
        }
    cfg = config_to_cfg(config, source_path=tmp_path / "task.yaml")
    assert len(cfg.cable_object) == 1
    assert isinstance(cfg.cable_object[0], CableObjectCfg)
    assert isinstance(cfg.cable_object[0].attrs, CablePhysicsCfg)
    assert cfg.cable_object[0].attrs.bend_stiffness == 0.1

    env = object.__new__(EmbodiedEnv)
    env.cfg = cfg
    env.sim = MagicMock()
    env._setup_interactive_objects()
    env.sim.add_cable_object.assert_called_once_with(cfg=cfg.cable_object[0])
    env.sim.prepare.assert_not_called()


@pytest.mark.parametrize("resource", ["registry", "relative"])
@pytest.mark.parametrize("component", [False, True])
def test_cable_visual_mesh_resolves_from_dataset_or_declaring_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource: str, component: bool
) -> None:
    from embodichain import data

    owner = tmp_path / "environment" if component else tmp_path
    owner.mkdir(exist_ok=True)
    mesh = owner / "wire.obj"
    mesh.write_text("v 0 0 0\n")
    reference = "CableAssets/wire.obj" if resource == "registry" else "./wire.obj"
    resolved = []

    def get_data_path(path: str) -> str:
        resolved.append(path)
        assert path == (reference if resource == "registry" else str(mesh))
        return str(mesh)

    monkeypatch.setattr(data, "get_data_path", get_data_path)
    cable = {
        "uid": "wire",
        "points": [[0, 0, 1], [0.2, 0, 1]],
        "radius": 0.01,
        "visual_shape": {"fpath": reference},
    }
    config = {
        "id": "EmbodiedEnv-v1",
        "physics": "newton",
        "env": {},
        "robot": {"uid": "robot"},
        "cable_object": [cable],
    }
    if component:
        (owner / "env.yaml").write_text(
            yaml.safe_dump(
                {
                    "environment_id": "EmbodiedEnv-v1",
                    "physics": "newton",
                    "env": {},
                    "simulation": {"cable_object": [cable]},
                }
            )
        )
        config = {
            "id": "EmbodiedEnv-v1",
            "environment": {"component": "environment/env.yaml"},
            "robot": {"uid": "robot"},
        }
    cfg = config_to_cfg(config, source_path=tmp_path / "task.yaml")
    assert cfg.cable_object[0].visual_shape.fpath == str(mesh)
    assert len(resolved) == 1
    assert cable["visual_shape"]["fpath"] == reference


def test_cable_visual_mesh_loads_inline_geometry() -> None:
    vertices = [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.01, 0.0]]
    triangles = [[0, 1, 2]]
    cable = {
        "uid": "wire",
        "points": [[0, 0, 1], [0.2, 0, 1]],
        "radius": 0.01,
        "visual_shape": {"vertices": vertices, "triangles": triangles},
    }
    cfg = config_to_cfg(
        {
            "id": "EmbodiedEnv-v1",
            "physics": "newton",
            "env": {},
            "robot": {"uid": "robot"},
            "cable_object": [cable],
        }
    )
    mesh = cfg.cable_object[0].visual_shape
    assert mesh.fpath is None
    assert mesh.vertices == vertices
    assert mesh.triangles == triangles


@pytest.mark.parametrize("uids", [None, ["wire"]])
def test_cable_recording_uses_shared_observation_history_and_t_plus_one_slots(
    uids,
) -> None:
    from types import SimpleNamespace

    import gymnasium.spaces
    import torch
    from tensordict import TensorDict

    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers import (
        ObservationCfg,
        ObservationManager,
        SceneEntityCfg,
    )
    from embodichain.lab.gym.envs.managers.observations import get_entity_state
    from embodichain.lab.gym.utils.gym_utils import (
        _init_raw_rollout,
        build_trajectory_buffer,
    )
    from embodichain.lab.gym.utils.trajectory_state import (
        export_observation_history,
        trajectory_observation_cfgs,
    )

    class CableData:
        def __init__(self, segments: int):
            self.num_segments = segments
            self.state = torch.zeros(2, segments, 13)
            self.state[..., 6] = 1
            self.reads = 0

        @property
        def segment_state_w(self):
            self.reads += 1
            return self.state

    wires = {
        uid: SimpleNamespace(data=CableData(segments))
        for uid, segments in (("wire", 2), ("other", 4))
    }
    pose = torch.zeros(2, 7)
    pose[:, 6] = 1
    robot = SimpleNamespace(
        uid="robot",
        dof=1,
        joint_names=["joint"],
        get_local_pose=lambda to_matrix=False: pose,
        get_qpos=lambda: torch.zeros(2, 1),
        get_qvel=lambda: torch.zeros(2, 1),
    )
    assets = {"robot": robot, **wires}
    context = EpisodeContext(2)
    context._begin_episode([0, 1])
    context._finish_initialization([0, 1])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        robot=robot,
        episode_context=context,
        sim=SimpleNamespace(
            _articulations={},
            _rigid_objects={},
            _cable_objects=wires,
            asset_uids=list(assets),
            get_asset=assets.get,
        ),
    )
    alias = ObservationCfg(
        func=get_entity_state,
        mode="add",
        name="wire_alias",
        params={
            "entity_cfg": SceneEntityCfg(uid="wire"),
            "quantity": "segment_state_w",
        },
        consumers=("policy", "recording"),
    )
    manager = ObservationManager(
        {"alias": alias}, env, recording=trajectory_observation_cfgs(env, uids)
    )
    env.observation_manager = manager
    trajectory = build_trajectory_buffer(
        env,
        max_steps=3,
        num_envs=2,
        device="cpu",
        uids=uids,
        action_space=gymnasium.spaces.Box(-1, 1, shape=(2, 1)),
    )
    storage = TensorDict({"trajectory": trajectory}, batch_size=[2])
    env._rollout_storage = storage
    fields = {
        "/".join(path): value
        for path, value in trajectory["states"].items(
            include_nested=True, leaves_only=True
        )
    }
    _init_raw_rollout(storage, context, fields, trajectory["actions"][:, :2])
    manager.bind_rollout_storage(storage)

    def capture(slot):
        return manager.capture(
            lambda: manager.compute(TensorDict({}, batch_size=[2])),
            episode_context=context,
            timestamp=slot * 0.04,
        )

    for slot in range(3):
        for wire in wires.values():
            wire.data.state[..., 0] = torch.tensor([slot, 100 + slot])[:, None]
        wires["wire"].data.state[1, :, 6] = 0 if slot == 1 else 1
        frame = capture(slot)
        # A policy consumer may mutate its view without corrupting recorded state.
        frame.policy["wire_alias"].add_(1000)
        if slot == 0:
            EmbodiedEnv._record_initial_observation(env, frame, torch.arange(2))
        else:
            env._traj_raw_action = torch.full((2, 1), float(slot))
            EmbodiedEnv._write_trajectory_step(env, frame)

    selected = set(wires) if uids is None else set(uids)
    assert set(trajectory["states", "cables"].keys()) == selected
    for uid, wire in wires.items():
        assert wire.data.reads == (3 if uid in selected else 0)
        if uid in selected:
            state = trajectory["states", "cables", uid, "segment_state_w"]
            assert state.shape == (2, 3, wire.data.num_segments, 13)
            torch.testing.assert_close(
                state[:, :, 0, 0],
                torch.tensor([[0.0, 1.0, 2.0], [100.0, 101.0, 102.0]]),
            )
    field = "cables/wire/segment_state_w"
    assert manager.field_ids["wire_alias"] == manager.field_ids[field]
    descriptor = manager.schema[field]
    assert descriptor.coordinate_frame == "world"
    assert descriptor.representation == "xyz_xyzw_com_lin_vel_ang_vel"
    assert descriptor.units == ("m", "unitless", "m/s", "rad/s")
    window = manager.read_history_window(
        [0], [2], fields=[field], episode_ids=[0], env_ids=[1]
    )
    assert window["fields"][field]["valid"][0].tolist() == [True, False, True]
    exported = export_observation_history(manager)
    assert exported["fields"][field]["descriptor"]["shape"] == (2, 13)
    assert storage["raw_steps"].tolist() == [2, 2]

    context._begin_episode([0])
    context._finish_initialization([0])
    wires["wire"].data.state[0, :, 0] = 99
    EmbodiedEnv._record_initial_observation(env, capture(3), torch.tensor([0]))
    assert storage["raw_steps"].tolist() == [0, 2]
    retained = manager.read_history([2], fields=[field], episode_ids=[0], env_ids=[1])
    assert retained.values[field][0, 0, 0] == 102
    with pytest.raises(ValueError, match="episode"):
        manager.read_history([0], fields=[field], episode_ids=[0], env_ids=[0])
