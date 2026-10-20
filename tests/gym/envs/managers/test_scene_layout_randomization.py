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

from copy import deepcopy
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs.managers import EventCfg, SceneEntityCfg
from embodichain.lab.gym.envs.managers.event_manager import EventManager
from embodichain.lab.gym.envs.managers.randomization import ExpandSceneLayout
from embodichain.lab.sim.scene.expansion import (
    generate_scene_variants,
    prepare_scene_randomization,
    save_scene_variants,
)
import numpy as np
import trimesh
from embodichain.lab.sim.scene.expansion import SceneLayoutInput


def make_source() -> SceneLayoutInput:
    meshes = {
        "workbench": trimesh.creation.box([1.5, 1.2, 0.1]),
        "tray": trimesh.creation.box([0.3, 0.3, 0.02]),
        "mug": trimesh.creation.box([0.04, 0.04, 0.08]),
        "z_target": trimesh.creation.box([0.1, 0.1, 0.1]),
    }
    poses = {uid: np.eye(4) for uid in meshes}
    poses["tray"][:3, 3] = [-0.3, 0, 0.061]
    poses["mug"][:3, 3] = [-0.3, 0, 0.111]
    poses["z_target"][:3, 3] = [0.4, 0.3, 0.101]
    return SceneLayoutInput(
        "authored-test",
        "workbench",
        poses,
        {uid: mesh.vertices for uid, mesh in meshes.items()},
        {uid: mesh.faces for uid, mesh in meshes.items()},
        {"tray": "workbench", "mug": "tray", "z_target": "workbench"},
    )


def make_policy(**kwargs: object) -> dict:
    return {
        "group_targets": ["tray"],
        "batch_size": 7,
        "intra_group": {"targets": {"mug": {}}},
        **kwargs,
    }


class RigidPort:
    def __init__(self, uid: str, source: SceneLayoutInput, count: int) -> None:
        self.uid = uid
        self.source = source
        self.poses = (
            torch.tensor(np.array(source.poses[uid])).expand(count, 4, 4).clone()
        )
        self.velocity = torch.ones(count, 3)
        self.writes = []

    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        assert to_matrix
        return self.poses.clone()

    def get_vertices(self, env_ids: list[int], scale: bool = False) -> torch.Tensor:
        assert scale
        return torch.tensor(np.array(self.source.vertices[self.uid])).expand(
            len(env_ids), -1, -1
        )

    def get_triangles(self, env_ids: list[int]) -> torch.Tensor:
        return torch.tensor(np.array(self.source.faces[self.uid])).expand(
            len(env_ids), -1, -1
        )

    def get_body_scale(self) -> torch.Tensor:
        return torch.ones(len(self.poses), 3)

    def clear_dynamics(self, env_ids: list[int]) -> None:
        self.velocity[env_ids] = 0

    def set_local_pose(self, poses: torch.Tensor, env_ids: list[int]) -> None:
        self.writes.append(tuple(env_ids))
        self.poses[env_ids] = poses.to(self.poses.dtype)


class SimPort:
    def __init__(self, source: SceneLayoutInput, count: int = 4) -> None:
        self.num_envs = count
        self.device = torch.device("cpu")
        self.asset_uids = list(source.poses)
        self.entities = {uid: RigidPort(uid, source, count) for uid in source.poses}
        self.time = 0
        self.syncs = 0

    def get_rigid_object_uid_list(self) -> list[str]:
        return list(self.entities)

    def get_rigid_object(self, uid: str) -> RigidPort:
        return self.entities[uid]

    def sync_render_state(self) -> None:
        self.syncs += 1


def make_env(*, source: dict | None = None, extra: bool = False) -> SimpleNamespace:
    scene = make_source()
    env = SimpleNamespace(
        num_envs=4,
        episode_context=EpisodeContext(4),
        device=torch.device("cpu"),
        cfg=SimpleNamespace(seed=17),
        sim=SimPort(scene),
        _profiler=None,
        event_manager=None,
        _demo_episode_metadata=[{} for _ in range(4)],
    )
    config = {
        "layout": EventCfg(
            func=ExpandSceneLayout,
            mode="reset",
            params={
                "entity_cfg": SceneEntityCfg(uid=scene.support_uid),
                "entity_cfgs": [SceneEntityCfg(uid=uid) for uid in scene.parent_by_uid],
                "parent_by_uid": dict(scene.parent_by_uid),
                "source": source or {"kind": "online"},
                "policy": make_policy(),
                "candidate_count": 8,
                "max_attempts": 1000,
            },
        )
    }
    if extra:
        config["unrelated"] = EventCfg(
            func=lambda env, env_ids: torch.rand(5), mode="reset"
        )
    env.event_manager = EventManager(config, env)
    return env


def apply(env, rows=(1, 3)) -> list:
    env.episode_context._begin_episode(list(rows))
    env.event_manager.reset(env_ids=list(rows))
    env.event_manager.apply("reset", env_ids=list(rows))
    env.episode_context._finish_initialization(list(rows))
    return env.episode_context.snapshot()


def test_event_handles_partial_reset_without_task_program() -> None:
    env = make_env()
    metadata = apply(env)
    assert not metadata[0]["initialized"] and not metadata[2]["initialized"]
    assert (
        metadata[1]["sources"]["events/layout"]["metadata"]["variant_id"]
        != metadata[3]["sources"]["events/layout"]["metadata"]["variant_id"]
    )
    assert env._demo_episode_metadata[1] == {}
    assert env.episode_context.filter_recordable().tolist() == [1, 3]
    assert env.sim.time == 0


def test_repeated_seed_rewinds_candidates_without_reextracting_geometry() -> None:
    env = make_env()
    env.event_manager.set_seed(17)
    first = apply(env)
    functor = env.event_manager.get_functor_cfg("layout").func
    geometry = functor._prepared
    second = apply(env)
    assert (
        first[1]["sources"]["events/layout"]["metadata"]["variant_id"]
        != second[1]["sources"]["events/layout"]["metadata"]["variant_id"]
    )
    revision = env.event_manager.seed_revision
    env.event_manager.set_seed(17)
    assert env.event_manager.seed_revision == revision + 1
    repeated = apply(env)
    assert [r["sources"] for r in first] == [r["sources"] for r in repeated]
    assert functor._prepared is geometry


def test_unrelated_event_does_not_consume_layout_random_stream() -> None:
    left, right = make_env(), make_env(extra=True)
    for _ in range(3):
        assert apply(left) == apply(right)


def make_disjoint_layout_env() -> SimpleNamespace:
    env = make_env()
    first = deepcopy(env.event_manager.get_functor_cfg("layout").params)
    first["parent_by_uid"].pop("z_target")
    first["entity_cfgs"] = [SceneEntityCfg(uid=uid) for uid in first["parent_by_uid"]]
    second = {
        **deepcopy(first),
        "entity_cfgs": [SceneEntityCfg(uid="z_target")],
        "parent_by_uid": {"z_target": "workbench"},
        "policy": {"group_targets": ["z_target"]},
    }
    env.event_manager = EventManager(
        {
            "tableware": EventCfg(func=ExpandSceneLayout, mode="reset", params=first),
            "target": EventCfg(func=ExpandSceneLayout, mode="reset", params=second),
        },
        env,
    )
    return env


def test_disjoint_layout_events_retain_each_named_provenance() -> None:
    env = make_disjoint_layout_env()
    metadata = apply(env)
    assert not metadata[0]["initialized"] and not metadata[2]["initialized"]
    for row in (1, 3):
        assert set(metadata[row]["sources"]) == {"events/tableware", "events/target"}
        for name in ("tableware", "target"):
            event = metadata[row]["sources"]["events/" + name]["metadata"]
            assert event["variant_id"] and event["source_sha256"]
            assert event["geometry_status"] == "passed"
        assert env._demo_episode_metadata[row] == {}
    env.event_manager.get_functor_cfg("tableware").func.reset([1])
    assert env.episode_context.snapshot() == metadata
    env.event_manager.set_seed(17)
    assert [r["sources"] for r in apply(env)] == [r["sources"] for r in metadata]


def test_one_layout_success_cannot_clear_another_layout_failure() -> None:
    env = make_disjoint_layout_env()
    apply(env)
    target = env.event_manager.get_functor_cfg("target").func
    target._pools.clear()
    target._available = 0
    target._candidate_count = target._max_attempts = 1
    target.reset([1, 3])
    with pytest.raises(ValueError, match="max_attempts"):
        target(env, [1, 3])
    tableware = env.event_manager.get_functor_cfg("tableware").func
    tableware.reset([1, 3])
    tableware(env, [1, 3])
    assert env.episode_context.filter_recordable().tolist() == []
    for row in (1, 3):
        assert not env.episode_context.snapshot([row])[0]["recordable"]


def test_saved_source_cycles_without_sampling(tmp_path, monkeypatch) -> None:
    result = generate_scene_variants(
        prepare_scene_randomization(make_source(), make_policy()), candidate_count=2
    )
    run = save_scene_variants(result, tmp_path / "run")
    env = make_env(source={"kind": "saved", "layout_dir": str(run)})
    import embodichain.lab.gym.envs.managers.randomization._scene_layout as module

    monkeypatch.setattr(
        module,
        "generate_scene_variants",
        lambda *args, **kwargs: pytest.fail("Replay resampled layouts"),
    )
    first = apply(env)
    second = apply(env)
    assert [r["sources"] for r in first] == [r["sources"] for r in second]
    assert (
        first[1]["sources"]["events/layout"]["metadata"]["variant_id"]
        == result.variants[0].variant_id
    )


def test_budget_failure_blocks_selected_recordings() -> None:
    env = make_env()
    functor = env.event_manager.get_functor_cfg("layout").func
    functor._max_attempts = 1
    functor._candidate_count = 1
    with pytest.raises(ValueError, match="max_attempts"):
        apply(env, rows=(0, 1, 3))
    assert not len(env.episode_context.filter_recordable())
    assert set(
        row["env_id"] for row in env.episode_context.snapshot() if row["rejections"]
    ) == {0, 1, 3}
    assert not any(entity.writes for entity in env.sim.entities.values())


def test_duplicate_pose_event_is_rejected_before_application() -> None:
    env = make_env()
    from embodichain.lab.gym.envs.managers.randomization import (
        randomize_rigid_object_pose,
    )

    conflicting = EventCfg(
        func=randomize_rigid_object_pose,
        mode="reset",
        params={"entity_cfg": SceneEntityCfg(uid="mug")},
    )
    cfg = {"layout": env.event_manager.get_functor_cfg("layout")}
    # Use a fresh declaration, rather than copying a live native-bound functor.
    cfg["layout"] = EventCfg(
        func=ExpandSceneLayout, mode="reset", params=deepcopy(cfg["layout"].params)
    )
    cfg["pose"] = conflicting
    env.event_manager = EventManager(cfg, env)
    with pytest.raises(ValueError, match="conflicts"):
        apply(env)
    assert not any(entity.writes for entity in env.sim.entities.values())


def test_yaml_event_decodes_through_production_environment_loader(tmp_path) -> None:
    from pathlib import Path
    import yaml
    from embodichain.lab.gym.utils.gym_utils import config_to_cfg

    root = Path(__file__).resolve().parents[4]
    task = (
        root
        / "extensions/embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml"
    )
    component = task.parent / "envs/default.yaml"
    physical = yaml.safe_load(component.read_text())
    physical["simulation"]["background"] = [
        {
            "uid": "bench",
            "body_type": "kinematic",
            "shape": {"shape_type": "Cube", "size": [1.0, 1.0, 0.1]},
        }
    ]
    physical["env"]["events"] = {
        "layout": {
            "func": "ExpandSceneLayout",
            "mode": "reset",
            "params": {
                "entity_cfg": {"uid": "bench"},
                "entity_cfgs": [{"uid": "cube"}],
                "parent_by_uid": {"cube": "bench"},
                "policy": {"backend": "batched", "device": "cpu"},
            },
        }
    }
    env_file = tmp_path / "environment.yaml"
    env_file.write_text(yaml.safe_dump(physical))
    deployment = yaml.safe_load(task.read_text())
    deployment.pop("task_program", None)
    deployment["id"] = "EmbodiedEnv-v1"
    deployment["environment"] = {"component": str(env_file)}
    cfg = config_to_cfg(deployment, source_path=task)
    assert cfg.task_program is None
    assert cfg.events.layout.func is ExpandSceneLayout
    assert isinstance(cfg.events.layout.params["entity_cfg"], SceneEntityCfg)
    assert all(
        isinstance(item, SceneEntityCfg)
        for item in cfg.events.layout.params["entity_cfgs"]
    )


def test_old_layout_functor_name_is_not_exported():
    import embodichain.lab.gym.envs.managers.randomization as randomization

    assert hasattr(randomization, "ExpandSceneLayout")
    assert not hasattr(randomization, "RandomizeSceneLayout")


def test_manager_bound_functor_name_is_read_only():
    env = make_env()
    functor = env.event_manager.get_functor_cfg("layout").func
    assert functor.name == "layout"
    with pytest.raises(AttributeError):
        functor.name = "renamed"
