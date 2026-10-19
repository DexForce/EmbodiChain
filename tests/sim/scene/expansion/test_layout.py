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

from dataclasses import replace
import json

import numpy as np
import pytest
import torch
import trimesh

from embodichain.lab.sim.scene.expansion import (
    SceneLayoutInput,
    SceneRandomizationPolicyCfg,
    ScenePoseChange,
    SceneVariant,
    apply_scene_variants,
    generate_scene_variants,
    load_scene_variants,
    prepare_scene_randomization,
    save_scene_variants,
    scene_layout_input_from_sim,
)


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


def test_owned_geometry_and_support_topology() -> None:
    source = make_source()
    with pytest.raises(ValueError):
        source.vertices["mug"][0, 0] = 10
    with pytest.raises(ValueError, match="acyclic"):
        replace(
            source,
            parent_by_uid={"tray": "mug", "mug": "tray", "z_target": "workbench"},
        )


@pytest.mark.parametrize("batch_size", [1, 7, 64])
def test_prefix_fixed_group_and_subtree_sampling(batch_size: int) -> None:
    source = make_source()
    expected = generate_scene_variants(
        prepare_scene_randomization(source, make_policy(batch_size=1)),
        seed=27,
        candidate_count=8,
        max_attempts=5000,
    )
    result = generate_scene_variants(
        prepare_scene_randomization(source, make_policy(batch_size=batch_size)),
        seed=27,
        candidate_count=8,
        max_attempts=5000,
    )
    assert result.status == "completed"
    assert result.attempts == expected.attempts
    assert [v.variant_id for v in result.variants] == [
        v.variant_id for v in expected.variants
    ]
    assert all(
        {c.entity_uid for c in v.pose_changes} == {"tray", "mug"}
        for v in result.variants
    )
    offsets = [np.asarray(v.pose_changes[0].arena_pose)[:2, 3] for v in result.variants]
    assert not np.allclose(offsets[0], offsets[-1])


def test_continuation_consumes_same_logical_stream() -> None:
    prepared = prepare_scene_randomization(make_source(), make_policy())
    first = generate_scene_variants(prepared, seed=5, candidate_count=3)
    second = generate_scene_variants(
        prepared, seed=5, candidate_count=3, start_ordinal=first.next_ordinal
    )
    together = generate_scene_variants(prepared, seed=5, candidate_count=6)
    assert [v.variant_id for v in (*first.variants, *second.variants)] == [
        v.variant_id for v in together.variants
    ]


def test_group_only_and_intra_only() -> None:
    source = make_source()
    for policy in (make_policy(intra_group=None), make_policy(randomize_groups=False)):
        result = generate_scene_variants(
            prepare_scene_randomization(source, policy), candidate_count=3
        )
        assert result.status == "completed"
        if not policy.get("randomize_groups", True):
            assert all(
                [c.entity_uid for c in v.pose_changes] == ["mug"]
                for v in result.variants
            )


def test_fixed_descendant_permission_is_not_implicitly_relaxed() -> None:
    source = replace(make_source(), movable_entity_ids=("tray",))
    with pytest.raises(ValueError, match="fixed"):
        prepare_scene_randomization(source, make_policy())


@pytest.mark.parametrize("storage", ["json", "packed"])
def test_v4_roundtrip_and_full_validation_before_consumption(
    tmp_path, storage: str
) -> None:
    source = make_source()
    result = generate_scene_variants(
        prepare_scene_randomization(source, make_policy()), candidate_count=5
    )
    run = save_scene_variants(result, tmp_path / "run", storage=storage)
    loaded = load_scene_variants(run, scene_input=source)
    assert [v.variant_id for v in loaded.variants] == [
        v.variant_id for v in result.variants
    ]
    candidate = next((run / "layouts").iterdir())
    values = json.loads(candidate.read_text())
    values[0]["poses"]["mug"][0][3] = 200
    candidate.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="changed"):
        load_scene_variants(run)


def test_selected_rows_have_independent_poses_and_unchanged_other_state() -> None:
    source = make_source()
    sim = SimPort(source)
    result = generate_scene_variants(
        prepare_scene_randomization(source, make_policy()), candidate_count=2
    )
    applied = apply_scene_variants(
        sim, result.variants, env_ids=[3, 1], movable_entity_ids=("tray", "mug")
    )
    assert applied.env_ids == (3, 1)
    assert sim.time == 0 and sim.syncs == 1
    for uid in ("tray", "mug"):
        entity = sim.entities[uid]
        assert torch.equal(entity.poses[0], torch.tensor(np.array(source.poses[uid])))
        assert torch.equal(entity.poses[2], entity.poses[0])
        assert torch.equal(entity.velocity[[0, 2]], torch.ones(2, 3))
        assert torch.equal(entity.velocity[[1, 3]], torch.zeros(2, 3))
    assert not sim.entities["z_target"].writes


def test_all_rows_preflight_before_any_write() -> None:
    source = make_source()
    sim = SimPort(source)
    valid = generate_scene_variants(
        prepare_scene_randomization(source, make_policy()), candidate_count=1
    ).variants[0]
    invalid = SceneVariant(
        source.scene_id, 0, 99, (ScenePoseChange("missing", np.eye(4)),)
    )
    with pytest.raises(ValueError, match="permitted"):
        apply_scene_variants(
            sim, [valid, invalid], env_ids=[0, 1], movable_entity_ids=("tray", "mug")
        )
    assert not any(entity.writes for entity in sim.entities.values())


def test_runtime_source_uses_complete_scaled_geometry() -> None:
    source = make_source()
    extracted = scene_layout_input_from_sim(
        SimPort(source),
        support_uid=source.support_uid,
        parent_by_uid=source.parent_by_uid,
        scene_id=source.scene_id,
    )
    assert extracted.fingerprint == source.fingerprint


@pytest.mark.gpu
def test_cuda_matches_cpu_geometry_and_prefix() -> None:
    source = make_source()
    cpu = generate_scene_variants(
        prepare_scene_randomization(source, make_policy()), seed=12, candidate_count=8
    )
    gpu = generate_scene_variants(
        prepare_scene_randomization(source, make_policy(device="cuda:0", batch_size=1)),
        seed=12,
        candidate_count=8,
    )
    assert cpu.attempts == gpu.attempts
    for left, right in zip(cpu.variants, gpu.variants):
        for first, second in zip(left.pose_changes, right.pose_changes):
            np.testing.assert_allclose(
                first.arena_pose, second.arena_pose, atol=1e-10, rtol=0
            )
