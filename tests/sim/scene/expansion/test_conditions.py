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

"""Anchored layout restrictions preserve geometry, ownership and addressed RNG."""

from __future__ import annotations
from dataclasses import replace
import numpy as np
import pytest

from embodichain.lab.sim.scene.expansion import (
    SceneLayoutCondition,
    prepare_scene_randomization,
    generate_scene_variants,
    save_scene_variants,
    load_scene_variants,
)
from scene_layout_support import make_source, make_policy


def _poses(variant):
    return {c.entity_uid: np.asarray(c.arena_pose) for c in variant.pose_changes}


def test_condition_owns_anchors_and_leaves_prepared_and_other_samples_independent():
    prepared = prepare_scene_randomization(make_source(), make_policy())
    anchor = np.array(prepared.source.poses["tray"])
    anchor[0, 3] += 0.1
    condition = SceneLayoutCondition({"tray": anchor})
    anchor[0, 3] = 99
    original_policy = prepared.policy
    result = generate_scene_variants(
        prepared, conditions=condition, seed=7, candidate_count=4
    )
    assert result.status == "completed"
    for variant in result.variants:
        np.testing.assert_allclose(
            _poses(variant)["tray"], condition.poses["tray"], rtol=0, atol=1e-12
        )
    assert not np.allclose(
        _poses(result.variants[0])["mug"], _poses(result.variants[-1])["mug"]
    )
    assert prepared.policy == original_policy
    projection = condition.to_dict()
    projection["tray"][0][3] = 77
    assert condition.to_dict()["tray"][0][3] == pytest.approx(-0.2)
    with pytest.raises(TypeError):
        condition.poses["tray"] = np.eye(4)


def test_member_anchor_requires_pinned_ancestors_and_preserves_full_pose():
    prepared = prepare_scene_randomization(make_source(), make_policy())
    first = generate_scene_variants(prepared, seed=9, candidate_count=1).variants[0]
    poses = _poses(first)
    with pytest.raises(ValueError, match="ancestor"):
        generate_scene_variants(
            prepared, conditions=SceneLayoutCondition({"mug": poses["mug"]})
        )
    condition = SceneLayoutCondition(poses)
    batch = generate_scene_variants(
        prepared, conditions=condition, candidate_count=2, max_attempts=10
    )
    assert batch.status == "completed"
    for variant in batch.variants:
        for uid, pose in poses.items():
            np.testing.assert_allclose(_poses(variant)[uid], pose, rtol=0, atol=1e-10)


@pytest.mark.parametrize("size", [1, 7, 64])
def test_condition_prefix_continuation_and_persistence(size, tmp_path):
    source = make_source()
    condition = SceneLayoutCondition({"tray": source.poses["tray"]})
    prepared = prepare_scene_randomization(source, make_policy(batch_size=size))
    one = generate_scene_variants(
        prepared, conditions=condition, seed=13, candidate_count=3
    )
    two = generate_scene_variants(
        prepared,
        conditions=condition,
        seed=13,
        candidate_count=2,
        start_ordinal=one.next_ordinal,
    )
    together = generate_scene_variants(
        prepared, conditions=condition, seed=13, candidate_count=5
    )
    assert tuple(one.variants) + tuple(two.variants) == tuple(together.variants)
    reference = generate_scene_variants(
        prepare_scene_randomization(source, make_policy(batch_size=1)),
        conditions=condition,
        seed=13,
        candidate_count=5,
    )
    assert tuple(reference.variants) == tuple(together.variants)
    saved = save_scene_variants(together, tmp_path / "layouts.json")
    loaded = load_scene_variants(saved)
    assert tuple(loaded.variants) == tuple(together.variants)


def test_bad_anchor_rejects_without_relaxing_domain_or_permissions():
    source = make_source()
    prepared = prepare_scene_randomization(
        source,
        make_policy(group_xy_offset_bounds_m={"tray": [[-0.1, -0.1], [0.1, 0.1]]}),
    )
    outside = np.array(source.poses["tray"])
    outside[0, 3] += 0.2
    with pytest.raises(ValueError, match="bounds"):
        generate_scene_variants(
            prepared, conditions=SceneLayoutCondition({"tray": outside})
        )
    tilted = np.array(source.poses["tray"])
    tilted[0, 0] = -1
    tilted[1, 1] = -1
    with pytest.raises(ValueError, match="rotation"):
        generate_scene_variants(
            prepared, conditions=SceneLayoutCondition({"tray": tilted})
        )
    fixed = np.array(source.poses["z_target"])
    fixed[0, 3] += 0.01
    with pytest.raises(ValueError, match="fixed"):
        generate_scene_variants(
            prepared, conditions=SceneLayoutCondition({"z_target": fixed})
        )
    with pytest.raises(ValueError, match="Unknown"):
        generate_scene_variants(
            prepared, conditions=SceneLayoutCondition({"missing": np.eye(4)})
        )


def test_geometrically_impossible_anchor_returns_exact_exhaustion_without_repair():
    source = make_source()
    prepared = prepare_scene_randomization(source, make_policy())
    overlap = np.array(source.poses["tray"])
    overlap[:2, 3] = np.array(source.poses["z_target"])[:2, 3]
    result = generate_scene_variants(
        prepared,
        conditions=SceneLayoutCondition({"tray": overlap}),
        candidate_count=1,
        max_attempts=9,
    )
    assert result.status == "budget_exhausted" and not result.variants
    assert result.attempts == 9 and result.next_ordinal == 9


def test_anchor_reserves_room_before_sampling_an_earlier_sibling():
    source = make_source()
    prepared = prepare_scene_randomization(
        source,
        make_policy(
            group_targets=["tray", "z_target"],
            intra_group={},
            collision_margin_m=0.02,
            group_xy_offset_bounds_m={
                "tray": [[0, -0.04], [0.35, 0.04]],
                "z_target": [[-0.8, -0.4], [0.4, 0.4]],
            },
        ),
    )
    target = np.array(source.poses["z_target"])
    target[:2, 3] = [-0.3, 0.0]
    result = generate_scene_variants(
        prepared,
        conditions=SceneLayoutCondition({"z_target": target}),
        seed=11,
        candidate_count=8,
        max_attempts=8,
    )
    assert result.status == "completed" and len(result.variants) == 8
    for variant in result.variants:
        poses = _poses(variant)
        np.testing.assert_allclose(poses["z_target"], target, rtol=0, atol=1e-10)
        # Their Y ranges always overlap. Clearance is possible only in X:
        # 0.15 m tray half-width + 0.05 m block half-width + 0.02 m margin.
        assert poses["tray"][0, 3] - target[0, 3] >= 0.22 - 1e-10


def test_no_condition_and_empty_condition_preserve_legacy_exact_sequence():
    prepared = prepare_scene_randomization(make_source(), make_policy())
    a = generate_scene_variants(prepared, seed=41, candidate_count=4)
    b = generate_scene_variants(
        prepared, seed=41, candidate_count=4, conditions=SceneLayoutCondition({})
    )
    assert tuple(a.variants) == tuple(b.variants) and a.attempts == b.attempts


@pytest.mark.gpu
def test_cuda_anchors_keep_cpu_geometry_certificate_and_selected_pose():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = make_source()
    condition = SceneLayoutCondition({"tray": source.poses["tray"]})
    prepared = prepare_scene_randomization(
        source, make_policy(device="cuda:0", batch_size=7)
    )
    result = generate_scene_variants(
        prepared, conditions=condition, seed=13, candidate_count=3
    )
    assert result.status == "completed"
    for candidate in result.variants:
        np.testing.assert_allclose(
            _poses(candidate)["tray"], source.poses["tray"], rtol=0, atol=1e-10
        )


def test_member_anchor_under_fixed_parent_and_invalid_height_are_explicit():
    prepared = prepare_scene_randomization(
        make_source(), make_policy(randomize_groups=False)
    )
    initial = generate_scene_variants(prepared, seed=19, candidate_count=1).variants[0]
    pose = _poses(initial)["mug"]
    result = generate_scene_variants(
        prepared, conditions=SceneLayoutCondition({"mug": pose}), candidate_count=1
    )
    assert result.status == "completed"
    np.testing.assert_allclose(
        _poses(result.variants[0])["mug"], pose, rtol=0, atol=1e-10
    )
    pose[2, 3] += 0.01
    with pytest.raises(ValueError, match="height"):
        generate_scene_variants(
            prepared, conditions=SceneLayoutCondition({"mug": pose})
        )
