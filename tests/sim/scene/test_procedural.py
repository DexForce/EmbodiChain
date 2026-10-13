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

"""Production recipe/composition contracts without model services or physics."""

from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.scene.composition import (
    SceneFragment,
    compose,
    declare_fragment,
)
from embodichain.lab.sim.scene.layout import LayoutCfg, LayoutProblem
from embodichain.lab.sim.scene.procedural import SceneRecipe, generate, check_layout
from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant

pytestmark = pytest.mark.no_sim


def _cfg(uid, pos=(0.0, 0.0, 0.0)):
    return RigidObjectCfg(uid=uid, shape=CubeCfg(size=[0.1, 0.1, 0.1]), init_pos=pos)


def _recipe():
    entities = [
        _cfg("workbench", (0, 0, -0.1)),
        _cfg("bin_a", (-0.7, 0.6, 0.05)),
        _cfg("bin_b", (0.7, 0.6, 0.05)),
        _cfg("part_a", (0, 0, 0.05)),
        _cfg("part_b", (0, 0, 0.05)),
    ]
    problem = LayoutProblem(
        "workbench",
        {
            "bin_a": (-0.7, 0.6),
            "bin_b": (0.7, 0.6),
            "part_a": (0, 0),
            "part_b": (0, 0),
            "keepout": (0, 0),
        },
        {
            "bin_a": (0.15, 0.15),
            "bin_b": (0.15, 0.15),
            "part_a": (0.05, 0.05),
            "part_b": (0.05, 0.05),
            "keepout": (0.2, 0.2),
        },
        {
            uid: ((-1.0, -1.0), (1.0, 1.0))
            for uid in ("bin_a", "bin_b", "part_a", "part_b")
        },
        {"bin_a": (-0.7, 0.6), "bin_b": (0.7, 0.6), "keepout": (0.0, 0.0)},
    )
    return SceneRecipe("fixture:v1:asset-revision-1", SceneFragment(entities), problem)


def test_recipe_candidate_reproducible_independent_of_call_order():
    recipe = _recipe()
    first = generate(recipe, seed=7, ordinal=4)
    generate(recipe, seed=8, ordinal=99)
    again = generate(recipe, seed=7, ordinal=4)
    assert first.checks.passed
    assert first.variant == again.variant
    assert first.variant.variant_id == again.variant.variant_id
    assert {c.entity_uid for c in first.variant.pose_changes} == {"part_a", "part_b"}
    assert (
        generate(recipe, seed=7, ordinal=5).variant.variant_id
        != first.variant.variant_id
    )
    assert recipe.fragment.entities[-1].init_local_pose is None


def test_generated_pose_validation_uses_measured_state_and_keepout():
    recipe = _recipe()
    result = generate(recipe, seed=7)
    assert result.fragment is not None
    from embodichain.lab.sim.scene.composition import _config_pose

    poses = {cfg.uid: _config_pose(cfg) for cfg in result.fragment.entities}
    assert check_layout(recipe, poses).passed
    poses["part_a"][:2, 3] = 0
    assert not check_layout(recipe, poses).passed
    assert not check_layout(recipe, {}).passed


def _group_proxy_recipe() -> SceneRecipe:
    # Two 0.125 m cubes have a joint center at (0.25, 0.125), offset
    # from the root origin. Binary-exact bounds lock the target center
    # at (0.75, 0.125), so the required arena translation is +0.5 m in X.
    root = _cfg("root", (0.125, 0.125, 0.75))
    member = _cfg("member", (0.375, 0.125, 0.875))
    root.shape = CubeCfg(size=[0.125] * 3)
    member.shape = CubeCfg(size=[0.125] * 3)
    return SceneRecipe(
        "group-proxy:v1",
        SceneFragment([_cfg("support"), root, member]),
        LayoutProblem(
            "support",
            {"root": (0.25, 0.125)},
            {"root": (0.1875, 0.0625)},
            {"root": ((0.5625, 0.0625), (0.9375, 0.1875))},
        ),
    )


def test_group_proxy_displacement_preserves_root_offset_without_moving_members() -> (
    None
):
    from embodichain.lab.sim.scene.composition import _config_pose

    recipe = _group_proxy_recipe()
    result = generate(recipe, seed=7)
    assert result.layout.status == "solved"
    assert result.layout.xy_by_uid["root"] == pytest.approx((0.75, 0.125))
    poses = {cfg.uid: _config_pose(cfg) for cfg in result.fragment.entities}
    np.testing.assert_allclose(poses["root"][:3, 3], (0.625, 0.125, 0.75))
    np.testing.assert_allclose(poses["member"][:3, 3], (0.375, 0.125, 0.875))
    assert [change.entity_uid for change in result.variant.pose_changes] == ["root"]
    baseline = {cfg.uid: _config_pose(cfg) for cfg in recipe.fragment.entities}
    np.testing.assert_allclose(baseline["root"][:3, 3], (0.125, 0.125, 0.75))


def test_measured_group_aabb_detects_member_left_behind_by_root_only_application() -> (
    None
):
    from embodichain.lab.sim.scene.composition import _config_pose

    recipe = _group_proxy_recipe()
    result = generate(recipe, seed=7)
    assert result.layout.status == "solved"
    poses = {cfg.uid: _config_pose(cfg) for cfg in result.fragment.entities}
    # Root-only evidence shifts the entire proxy and cannot detect the member
    # still at x=0.375. Its actual cube extends outside the allowed region.
    assert check_layout(recipe, poses).passed
    report = check_layout(
        recipe,
        poses,
        actual_aabbs_xy_by_uid={"root": ((0.3125, 0.0625), (0.6875, 0.1875))},
    )
    assert not report.passed
    assert report.checks[0].status == "failed"
    # After the consumer also translates the member to x=0.875, the measured
    # union matches the solved whole-group bounds and validation passes.
    poses["member"][0, 3] = 0.875
    assert check_layout(
        recipe,
        poses,
        actual_aabbs_xy_by_uid={"root": ((0.5625, 0.0625), (0.9375, 0.1875))},
    ).passed


def test_changed_orientation_does_not_reuse_stale_proxy():
    recipe = _recipe()
    result = generate(recipe, seed=7)
    from embodichain.lab.sim.scene.composition import _config_pose

    poses = {cfg.uid: _config_pose(cfg) for cfg in result.fragment.entities}
    poses["part_a"][:3, :3] = Rotation.from_euler("z", 0.4).as_matrix()
    report = check_layout(recipe, poses)
    assert not report.passed
    assert report.checks[0].status == "unverified"


def test_failed_recipe_does_not_publish_applicable_variant():
    recipe = _recipe()
    problem = recipe.layout
    problem.bounds_xy_by_uid["part_a"] = ((0, 0), (0.01, 0.01))
    failed = generate(SceneRecipe(recipe.recipe_id, recipe.fragment, problem))
    assert not failed.checks.passed
    assert failed.fragment is None and failed.variant is None


def test_recipe_invalid_bounds_fail_explicitly():
    recipe = _recipe()
    problem = recipe.layout
    problem.bounds_xy_by_uid["part_a"] = ((0, 0), (np.inf, 1))
    with pytest.raises(ValueError, match="finite"):
        generate(SceneRecipe(recipe.recipe_id, recipe.fragment, problem))


def test_candidate_compatibility_imports_and_known_identity():
    from embodichain.lab.sim.scene_expansion import SceneVariant as OldVariant
    from embodichain.lab.sim.scene_expansion.contracts import ScenePoseChange as OldPose

    assert OldVariant is SceneVariant and OldPose is ScenePoseChange
    pose = np.eye(4)
    pose[:3, 3] = [0.2, 0.3, 0.4]
    variant = SceneVariant("scene:v1", 0, 1, (ScenePoseChange("part", pose.tolist()),))
    # Independent canonical JSON fixture from the original #749 schema.
    import hashlib, json

    payload = {
        "schema": "scene-variant/v1",
        "parent_scene_id": "scene:v1",
        "seed": 0,
        "ordinal": 1,
        "pose_changes": [["part", pose.tolist()]],
        "metadata": [],
    }
    assert (
        variant.variant_id
        == "scene-variant:"
        + hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )


def _single_recipe():
    return SceneRecipe(
        "single:v1",
        SceneFragment([_cfg("support"), _cfg("item")]),
        LayoutProblem(
            "support",
            {"item": (0.0, 0.0)},
            {"item": (0.1, 0.2)},
            {"item": ((-1.0, -1.0), (1.0, 1.0))},
        ),
    )


def test_measured_aabb_allows_rotated_object_without_changing_recipe():
    recipe = _single_recipe()
    original = recipe.layout
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    poses = {"support": np.eye(4), "item": pose}
    assert not check_layout(recipe, poses).passed
    report = check_layout(
        recipe, poses, actual_aabbs_xy_by_uid={"item": ((-0.2, -0.1), (0.2, 0.1))}
    )
    assert report.passed
    assert recipe.layout == original
    np.testing.assert_array_equal(poses["item"], pose)


def test_measured_aabb_that_exceeds_boundary_fails_without_repair():
    recipe = _single_recipe()
    poses = {"support": np.eye(4), "item": np.eye(4)}
    report = check_layout(
        recipe, poses, actual_aabbs_xy_by_uid={"item": ((-0.1, -0.1), (1.1, 0.1))}
    )
    assert not report.passed
    np.testing.assert_array_equal(poses["item"], np.eye(4))


@pytest.mark.parametrize(
    "boxes",
    [
        {"unknown": ((0.0, 0.0), (1.0, 1.0))},
        {"item": ((0.0, 0.0), (-1.0, 1.0))},
        {"item": ((0.0, 0.0), (float("nan"), 1.0))},
        {"item": (0.0, 0.0, 1.0, 1.0)},
    ],
)
def test_invalid_measured_aabb_fails_explicitly(boxes):
    with pytest.raises(ValueError):
        check_layout(
            _single_recipe(),
            {"support": np.eye(4), "item": np.eye(4)},
            actual_aabbs_xy_by_uid=boxes,
        )


def test_fixed_pose_z_movement_fails_even_with_measured_aabb():
    recipe = _single_recipe()
    problem = replace(recipe.layout, fixed_xy_by_uid={"item": (0.0, 0.0)})
    recipe = SceneRecipe(recipe.recipe_id, recipe.fragment, problem)
    moved = np.eye(4)
    moved[2, 3] = 0.1
    report = check_layout(
        recipe,
        {"support": np.eye(4), "item": moved},
        actual_aabbs_xy_by_uid={"item": ((-0.1, -0.2), (0.1, 0.2))},
    )
    assert report.checks[0].status == "failed"


def test_support_movement_is_unverified_even_with_new_geometry():
    moved = np.eye(4)
    moved[0, 3] = 0.5
    report = check_layout(
        _single_recipe(),
        {"support": moved, "item": np.eye(4)},
        actual_aabbs_xy_by_uid={"support": ((0.0, 0.0), (1.0, 1.0))},
    )
    assert report.checks[0].status == "unverified"


def test_virtual_fixed_position_is_not_overwritten_by_seed_during_actual_check():
    recipe = _single_recipe()
    problem = recipe.layout
    problem.seed_xy_by_uid["keepout"] = (0.0, 0.0)
    problem.half_extents_xy_by_uid["keepout"] = (0.1, 0.1)
    problem.fixed_xy_by_uid["keepout"] = (0.8, 0.8)
    recipe = SceneRecipe(recipe.recipe_id, recipe.fragment, problem)
    assert check_layout(recipe, {"support": np.eye(4), "item": np.eye(4)}).passed


def test_original_fixed_fixed_exemption_survives_actual_check():
    recipe = _single_recipe()
    problem = replace(recipe.layout, skip_fixed_collisions=True)
    for uid in ("zone_a", "zone_b"):
        problem.seed_xy_by_uid[uid] = (0.8, 0.8)
        problem.half_extents_xy_by_uid[uid] = (0.1, 0.1)
        problem.fixed_xy_by_uid[uid] = (0.8, 0.8)
    recipe = SceneRecipe(recipe.recipe_id, recipe.fragment, problem)
    poses = {"support": np.eye(4), "item": np.eye(4)}
    assert check_layout(recipe, poses).passed
    strict = SceneRecipe(
        recipe.recipe_id, recipe.fragment, replace(problem, skip_fixed_collisions=False)
    )
    assert not check_layout(strict, poses).passed


def test_recipe_rejects_fixed_config_positions_that_generation_would_not_apply():
    recipe = _single_recipe()
    with pytest.raises(ValueError, match="authored"):
        SceneRecipe(
            recipe.recipe_id,
            recipe.fragment,
            replace(recipe.layout, fixed_xy_by_uid={"item": (0.7, 0.7)}),
        )


def test_recipe_rejects_unknown_constraint_uid_before_checks_can_drop_it():
    recipe = _single_recipe()
    with pytest.raises(ValueError, match="unknown UID"):
        SceneRecipe(
            recipe.recipe_id,
            recipe.fragment,
            replace(recipe.layout, fixed_xy_by_uid={"missing": (0.0, 0.0)}),
        )


def test_recipe_rejects_moving_support():
    recipe = _single_recipe()
    problem = recipe.layout
    problem.seed_xy_by_uid["support"] = (0.0, 0.0)
    problem.half_extents_xy_by_uid["support"] = (0.1, 0.1)
    problem.bounds_xy_by_uid["support"] = ((-1.0, -1.0), (1.0, 1.0))
    with pytest.raises(ValueError, match="support cannot"):
        SceneRecipe(recipe.recipe_id, recipe.fragment, problem)


def test_recipe_rejects_negative_geometry_before_sampling():
    recipe = _single_recipe()
    problem = recipe.layout
    problem.half_extents_xy_by_uid["item"] = (-0.1, 0.2)
    with pytest.raises(ValueError, match="nonnegative"):
        SceneRecipe(recipe.recipe_id, recipe.fragment, problem)


def test_recipe_rejects_nonboolean_fixed_pair_policy():
    recipe = _single_recipe()
    with pytest.raises(ValueError, match="boolean"):
        SceneRecipe(
            recipe.recipe_id,
            recipe.fragment,
            replace(recipe.layout, skip_fixed_collisions=1),
        )
