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
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

from embodichain.gen_sim.scene_engine.randomization import generation
from embodichain.gen_sim.scene_engine.randomization.baseline import (
    prepare_scene_baseline,
)
from embodichain.gen_sim.scene_engine.randomization.cfg import (
    SceneRandomizationCfg,
    load_randomization_config,
)
from embodichain.lab.sim.scene.layout import LayoutSolution
from embodichain.lab.sim.scene.procedural import (
    GeneratedScene,
    SceneRecipe,
    generate as shared_generate,
)
from embodichain.lab.sim.scene.validation import SceneCheck, SceneCheckReport

from .test_randomization_baseline import _export


@pytest.fixture
def grouped_export(tmp_path: Path) -> tuple[Path, SceneRandomizationCfg, dict]:
    """Reuse a nested export with asymmetric scaled geometry and rotated roots."""
    root, cfg = _export(tmp_path)
    path = root / "scene_config.json"
    config = json.loads(path.read_text())
    entries = {
        entry["uid"]: entry for entry in config["background"] + config["rigid_object"]
    }
    entries["tray"]["body_scale"] = [1.1, 0.8, 1.2]
    entries["cup"]["body_scale"] = [0.7, 1.3, 0.9]
    entries["bottle"]["body_scale"] = [1.2, 0.8, 0.9]
    entries["bottle"]["init_rot"] = [8.0, -13.0, 25.0]
    path.write_text(json.dumps(config))
    return root, cfg, entries


def _entry_pose(entry: dict) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler(
        "XYZ", entry["init_rot"], degrees=True
    ).as_matrix()
    pose[:3, 3] = entry["init_pos"]
    return pose


def _mesh_aabb(root: Path, entry: dict, pose: np.ndarray) -> np.ndarray:
    """Measure final exported vertices independently of baseline extraction."""
    mesh = trimesh.load(root / entry["shape"]["fpath"], force="mesh")
    points = np.asarray(mesh.vertices) * entry["body_scale"]
    # Exported asset geometry remains Y-up; runtime entity poses are Z-up.
    points = np.column_stack((points[:, 0], -points[:, 2], points[:, 1]))
    points = points @ pose[:3, :3].T + pose[:3, 3]
    return np.asarray((points[:, :2].min(axis=0), points[:, :2].max(axis=0)))


@pytest.mark.parametrize("hull_budget", [None, 1, 16])
def test_generated_fragments_preserve_exported_collision_configuration(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
    hull_budget: int | None,
) -> None:
    root, cfg, _ = grouped_export
    config_path = root / "scene_config.json"
    config = json.loads(config_path.read_text())
    if hull_budget is not None:
        for entry in config["background"] + config["rigid_object"]:
            entry["max_convex_hull_num"] = hull_budget
        config_path.write_text(json.dumps(config))

    result = generation.randomize_scene(cfg, candidate_count=1)

    assert result.status == "completed"
    fragment = result.candidates[0].fragment
    assert fragment is not None
    for entity in fragment.entities:
        collision = entity.shape.collision
        assert collision is not None
        if hull_budget == 1:
            assert collision.approximation == "convex_hull"
        else:
            assert collision.approximation == "convex_decomposition"
            assert collision.max_hulls == (32 if hull_budget is None else hull_budget)
            assert collision.acd_method == "vhacd"


@pytest.mark.parametrize(
    "collision",
    [
        {"approximation": "convex_hull"},
        {
            "approximation": "convex_decomposition",
            "max_hulls": 4,
            "acd_method": "coacd",
        },
    ],
)
def test_explicit_collision_policy_survives_legacy_layout_materialization(
    grouped_export, collision
) -> None:
    root, cfg, _ = grouped_export
    path = root / "scene_config.json"
    data = json.loads(path.read_text())
    for entry in data["background"] + data["rigid_object"]:
        entry["shape"]["collision"] = collision
    path.write_text(json.dumps(data))
    candidate = generation.randomize_scene(cfg, candidate_count=1).candidates[0]
    for entity in candidate.fragment.entities:
        for key, value in collision.items():
            assert getattr(entity.shape.collision, key) == value


def test_generated_groups_move_all_members_and_satisfy_measured_aabbs(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
) -> None:
    root, cfg, entries = grouped_export
    margin = 0.03
    result = generation.randomize_scene(
        cfg, seed=42, candidate_count=8, max_attempts=30, collision_margin_m=margin
    )
    assert result.status == "completed"
    assert len(result.candidates) == 8
    table = result.baseline.data["table"]
    table_center = np.asarray(table["center_xy"])
    table_half = np.asarray(table["size_xy"]) / 2
    table_bounds = np.asarray((table_center - table_half, table_center + table_half))
    source_poses = {uid: _entry_pose(entry) for uid, entry in entries.items()}
    for candidate in result.candidates:
        assert candidate.layout.status == "solved" and candidate.checks.passed
        assert candidate.fragment is not None and candidate.variant is not None
        entities = {entity.uid: entity for entity in candidate.fragment.entities}
        poses = {
            uid: (
                np.asarray(entity.init_local_pose)
                if entity.init_local_pose is not None
                else _entry_pose(
                    {"init_pos": entity.init_pos, "init_rot": entity.init_rot}
                )
            )
            for uid, entity in entities.items()
        }
        changes = {
            change.entity_uid: np.asarray(change.arena_pose)
            for change in candidate.variant.pose_changes
        }
        assert set(entities) == set(entries)
        assert set(changes) == set(entries) - {"table"}
        np.testing.assert_allclose(poses["table"], source_poses["table"], atol=1e-12)
        group_bounds = {}
        for group in result.baseline.data["groups"]:
            root_id = group["root_id"]
            delta = np.asarray(candidate.layout.xy_by_uid[root_id]) - np.asarray(
                group["aabb_2d"]["center_xy"]
            )
            member_ids = [root_id] + [m["entity_id"] for m in group["members"]]
            boxes = []
            for uid in member_ids:
                expected = source_poses[uid].copy()
                expected[:2, 3] += delta
                np.testing.assert_allclose(poses[uid], expected, atol=1e-9)
                np.testing.assert_allclose(changes[uid], expected, atol=1e-9)
                np.testing.assert_allclose(entities[uid].init_pos, expected[:3, 3])
                np.testing.assert_allclose(
                    entities[uid].body_scale, entries[uid]["body_scale"]
                )
                boxes.append(_mesh_aabb(root, entries[uid], poses[uid]))
            bounds = np.asarray(boxes)
            bounds = np.asarray((bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)))
            group_bounds[root_id] = bounds
            assert np.all(bounds[0] >= table_bounds[0] - 1e-7)
            assert np.all(bounds[1] <= table_bounds[1] + 1e-7)
            np.testing.assert_allclose(
                bounds.mean(axis=0), candidate.layout.xy_by_uid[root_id], atol=1e-9
            )
        for first, second in combinations(group_bounds.values(), 2):
            axis_gaps = np.maximum(first[0] - second[1], second[0] - first[1])
            assert np.max(axis_gaps) >= margin - 1e-7


def test_repeat_seed_reproduces_candidates_without_writing_layouts(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
) -> None:
    root, cfg, _ = grouped_export
    prepare_scene_baseline(cfg)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.parent.rglob("*")
        if path.is_file()
    }
    first = generation.randomize_scene(cfg, seed=17, candidate_count=3)
    second = generation.randomize_scene(cfg, seed=17, candidate_count=3)
    other = generation.randomize_scene(cfg, seed=18, candidate_count=3)
    assert first.baseline.cache_hit and second.baseline.cache_hit
    assert first.baseline.data == second.baseline.data
    assert first.attempts == second.attempts
    assert [c.variant for c in first.candidates] == [
        c.variant for c in second.candidates
    ]
    assert [c.layout.xy_by_uid for c in first.candidates] != [
        c.layout.xy_by_uid for c in other.candidates
    ]
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.parent.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("from_yaml", [False, True])
def test_adapter_uses_shared_recipe_sampler_and_solver(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
    monkeypatch: pytest.MonkeyPatch,
    from_yaml: bool,
) -> None:
    root, cfg, _ = grouped_export
    if from_yaml:
        config_path = root.parent / "randomization.yaml"
        config_path.write_text(
            "schema_version: 1\nseed: 13\ncandidate_count: 4\n"
            "max_attempts: 20\ncollision_margin_m: 0.04\n"
        )
        cfg = load_randomization_config(config_path, output_root=root.parent)
    calls = []

    def record_generate(recipe: SceneRecipe, **kwargs: object) -> GeneratedScene:
        direct = shared_generate(recipe, **kwargs)
        calls.append((recipe, kwargs, direct))
        return direct

    monkeypatch.setattr(generation, "generate", record_generate)
    result = (
        generation.randomize_scene(cfg)
        if from_yaml
        else generation.randomize_scene(
            cfg, seed=13, candidate_count=4, collision_margin_m=0.04
        )
    )
    assert len(calls) == result.attempts
    successful = [call for call in calls if call[2].layout.status == "solved"]
    assert len(successful) == len(result.candidates) == 4
    for candidate, (recipe, kwargs, direct) in zip(result.candidates, successful):
        assert not recipe.layout.fixed_xy_by_uid
        assert kwargs["seed"] == 13
        assert kwargs["cfg"].collision_margin_m == 0.04
        assert (
            candidate.variant.parent_scene_id
            == result.baseline.data["source_scene"]["content_sha256"]
        )
        assert (
            dict(candidate.variant.metadata)["baseline_sha256"]
            == result.baseline.data["data_sha256"]
        )
        assert dict(candidate.variant.metadata)["proposal_ordinal"] == str(
            kwargs["ordinal"]
        )
        assert candidate.layout == direct.layout
        repeated = shared_generate(recipe, **kwargs)
        assert repeated.layout == candidate.layout


def _failed_attempt(status: str) -> GeneratedScene:
    return GeneratedScene(
        None,
        None,
        LayoutSolution(status, {}, {}, "Synthetic solver outcome"),
        SceneCheckReport(
            (
                SceneCheck(
                    "scene.layout",
                    "unverified" if status == "unsupported" else "failed",
                ),
            )
        ),
    )


def test_configured_attempt_budget_and_python_overrides(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cfg, _ = grouped_export
    cfg = cfg.replace(
        seed=19, candidate_count=2, max_attempts=3, collision_margin_m=0.03
    )
    calls = []

    def no_solution(recipe: SceneRecipe, **kwargs: object) -> GeneratedScene:
        calls.append(kwargs)
        return _failed_attempt("no_solution")

    monkeypatch.setattr(generation, "generate", no_solution)
    result = generation.randomize_scene(cfg)
    assert result.status == "budget_exhausted" and result.attempts == 3
    assert result.resolved_parameters == {
        "seed": 19,
        "candidate_count": 2,
        "max_attempts": 3,
        "collision_margin_m": 0.03,
    }
    assert [call["ordinal"] for call in calls] == [0, 1, 2]
    assert all(call["seed"] == 19 for call in calls)
    assert all(call["cfg"].collision_margin_m == 0.03 for call in calls)

    calls.clear()
    overridden = generation.randomize_scene(
        cfg, seed=0, max_attempts=2, collision_margin_m=0.0
    )
    assert len(calls) == 2
    assert all(call["seed"] == 0 for call in calls)
    assert all(call["cfg"].collision_margin_m == 0.0 for call in calls)
    assert overridden.resolved_parameters == {
        "seed": 0,
        "candidate_count": 2,
        "max_attempts": 2,
        "collision_margin_m": 0.0,
    }
    cfg.seed = 100
    assert result.resolved_parameters["seed"] == 19
    assert overridden.resolved_parameters["seed"] == 0
    cfg.seed = 19
    assert (cfg.seed, cfg.max_attempts, cfg.collision_margin_m) == (19, 3, 0.03)


def test_failed_attempts_leave_gaps_in_proposals_not_candidate_ordinals(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cfg, _ = grouped_export
    ordinals = []

    def sequenced_generate(recipe: SceneRecipe, **kwargs: object) -> GeneratedScene:
        ordinal = kwargs["ordinal"]
        ordinals.append(ordinal)
        if ordinal in (0, 2):
            return _failed_attempt(
                "no_solution" if ordinal == 0 else "budget_exhausted"
            )
        return shared_generate(recipe, **kwargs)

    monkeypatch.setattr(generation, "generate", sequenced_generate)
    result = generation.randomize_scene(cfg, seed=42, candidate_count=2, max_attempts=4)
    assert result.status == "completed" and result.attempts == 4
    assert ordinals == [0, 1, 2, 3]
    assert [c.variant.ordinal for c in result.candidates] == [0, 1]
    assert [
        dict(c.variant.metadata)["proposal_ordinal"] for c in result.candidates
    ] == ["1", "3"]
    assert result.solver_status_counts == {
        "no_solution": 1,
        "budget_exhausted": 1,
        "solved": 2,
    }


@pytest.mark.parametrize(
    ("solver_status", "run_status", "expected_attempts"),
    [("no_solution", "budget_exhausted", 3), ("unsupported", "unsupported", 1)],
)
def test_failure_budget_and_unsupported_stop_without_candidates(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
    monkeypatch: pytest.MonkeyPatch,
    solver_status: str,
    run_status: str,
    expected_attempts: int,
) -> None:
    _, cfg, _ = grouped_export
    monkeypatch.setattr(
        generation, "generate", lambda *args, **kwargs: _failed_attempt(solver_status)
    )
    result = generation.randomize_scene(cfg, candidate_count=2, max_attempts=3)
    assert result.status == run_status and result.attempts == expected_attempts
    assert result.candidates == ()
    assert result.solver_status_counts == {solver_status: expected_attempts}


def test_budget_exhaustion_retains_only_successful_candidates(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cfg, _ = grouped_export

    def partly_successful(recipe: SceneRecipe, **kwargs: object) -> GeneratedScene:
        if kwargs["ordinal"] == 0:
            return shared_generate(recipe, **kwargs)
        return _failed_attempt("budget_exhausted")

    monkeypatch.setattr(generation, "generate", partly_successful)
    result = generation.randomize_scene(cfg, candidate_count=2, max_attempts=2)
    assert result.status == "budget_exhausted" and result.attempts == 2
    assert len(result.candidates) == 1
    assert result.candidates[0].layout.status == "solved"
    assert result.candidates[0].variant.ordinal == 0
    assert result.solver_status_counts == {"solved": 1, "budget_exhausted": 1}


def test_failed_layout_is_not_published_even_if_a_fragment_is_present(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, cfg, _ = grouped_export

    def failed_with_fragment(recipe: SceneRecipe, **kwargs: object) -> GeneratedScene:
        proposal = shared_generate(recipe, **kwargs)
        failure = _failed_attempt("no_solution")
        return replace(proposal, layout=failure.layout, checks=failure.checks)

    monkeypatch.setattr(generation, "generate", failed_with_fragment)
    result = generation.randomize_scene(cfg, candidate_count=1, max_attempts=1)
    assert result.status == "budget_exhausted" and result.candidates == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 1.5},
        {"candidate_count": 0},
        {"candidate_count": True},
        {"candidate_count": 1.5},
        {"max_attempts": 0},
        {"max_attempts": True},
        {"max_attempts": 1.5},
        {"candidate_count": 3, "max_attempts": 2},
        {"collision_margin_m": -0.01},
        {"collision_margin_m": float("nan")},
        {"collision_margin_m": float("inf")},
        {"collision_margin_m": True},
        {"collision_margin_m": "0.02"},
    ],
)
def test_invalid_controls_fail_before_reading_scene(
    monkeypatch: pytest.MonkeyPatch, overrides: dict
) -> None:
    monkeypatch.setattr(
        generation,
        "prepare_scene_baseline",
        lambda *args, **kwargs: pytest.fail(
            "Invalid controls must not touch scene files"
        ),
    )
    with pytest.raises(ValueError):
        generation.randomize_scene(SceneRandomizationCfg(), **overrides)


def test_oversized_group_fails_before_sampling(
    grouped_export: tuple[Path, SceneRandomizationCfg, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, cfg, _ = grouped_export
    path = root / "scene_config.json"
    config = json.loads(path.read_text())
    next(entry for entry in config["rigid_object"] if entry["uid"] == "bottle")[
        "body_scale"
    ] = [100.0, 1.0, 100.0]
    path.write_text(json.dumps(config))
    monkeypatch.setattr(
        generation,
        "generate",
        lambda *args, **kwargs: pytest.fail("Empty center range must not sample"),
    )
    with pytest.raises(ValueError, match="bottle"):
        generation.randomize_scene(cfg)
