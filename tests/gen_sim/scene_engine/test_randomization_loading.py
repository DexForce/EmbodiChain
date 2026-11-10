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

"""Load immutable saved layouts and cycle their absolute poses without physics."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh
import yaml

from embodichain.gen_sim.scene_engine.randomization import (
    _geometry,
    baseline,
    generation,
    loading,
    persistence,
)
from embodichain.gen_sim.scene_engine.randomization.generation import (
    SceneRandomizationResult,
)

from .test_randomization_baseline import _export


@pytest.fixture
def saved_pool(tmp_path: Path) -> tuple[Path, Path, SceneRandomizationResult]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, cfg = _export(workspace)
    result = generation.randomize_scene(
        cfg, seed=42, candidate_count=2, max_attempts=10
    )
    assert result.status == "completed"
    run = persistence.save_layouts(result, output_root=workspace)
    return workspace, run / "layouts", result


def _snapshot(root: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _source_poses(workspace: Path) -> dict[str, np.ndarray]:
    config = json.loads((workspace / "scene_export" / "scene_config.json").read_text())
    poses = {}
    for entry in config["background"] + config["rigid_object"]:
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler(
            "XYZ", entry["init_rot"], degrees=True
        ).as_matrix()
        pose[:3, 3] = entry["init_pos"]
        poses[entry["uid"]] = pose
    return poses


def test_load_reconstructs_complete_variants_without_generation_or_writes(
    saved_pool: tuple[Path, Path, SceneRandomizationResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, directory, result = saved_pool
    before = _snapshot(workspace)
    source_poses = _source_poses(workspace)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail(
            "Loading saved layouts must not generate, parse meshes, or write caches"
        )

    monkeypatch.setattr(generation, "generate", forbidden)
    monkeypatch.setattr(baseline, "prepare_scene_baseline", forbidden)
    monkeypatch.setattr(_geometry, "measure_scene_geometry", forbidden)
    monkeypatch.setattr(trimesh, "load", forbidden)
    monkeypatch.setattr(persistence, "_write_atomic", forbidden)

    loaded = loading.load_layouts(directory)

    assert loaded.output_root == workspace
    assert loaded.names == tuple(path.stem for path in sorted(directory.glob("*.json")))
    assert set(loaded.baseline_poses) == set(source_poses)
    for uid, pose in source_poses.items():
        np.testing.assert_allclose(loaded.baseline_poses[uid], pose, atol=1e-12)
    assert len(loaded.variants) == len(result.candidates) == 2
    for actual, expected in zip(loaded.variants, result.candidates, strict=True):
        assert expected.variant is not None
        assert actual.variant_id == expected.variant.variant_id
        assert actual.metadata == expected.variant.metadata
        poses = {change.entity_uid: change.arena_pose for change in actual.pose_changes}
        assert set(poses) == set(source_poses) - {"table"}
        for change in expected.variant.pose_changes:
            np.testing.assert_allclose(
                poses[change.entity_uid], change.arena_pose, atol=1e-12
            )
    assert _snapshot(workspace) == before


@pytest.mark.parametrize("artifact", ["source", "baseline", "policy"])
def test_load_rejects_changed_provenance(
    saved_pool: tuple[Path, Path, SceneRandomizationResult], artifact: str
) -> None:
    workspace, directory, result = saved_pool
    if artifact == "source":
        path = workspace / "scene_export" / "scene_config.json"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path = (
            result.baseline.path
            if artifact == "baseline"
            else directory.parent / "policy_resolved.yaml"
        )
        data = yaml.safe_load(path.read_text())
        if artifact == "baseline":
            data["table"]["center_xy"][0] += 0.1
        else:
            data["seed"] += 1
        path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError):
        loading.load_layouts(directory)


@pytest.mark.parametrize(
    "mutation",
    [
        "syntax",
        "duplicate_key",
        "nan",
        "unknown_group",
        "duplicate_group",
        "missing_group",
        "yaw",
        "bounds",
        "variant_id",
    ],
)
def test_load_rejects_malformed_or_inconsistent_candidate(
    saved_pool: tuple[Path, Path, SceneRandomizationResult], mutation: str
) -> None:
    _, directory, _ = saved_pool
    path = next(iter(sorted(directory.glob("*.json"))))
    raw = path.read_text()
    payload = json.loads(raw)
    if mutation == "syntax":
        raw = "{"
    elif mutation == "duplicate_key":
        raw = raw.replace("{", '{"schema_version": 1,', 1)
    else:
        if mutation == "nan":
            payload["groups"][0]["center_xy"][0] = float("nan")
        elif mutation == "unknown_group":
            payload["groups"][0]["root_id"] = "missing-root"
        elif mutation == "duplicate_group":
            payload["groups"].append(payload["groups"][0])
        elif mutation == "missing_group":
            payload["groups"].pop()
        elif mutation == "yaw":
            payload["groups"][0]["yaw_delta_deg"] = 1.0
        elif mutation == "bounds":
            payload["groups"][0]["center_xy"] = [100.0, 100.0]
        else:
            payload["variant_id"] = "scene-variant:" + "0" * 64
        raw = json.dumps(payload)
    path.write_text(raw)

    with pytest.raises(ValueError):
        loading.load_layouts(directory)


def test_load_requires_completed_manifest(
    saved_pool: tuple[Path, Path, SceneRandomizationResult],
) -> None:
    _, directory, _ = saved_pool
    (directory.parent / "run_result.json").unlink()

    with pytest.raises((ValueError, FileNotFoundError)):
        loading.load_layouts(directory)


def test_load_rejects_a_published_pool_without_successful_layouts(
    saved_pool: tuple[Path, Path, SceneRandomizationResult],
) -> None:
    workspace, _, result = saved_pool
    attempts = int(result.resolved_parameters["max_attempts"])
    empty = replace(
        result,
        candidates=(),
        status="budget_exhausted",
        attempts=attempts,
        solver_status_counts={"no_solution": attempts},
    )
    run = persistence.save_layouts(empty, output_root=workspace)

    with pytest.raises(ValueError):
        loading.load_layouts(run / "layouts")


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_preview_rejects_invalid_interval_before_reading_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interval: float
) -> None:
    monkeypatch.setattr(
        loading,
        "load_layouts",
        lambda *args, **kwargs: pytest.fail(
            "Invalid interval must fail before file access"
        ),
    )

    with pytest.raises(ValueError, match="interval"):
        loading.preview_layouts(tmp_path / "unused", interval=interval)


class _Rigid:
    body_type = "static"

    def __init__(self) -> None:
        self.pose = np.zeros((4, 4))
        self.writes = 0
        self.clears = 0

    def clear_dynamics(self, *, env_ids: list[int]) -> None:
        assert env_ids == [0]
        self.clears += 1

    def set_local_pose(self, pose: Any, *, env_ids: list[int]) -> None:
        assert env_ids == [0]
        self.pose = pose.detach().cpu().numpy()[0].copy()
        self.writes += 1


class _Sim:
    device = "cpu"
    num_envs = 1

    def __init__(self, uids: set[str]) -> None:
        self.entities = {uid: _Rigid() for uid in uids}
        self.syncs = 0

    def get_rigid_object_uid_list(self) -> list[str]:
        return list(self.entities)

    def get_rigid_object(self, uid: str) -> _Rigid:
        return self.entities[uid]

    def get_robot_uid_list(self) -> list[str]:
        return []

    def get_articulation_uid_list(self) -> list[str]:
        return []

    def sync_render_state(self) -> None:
        self.syncs += 1


def test_preview_uses_one_scene_and_cycles_absolute_poses_on_deadlines(
    saved_pool: tuple[Path, Path, SceneRandomizationResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from embodichain.gen_sim.scene_engine.cli import preview

    workspace, directory, result = saved_pool
    baseline_poses = _source_poses(workspace)
    sim = _Sim(set(baseline_poses))
    candidate_paths = {path.resolve() for path in directory.glob("*.json")}
    read_counts: Counter[Path] = Counter()
    read_bytes = Path.read_bytes
    read_text = Path.read_text

    def count_read_bytes(path: Path) -> bytes:
        if path.resolve() in candidate_paths:
            read_counts[path.resolve()] += 1
        return read_bytes(path)

    def count_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.resolve() in candidate_paths:
            read_counts[path.resolve()] += 1
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", count_read_bytes)
    monkeypatch.setattr(Path, "read_text", count_read_text)
    now = [100.0]
    monkeypatch.setattr(loading.time, "monotonic", lambda: now[0])
    calls = []
    observed = []

    def preview_once(**kwargs: Any) -> None:
        calls.append(kwargs)
        callback = kwargs["on_frame"]
        assert kwargs["advance_physics"] is False
        for timestamp, changed in (
            (100.0, True),
            (100.5, False),
            (101.0, True),
            (101.5, False),
            (102.0, True),
            (103.0, True),
            (104.0, True),
            (105.0, True),
            (106.0, True),
        ):
            now[0] = timestamp
            writes_before = sum(entity.writes for entity in sim.entities.values())
            assert callback(sim) is changed
            writes_after = sum(entity.writes for entity in sim.entities.values())
            expected_writes = 0
            if changed:
                # Candidate poses cover every movable member; the fixed table
                # only needs an explicit write when restoring the original scene.
                expected_writes = len(sim.entities) - (timestamp not in (100, 103, 106))
            assert writes_after - writes_before == expected_writes
            if changed:
                observed.append(
                    {uid: entity.pose.copy() for uid, entity in sim.entities.items()}
                )

    monkeypatch.setattr(preview, "preview_scene_export", preview_once)
    loading.preview_layouts(directory, interval=1.0, device="cpu", headless=True)

    assert len(calls) == 1
    assert calls[0]["output_root"] == workspace
    assert calls[0]["device"] == "cpu"
    assert calls[0]["headless"] is True
    expected = [baseline_poses]
    for candidate in result.candidates:
        assert candidate.variant is not None
        expected.append(
            baseline_poses
            | {
                change.entity_uid: np.asarray(change.arena_pose)
                for change in candidate.variant.pose_changes
            }
        )
    expected = expected + expected + [baseline_poses]
    assert len(observed) == len(expected) == 7
    for actual, target in zip(observed, expected, strict=True):
        assert set(actual) == set(target)
        for uid, pose in target.items():
            np.testing.assert_allclose(actual[uid], pose, atol=1e-7)
    assert sim.syncs == 7
    assert sim.entities["table"].clears == sim.entities["table"].writes == 3
    assert all(
        entity.clears == entity.writes == 7
        for uid, entity in sim.entities.items()
        if uid != "table"
    )
    assert read_counts == Counter({path: 1 for path in candidate_paths})
