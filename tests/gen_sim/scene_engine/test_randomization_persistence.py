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

"""Verify durable layout pools, provenance, and atomic publication."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
from threading import Barrier
from typing import Any

import pytest
import yaml

from embodichain.gen_sim.scene_engine.randomization import generation, persistence
from embodichain.gen_sim.scene_engine.randomization.generation import (
    SceneRandomizationResult,
)

from .test_randomization_baseline import _export


@pytest.fixture
def completed_pool(tmp_path: Path) -> tuple[Path, SceneRandomizationResult]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _, cfg = _export(workspace)
    result = generation.randomize_scene(
        cfg,
        seed=23,
        candidate_count=2,
        max_attempts=12,
        collision_margin_m=0.037,
    )
    assert result.status == "completed"
    return workspace, result


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(root: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_references(path: Path, payload: dict[str, Any], workspace: Path) -> None:
    source = payload["source_scene"]
    baseline = payload["base_data"]
    for value in (source["export_dir"], baseline["path"], payload["policy_file"]):
        assert not Path(value).is_absolute()
    assert (path.parent / source["export_dir"]).resolve() == (
        workspace / "scene_export"
    ).resolve()
    baseline_path = (path.parent / baseline["path"]).resolve()
    assert (
        baseline_path
        == (workspace / "scene_export" / "scene_randomization_base.yaml").resolve()
    )
    baseline_data = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
    assert baseline_data["data_sha256"] == baseline["data_sha256"]
    assert baseline_data["source_scene"]["content_sha256"] == source["content_sha256"]
    policy_path = (path.parent / payload["policy_file"]).resolve()
    assert policy_path.name == "policy_resolved.yaml"
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    canonical_policy = json.dumps(
        policy, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert hashlib.sha256(canonical_policy).hexdigest() == payload["policy_sha256"]


def test_save_persists_only_group_layouts_and_actual_policy(
    completed_pool: tuple[Path, SceneRandomizationResult],
) -> None:
    workspace, result = completed_pool
    before = _snapshot(workspace)

    run = persistence.save_layouts(result, output_root=workspace)

    assert run == workspace / "scene_layouts" / "run_001"
    manifest_path = run / "run_result.json"
    manifest = _json(manifest_path)
    policy = yaml.safe_load((run / "policy_resolved.yaml").read_text(encoding="utf-8"))
    assert policy == {
        "schema_version": 1,
        "seed": 23,
        "candidate_count": 2,
        "max_attempts": 12,
        "collision_margin_m": 0.037,
    }
    assert manifest["status"] == "completed"
    assert manifest["requested_count"] == 2
    assert manifest["generated_count"] == 2
    assert manifest["max_attempts"] == 12
    assert manifest["attempts"] == result.attempts
    assert manifest["seed"] == 23
    assert manifest["algorithm_version"] == result.algorithm_version
    assert manifest["solver_status_counts"] == result.solver_status_counts
    _assert_references(manifest_path, manifest, workspace)
    assert len(manifest["layouts"]) == len(result.candidates)
    for entry, candidate in zip(manifest["layouts"], result.candidates, strict=True):
        assert candidate.variant is not None
        ordinal = int(dict(candidate.variant.metadata)["proposal_ordinal"])
        candidate_path = run / entry["path"]
        assert candidate_path == run / "layouts" / f"candidate_{ordinal:06d}.json"
        payload = _json(candidate_path)
        assert payload["candidate_id"] == entry["candidate_id"] == candidate_path.stem
        assert payload["proposal_ordinal"] == entry["proposal_ordinal"] == ordinal
        assert (
            payload["variant_id"] == entry["variant_id"] == candidate.variant.variant_id
        )
        assert payload["variant_ordinal"] == candidate.variant.ordinal
        assert payload["variant_metadata"] == dict(candidate.variant.metadata)
        assert payload["algorithm_version"] == result.algorithm_version
        assert payload["seed"] == 23
        assert payload["coordinate_frame"] == "local_arena_z_up"
        assert payload["geometry_model"] == "fixed_orientation_aabb"
        groups = {group["root_id"]: group for group in payload["groups"]}
        assert set(groups) == set(candidate.layout.xy_by_uid)
        for uid, center in candidate.layout.xy_by_uid.items():
            assert groups[uid] == {
                "root_id": uid,
                "center_xy": pytest.approx(center),
                "yaw_delta_deg": 0.0,
            }
        assert (
            not {"fragment", "entities", "members", "pose_changes", "rigid_object"}
            & payload.keys()
        )
        _assert_references(candidate_path, payload, workspace)
    assert len([path for path in run.rglob("*") if path.is_file()]) == 4
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in before
    } == before


def test_pool_references_survive_relocating_the_workspace(
    completed_pool: tuple[Path, SceneRandomizationResult], tmp_path: Path
) -> None:
    workspace, result = completed_pool
    run = persistence.save_layouts(result, output_root=workspace)
    relocated = tmp_path / "relocated"
    shutil.copytree(workspace, relocated)
    relocated_run = relocated / run.relative_to(workspace)

    for original in [
        run / "run_result.json",
        *sorted((run / "layouts").glob("*.json")),
    ]:
        moved = relocated_run / original.relative_to(run)
        assert moved.read_bytes() == original.read_bytes()
        _assert_references(moved, _json(moved), relocated)


def test_saved_names_retain_original_proposal_gaps(
    completed_pool: tuple[Path, SceneRandomizationResult],
) -> None:
    workspace, result = completed_pool
    candidates = []
    # Accepted candidates are dense, but their original proposal ordinals need not be.
    for candidate, proposal_ordinal in zip(result.candidates, (1, 3), strict=True):
        assert candidate.variant is not None
        metadata = dict(candidate.variant.metadata)
        metadata["proposal_ordinal"] = str(proposal_ordinal)
        candidates.append(
            replace(
                candidate,
                variant=replace(candidate.variant, metadata=tuple(metadata.items())),
            )
        )
    result = replace(
        result,
        candidates=tuple(candidates),
        attempts=4,
        solver_status_counts={"solved": 2, "no_solution": 2},
    )

    run = persistence.save_layouts(result, output_root=workspace)

    manifest = _json(run / "run_result.json")
    assert [entry["proposal_ordinal"] for entry in manifest["layouts"]] == [1, 3]
    assert {path.name for path in (run / "layouts").iterdir()} == {
        "candidate_000001.json",
        "candidate_000003.json",
    }
    assert manifest["attempts"] == 4
    assert manifest["solver_status_counts"] == {"solved": 2, "no_solution": 2}


@pytest.mark.parametrize(
    ("status", "generated_count", "attempts", "counts"),
    [
        ("budget_exhausted", 1, 12, {"solved": 1, "no_solution": 11}),
        ("budget_exhausted", 0, 12, {"no_solution": 12}),
        ("unsupported", 0, 1, {"unsupported": 1}),
    ],
)
def test_incomplete_runs_still_publish_policy_and_manifest(
    completed_pool: tuple[Path, SceneRandomizationResult],
    status: str,
    generated_count: int,
    attempts: int,
    counts: dict[str, int],
) -> None:
    workspace, result = completed_pool
    result = replace(
        result,
        candidates=result.candidates[:generated_count],
        status=status,
        attempts=attempts,
        solver_status_counts=counts,
    )

    run = persistence.save_layouts(result, output_root=workspace)

    manifest = _json(run / "run_result.json")
    assert (run / "policy_resolved.yaml").is_file()
    assert manifest["status"] == status
    assert manifest["requested_count"] == 2
    assert manifest["generated_count"] == generated_count
    assert manifest["attempts"] == attempts
    assert manifest["solver_status_counts"] == counts
    assert len(manifest["layouts"]) == generated_count
    assert (
        len([path for path in run.rglob("*") if path.is_file()]) == generated_count + 2
    )


def test_each_save_reserves_a_fresh_run_even_when_calls_overlap(
    completed_pool: tuple[Path, SceneRandomizationResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, result = completed_pool
    original = persistence.save_layouts(result, output_root=workspace)
    before = _snapshot(original)
    contenders = Barrier(3)
    real_mkdir = Path.mkdir

    def competing_mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == workspace / "scene_layouts" / "run_002":
            contenders.wait(timeout=10)
        real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", competing_mkdir)

    with ThreadPoolExecutor(max_workers=3) as executor:
        runs = list(
            executor.map(
                lambda _: persistence.save_layouts(result, output_root=workspace),
                range(3),
            )
        )

    assert {run.name for run in runs} == {"run_002", "run_003", "run_004"}
    assert _snapshot(original) == before
    for run in runs:
        manifest = _json(run / "run_result.json")
        assert manifest["generated_count"] == 2
        assert len(manifest["layouts"]) == 2


def test_write_failure_never_publishes_a_manifest_or_partial_json(
    completed_pool: tuple[Path, SceneRandomizationResult],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, result = completed_pool
    original = persistence.save_layouts(result, output_root=workspace)
    before = _snapshot(original)
    real_replace = persistence.os.replace
    candidate_publications = 0

    def fail_second_candidate(source: str | Path, destination: str | Path) -> None:
        nonlocal candidate_publications
        destination = Path(destination)
        if destination.name.startswith("candidate_") and destination.suffix == ".json":
            candidate_publications += 1
            if candidate_publications == 2:
                raise OSError("Injected publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", fail_second_candidate)

    with pytest.raises(OSError, match="Injected publication failure"):
        persistence.save_layouts(result, output_root=workspace)

    assert candidate_publications == 2
    assert _snapshot(original) == before
    failed_run = workspace / "scene_layouts" / "run_002"
    assert not (failed_run / "run_result.json").exists()
    candidate_files = sorted(failed_run.glob("layouts/*.json"))
    assert len(candidate_files) <= 1
    for path in candidate_files:
        payload = _json(path)
        assert payload["groups"]
    assert not list(failed_run.rglob("*.tmp"))


@pytest.mark.parametrize("changed_file", ["baseline", "scene_config", "mesh"])
def test_stale_source_or_baseline_is_rejected_before_manifest_publication(
    completed_pool: tuple[Path, SceneRandomizationResult], changed_file: str
) -> None:
    workspace, result = completed_pool
    if changed_file == "baseline":
        path = result.baseline.path
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["groups"][0]["aabb_2d"]["center_xy"][0] += 0.1
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    elif changed_file == "scene_config":
        path = workspace / "scene_export" / "scene_config.json"
        path.write_bytes(path.read_bytes() + b"\n")
    else:
        path = next((workspace / "scene_export" / "mesh_assets").rglob("*.glb"))
        path.write_bytes(path.read_bytes() + b"\n")
    changed = path.read_bytes()

    with pytest.raises(ValueError, match="(?i)(changed|stale|baseline|source)"):
        persistence.save_layouts(result, output_root=workspace)

    assert path.read_bytes() == changed
    assert not list((workspace / "scene_layouts").glob("*/run_result.json"))


def test_pose_inconsistent_with_saved_centers_is_rejected_before_output(
    completed_pool: tuple[Path, SceneRandomizationResult],
) -> None:
    workspace, result = completed_pool
    candidate = result.candidates[0]
    assert candidate.variant is not None
    change = candidate.variant.pose_changes[0]
    pose = [list(row) for row in change.arena_pose]
    pose[0][3] += 0.1
    variant = replace(
        candidate.variant,
        pose_changes=(replace(change, arena_pose=pose),)
        + candidate.variant.pose_changes[1:],
    )
    inconsistent = replace(
        result,
        candidates=(replace(candidate, variant=variant),) + result.candidates[1:],
    )

    with pytest.raises(ValueError, match="poses do not match"):
        persistence.save_layouts(inconsistent, output_root=workspace)

    assert not (workspace / "scene_layouts").exists()


def test_inconsistent_accounting_is_rejected_before_output(
    completed_pool: tuple[Path, SceneRandomizationResult],
) -> None:
    workspace, result = completed_pool
    counts = dict(result.solver_status_counts)
    counts["solved"] += 1
    inconsistent = replace(result, solver_status_counts=counts)

    with pytest.raises(ValueError, match="accounting is inconsistent"):
        persistence.save_layouts(inconsistent, output_root=workspace)

    assert not (workspace / "scene_layouts").exists()
