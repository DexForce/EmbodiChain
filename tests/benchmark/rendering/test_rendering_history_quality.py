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

"""Real file-proof validation and alias gates; no render or learned-model work."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.rendering import evaluate, quality
from scripts.benchmark.rendering.frozen_source import artifact, sha256
from scripts.benchmark.rendering.history_contract import validate_history_proof
from scripts.benchmark.rendering.publication_report import _timed_history_reasons
from scripts.benchmark.rendering import publication_report
from scripts.benchmark.rendering.quality_protocol import (
    protocol_for_modalities,
    protocol_requirements,
)


def _fixture(root: Path, *, numerical: bool = False) -> dict[str, Any]:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]

    def arrays(path: Path, index: int, metadata: dict[str, Any]) -> None:
        import numpy as np

        metadata["camera"]["convention"] = "opencv"
        metadata["view_sha256"] = stable_hash(metadata["camera"])
        metadata["depth_semantics"] = "optical_axis_metres"
        metadata["segmentation_semantics"] = "canonical_instance_ids_zero_background"
        np.savez(
            path,
            rgb=np.full((4, 4, 3), 128, dtype=np.uint8),
            depth=np.ones((4, 4), dtype=np.float32),
            seg=np.ones((4, 4), dtype=np.int32),
        )

    fixture = builder(root, write_arrays=arrays if numerical else None)
    proof = validate_history_proof(root, fixture["path"], fixture["expected_original"])
    assert proof["valid"], proof["reasons"]
    candidates = [
        {"metadata": deepcopy(frame["metadata"])} for frame in proof["frames"]
    ]
    references = []
    for candidate in candidates:
        meta = deepcopy(candidate["metadata"])
        raw_id = meta["frame_id"]
        ref_path = root / "reference" / f"frame_{raw_id}.json"
        array_path = ref_path.with_suffix(".npz")
        if not ref_path.exists():
            ref_path.parent.mkdir(exist_ok=True)
            if numerical:
                frame = next(
                    item
                    for item in proof["frames"]
                    if item["metadata"]["frame_id"] == raw_id
                )
                array_path.write_bytes(frame["arrays_path"].read_bytes())
            else:
                array_path.write_bytes(
                    b"file-binding fixture, not image quality evidence"
                )
            raw_meta = deepcopy(meta)
            raw_meta.pop("history")
            write_json(
                ref_path,
                {
                    "metadata": raw_meta,
                    "arrays": {"path": array_path.name, "sha256": sha256(array_path)},
                },
            )
        reference = {
            "metadata": meta,
            "reference_provenance": {
                "kind": "synthetic_file_fixture",
                "reference_id": f"fixture-{raw_id}",
            },
            "artifacts": [
                {
                    **artifact(root, ref_path, "reference_metadata"),
                    "path": str(ref_path),
                },
                {
                    **artifact(root, array_path, "reference_arrays"),
                    "path": str(array_path),
                },
            ],
        }
        if numerical:
            import numpy as np

            with np.load(array_path, allow_pickle=False) as data:
                reference.update({name: data[name].copy() for name in data.files})
        references.append(reference)
    evidence = {
        **artifact(root, fixture["path"], "timed_renderer_history"),
        "root": str(root),
        "expected_original": fixture["expected_original"],
    }
    return {
        **fixture,
        "proof": proof,
        "candidates": candidates,
        "references": references,
        "evidence": evidence,
    }


def test_actual_history_validator_binds_aliases_without_rewriting_raw_time(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    original = deepcopy(fixture["candidates"])
    checks, descriptor, timeline, artifacts = quality._timed_history_context(
        fixture["candidates"], fixture["references"], fixture["evidence"]
    )
    assert all(check["status"] == "passed" for check in checks.values())
    assert descriptor["measured_frames"] == 6 and descriptor["sequence_frames"] == 7
    assert len(timeline["aliases"]) == 7 and artifacts
    assert [item["replay_frame_id"] for item in timeline["aliases"]] == [
        1,
        0,
        1,
        2,
        0,
        1,
        2,
    ]
    assert fixture["candidates"] == original
    assert timeline["aliases"][0]["replay_time_s"] == 1 / 30
    assert timeline["aliases"][0]["history"]["scoring_time_s"] == 0
    assert all(
        not Path(item["candidate_metadata"]["path"]).is_absolute()
        for item in timeline["aliases"]
    )


@pytest.mark.parametrize(
    "changed", ["raw_time", "scoring_alias", "missing_guard", "reference_alias"]
)
def test_in_memory_alias_edits_cannot_override_verified_sources(
    tmp_path: Path, changed: str
) -> None:
    fixture = _fixture(tmp_path)
    if changed == "raw_time":
        fixture["candidates"][1]["metadata"]["time_s"] = 999
    elif changed == "scoring_alias":
        fixture["candidates"][1]["metadata"]["history"]["scoring_index"] = 999
    elif changed == "missing_guard":
        fixture["candidates"].pop(0)
    else:
        fixture["references"][1]["metadata"].pop("history")
    checks, _, _, _ = quality._timed_history_context(
        fixture["candidates"], fixture["references"], fixture["evidence"]
    )
    assert checks["history_aliases"]["status"] == "failed"


def _report_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    fixture = _fixture(root)
    checks, descriptor, timeline, _ = quality._timed_history_context(
        fixture["candidates"], fixture["references"], fixture["evidence"]
    )
    assert all(check["status"] == "passed" for check in checks.values())
    path = write_json(root / "exposure_timeline.json", timeline)
    timeline_artifact = artifact(root, path, "scoring_exposure_timeline")
    metadata = [frame["metadata"] for frame in fixture["candidates"]]
    pairs = []
    for index in range(1, len(metadata)):
        previous, current = metadata[index - 1], metadata[index]
        pairs.append(
            {
                "previous_frame": index - 1,
                "current_frame": index,
                "status": "passed",
                "previous_replay_frame_id": previous["frame_id"],
                "current_replay_frame_id": current["frame_id"],
                "transition": (
                    "warmup_entry"
                    if index == 1
                    else (
                        "replay_wrap"
                        if current["history"]["cycle"] != previous["history"]["cycle"]
                        else "continuous"
                    )
                ),
            }
        )
    sources = [*fixture["proof"]["artifacts"], descriptor, timeline_artifact]
    for alias in timeline["aliases"]:
        sources.extend(
            [
                alias["candidate_metadata"],
                alias["candidate_arrays"],
                *alias["reference_sources"].values(),
            ]
        )
    record = {
        "protocol_id": fixture["manifest"]["history_quality_protocol"]["id"],
        "evaluation_kind": "sequence",
        "timed_history": descriptor,
        "coverage": {"expected_samples": 7, "sequence_frames": 7},
        "artifacts": sources,
        "sequences": [
            {
                "env_id": 0,
                "camera_name": "external_front",
                "timed_history": descriptor,
                "exposure_timeline": timeline_artifact,
                "pairs": pairs,
            }
        ],
    }
    return fixture["expected_original"], record, path


def test_offline_report_revalidates_real_history_and_all_alias_sources(
    tmp_path: Path,
) -> None:
    original, record, _ = _report_fixture(tmp_path)
    assert _timed_history_reasons(tmp_path, original, record, {}) == []


@pytest.mark.parametrize(
    "changed",
    [
        "short_population",
        "missing_guard",
        "raw_time",
        "wrong_reference",
        "missing_wrap",
        "original_metrics",
    ],
)
def test_resealed_history_claims_do_not_bypass_offline_semantic_gates(
    tmp_path: Path, changed: str
) -> None:
    original, record, path = _report_fixture(tmp_path)
    timeline = json.loads(path.read_text())
    if changed == "short_population":
        record["timed_history"]["measured_frames"] = 3
    elif changed == "missing_guard":
        timeline["aliases"].pop(0)
    elif changed == "raw_time":
        timeline["aliases"][1]["replay_time_s"] = 1 / 30
    elif changed == "wrong_reference":
        timeline["aliases"][1]["reference_sources"] = deepcopy(
            timeline["aliases"][0]["reference_sources"]
        )
    elif changed == "missing_wrap":
        record["sequences"][0]["pairs"][3]["transition"] = "continuous"
    else:
        original["metrics"] = {"batch_steps_per_s": 999}
    write_json(path, timeline)
    digest = sha256(path)
    for item in record["artifacts"]:
        if item["path"] == path.name:
            item["sha256"] = digest
    record["sequences"][0]["exposure_timeline"]["sha256"] = digest
    assert "quality_timed_history" in _timed_history_reasons(
        tmp_path, original, record, {}
    )


def test_history_loader_rechecks_geometry_against_original_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(
        evaluate, "load_scene", lambda path: json.loads(Path(path).read_text())
    )
    monkeypatch.setattr(
        evaluate, "load_replay", lambda path, scene: json.loads(Path(path).read_text())
    )
    run = {
        "row": fixture["expected_original"],
        "case": SimpleNamespace(**fixture["config"]["case"]),
        "config": {**fixture["config"], "replay_path": "original/replay.json"},
        "scene_path": tmp_path / "original/scene.json",
        "input_artifacts": [],
        "coverage": {"representative_env_ids": [0]},
    }
    loaded = evaluate._load_history(tmp_path, run, fixture["path"])
    assert loaded["coverage"]["expected_samples"] == 7
    assert [item["metadata"]["frame_id"] for item in loaded["descriptors"]] == [
        1,
        0,
        1,
        2,
        0,
        1,
        2,
    ]
    manifest = json.loads(fixture["frames_manifest_path"].read_text())
    entry = manifest["frames"][1]
    frame_path = fixture["frames_manifest_path"].parent / entry["path"]
    frame = json.loads(frame_path.read_text())
    frame["metadata"]["camera"]["intrinsics"][0][0] = 20
    frame["metadata"]["view_sha256"] = stable_hash(frame["metadata"]["camera"])
    write_json(frame_path, frame)
    entry["sha256"] = sha256(frame_path)
    write_json(fixture["frames_manifest_path"], manifest)
    fixture["manifest"]["frames_manifest"]["sha256"] = sha256(
        fixture["frames_manifest_path"]
    )
    write_json(fixture["path"], fixture["manifest"])
    assert validate_history_proof(
        tmp_path, fixture["path"], fixture["expected_original"]
    )["valid"]
    with pytest.raises(ValueError, match="History camera differs"):
        evaluate._load_history(tmp_path, run, fixture["path"])


def test_full_evaluator_uses_every_history_exposure_and_preserves_original_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, numerical=True)
    source_policy = protocol_for_modalities(("rgb",), temporal_dt_s=1 / 30)
    source_contract = {
        **protocol_requirements(source_policy["id"], "sequence"),
        "temporal_quality": "assessed_by_declared_policy",
    }
    config = {
        **fixture["config"],
        "scene_path": "original/scene.json",
        "replay_path": "original/replay.json",
        "quality_protocol": source_policy,
        "quality_requirements": source_contract,
    }
    config_path = write_json(tmp_path / "original/worker_config.json", config)
    fixture["expected_original"]["_history_artifact_hashes"]["worker_config"] = sha256(
        config_path
    )
    for item in fixture["manifest"]["original"]["artifacts"]:
        if item["role"] == "worker_config":
            item["sha256"] = sha256(config_path)
    for item in fixture["manifest"]["producer_artifacts"]:
        if item["role"] == "request":
            path = tmp_path / item["path"]
            request = json.loads(path.read_text())
            request["original"] = deepcopy(fixture["manifest"]["original"])
            write_json(path, request)
            item["sha256"] = sha256(path)
    write_json(fixture["path"], fixture["manifest"])
    monkeypatch.setattr(
        evaluate, "load_scene", lambda path: json.loads(Path(path).read_text())
    )
    monkeypatch.setattr(
        evaluate, "load_replay", lambda path, scene: json.loads(Path(path).read_text())
    )

    def no_model(*args: Any) -> Any:
        raise FileNotFoundError(
            "No learned model used in file-contract orchestration fixture"
        )

    monkeypatch.setattr(quality, "_lpips", no_model)
    seen = []

    def independent(
        root: Path, run: dict[str, Any], item: dict[str, Any], settings: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        index = item["metadata"]["history"]["scoring_index"]
        seen.append(item["metadata"]["frame_id"])
        return deepcopy(fixture["references"][index]), {
            "history": deepcopy(item["metadata"]["history"])
        }

    monkeypatch.setattr(evaluate, "_independent_reference", independent)
    run = {
        "row": fixture["expected_original"],
        "case": SimpleNamespace(**config["case"]),
        "config": config,
        "scene_path": tmp_path / "original/scene.json",
        "input_artifacts": [],
        "coverage": {"representative_env_ids": [0]},
    }
    loaded = evaluate._load_history(tmp_path, run, fixture["path"])
    before = {
        path: sha256(path)
        for path in (tmp_path / "original").rglob("*")
        if path.is_file()
    }
    record = evaluate._evaluate_run(
        tmp_path,
        tmp_path / "evaluation",
        loaded,
        {"spp": 1, "threads": 1, "seed": 0},
        None,
    )
    assert seen == [1, 0, 1, 2, 0, 1, 2]
    assert record["coverage"]["sequence_frames"] == 7
    assert record["coverage"]["evaluated_samples"] == 7
    assert record["checks"]["timed_history"]["status"] == "passed"
    assert record["checks"]["temporal_evidence"]["status"] == "passed"
    assert record["original_quality_contract"] == source_contract
    assert (
        record["quality_status"] == "not_qualified"
    )  # File fixtures cannot certify image quality.
    assert (
        _timed_history_reasons(tmp_path, fixture["expected_original"], record, {}) == []
    )
    definition = {
        "quality_protocol": {
            "case_protocols": {run["row"]["case_id"]: source_contract},
            "policies": {source_policy["id"]: source_policy},
        }
    }
    write_json(tmp_path / "quality.json", {"runs": [record]})
    row = deepcopy(fixture["expected_original"])
    publication_report._attach_quality(tmp_path, [row], definition)
    assert not any(
        reason.startswith("frozen_quality_assignment")
        for reason in row["quality_reasons"]
    )
    assert row["qualified_for_comparison"] is False
    record["original_quality_contract"]["required_metrics"] = []
    write_json(tmp_path / "quality.json", {"runs": [record]})
    row = deepcopy(fixture["expected_original"])
    publication_report._attach_quality(tmp_path, [row], definition)
    assert "frozen_quality_assignment:protocol_id" in row["quality_reasons"]
    assert {path: sha256(path) for path in before} == before
