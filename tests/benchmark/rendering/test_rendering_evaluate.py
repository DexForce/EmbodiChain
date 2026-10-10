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

"""Offline evidence orchestration with explicit synthetic CPU renderer fixtures."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
from typing import Any

import numpy as np
import pytest

from scripts.benchmark.core.artifacts import read_json_object, stable_hash, write_json
from scripts.benchmark.rendering import evaluate, quality, reference, publication_report
from scripts.benchmark.rendering.formal import RenderCase
from scripts.benchmark.rendering.quality_protocol import protocol_requirements
from scripts.benchmark.rendering.scenes import load_scene, scene_fingerprint


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_batch_registration_preserves_kind_and_rejects_unbound_evidence(
    tmp_path: Path,
) -> None:
    from scripts.benchmark.core.artifacts import ArtifactStore

    path = tmp_path / "source.json"
    path.write_text("immutable source")
    store = ArtifactStore(tmp_path)
    original = store.register_artifact(path, kind="original_source")
    descriptor = {"path": path.name, "sha256": sha(path), "kind": "quality_detail"}
    evaluate._register_batch(store, [descriptor])
    assert read_json_object(tmp_path / "artifact_index.json")["artifacts"] == [original]
    before = (tmp_path / "artifact_index.json").read_bytes()
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate._register_batch(store, [{"path": path.name, "kind": "unbound"}])
    path.write_text("changed")
    with pytest.raises(ValueError):
        evaluate._register_batch(store, [descriptor])
    assert (tmp_path / "artifact_index.json").read_bytes() == before


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "reference_calls": [],
        "frame_calls": [],
        "sequence_calls": [],
        "report_calls": [],
    }

    def render(
        scene_path: Path,
        specs: list[dict[str, Any]],
        width: int,
        height: int,
        output: Path,
        **settings: Any,
    ) -> dict[str, Any]:
        assert len(specs) == 1 and "rgb" not in specs[0]
        state["reference_calls"].append((scene_path, deepcopy(specs), settings))
        output.mkdir(parents=True, exist_ok=False)
        scene = load_scene(scene_path)
        provenance = {
            "kind": "independent_offline",
            "renderer": "mitsuba",
            "device": "cpu",
            "fixture": "synthetic CPU orchestration test; no rendering claims",
            "source_scene_sha256": scene_fingerprint(scene),
            "assets_verified": True,
            "implementation_sha256": "fixture",
            "spec_sha256": stable_hash(specs),
            **settings,
        }
        provenance["reference_id"] = stable_hash(provenance)
        arrays = output / "frame_0000.npz"
        np.savez_compressed(
            arrays,
            rgb=np.full((height, width, 3), 0.5, dtype=np.float32),
            depth=np.ones((height, width), dtype=np.float32),
            seg=np.ones((height, width), dtype=np.int32),
        )
        frame = {
            "schema_version": 1,
            "metadata": {
                "backend": "mitsuba",
                "scene_sha256": scene_fingerprint(scene),
                "camera": reference.camera_metadata(scene, specs[0], width, height),
                "frame_id": specs[0]["frame_id"],
                "frame_index": specs[0]["frame_id"],
                "time_s": specs[0]["time_s"],
            },
            "reference_provenance": provenance,
            "reference_validation": {"metric": "synthetic_test_only"},
            "arrays": {"path": arrays.name, "sha256": sha(arrays)},
        }
        path = write_json(output / "frame_0000.json", frame)
        manifest = {
            "schema_version": 1,
            "provenance": provenance,
            "frames": [{"path": path.name, "sha256": sha(path)}],
        }
        write_json(output / "reference.json", manifest)
        return manifest

    def frame(
        candidate: dict[str, Any],
        ref: dict[str, Any] | None,
        protocol: dict[str, Any],
        *,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        contract = protocol_requirements(protocol["id"], "frame")
        state["frame_calls"].append((deepcopy(candidate["metadata"]), contract))
        if ref is not None:
            assert ref["metadata"]["backend"] == "mitsuba"
            assert np.all(ref["rgb"] == 0.5)  # Candidate pixels never supply GT.
        calibrated = isinstance(
            candidate["metadata"].get("photometric_calibration"), dict
        )
        checks = {name: {"status": "passed"} for name in contract["required_checks"]}
        if not calibrated:
            checks["photometric_calibration"] = {
                "status": "missing",
                "reason": "No measured calibration",
            }
        requested_present = set(contract["requested_modalities"]).issubset(candidate)
        if not requested_present:
            checks["requested_outputs"] = {
                "status": "failed",
                "reason": "Missing requested modality",
            }
        loss = (
            0.2 if candidate["metadata"]["env_id"] == state.get("outlier_env") else 0.01
        )
        metrics = {}
        for name, (comparison, threshold, unit) in protocol["frame_thresholds"].items():
            required = name in contract["required_metrics"]
            value = (
                1 - loss
                if name == "ssim"
                else loss if name == "lpips" else 1.0 if comparison == "ge" else 0.0
            )
            metrics[name] = {
                "value": value if required and requested_present else None,
                "status": (
                    (
                        "passed"
                        if (
                            value >= threshold
                            if comparison == "ge"
                            else value <= threshold
                        )
                        else "failed"
                    )
                    if required and requested_present
                    else "missing" if required else "not_requested"
                ),
                "required": required,
                "reason": None if required else "modality_not_requested",
                "comparison": comparison,
                "threshold": threshold,
                "unit": unit,
                "denominator": 48 if required else 0,
            }
        result = {
            **contract,
            "checks": checks,
            "metrics": metrics,
            "quality_status": (
                "qualified"
                if calibrated and requested_present and loss < 0.05
                else "not_qualified"
            ),
            "qualification": {
                "passed": calibrated and loss < 0.05,
                "failed_checks": [],
                "missing_checks": [] if calibrated else ["photometric_calibration"],
            },
            "artifacts": list((ref or {}).get("artifacts", [])),
        }
        if output_dir is not None:
            write_json(output_dir / "quality_evidence.json", result)
        return result

    def sequence(
        candidates: list[dict[str, Any]],
        refs: list[dict[str, Any]] | None,
        protocol: dict[str, Any],
        *,
        output_dir: Path | None = None,
    ) -> dict[str, Any]:
        state["sequence_calls"].append(
            (
                len(candidates),
                protocol["id"],
                [item["metadata"]["frame_id"] for item in candidates],
            )
        )
        assert protocol["sequence_dt_s"] == 1 / 30
        contract = protocol_requirements(protocol["id"], "sequence")
        frames = [
            frame(item, refs[index] if refs else None, protocol)
            for index, item in enumerate(candidates)
        ]
        if refs:
            assert [item["metadata"]["frame_id"] for item in refs] == [
                item["metadata"]["frame_id"] for item in candidates
            ]
        checks = {name: {"status": "passed"} for name in contract["required_checks"]}
        checks["all_frame_quality"] = {
            "status": (
                "failed"
                if any(item["quality_status"] != "qualified" for item in frames)
                else "passed"
            )
        }
        checks["sequence_length"] = {
            "status": "passed" if len(candidates) == 100 else "failed"
        }
        metrics = {
            name: {
                "value": 1.0 if rule[0] == "ge" else 0.0,
                "status": "passed",
                "required": True,
                "comparison": rule[0],
                "threshold": rule[1],
                "unit": rule[2],
                "denominator": max(0, len(candidates) - 1),
            }
            for name, rule in protocol["temporal_thresholds"].items()
        }
        result = {
            **contract,
            "checks": checks,
            "metrics": metrics,
            "frames": frames,
            "artifacts": list(
                {
                    item["path"]: item
                    for report in frames
                    for item in report["artifacts"]
                }.values()
            ),
            "quality_status": "not_qualified",
        }
        if output_dir is not None:
            write_json(output_dir / "quality_evidence.json", result)
        return result

    def report(root: Path) -> Path:
        state["report_calls"].append(root)
        path = root / "report.md"
        path.write_text("Synthetic test rebuild; no performance claims.\n")
        return path

    monkeypatch.setattr(reference, "render_reference", render)
    monkeypatch.setattr(quality, "evaluate_frame", frame)
    monkeypatch.setattr(quality, "evaluate_sequence", sequence)
    monkeypatch.setattr(publication_report, "rebuild_report", report)
    return state


def experiment(
    root: Path,
    case: RenderCase,
    monkeypatch: pytest.MonkeyPatch,
    *,
    backends: tuple[str, ...] = ("embodichain",),
    sequence_quality: bool = False,
) -> list[dict[str, Any]]:
    # Reuse the simulator-free worker integration fixture to produce the real
    # worker schema. It uses only fake device buffers and a fake NVML sampler.
    fixture = runpy.run_path(str(Path(__file__).with_name("test_formal_worker.py")))
    fixture["runtime"].__wrapped__(monkeypatch)
    config_path = fixture["config_file"](root / "inputs", case, measured=3, warmup=0)
    if sequence_quality:
        from scripts.benchmark.rendering.formal_runner import _case_policy

        config = read_json_object(config_path)
        config["post_measurement_quality_capture"] = "frozen_sequence_100"
        config["quality_protocol"], config["quality_requirements"] = _case_policy(
            case.to_dict()
        )
        write_json(config_path, config)
    rows = []
    for backend in backends:
        run_id = "r00_" + backend
        result = fixture["worker"].run_worker(config_path, root / run_id, backend, 0)
        assert result["status"] == "completed", result.get("error")
        rows.append({**result, "run_id": run_id, "run_dir": run_id, "attempt": 0})
    write_json(root / "runs.json", rows)
    write_json(
        root / "definition.json",
        {
            "schema_version": 1,
            "definition_version": "2.0",
            "experiment_id": case.experiment_id,
        },
    )
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "runs": [
                {key: row[key] for key in ("run_id", "backend", "case_id", "repeat")}
                for row in rows
            ],
        },
    )
    write_json(root / "assets_manifest.json", {"assets": []})
    write_json(root / "artifact_index.json", {"artifacts": []})
    write_json(
        root / "quality.json",
        {
            "schema_version": 1,
            "runs": [
                {"run_id": row["run_id"], "quality_status": "not_qualified"}
                for row in rows
            ],
        },
    )
    (root / "raw.jsonl").write_text(
        "".join(json.dumps({"record_id": row["run_id"], **row}) + "\n" for row in rows)
    )
    return rows


def test_deduplication_preserves_raw_timings_and_calibration_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    case = RenderCase("R-04", "franka_workstation", width=8, height=6, num_envs=3)
    rows = experiment(tmp_path, case, monkeypatch, backends=("embodichain", "isaaclab"))
    raw = (tmp_path / "raw.jsonl").read_bytes()
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1, seed=4)
    assert len(offline["reference_calls"]) == 1
    assert len(offline["frame_calls"]) == 6
    assert all(
        metadata["effective_lighting_controls"] is None
        for metadata, _ in offline["frame_calls"]
    )
    assert all(
        metadata["postprocess_evidence"] is None
        for metadata, _ in offline["frame_calls"]
    )
    assert (tmp_path / "raw.jsonl").read_bytes() == raw
    updated = json.loads((tmp_path / "runs.json").read_text())
    assert [row["metrics"] for row in updated] == [row["metrics"] for row in rows]
    quality_rows = read_json_object(tmp_path / "quality.json")["runs"]
    assert all(row["quality_status"] == "not_qualified" for row in quality_rows)
    assert all(
        row["checks"]["photometric_calibration"]["status"] == "missing"
        for row in quality_rows
    )
    assert all(
        row["metrics"]["depth_rmse_m"]["status"] == "not_requested"
        for row in quality_rows
    )
    assert quality_rows[0]["reference_id"] == quality_rows[1]["reference_id"]
    assert all(
        not Path(item["path"]).is_absolute()
        for row in quality_rows
        for item in row["artifacts"]
    )
    assert (tmp_path / "quality_evaluations/original_runs.json").is_file()
    assert (
        tmp_path / "quality_evaluations/evaluation_0001/report_inputs.json"
    ).is_file()
    assert offline["report_calls"] == [tmp_path]


def test_staged_evaluations_reuse_immutable_reference_and_keep_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    rows = experiment(
        tmp_path,
        RenderCase("R-03", "calibration", width=8, height=6),
        monkeypatch,
        backends=("embodichain", "isaaclab"),
    )
    evaluate.evaluate_experiment(
        tmp_path, spp=2, threads=1, run_ids=[rows[0]["run_id"]]
    )
    references = {
        path: path.read_bytes()
        for path in (tmp_path / "references").rglob("*")
        if path.is_file()
    }
    evaluate.evaluate_experiment(
        tmp_path, spp=2, threads=1, run_ids=[rows[1]["run_id"]]
    )
    assert len(offline["reference_calls"]) == 1
    assert all(path.read_bytes() == value for path, value in references.items())
    assert len((tmp_path / "quality_events.jsonl").read_text().splitlines()) == 2


def test_candidate_uses_observed_renderer_evidence_without_changing_capture_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    rows = experiment(
        tmp_path, RenderCase("R-03", "calibration", width=8, height=6), monkeypatch
    )
    row = rows[0]
    observed = {
        "directional_intensity": 2.5,
        "environment_intensity": 0.2,
        "environment_color": [1.0, 1.0, 1.0],
    }
    postprocess = {
        "scope": "runtime_carb_readback",
        "settings": {"tone_mapping": False},
    }
    row["renderer"]["effective_lighting_controls"] = observed
    row["renderer"]["postprocess_evidence"] = postprocess
    result_path = tmp_path / row["run_id"] / "result.json"
    result = read_json_object(result_path)
    result["renderer"] = deepcopy(row["renderer"])
    write_json(result_path, result)
    write_json(tmp_path / "runs.json", rows)
    (tmp_path / "raw.jsonl").write_text(
        json.dumps({"record_id": row["run_id"], **row}) + "\n"
    )
    frames = {
        path: path.read_bytes()
        for path in (tmp_path / row["run_id"] / "frames").glob("*.json")
    }
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    metadata = offline["frame_calls"][0][0]
    assert metadata["effective_lighting_controls"] == observed
    assert metadata["postprocess_evidence"] == postprocess
    assert all(path.read_bytes() == original for path, original in frames.items())


def test_worst_sample_aggregation_never_averages_a_failed_camera_away(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    offline["outlier_env"] = 2
    experiment(
        tmp_path,
        RenderCase("R-04", "franka_workstation", width=8, height=6, num_envs=3),
        monkeypatch,
    )
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    record = read_json_object(tmp_path / "quality.json")["runs"][0]
    assert record["metrics"]["ssim"]["value"] == 0.8
    assert record["metrics"]["lpips"]["value"] == 0.2
    assert record["metrics"]["ssim"]["sample_denominator"] == 3
    assert record["metrics"]["ssim"]["status"] == "failed"


def test_moving_singleton_retains_missing_temporal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    experiment(
        tmp_path,
        RenderCase("R-03", "calibration", width=8, height=6, temporal_mode="moving"),
        monkeypatch,
    )
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    record = read_json_object(tmp_path / "quality.json")["runs"][0]
    assert record["checks"]["temporal_evidence"]["status"] == "missing"
    assert not offline["sequence_calls"]
    assert record["quality_status"] == "not_qualified"


def test_full_modality_moving_singleton_keeps_missing_hundred_frame_population(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    experiment(
        tmp_path,
        RenderCase(
            "R-03",
            "calibration",
            width=8,
            height=6,
            temporal_mode="moving",
            modalities=("rgb", "depth", "seg"),
        ),
        monkeypatch,
    )
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    record = read_json_object(tmp_path / "quality.json")["runs"][0]
    assert record["protocol_id"] == "pure-render-quality/temporal30-v1"
    assert record["coverage"]["expected_samples"] == 100
    assert record["coverage"]["evaluated_samples"] == 1
    assert record["checks"]["temporal_evidence"]["status"] == "missing"
    assert offline["sequence_calls"][0][0] == 1


@pytest.mark.parametrize("modalities", [("rgb",), ("rgb", "depth", "seg")])
def test_post100_cannot_certify_moving_performance_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: dict[str, Any],
    modalities: tuple[str, ...],
) -> None:
    rows = experiment(
        tmp_path,
        RenderCase(
            "R-03",
            "calibration",
            width=8,
            height=6,
            temporal_mode="moving",
            modalities=modalities,
        ),
        monkeypatch,
        sequence_quality=True,
    )
    raw = (tmp_path / "raw.jsonl").read_bytes()
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    record = read_json_object(tmp_path / "quality.json")["runs"][0]
    assert record["evaluation_kind"] == "sequence"
    assert record["coverage"]["complete"] is True
    assert record["coverage"]["expected_samples"] == 100
    assert record["checks"]["temporal_evidence"]["status"] == "missing"
    assert "timed renderer history" in record["checks"]["temporal_evidence"]["reason"]
    assert record["quality_status"] == "not_qualified"
    assert offline["sequence_calls"][0][0] == 100
    assert (tmp_path / "raw.jsonl").read_bytes() == raw
    assert (
        json.loads((tmp_path / "runs.json").read_text())[0]["metrics"]
        == rows[0]["metrics"]
    )


def test_missing_required_metric_is_retained_in_aggregate_denominator() -> None:
    from scripts.benchmark.rendering.quality_protocol import protocol_for_modalities

    policy = protocol_for_modalities(("rgb",))
    required = protocol_requirements(policy["id"])
    reports = [
        {
            "checks": {},
            "metrics": {"ssim": {"value": 0.99, "status": "passed", "denominator": 48}},
        },
        {
            "checks": {},
            "metrics": {
                "ssim": {
                    "value": None,
                    "status": "missing",
                    "reason": "missing requested RGB",
                    "denominator": 0,
                }
            },
        },
    ]
    _, metrics = evaluate._aggregate(reports, required, policy, 2)
    assert metrics["ssim"]["status"] == "missing"
    assert metrics["ssim"]["missing_samples"] == 1
    assert metrics["ssim"]["sample_denominator"] == 2


def test_r01_requires_all_twenty_views_and_representative_environments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    experiment(
        tmp_path,
        RenderCase(
            "R-01",
            "calibration",
            width=8,
            height=6,
            num_envs=3,
            view_count=20,
            modalities=("rgb", "depth", "seg"),
            delivery="host_ready",
        ),
        monkeypatch,
    )
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    record = read_json_object(tmp_path / "quality.json")["runs"][0]
    assert (
        record["coverage"]["expected_samples"]
        == record["coverage"]["evaluated_samples"]
        == 60
    )
    assert len(offline["frame_calls"]) == 60
    assert len(offline["reference_calls"]) == 20


def test_r02_groups_hundred_frame_camera_streams_at_thirty_hz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    experiment(
        tmp_path,
        RenderCase(
            "R-02",
            "franka_workstation",
            width=8,
            height=6,
            cameras_per_env=3,
            temporal_mode="moving",
            modalities=("rgb", "depth", "seg"),
            delivery="host_ready",
        ),
        monkeypatch,
    )
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    record = read_json_object(tmp_path / "quality.json")["runs"][0]
    assert record["protocol_id"] == "pure-render-quality/temporal30-v1"
    assert record["coverage"]["sequence_groups"] == len(record["sequences"]) == 3
    assert all(
        count == 100 and ids == list(range(100))
        for count, _, ids in offline["sequence_calls"]
    )


@pytest.mark.parametrize("tamper", ["candidate_npz", "candidate_json", "timing"])
def test_modified_capture_or_snapshot_is_rejected_before_reference_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    offline: dict[str, Any],
    tamper: str,
) -> None:
    rows = experiment(
        tmp_path, RenderCase("R-03", "calibration", width=8, height=6), monkeypatch
    )
    run = tmp_path / rows[0]["run_id"]
    entry = read_json_object(run / "frames.json")["frames"][0]
    if tamper == "timing":
        rows[0]["metrics"]["camera_frames_per_s"] = 999999
        write_json(tmp_path / "runs.json", rows)
    else:
        path = run / entry["arrays_path" if tamper == "candidate_npz" else "path"]
        path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum|immutable raw"):
        evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    assert not offline["reference_calls"]


def test_altered_cached_reference_or_aggregate_evidence_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    experiment(
        tmp_path, RenderCase("R-03", "calibration", width=8, height=6), monkeypatch
    )
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    path = next((tmp_path / "references").rglob("*.npz"))
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    assert len(offline["reference_calls"]) == 1


def test_manual_calibration_status_and_arbitrary_hashed_file_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: dict[str, Any]
) -> None:
    rows = experiment(
        tmp_path, RenderCase("R-03", "calibration", width=8, height=6), monkeypatch
    )
    evaluate.evaluate_experiment(tmp_path, spp=2, threads=1)
    record = read_json_object(tmp_path / "quality.json")["runs"][0]
    evidence = write_json(
        tmp_path / "measured-calibration.json",
        {"fixture": "synthetic unit-test measurement only"},
    )
    calibration = {
        **{
            key: rows[0][key]
            for key in (
                "backend",
                "case_id",
                "config_sha256",
                "scene_sha256",
                "view_sha256",
                "profile_sha256",
            )
        },
        "reference_id": record["reference_id"],
        "photometric_calibration": {
            "status": "qualified",
            "reference_id": record["reference_id"],
            "profile_sha256": rows[0]["profile_sha256"],
            "evidence": [{"path": evidence.name, "sha256": sha(evidence)}],
        },
    }
    path = write_json(tmp_path / "calibration.json", {"records": [calibration]})
    with pytest.raises(ValueError, match="measured photometric_calibration_receipt"):
        evaluate.evaluate_experiment(
            tmp_path, spp=2, threads=1, calibration_evidence=path
        )
    updated = read_json_object(tmp_path / "quality.json")["runs"][0]
    assert updated["quality_status"] == "not_qualified"
    assert updated["checks"]["photometric_calibration"]["status"] == "missing"
    calibration["photometric_calibration"]["evidence"][0][
        "kind"
    ] = "photometric_calibration_receipt"
    tagged = write_json(
        tmp_path / "tagged-calibration.json", {"records": [calibration]}
    )
    with pytest.raises(ValueError, match="Measured calibration proof did not pass"):
        evaluate.evaluate_experiment(
            tmp_path, spp=2, threads=1, calibration_evidence=tagged
        )
    calibration["config_sha256"] = "wrong-case"
    wrong = write_json(tmp_path / "wrong-calibration.json", {"records": [calibration]})
    with pytest.raises(ValueError, match="bound to this frozen case"):
        evaluate.evaluate_experiment(
            tmp_path, spp=2, threads=1, calibration_evidence=wrong
        )


@pytest.mark.parametrize("status", ["passed", "failed", "missing"])
def test_calibration_attachment_follows_measured_verifier_and_retains_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    from scripts.benchmark.rendering import calibration as measured_calibration
    from scripts.benchmark.rendering.quality_protocol import protocol_for_modalities

    receipt = write_json(tmp_path / "receipt.json", {"fixture": "verifier wiring only"})
    source = write_json(tmp_path / "source.json", {"fixture": "verified source wiring"})
    row = {
        "backend": "embodichain",
        "case_id": "case",
        "config_sha256": "config",
        "scene_sha256": "scene",
        "view_sha256": "view",
        "profile_sha256": "profile",
    }
    claim = {
        "status": "qualified",
        "reference_id": "reference",
        "profile_sha256": "profile",
        "evidence": [
            {
                "kind": "photometric_calibration_receipt",
                "path": receipt.name,
                "sha256": sha(receipt),
            }
        ],
    }
    document = {
        "records": [
            {**row, "reference_id": "reference", "photometric_calibration": claim}
        ]
    }
    candidate = {"metadata": {**row, "photometric_calibration": "not_qualified"}}
    reference_frame = {"reference_provenance": {"reference_id": "reference"}}
    policy = protocol_for_modalities(("rgb",))
    calls = []

    def verify(
        value: dict[str, Any],
        metadata: dict[str, Any],
        ref: dict[str, Any],
        protocol: dict[str, Any],
    ) -> dict[str, Any]:
        calls.append(value)
        assert metadata is candidate["metadata"]
        assert ref is reference_frame
        assert protocol == policy
        assert value["evidence"][0]["path"] == str(receipt)
        return {
            "status": status,
            "reason": "heldout proof outcome",
            "evidence": {
                "source_evidence": [
                    {
                        "path": source.name,
                        "sha256": sha(source),
                        "kind": "calibration_frame_arrays",
                    }
                ]
            },
        }

    monkeypatch.setattr(measured_calibration, "verify_calibration", verify)
    if status == "passed":
        evaluate._calibration(
            tmp_path, document, row, candidate, reference_frame, policy
        )
        attached = candidate["metadata"]["photometric_calibration"]
        assert {item["path"] for item in attached["evidence"]} == {
            str(receipt),
            str(source),
        }
        assert len(claim["evidence"]) == 1
    else:
        with pytest.raises(ValueError, match="heldout proof outcome"):
            evaluate._calibration(
                tmp_path, document, row, candidate, reference_frame, policy
            )
        assert candidate["metadata"]["photometric_calibration"] == "not_qualified"
    assert len(calls) == 1
