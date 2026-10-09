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

"""Offline camera-count pairing, uncertainty and traceability contracts."""

from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import pytest

from scripts.benchmark.core.artifacts import read_json, stable_hash, write_json
from scripts.benchmark.rendering.camera_scaling_report import (
    rebuild_camera_scaling_report,
)
from scripts.benchmark.rendering.formal import view_fingerprint
from scripts.benchmark.rendering.quality_protocol import (
    protocol_for_modalities,
    protocol_requirements,
)
from scripts.benchmark.rendering.scenes import camera_intrinsics

ROOT = Path(__file__).resolve().parents[3]
MIB = 2**20


def _scene() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scene_id": "franka_workstation",
        "units": "metres",
        "up_axis": "Z",
        "physics_steps": 0,
        "camera_model": {"horizontal_fov_deg": 55.0, "near_m": 0.01, "far_m": 8.0},
        "objects": [
            {
                "uid": "table",
                "instance_id": 1,
                "geometry": {"kind": "box", "size": [1, 1, 0.1]},
            }
        ],
        "cameras": [
            {"name": name, "eye": [index, 0, 2], "target": [0, 0, 0], "up": [0, 1, 0]}
            for index, name in enumerate(("external_front", "external_side", "wrist"))
        ],
    }


def _quality(root: Path, row: dict[str, Any], cameras: int) -> dict[str, Any]:
    policy = protocol_for_modalities(tuple(row["config"]["modalities"]))
    requirements = protocol_requirements(policy["id"])
    proof = write_json(
        root / "evidence" / f"{row['run_id']}.json",
        {"fixture": "CPU-only measured-proof contract", "run_id": row["run_id"]},
    )
    metrics = {}
    for name, (comparison, threshold, unit) in policy["frame_thresholds"].items():
        required = name in requirements["required_metrics"]
        metrics[name] = {
            "value": (1.0 if comparison == "ge" else 0.0) if required else None,
            "comparison": comparison,
            "threshold": threshold,
            "unit": unit,
            "denominator": 128 * 128 if required else 0,
            "required": required,
            "status": "passed" if required else "not_requested",
            "reason": None if required else "modality_not_requested",
        }
    return {
        **{
            name: row[name]
            for name in (
                "run_id",
                "config_sha256",
                "scene_sha256",
                "view_sha256",
                "profile_sha256",
            )
        },
        **requirements,
        "reference_id": f"reference-{cameras}-views",
        "quality_status": "qualified",
        "qualification": {"passed": True, "failed_checks": [], "missing_checks": []},
        "checks": {
            name: {"status": "passed"} for name in requirements["required_checks"]
        },
        "metrics": metrics,
        "coverage": {
            "expected_samples": cameras,
            "evaluated_samples": cameras,
            "camera_names": [view["name"] for view in _scene()["cameras"][:cameras]],
        },
        "artifacts": [
            {
                "path": proof.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(proof.read_bytes()).hexdigest(),
                "kind": "quality_evidence",
            }
        ],
    }


def _freeze(
    root: Path,
    role: str,
    *,
    ns: tuple[int, ...] = (1, 16),
    repeats: int = 2,
    qualified: bool = False,
    backend: str = "embodichain",
) -> list[dict[str, Any]]:
    root.mkdir()
    cameras = 1 if role == "R-04" else 3
    scene = _scene()
    scene_sha = stable_hash(scene)
    profile = {"name": "realtime", "version": "frozen-fixture", "same_settings": True}
    write_json(root / "scenes/franka_workstation/scene.json", scene)
    write_json(
        root / "approval.json",
        {
            "status": "approved",
            "scope": "scene_layout_and_camera_scheme",
            "scenes": {"franka_workstation": scene_sha},
        },
    )
    rows, cases = [], []
    for n in ns:
        config = {
            "experiment_id": role,
            "definition_version": "2.0",
            "scene_id": "franka_workstation",
            "num_envs": n,
            "cameras_per_env": cameras,
            "width": 128,
            "height": 128,
            "modalities": ["rgb"],
            "profile": "realtime",
            "temporal_mode": "static",
            "delivery": "device_ready",
            "view_count": 1,
            "diagnostic": False,
        }
        case_id = f"{role}-n{n}"
        cases.append(config)
        write_json(
            root / "cases" / case_id / "worker-config.json",
            {
                "case": config,
                "profile": profile,
                "warmup_frames": 30,
                "measured_frames": 300,
            },
        )
        poses = [
            [[1, 0, 0, i], [0, 1, 0, 0], [0, 0, 1, 2], [0, 0, 0, 1]]
            for i in range(cameras)
        ]
        for repeat in range(repeats):
            rate = (10.0 + 10 * repeat) if cameras == 1 else (6.0 + 2 * repeat)
            row = {
                "schema_version": 1,
                "experiment_id": role,
                "run_id": f"n{n}-r{repeat}",
                "case_id": case_id,
                "backend": backend,
                "repeat": repeat,
                "status": "completed",
                "hardware_id": "GPU-fixture",
                "quality_status": "qualified" if qualified else "not_qualified",
                "config": deepcopy(config),
                "config_sha256": stable_hash(config),
                "scene_sha256": scene_sha,
                "view_sha256": view_fingerprint(SimpleNamespace(**config), scene),
                "profile_sha256": stable_hash(profile),
                "counts": {"batch_steps": 300},
                "warmup_counts": {"batch_steps": 30},
                "metrics": {
                    "batch_steps_per_s": rate,
                    "camera_frames_per_s": rate * n * cameras,
                    "gpu_process_steady_bytes": (
                        (100 + 10 * repeat) if cameras == 1 else (130 + 50 * repeat)
                    )
                    * MIB,
                    "gpu_process_peak_bytes": (
                        (200 + 10 * repeat) if cameras == 1 else (260 + 50 * repeat)
                    )
                    * MIB,
                    "boundary": "render_to_device_ready",
                    "completion": "selected_device_synchronized",
                },
                "renderer": {
                    "camera_names": [
                        view["name"] for view in scene["cameras"][:cameras]
                    ],
                    "camera_poses": deepcopy(poses),
                    "intrinsics": [
                        camera_intrinsics(scene, 128, 128) for _ in range(cameras)
                    ],
                    "effective_profile": profile,
                },
            }
            if backend == "isaaclab":
                row["renderer"].pop("camera_poses")
                row["renderer"]["camera_positions"] = [
                    [env * 4, camera, 2]
                    for env in range(n)
                    for camera in range(cameras)
                ]
                row["renderer"]["camera_quaternions_xyzw"] = [
                    [0, 0, 0, 1] for _ in range(n * cameras)
                ]
                row["renderer"]["intrinsics"] = [
                    camera_intrinsics(scene, 128, 128) for _ in range(n * cameras)
                ]
            rows.append(row)
    write_json(
        root / "definition.json",
        {
            "schema_version": 1,
            "definition_version": "2.0",
            "experiment_id": role,
            "parameter_matrix": {"case": cases},
        },
    )
    write_json(root / "runs.json", rows)
    write_json(
        root / "manifest.json",
        {
            "runs": [
                {key: row[key] for key in ("run_id", "backend", "case_id", "repeat")}
                for row in rows
            ]
        },
    )
    write_json(
        root / "quality.json",
        {
            "schema_version": 1,
            "runs": [_quality(root, row, cameras) for row in rows] if qualified else [],
        },
    )
    return rows


def _report(tmp_path: Path) -> dict[str, Any]:
    rebuild_camera_scaling_report(
        tmp_path / "r04", tmp_path / "r05", tmp_path / "comparison"
    )
    return read_json(tmp_path / "comparison/summary.json")


def _group(summary: dict[str, Any], n: int = 1) -> dict[str, Any]:
    return next(group for group in summary["groups"] if group["scope"]["num_envs"] == n)


def test_paired_scaling_ratios_deltas_uncertainty_and_vector_traceability(
    tmp_path: Path,
) -> None:
    left = _freeze(tmp_path / "r04", "R-04")
    right = _freeze(tmp_path / "r05", "R-05")
    before = {
        path: path.read_bytes()
        for root in (tmp_path / "r04", tmp_path / "r05")
        for path in root.rglob("*")
        if path.is_file()
    }
    summary = _report(tmp_path)
    group = _group(summary)
    metric = group["metrics"]["batch_steps_per_s:ratio"]
    assert metric["median"] == pytest.approx(0.5)  # median paired ratios, not 7/15
    assert metric["uncertainty"]["value"] == pytest.approx(0.1)
    assert metric["valid_pairs"] == metric["planned_pair_slots"] == 2
    assert group["metrics"]["camera_frames_per_s:ratio"]["median"] == pytest.approx(1.5)
    assert group["metrics"]["camera_frames_per_s:delta"]["median"] == 6
    assert group["metrics"]["gpu_process_steady_bytes:delta"]["median"] == 50 * MIB
    assert group["metrics"]["gpu_process_peak_bytes:delta"]["median"] == 80 * MIB
    assert not metric["equivalence_qualified"]
    assert summary["matched_pairs"] == 4
    assert summary["source_trial_count"] == 8
    assert not summary["equal_quality_speedup_claim"]
    assert len(summary["figures"]) == 6
    for figure in summary["figures"]:
        svg = tmp_path / "comparison" / figure["svg"]
        pdf = tmp_path / "comparison" / figure["pdf"]
        tree = ElementTree.parse(svg)
        metadata = tree.find("{http://www.w3.org/2000/svg}metadata").text
        assert "R-04:n1-r0" in metadata and "R-05:n1-r0" in metadata
        assert (
            left[0]["config_sha256"] in metadata
            and right[0]["config_sha256"] in metadata
        )
        assert pdf.read_bytes().startswith(b"%PDF-1.4")
        assert b"R-05:n1-r0" in pdf.read_bytes()
    with (tmp_path / "comparison/figure_data.csv").open() as stream:
        records = list(csv.DictReader(stream))
    memory = next(
        record for record in records if record["metric"] == "gpu_process_steady_bytes"
    )
    assert float(memory["y"]) == 50  # Plot unit is MiB; paired CSV retains bytes.
    assert all(path.read_bytes() == original for path, original in before.items())


@pytest.mark.parametrize(
    "field",
    [
        "hardware_id",
        "profile_sha256",
        "scene_sha256",
        "delivery",
        "width",
        "modalities",
        "completion",
        "window",
        "front_pose",
        "front_intrinsics",
        "view_sha256",
    ],
)
def test_invariant_mismatch_blocks_arithmetic_and_retains_trials(
    tmp_path: Path, field: str
) -> None:
    _freeze(tmp_path / "r04", "R-04", ns=(1,), repeats=1)
    rows = _freeze(tmp_path / "r05", "R-05", ns=(1,), repeats=1)
    row = rows[0]
    if field in {"delivery", "width", "modalities"}:
        row["config"][field] = {
            "delivery": "host_ready",
            "width": 256,
            "modalities": ["rgb", "depth"],
        }[field]
    elif field == "completion":
        row["metrics"][field] = "asynchronous_submission"
    elif field == "window":
        row["counts"]["batch_steps"] = 4
    elif field == "front_pose":
        row["renderer"]["camera_poses"][0][0][3] = 99
    elif field == "front_intrinsics":
        row["renderer"]["intrinsics"][0][0][0] += 10
    else:
        row[field] = None if field == "hardware_id" else "wrong"
    write_json(tmp_path / "r05/runs.json", rows)
    summary = _report(tmp_path)
    assert summary["source_trial_count"] == 2
    assert summary["matched_pairs"] == 0
    trial = summary["trials"][0]
    assert trial["reasons"]
    assert trial["metrics"]["camera_frames_per_s"]["ratio"] is None
    assert trial["metrics"]["gpu_process_peak_bytes"]["delta"] is None


def test_failures_missing_records_repeat_gaps_and_partial_metrics_stay_explicit(
    tmp_path: Path,
) -> None:
    _freeze(tmp_path / "r04", "R-04", ns=(1,), repeats=3)
    rows = _freeze(tmp_path / "r05", "R-05", ns=(1,), repeats=4)
    rows[0].update(status="failed", failure_reason_code="oom")
    rows[0]["metrics"]["batch_steps_per_s"] = 1e12
    rows[1]["metrics"]["gpu_process_steady_bytes"] = None
    rows[1]["metric_reasons"] = {"gpu_process_steady_bytes": "sampler_unavailable"}
    rows = [
        row for row in rows if row["repeat"] != 2
    ]  # Still present in the frozen plan.
    write_json(tmp_path / "r05/runs.json", rows)
    summary = _report(tmp_path)
    assert summary["source_trial_count"] == 7
    assert summary["planned_pair_slots"] == 4
    assert summary["status_counts"] == {"completed": 5, "failed": 1, "not_run": 1}
    by_repeat = {trial["repeat"]: trial for trial in summary["trials"]}
    assert by_repeat[0]["metrics"]["batch_steps_per_s"]["ratio"] is None
    assert "R-05:status:failed:oom" in by_repeat[0]["reasons"]
    assert by_repeat[1]["metrics"]["batch_steps_per_s"]["ratio"] is not None
    assert by_repeat[1]["metrics"]["gpu_process_steady_bytes"]["delta"] is None
    assert (
        "R-05:sampler_unavailable"
        in by_repeat[1]["metrics"]["gpu_process_steady_bytes"]["reasons"]
    )
    assert "missing:R-04" in by_repeat[3]["reasons"]
    ratio = _group(summary)["metrics"]["batch_steps_per_s:ratio"]
    assert ratio["uncertainty"]["value"] is None


def test_duplicate_repeat_is_not_silently_overwritten(tmp_path: Path) -> None:
    _freeze(tmp_path / "r04", "R-04", ns=(1,), repeats=1)
    rows = _freeze(tmp_path / "r05", "R-05", ns=(1,), repeats=1)
    duplicate = deepcopy(rows[0])
    duplicate["run_id"] = "duplicate-process"
    rows.append(duplicate)
    write_json(tmp_path / "r05/runs.json", rows)
    summary = _report(tmp_path)
    assert summary["source_trial_count"] == 3
    assert summary["matched_pairs"] == 0
    assert "ambiguous_repeat:R-05" in summary["trials"][0]["reasons"]


def test_isaac_front_pose_matches_every_environment_not_only_first(
    tmp_path: Path,
) -> None:
    _freeze(tmp_path / "r04", "R-04", ns=(16,), repeats=1, backend="isaaclab")
    rows = _freeze(tmp_path / "r05", "R-05", ns=(16,), repeats=1, backend="isaaclab")
    assert _report(tmp_path)["matched_pairs"] == 1
    rows[0]["renderer"]["camera_quaternions_xyzw"][15 * 3] = [0, 0, 0, -1]
    write_json(tmp_path / "r05/runs.json", rows)
    assert (
        _report(tmp_path)["matched_pairs"] == 1
    )  # Quaternion sign is not a changed view.
    rows[0]["renderer"]["camera_positions"][15 * 3][0] += 1
    write_json(tmp_path / "r05/runs.json", rows)
    summary = _report(tmp_path)
    assert summary["matched_pairs"] == 0
    assert "mismatch:observed_external_front" in summary["trials"][0]["reasons"]


def test_malformed_camera_metadata_does_not_establish_matching_view(
    tmp_path: Path,
) -> None:
    _freeze(tmp_path / "r04", "R-04", ns=(1,), repeats=1)
    rows = _freeze(tmp_path / "r05", "R-05", ns=(1,), repeats=1)
    rows[0]["renderer"]["intrinsics"][0] = []
    write_json(tmp_path / "r05/runs.json", rows)
    summary = _report(tmp_path)
    assert summary["matched_pairs"] == 0
    assert "R-05:observed_external_front_missing" in summary["trials"][0]["reasons"]
    assert len(summary["figures"]) == 1


def test_both_bound_quality_proofs_required_for_qualified_label(tmp_path: Path) -> None:
    _freeze(tmp_path / "r04", "R-04", ns=(1,), qualified=True)
    _freeze(tmp_path / "r05", "R-05", ns=(1,), qualified=True)
    summary = _report(tmp_path)
    assert all(trial["both_quality_proofs_verified"] for trial in summary["trials"])
    assert _group(summary)["metrics"]["batch_steps_per_s:ratio"][
        "equivalence_qualified"
    ]
    proof = tmp_path / "r05/evidence/n1-r0.json"
    proof.write_text("altered proof")
    summary = _report(tmp_path)
    assert summary["matched_pairs"] == 2  # Descriptive scaling remains available.
    assert not summary["trials"][0]["equivalence_qualified"]
    assert "R-05:quality_artifact_checksum" in summary["trials"][0]["quality_reasons"]
    assert not _group(summary)["metrics"]["batch_steps_per_s:ratio"][
        "equivalence_qualified"
    ]
    assert not summary["equal_quality_speedup_claim"]


def test_offline_cli_imports_no_runtime_and_rejects_legacy_r05(tmp_path: Path) -> None:
    _freeze(tmp_path / "r04", "R-04", ns=(1,), repeats=1)
    _freeze(tmp_path / "r05", "R-05", ns=(1,), repeats=1)
    script = """
import sys
from scripts.benchmark.rendering.camera_scaling_report import main
main(sys.argv[1:])
assert not any(name in sys.modules for name in ('torch', 'numpy', 'sapien', 'isaaclab', 'dexsim'))
"""
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            script,
            "--r04",
            str(tmp_path / "r04"),
            "--r05",
            str(tmp_path / "r05"),
            "--output",
            str(tmp_path / "offline"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    definition = read_json(tmp_path / "r05/definition.json")
    definition["definition_version"] = "1.0"
    write_json(tmp_path / "r05/definition.json", definition)
    with pytest.raises(ValueError, match="formal v2 R-05"):
        _report(tmp_path)
