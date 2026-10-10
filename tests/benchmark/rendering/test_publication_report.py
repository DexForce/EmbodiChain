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

"""Frozen-fixture tests for pure-rendering publication reports; no GPU needed."""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from xml.etree import ElementTree

import pytest

from scripts.benchmark.core.artifacts import stable_hash
from scripts.benchmark.rendering.publication_report import rebuild_report
from scripts.benchmark.rendering.quality_protocol import (
    protocol_for_modalities,
    protocol_requirements,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _rows(
    *, experiment: str = "R-04", n: int = 1, cameras: int = 1, repeats: int = 2
) -> list[dict[str, Any]]:
    config = {
        "scene_id": "tabletop",
        "num_envs": n,
        "cameras_per_env": cameras,
        "width": 64,
        "height": 48,
        "modalities": ["rgb", "depth"],
        "profile": "realtime",
        "temporal_mode": "static",
        "delivery": "host_ready",
    }
    rows = []
    for backend in ("embodichain", "isaaclab"):
        for repeat in range(repeats):
            value = (10.0 if backend == "embodichain" else 5.0) * (repeat + 1)
            rows.append(
                {
                    "schema_version": 1,
                    "experiment_id": experiment,
                    "case_id": f"{experiment}-n{n}",
                    "run_id": f"{experiment}-n{n}-{backend}-{repeat}",
                    "backend": backend,
                    "repeat": repeat,
                    "status": "completed",
                    "quality_status": "qualified",
                    "hardware_id": "gpu-A",
                    "config_sha256": stable_hash(config),
                    "config": dict(config),
                    "scene_sha256": "scene-sha",
                    "view_sha256": "view-sha",
                    "profile_sha256": "profile-sha",
                    "metrics": {
                        "batch_steps_per_s": value,
                        "camera_frames_per_s": value * n * cameras,
                        "gpixel_per_s": value * n * cameras * 64 * 48 / 1e9,
                        "latency_p50_s": 1 / value,
                        "latency_p95_s": 2 / value,
                        "state_publish_p50_s": 0.001,
                        "device_ready_p50_s": 0.01,
                        "host_ready_p50_s": 0.012,
                        "gpu_process_peak_bytes": 2**30,
                        "gpu_process_steady_bytes": 2**29,
                        "gpu_device_peak_bytes": 2**31,
                        "cpu_peak_rss_bytes": 2**28,
                        "torch_peak_allocated_bytes": 2**27,
                        "d2h_bytes_per_capture": 64 * 48 * 4,
                        "host_copy_bytes_per_capture": 0,
                        "native_render_s": None,
                        "boundary": "state_publish_to_host_ready",
                        "completion": "blocking_host_copy",
                    },
                    "metric_reasons": {"native_render_s": "native_timer_unavailable"},
                }
            )
    return rows


def _freeze(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _write(
        root / "definition.json",
        {
            "schema_version": 1,
            "definition_version": "2.0",
            "experiment_id": "pure-rendering",
        },
    )
    _write(root / "runs.json", rows)
    evidence_dir = root / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    quality = []
    for row in rows:
        policy = protocol_for_modalities(tuple(row["config"]["modalities"]))
        requirements = protocol_requirements(policy["id"])
        metrics = {}
        for name, (comparison, threshold, unit) in policy["frame_thresholds"].items():
            required = name in requirements["required_metrics"]
            metrics[name] = {
                "comparison": comparison,
                "threshold": threshold,
                "unit": unit,
                "value": (
                    (
                        0.95 + row["repeat"] * 0.01
                        if name == "ssim"
                        else 1.0 if comparison == "ge" else 0.0
                    )
                    if required
                    else None
                ),
                "denominator": 64 * 48 if required else 0,
                "required": required,
                "status": "passed" if required else "not_requested",
                "reason": None if required else "modality_not_requested",
            }
        path = evidence_dir / f"{row['run_id']}.json"
        _write(path, {"run_id": row["run_id"], "checks": "frozen measured fixture"})
        quality.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "run_id",
                        "config_sha256",
                        "scene_sha256",
                        "view_sha256",
                        "profile_sha256",
                    )
                },
                "quality_status": "qualified",
                **requirements,
                "reference_id": "frozen-reference",
                "qualification": {
                    "passed": True,
                    "failed_checks": [],
                    "missing_checks": [],
                },
                "checks": {
                    name: {"status": "passed"}
                    for name in requirements["required_checks"]
                },
                "artifacts": [
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "kind": "quality_evidence",
                    }
                ],
                "metrics": metrics,
            }
        )
    _write(root / "quality.json", {"schema_version": 1, "runs": quality})
    return quality


def _summary(root: Path) -> dict[str, Any]:
    rebuild_report(root)
    return json.loads((root / "summary.json").read_text())


def _comparison(
    summary: dict[str, Any], metric: str = "camera_frames_per_s"
) -> dict[str, Any]:
    return next(item for item in summary["comparisons"] if item["metric"] == metric)


def test_partial_matrix_preserves_trials_missing_reasons_and_oom_endpoints(
    tmp_path: Path,
) -> None:
    rows = _rows() + _rows(n=4) + _rows(experiment="R-05", n=1, cameras=3)
    rows[0]["metrics"]["cpu_peak_rss_bytes"] = None
    rows[0]["metric_reasons"]["cpu_peak_rss_bytes"] = "sampler_unavailable"
    for index, (status, reason) in enumerate(
        (
            ("failed", "oom"),
            ("failed", "unsupported"),
            ("timeout", "timeout"),
            ("interrupted", "interrupted"),
            ("not_run", "not_run"),
            ("failed", "dependency_error"),
        )
    ):
        row = _rows(n=16)[0]
        row.update(
            run_id=f"failure-{index}",
            repeat=index,
            status=status,
            failure_reason_code=reason,
            quality_status="missing",
        )
        # Stale failure values must not become throughput observations.
        row["metrics"]["camera_frames_per_s"] = 1e12
        rows.append(row)
    _freeze(tmp_path, rows)
    planned = {**_rows(n=32)[0], "run_id": "planned-no-result"}
    _write(tmp_path / "manifest.json", {"schema_version": 1, "runs": [planned]})
    frozen = {
        name: (tmp_path / name).read_bytes()
        for name in ("runs.json", "quality.json", "manifest.json", "definition.json")
    }

    summary = _summary(tmp_path)

    assert summary["scheduled_runs"] == len(rows) + 1
    assert summary["status_counts"] == {
        "completed": 12,
        "failed": 3,
        "timeout": 1,
        "interrupted": 1,
        "not_run": 2,
    }
    failed = next(
        group for group in summary["groups"] if group["case_id"] == "R-04-n16"
    )
    assert failed["scheduled"] == 6
    assert failed["metrics"]["camera_frames_per_s"]["median"] is None
    assert failed["failure_reason_counts"]["oom"] == 1
    first = next(
        group
        for group in summary["groups"]
        if group["case_id"] == "R-04-n1" and group["backend"] == "embodichain"
    )
    assert first["metrics"]["cpu_peak_rss_bytes"]["missing_reasons"] == {
        "sampler_unavailable": 1
    }
    assert first["metrics"]["native_render_s"]["median"] is None
    capacity = next(
        item
        for item in summary["capacity"]
        if item["experiment_id"] == "R-04" and item["backend"] == "embodichain"
    )
    assert capacity["max_observed_completed_num_envs"] == 4
    assert capacity["min_observed_oom_num_envs"] == 16
    with (tmp_path / "metrics.csv").open() as stream:
        trials = list(csv.DictReader(stream))
    assert len(trials) == len(rows) + 1
    assert (
        next(item for item in trials if item["run_id"] == "planned-no-result")["status"]
        == "not_run"
    )
    assert {name: (tmp_path / name).read_bytes() for name in frozen} == frozen
    before = (tmp_path / "summary.json").read_bytes()
    rebuild_report(tmp_path)
    assert (tmp_path / "summary.json").read_bytes() == before


def test_ratios_use_paired_repeat_values_and_mad_not_ratio_of_medians(
    tmp_path: Path,
) -> None:
    rows = _rows()
    # Independent paired ratios are 10/5=2 and 100/100=1; their median is 1.5.
    rows[1]["metrics"]["camera_frames_per_s"] = 100.0
    rows[3]["metrics"]["camera_frames_per_s"] = 100.0
    _freeze(tmp_path, rows)

    comparison = _comparison(_summary(tmp_path))

    assert comparison["eligible"]
    assert comparison["pairs"] == 2
    assert comparison["ratio"] == pytest.approx(1.5)
    assert comparison["uncertainty"]["value"] == pytest.approx(0.5)
    assert comparison["ratio"] != pytest.approx(55.0 / 52.5)
    assert {
        run for pair in comparison["paired_runs"] for run in pair["source_runs"]
    } == {row["run_id"] for row in rows}


@pytest.mark.parametrize(
    "path,value",
    [
        ("hardware_id", "gpu-B"),
        ("hardware_id", None),
        ("scene_sha256", "different-scene"),
        ("view_sha256", "different-view"),
        ("profile_sha256", "different-profile"),
        ("config_sha256", "different-config"),
        ("config.profile", "high_quality"),
        ("config.delivery", "device_ready"),
        ("config.temporal_mode", "dynamic"),
        ("metrics.boundary", "other-boundary"),
        ("metrics.completion", "host_return_without_sync"),
    ],
)
def test_frozen_invariant_mismatch_prohibits_ratios(
    tmp_path: Path, path: str, value: object
) -> None:
    rows = _rows()
    target = rows[-1]
    parts = path.split(".")
    if len(parts) == 1:
        target[path] = value
    else:
        target[parts[0]][parts[1]] = value
    _freeze(tmp_path, rows)

    comparison = _comparison(_summary(tmp_path))

    assert not comparison["eligible"]
    assert path in comparison["reasons"]
    assert comparison["ratio"] is None
    assert all(pair["ratio"] is None for pair in comparison["paired_runs"])


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("missing_file", "quality_artifact_missing"),
        ("bad_checksum", "quality_artifact_checksum"),
        ("outside_root", "quality_artifact_outside_root"),
        ("unbound", "quality_binding:config_sha256"),
        ("no_artifacts", "quality_artifacts_missing"),
        ("failed_check", "quality_checks"),
        ("failed_metric", "quality_metrics"),
        ("not_qualified", "quality_evidence_status"),
        ("no_record", "quality_evidence_status"),
    ],
)
def test_quality_claim_requires_bound_measured_evidence(
    tmp_path: Path, problem: str, reason: str
) -> None:
    rows = _rows()
    quality = _freeze(tmp_path, rows)
    record = quality[0]
    if problem == "missing_file":
        (tmp_path / record["artifacts"][0]["path"]).unlink()
    elif problem == "bad_checksum":
        record["artifacts"][0]["sha256"] = "wrong"
    elif problem == "outside_root":
        record["artifacts"][0]["path"] = "../evidence.json"
    elif problem == "unbound":
        record["config_sha256"] = "wrong"
    elif problem == "no_artifacts":
        record["artifacts"] = []
    elif problem == "failed_check":
        record["checks"]["rgb_semantics"]["status"] = "failed"
    elif problem == "failed_metric":
        record["metrics"]["ssim"]["status"] = "failed"
    elif problem == "not_qualified":
        record["quality_status"] = "not_qualified"
    elif problem == "no_record":
        quality.pop(0)
    _write(tmp_path / "quality.json", {"schema_version": 1, "runs": quality})

    summary = _summary(tmp_path)

    assert _comparison(summary)["ratio"] is None
    assert reason in _comparison(summary)["reasons"]
    assert summary["groups"][0]["metrics"]["camera_frames_per_s"]["median"] is not None


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("single_repeat", "need_at_least_two_independent_paired_repeats"),
        ("unmatched", "unmatched_pairs"),
        ("missing_value", "missing_paired_metric_values"),
        ("duplicate_repeat", "duplicate_pair_identity"),
    ],
)
def test_repeated_pairing_is_required(
    tmp_path: Path, problem: str, reason: str
) -> None:
    rows = _rows()
    if problem == "single_repeat":
        rows = [row for row in rows if row["repeat"] == 0]
    elif problem == "unmatched":
        rows.pop()
    elif problem == "missing_value":
        rows[-1]["metrics"]["camera_frames_per_s"] = None
    elif problem == "duplicate_repeat":
        duplicate = dict(rows[0])
        duplicate["run_id"] = "another-process-same-pair-slot"
        rows.append(duplicate)
    _freeze(tmp_path, rows)

    comparison = _comparison(_summary(tmp_path))

    assert comparison["ratio"] is None
    assert reason in comparison["reasons"]


def test_hardware_is_not_pooled_and_charts_embed_csv_source_identities(
    tmp_path: Path,
) -> None:
    rows = (
        _rows()
        + _rows(n=4)
        + _rows(experiment="R-05", n=1, cameras=3)
        + _rows(experiment="R-09")
    )
    for row in rows:
        if row["experiment_id"] == "R-04" and row["repeat"] == 1:
            row["hardware_id"] = "gpu-B"
    _freeze(tmp_path, rows)

    summary = _summary(tmp_path)

    assert {g["hardware_id"] for g in summary["groups"]} == {"gpu-A", "gpu-B"}
    assert all(
        g["scheduled"] == 1 for g in summary["groups"] if g["experiment_id"] == "R-04"
    )
    assert {f["kind"] for f in summary["figures"]} >= {
        "scaling",
        "latency",
        "memory",
        "capacity",
        "delivery",
        "quality-performance",
    }
    figure = next(
        f
        for f in summary["figures"]
        if f["kind"] == "scaling" and f["scope"]["hardware_id"] == "gpu-B"
    )
    svg = ElementTree.parse(tmp_path / figure["svg"])
    provenance = json.loads(svg.find("{http://www.w3.org/2000/svg}metadata").text)
    plotted = [point for series in provenance["series"].values() for point in series]
    assert plotted and all("config_sha256" in point for point in plotted)
    assert all("-1" in run_id for point in plotted for run_id in point["run_ids"])
    pdf = (tmp_path / figure["pdf"]).read_bytes()
    assert pdf.startswith(b"%PDF-1.4") and pdf.endswith(b"%%EOF\n")
    assert b"/Subject" in pdf and rows[0]["config_sha256"].encode() in pdf
    assert plotted[0]["run_ids"][0].encode() in pdf
    xref = int(pdf.split(b"startxref\n")[1].splitlines()[0])
    assert pdf[xref:].startswith(b"xref\n")
    with (tmp_path / "figure_data.csv").open() as stream:
        data = list(csv.DictReader(stream))
    quality = [record for record in data if record["kind"] == "quality-performance"]
    assert quality and all(json.loads(item["quality_source_runs"]) for item in quality)
    assert all(item["low"] and item["high"] for item in quality)


def test_report_rebuild_works_without_site_packages_or_runtime_imports(
    tmp_path: Path,
) -> None:
    _freeze(tmp_path, [])
    script = """
import importlib.abc
import sys
from pathlib import Path
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'matplotlib', 'scipy', 'embodichain', 'dexsim', 'warp', 'isaaclab', 'isaacsim'}:
            raise AssertionError('Runtime import: ' + fullname)
        if fullname.startswith(('scripts.benchmark.rendering.backends', 'scripts.benchmark.rendering.suite')) or fullname == 'scripts.benchmark.rendering.quality':
            raise AssertionError('Domain worker import: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
from scripts.benchmark.rendering.publication_report import rebuild_report
print(rebuild_report(Path(sys.argv[1])))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "report.md").is_file()
    assert len(list((tmp_path / "figures").glob("*.pdf"))) >= 7


def test_missing_quality_and_hardware_remain_explicit(tmp_path: Path) -> None:
    rows = _rows()
    _freeze(tmp_path, rows)
    (tmp_path / "quality.json").unlink()
    for row in rows:
        row.pop("hardware_id")
    _write(tmp_path / "runs.json", rows)

    summary = _summary(tmp_path)

    comparison = _comparison(summary)
    assert comparison["ratio"] is None
    assert "hardware_id" in comparison["reasons"]
    assert "quality_artifacts_missing" in comparison["reasons"]
    assert summary["groups"][0]["hardware_id"] is None
    assert "missing" in (tmp_path / "report.md").read_text()


def test_conflicting_run_identities_and_wrong_definition_are_rejected(
    tmp_path: Path,
) -> None:
    rows = _rows()
    _freeze(tmp_path, rows)
    _write(tmp_path / "runs.json", rows + [rows[0]])
    with pytest.raises(ValueError, match="unique nonempty run_id"):
        rebuild_report(tmp_path)
    _write(tmp_path / "runs.json", rows)
    _write(
        tmp_path / "definition.json", {"schema_version": 1, "definition_version": "1.0"}
    )
    with pytest.raises(ValueError, match="definition_version"):
        rebuild_report(tmp_path)


def test_inconsistent_config_values_are_not_pooled_despite_same_hash(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows[1]["config"]["profile"] = "high_quality"
    _freeze(tmp_path, rows)

    summary = _summary(tmp_path)

    groups = [group for group in summary["groups"] if group["backend"] == "embodichain"]
    assert len(groups) == 2
    assert all(
        group["metrics"]["camera_frames_per_s"]["valid_runs"] == 1 for group in groups
    )
    assert not _comparison(summary)["eligible"]


def test_rgb_policy_qualifies_with_only_explicitly_unrequested_geometry_metrics(
    tmp_path: Path,
) -> None:
    rows = _rows()
    for row in rows:
        row["config"]["modalities"] = ["rgb"]
        row["config_sha256"] = stable_hash(row["config"])
    quality = _freeze(tmp_path, rows)

    assert _comparison(_summary(tmp_path))["eligible"]
    depth = quality[0]["metrics"]["depth_rmse_m"]
    assert depth["status"] == "not_requested" and depth["value"] is None
    assert depth["required"] is False
    depth["status"] = "missing"
    _write(tmp_path / "quality.json", {"schema_version": 1, "runs": quality})
    comparison = _comparison(_summary(tmp_path))
    assert comparison["ratio"] is None
    assert "quality_optional_metric:depth_rmse_m" in comparison["reasons"]


@pytest.mark.parametrize(
    "problem,reason",
    [
        ("trim_requirements", "quality_contract:required_metrics"),
        ("missing_requested", "quality_metric_schema"),
        ("missing_check", "quality_required_checks"),
        ("forged_status", "quality_metrics"),
        ("relaxed_threshold", "quality_metric_definition:ssim"),
        ("shared_wrong_hash", "quality_contract:protocol_sha256"),
    ],
)
def test_policy_requirements_cannot_be_shortened_or_forged(
    tmp_path: Path, problem: str, reason: str
) -> None:
    quality = _freeze(tmp_path, _rows())
    if problem == "trim_requirements":
        quality[0]["required_metrics"].remove("lpips")
    elif problem == "missing_requested":
        del quality[0]["metrics"]["lpips"]
    elif problem == "missing_check":
        del quality[0]["checks"]["photometric_calibration"]
    elif problem == "forged_status":
        quality[0]["metrics"]["ssim"]["value"] = 0.1
    elif problem == "relaxed_threshold":
        quality[0]["metrics"]["ssim"]["threshold"] = 0.1
    elif problem == "shared_wrong_hash":
        for record in quality:
            record["protocol_sha256"] = "same-unrecognized-policy"
    _write(tmp_path / "quality.json", {"schema_version": 1, "runs": quality})

    comparison = _comparison(_summary(tmp_path))

    assert comparison["ratio"] is None
    assert reason in comparison["reasons"]


@pytest.mark.parametrize(
    "modalities,temporal,reason",
    [
        (["rgb", "normals"], "static", "quality_unassessed_modalities"),
        (["rgb"], "moving", "quality_temporal_coverage"),
    ],
)
def test_unassessed_modality_or_temporal_scope_prohibits_equal_quality_ratio(
    tmp_path: Path, modalities: list[str], temporal: str, reason: str
) -> None:
    rows = _rows()
    for row in rows:
        row["config"].update(modalities=modalities, temporal_mode=temporal)
        row["config_sha256"] = stable_hash(row["config"])
    _freeze(tmp_path, rows)

    comparison = _comparison(_summary(tmp_path))

    assert comparison["ratio"] is None
    assert reason in comparison["reasons"]


@pytest.mark.parametrize("experiment", ["R-02", "R-03", "R-03-as-R-02"])
def test_three_camera_sequence_proof_requires_every_frame_contract(
    tmp_path: Path,
    experiment: str,
) -> None:
    rows = _rows(
        experiment="R-03" if experiment == "R-03-as-R-02" else experiment, cameras=3
    )
    for row in rows:
        if experiment == "R-03-as-R-02":
            row["experiment_id"] = "R-02"
            row["config"]["experiment_id"] = "R-03"
        row["config"].update(modalities=["rgb", "depth", "seg"], temporal_mode="moving")
        row["config_sha256"] = stable_hash(row["config"])
    quality = _freeze(tmp_path, rows)
    policy = protocol_for_modalities(("rgb", "depth", "seg"), temporal_dt_s=1 / 30)
    sequence_requirements = protocol_requirements(policy["id"], "sequence")
    for record in quality:
        frame = deepcopy(record)
        frame.update(protocol_requirements(policy["id"], "frame"))
        sequence = {
            **deepcopy(record),
            **sequence_requirements,
            "checks": {
                key: {"status": "passed"}
                for key in sequence_requirements["required_checks"]
            },
            "metrics": {
                key: {
                    "comparison": comparison,
                    "threshold": threshold,
                    "unit": unit,
                    "required": True,
                    "value": 1.0 if comparison == "ge" else 0.0,
                    "status": "passed",
                    "denominator": 99,
                }
                for key, (comparison, threshold, unit) in policy[
                    "temporal_thresholds"
                ].items()
            },
            "frames": [deepcopy(frame) for _ in range(100)],
        }
        record.update(
            {key: value for key, value in sequence.items() if key != "frames"}
        )
        record.update(
            sequences=[deepcopy(sequence) for _ in range(3)],
            coverage={
                "expected_samples": 300,
                "evaluated_samples": 300,
                "sequence_groups": 3,
            },
        )
        if experiment != "R-02":
            record["checks"]["temporal_evidence"] = {"status": "passed"}
    _write(tmp_path / "quality.json", {"schema_version": 1, "runs": quality})

    comparison = _comparison(_summary(tmp_path))
    if experiment != "R-02":
        assert comparison["eligible"] is False
        assert "quality_timed_history" in comparison["reasons"]
        if experiment == "R-03-as-R-02":
            assert "quality_experiment_binding" in comparison["reasons"]
        return
    assert comparison["eligible"]
    quality[0]["sequences"][2]["frames"][99]["metrics"]["lpips"]["value"] = 1.0
    _write(tmp_path / "quality.json", {"schema_version": 1, "runs": quality})
    comparison = _comparison(_summary(tmp_path))
    assert comparison["ratio"] is None
    assert "quality_sequence_frame_contract" in comparison["reasons"]


def test_quality_policy_must_match_the_frozen_case_assignment(tmp_path: Path) -> None:
    rows = _rows()
    _freeze(tmp_path, rows)
    _write(
        tmp_path / "definition.json",
        {
            "schema_version": 1,
            "definition_version": "2.0",
            "quality_protocol": {
                "case_protocols": {
                    rows[0]["case_id"]: {
                        "protocol_id": "pure-render-quality/v1",
                        "protocol_sha256": "other-frozen-policy",
                        "evaluation_kind": "frame",
                    }
                }
            },
        },
    )

    comparison = _comparison(_summary(tmp_path))

    assert comparison["ratio"] is None
    assert "frozen_quality_assignment:protocol_id" in comparison["reasons"]
