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

import json
from pathlib import Path

import pytest

from examples.ods.simple_bc_demo.compare import (
    build_comparison,
    load_run,
    main,
    write_comparison,
)


def _run(
    root: Path,
    mode: str,
    seed: int,
    *,
    t80: float | None = 4,
    threshold_test: float | None = 0.9,
    data_ready: bool = False,
    resources: bool = False,
) -> Path:
    path = root / f"{mode}-{seed}"
    path.mkdir(parents=True)
    efficiency = {
        "observed_seconds": 10,
        "horizon_seconds": 20,
        "horizon_prespecified": True,
        "data_ready": data_ready,
        "end_reason": "updates",
        "efficiency/reached_80": t80 is not None,
        "efficiency/t80_dev_seconds": t80,
        "efficiency/censor_seconds": 10 if t80 is None else None,
        "efficiency/auc_success_time": 0.7,
        "cost/allocated_physical_gpu_ids": ["device0", "device0"],
        "cost/allocated_gpu_hours": 10 / 3600,
        "pipeline/data_wait_fraction": None,
    }
    if resources:
        efficiency["cost/host_pss_sampled_peak_bytes"] = 1024**2
        efficiency["cost/data_disk_peak_bytes"] = 2 * 1024**2
    summary = {
        "mode": mode,
        "policy": "mlp-pose",
        "stage": "showcase",
        "smoke": False,
        "threshold_test_success": threshold_test,
        "final_success_rate": 0.9,
    }
    config = {
        "seed": seed,
        "qualification_scenes": 40,
        "dev_scenes": 20,
        "final_test_scenes": 100,
        "batch_size": 256,
        "updates_per_round": 1000,
        "rounds": 5,
        "no_viewer": True,
        "no_video": True,
        "learning_rate": 0.001,
        "pose_ik": "dls",
    }
    timeline = {
        "events": [
            {"kind": "evaluation", "seconds": 2, "success_rate": 0.2},
            {"kind": "update", "seconds": 3, "targets": 256},
            {"kind": "evaluation", "seconds": 4, "success_rate": 0.6},
            {"kind": "evaluation", "seconds": 6, "success_rate": 0.3},
        ],
        "intervals": {},
    }
    data_manifest = {
        "schema": {
            "observation": {"names": ["joint", "cube_height"]},
            "action": {"names": ["pose_request"]},
        },
        # Paired seeds share initialization; different seeds may differ.
        "initial_model_sha256": f"{seed:064x}",
        "normalization_sha256": "b" * 64,
    }
    environment = {
        "python": "3.11.14",
        "torch": "2.7.0+cu128",
        "dexsim": "0.1.fixture",
        "commit": "fixture-commit",
        "source_sha256": {"fixture-source.py": "a" * 64},
        "allocated_gpu_hardware": "0, GPU-fixture-uuid, NVIDIA Fixture GPU, 24576 MiB",
        "cpu_affinity": [0, 1],
        "torch_num_threads": 1,
    }
    for name, value in (
        ("summary", summary),
        ("efficiency", efficiency),
        ("config", config),
        ("timeline", timeline),
        ("data_manifest", data_manifest),
        ("environment", environment),
    ):
        (path / f"{name}.json").write_text(json.dumps(value))
    return path


def test_run_loader_keeps_unknown_measurements_null_and_finds_training_seed(
    tmp_path: Path,
) -> None:
    record = load_run(_run(tmp_path, "online", 42))
    assert record["training_seed"] == 42
    assert record["data_wait_fraction"] is None
    assert record["host_pss_sampled_peak_bytes"] is None
    assert record["data_disk_peak_bytes"] is None
    assert record["allocated_gpu_count"] == 1  # Shared physical device counted once.


def test_development_curve_updates_only_at_result_arrival_and_holds_best_to_horizon(
    tmp_path: Path,
) -> None:
    curve = load_run(_run(tmp_path, "online", 42))["development_curve"]
    assert curve["seconds"] == [0, 2, 4, 6, 20]
    assert curve["best_success"] == [0, 0.2, 0.6, 0.6, 0.6]


def test_group_median_labels_attainers_and_keeps_censored_runs(tmp_path: Path) -> None:
    paths = [_run(tmp_path, "online", 1, t80=4), _run(tmp_path, "online", 2, t80=None)]
    report = build_comparison([load_run(p) for p in paths])
    group = report["groups"][0]
    assert group["runs"] == 2
    assert group["attained"] == 1
    assert group["censored"] == 1
    assert group["median_t80_dev_seconds_attainers"] == 4
    assert group["measured_runs"]["t80_dev_seconds"] == 1
    assert any("attainers only" in warning for warning in report["warnings"])


def test_all_qualified_pairs_report_individual_ratios_and_median(
    tmp_path: Path,
) -> None:
    paths = []
    for seed, offline_time in ((1, 8), (2, 12)):
        paths.extend(
            (
                _run(tmp_path, "online", seed, t80=4),
                _run(tmp_path, "offline-full", seed, t80=offline_time),
            )
        )
    comparison = build_comparison([load_run(p) for p in paths])["paired_comparisons"][0]
    assert comparison["ratios_by_seed"] == {"1": 2, "2": 3}
    assert comparison["median_ratio"] == 2.5
    assert comparison["ratios_supported"] is True
    assert (
        comparison["headline_efficiency_supported"] is False
    )  # Two seeds remain preliminary.


@pytest.mark.parametrize(
    "reason", ["censored", "failed_threshold_test", "missing_threshold_test"]
)
def test_one_unqualified_pair_omits_all_ratios_including_surviving_pairs(
    tmp_path: Path, reason: str
) -> None:
    paths = [
        _run(tmp_path, "online", 1),
        _run(tmp_path, "offline-full", 1, t80=8),
        _run(tmp_path, "online", 2),
    ]
    options = (
        {"t80": None}
        if reason == "censored"
        else {"threshold_test": 0.79 if reason == "failed_threshold_test" else None}
    )
    paths.append(_run(tmp_path, "offline-full", 2, **options))
    report = build_comparison([load_run(p) for p in paths])
    comparison = report["paired_comparisons"][0]
    assert comparison["complete_pairs"] == 2
    assert comparison["ratios_supported"] is False
    assert comparison["ratios_by_seed"] is None
    assert comparison["median_ratio"] is None
    assert comparison["omission_reasons"]
    assert any(
        "headline efficiency is unsupported" in warning
        for warning in report["warnings"]
    )


def test_unmatched_seed_blocks_surviving_pair_average(tmp_path: Path) -> None:
    paths = [
        _run(tmp_path, "online", 1),
        _run(tmp_path, "offline-full", 1),
        _run(tmp_path, "online", 2),
    ]
    comparison = build_comparison([load_run(p) for p in paths])["paired_comparisons"][0]
    assert comparison["intended_pairs"] == 2
    assert comparison["complete_pairs"] == 1
    assert comparison["ratios_by_seed"] is None


def test_cold_and_data_ready_scenarios_are_never_paired(tmp_path: Path) -> None:
    paths = [
        _run(tmp_path, "online", 1),
        _run(tmp_path, "offline-full", 1, data_ready=True),
    ]
    report = build_comparison([load_run(p) for p in paths])
    assert len(report["groups"]) == 2
    assert report["paired_comparisons"] == []
    assert report["headline_efficiency_supported"] is False


def test_full_five_pair_matrix_can_satisfy_headline_requirements(
    tmp_path: Path,
) -> None:
    paths = [
        _run(tmp_path, mode, seed)
        for seed in range(5)
        for mode in ("online", "offline-full")
    ]
    report = build_comparison([load_run(p) for p in paths])
    assert report["headline_efficiency_supported"] is True
    assert report["paired_comparisons"][0]["median_ratio"] == 1


def test_unmatched_serial_schedule_reports_descriptive_ratios_without_overlap_headline(
    tmp_path: Path,
) -> None:
    paths = [
        _run(tmp_path, mode, seed) for seed in range(5) for mode in ("online", "serial")
    ]
    for path in paths:
        if path.name.startswith("serial"):
            summary_path = path / "summary.json"
            summary = json.loads(summary_path.read_text())
            summary["matching"] = "unmatched_refresh_schedule"
            summary_path.write_text(json.dumps(summary))
    report = build_comparison([load_run(path) for path in paths])
    comparison = report["paired_comparisons"][0]
    assert comparison["matching"] == "unmatched_refresh_schedule"
    assert comparison["ratios_supported"] is True
    assert comparison["median_ratio"] == 1
    assert comparison["causal_overlap_headline_supported"] is False
    assert comparison["headline_efficiency_supported"] is False
    assert report["headline_efficiency_supported"] is False
    assert any(
        "descriptive" in note and "causal overlap" in note
        for note in comparison["qualification_notes"]
    )
    assert any(
        "matching=unmatched_refresh_schedule" in warning
        for warning in report["warnings"]
    )


def test_same_policy_with_different_controllers_cannot_support_efficiency_headline(
    tmp_path: Path,
) -> None:
    paths = [
        _run(tmp_path, mode, seed)
        for seed in range(5)
        for mode in ("online", "offline-full")
    ]
    for path in paths:
        if path.name.startswith("offline-full"):
            config_path = path / "config.json"
            config = json.loads(config_path.read_text())
            config["pose_ik"] = "solver"
            config_path.write_text(json.dumps(config))
    report = build_comparison([load_run(path) for path in paths])
    comparison = report["paired_comparisons"][0]
    assert comparison["ratios_supported"] is True
    assert comparison["ratios_descriptive_only"] is True
    assert comparison["comparability_supported"] is False
    assert comparison["headline_efficiency_supported"] is False
    assert report["headline_efficiency_supported"] is False
    assert any(
        "Different pose_ik" in note for note in comparison["comparability_notes"]
    )


def test_missing_environment_and_schema_evidence_is_not_treated_as_equality(
    tmp_path: Path,
) -> None:
    paths = [
        _run(tmp_path, mode, seed)
        for seed in range(5)
        for mode in ("online", "offline-full")
    ]
    for path in paths:
        (path / "data_manifest.json").unlink()
        (path / "environment.json").unlink()
    report = build_comparison([load_run(path) for path in paths])
    comparison = report["paired_comparisons"][0]
    assert comparison["ratios_supported"] is True
    assert comparison["ratios_descriptive_only"] is True
    assert report["headline_efficiency_supported"] is False
    assert any(
        "Missing observation_schema" in note
        for note in comparison["comparability_notes"]
    )
    assert any(
        "Missing gpu_model_memory_count" in note
        for note in comparison["comparability_notes"]
    )
    assert any(
        "initial_model_sha256" in note for note in comparison["comparability_notes"]
    )


@pytest.mark.parametrize(
    "filename,keys,value,evidence",
    [
        ("config", ("learning_rate",), 0.002, "learning_rate"),
        (
            "data_manifest",
            ("schema", "action", "names"),
            ["different_action"],
            "action_schema",
        ),
        (
            "data_manifest",
            ("schema", "observation", "names"),
            ["different_observation"],
            "observation_schema",
        ),
        (
            "environment",
            ("source_sha256", "fixture-source.py"),
            "c" * 64,
            "source_sha256",
        ),
        ("environment", ("dexsim",), "different-version", "source_versions"),
        (
            "environment",
            ("allocated_gpu_hardware",),
            "0, GPU-fixture, NVIDIA Other GPU, 24576 MiB",
            "gpu_model_memory_count",
        ),
        ("environment", ("cpu_affinity",), [2, 3], "cpu_affinity"),
        ("environment", ("torch_num_threads",), 2, "torch_num_threads"),
        ("data_manifest", ("initial_model_sha256",), "c" * 64, "initial_model_sha256"),
        ("data_manifest", ("normalization_sha256",), "c" * 64, "normalization_sha256"),
    ],
    ids=[
        "learning_rate",
        "action_schema",
        "observation_schema",
        "source_hashes",
        "native_version",
        "device_model",
        "cpu_affinity",
        "thread_count",
        "initial_weights",
        "normalization",
    ],
)
def test_changed_fingerprints_keep_qualified_time_ratios_descriptive(
    tmp_path: Path, filename: str, keys: tuple[str, ...], value: object, evidence: str
) -> None:
    paths = [
        _run(tmp_path, mode, seed)
        for seed in range(5)
        for mode in ("online", "offline-full")
    ]
    target = paths[1] / f"{filename}.json"
    payload = json.loads(target.read_text())
    container = payload
    for key in keys[:-1]:
        container = container[key]
    container[keys[-1]] = value
    target.write_text(json.dumps(payload))
    report = build_comparison([load_run(path) for path in paths])
    comparison = report["paired_comparisons"][0]
    assert comparison["ratios_supported"] is True
    assert comparison["ratios_descriptive_only"] is True
    assert report["headline_efficiency_supported"] is False
    assert any(
        evidence in note and "confound" in note
        for note in comparison["comparability_notes"]
    )


def test_equal_device_models_ignore_physical_uuid_index_and_cpu_affinity_order(
    tmp_path: Path,
) -> None:
    paths = [
        _run(tmp_path, mode, seed)
        for seed in range(5)
        for mode in ("online", "offline-full")
    ]
    for path in paths:
        if path.name.startswith("offline-full"):
            target = path / "environment.json"
            environment = json.loads(target.read_text())
            environment["allocated_gpu_hardware"] = (
                "3, GPU-other-physical-uuid, NVIDIA Fixture GPU, 24576 MiB"
            )
            environment["cpu_affinity"] = [1, 0]
            target.write_text(json.dumps(environment))
    report = build_comparison([load_run(path) for path in paths])
    assert report["headline_efficiency_supported"] is True
    assert report["paired_comparisons"][0]["comparability_supported"] is True


def test_missing_artifacts_are_reported_without_inventing_curves(
    tmp_path: Path,
) -> None:
    path = _run(tmp_path, "online", 42)
    (path / "timeline.json").unlink()
    report = build_comparison([load_run(path)])
    assert report["runs"][0]["development_curve"] is None
    assert report["runs"][0]["missing_artifacts"] == ["timeline.json"]
    assert any("artifacts are missing" in warning for warning in report["warnings"])


def test_report_writes_json_markdown_and_available_figures(tmp_path: Path) -> None:
    path = _run(tmp_path, "online", 42, resources=True)
    output = tmp_path / "report"
    report = write_comparison([path], output)
    assert set(report["figures"]) == {
        "development_success.png",
        "allocated_gpu_hours.png",
        "resource_peaks.png",
    }
    assert all((output / figure).stat().st_size > 0 for figure in report["figures"])
    saved = json.loads((output / "comparison.json").read_text())
    assert saved["runs"][0]["data_disk_peak_bytes"] == 2 * 1024**2
    markdown = (output / "comparison.md").read_text()
    assert "headline efficiency is unsupported" in markdown
    assert "Attained / total" in markdown


def test_missing_efficiency_does_not_invent_censoring_or_attainment(
    tmp_path: Path,
) -> None:
    path = _run(tmp_path, "online", 42)
    (path / "efficiency.json").unlink()
    report = build_comparison([load_run(path)])
    assert report["runs"][0]["reached_80"] is None
    assert report["runs"][0]["right_censored"] is None
    assert report["groups"][0]["censored"] == 0
    assert report["groups"][0]["unobserved"] == 1


@pytest.mark.parametrize("use_option", [False, True])
def test_cli_accepts_positional_or_runs_paths(tmp_path: Path, use_option: bool) -> None:
    path = _run(tmp_path, "online", 42)
    # Missing timelines/resources skip all figures; no data are substituted.
    (path / "timeline.json").unlink()
    output = tmp_path / "report"
    arguments = ["--runs", str(path)] if use_option else [str(path)]
    main([*arguments, "--output", str(output)])
    saved = json.loads((output / "comparison.json").read_text())
    assert saved["figures"] == []
    assert saved["runs"][0]["data_disk_peak_bytes"] is None


@pytest.mark.parametrize(
    "setting,value",
    [("action_horizon", 1), ("prediction_horizon", 128), ("observation_noise", 0.5)],
)
def test_chunk_policy_settings_are_part_of_comparison_evidence(
    tmp_path, setting, value
) -> None:
    from examples.ods.simple_bc_demo.compare import _fingerprint

    path = _run(tmp_path, "online", 0)
    run = load_run(path)
    run["policy"] = "chunk-pose"
    run["config"].update(
        prediction_horizon=64, action_horizon=16, observation_noise=0.01
    )
    run["data_manifest"]["normalization_fields"] = [
        "mean",
        "std",
        "target_mean",
        "target_std",
    ]
    first = _fingerprint(run)
    run["config"][setting] = value
    assert _fingerprint(run) != first
    del run["data_manifest"]["normalization_fields"]
    assert _fingerprint(run)["normalization_fields"] is None
