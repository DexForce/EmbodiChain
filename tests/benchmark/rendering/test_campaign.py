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

"""Simulator-free campaign conservation, storage and provenance tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.rendering import campaign
from scripts.benchmark.rendering.routing_proof import routing_env_ids


def _campaign(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(campaign, "git_revision", lambda root: campaign.EA_COMMIT)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *args, **kwargs: "")
    monkeypatch.setattr(
        campaign,
        "_runtime_evidence",
        lambda python: {
            "packages": {"isaacsim": "6.1.0.0", "nvidia-ml-py": "13.595.45"}
        },
    )
    monkeypatch.setattr(campaign, "_source_files", lambda: {"file.py": "frozen"})
    root = tmp_path / "campaign"
    campaign.initialize_campaign(
        root,
        scene_root=tmp_path / "scenes",
        embodichain_python=Path("python"),
        isaaclab_root=tmp_path / "isaac",
        isaaclab_python=Path("python"),
    )
    return root


def test_matrix_freezes_all_cases_and_exact_publication_core() -> None:
    cases = campaign.campaign_cases()
    assert len(cases) == 82
    assert len({c["case_id"] for c in cases}) == 82
    assert sum(c["publication_core"] for c in cases) == 16
    assert sum(c["case"]["deformation"] for c in cases) == 2
    selected = [c["case"] for c in cases if c["publication_core"]]
    assert sum(c["experiment_id"] == "R-03" for c in selected) == 8
    assert all(
        c["modalities"] == ["rgb"] or c["modalities"] == ("rgb",) for c in selected
    )


def test_initial_plan_retains_all_phase_populations_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _campaign(tmp_path, monkeypatch)
    definition, state = campaign._load(root)
    coverage = [j for j in state["jobs"] if j["phase"] == "coverage"]
    publication = [j for j in state["jobs"] if j["phase"] == "publication"]
    assert len(coverage) == 82 and len(publication) == 16
    assert all(j["repeats"] == 10 for j in publication)
    assert all(j["status"] == "not_run" for j in state["jobs"])
    with pytest.raises(FileExistsError):
        campaign.initialize_campaign(
            root,
            scene_root=root,
            embodichain_python=Path("python"),
            isaaclab_root=root,
            isaaclab_python=Path("python"),
        )
    definition["repeat_policy"]["publication"] = 1
    write_json(root / "campaign.json", definition)
    with pytest.raises(ValueError, match="definition changed"):
        campaign._load(root)


def test_storage_gate_counts_hardlinks_once_and_reserves_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _campaign(tmp_path, monkeypatch)
    definition, _ = campaign._load(root)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    original = evidence / "a"
    original.write_bytes(b"x" * 32)
    (evidence / "b").hardlink_to(original)
    definition["evidence_roots"] = [str(evidence), str(evidence)]
    definition["budget"] = {"evidence_bytes": 64, "minimum_free_bytes": 20}
    monkeypatch.setattr(
        campaign.shutil, "disk_usage", lambda path: SimpleNamespace(free=100)
    )
    gate = campaign.storage_gate(definition, reservation_bytes=16)
    assert gate["passed"] and gate["used_bytes"] == 32
    assert (
        campaign.storage_gate(definition, reservation_bytes=40)["reason"]
        == "evidence_budget"
    )
    monkeypatch.setattr(
        campaign.shutil, "disk_usage", lambda path: SimpleNamespace(free=30)
    )
    assert (
        campaign.storage_gate(definition, reservation_bytes=16)["reason"]
        == "disk_headroom"
    )


def test_generic_crash_and_freshness_failure_do_not_prove_capacity() -> None:
    assert campaign._capacity_reason([{"failure_reason_code": "crash"}]) is None
    assert (
        campaign._capacity_reason([{"failure_reason_code": "freshness_probe_failed"}])
        is None
    )
    assert campaign._capacity_reason([{"failure_reason_code": "oom"}]) == "oom"
    assert (
        campaign._capacity_reason(
            [
                {
                    "failure_reason_code": "freshness_probe_failed",
                    "renderer_validation": {
                        "status": "failed",
                        "events": [{"reason_code": "vulkan_sampler_allocation_limit"}],
                    },
                }
            ]
        )
        == "renderer_capacity_limit"
    )


def test_full_not_run_report_rebuild_is_offline_and_does_not_claim_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _campaign(tmp_path, monkeypatch)
    path = campaign.rebuild_campaign_report(root)
    summary = json.loads((root / "campaign-summary.json").read_text())
    assert len([r for r in summary["coverage"] if r["phase"] == "coverage"]) == 82
    assert all(
        r["status_counts"] == {} and r["qualified_runs"] == 0
        for r in summary["coverage"]
    )
    assert "not_run" in path.read_text()
    assert (root / "campaign-coverage.csv").exists()


def test_execution_rejects_source_drift_before_any_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(campaign, "_source_files", lambda: {"file.py": "changed"})
    with pytest.raises(ValueError, match="source changed"):
        campaign.run_campaign(root, phase="coverage")
    assert not (root / "coverage").exists()


@pytest.mark.parametrize(
    "n,expected",
    [
        (1, [0]),
        (16, [0, 8, 15]),
        (64, [0, 31, 32, 63]),
        (1024, [0, 31, 32, 511, 512, 991, 992, 1023]),
    ],
)
def test_routing_boundary_selection_is_bounded_and_explicit(
    n: int, expected: list[int]
) -> None:
    assert routing_env_ids(n) == expected


def test_capacity_stop_is_backend_specific_and_reservations_are_conserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmark.rendering import formal_runner

    root = _campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(
        campaign, "validate_approval", lambda path: {"status": "approved"}
    )
    monkeypatch.setattr(
        campaign, "rebuild_campaign_report", lambda root: root / "report.md"
    )
    monkeypatch.setattr(
        campaign, "storage_gate", lambda *args, **kwargs: {"passed": True}
    )
    selected = [
        r
        for r in campaign.campaign_cases()
        if r["case"]["experiment_id"] == "R-04" and r["case"]["num_envs"] in (16, 64)
    ]
    calls = []

    def launch(argv):
        backend = argv[argv.index("--backend") + 1]
        calls.append(backend)
        case_id = argv[argv.index("--case-id") + 1]
        output = Path(argv[argv.index("--output") + 1]) / "invocation"
        experiment = output / "R-04"
        experiment.mkdir(parents=True)
        write_json(output / "index.json", {"experiments": [{"path": "R-04"}]})
        rows = [
            {
                "selected": True,
                "backend": b,
                "case_id": case_id,
                "repeat": i,
                "run_id": f"{b}-{i}",
                "status": "failed" if b == "embodichain" else "completed",
                "failure_reason_code": "oom" if b == "embodichain" else None,
            }
            for b in (("embodichain", "isaaclab") if backend == "both" else (backend,))
            for i in range(3)
        ]
        write_json(experiment / "runs.json", rows)

    monkeypatch.setattr(formal_runner, "main", launch)
    campaign.run_campaign(
        root, phase="coverage", case_ids=tuple(r["case_id"] for r in selected)
    )
    _, state = campaign._load(root)
    jobs = [
        j
        for j in state["jobs"]
        if j["phase"] == "coverage" and j["case_id"] in {r["case_id"] for r in selected}
    ]
    assert calls == ["both", "isaaclab"]
    assert state["reserved_worker_runs"] == 9
    assert jobs[0]["attempts"][0]["status_counts"] == {"failed": 3, "completed": 3}
    assert jobs[1]["blocked_backends"]["embodichain"] == "oom:N=16"


def test_proof_selection_requires_original_completed_runs_and_separates_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmark.rendering.campaign_proofs import proof_population

    root = _campaign(tmp_path, monkeypatch)
    definition, state = campaign._load(root)
    core = next(
        r
        for r in definition["cases"]
        if r["publication_core"] and r["case"]["temporal_mode"] == "moving"
    )
    job = next(
        j
        for j in state["jobs"]
        if j["phase"] == "publication" and j["case_id"] == core["case_id"]
    )
    experiment = root / "example"
    experiment.mkdir()
    write_json(
        experiment / "runs.json",
        [
            {
                "selected": True,
                "status": status,
                "run_id": f"run-{i}",
                "backend": "isaaclab",
                "repeat": i,
                "case_id": core["case_id"],
            }
            for i, status in enumerate(("completed", "failed", "not_run"))
        ],
    )
    job["attempts"] = [{"experiments": ["example"]}]
    write_json(root / "campaign-state.json", state)
    assert [r["run_id"] for r in proof_population(root, "history")] == ["run-0"]
    assert [r["run_id"] for r in proof_population(root, "quality")] == ["run-0"]
    assert proof_population(root, "routing") == []


def test_campaign_figures_keep_phase_attempt_and_raw_source_identity(
    tmp_path: Path,
) -> None:
    metrics = {
        name: {
            "median": 10.0,
            "unit": (
                "1/s"
                if name == "camera_frames_per_s"
                else "byte" if "bytes" in name else "s"
            ),
            "valid_runs": 3,
            "uncertainty": {"value": 1.0},
            "source_runs": ["r0", "r1", "r2"],
        }
        for name in (
            "camera_frames_per_s",
            "latency_p50_s",
            "latency_p95_s",
            "gpu_process_steady_bytes",
            "gpu_process_peak_bytes",
        )
    }
    group = {
        "config": {
            "scene_id": "franka_workstation",
            "profile": "realtime",
            "temporal_mode": "static",
            "diagnostic": False,
            "cameras_per_env": 1,
            "delivery": "device_ready",
            "modalities": ["rgb"],
            "num_envs": 16,
        },
        "experiment_id": "R-04",
        "hardware_id": "uuid",
        "backend": "isaaclab",
        "case_id": "case",
        "config_sha256": "config",
        "source_runs": ["r0", "r1", "r2"],
        "qualified_runs": 0,
        "metrics": metrics,
    }
    records = [
        {
            "group": group,
            "phase": phase,
            "attempt": 0,
            "experiment": f"{phase}/original",
            "profile_sha256": "profile",
            "source_artifact_sha256": "raw-sha",
        }
        for phase in ("coverage", "publication")
    ]
    figures = campaign._campaign_figures(tmp_path, records)
    assert len(figures) == 10
    assert {f["scope"]["phase"] for f in figures} == {"coverage", "publication"}
    for figure in figures:
        assert (tmp_path / figure["svg"]).is_file() and (
            tmp_path / figure["pdf"]
        ).is_file()
        assert "raw-sha" in (tmp_path / figure["svg"]).read_text()
    assert "coverage/original" in (tmp_path / "campaign-figure-data.csv").read_text()


def test_worker_completion_is_selected_by_frozen_plan_when_flag_is_absent(
    tmp_path: Path,
) -> None:
    from scripts.benchmark.rendering.campaign import selected_case_rows

    planned = {
        "run_id": "actual",
        "backend": "isaaclab",
        "case_id": "chosen",
        "repeat": 0,
        "selected": True,
    }
    excluded = {**planned, "run_id": "excluded", "selected": False}
    write_json(tmp_path / "selection.json", {"trials": [planned, excluded]})
    worker = {k: v for k, v in planned.items() if k != "selected"}
    write_json(
        tmp_path / "runs.json",
        [{**worker, "status": "completed"}, {**excluded, "status": "not_run"}],
    )
    assert [row["run_id"] for row in selected_case_rows(tmp_path, "chosen")] == [
        "actual"
    ]
    write_json(
        tmp_path / "runs.json",
        [{**worker, "backend": "embodichain", "status": "completed"}],
    )
    with pytest.raises(ValueError, match="identity"):
        selected_case_rows(tmp_path, "chosen")
