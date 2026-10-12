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

"""Frozen RTX 5090 / Isaac Lab EA campaign orchestration and offline coverage.

Run: python -S -m scripts.benchmark.rendering.campaign --help
The existing formal launcher owns every trial. Campaign state references those
immutable invocations and keeps coverage and ten-repeat publication populations
separate. Missing quality or capacity evidence never becomes a performance claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

from scripts.benchmark.core.artifacts import read_json_object, write_json, stable_hash
from scripts.benchmark.core.provenance import git_revision
from scripts.benchmark.reporting.tables import render_markdown_table
from .formal import (
    EXPERIMENTS,
    RenderCase,
    experiment_cases,
    render_profile,
    validate_approval,
)

__all__ = [
    "selected_case_rows",
    "campaign_cases",
    "initialize_campaign",
    "run_campaign",
    "rebuild_campaign_report",
    "storage_gate",
    "main",
]


def selected_case_rows(experiment: Path, case_id: str) -> list[dict[str, Any]]:
    """Resolve selected workers from the frozen plan, including flagless results.

    Args:
        experiment: Formal experiment containing its plan and result projection.
        case_id: Exact frozen campaign case identity.

    Returns:
        Original selected rows without adding or changing measurement fields.
    """
    rows = json.loads((experiment / "runs.json").read_text())
    plan_path = experiment / "selection.json"
    if not plan_path.exists():
        return [
            row for row in rows if row.get("selected") and row["case_id"] == case_id
        ]
    plans = {
        row["run_id"]: row
        for row in read_json_object(plan_path)["trials"]
        if row.get("selected") and row["case_id"] == case_id
    }
    selected = []
    for row in rows:
        planned = plans.get(row["run_id"])
        if planned is None:
            continue
        if any(
            row.get(key) != planned[key] for key in ("backend", "case_id", "repeat")
        ):
            raise ValueError("Worker identity differs from frozen selected plan")
        selected.append(row)
    return selected


ROOT = Path(__file__).resolve().parents[3]
EA_COMMIT = "ae37b028ea415c91ea2bc32609efcd759ed2b974"
_GIB = 1024**3
_BACKENDS = ("embodichain", "isaaclab")


def campaign_cases() -> list[dict[str, Any]]:
    """Return all 82 cells and the prespecified 16 publication points."""
    records = []
    for experiment in EXPERIMENTS:
        for case in experiment_cases(experiment):
            core = (
                (
                    experiment == "R-03"
                    and case.scene_id == "franka_workstation"
                    and case.width in (256, 1920)
                    and case.modalities == ("rgb",)
                )
                or (experiment in ("R-04", "R-05") and case.num_envs in (1, 16, 64))
                or (
                    experiment == "R-09"
                    and not case.diagnostic
                    and case.delivery in ("device_ready", "host_ready")
                )
            )
            smoke = (
                (
                    experiment == "R-03"
                    and case.scene_id == "franka_workstation"
                    and case.width == 256
                    and case.modalities == ("rgb", "depth", "seg")
                )
                or (experiment in ("R-04", "R-05") and case.num_envs in (1, 16))
                or (experiment == "R-09" and core)
            )
            records.append(
                {
                    "case_id": case.case_id,
                    "case": case.to_dict(),
                    "publication_core": core,
                    "smoke": smoke,
                }
            )
    return records


def _runtime_evidence(python: Path) -> dict[str, Any]:
    script = "import json,sys; from importlib.metadata import version,PackageNotFoundError; names=['isaaclab','isaacsim','torch','numpy','nvidia-ml-py','dexsim-engine']; values={};\nfor name in names:\n try: values[name]=version(name)\n except PackageNotFoundError: values[name]=None\nprint(json.dumps({'python':sys.version,'packages':values}))"
    return json.loads(
        subprocess.check_output([str(python.absolute()), "-c", script], text=True)
    )


def _source_files() -> dict[str, str]:
    paths = (
        sorted((ROOT / "scripts/benchmark/core").glob("*.py"))
        + sorted((ROOT / "scripts/benchmark/rendering").glob("*.py"))
        + sorted((ROOT / "scripts/benchmark/rendering/backends").glob("*.py"))
    )
    return {
        p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def initialize_campaign(
    root: Path,
    *,
    scene_root: Path,
    embodichain_python: Path,
    isaaclab_root: Path,
    isaaclab_python: Path,
    evidence_roots: tuple[Path, ...] = (),
) -> Path:
    """Freeze cases, profiles, runtime identity and all planned phase slots.

    Args:
        root: New campaign directory, which may already contain previews.
        scene_root: Exact reviewed canonical scene package.
        embodichain_python: DexSim-capable interpreter.
        isaaclab_root: Clean v3.0.0-EA source checkout.
        isaaclab_python: Matching Isaac Sim 6.1 environment.
        evidence_roots: Additional history/reference directories within budget.

    Returns:
        Saved immutable campaign definition. Execution state is separate.
    """
    root = Path(root).absolute()
    if (root / "campaign.json").exists():
        raise FileExistsError("Campaign is already frozen; use a new directory")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ):
        raise ValueError("Commit benchmark source before freezing a campaign")
    isaaclab_root = Path(isaaclab_root).absolute()
    if git_revision(isaaclab_root) != EA_COMMIT:
        raise ValueError("Campaign requires the exact v3.0.0-EA source commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=isaaclab_root,
        text=True,
    )
    if dirty:
        raise ValueError("Campaign requires clean external Isaac source")
    isaac = _runtime_evidence(isaaclab_python)
    if (
        isaac["packages"]["isaacsim"] != "6.1.0.0"
        or not isaac["packages"]["nvidia-ml-py"]
    ):
        raise ValueError(
            "EA requires Isaac Sim 6.1.0.0 and NVML in its selected environment"
        )
    definition = {
        "schema_version": 1,
        "campaign_id": "rtx5090-isaaclab-ea-v1",
        "definition_version": "2.0",
        "target_hardware": "NVIDIA GeForce RTX 5090",
        "source_commit": git_revision(ROOT),
        "source_files": _source_files(),
        "isaaclab_commit": EA_COMMIT,
        "runtime": {
            "isaaclab": isaac,
            "embodichain": _runtime_evidence(embodichain_python),
        },
        "scene_root": str(Path(scene_root).absolute()),
        "interpreters": {
            "embodichain": str(embodichain_python.absolute()),
            "isaaclab": str(isaaclab_python.absolute()),
        },
        "isaaclab_root": str(isaaclab_root),
        "gpu_id": 0,
        "profiles": {
            name: render_profile(name) for name in ("realtime", "high_quality")
        },
        "repeat_policy": {
            "coverage": 3,
            "publication": 10,
            "quality_capture": 1,
            "smoke": 1,
        },
        "warmup_frames": 30,
        "measured_frames": 300,
        "timeout_s": 900,
        "history_timeout_s": 3600,
        "evidence_policy": "complete_core_history; diagnostic_noncore; no_history_no_temporal_qualification",
        "reference_precision_policy": "reference-noise/4096-16384-v1",
        "budget": {
            "evidence_bytes": 100 * _GIB,
            "minimum_free_bytes": 20 * _GIB,
            "retry_fraction_of_total": 0.2,
        },
        "proof_root": str(ROOT.parent / "evidence" / root.name),
        "evidence_roots": [
            str(root),
            str(ROOT.parent / "evidence" / root.name),
            *[str(p.absolute()) for p in evidence_roots],
        ],
        "cases": campaign_cases(),
    }
    jobs = []
    for phase in ("smoke", "coverage", "publication"):
        for record in definition["cases"]:
            if (
                phase == "smoke"
                and not record["smoke"]
                or phase == "publication"
                and not record["publication_core"]
            ):
                continue
            quality = record["case"]["experiment_id"] in ("R-01", "R-02")
            repeats = (
                1 if quality or phase == "smoke" else definition["repeat_policy"][phase]
            )
            jobs.append(
                {
                    "job_id": f"{phase}-{record['case_id']}",
                    "phase": phase,
                    "case_id": record["case_id"],
                    "repeats": repeats,
                    "status": "not_run",
                    "attempts": [],
                    "blocked_backends": {},
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    definition["budget"]["planned_worker_runs"] = sum(
        job["repeats"] * 2 for job in jobs
    )
    definition["budget"]["maximum_worker_runs"] = math.ceil(
        definition["budget"]["planned_worker_runs"] / 0.8
    )
    write_json(root / "campaign.json", definition)
    write_json(
        root / "campaign-state.json",
        {
            "definition_sha256": stable_hash(definition),
            "jobs": jobs,
            "reserved_worker_runs": 0,
        },
    )
    return root / "campaign.json"


def _load(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    definition = read_json_object(root / "campaign.json")
    state = read_json_object(root / "campaign-state.json")
    if state["definition_sha256"] != stable_hash(definition):
        raise ValueError("Frozen campaign definition changed")
    return definition, state


def storage_gate(
    definition: dict[str, Any], *, reservation_bytes: int = 0
) -> dict[str, Any]:
    """Check all evidence roots and filesystem headroom before a new operation.

    Args:
        definition: Frozen campaign budget and evidence locations.
        reservation_bytes: Uncompressed upper estimate for the next acquisition.

    Returns:
        Measured bytes and an explicit blocked reason or a passing gate.
    """
    if reservation_bytes < 0:
        raise ValueError("Storage reservation must be nonnegative")
    seen = set()
    used = 0
    for root in definition["evidence_roots"]:
        for path in Path(root).rglob("*"):
            if path.is_file():
                stat = path.stat()
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    seen.add(identity)
                    used += stat.st_size
    free = shutil.disk_usage(definition["evidence_roots"][0]).free
    budget = definition["budget"]
    reason = (
        "evidence_budget"
        if used + reservation_bytes > budget["evidence_bytes"]
        else (
            "disk_headroom"
            if free - reservation_bytes < budget["minimum_free_bytes"]
            else None
        )
    )
    return {
        "passed": reason is None,
        "reason": reason,
        "used_bytes": used,
        "free_bytes": free,
        "reservation_bytes": reservation_bytes,
    }


def _reservation(case: dict[str, Any]) -> int:
    pixel_bytes = sum(
        {"rgb": 3, "depth": 4, "seg": 4, "normals": 12}[m] for m in case["modalities"]
    )
    samples = 100 if case["temporal_mode"] == "moving" else case["view_count"]
    return max(
        256 * 1024**2,
        case["width"]
        * case["height"]
        * pixel_bytes
        * (
            4 * case["num_envs"] * case["cameras_per_env"]
            + samples * min(3, case["num_envs"]) * case["cameras_per_env"]
        ),
    )


def _capacity_reason(rows: list[dict[str, Any]]) -> str | None:
    for row in rows:
        if row.get("failure_reason_code") in ("oom", "renderer_capacity_limit"):
            return row["failure_reason_code"]
        if row.get("renderer_validation", {}).get("status") == "failed" and row[
            "renderer_validation"
        ].get("events"):
            return "renderer_capacity_limit"
    return None


def run_campaign(
    root: Path,
    *,
    phase: str,
    case_ids: tuple[str, ...] = (),
    retry_failed: bool = False,
) -> None:
    """Execute pending jobs through the existing formal runner and persist progress.

    Args:
        root: Initialized campaign directory.
        phase: Smoke, three-repeat coverage, or independent ten-repeat publication.
        case_ids: Optional explicit subset; unselected jobs retain their state.
        retry_failed: Reserve a new attempt for explicitly selected failures.
    """
    from .formal_runner import main as formal_main

    root = Path(root).absolute()
    definition, state = _load(root)
    if (
        _source_files() != definition["source_files"]
        or git_revision(ROOT) != definition["source_commit"]
    ):
        raise ValueError("Campaign source changed after freezing")
    if git_revision(Path(definition["isaaclab_root"])) != definition["isaaclab_commit"]:
        raise ValueError("External EA revision changed")
    for owner in (ROOT, Path(definition["isaaclab_root"])):
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=owner,
            text=True,
        ):
            raise ValueError("Frozen source checkout is dirty")
    for backend, interpreter in definition["interpreters"].items():
        if _runtime_evidence(Path(interpreter)) != definition["runtime"][backend]:
            raise ValueError(f"Frozen worker runtime changed: {backend}")
    validate_approval(Path(definition["scene_root"]))
    cases = {c["case_id"]: c["case"] for c in definition["cases"]}
    if set(case_ids) - cases.keys():
        raise ValueError("Unknown campaign case selection")
    if retry_failed and not case_ids:
        raise ValueError("Retries require an explicit case selection")
    for job in state["jobs"]:
        if (
            retry_failed
            and job["phase"] == phase
            and job["case_id"] in case_ids
            and job["status"] in ("failed", "interrupted")
        ):
            job["status"] = "not_run"
        if (
            job["phase"] != phase
            or job["status"] != "not_run"
            or case_ids
            and job["case_id"] not in case_ids
        ):
            continue
        case = cases[job["case_id"]]
        if case["deformation"]:
            job.update(status="blocked", reason="unsupported_frozen_deformation_input")
            write_json(root / "campaign-state.json", state)
            continue
        if case["experiment_id"] in ("R-04", "R-05"):
            for previous in state["jobs"]:
                other = cases[previous["case_id"]]
                if (
                    previous["phase"] != phase
                    or other["experiment_id"] != case["experiment_id"]
                    or other["num_envs"] >= case["num_envs"]
                ):
                    continue
                for attempt in previous["attempts"]:
                    for backend in _BACKENDS:
                        reason = attempt.get("capacity", {}).get(backend)
                        if reason:
                            job["blocked_backends"][
                                backend
                            ] = f"{reason}:N={other['num_envs']}"
        backends = [b for b in _BACKENDS if b not in job["blocked_backends"]]
        gate = storage_gate(definition, reservation_bytes=_reservation(case))
        if not gate["passed"]:
            job.update(reason=gate["reason"], storage_gate=gate)
            write_json(root / "campaign-state.json", state)
            break
        if not backends:
            job.update(status="blocked", reason="observed_capacity_stop")
            write_json(root / "campaign-state.json", state)
            continue
        reservation = job["repeats"] * len(backends)
        if (
            state["reserved_worker_runs"] + reservation
            > definition["budget"]["maximum_worker_runs"]
        ):
            job.update(reason="worker_attempt_budget")
            write_json(root / "campaign-state.json", state)
            break
        output = root / phase / job["case_id"] / f"attempt-{len(job['attempts']):03d}"
        output.mkdir(parents=True, exist_ok=False)
        argv = [
            "--scene-root",
            definition["scene_root"],
            "--case-id",
            job["case_id"],
            "--backend",
            "both" if len(backends) == 2 else backends[0],
            "--gpu-id",
            str(definition["gpu_id"]),
            "--embodichain-python",
            definition["interpreters"]["embodichain"],
            "--isaaclab-root",
            definition["isaaclab_root"],
            "--isaaclab-python",
            definition["interpreters"]["isaaclab"],
            "--repeats",
            str(job["repeats"]),
            "--timeout-s",
            str(definition["timeout_s"]),
            "--output",
            str(output),
        ]
        if phase == "smoke":
            argv.append("--smoke")
        elif phase == "publication":
            argv.append("--publication")
        attempt = {
            "attempt_id": len(job["attempts"]),
            "argv": argv,
            "output": output.relative_to(root).as_posix(),
            "status": "running",
            "experiments": [],
            "reserved_worker_runs": reservation,
        }
        state["reserved_worker_runs"] += reservation
        job["attempts"].append(attempt)
        job["status"] = "running"
        write_json(root / "campaign-state.json", state)
        print(
            f"{phase}: {job['case_id']} ({','.join(backends)}, repeats={job['repeats']})",
            flush=True,
        )
        try:
            with (
                (output / "launcher.log").open("w") as log,
                redirect_stdout(log),
                redirect_stderr(log),
            ):
                formal_main(argv)
        except KeyboardInterrupt:
            attempt["status"] = "interrupted"
            job["status"] = "interrupted"
            write_json(root / "campaign-state.json", state)
            raise
        except (Exception, SystemExit) as error:
            attempt["error"] = str(error)
        roots = [p.parent for p in output.glob("*/index.json")]
        rows = []
        for invocation in roots:
            for entry in read_json_object(invocation / "index.json")["experiments"]:
                experiment = invocation / entry["path"]
                attempt["experiments"].append(experiment.relative_to(root).as_posix())
                selected = selected_case_rows(experiment, job["case_id"])
                rows += selected
        attempt["capacity"] = {
            b: reason
            for b in _BACKENDS
            if (reason := _capacity_reason([r for r in rows if r["backend"] == b]))
        }
        attempt["status_counts"] = dict(Counter(r["status"] for r in rows))
        attempt["status"] = (
            "completed"
            if rows and all(r["status"] == "completed" for r in rows)
            else "failed"
        )
        job["status"] = attempt["status"]
        write_json(root / "campaign-state.json", state)
        rebuild_campaign_report(root)


def _campaign_figures(
    root: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Plot verified per-invocation groups with separate phase/attempt scopes."""
    from ._publication_figures import write_chart
    import csv

    groups: dict[str, list[dict[str, Any]]] = {}
    points = []
    for record in records:
        group = record["group"]
        config = group["config"]
        experiment = group["experiment_id"]
        if experiment not in ("R-03", "R-04", "R-05", "R-09", "modality-ablation"):
            continue
        if experiment == "R-03":
            x = config["width"] * config["height"]
            axis = "Output pixels"
        elif experiment in ("R-04", "R-05"):
            x = config["num_envs"]
            axis = "Environments"
        elif experiment == "R-09":
            x = ["device_ready", "host_ready", "duplicate_d2h", "host_copy"].index(
                config["delivery"]
            )
            axis = "Delivery treatment"
        else:
            x = [
                ("rgb",),
                ("rgb", "depth"),
                ("rgb", "normals"),
                ("rgb", "depth", "seg"),
            ].index(tuple(config["modalities"]))
            axis = "Requested modalities"
        scope = {
            key: config.get(key)
            for key in (
                "scene_id",
                "profile",
                "temporal_mode",
                "diagnostic",
                "cameras_per_env",
            )
        }
        if experiment != "R-09":
            scope["delivery"] = config.get("delivery")
        if experiment != "modality-ablation":
            scope["modalities"] = config.get("modalities")
        scope.update(
            phase=record["phase"],
            attempt=record["attempt"],
            hardware_id=group["hardware_id"],
            experiment=experiment,
            profile_sha256=record["profile_sha256"],
        )
        for metric in (
            "camera_frames_per_s",
            "latency_p50_s",
            "latency_p95_s",
            "gpu_process_steady_bytes",
            "gpu_process_peak_bytes",
        ):
            value = group["metrics"][metric]
            chart_scope = {
                **scope,
                "metric": metric,
                "unit": value["unit"],
                "x_label": axis,
            }
            point = {
                "x": x,
                "y": value["median"],
                "mad": value.get("uncertainty", {}).get("value"),
                "backend": group["backend"],
                "case_id": group["case_id"],
                "source_runs": value.get("source_runs", group["source_runs"]),
                "config_sha256": group["config_sha256"],
                "scope": chart_scope,
                "qualified_runs": group["qualified_runs"],
                "experiment": record["experiment"],
                "source_artifact_sha256": record["source_artifact_sha256"],
                "valid_runs": value["valid_runs"],
            }
            groups.setdefault(json.dumps(chart_scope, sort_keys=True), []).append(point)
            points.append(point)
    figure_dir = root / "figures"
    figure_dir.mkdir(exist_ok=True)
    figures = []
    for encoded, members in sorted(groups.items()):
        scope = json.loads(encoded)
        series = []
        for backend in _BACKENDS:
            values = []
            for point in sorted(
                (p for p in members if p["backend"] == backend), key=lambda p: p["x"]
            ):
                spread = point["mad"] or 0
                values.append(
                    {
                        "x": point["x"],
                        "y": point["y"],
                        "low": (
                            max(0, point["y"] - spread)
                            if point["y"] is not None
                            else None
                        ),
                        "high": point["y"] + spread if point["y"] is not None else None,
                    }
                )
            if values:
                series.append({"label": backend, "points": values})
        stem = (
            figure_dir
            / f"{scope['phase']}-{scope['experiment']}-{stable_hash(scope)[:12]}"
        )
        ticks = (
            {
                float(i): label
                for i, label in enumerate(
                    ("GPU ready", "CPU ready", "Duplicate D2H", "CPU copy")
                )
            }
            if scope["experiment"] == "R-09"
            else (
                {
                    float(i): label
                    for i, label in enumerate(("RGB", "RGB+D", "RGB+N", "RGB+D+S"))
                }
                if scope["experiment"] == "modality-ablation"
                else None
            )
        )
        svg, pdf = write_chart(
            stem,
            title=f"Descriptive {scope['experiment']} / {scope['phase']} / {scope['profile']}",
            x_label=scope["x_label"],
            y_label=f"{scope['metric']} ({scope['unit']})",
            series=series,
            provenance={
                "scope": scope,
                "points": members,
                "claim": "descriptive measurements; quality eligibility is separate",
            },
            log_x=scope["experiment"] in ("R-04", "R-05"),
            x_ticks=ticks,
        )
        figures.append(
            {
                "svg": svg.relative_to(root).as_posix(),
                "pdf": pdf.relative_to(root).as_posix(),
                "scope": scope,
                "source_runs": sorted({r for p in members for r in p["source_runs"]}),
            }
        )
    with (root / "campaign-figure-data.csv").open("w", newline="") as stream:
        fields = list(points[0]) if points else ["x", "y", "source_runs", "scope"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in point.items()
                }
            )
    return figures


def rebuild_campaign_report(root: Path) -> Path:
    """Revalidate source reports and export full coverage without pooling phases.

    Args:
        root: Campaign directory; this function imports only standard-library code.

    Returns:
        Campaign Markdown overview with linked verified experiment reports.
    """
    from .publication_report import rebuild_report

    root = Path(root).absolute()
    definition, state = _load(root)
    reports = []
    coverage = []
    chart_records = []
    for job in state["jobs"]:
        source_runs = []
        counts = Counter()
        qualified = 0
        for attempt in job["attempts"]:
            for relative in attempt["experiments"]:
                experiment = root / relative
                rebuild_report(experiment)
                summary = read_json_object(experiment / "summary.json")
                for group in summary["groups"]:
                    if group["case_id"] == job["case_id"]:
                        source_runs.extend(
                            f"{relative}#{run}" for run in group["source_runs"]
                        )
                        qualified += group.get("qualified_runs", 0)
                        candidates = [
                            r
                            for r in json.loads((experiment / "runs.json").read_text())
                            if r["backend"] == group["backend"]
                            and r["case_id"] == job["case_id"]
                            and r.get("profile_sha256")
                        ]
                        chart_records.append(
                            {
                                "group": group,
                                "experiment": relative,
                                "source_artifact_sha256": hashlib.sha256(
                                    (experiment / "runs.json").read_bytes()
                                ).hexdigest(),
                                "phase": job["phase"],
                                "attempt": attempt["attempt_id"],
                                "profile_sha256": (
                                    candidates[0]["profile_sha256"]
                                    if candidates
                                    else None
                                ),
                            }
                        )
                rows = selected_case_rows(experiment, job["case_id"])
                counts.update(r["status"] for r in rows)
                reports.append(
                    {
                        "phase": job["phase"],
                        "case_id": job["case_id"],
                        "report": str(Path(relative) / "report.md"),
                        "source_runs": [row["run_id"] for row in rows],
                    }
                )
        coverage.append(
            {
                "phase": job["phase"],
                "case_id": job["case_id"],
                "status": job["status"],
                "repeats": job["repeats"],
                "status_counts": dict(counts),
                "qualified_runs": qualified,
                "reason": job.get("reason"),
                "blocked_backends": job["blocked_backends"],
                "source_runs": sorted(set(source_runs)),
            }
        )
    summary = {
        "campaign_sha256": stable_hash(definition),
        "runtime": definition["runtime"],
        "coverage": coverage,
        "reports": reports,
        "storage": storage_gate(definition),
        "source_commit": definition["source_commit"],
        "claims": "Only independently verified matching quality pairs in linked reports may support equal-quality ratios; phases and historical hardware are separate.",
    }
    summary["figures"] = _campaign_figures(root, chart_records)
    summary["proofs"] = state.get("proofs", [])
    write_json(root / "campaign-summary.json", summary)
    import csv

    with (root / "campaign-coverage.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(coverage[0]) if coverage else []
        )
        writer.writeheader()
        for row in coverage:
            writer.writerow(
                {
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
                    for k, v in row.items()
                }
            )
    table = [
        {
            "Phase": r["phase"],
            "Case": r["case_id"],
            "Status": r["status"],
            "Planned repeats": r["repeats"],
            "Trial statuses": json.dumps(r["status_counts"]),
            "Qualified": r["qualified_runs"],
            "Reason": r["reason"] or json.dumps(r["blocked_backends"]),
        }
        for r in coverage
    ]
    lines = [
        "# RTX 5090 × Isaac Lab 3.0.0-EA campaign",
        "",
        f"Source: `{definition['source_commit']}`. EA: `{definition['isaaclab_commit']}` / Isaac Sim 6.1.0.0.",
        "",
        "82 declared cells; 16 prespecified publication cells. Coverage, smoke and independent ten-repeat publication populations remain separate.",
        "",
        summary["claims"],
        "",
        render_markdown_table(table),
        "",
        "## Experiment evidence",
        "",
    ]
    lines += [
        f"- {r['phase']} / {r['case_id']}: [report]({r['report']})" for r in reports
    ]
    lines += ["", "## Descriptive curves", ""]
    lines += [
        f"![{f['scope']['phase']} {f['scope']['experiment']} {f['scope']['metric']}]({f['svg']})"
        for f in summary["figures"]
    ]
    path = root / "campaign-report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None) -> None:
    """Inspect, initialize, execute or rebuild the frozen comparison campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument("--embodichain-python", type=Path)
    parser.add_argument("--isaaclab-root", type=Path)
    parser.add_argument("--isaaclab-python", type=Path)
    parser.add_argument("--phase", choices=("smoke", "coverage", "publication"))
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--evidence-root", type=Path, action="append", default=[])
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(campaign_cases(), indent=2))
        return
    if args.initialize:
        if any(
            getattr(args, key) is None
            for key in (
                "scene_root",
                "embodichain_python",
                "isaaclab_root",
                "isaaclab_python",
            )
        ):
            parser.error("Initialization requires scene, source and both runtime paths")
        initialize_campaign(
            args.root,
            scene_root=args.scene_root,
            embodichain_python=args.embodichain_python,
            isaaclab_root=args.isaaclab_root,
            isaaclab_python=args.isaaclab_python,
            evidence_roots=tuple(args.evidence_root),
        )
    if args.phase:
        run_campaign(
            args.root,
            phase=args.phase,
            case_ids=tuple(args.case_id),
            retry_failed=args.retry_failed,
        )
    print(rebuild_campaign_report(args.root))


if __name__ == "__main__":
    main()
