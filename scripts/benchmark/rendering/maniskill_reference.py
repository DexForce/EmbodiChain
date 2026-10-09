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

"""Run two isolated, unqualified ManiSkill raster reference points.

The approved Franka scene is replicated at N=1/16, C=1, 128x128 RGB. This
separate experiment does not extend the two-backend formal quality comparison.
List, dry-run and report rebuild are standard-library-only operations.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

from scripts.benchmark.core.artifacts import (
    ArtifactStore,
    create_experiment_directory,
    read_json,
    stable_hash,
    write_json,
)
from scripts.benchmark.core.contracts import Budget, ExperimentDefinition
from scripts.benchmark.core.execution import run_experiment
from scripts.benchmark.core.records import RunSpec
from scripts.benchmark.reporting.aggregation import aggregate_runs
from scripts.benchmark.reporting.report import write_technical_report
from .backends.scene_maniskill import raster_profile
from .formal import metric_definitions, validate_approval, view_fingerprint
from .formal_runner import _runtime_environment
from .scenes import load_scene, scene_fingerprint

__all__ = ["main", "rebuild_report", "reference_cases"]

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "maniskill-raster-reference"


def reference_cases(
    shader: str = "default", delivery: str = "device_ready"
) -> list[dict[str, Any]]:
    """Return the two explicit workload points, with portable case identities.

    Args:
        shader: Pinned default/minimal raster profile, separate from ray tracing.
        delivery: Completed GPU observation or one blocking host delivery.

    Returns:
        Fixed N=1/16, C=1, 128x128 RGB static scene cases.
    """
    profile = raster_profile(shader)
    if delivery not in {"device_ready", "host_ready"}:
        raise ValueError("Reference delivery must be device_ready or host_ready")
    return [
        {
            "experiment_id": EXPERIMENT_ID,
            "case_id": f"ms-ref-franka-n{num_envs}c1-128-rgb-{shader}-{delivery}",
            "scene_id": "franka_workstation",
            "num_envs": num_envs,
            "cameras_per_env": 1,
            "width": 128,
            "height": 128,
            "modalities": ["rgb"],
            "profile": profile["id"],
            "shader": shader,
            "temporal_mode": "static",
            "view_count": 1,
            "delivery": delivery,
        }
        for num_envs in (1, 16)
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/maniskill-reference")
    )
    parser.add_argument(
        "--python", type=Path, default=ROOT / ".venv-maniskill-reference/bin/python"
    )
    parser.add_argument(
        "--physx-gpu-library",
        type=Path,
        default=ROOT / ".venv-maniskill-reference/physx-runtime/libPhysXGpu_64.so",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--shader", choices=("default", "minimal"), default="default")
    parser.add_argument(
        "--delivery", choices=("device_ready", "host_ready"), default="device_ready"
    )
    parser.add_argument(
        "--num-envs", type=int, choices=(1, 16), nargs="+", default=[1, 16]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--measured-frames", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=900)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One repeat, two warmups and four measured frames; preserve N/C/resolution.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", type=Path)
    return parser


def _selection(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "definition_version": "1.0",
        "quality_status": "not_qualified",
        "equal_quality_comparison": False,
        "publication_claim": False,
        "repeats": 1 if args.smoke else args.repeats,
        "warmup_frames": 2 if args.smoke else args.warmup_frames,
        "measured_frames": 4 if args.smoke else args.measured_frames,
        "repeat_population": "independent fresh worker processes",
        "cases": [
            {
                "case": case,
                "selected": case["num_envs"] in args.num_envs,
                "selection_reason": (
                    "selected"
                    if case["num_envs"] in args.num_envs
                    else "num_envs_filter"
                ),
            }
            for case in reference_cases(args.shader, args.delivery)
        ],
    }


def _finalize(root: Path, selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_json(root / "runs.json")
    cases = {item["case"]["case_id"]: item for item in selection["cases"]}
    present = {(row["case_id"], row["repeat"]) for row in rows}
    store = ArtifactStore(root)
    for case_id, item in cases.items():
        for repeat in range(selection["repeats"]):
            if (case_id, repeat) in present:
                continue
            row = {
                "schema_version": 1,
                "experiment_id": EXPERIMENT_ID,
                "run_id": f"r{repeat:02d}_{case_id}",
                "case_id": case_id,
                "backend": "maniskill",
                "repeat": repeat,
                "status": "not_run",
                "quality_status": "not_qualified",
                "hardware_id": None,
                "config": item["case"],
                "config_sha256": stable_hash(item["case"]),
                "selection_reason": item["selection_reason"],
                "failure_reason_code": "not_selected",
                "metrics": {},
            }
            rows.append(row)
            store.append_raw({**row, "record_id": row["run_id"]})
    for row in rows:
        item = cases[row["case_id"]]
        row.setdefault("config", item["case"])
        row.setdefault("config_sha256", stable_hash(item["case"]))
        row.setdefault("quality_status", "not_qualified")
        row["selection_reason"] = item["selection_reason"]
    write_json(root / "runs.json", rows)
    store.write_quality(
        {
            "schema_version": 1,
            "runs": [
                {
                    "run_id": row["run_id"],
                    "backend": "maniskill",
                    "status": row["status"],
                    "quality_status": "not_qualified",
                    "reason": "separate_raster_profile_without_independent_calibration",
                    "evidence": row.get("evidence", []),
                }
                for row in rows
            ],
        }
    )
    return rows


def rebuild_report(root: Path) -> Path:
    """Rebuild a separate truthful reference report from frozen standard artifacts.

    Args:
        root: Experiment directory containing ``runs.json`` and its selection.

    Returns:
        Markdown report; CSV and JSON retain run IDs, hashes and repeat uncertainty.
    """
    root = Path(root)
    rows = read_json(root / "runs.json")
    if not isinstance(rows, list) or any(
        row.get("backend") != "maniskill" for row in rows
    ):
        raise ValueError("Expected only ManiSkill reference run records")
    groups = []
    for hardware_id in sorted({row.get("hardware_id") for row in rows}, key=str):
        subset = [row for row in rows if row.get("hardware_id") == hardware_id]
        part = aggregate_runs(subset, metric_definitions())
        for group in part["groups"]:
            for metric in group["metrics"].values():
                if metric["valid_runs"] < 2:
                    metric["uncertainty"] = None
                    metric["uncertainty_reason"] = "fewer_than_two_independent_repeats"
        groups.extend({**group, "hardware_id": hardware_id} for group in part["groups"])
    summary = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "groups": groups,
        "scheduled": len(rows),
        "completed_runs": sum(row["status"] == "completed" for row in rows),
        "not_run": sum(row["status"] == "not_run" for row in rows),
        "quality_status": "not_qualified",
        "equal_quality_comparison": False,
    }
    store = ArtifactStore(root)
    store.write_metrics(summary)
    write_json(root / "summary.json", summary)
    columns = (
        "run_id",
        "case_id",
        "config_sha256",
        "hardware_id",
        "repeat",
        "status",
        "quality_status",
        "num_envs",
        "profile",
        "delivery",
        "batch_steps_per_s",
        "camera_frames_per_s",
        "latency_p50_s",
        "gpu_process_peak_bytes",
        "reason",
    )
    metric_rows = [
        {
            **{key: row.get(key) for key in columns},
            **{
                key: row.get("config", {}).get(key)
                for key in ("num_envs", "profile", "delivery")
            },
            **{
                key: row.get("metrics", {}).get(key)
                for key in (
                    "batch_steps_per_s",
                    "camera_frames_per_s",
                    "latency_p50_s",
                    "gpu_process_peak_bytes",
                )
            },
            "reason": row.get("failure_reason_code")
            or row.get("error")
            or "raster_quality_not_qualified",
        }
        for row in rows
    ]
    with (root / "reference_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(metric_rows)
    report = write_technical_report(
        root / "report.md",
        title="ManiSkill raster reference points — unqualified",
        summary=summary,
        metric_rows=metric_rows,
        validation_rows=[
            {
                "run_id": row["run_id"],
                "status": row["status"],
                "quality": row.get("quality_status", "missing"),
                "reason": row.get("error", "No independent quality qualification"),
            }
            for row in rows
        ],
        notes=[
            "These two raster reference points have different rendering/lighting/colour semantics from the formal ray-tracing profiles. No equal-quality speedup or universal leaderboard is computed.",
            "N=1/16, one external-front camera, 128x128 RGB, static approved Franka geometry. Camera frames/s = batch steps/s × N; modalities do not multiply exposures.",
            "Independent process-repeat median/MAD and source run IDs are in summary.json; frame samples are not independent repeats. Fewer than two repeats have no uncertainty estimate. Hardware groups remain separate and missing UUIDs remain explicit.",
            "Selected-device completion includes Vulkan-to-CUDA semaphore waits and RGB packing. Host-ready adds one blocking RGB batch D2H. Physics GPU-buffer warm-start occurs only during setup; capture does not step physics.",
            "Native gamma-2.2 RGB is preserved, not labelled exact IEC sRGB. Native GPU kernel time and internal-resolution equivalence remain unverified.",
            "Freshness checks and saved all-environment RGB/pose/intrinsics evidence are outside the primary timing window. Short smoke runs establish no publication performance result.",
        ],
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run or inspect the bounded raster reference CLI.

    Args:
        argv: Optional command-line arguments, excluding the module name.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.report_only:
        print(rebuild_report(args.report_only))
        return
    if (
        args.repeats < 1
        or args.warmup_frames < 0
        or args.measured_frames < 1
        or args.gpu_id < 0
    ):
        parser.error(
            "repeats/measured frames must be positive; warmup/GPU index nonnegative"
        )
    budget = Budget(
        max_runs=len(set(args.num_envs)) * (1 if args.smoke else args.repeats),
        wall_time_s=args.timeout_s,
    )
    selection = _selection(args)
    if args.list or args.dry_run:
        print(json.dumps({**selection, "filesystem_mutations": False}, indent=2))
        return
    if args.scene_root is None:
        parser.error("--scene-root is required for execution")
    approval = validate_approval(args.scene_root, args.approval)
    scene_path = args.scene_root / "franka_workstation/scene.json"
    scene = load_scene(scene_path)
    definition = ExperimentDefinition(
        experiment_id=EXPERIMENT_ID,
        definition_version="1.0",
        parameter_matrix={"case": tuple(item["case"] for item in selection["cases"])},
        budget=Budget(
            max_runs=budget.max_runs,
            max_attempts=budget.max_runs,
            wall_time_s=args.timeout_s * budget.max_runs,
        ),
        quality_protocol={
            "status": "not_qualified",
            "reason": "separate_raster_profile_without_independent_calibration",
            "equal_quality_comparison": False,
        },
        comparison_invariants=(
            "config_sha256",
            "hardware_id",
            "scene_sha256",
            "view_sha256",
            "profile_sha256",
        ),
        metric_definitions=metric_definitions(),
    )
    root = create_experiment_directory(
        args.output,
        experiment_id=EXPERIMENT_ID,
        config=selection,
        definition=definition,
        assets_manifest={
            "approval": approval,
            "scene_sha256": scene_fingerprint(scene),
            "stock_asset_downloads": False,
        },
    )
    frozen_scene = root / "scenes/franka_workstation/scene.json"
    frozen_scene.parent.mkdir(parents=True)
    shutil.copy2(scene_path, frozen_scene)
    for obj in scene["objects"]:
        geometry = obj["geometry"]
        if geometry["kind"] == "mesh":
            target = frozen_scene.parent / geometry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(scene_path.parent / geometry["path"], target)
    write_json(root / "approval.json", approval)
    write_json(root / "selection.json", selection)
    runs = []
    configs = {}
    for item in selection["cases"]:
        case = item["case"]
        profile = raster_profile(case["shader"])
        config = {
            "case": case,
            "scene_path": str(frozen_scene),
            "profile": profile,
            "config_sha256": stable_hash(case),
            "scene_sha256": scene_fingerprint(scene),
            "view_sha256": view_fingerprint(SimpleNamespace(**case), scene),
            "profile_sha256": stable_hash(profile),
            "approved_scene_receipt": approval,
            "warmup_frames": selection["warmup_frames"],
            "measured_frames": selection["measured_frames"],
            "physx_gpu_library": str(args.physx_gpu_library.absolute()),
        }
        configs[case["case_id"]] = write_json(
            root / "configs" / f"{case['case_id']}.json", config
        )
    environment = _runtime_environment(ROOT, args.python, None)
    for repeat in range(selection["repeats"]):
        items = (
            selection["cases"]
            if repeat % 2 == 0
            else list(reversed(selection["cases"]))
        )
        for item in items:
            if not item["selected"]:
                continue
            case_id = item["case"]["case_id"]
            output = root / f"r{repeat:02d}_{case_id}"
            runs.append(
                RunSpec(
                    backend="maniskill",
                    repeat=repeat,
                    case_id=case_id,
                    output=output,
                    timeout_s=args.timeout_s,
                    env=environment,
                    command=(
                        str(args.python.absolute()),
                        "-m",
                        "scripts.benchmark.rendering.maniskill_reference_worker",
                        "--config",
                        str(configs[case_id]),
                        "--output",
                        str(output),
                        "--gpu-id",
                        str(args.gpu_id),
                        "--repeat",
                        str(repeat),
                    ),
                )
            )
    store = ArtifactStore(root)
    for path in [
        *sorted((root / "scenes").rglob("*")),
        *configs.values(),
        root / "approval.json",
        root / "selection.json",
    ]:
        if path.is_file():
            store.register_artifact(path, kind="frozen_reference_input")
    print(root, flush=True)
    try:
        run_experiment(
            root, runs, experiment_id=EXPERIMENT_ID, budget=definition.budget
        )
    finally:
        if (root / "runs.json").exists():
            rows = _finalize(root, selection)
            for row in rows:
                for name in row.get("evidence", []):
                    path = root / row["run_id"] / name
                    if path.is_file():
                        store.register_artifact(path, kind="reference_worker_evidence")
            print(rebuild_report(root), flush=True)


if __name__ == "__main__":
    main()
