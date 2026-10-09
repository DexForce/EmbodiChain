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

"""Plan and run the approved v2 rendering matrix in isolated serial processes.

Run: python -m scripts.benchmark.rendering.formal_runner --list
Inspection and dry runs use only the standard library and never prepare assets
or create output directories. Execution requires a matching scene approval.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from scripts.benchmark.core.artifacts import (
    ArtifactStore,
    create_experiment_directory,
    read_json,
    read_json_object,
    stable_hash,
    write_json,
)
from scripts.benchmark.core.contracts import Budget, ExperimentDefinition
from scripts.benchmark.core.execution import repeat_schedule, run_experiment
from scripts.benchmark.core.records import RunSpec, validate_run_result

from .formal import (
    DEFINITION_VERSION,
    EXPERIMENTS,
    RenderCase,
    experiment_cases,
    metric_definitions,
    render_profile,
    validate_approval,
    view_fingerprint,
)
from .quality_protocol import protocol_for_modalities, protocol_requirements

__all__ = ["main"]

_BACKENDS = ("embodichain", "isaaclab")
_SCENES = ("calibration", "workshop", "franka_workstation")
_QUALITY_EXPERIMENTS = {"R-01", "R-02"}
_INVARIANTS = (
    "config_sha256",
    "hardware_id",
    "scene_sha256",
    "view_sha256",
    "profile_sha256",
    "config.profile",
    "config.temporal_mode",
    "config.delivery",
    "metrics.boundary",
    "metrics.completion",
)


def _experiment(value: str) -> str:
    """Accept the report's R01 shorthand while recording canonical R-01 IDs."""
    upper = value.upper()
    if upper.startswith("R") and len(upper) == 3:
        return "R-" + upper[1:]
    return upper if upper.startswith("R-") else value


def _parser() -> argparse.ArgumentParser:
    """Build CLI flags without touching runtime environments or assets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment", type=_experiment, choices=("all", *EXPERIMENTS), default="all"
    )
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/benchmarks/rendering-publication")
    )
    parser.add_argument(
        "--backend", choices=("both", "E", "I", *_BACKENDS), default="both"
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--embodichain-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--isaaclab-root", type=Path)
    parser.add_argument("--isaaclab-python", type=Path)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Independent performance repeats; R-01/R-02 quality captures use one per backend.",
    )
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--measured-frames", type=int, default=300)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One repeat, two warmups, four timed frames; preserve all case semantics.",
    )
    parser.add_argument(
        "--publication",
        action="store_true",
        help="At least ten independent repeats for selected performance cells.",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--num-envs", type=int, nargs="+", action="extend")
    parser.add_argument(
        "--max-cases",
        type=int,
        help="Explicit global prefix limit after other filters; exclusions are recorded.",
    )
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--show-excluded",
        action="store_true",
        help="Include excluded cases when listing the matrix.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Use structured JSON for --list; --dry-run always emits the full JSON plan.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved selection without reads of scene assets or any writes.",
    )
    parser.add_argument("--report-only", type=Path)
    return parser


def _print_list(
    selection: dict[str, Any], *, show_excluded: bool, as_json: bool
) -> None:
    """List selected cases concisely, retaining an explicit full-universe count."""
    cases = [
        record for record in selection["cases"] if record["selected"] or show_excluded
    ]
    if as_json:
        print(
            json.dumps(
                {
                    **selection,
                    "cases": cases,
                    "inspection": "list",
                    "listing_scope": "all" if show_excluded else "selected",
                    "filesystem_mutations": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    print(
        f"Formal rendering v{DEFINITION_VERSION}: {selection['selected_case_count']} selected / {selection['full_matrix_case_count']} declared cases; {selection['excluded_case_count']} excluded"
    )
    print(
        f"Repeats: performance={selection['repeat_policy']['performance']}, quality={selection['repeat_policy']['quality_capture']}; backends={','.join(selection['backends'])}; warmup={selection['warmup_frames']}, measured={selection['measured_frames']}"
    )
    for record in cases:
        case = record["case"]
        tags = [name for name in ("diagnostic", "deformation") if case[name]]
        status = (
            "selected"
            if record["selected"]
            else "excluded:" + ",".join(record["selection_reasons"])
        )
        print(
            f"{record['case_id']}  {case['width']}x{case['height']} N={case['num_envs']} C={case['cameras_per_env']} {'+'.join(case['modalities'])} {case['profile']} {case['temporal_mode']} {case['delivery']} repeats={record['repeats']} {status}"
            + (" " + ",".join(tags) if tags else "")
        )


def _case_policy(case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze only quality evidence supported by the actual requested outputs."""
    moving = case["temporal_mode"] == "moving"
    sequence = moving and set(case["modalities"]) == {"rgb", "depth", "seg"}
    policy = protocol_for_modalities(
        tuple(case["modalities"]), temporal_dt_s=1 / 30 if sequence else None
    )
    requirements = protocol_requirements(
        policy["id"], "sequence" if sequence else "frame"
    )
    requirements["temporal_quality"] = (
        "unassessed" if moving and not sequence else "assessed_by_declared_policy"
    )
    return policy, requirements


def _selection(args: argparse.Namespace) -> dict[str, Any]:
    """Freeze the entire case universe with explicit filter and repeat policies."""
    backends = (
        _BACKENDS
        if args.backend == "both"
        else ({"E": "embodichain", "I": "isaaclab"}.get(args.backend, args.backend),)
    )
    performance_repeats = (
        1 if args.smoke else max(args.repeats, 10 if args.publication else 1)
    )
    records = []
    selected_count = 0
    for experiment_id in EXPERIMENTS:
        for case in experiment_cases(experiment_id):
            reasons = []
            if args.experiment not in ("all", experiment_id):
                reasons.append("experiment_filter")
            if args.case_id and case.case_id not in args.case_id:
                reasons.append("case_id_filter")
            if args.num_envs is not None and case.num_envs not in args.num_envs:
                reasons.append("num_envs_filter")
            if (
                not reasons
                and args.max_cases is not None
                and selected_count >= args.max_cases
            ):
                reasons.append("max_cases_limit")
            selected_count += not reasons
            records.append(
                {
                    "experiment_id": experiment_id,
                    "case_id": case.case_id,
                    "case": case.to_dict(),
                    "config_sha256": stable_hash(case.to_dict()),
                    "quality_protocol": _case_policy(case.to_dict())[1],
                    "selected": not reasons,
                    "selection_reasons": reasons,
                    "repeats": (
                        1
                        if experiment_id in _QUALITY_EXPERIMENTS
                        else performance_repeats
                    ),
                }
            )
    unknown = set(args.case_id) - {record["case_id"] for record in records}
    if unknown:
        raise ValueError("Unknown case ID(s): " + ", ".join(sorted(unknown)))
    return {
        "schema_version": 1,
        "definition_version": DEFINITION_VERSION,
        "mode": (
            "smoke" if args.smoke else "publication" if args.publication else "pilot"
        ),
        "backends": list(backends),
        "gpu_id": args.gpu_id,
        "warmup_frames": 2 if args.smoke else args.warmup_frames,
        "measured_frames": 4 if args.smoke else args.measured_frames,
        "repeat_policy": {
            "performance": performance_repeats,
            "quality_capture": 1,
            "quality_experiments": sorted(_QUALITY_EXPERIMENTS),
            "population": "fresh process per backend/case/repeat",
            "publication_minimum": 10 if args.publication else None,
        },
        "filters": {
            "experiment": args.experiment,
            "case_id": args.case_id,
            "num_envs": args.num_envs,
            "max_cases": args.max_cases,
        },
        "full_matrix_case_count": len(records),
        "selected_case_count": selected_count,
        "excluded_case_count": len(records) - selected_count,
        "cases": records,
    }


def _runtime_environment(
    repo: Path, python: Path, isaaclab_root: Path | None
) -> dict[str, str]:
    """Pin source imports and retain the chosen virtualenv's lexical path."""
    environment = os.environ.copy()
    for key in list(environment):
        if (
            key == "CUDA_VISIBLE_DEVICES"
            or key.startswith("CONDA_")
            or key in {"_CE_CONDA", "_CE_M"}
        ):
            environment.pop(key)
    environment.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONUNBUFFERED="1",
        VIRTUAL_ENV=str(python.absolute().parent.parent),
    )
    imports = [str(repo)]
    if isaaclab_root is not None:
        imports += [
            str(path)
            for path in sorted((isaaclab_root / "source").iterdir())
            if path.is_dir()
        ]
        environment.update(ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES")
    environment["PYTHONPATH"] = os.pathsep.join(imports)
    return environment


def _worker_command(
    python: Path, config: Path, output: Path, backend: str, gpu_id: int, repeat: int
) -> tuple[str, ...]:
    """Build a shell-free invocation of the agreed formal worker protocol."""
    return (
        str(python.absolute()),
        "-m",
        "scripts.benchmark.rendering.formal_worker",
        "--config",
        str(config),
        "--output",
        str(output),
        "--backend",
        backend,
        "--gpu-id",
        str(gpu_id),
        "--repeat",
        str(repeat),
    )


def _file_hash(path: Path) -> str:
    """Hash frozen assets without loading large arrays into the launcher."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _copy_scenes(source: Path, destination: Path) -> dict[str, dict[str, Any]]:
    """Copy only the approved manifests and referenced canonical mesh buffers."""
    from .scenes import load_scene

    scenes = {}
    for name in _SCENES:
        source_path = source / name / "scene.json"
        scene = load_scene(source_path)
        target = destination / name
        target.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_path, target / "scene.json")
        for obj in scene["objects"]:
            if obj["geometry"]["kind"] == "mesh":
                relative = Path(obj["geometry"]["path"])
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path.parent / relative, target / relative)
        scenes[name] = load_scene(target / "scene.json")
    return scenes


def _prepare_replays(
    root: Path, selected: list[dict[str, Any]]
) -> tuple[dict[str, Path], dict[str, dict[str, str]], bool]:
    """Prepare each needed CPU replay once, before any rendering worker starts."""
    from .replay import prepare_replay

    needed = sorted(
        {
            record["case"]["scene_id"]
            for record in selected
            if record["case"]["temporal_mode"] == "moving"
            and not record["case"]["deformation"]
        }
    )
    paths: dict[str, Path] = {}
    errors: dict[str, dict[str, str]] = {}
    interrupted = False
    for name in needed:
        if interrupted:
            errors[name] = {
                "code": "not_run",
                "reason": "Launcher interrupted before replay preparation",
            }
            continue
        try:
            paths[name] = prepare_replay(
                root / "scenes" / name / "scene.json",
                root / "replays" / f"{name}.json",
                frames=100,
                fps=30,
            )
        except KeyboardInterrupt:
            interrupted = True
            errors[name] = {
                "code": "not_run",
                "reason": "Launcher interrupted during replay preparation",
            }
        except Exception as exc:
            errors[name] = {
                "code": (
                    "dependency_error"
                    if isinstance(exc, (ImportError, OSError))
                    else "replay_preparation_error"
                ),
                "reason": f"{type(exc).__name__}: {exc}",
            }
    return paths, errors, interrupted


def _device_evidence(gpu_id: int) -> dict[str, Any]:
    """Read selected-device identity without creating a CUDA context."""
    from .resources import ResourceSampler

    with ResourceSampler(gpu_id, interval_s=10.0) as sampler:
        pass
    result = sampler.summary()
    return {
        "method": "read_only_nvml",
        "gpu": result["gpu"],
        "missing_reasons": result["metric_reasons"],
    }


def _definition(
    experiment_id: str, cases: list[dict[str, Any]], runnable: int, timeout_s: float
) -> ExperimentDefinition:
    """Keep the full declared matrix separate from the selected attempt budget."""
    policies, assignments = {}, {}
    for record in cases:
        policy, requirements = _case_policy(record["case"])
        policies[policy["id"]] = policy
        assignments[record["case_id"]] = requirements
    return ExperimentDefinition(
        experiment_id=experiment_id,
        definition_version=DEFINITION_VERSION,
        parameter_matrix={"case": tuple(record["case"] for record in cases)},
        budget=Budget(
            max_runs=runnable or None,
            max_attempts=runnable or None,
            wall_time_s=timeout_s * runnable if runnable else None,
        ),
        quality_protocol={
            "policies": policies,
            "case_protocols": assignments,
            "qualification": "offline_measured_evidence_required",
            "status": "not_qualified",
        },
        comparison_invariants=_INVARIANTS,
        metric_definitions=metric_definitions(),
    )


def _freeze_experiment(
    invocation: Path,
    args: argparse.Namespace,
    selection: dict[str, Any],
    experiment_id: str,
    replays: dict[str, Path],
    preparation_errors: dict[str, dict[str, str]],
    approval: dict[str, Any],
) -> tuple[Path, list[RunSpec], Budget]:
    """Freeze case configs, full planned trials and isolated runnable processes."""
    cases = [
        record
        for record in selection["cases"]
        if record["experiment_id"] == experiment_id
    ]

    def blocked(record: dict[str, Any]) -> dict[str, str] | None:
        case = record["case"]
        return (
            preparation_errors.get(case["scene_id"])
            if case["temporal_mode"] == "moving" and not case["deformation"]
            else None
        )

    runnable = sum(
        record["repeats"] * len(selection["backends"])
        for record in cases
        if record["selected"] and not blocked(record)
    )
    definition = _definition(experiment_id, cases, runnable, args.timeout_s)
    root = create_experiment_directory(
        invocation / experiment_id,
        experiment_id=experiment_id,
        config={
            "cases": {record["case_id"]: record["case"] for record in cases},
            "selection": selection["filters"],
            "repeat_policy": selection["repeat_policy"],
            "warmup_frames": selection["warmup_frames"],
            "measured_frames": selection["measured_frames"],
            "gpu_id": args.gpu_id,
        },
        definition=definition,
        assets_manifest={
            "scene_approval": approval,
            "scene_manifests": {name: f"scenes/{name}/scene.json" for name in _SCENES},
        },
    )
    scenes = _copy_scenes(invocation / "scenes", root / "scenes")
    shutil.copy2(invocation / "approval.json", root / "approval.json")
    store = ArtifactStore(root)
    for path in sorted((root / "scenes").rglob("*")):
        if path.is_file():
            store.register_artifact(path, kind="approved_scene_input")
    store.register_artifact(root / "approval.json", kind="scene_approval")
    copied_replays = {}
    for name in sorted(
        {
            record["case"]["scene_id"]
            for record in cases
            if record["selected"]
            and record["case"]["temporal_mode"] == "moving"
            and not record["case"]["deformation"]
        }
    ):
        if name in replays:
            target = root / "replays" / f"{name}.json"
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(replays[name], target)
            copied_replays[name] = target
            store.register_artifact(target, kind="cpu_kinematic_replay")
    worker_configs = {}
    for record in cases:
        case = RenderCase(
            **{**record["case"], "modalities": tuple(record["case"]["modalities"])}
        )
        scene = scenes[case.scene_id]
        replay_path = (
            copied_replays.get(case.scene_id)
            if case.temporal_mode == "moving" and not case.deformation
            else None
        )
        replay = read_json_object(replay_path) if replay_path else None
        config = {
            "case": case.to_dict(),
            "scene_path": str(root / "scenes" / case.scene_id / "scene.json"),
            "replay_path": str(replay_path) if replay_path else None,
            "profile": render_profile(case.profile),
            "warmup_frames": selection["warmup_frames"],
            "measured_frames": selection["measured_frames"],
            "config_sha256": record["config_sha256"],
            "scene_sha256": stable_hash(scene),
            "view_sha256": (
                view_fingerprint(case, scene, replay)
                if case.temporal_mode == "static" or replay
                else None
            ),
            "profile_sha256": stable_hash(render_profile(case.profile)),
            "isaaclab_root": (
                str(args.isaaclab_root.absolute()) if args.isaaclab_root else None
            ),
            "approved_scene_receipt": approval,
            "quality_protocol": _case_policy(case.to_dict())[0],
            "quality_requirements": _case_policy(case.to_dict())[1],
        }
        worker_configs[case.case_id] = config
        if record["selected"]:
            path = write_json(
                root / "cases" / case.case_id / "worker-config.json", config
            )
            store.register_artifact(path, kind="frozen_worker_config")
    repo = Path(__file__).resolve().parents[3]
    interpreters = {
        "embodichain": args.embodichain_python,
        "isaaclab": args.isaaclab_python,
    }
    environments = {
        backend: _runtime_environment(
            repo,
            interpreters[backend],
            args.isaaclab_root if backend == "isaaclab" else None,
        )
        for backend in selection["backends"]
    }
    trials = []
    specs = []
    repeats = cases[0]["repeats"]
    for backend, repeat in repeat_schedule(_BACKENDS, repeats):
        for record in cases:
            reasons = list(record["selection_reasons"])
            if backend not in selection["backends"]:
                reasons.append("backend_filter")
            selected = not reasons
            preparation = blocked(record) if selected else None
            run_id = f"r{repeat:02d}_{backend}_{record['case_id']}"
            config = worker_configs[record["case_id"]]
            trial = {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "run_id": run_id,
                "run_dir": run_id,
                "backend": backend,
                "case_id": record["case_id"],
                "repeat": repeat,
                "attempt": 0,
                "status": "not_run",
                "quality_status": "not_qualified" if selected else "missing",
                "selected": selected,
                "runnable": selected and preparation is None,
                "selection_reason": "selected" if selected else ",".join(reasons),
                "failure_reason_code": (
                    preparation["code"] if preparation else "not_run"
                ),
                "failure_reason": (
                    preparation["reason"]
                    if preparation
                    else (
                        "Not started"
                        if selected
                        else "Explicitly excluded: " + ", ".join(reasons)
                    )
                ),
                "config": record["case"],
                "config_sha256": record["config_sha256"],
                "scene_sha256": config["scene_sha256"],
                "view_sha256": config["view_sha256"],
                "profile_sha256": config["profile_sha256"],
                "hardware_id": None,
                "metrics": {},
                "execution_attempted": False,
            }
            if trial["runnable"]:
                command = _worker_command(
                    interpreters[backend],
                    root / "cases" / record["case_id"] / "worker-config.json",
                    root / run_id,
                    backend,
                    args.gpu_id,
                    repeat,
                )
                specs.append(
                    RunSpec(
                        backend=backend,
                        repeat=repeat,
                        case_id=record["case_id"],
                        command=command,
                        output=root / run_id,
                        timeout_s=args.timeout_s,
                        env=environments[backend],
                    )
                )
                trial.update(command=list(command), timeout_s=args.timeout_s)
            trials.append(trial)
    frozen = {
        "schema_version": 1,
        "definition_version": DEFINITION_VERSION,
        "experiment_id": experiment_id,
        "mode": selection["mode"],
        "filters": selection["filters"],
        "repeat_policy": selection["repeat_policy"],
        "cases": cases,
        "trials": trials,
        "counts": {
            "full_planned_trials": len(trials),
            "selected_trials": sum(row["selected"] for row in trials),
            "runnable_trials": len(specs),
        },
    }
    store.register_artifact(
        write_json(root / "selection.json", frozen), kind="frozen_matrix_selection"
    )
    manifest = read_json_object(root / "manifest.json")
    manifest.update(
        selection="selection.json",
        repeat_policy=selection["repeat_policy"],
        runs=trials,
    )
    write_json(root / "manifest.json", manifest)
    return root, specs, definition.budget


def _finalize_rows(root: Path, *, interrupted: bool = False) -> list[dict[str, Any]]:
    """Merge plan-only cells without replacing already appended worker evidence."""
    selection = read_json_object(root / "selection.json")
    existing = read_json(root / "runs.json") if (root / "runs.json").exists() else []
    by_id = {row["run_id"]: row for row in existing}
    raw_records = {
        record["record_id"]: record
        for line in (root / "raw.jsonl").read_text().splitlines()
        if line.strip()
        for record in (json.loads(line),)
    }
    raw_ids = set(raw_records)
    planned_ids = {row["run_id"] for row in selection["trials"]}
    if set(by_id) - planned_ids:
        raise ValueError("Run records contain identities outside the frozen selection")
    rows = []
    store = ArtifactStore(root)
    quality = read_json_object(root / "quality.json")
    quality_ids = {row["run_id"] for row in quality["runs"]}
    for planned in selection["trials"]:
        run_id = planned["run_id"]
        row = by_id.get(run_id)
        if row is None and run_id in raw_records:
            row = {
                key: value
                for key, value in raw_records[run_id].items()
                if key != "record_id"
            }
        if row is None:
            row = dict(planned)
        elif run_id not in raw_ids and row["status"] != "completed":
            row = {**planned, **row}
        if (
            interrupted
            and run_id not in raw_ids
            and row["status"] == "not_run"
            and planned["runnable"]
        ):
            row = {
                **row,
                "failure_reason_code": "not_run",
                "failure_reason": "Launcher interrupted before this trial started",
            }
        validate_run_result(row)
        if run_id not in raw_ids:
            store.append_raw({"record_id": run_id, **row})
        if run_id not in quality_ids:
            quality["runs"].append(
                {
                    "run_id": run_id,
                    "backend": row["backend"],
                    "status": row["status"],
                    "quality_status": row.get("quality_status", "missing"),
                    "reason": row.get("failure_reason") or row.get("error"),
                    "evidence": row.get("evidence", []),
                }
            )
        rows.append(row)
    write_json(root / "runs.json", rows)
    store.write_quality(quality)
    manifest = read_json_object(root / "manifest.json")
    manifest["runs"] = selection["trials"]
    write_json(root / "manifest.json", manifest)
    return rows


def _report(root: Path) -> Path:
    """Recover the full projection, then use the shared offline report entry."""
    from .publication_report import rebuild_report

    if (root / "selection.json").exists():
        _finalize_rows(root)
    return rebuild_report(root)


def _acceptance(
    invocation: Path,
    device: dict[str, Any],
    selection: dict[str, Any],
    roots: list[Path],
) -> None:
    """Identify absent RTX 5090 execution without treating another GPU as a proxy."""
    records = []
    for root in roots:
        summary = read_json_object(root / "summary.json")
        records.append(
            {
                "experiment_id": summary["experiment_id"],
                "path": root.relative_to(invocation).as_posix(),
                "report": (root / "report.md").relative_to(invocation).as_posix(),
                "status_counts": summary["status_counts"],
                "selection_coverage": summary.get("selection_coverage"),
            }
        )
    name = device.get("gpu", {}).get("name")
    is_target = isinstance(name, str) and "RTX5090" in "".join(name.upper().split())
    started = sum(
        bool(row.get("resources", {}).get("gpu", {}).get("uuid"))
        and row.get("execution_attempted") is not False
        for root in roots
        for row in read_json(root / "runs.json")
    )
    acceptance = {
        "schema_version": 1,
        "definition_version": DEFINITION_VERSION,
        "selected_device": device,
        "target_hardware": "NVIDIA GeForce RTX 5090",
        "status": "partial" if is_target and started else "not_run",
        "reason": (
            "Execution recorded; full-matrix and quality acceptance remain separate"
            if is_target and started
            else (
                "Selected GPU is not an RTX 5090"
                if name and not is_target
                else "No RTX 5090 execution evidence"
            )
        ),
        "full_matrix_case_count": selection["full_matrix_case_count"],
        "selected_case_count": selection["selected_case_count"],
        "quality_acceptance": "not_qualified",
    }
    write_json(invocation / "acceptance.json", acceptance)
    write_json(
        invocation / "index.json",
        {
            "schema_version": 1,
            "definition_version": DEFINITION_VERSION,
            "selection": "selection.json",
            "acceptance": "acceptance.json",
            "experiments": records,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Inspect, run or rebuild an explicitly selected formal rendering matrix.

    Args:
        argv: CLI arguments, or process arguments when omitted.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.report_only is not None:
        print(_report(args.report_only.resolve()))
        return
    if (
        args.repeats < 1
        or args.warmup_frames < 0
        or args.measured_frames < 1
        or args.gpu_id < 0
        or not 0 < args.timeout_s < float("inf")
    ):
        parser.error(
            "repeats/measured-frames/timeout must be positive; warmup/gpu-id nonnegative"
        )
    if (
        args.max_cases is not None
        and args.max_cases < 1
        or args.num_envs is not None
        and any(value < 1 for value in args.num_envs)
    ):
        parser.error("max-cases and num-envs must be positive")
    if args.smoke and args.publication:
        parser.error(
            "--smoke and --publication have different repeat policies; choose one"
        )
    try:
        selection = _selection(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(
            json.dumps(
                {
                    **selection,
                    "inspection": "dry_run",
                    "filesystem_mutations": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.list:
        _print_list(selection, show_excluded=args.show_excluded, as_json=args.json)
        return
    if not selection["selected_case_count"]:
        parser.error("No cases selected by the explicit filters")
    if args.scene_root is None:
        parser.error("--scene-root is required for execution")
    if "embodichain" in selection["backends"] and not args.embodichain_python.is_file():
        parser.error("--embodichain-python must identify an installed interpreter")
    if "isaaclab" in selection["backends"]:
        if args.isaaclab_root is None or not (args.isaaclab_root / "source").is_dir():
            parser.error("--isaaclab-root must contain the frozen source directory")
        args.isaaclab_root = args.isaaclab_root.absolute()
        args.isaaclab_python = (
            args.isaaclab_python or args.isaaclab_root / "env_isaaclab/bin/python"
        )
        if not args.isaaclab_python.is_file():
            parser.error("--isaaclab-python must identify an installed interpreter")
    approval = validate_approval(args.scene_root, args.approval)
    selected = [record for record in selection["cases"] if record["selected"]]
    invocation = create_experiment_directory(
        args.output,
        experiment_id="pure-rendering-v2",
        config=selection,
        definition={
            "schema_version": 1,
            "definition_version": DEFINITION_VERSION,
            "experiment_id": "pure-rendering-v2",
        },
        assets_manifest={"scene_approval": approval},
    )
    print(f"Invocation directory: {invocation}", flush=True)
    _copy_scenes(args.scene_root, invocation / "scenes")
    shutil.copy2(
        args.approval or args.scene_root / "approval.json", invocation / "approval.json"
    )
    write_json(invocation / "selection.json", selection)
    device = _device_evidence(args.gpu_id)
    replays, preparation_errors, interrupted = _prepare_replays(invocation, selected)
    plans = [
        _freeze_experiment(
            invocation,
            args,
            selection,
            experiment_id,
            replays,
            preparation_errors,
            approval,
        )
        for experiment_id in EXPERIMENTS
        if any(record["experiment_id"] == experiment_id for record in selected)
    ]
    try:
        if not interrupted:
            for root, specs, budget in plans:
                if specs:
                    run_experiment(
                        root,
                        specs,
                        experiment_id=read_json_object(root / "definition.json")[
                            "experiment_id"
                        ],
                        budget=budget,
                    )
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for root, _, _ in plans:
            _finalize_rows(root, interrupted=interrupted)
            print(_report(root), flush=True)
        _acceptance(invocation, device, selection, [root for root, _, _ in plans])
        print(invocation / "index.json", flush=True)
    if interrupted:
        raise SystemExit(130)
    if any(
        row["status"] != "completed"
        and row["run_id"]
        in {
            planned["run_id"]
            for planned in read_json_object(root / "selection.json")["trials"]
            if planned["selected"]
        }
        for root, _, _ in plans
        for row in read_json(root / "runs.json")
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
