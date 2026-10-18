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
"""Freeze and optionally execute the staged live ODS reuse screen.

Run: python -m scripts.benchmark.data_pipeline.run_ods_usage_study --help

No GPU job starts without --execute. Each invocation owns one explicit GPU and
runs its selected cells sequentially. Different invocations require distinct GPU
UUIDs and CPU affinities. Reports are development diagnostics, never final-test
or stability-confirmation claims. The formal 610000 test panel is never used.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

__all__ = ["build_manifest", "parse_cpu_affinity", "run_study"]

_REPO = Path(__file__).resolve().parents[3]
_CELLS = {
    f"c{capacity}-r{reuse}": (capacity, reuse)
    for capacity in (16, 32, 64)
    for reuse in (8, 32, 128)
}
_CELLS["reference"] = (32, None)


def parse_cpu_affinity(value: str) -> list[int]:
    """Expand explicit comma-separated CPU IDs and inclusive ranges."""
    result: set[int] = set()
    try:
        for part in value.split(","):
            bounds = [int(item) for item in part.split("-")]
            if len(bounds) == 1:
                first = last = bounds[0]
            elif len(bounds) == 2:
                first, last = bounds
            else:
                raise ValueError
            if first < 0 or last < first or last - first > 65536:
                raise ValueError
            result.update(range(first, last + 1))
    except ValueError:
        raise ValueError("CPUs must be explicit IDs/ranges such as 0-7,16-23") from None
    if not result:
        raise ValueError("CPU affinity cannot be empty")
    return sorted(result)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build exact commands, resource isolation and immutable source identities."""
    if not re.fullmatch(
        r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", args.gpu_uuid
    ):
        raise ValueError("Use a full physical GPU UUID, not an index or UUID prefix")
    if args.producer_envs not in (1, 2, 4, 8):
        raise ValueError("Choose an explicitly qualified P from 1, 2, 4, 8")
    if args.stage == "confirmation" and not args.cells:
        raise ValueError("Confirmation requires explicitly selected --cells")
    cells = args.cells or list(_CELLS)
    if len(set(cells)) != len(cells) or any(cell not in _CELLS for cell in cells):
        raise ValueError("Unknown or duplicate cell; use c16-r8, ... or reference")
    cpus = parse_cpu_affinity(args.cpus)
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("Requested CPUs are outside this process's allowed affinity")
    executable = Path(args.python).resolve()
    if not executable.is_file():
        raise ValueError("Python executable does not exist")
    pythonpath = [_REPO, _REPO / "extensions", _REPO / "extensions/embodichain_tasks"]
    overlay_sources = {}
    if args.runtime_overlay:
        overlay = args.runtime_overlay.resolve()
        if not overlay.is_dir():
            raise ValueError("Runtime overlay directory does not exist")
        pythonpath.append(overlay)
        overlay_sources = {
            str(path.relative_to(overlay)): _sha256(path)
            for path in sorted(overlay.rglob("*.py"))
        }
    environment = {
        "CUDA_VISIBLE_DEVICES": args.gpu_uuid,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": os.pathsep.join(map(str, pythonpath)),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "EMBODICHAIN_SIM_EXIT_PROCESS": "0",
    }
    seeds = [13001] if args.stage == "screen" else [14001, 14002, 14003]
    jobs = []
    for cell in cells:
        capacity, reuse = _CELLS[cell]
        if capacity < 2 * args.producer_envs:
            raise ValueError("Each cache must hold at least two producer batches")
        for seed in seeds:
            identity = f"{cell}-s{seed}"
            output = args.output.resolve() / "runs" / identity
            command = [
                "taskset",
                "--cpu-list",
                ",".join(map(str, cpus)),
                str(executable),
                "-m",
                "examples.data_pipeline.ods_pick_place_showcase",
                "--output",
                str(output),
                "--mode",
                "online",
                "--policy",
                "chunk-pose",
                "--pose-ik",
                "dls",
                "--train-device",
                "cuda:0",
                "--sim-device",
                "cuda:0",
                "--buffer-size",
                str(capacity),
                "--batch-size",
                "256",
                "--producer-envs",
                str(args.producer_envs),
                "--seed",
                str(seed),
                "--producer-seed",
                str(seed + 1000000),
                "--sampling-seed",
                str(seed + 10000),
                "--rounds",
                "10",
                "--updates-per-round",
                "5000",
                "--max-generated-episodes",
                "256",
                "--wall-time-budget",
                "5400",
                "--dev-seed-base",
                "810000",
                "--dev-scenes",
                "20",
                "--final-test-scenes",
                "0",
                "--test-seed-base",
                "830000",
                "--eval-every",
                "5000",
                "--usage-statistics",
                "--defer-evaluation",
                "--no-viewer",
                "--no-video",
                "--usage-schedule",
                "fixed" if reuse is None else "anchor_budget",
            ]
            qualification = getattr(args, "qualification_report", None)
            if qualification is not None:
                qualification = Path(qualification).resolve()
                if not qualification.is_file():
                    raise ValueError("Qualification report does not exist")
                command += ["--qualification-report", str(qualification)]
            if reuse is not None:
                command += ["--anchor-reuse", str(reuse)]
            jobs.append(
                {
                    "id": identity,
                    "cell": cell,
                    "seed": seed,
                    "output": str(output),
                    "argv": command,
                    "confirmation": args.stage == "confirmation",
                    "confirmation_seeds": (
                        list(range(820000, 820100))
                        if args.stage == "confirmation"
                        else []
                    ),
                    "production_cap": 256,
                    "update_cap": 50000,
                    "stop_semantics": (
                        "credits exhausted after production cap, or update/time cap"
                        if reuse
                        else "production stops at cap; keep training last cache to update/time cap"
                    ),
                }
            )
    sources = list((_REPO / "examples/data_pipeline").glob("*ods_pick_place*.py"))
    sources += [
        _REPO / "embodichain/data_pipeline/engine/data.py",
        Path(__file__).resolve(),
        _REPO
        / "extensions/embodichain_tasks/embodichain_tasks/manipulation/pick_place.py",
    ]
    sources.extend(
        _REPO / relative
        for relative in (
            "extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/task.franka.yaml",
            "extensions/embodichain_tasks/configs/tasks/manipulation/pick_place/env.yaml",
            "extensions/embodichain_tasks/configs/components/embodiments/franka_panda.yaml",
            "extensions/embodichain_robots/franka_panda.py",
        )
    )
    return {
        "version": 1,
        "repository_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip(),
        "stage": args.stage,
        "cwd": str(_REPO),
        "gpu_uuid": args.gpu_uuid,
        "cpus": cpus,
        "environment": environment,
        "runtime_overlay_sources": overlay_sources,
        "source_sha256": {
            str(path.relative_to(_REPO)): _sha256(path) for path in sorted(sources)
        },
        "gpu_preflight": {
            "required_utilization_percent": 0,
            "minimum_free_mib": args.minimum_free_mib,
        },
        "qualification_report_sha256": (
            _sha256(Path(args.qualification_report))
            if getattr(args, "qualification_report", None) is not None
            else None
        ),
        "formal_test_used": False,
        "confirmation_is_development_only": True,
        "jobs": jobs,
    }


def _write_immutable(path: Path, value: dict[str, Any], resume: bool) -> None:
    """Refuse mutation or a resumed command/configuration mismatch."""
    if path.exists():
        if not resume or json.loads(path.read_text()) != value:
            raise FileExistsError(
                f"Existing manifest differs or --resume missing: {path}"
            )
        return
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _completed(job: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Validate finished status and cleanup, never silently resume a partial run."""
    output = Path(job["output"])
    if status.get("status") != "completed" or status.get("returncode") != 0:
        raise RuntimeError(f"Partial/failed cell needs a new output root: {job['id']}")
    cleanup = json.loads((output / "cleanup.json").read_text())
    if (
        cleanup.get("engine_cleanup_error") is not None
        or cleanup.get("evaluator_exitcode") != 0
    ):
        raise RuntimeError(f"Cell cleanup was not successful: {job['id']}")
    for name in ("failure.json", "budget_exhausted.json"):
        if (output / name).exists():
            raise RuntimeError(f"Cell has terminal failure evidence: {output / name}")
    summary = json.loads((output / "summary.json").read_text())
    deferred = json.loads((output / "deferred_evaluation.json").read_text())
    if summary.get("final_success_rate") is not None:
        raise RuntimeError("Study cell unexpectedly contains final-test results")
    return {
        "id": job["id"],
        "status": "completed",
        "end_reason": summary.get("end_reason"),
        "training_step": summary.get("training_step"),
        "training_finished_seconds": deferred["training_finished_seconds"],
        "best_development_success": summary.get("best_dev_success"),
        "latest_development_success": deferred["last_report"]["success_rate"],
        "supervised_generations": summary.get("supervised_generations"),
        "process_seconds": status["process_seconds"],
        "memory": "see run efficiency resource snapshots; not aggregated",
    }


def _policy_identity(checkpoint: Path, task_config: dict[str, Any]) -> str:
    """Hash executable policy state on CPU, excluding optimizer/step metadata."""
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    identity = {
        "policy_config": saved["policy_config"],
        "task_config": task_config,
        "controller": {
            key: saved.get("config", {}).get(key)
            for key in ("pose_ik", "arm_command_mode")
        },
    }
    digest.update(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )
    for name, tensor in sorted(saved["model"].items()):
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError("Policy identity requires a dense tensor state dictionary")
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            [name, str(value.dtype), list(value.shape)], separators=(",", ":")
        ).encode()
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        payload = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _confirmation_selection(job: dict[str, Any]) -> dict[str, Any]:
    """Freeze the two roles using selection-panel artifacts before confirmation."""
    output = Path(job["output"])
    summary_path = output / "summary.json"
    deferred_path = output / "deferred_evaluation.json"
    summary = json.loads(summary_path.read_text())
    deferred = json.loads(deferred_path.read_text())
    task_path = output / "task_config.json"
    task_config = json.loads(task_path.read_text())
    panels: list[dict[str, Any]] = []
    role_checkpoints = {}
    for role, value in (
        ("best", summary["best_checkpoint"]),
        ("latest", deferred["last_checkpoint"]),
    ):
        checkpoint = Path(value).resolve()
        if not checkpoint.is_relative_to(output.resolve()):
            raise ValueError("Confirmation checkpoint must belong to its training run")
        digest = _sha256(checkpoint)
        policy_digest = _policy_identity(checkpoint, task_config)
        role_checkpoints[role] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "policy_sha256": policy_digest,
        }
        matching = next(
            (p for p in panels if p["policy_sha256"] == policy_digest), None
        )
        if matching is not None:
            matching["roles"].append(role)
        else:
            panels.append(
                {
                    "id": f"panel_{len(panels)}",
                    "roles": [role],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": digest,
                    "policy_sha256": policy_digest,
                }
            )
    return {
        "selection_panel_seeds": list(range(810000, 810020)),
        "confirmation_seeds": job["confirmation_seeds"],
        "role_checkpoints": role_checkpoints,
        "task_config_sha256": _sha256(task_path),
        "summary_sha256": _sha256(summary_path),
        "deferred_sha256": _sha256(deferred_path),
        "formal_test_used": False,
        "panels": panels,
    }


def _read_confirmation_panel(
    output: Path, panel: dict[str, Any], seeds: list[int]
) -> dict[str, Any]:
    """Require result, source identity and proven clean evaluator termination."""
    cleanup = json.loads((output / "cleanup.json").read_text())
    if cleanup.get("forced") is not False or cleanup.get("exitcode") != 0:
        raise RuntimeError(f"Confirmation cleanup failed or is unknown: {output}")
    report = json.loads((output / "report.json").read_text())
    if (
        report.get("checkpoint_sha256") != panel["checkpoint_sha256"]
        or report.get("seeds") != seeds
        or [record.get("seed") for record in report.get("records", [])] != seeds
    ):
        raise RuntimeError(f"Confirmation result identity/panel mismatch: {output}")
    return report


def _confirmation_results(job: dict[str, Any]) -> dict[str, Any]:
    """Read only frozen best/latest confirmation, never select with these scores."""
    root = Path(job["output"]) / "confirmation"
    selection = _confirmation_selection(job)
    if json.loads((root / "selection.json").read_text()) != selection:
        raise RuntimeError("Frozen confirmation selection changed")
    roles = {}
    elapsed_seconds = 0.0
    for panel in selection["panels"]:
        report = _read_confirmation_panel(
            root / panel["id"], panel, selection["confirmation_seeds"]
        )
        elapsed_seconds += report.get("elapsed_seconds", 0.0)
        for role in panel["roles"]:
            roles[role] = {
                "success_rate": report["success_rate"],
                **selection["role_checkpoints"][role],
                "report": str(root / panel["id"] / "report.json"),
            }
    return {
        "confirmation_best_success": roles["best"]["success_rate"],
        "confirmation_latest_success": roles["latest"]["success_rate"],
        "confirmation_roles": roles,
        "confirmation_seconds": elapsed_seconds,
        "confirmation_scenes": len(selection["confirmation_seeds"]),
    }


def _run_confirmation(job: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Launch one isolated fresh evaluator per frozen executable policy."""
    root = Path(job["output"]) / "confirmation"
    root.mkdir(exist_ok=False)
    selection = _confirmation_selection(job)
    _write_immutable(root / "selection.json", selection, False)
    environment = dict(os.environ)
    environment.update(manifest["environment"])
    environment.pop("PYTHONHOME", None)
    for panel in selection["panels"]:
        output = root / panel["id"]
        command = job["argv"][:4] + [
            "-m",
            "scripts.benchmark.data_pipeline.run_ods_usage_study",
            "_confirm-one",
            "--execute",
            "--gpu-uuid",
            manifest["gpu_uuid"],
            "--checkpoint",
            panel["checkpoint"],
            "--output",
            str(output),
        ]
        _write_immutable(
            root / (panel["id"] + ".command.json"),
            {
                "argv": command,
                "environment": manifest["environment"],
                "cwd": manifest["cwd"],
            },
            False,
        )
        with (root / (panel["id"] + ".log")).open("x") as log:
            process = subprocess.Popen(
                command,
                cwd=manifest["cwd"],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=2400)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        if returncode != 0:
            raise RuntimeError(
                f"Confirmation process failed: {output}; exit={returncode}"
            )
        _read_confirmation_panel(output, panel, selection["confirmation_seeds"])
    return _confirmation_results(job)


def _confirmation_worker(
    config: dict[str, Any], requests: Any, results: Any, output: str
) -> None:
    """Import simulation only in an environment-isolated worker process."""
    import torch
    from examples.data_pipeline.ods_pick_place_showcase import _evaluate_worker

    torch.set_num_threads(1)
    _evaluate_worker(config, requests, results, output, False, 8099, False)


def _confirm_one(argv: list[str]) -> None:
    """Internal isolated executable for a single fixed confirmation panel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError(
            "Confirmation GPU isolation must be established before import"
        )
    import multiprocessing as mp
    import torch
    from examples.data_pipeline.ods_pick_place_showcase import (
        load_task_config,
        _receive,
    )

    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint_hash = _sha256(args.checkpoint)
    seeds = list(range(820000, 820100))
    config = load_task_config("cuda:0", "pose", "dls")
    config["seed"] = seeds[0]
    context = mp.get_context("spawn")
    requests, results = context.Queue(), context.Queue()
    process = context.Process(
        target=_confirmation_worker, args=(config, requests, results, str(args.output))
    )
    report = None
    cleanup = {"forced": False, "exitcode": None}
    started = time.monotonic()
    try:
        process.start()
        _receive(process, results, timeout=300)
        requests.put(
            {
                "kind": "student",
                "name": "confirmation",
                "checkpoint": str(args.checkpoint),
                "seeds": seeds,
                "trace": False,
            }
        )
        report = _receive(process, results, timeout=1800)
        report.update(
            checkpoint_sha256=checkpoint_hash,
            seeds=seeds,
            elapsed_seconds=time.monotonic() - started,
        )
    finally:
        if process.pid is not None:
            requests.put(None)
            process.join(30)
            if process.is_alive():
                cleanup["forced"] = True
                process.terminate()
                process.join(10)
            if process.is_alive():
                process.kill()
                process.join(10)
            cleanup.update(pid=process.pid, exitcode=process.exitcode)
        requests.close()
        results.close()
        requests.join_thread()
        results.join_thread()
        _atomic_json(args.output / "cleanup.json", cleanup)
    if cleanup["forced"] or cleanup["exitcode"] != 0 or report is None:
        raise RuntimeError("Confirmation did not finish with clean shutdown")
    if _sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("Frozen checkpoint changed during confirmation")
    _atomic_json(args.output / "report.json", report)


def _gpu_preflight(manifest: dict[str, Any]) -> list[dict[str, int]]:
    """Require three idle, sufficiently free readings of the assigned UUID."""
    readings = []
    for _ in range(3):
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                manifest["gpu_uuid"],
                "--query-gpu=utilization.gpu,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        utilization, free = map(int, raw.strip().split(","))
        if utilization != 0 or free < manifest["gpu_preflight"]["minimum_free_mib"]:
            raise RuntimeError(
                "Assigned GPU is not idle or has insufficient free memory"
            )
        readings.append({"utilization_percent": utilization, "free_mib": free})
        time.sleep(0.5)
    return readings


def _report(root: Path, rows: list[dict[str, Any]]) -> None:
    """Write descriptive development results without declaring a winner."""
    _atomic_json(root / "results.json", {"formal_test_used": False, "runs": rows})
    lines = [
        "# ODS reuse study execution",
        "",
        "Development diagnostics only; no stability or final-test claim.",
        "",
        "## Time & Memory",
        "",
        "| Cell | Training process s | Training finish s | Confirmation s | Memory |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row.get('process_seconds')} | {row.get('training_finished_seconds')} | {row.get('confirmation_seconds')} | {row.get('memory', 'unmeasured')} |"
        )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Cell | Status | Latest dev20 | Best dev20 | Latest confirm100 | Best confirm100 | Updates |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | {row['status']} | {row.get('latest_development_success')} | {row.get('best_development_success')} | {row.get('confirmation_latest_success')} | {row.get('confirmation_best_success')} | {row.get('training_step')} |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "Descriptive ordering only; uncompleted cells remain listed.",
        "",
        "| Order | Cell | Latest development success |",
        "| ---: | --- | ---: |",
    ]
    for order, row in enumerate(
        sorted(rows, key=lambda r: -(r.get("latest_development_success") or 0)), 1
    ):
        lines.append(
            f"| {order} | {row['id']} | {row.get('latest_development_success')} |"
        )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_study(args: argparse.Namespace) -> None:
    """Freeze commands; execute sequentially only with explicit execution opt-in."""
    manifest = build_manifest(args)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_immutable(root / "manifest.json", manifest, args.resume)
    for job in manifest["jobs"]:
        _write_immutable(
            root / (job["id"] + ".command.json"),
            {
                "argv": job["argv"],
                "environment": manifest["environment"],
                "cwd": manifest["cwd"],
            },
            args.resume,
        )
    if not args.execute:
        print(
            f"Frozen {len(manifest['jobs'])} commands in {root}; no jobs launched",
            flush=True,
        )
        return
    rows = [{"id": job["id"], "status": "pending"} for job in manifest["jobs"]]
    _report(root, rows)
    lock_path = Path("/tmp") / f"embodichain-ods-study-{args.gpu_uuid}.lock"
    with lock_path.open("a") as device_lock:
        fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, job in enumerate(manifest["jobs"]):
            status_path = root / (job["id"] + ".status.json")
            if Path(job["output"]).exists() or status_path.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"Existing cell requires --resume: {job['id']}"
                    )
                rows[index] = _completed(job, json.loads(status_path.read_text()))
                if job.get("confirmation"):
                    rows[index].update(_confirmation_results(job))
                _report(root, rows)
                continue
            readings = _gpu_preflight(manifest)
            environment = dict(os.environ)
            environment.update(manifest["environment"])
            environment.pop("PYTHONHOME", None)
            status = {"status": "running", "gpu_preflight": readings}
            _atomic_json(status_path, status)
            begin = time.perf_counter()
            with (root / (job["id"] + ".log")).open("x") as stream:
                process = subprocess.Popen(
                    job["argv"],
                    cwd=manifest["cwd"],
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait()
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    _atomic_json(
                        status_path,
                        {
                            **status,
                            "status": "interrupted",
                            "returncode": process.returncode,
                            "process_seconds": time.perf_counter() - begin,
                        },
                    )
                    raise
            status.update(
                status="completed" if returncode == 0 else "failed",
                returncode=returncode,
                process_seconds=time.perf_counter() - begin,
            )
            _atomic_json(status_path, status)
            try:
                rows[index] = _completed(job, status)
                if job.get("confirmation"):
                    rows[index].update(_run_confirmation(job, manifest))
            except BaseException as error:
                status.update(status="invalid_result", error=str(error))
                _atomic_json(status_path, status)
                rows[index] = {
                    "id": job["id"],
                    **status,
                    "status": "failed",
                    "error": str(error),
                }
                _report(root, rows)
                raise
            _report(root, rows)


def main() -> None:
    """Parse explicit study resources and freeze or execute selected cells."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("screen", "confirmation"), default="screen")
    parser.add_argument("--cells", nargs="+", choices=list(_CELLS))
    parser.add_argument("--producer-envs", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runtime-overlay", type=Path)
    parser.add_argument("--qualification-report", type=Path)
    parser.add_argument("--minimum-free-mib", type=int, default=16384)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.minimum_free_mib < 1:
        parser.error("Minimum free GPU memory must be positive")
    run_study(args)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_confirm-one":
        _confirm_one(sys.argv[2:])
    else:
        main()
