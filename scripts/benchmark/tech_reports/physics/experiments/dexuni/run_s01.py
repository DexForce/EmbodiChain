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
"""Run S01 fixed-configuration repetitions in isolated platform processes.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/run_s01.py
All generated inputs, logs, measurements and media live under outputs/benchmarks/tech_reports/dexuni/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[6]
TECH = REPO / "scripts/benchmark/tech_reports"
PLATFORMS = ("embodichain", "isaaclab")
__all__ = ["summarize_steps", "main"]


def summarize_steps(steps: np.ndarray, warmup: int = 30) -> dict[str, float]:
    """Summarize the declared post-settling timing window.

    Args:
        steps: Completed outer-step wall durations in seconds.
        warmup: Initial episode frames excluded before any measurements.

    Returns:
        Mean and median milliseconds, plus startup-window wall seconds.
    """
    if len(steps) <= warmup or not np.isfinite(steps).all() or (steps <= 0).any():
        raise ValueError("Invalid or insufficient completed-step measurements")
    return {
        "step_ms": float(1000 * steps[warmup:].mean()),
        "step_median_ms": float(1000 * np.median(steps[warmup:])),
        "startup_steps_s": float(steps[:warmup].sum()),
    }


def _analyzer(task: str, inputs: Path):
    spec = importlib.util.spec_from_file_location(
        f"metrics_{task}", TECH / f"reporting/dexuni/{task}/analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.INPUTS = inputs / task
    return module


def _metrics(task: str, batch: Path, run_path: Path) -> dict:
    analyzer = _analyzer(task, batch / "inputs")
    run = analyzer.load_run(run_path)
    metrics = analyzer.task_metrics(run)
    steps = run["step_seconds"]
    metrics["episode_all_steps_mean_ms"] = float(1000 * steps.mean())
    metrics.update(summarize_steps(steps))
    metrics["step_ms_mean"] = metrics["step_ms"]
    metrics["step_ms_p50"] = metrics["step_median_ms"]
    metrics["step_ms_p95"] = float(1000 * np.percentile(steps[30:], 95))
    metrics["realtime_factor"] = (1 / run["config"]["trajectory"]["frame_rate"]) / (
        metrics["step_ms"] / 1000
    )
    metrics["measured_frames"] = len(steps) - 30
    timing = np.genfromtxt(
        run_path / "timing.csv", delimiter=",", names=True, dtype=None, encoding=None
    )
    if "readback_record_capture_seconds" in timing.dtype.names:
        metrics["readback_record_capture_ms"] = float(
            1000 * np.mean(timing["readback_record_capture_seconds"][30:])
        )
    return metrics


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )


def _mean_std(rows: list[dict], key: str) -> str:
    values = [r[key] for r in rows if isinstance(r.get(key), (float, int, np.number))]
    if not values:
        return "N/A"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1) if len(values) > 1 else 0:.3f}"


def _report(batch: Path, attempts: list[dict]) -> None:
    lines = [
        "# S01 fixed-configuration platform comparison",
        "",
        "Each platform uses its native control implementation. This measures complete platform steps, not an isolated solver.",
        "Fixed s00/nominal; independent processes; no parameter matrix. Mean ± sample standard deviation across successful runs.",
        "First 30 episode frames (0.5 s simulated initial settling) are excluded by a rule fixed before running; state is not reset.",
        "The timing window covers the remaining episode, with platform-dependent duration under the same closed-loop policy.",
        "Step includes platform stepping/bookkeeping and completion synchronization; control target construction, host readback, state recording and disk serialization are outside this interval.",
        "No-camera samples retain full states for quality checks. GPU memory is a sampled process maximum, including initialization; not an exact instantaneous peak.",
        "Material calibration is unverified; quality measurements describe geometry only. Desktop GPU activity may affect timings. Attempt wall includes worker launch and parent quality extraction.",
        "",
        "## Time & Memory",
        "",
        "| Task | Platform | Successful samples | Step ms | Process GPU max MiB | Init s | Startup steps s | Readback/record ms | Attempt wall s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    setup = []
    for platform in PLATFORMS:
        sample = next(
            (a for a in attempts if a["platform"] == platform and "run" in a), None
        )
        if sample is None:
            continue
        manifest = json.loads((batch / sample["run"] / "manifest.json").read_text())
        versions = ", ".join(
            f"{key}={manifest[key]}"
            for key in (
                "dexsim_version",
                "isaaclab_version",
                "isaaclab_newton_version",
                "newton_version",
                "mujoco_warp_version",
                "warp_version",
                "torch_version",
            )
            if key in manifest
        )
        setup.append(
            f"{platform}: {versions}. GPU: {manifest.get('gpus')}. Host: {manifest.get('host')}."
        )
    lines[9:9] = setup
    for task in ("s01c", "s01t"):
        for platform in PLATFORMS:
            rows = [
                a["metrics"]
                for a in attempts
                if a["task"] == task
                and a["platform"] == platform
                and a.get("success")
                and a["mode"] == "measurement"
            ]
            lines.append(
                f"| {task} | {platform} | {len(rows)}/5 | {_mean_std(rows, 'step_ms')} | {_mean_std(rows, 'gpu_peak_mib')} | {_mean_std(rows, 'init_s')} | {_mean_std(rows, 'startup_steps_s')} | {_mean_std(rows, 'readback_record_capture_ms')} | {_mean_std(rows, 'process_wall_s')} |"
            )
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Task | Platform | Passed / attempted | Deformation % | Recovery error % | Grip drift/slip mm |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for task in ("s01c", "s01t"):
        for platform in PLATFORMS:
            selected = [
                a
                for a in attempts
                if a["task"] == task
                and a["platform"] == platform
                and a["mode"] == "measurement"
            ]
            rows = [a["metrics"] for a in selected if a.get("success")]
            deformation = (
                "hold_compression_pct" if task == "s01c" else "hold_strain_pct"
            )
            drift = "grip_slip_episode_max_mm" if task == "s01c" else "grip_drift_mm"
            lines.append(
                f"| {task} | {platform} | {len(rows)}/{len(selected)} | {_mean_std(rows, deformation)} | {_mean_std(rows, 'recovery_error_pct')} | {_mean_std(rows, drift)} |"
            )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Rank | Platform | Scope |",
        "| --- | --- | --- |",
    ]
    for platform in PLATFORMS:
        lines.append(
            f"| N/A | {platform} | Native platform comparison; five fixed-state repeats do not establish reliability or solver accuracy ranking |"
        )
    lines += [
        "",
        "Attempt details and failure reasons: [attempts.json](attempts.json). Inputs and script snapshots are stored in this batch.",
        "",
    ]
    if (batch / "videos").exists():
        lines += ["## Media", ""]
        for task in ("s01c", "s01t"):
            if (batch / "videos" / f"{task}_comparison.mp4").exists():
                lines += [
                    f"[{task.upper()} comparison video](videos/{task}_comparison.mp4) · [timestamp mapping](videos/{task}_time_mapping.json)",
                    "",
                    f"![{task.upper()} geometric response](videos/{task}_response.png)",
                    "",
                ]
    (batch / "REPORT.md").write_text("\n".join(lines))
    _write_json(batch / "attempts.json", attempts)


def main() -> None:
    """Freeze inputs/scripts and run five paired repetitions plus separate media."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--skip-media", action="store_true")
    parser.add_argument("--rebuild-report", action="store_true")
    args = parser.parse_args()
    if args.repeats != 5:
        parser.error("This protocol requires exactly five independent repetitions")
    batch = (
        args.batch
        or TECH.parents[2]
        / "outputs/benchmarks/tech_reports/dexuni"
        / dt.datetime.now(dt.timezone.utc).strftime("s01_%Y%m%dT%H%M%SZ")
    ).resolve()
    if TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni" not in batch.parents:
        parser.error(
            "Batch must be a new directory inside outputs/benchmarks/tech_reports/dexuni/"
        )
    if args.rebuild_report:
        attempts = json.loads((batch / "attempts.json").read_text())
        for attempt in attempts:
            if "run" not in attempt:
                continue
            try:
                metrics = _metrics(attempt["task"], batch, batch / attempt["run"])
                metrics["process_wall_s"] = attempt.get("metrics", {}).get(
                    "process_wall_s"
                )
                attempt["metrics"] = metrics
                attempt["success"] = bool(metrics["success"])
            except Exception as exc:
                attempt["error"] = repr(exc)
                attempt["success"] = False
        _report(batch, attempts)
        shutil.copy2(__file__, batch / "report_builder.py")
        return
    batch.mkdir(parents=True, exist_ok=False)
    cache = Path.home() / ".cache/embodichain/report-assets/local/dexuni/pilot_v1"
    inputs = batch / "inputs"
    for name in ("s01c", "s01t", "generated"):
        shutil.copytree(cache / name, inputs / name)
    snapshot = batch / "source"
    shutil.copytree(
        TECH,
        snapshot,
        ignore=shutil.ignore_patterns("runs", "maintenance", "__pycache__", "*.log"),
    )
    identity = {
        str(p.relative_to(inputs)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in inputs.rglob("*")
        if p.is_file()
    }
    _write_json(
        batch / "protocol.json",
        {
            "protocol": "dexuni-fixed-five-v1",
            "state": "s00",
            "variant": "nominal",
            "repeats": 5,
            "warmup_frames": 30,
            "control": "native; not matched replay",
            "inputs_sha256": identity,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    subprocess.run(
        ["git", "diff", "HEAD"],
        cwd=REPO,
        stdout=(batch / "working_tree.patch").open("w"),
        check=True,
    )
    env = dict(
        os.environ,
        DEXUNI_ALLOW_PILOT="1",
        DEXUNI_INPUT_ROOT=str(inputs),
        PYTHONUNBUFFERED="1",
    )
    python = {
        "embodichain": REPO / ".venv/bin/python",
        "isaaclab": Path.home() / "app/IsaacLab/.venv/bin/python",
    }
    attempts: list[dict] = []
    (batch / "logs").mkdir()
    plan = [
        (task, platform, repeat, "measurement")
        for task in ("s01c", "s01t")
        for repeat in range(5)
        for platform in (PLATFORMS if repeat % 2 == 0 else PLATFORMS[::-1])
    ]
    if not args.skip_media:
        plan += [
            (task, platform, 0, "media")
            for task in ("s01c", "s01t")
            for platform in PLATFORMS
        ]
    print(f"Batch: {batch}", flush=True)
    for index, (task, platform, repeat, mode) in enumerate(plan):
        output = batch / mode / task
        command = [
            str(python[platform]),
            str(TECH / f"physics/platforms/dexuni/{platform}/{task}.py"),
            "--state",
            "s00",
            "--variant",
            "nominal",
            "--repeat",
            str(repeat),
            "--kind",
            "full",
            "--images",
            "full" if mode == "media" else "none",
            "--out-root",
            str(output),
        ]
        log = batch / "logs" / f"{index:02d}_{task}_{platform}_{mode}_{repeat}.log"
        attempt = {
            "task": task,
            "platform": platform,
            "repeat": repeat,
            "mode": mode,
            "command": command,
            "log": str(log.relative_to(batch)),
        }
        attempts.append(attempt)
        _report(batch, attempts)
        print(
            f"[{index + 1}/{len(plan)}] {task} {platform} {mode} r{repeat}", flush=True
        )
        started = time.perf_counter()
        try:
            with log.open("w") as stream:
                result = subprocess.run(
                    command,
                    cwd=TECH,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                )
            attempt["returncode"] = result.returncode
            runs = sorted(
                (output / "raw_data" / platform).glob(f"{task}_*r{repeat:02d}_*")
            )
            if result.returncode != 0 or len(runs) != 1:
                raise RuntimeError(
                    f"worker exit={result.returncode}; found {len(runs)} run directories"
                )
            attempt["run"] = str(runs[0].relative_to(batch))
            analyzer = _analyzer(task, inputs)
            run = analyzer.load_run(runs[0])
            if run["manifest"]["status"] != "completed":
                raise RuntimeError("Worker did not complete")
            metrics = _metrics(task, batch, runs[0])
            metrics["process_wall_s"] = time.perf_counter() - started
            attempt["metrics"] = metrics
            attempt["success"] = bool(metrics["success"])
            print(
                f"  success={attempt['success']} step={metrics['step_ms']:.2f}ms {metrics.get('failed_checks', '')}",
                flush=True,
            )
        except Exception as exc:
            attempt["success"] = False
            attempt["error"] = repr(exc)
            print(f"  FAILED {exc}", flush=True)
        _report(batch, attempts)
    if not args.skip_media and all(
        a.get("run") for a in attempts if a["mode"] == "media"
    ):
        subprocess.run(
            [
                sys.executable,
                str(TECH / "reporting/dexuni/build_s01_media.py"),
                "--batch",
                str(batch),
            ],
            cwd=TECH,
            check=True,
        )
        _report(batch, attempts)
    print(f"Report: {batch / 'REPORT.md'}", flush=True)
    if not all(a.get("success") for a in attempts):
        sys.exit(1)


if __name__ == "__main__":
    main()
