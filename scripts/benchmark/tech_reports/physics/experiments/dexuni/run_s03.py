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
"""Run S03 fixed-configuration repetitions in isolated platform processes.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/run_s03.py
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
__all__ = ["summarize_steps", "performance_eligibility", "main"]


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


def performance_eligibility(run: dict) -> dict:
    """Accept completed finite episodes without judging the final cloth shape."""
    states = run["states"]
    expected = round(
        run["manifest"]["controller"]["events"]["done"]
        * run["config"]["trajectory"]["frame_rate"]
    )
    checks = {
        "completed": run["manifest"]["status"] == "completed"
        and bool(run["manifest"]["controller"]["finished"]),
        "finite_state": all(
            np.isfinite(states[key]).all()
            for key in ("particles", "tcp", "qpos", "target_q", "time")
        ),
        "full_episode": len(run["step_seconds"]) == expected
        and len(states["time"]) == expected + 1
        and np.allclose(
            states["time"],
            np.arange(expected + 1) / run["config"]["trajectory"]["frame_rate"],
            rtol=0,
            atol=1e-8,
        ),
    }
    return {
        "success": all(checks.values()),
        "performance_checks": checks,
        "failed_checks": ";".join(k for k, value in checks.items() if not value),
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
    protocol = json.loads((batch / "protocol.json").read_text())
    if protocol.get("scope") == "performance_only":
        metrics["legacy_geometry_success"] = metrics["success"]
        metrics["legacy_geometry_failed_checks"] = metrics["failed_checks"]
        metrics.update(performance_eligibility(run))
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


def _performance_report(batch: Path, attempts: list[dict]) -> None:
    """Report native-platform cost without a physical-quality leaderboard."""
    config = json.loads((batch / "inputs/s03/s03_config.json").read_text())
    protocol = json.loads((batch / "protocol.json").read_text())
    preview = protocol.get(
        "approved_preview",
        str(
            TECH.parents[2]
            / "outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/videos/s03_comparison.mp4"
        ),
    )
    preview_link = Path(os.path.relpath(preview, batch)).as_posix()
    solver = config["solver"]
    lines = [
        "# S03 approved grasp: performance measurements",
        "",
        "The user approved the linked scene preview. Final cloth states may differ; no physical-quality or fidelity ranking is made. No run is excluded or repeated based on the final fold or legacy release-height threshold.",
        "",
        f"Fixed configuration, five independent processes per platform, alternating execution order. Single environment: FR3/Franka Hand, {config['cloth']['vertices_per_side']}×{config['cloth']['vertices_per_side']} cloth nodes, {config['trajectory']['frame_rate']} Hz, {solver['num_substeps']} substeps, {solver['vbd_iterations']} VBD iterations, self-contact enabled={solver['vbd_self_contact']['particle_enable_self_contact']}, CUDA graph requested={solver['use_cuda_graph']}. Identical frozen cloth mesh, IK commands and requested time grid; native controls and fixed-link representation remain platform-specific. This is a platform-step cost comparison, not an isolated solver benchmark or matched state replay.",
        "",
        "Timing: CUDA-synchronized completed outer step; initial 30 episode frames excluded. EmbodiChain times sim.update; Isaac Lab includes scene.write_data_to_sim, sim.step and scene.update. Controls, state readback, rendering, recording and serialization are outside the timed step. All samples have cameras disabled. Init cost and process wall cost are reported separately; caches may already be warm.",
        "",
        "Memory: sampled NVML process framebuffer maximum including initialization; separate from CPU retained delta or incremental allocator peak (both unmeasured). Sampling is outside the step timer in the same process, not a separate profiling pass. Mean ± sample standard deviation across the five process-level values; p95 is computed within each run then averaged.",
        "",
        "## Time & Memory",
        "",
        "| Platform | Samples | Step mean ms | Step median ms | Step p95 ms | GPU max MiB | Init s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    grouped = {}
    for platform in PLATFORMS:
        chosen = [
            a
            for a in attempts
            if a["platform"] == platform and a["mode"] == "measurement"
        ]
        rows = [a["metrics"] for a in chosen if a.get("success")]
        grouped[platform] = (chosen, rows)
        lines.append(
            f"| {platform} | {len(rows)}/5 | "
            + " | ".join(
                _mean_std(rows, k)
                for k in (
                    "step_ms",
                    "step_median_ms",
                    "step_ms_p95",
                    "gpu_peak_mib",
                    "init_s",
                )
            )
            + " |"
        )
    lines += [
        "",
        "## Execution coverage and observations",
        "",
        "Completion below means a full finite episode, not a grasp-quality success rate. Heights describe the recorded motion and are not used to select samples or rank platforms.",
        "",
        "| Platform | Completed finite / attempted | Hold corner lift mm | Final corner above table mm |",
        "| --- | --- | --- | --- |",
    ]
    for platform, (chosen, rows) in grouped.items():
        lines.append(
            f"| {platform} | {len(rows)}/{len(chosen)} | {_mean_std(rows, 'corner_hold_mean_mm')} | {_mean_std(rows, 'release_corner_height_mm')} |"
        )
    lines += [
        "",
        "## Performance summary",
        "",
        "Derived steps/s and real-time factor use synchronized step cost only; they exclude readback and application overhead. No physical-quality leaderboard is produced, following the user's explicit instruction.",
        "",
        "| Platform | Derived steps/s | Real-time factor | Readback/record ms | Process wall s |",
        "| --- | --- | --- | --- | --- |",
    ]
    for platform, (_, rows) in grouped.items():
        for row in rows:
            row["derived_steps_per_s"] = 1000 / row["step_ms"]
        lines.append(
            f"| {platform} | "
            + " | ".join(
                _mean_std(rows, k)
                for k in (
                    "derived_steps_per_s",
                    "realtime_factor",
                    "readback_record_capture_ms",
                    "process_wall_s",
                )
            )
            + " |"
        )
    for platform, (chosen, _) in grouped.items():
        first = next((a for a in chosen if "run" in a), None)
        if first:
            m = json.loads((batch / first["run"] / "manifest.json").read_text())
            lines += [
                "",
                f"{platform}: Newton {m.get('newton_version')}, Warp {m.get('warp_version')}; GPU {m.get('gpus')}.",
            ]
    lines += [
        "",
        f"[Approved scene preview]({preview_link}) · [Protocol and input hashes](protocol.json) · [All attempts](attempts.json)",
        "",
        "[Performance CSV](results/s03_performance.csv) · [Per-step timing CSV](results/s03_step_timings.csv) · [Descriptive state metrics](results/s03_measurements.csv) · [Height curves](results/s03_height_curves.csv)",
        "",
        "![Five repetitions: performance](results/s03_performance.png)",
        "",
    ]
    for note in ("NOTES.md", "VALIDATION.md"):
        if (batch / note).exists():
            lines += [f"[{note}]({note})", ""]
    (batch / "REPORT.md").write_text("\n".join(lines))
    _write_json(batch / "attempts.json", attempts)


def _report(batch: Path, attempts: list[dict]) -> None:
    if (
        json.loads((batch / "protocol.json").read_text()).get("scope")
        == "performance_only"
    ):
        _performance_report(batch, attempts)
        return
    lines = [
        "# S03 cloth-corner lift platform comparison",
        "",
        "Fixed s00/nominal, physical 10mm corner grasp of a flat 31x31 sheet. All particles remain dynamic; no pinned nodes or attachments. Native platform controls are retained.",
        "Five independent no-camera samples per platform; mean ± sample standard deviation. Initial 30 frames are excluded from synchronized step timing. Readback/recording, capture and serialization are outside step. Separate media runs are excluded from performance samples.",
        "Geometric task comparison only; no calibrated material accuracy or reliability ranking. Native backend diagnostics are ephemeral and not retained. GPU memory is sampled process framebuffer usage including initialization, not a warmed allocator-only peak; CPU retained delta and isolated incremental GPU delta are unmeasured.",
        "",
    ]
    for note in ("NOTES.md", "VALIDATION.md"):
        if (batch / note).exists():
            lines += [f"[{note}]({note})", ""]
    for platform in PLATFORMS:
        first = next(
            (a for a in attempts if a["platform"] == platform and "run" in a), None
        )
        if first:
            m = json.loads((batch / first["run"] / "manifest.json").read_text())
            lines += [
                f"{platform}: Newton {m.get('newton_version')}, Warp {m.get('warp_version')}; GPU {m.get('gpus')}.",
                "",
            ]
    lines += [
        "## Time & Memory",
        "",
        "| Platform | Samples | Step ms | GPU max MiB | Init s | Readback/record ms | Attempt wall s |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for platform in PLATFORMS:
        rows = [
            a["metrics"]
            for a in attempts
            if a["platform"] == platform
            and a["mode"] == "measurement"
            and a.get("success")
        ]
        values = [
            _mean_std(rows, k)
            for k in (
                "step_ms",
                "gpu_peak_mib",
                "init_s",
                "readback_record_capture_ms",
                "process_wall_s",
            )
        ]
        lines.append(f"| {platform} | {len(rows)}/5 | " + " | ".join(values) + " |")
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Platform | Passed / attempted | Hold corner min mm | Hold drop mm | End-hold corner min mm | Descent drift mm | Released height mm | Table penetration mm |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for platform in PLATFORMS:
        chosen = [
            a
            for a in attempts
            if a["platform"] == platform and a["mode"] == "measurement"
        ]
        rows = [a["metrics"] for a in chosen if a.get("success")]
        values = [
            _mean_std(rows, k)
            for k in (
                "corner_hold_min_mm",
                "corner_hold_drop_mm",
                "transport_corner_min_mm",
                "descent_grasp_drift_mm",
                "release_corner_height_mm",
                "table_penetration_mm",
            )
        ]
        lines.append(
            f"| {platform} | {len(rows)}/{len(chosen)} | " + " | ".join(values) + " |"
        )
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Rank | Platform | Observed success rate | Scope |",
        "| --- | --- | --- | --- |",
    ]

    def observed_rate(platform: str) -> float:
        chosen = [
            a
            for a in attempts
            if a["platform"] == platform and a["mode"] == "measurement"
        ]
        return (
            sum(bool(a.get("success")) for a in chosen) / len(chosen) if chosen else 0.0
        )

    for platform in sorted(PLATFORMS, key=observed_rate, reverse=True):
        lines.append(
            f"| {1 + sum(observed_rate(other) > observed_rate(platform) for other in PLATFORMS)} | {platform} | {observed_rate(platform):.1%} | Fixed-state samples; no robustness ranking |"
        )
    lines += [
        "",
        "All attempts, stage checks and raw run paths: [attempts.json](attempts.json).",
        "",
    ]
    if (batch / "videos/s03_comparison.mp4").exists():
        lines += [
            "[Comparison video](videos/s03_comparison.mp4) · [Time mapping](videos/s03_time_mapping.json)",
            "",
            "![Media response](videos/s03_response.png)",
            "",
        ]
    if (batch / "results/s03_repetitions.png").exists():
        lines += [
            "![All five repetitions](results/s03_repetitions.png)",
            "",
            "[Measurement CSV](results/s03_measurements.csv) · [Raw height curves CSV](results/s03_height_curves.csv)",
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
    parser.add_argument(
        "--performance-only",
        action="store_true",
        help="Report cost; final cloth shapes are descriptive, not acceptance filters",
    )
    parser.add_argument(
        "--complete-successes",
        action="store_true",
        help="Append missing successful samples/media without replacing any recorded failure",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a completed successful prefix after an orchestration interruption",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=TECH.parents[2]
        / "outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/inputs",
    )
    parser.add_argument(
        "--approved-preview",
        type=Path,
        default=TECH.parents[2]
        / "outputs/benchmarks/tech_reports/dexuni/s03_cloth_corner_lift/videos/s03_comparison.mp4",
        help="Video already accepted by the user; record its identity in this batch",
    )
    parser.add_argument("--rebuild-report", action="store_true")
    args = parser.parse_args()
    if args.complete_successes:
        args.resume = True
    if args.repeats != 5:
        parser.error("This protocol requires exactly five independent repetitions")
    batch = (
        args.batch
        or TECH.parents[2]
        / "outputs/benchmarks/tech_reports/dexuni"
        / dt.datetime.now(dt.timezone.utc).strftime("s03_%Y%m%dT%H%M%SZ")
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
    inputs = batch / "inputs"
    if args.resume:
        protocol = json.loads((batch / "protocol.json").read_text())
        for name, expected in protocol["inputs_sha256"].items():
            if hashlib.sha256((inputs / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen input changed: {name}")
        # Resume only orchestration; physical workers and their dependencies stay frozen.
        execution_files = [
            "physics/experiments/dexuni/paths.py",
            "physics/experiments/dexuni/s03_common.py",
            *[
                f"physics/platforms/dexuni/{platform}/s03.py"
                for platform in (*PLATFORMS, "dexsim")
            ],
        ]
        for name in execution_files:
            if (batch / "source" / name).read_bytes() != (TECH / name).read_bytes():
                raise ValueError(f"Frozen execution source changed: {name}")
    else:
        preview = args.approved_preview.resolve()
        if args.performance_only and not preview.is_file():
            parser.error(f"Approved scene preview does not exist: {preview}")
        batch.mkdir(parents=True, exist_ok=False)
        cache = args.input_root.resolve()
        shutil.copytree(cache / "s03", inputs / "s03")
        shutil.copytree(
            Path.home()
            / ".cache/embodichain/report-assets/local/dexuni/pilot_v1/generated",
            inputs / "generated",
        )
        snapshot = batch / "source"
        shutil.copytree(
            TECH,
            snapshot,
            ignore=shutil.ignore_patterns(
                "runs", "maintenance", "__pycache__", "*.log"
            ),
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
                "scope": (
                    "performance_only" if args.performance_only else "geometric_task"
                ),
                "inputs_sha256": identity,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "approval": (
                    "User approved the recorded scene preview and its configuration, requested five runs on both platforms, and explicitly limited conclusions to performance without judging differing final cloth states."
                    if args.performance_only
                    else "User approved raised-sheet single-view scene and requested measurement on EmbodiChain and Isaac Lab."
                ),
                "approved_preview": str(preview) if args.performance_only else None,
                "approved_preview_sha256": (
                    hashlib.sha256(preview.read_bytes()).hexdigest()
                    if args.performance_only
                    else None
                ),
            },
        )
        subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=REPO,
            stdout=(batch / "working_tree.patch").open("w"),
            check=True,
        )
    protocol = json.loads((batch / "protocol.json").read_text())
    adapter_scope = (
        ["--performance-only"] if protocol.get("scope") == "performance_only" else []
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
    attempts: list[dict] = (
        json.loads((batch / "attempts.json").read_text()) if args.resume else []
    )
    if not args.complete_successes and any(not a.get("success") for a in attempts):
        raise ValueError(
            "Resume requires a successful completed prefix; failures are never replaced"
        )
    (batch / "logs").mkdir(exist_ok=args.resume)
    plan = [
        (task, platform, repeat, "measurement")
        for task in ("s03",)
        for repeat in range(5)
        for platform in (PLATFORMS if repeat % 2 == 0 else PLATFORMS[::-1])
    ]
    if not args.skip_media:
        plan += [
            (task, platform, 0, "media") for task in ("s03",) for platform in PLATFORMS
        ]
    if args.complete_successes:
        if json.loads((batch / "adapter_check.json").read_text())["status"] != "passed":
            raise ValueError(
                "Successful adapter qualification is required before supplementation"
            )
        plan = []
        for platform in PLATFORMS:
            measured = [
                a
                for a in attempts
                if a["platform"] == platform and a["mode"] == "measurement"
            ]
            missing = max(0, 5 - sum(bool(a.get("success")) for a in measured))
            first_repeat = 1 + max((a["repeat"] for a in measured), default=-1)
            plan.extend(
                ("s03", platform, first_repeat + index, "measurement")
                for index in range(missing)
            )
        if not args.skip_media:
            for platform in PLATFORMS:
                media = [
                    a
                    for a in attempts
                    if a["platform"] == platform and a["mode"] == "media"
                ]
                if not any(a.get("success") for a in media):
                    plan.append(
                        (
                            "s03",
                            platform,
                            1 + max((a["repeat"] for a in media), default=-1),
                            "media",
                        )
                    )
    print(f"Batch: {batch}", flush=True)
    if args.resume and not args.complete_successes:
        for attempt, expected in zip(attempts, plan):
            if (
                tuple(attempt[k] for k in ("task", "platform", "repeat", "mode"))
                != expected
            ):
                raise ValueError("Recorded attempts are not a prefix of this plan")
        if not attempts:
            raise ValueError("Resume requires an existing successful reference")
        subprocess.run(
            [
                str(python["embodichain"]),
                str(TECH / "physics/experiments/dexuni/check_s03_adapter.py"),
                "--reference-run",
                str(batch / attempts[0]["run"]),
                *adapter_scope,
            ],
            cwd=TECH,
            env=env,
            check=True,
        )
    completed = 0 if args.complete_successes else len(attempts)
    for index, (task, platform, repeat, mode) in enumerate(plan):
        if index < completed:
            continue
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
        if index == 0 and not args.complete_successes:
            if not attempt.get("success"):
                raise RuntimeError(
                    "EmbodiChain preflight failed; stop before adapter qualification"
                )
            print(
                "Checking native backend equivalence (ephemeral, not recorded)...",
                flush=True,
            )
            subprocess.run(
                [
                    str(python["embodichain"]),
                    str(TECH / "physics/experiments/dexuni/check_s03_adapter.py"),
                    "--reference-run",
                    str(batch / attempt["run"]),
                    *adapter_scope,
                ],
                cwd=TECH,
                env=env,
                check=True,
            )
            _write_json(
                batch / "adapter_check.json",
                {
                    "status": "passed",
                    "scope": (
                        "model, control, TCP and held corner; final free fold descriptive"
                        if adapter_scope
                        else "task"
                    ),
                    "native_results_retained": False,
                },
            )
    if not args.skip_media and all(
        a.get("run") for a in attempts if a["mode"] == "media"
    ):
        subprocess.run(
            [
                sys.executable,
                str(TECH / "reporting/dexuni/build_s03_media.py"),
                "--batch",
                str(batch),
            ],
            cwd=TECH,
            check=True,
        )
        _report(batch, attempts)
    subprocess.run(
        [
            sys.executable,
            str(TECH / "reporting/dexuni/plot_s03_repetitions.py"),
            "--batch",
            str(batch),
        ],
        cwd=TECH,
        check=True,
    )
    _report(batch, attempts)
    print(f"Report: {batch / 'REPORT.md'}", flush=True)
    enough_samples = all(
        sum(
            a["platform"] == platform
            and a["mode"] == "measurement"
            and bool(a.get("success"))
            for a in attempts
        )
        >= 5
        for platform in PLATFORMS
    )
    enough_media = args.skip_media or all(
        any(
            a["platform"] == platform and a["mode"] == "media" and a.get("success")
            for a in attempts
        )
        for platform in PLATFORMS
    )
    completed_successfully = (
        enough_samples and enough_media
        if args.complete_successes
        else all(a.get("success") for a in attempts)
    )
    if not completed_successfully:
        sys.exit(1)


if __name__ == "__main__":
    main()
