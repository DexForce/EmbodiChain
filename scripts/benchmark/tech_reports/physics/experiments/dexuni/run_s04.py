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
"""Measure S04 in five independent no-camera processes per platform.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/run_s04.py
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_common import OUTPUT
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    completion_checks,
    file_hashes,
    write_json,
)

TECH = REPO / "scripts/benchmark/tech_reports"
PLATFORMS = ("embodichain", "isaaclab")
__all__ = ["run_all_benchmarks", "build_report", "main"]


def _analyze(attempt: dict) -> dict:
    """Read full raw output and audit initialized physics against the accepted preview."""
    path = OUTPUT / attempt["run"]
    manifest = json.loads((path / "manifest.json").read_text())
    with np.load(path / "states.npz") as archive:
        states = dict(archive)
    timing = np.genfromtxt(
        path / "timing.csv", delimiter=",", names=True, dtype=None, encoding=None
    )
    seconds = np.atleast_1d(timing["step_s"])
    checks = completion_checks(manifest, states, seconds)
    protocol = json.loads((OUTPUT / "measurement/protocol.json").read_text())
    checks["immutable_inputs"] = manifest["input_sha256"] == protocol["input_sha256"]
    with (
        np.load(path / "model.npz") as current,
        np.load(OUTPUT / "preview" / attempt["platform"] / "model.npz") as preview,
    ):
        model_audit = {
            key: key in current
            and current[key].shape == preview[key].shape
            and np.allclose(current[key], preview[key], rtol=1e-6, atol=1e-7)
            for key in preview.files
        }
    checks["model_matches_preview"] = all(model_audit.values())
    metrics = dict(manifest.get("summary", {}))
    metrics.update(
        {
            "success": all(checks.values()) and attempt["returncode"] == 0,
            "checks": checks,
            "model_audit": model_audit,
            "process_wall_s": attempt["process_wall_s"],
            "init_s": manifest["init_s"],
            "episode_wall_s": manifest.get("episode_wall_s"),
            "gpu_process_sampled_max_mib": manifest.get("gpu_process_sampled_max_mib"),
            **{
                key: manifest.get("memory_call", {}).get(key)
                for key in (
                    "cpu_delta_mb",
                    "gpu_delta_mb",
                    "peak_gpu_mb",
                    "peak_gpu_delta_mb",
                )
            },
            "recorded_states": len(states["time"]),
            "timed_steps": len(seconds),
            "simulated_elapsed_s": manifest.get("effective", {}).get(
                "simulated_elapsed_s"
            ),
            "graph": manifest.get("effective", {}).get("cuda_graph_status"),
            "rigid_peak_lift_m": float(
                np.max(states["rigid_pose"][:, 2, 3]) - states["rigid_pose"][0, 2, 3]
            ),
            "soft_peak_center_lift_m": float(
                np.max(states["particles"].mean(axis=1)[:, 2])
                - states["particles"][0].mean(axis=0)[2]
            ),
            "command_ms": float(1000 * np.mean(timing["command_s"][30:])),
            "readback_record_ms": float(
                1000 * np.mean(timing["readback_record_s"][30:])
            ),
        }
    )
    write_json(path / "validation.json", metrics)
    return metrics


def _mean_std(rows: list[dict], key: str) -> str:
    """Format process-level mean and sample standard deviation."""
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return "N/A"
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1) if len(values) > 1 else 0:.3f}"


def build_report(attempts: list[dict]) -> None:
    """Build exactly three performance tables, per-run CSV and raw timing plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = OUTPUT / "measurement/results"
    results.mkdir(exist_ok=True)
    rows = []
    raw = []
    for attempt in attempts:
        row = {
            "platform": attempt["platform"],
            "repeat": attempt["repeat"],
            "returncode": attempt["returncode"],
            "run": attempt["run"],
        }
        try:
            attempt["metrics"] = _analyze(attempt)
            row.update(
                {
                    k: v
                    for k, v in attempt["metrics"].items()
                    if k not in ("checks", "model_audit")
                }
            )
            timing = np.genfromtxt(
                OUTPUT / attempt["run"] / "timing.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding=None,
            )
            raw.extend(
                (
                    row["platform"],
                    row["repeat"],
                    int(t["frame"]),
                    t["time_s"],
                    bool(t["steady"]),
                    1000 * t["step_s"],
                    1000 * t["command_s"],
                    1000 * t["readback_record_s"],
                )
                for t in np.atleast_1d(timing)
            )
        except Exception as error:
            attempt["analysis_error"] = repr(error)
            row["success"] = False
        rows.append(row)
    write_json(OUTPUT / "measurement/attempts.json", attempts)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (results / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (results / "steps.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "platform",
                "repeat",
                "frame",
                "time_s",
                "steady",
                "step_ms",
                "command_ms",
                "readback_record_ms",
            ]
        )
        writer.writerows(raw)
    grouped = {p: [r for r in rows if r["platform"] == p] for p in PLATFORMS}
    # Every attempt is present; failed data are labeled rather than replaced.
    aggregate = {}
    for p, chosen in grouped.items():
        available = [r for r in chosen if r.get("cost_time_ms") is not None]
        aggregate[p] = {
            "attempts": len(chosen),
            "completed_protocol_runs": sum(bool(r.get("success")) for r in chosen),
        }
        for key in (
            "cost_time_ms",
            "median_ms",
            "p95_ms",
            "gpu_process_sampled_max_mib",
            "init_s",
            "process_wall_s",
        ):
            values = [r[key] for r in available if r.get(key) is not None]
            aggregate[p][key] = (
                {
                    "mean": float(np.mean(values)),
                    "sample_std": (
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    ),
                }
                if values
                else None
            )
    write_json(results / "summary.json", aggregate)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for index, p in enumerate(PLATFORMS):
        chosen = [r for r in grouped[p] if r.get("cost_time_ms") is not None]
        axes[0].scatter(
            [r["repeat"] + index * 0.12 for r in chosen],
            [r["cost_time_ms"] for r in chosen],
            label=p,
        )
        for repeat in range(1, 6):
            data = [r for r in raw if r[0] == p and r[1] == repeat and r[4]]
            if data:
                # 30-step block averages derive directly from the exported raw CSV.
                blocks = [data[i : i + 30] for i in range(0, len(data), 30)]
                axes[1].plot(
                    [np.mean([v[3] for v in b]) for b in blocks],
                    [np.mean([v[5] for v in b]) for b in blocks],
                    alpha=0.6,
                    color=f"C{index}",
                    label=p if repeat == 1 else None,
                )
    axes[0].set(
        xlabel="Independent process repeat",
        ylabel="Mean completed outer step (ms)",
        xticks=range(1, 6),
    )
    axes[1].set(xlabel="Scene time (s)", ylabel="30-step block mean (ms)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(results / "performance.png", dpi=160)
    plt.close(fig)
    first = next(
        (a for a in attempts if (OUTPUT / a["run"] / "manifest.json").exists()), None
    )
    hardware = (
        json.loads((OUTPUT / first["run"] / "manifest.json").read_text())["environment"]
        if first
        else {}
    )
    lines = [
        "# S04 rigid / soft squeeze-grasp performance",
        "",
        "User authorization: “测量一下性能吧”. [Accepted two-platform preview](videos/s04_comparison.mp4). Five independent processes per platform, alternating first platform per pair. All attempts are retained; final object shapes and lift heights are observations, never sample filters or physical-quality rankings.",
        "",
        "Fixed scene: W1 V030 parallel grippers, one rigid block and 441-particle / 1440-tetrahedron / 528-surface-triangle soft block, 24 named joint commands, 720 steps at 60 Hz (12 simulated seconds), 10 substeps × 12 VBD iterations, two-way coupling, CUDA Graph requested. Bundled mesh, commands, URDF/USD and config hashes are verified before each run. Initialized physics arrays are checked against each platform's accepted preview.",
        "",
        "Timing: Warp device-wide CUDA synchronization before and after each completed outer step; first 30 episode steps excluded, 690 measured steps per run. EC times sim.update; Isaac Lab times scene.write_data_to_sim + sim.step(render=False) + scene.update. Target setup, readback, recording and serialization are excluded. No camera, light or active renderer is instantiated. This measures native platform stepping; it is not isolated solver timing or matched-state replay. Init begins at worker main after imports; process wall includes imports, teardown and persistence. Realtime factor is dt / step latency, derived rather than measured as overlap throughput.",
        "",
        "Memory: GPU sampled max is NVML PID-attributed process framebuffer maximum from after scene initialization and 30-step snapshots, outside timers; it includes context, persistent buffers and caches, and is not a true transient peak or cold-init maximum. cpu_delta_mb is retained RSS change in one separate untimed warmed final-target step after the episode, retaining one host readback. gpu_delta_mb and peak_gpu_mb cover only the Torch allocator in that call, excluding Warp/native allocations; incremental Torch peak is in runs.csv. This extra call is neither a state row nor a timing sample. Unavailable readings are N/A.",
        "",
        f"Hardware: {hardware.get('gpu')}, driver {hardware.get('driver')}, {hardware.get('gpu_total_mib', 0):.0f} MiB VRAM. float32, cuda:0. Full versions, CPU affinity and thread counts are in each run manifest.",
        "",
        "Software/control differences: EC uses its installed DexSim coupled solver and Newton 1.7.0; Isaac Lab uses Newton 1.6.1 with CouplerProxy. Warp 1.17.0 on both. MuJoCo implicitfast, default 100 solver / 50 line-search iterations. EC VBD enable_cuda_fast_path=false is separate from CUDA Graph, which is checked as captured. Persistent fixed-link representation, contact capacities and platform control/publication costs differ. Cross-platform performance does not establish physical fidelity.",
        "",
        "The initial batch had an EC outer-step cadence error (default step=10). Four completed processes and one interrupted initialization are retained under [invalid_cadence](measurement/invalid_cadence/README.md), with sources and logs. They are excluded solely for mismatched simulated duration. Corrected workers explicitly advance one outer step and validate actual backend simulation time against 12 seconds. The user accepted the corrected preview with “可以，按修正版重测” before this final batch.",
        "",
        "Environment differences: EC uses Python 3.11, Torch 2.10.0 and MuJoCo/MuJoCo Warp 3.15.0; Isaac Lab uses Python 3.12, Torch 2.11.0+cu130 and MuJoCo/MuJoCo Warp 3.12.0. Both have 16 Torch intra-op / 32 inter-op threads. EC DexSim runtime identity (which has no installed distribution metadata) is recorded separately in [runtime_identity.json](measurement/runtime_identity.json). These version and implementation differences are part of this platform comparison.",
        "",
        "Values are mean ± sample standard deviation across five process-level values; median and p95 are calculated within each run then averaged.",
        "",
        "## Time & Memory",
        "",
        "| Platform | Runs | cost_time_ms | Median ms | p95 ms | GPU sampled max MiB | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb (Torch) | Init s | Process wall s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p, chosen in grouped.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    p,
                    str(len(chosen)),
                    *[
                        _mean_std(chosen, k)
                        for k in (
                            "cost_time_ms",
                            "median_ms",
                            "p95_ms",
                            "gpu_process_sampled_max_mib",
                            "cpu_delta_mb",
                            "gpu_delta_mb",
                            "peak_gpu_mb",
                            "init_s",
                            "process_wall_s",
                        )
                    ],
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Completion & Observations",
        "",
        "| Platform | Complete protocol runs | success_rate | States / timed steps per run | CUDA Graph captured | Rigid peak lift m | Soft peak center lift m |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p, chosen in grouped.items():
        lines.append(
            f"| {p} | {sum(bool(r.get('success')) for r in chosen)}/{len(chosen)} | {sum(bool(r.get('success')) for r in chosen)/len(chosen):.0%} | "
            + "; ".join(
                f"r{r['repeat']:02d}: {r.get('recorded_states', 'N/A')}/{r.get('timed_steps', 'N/A')}"
                for r in chosen
            )
            + f" | {sum(r.get('graph') == 'captured' for r in chosen)}/{len(chosen)} | {_mean_std(chosen, 'rigid_peak_lift_m')} | {_mean_std(chosen, 'soft_peak_center_lift_m')} |"
        )
    lines += [
        "",
        "## Performance Summary",
        "",
        "| Platform | Mean step ms | Derived realtime factor | Relative mean step cost (EC=1) |",
        "| --- | --- | --- | --- |",
    ]
    ec_mean = aggregate["embodichain"].get("cost_time_ms")
    for p in PLATFORMS:
        mean = aggregate[p].get("cost_time_ms")
        if mean and ec_mean:
            lines.append(
                f"| {p} | {_mean_std(grouped[p], 'cost_time_ms')} | {1000/60/mean['mean']:.3f}× | {mean['mean']/ec_mean['mean']:.3f} |"
            )
    lines += [
        "",
        "![All five process repetitions and scene timing](measurement/results/performance.png)",
        "",
        "[Per-run measurements](measurement/results/runs.csv) · [All raw step/command/readback timings and plotting inputs](measurement/results/steps.csv) · [All attempts](measurement/attempts.json) · [Protocol and identities](measurement/protocol.json)",
        "",
        "Per-run directories contain states.npz, timing.csv, initialized model.npz, environment/graph/memory manifests, validation.json and process logs. Measurement sources are frozen under measurement/source/.",
        "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines))
    print(f"Report: {OUTPUT / 'REPORT.md'}", flush=True)


def run_all_benchmarks(resume: bool = False) -> None:
    """Serialize ten GPU processes, preserving failures without automatic replacement."""
    measurement = OUTPUT / "measurement"
    sources = [
        "physics/experiments/dexuni/s04_common.py",
        "physics/experiments/dexuni/s04_measurement.py",
        "physics/experiments/dexuni/run_s04.py",
        "tests/test_s04_protocol.py",
        *[f"physics/platforms/dexuni/{p}/s04.py" for p in PLATFORMS],
        "physics/platforms/dexuni/_lib/benchlib/gpu_memory.py",
    ]
    identity = {p: hashlib.sha256((TECH / p).read_bytes()).hexdigest() for p in sources}
    if resume:
        protocol = json.loads((measurement / "protocol.json").read_text())
        if (
            file_hashes(OUTPUT / "inputs") != protocol["input_sha256"]
            or identity != protocol["source_sha256"]
        ):
            raise ValueError("Frozen inputs or measurement sources changed")
        attempts = json.loads((measurement / "attempts.json").read_text())
    else:
        if not json.loads((OUTPUT / "implementation.json").read_text())[
            "measurement_authorized"
        ]:
            raise ValueError("Latest user measurement authorization is required")
        if measurement.exists() and any(
            p.name != "invalid_cadence" for p in measurement.iterdir()
        ):
            raise ValueError(
                "Existing effective measurements must be resumed, never overwritten"
            )
        measurement.mkdir(exist_ok=True)
        for name in sources:
            target = measurement / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(TECH / name, target)
        protocol = {
            "task": "S04",
            "authorized": True,
            "approval_reply": "可以，按修正版重测",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repeats": 5,
            "warmup_steps": 30,
            "scope": "native platform performance; no physical-quality ranking",
            "input_sha256": file_hashes(OUTPUT / "inputs"),
            "source_sha256": identity,
            "approved_preview": "videos/s04_comparison.mp4",
            "approved_preview_sha256": hashlib.sha256(
                (OUTPUT / "preview/comparison.mp4").read_bytes()
            ).hexdigest(),
        }
        write_json(measurement / "protocol.json", protocol)
        attempts = []
    plan = [
        (p, repeat)
        for repeat in range(1, 6)
        for p in (PLATFORMS if repeat % 2 else PLATFORMS[::-1])
    ]
    if [(a["platform"], a["repeat"]) for a in attempts] != plan[: len(attempts)]:
        raise ValueError("Attempts are not a prefix of the serialized plan")
    interpreters = {
        "embodichain": REPO / ".venv/bin/python",
        "isaaclab": Path("/home/oem/app/IsaacLab/.venv/bin/python"),
    }
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    manifest.update(
        performance_measurements_started=True,
        current_phase="performance_measurement_running",
    )
    write_json(OUTPUT / "manifest.json", manifest)
    for p, repeat in plan[len(attempts) :]:
        run = Path("measurement") / p / f"r{repeat:02d}"
        log = measurement / f"{p}_r{repeat:02d}.log"
        if (OUTPUT / run).exists() or log.exists():
            raise ValueError(f"Unrecorded existing attempt requires inspection: {run}")
        command = [
            str(interpreters[p]),
            str(TECH / f"physics/platforms/dexuni/{p}/s04.py"),
            "--measure",
            "--repeat",
            str(repeat),
        ]
        print(f"[{len(attempts)+1}/10] {p} r{repeat:02d}", flush=True)
        start = time.perf_counter()
        with log.open("w") as stream:
            result = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        attempt = {
            "platform": p,
            "repeat": repeat,
            "run": str(run),
            "log": str(log.relative_to(OUTPUT)),
            "command": command,
            "returncode": result.returncode,
            "process_wall_s": time.perf_counter() - start,
        }
        attempts.append(attempt)
        write_json(measurement / "attempts.json", attempts)
        print(
            f"Finished {p} r{repeat:02d}: rc={result.returncode}, wall={attempt['process_wall_s']:.2f}s",
            flush=True,
        )
        if result.returncode:
            build_report(attempts)
            raise RuntimeError(f"Attempt failed; retained log: {log}")
    build_report(attempts)
    success = all(a.get("metrics", {}).get("success") for a in attempts)
    manifest.update(
        current_phase=(
            "performance_measurement_completed"
            if success
            else "performance_measurement_requires_review"
        ),
        measurement_summary="measurement/results/summary.json",
        measurement_attempts=len(attempts),
    )
    write_json(OUTPUT / "manifest.json", manifest)
    if not success:
        raise RuntimeError(
            "Finished samples failed protocol validation; inspect validation.json"
        )


def main() -> None:
    """Execute the fixed benchmark or rebuild its report from retained raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-report", action="store_true")
    args = parser.parse_args()
    if args.rebuild_report:
        build_report(json.loads((OUTPUT / "measurement/attempts.json").read_text()))
    else:
        run_all_benchmarks(args.resume)


if __name__ == "__main__":
    main()
