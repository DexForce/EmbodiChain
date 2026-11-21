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
"""Run one fixed-five M03 batch, preserving accepted inputs and videos.

Run: .venv/bin/python scripts/benchmark/tech_reports/physics/experiments/dexuni/run_m03.py
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_common import (
    OUTPUT,
    digest,
    write_json,
    load_inputs,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    file_hashes,
)

TECH = REPO / "scripts/benchmark/tech_reports"
MEASUREMENT = OUTPUT / "measurement"
PYTHONS = {
    "embodichain": REPO / ".venv/bin/python",
    "isaaclab": Path("/home/oem/app/IsaacLab/.venv/bin/python"),
}
__all__ = ["prepare", "run_all_benchmarks", "main"]


def gpu_status() -> str:
    """Record device load outside all latency samples."""
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,pstate,temperature.gpu,clocks.sm,utilization.gpu,power.draw",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def compute_processes() -> str:
    """List compute users without touching another agent's processes."""
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


def prepare() -> None:
    """Freeze a single accepted scene, two solver policies and ten process attempts."""
    inputs = load_inputs()
    MEASUREMENT.mkdir(exist_ok=False)
    names = [
        "physics/experiments/dexuni/run_m03.py",
        "physics/experiments/dexuni/m03_measurement.py",
        "physics/experiments/dexuni/m03_common.py",
        "physics/experiments/dexuni/s04_measurement.py",
        "physics/experiments/dexuni/sdk_identity.py",
        "physics/platforms/dexuni/_lib/benchlib/gpu_memory.py",
        "physics/platforms/dexuni/isaaclab/m02.py",
        "reporting/dexuni/build_m03_preview.py",
        "reporting/dexuni/build_m03_performance.py",
        *[f"physics/platforms/dexuni/{p}/m03.py" for p in (*PYTHONS, "dexsim")],
    ]
    sources = {str(TECH / n): digest(TECH / n) for n in names}
    sdk = REPO / ".venv/lib/python3.11/site-packages/dexsim/engine/newton_physics"
    sources.update({str(p): digest(p) for p in sorted(sdk.rglob("*.py"))})
    previews = [
        "preview/comparison.mp4",
        "preview/dexsim.mp4",
        "preview/embodichain/video.mp4",
        "preview/isaaclab/video.mp4",
    ]
    for p in PYTHONS:
        m = json.loads((OUTPUT / f"preview/{p}/manifest.json").read_text())
        if (
            m["status"] != "completed"
            or m["effective"]["cuda_graph_status"] != "captured"
        ):
            raise ValueError("Incomplete accepted scene")
    plan = [
        dict(platform=p, repeat=i)
        for i in range(1, 6)
        for p in (list(PYTHONS) if i % 2 else list(PYTHONS)[::-1])
    ]
    protocol = dict(
        task="M03",
        authorized=True,
        authorization="Fixed-configuration performance measurement after preview acceptance",
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        repeats_per_platform=5,
        outer_steps=inputs["steps"],
        warmup_steps=30,
        measured_steps_per_episode=inputs["steps"] - 30,
        duration_s=inputs["config"]["motion"]["duration"],
        dt=inputs["config"]["solver"]["dt"],
        num_substeps=10,
        policies={
            "embodichain": "surface-fast base8 + preset adaptive fallback",
            "isaaclab": "ordinary upstream VBD20",
        },
        comparison="same accepted physical materials and kinematic replay; differing solver algorithms and iteration budgets; no physical quality ranking",
        timing="device-synchronized completed no-camera outer step; readback, validation, memory sampling and serialization excluded; first30 steps excluded as warmup",
        memory="separate warmed final-target call; Torch allocator limited coverage; process NVML sampled maximum, not transient peak",
        sampling="five fresh processes per platform, alternating order, serialized GPU; no cloth-shape exclusion",
        plan=plan,
        input_sha256=file_hashes(OUTPUT / "inputs"),
        source_sha256=sources,
        preview_sha256={p: digest(OUTPUT / p) for p in previews},
        preview_observations={
            p: json.loads((OUTPUT / f"preview/{p}/observations.json").read_text())
            for p in PYTHONS
        },
        repository_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    write_json(MEASUREMENT / "protocol.json", protocol)
    write_json(MEASUREMENT / "attempts.json", [])


def run_all_benchmarks() -> None:
    """Execute each fresh process only after other GPU compute users finish."""
    protocol = json.loads((MEASUREMENT / "protocol.json").read_text())
    attempts = json.loads((MEASUREMENT / "attempts.json").read_text())
    if any(not a.get("performance_valid") for a in attempts):
        raise ValueError("Inspect the retained incomplete attempt before resuming")
    if [dict(platform=a["platform"], repeat=a["repeat"]) for a in attempts] != protocol[
        "plan"
    ][: len(attempts)]:
        raise ValueError("Attempt prefix differs from frozen plan")
    for item in protocol["plan"][len(attempts) :]:
        while compute_processes():
            print("Waiting for existing GPU compute processes to finish...", flush=True)
            time.sleep(10)
        p, i = item["platform"], item["repeat"]
        cmd = [
            str(PYTHONS[p]),
            str(TECH / f"physics/platforms/dexuni/{p}/m03.py"),
            "--measure",
            "--repeat",
            str(i),
        ]
        attempt = dict(
            item,
            command=cmd,
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            gpu_before=gpu_status(),
        )
        attempts.append(attempt)
        write_json(MEASUREMENT / "attempts.json", attempts)
        print(f"[{len(attempts)}/10] {p} r{i:02d}", flush=True)
        start = time.perf_counter()
        try:
            with (OUTPUT / f"logs/{p}_measure_r{i:02d}.log").open("x") as stream:
                result = subprocess.run(
                    cmd,
                    cwd=REPO,
                    env=dict(os.environ, PYTHONUNBUFFERED="1"),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=1200,
                )
            manifest = json.loads(
                (MEASUREMENT / f"{p}/r{i:02d}_manifest.json").read_text()
            )
            attempt.update(
                returncode=result.returncode,
                performance_valid=result.returncode == 0
                and manifest["status"] == "completed"
                and all(manifest["performance_checks"].values()),
                mean_ms=manifest.get("summary", {}).get("cost_time_ms"),
                task_placement=manifest.get("observations", {}).get(
                    "all_four_candidate_placement"
                ),
            )
        except BaseException as error:
            attempt.update(performance_valid=False, error=repr(error))
        attempt.update(
            process_wall_s=time.perf_counter() - start,
            gpu_after=gpu_status(),
            ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        write_json(MEASUREMENT / "attempts.json", attempts)
        if not attempt["performance_valid"]:
            raise RuntimeError(f"Retained invalid attempt: {p} r{i:02d}")
        print(f"  completed {attempt['mean_ms']:.3f} ms/outer step", flush=True)
    from scripts.benchmark.tech_reports.reporting.dexuni.build_m03_performance import (
        build_report,
    )

    build_report()


def main() -> None:
    """Prepare or resume this batch without modifying the accepted preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-report", action="store_true")
    args = parser.parse_args()
    if args.rebuild_report:
        from scripts.benchmark.tech_reports.reporting.dexuni.build_m03_performance import (
            build_report,
        )

        build_report()
    else:
        if not args.resume:
            prepare()
        run_all_benchmarks()


if __name__ == "__main__":
    main()
