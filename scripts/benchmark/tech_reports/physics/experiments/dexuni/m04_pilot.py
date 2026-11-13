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
"""Record synchronized M04 episodes without cameras, retaining every state.

Run: platform m04.py --pilot-performance (after freezing pilot_performance/protocol.json).
This is separate from formal repeated measurements and accepted preview artifacts.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
    OUTPUT,
    digest,
    write_json,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    _environment,
    completed_step_seconds,
    summarize_steps,
)

__all__ = ["pilot_directory", "record_episode"]
WARMUP = 30


def pilot_directory(platform: str) -> Path:
    """Check frozen artifacts and reserve one pilot without replacing any run."""
    if platform not in ("embodichain", "isaaclab"):
        raise ValueError(platform)
    root = OUTPUT / "pilot_performance"
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["scope"] != "informal pilot only; one episode per platform":
        raise ValueError("Expected the one-episode pilot protocol")
    for relative, expected in protocol["protected_sha256"].items():
        if digest(OUTPUT / relative) != expected:
            raise ValueError(f"Accepted artifact changed: {relative}")
    for relative, expected in protocol["source_sha256"].items():
        if digest(OUTPUT.parents[1] / relative) != expected:
            raise ValueError(f"Pilot execution source changed: {relative}")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    if any(int(p.strip()) != os.getpid() for p in processes.splitlines() if p.strip()):
        raise RuntimeError("Wait for foreign GPU compute processes before the pilot")
    output = root / platform
    output.mkdir(exist_ok=False)
    write_json(output / "manifest.json", dict(status="initializing", platform=platform))
    return output


def record_episode(
    platform: str,
    inputs: dict[str, Any],
    output: Path,
    step: Callable[[], None],
    readings: Callable[[], dict[str, np.ndarray]],
    synchronize: Callable[[], None],
    diagnostics: Callable[[], dict[str, Any]],
    audit: dict[str, Any],
    link_names: list[str],
    *,
    repeat: int | None = None,
) -> None:
    """Time synchronized completed steps; retain checks and readback outside timing."""
    from scripts.benchmark.tech_reports.physics.platforms.dexuni._lib.benchlib.gpu_memory import (
        GpuProcessMemoryTracker,
    )

    commands, cfg = inputs["commands"], inputs["config"]
    last = len(commands["time"]) - 1
    tracker = GpuProcessMemoryTracker()
    tracker.start()
    manifest = dict(
        task="M04",
        platform=platform,
        status="running",
        scope=(
            "fixed five formal episodes per platform"
            if repeat is not None
            else "informal pilot only; one episode per platform"
        ),
        formal_measurement=repeat is not None,
        repeat=repeat,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        environment=_environment(),
        camera_count=0,
        renderer="no-render",
        outer_steps=last,
        warmup_steps=WARMUP,
        config=cfg,
        initial_model=audit,
        joint_state_order=inputs["names"],
        link_pose_order=link_names,
        input_sha256=inputs["native"]["input_sha256"],
        protocol_sha256=digest(
            OUTPUT
            / ("measurement" if repeat is not None else "pilot_performance")
            / "protocol.json"
        ),
        timing_scope="Warp device-synchronized completed outer step; EC sim.update(physics_dt=1/60,step=1); IL scene.write_data_to_sim/sim.step/scene.update; readback, checks, memory sampling and serialization excluded",
        control_scope="same frozen graph-resident named kinematic replay and substep FK; one_way; no dynamic articulated feedback",
        memory_scope="sampled process framebuffer memory outside timers; not transient peak or per-call allocation; CPU and incremental allocation unmeasured",
    )
    write_json(output / "manifest.json", manifest)
    states: dict[str, list[np.ndarray]] = {}
    rows = []
    process_samples = []
    started = time.perf_counter()
    try:
        for frame in range(last + 1):
            elapsed = completed_step_seconds(step, synchronize) if frame else 0.0
            before = time.perf_counter()
            data = readings()
            synchronize()
            read_s = time.perf_counter() - before
            for key, value in data.items():
                states.setdefault(key, []).append(np.array(value, copy=True))
            if not all(np.isfinite(v).all() for v in data.values()):
                raise RuntimeError(f"Nonfinite state at {frame}")
            if abs(float(data["time"]) - commands["time"][frame]) > 1e-7:
                raise RuntimeError(f"Actual clock mismatch at {frame}")
            if not np.allclose(
                data["qpos"], commands["targets"][frame], rtol=0, atol=2e-5
            ):
                raise RuntimeError(f"Actual named joint replay mismatch at {frame}")
            if (
                np.linalg.norm(data["tcp"] - commands["tcp_baked"][frame], axis=1).max()
                > 2e-5
            ):
                raise RuntimeError(f"Actual TCP mismatch at {frame}")
            if frame:
                rows.append(
                    (frame, float(data["time"]), frame > WARMUP, elapsed, read_s)
                )
            if frame % 120 == 0:
                tracker.sample(f"frame_{frame}")
                text = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                process_samples.append(
                    dict(
                        frame=frame,
                        foreign_processes=[
                            line
                            for line in text.splitlines()
                            if line.strip()
                            and int(line.split(",", 1)[0]) != os.getpid()
                        ],
                    )
                )
                print(
                    f"{platform} {'pilot' if repeat is None else f'r{repeat:02d}'}: {frame}/{last}",
                    flush=True,
                )
        effective = diagnostics()
        seconds = np.asarray([r[3] for r in rows])
        arrays = {k: np.stack(v) for k, v in states.items()}
        checks = dict(
            full_finite_states=all(
                len(v) == last + 1 and np.isfinite(v).all() for v in arrays.values()
            ),
            actual_time_grid=np.allclose(
                arrays["time"], commands["time"], rtol=0, atol=1e-7
            ),
            named_joint_replay=np.allclose(
                arrays["qpos"], commands["targets"], rtol=0, atol=2e-5
            ),
            tcp_replay=float(
                np.linalg.norm(arrays["tcp"] - commands["tcp_baked"], axis=-1).max()
            )
            < 2e-5,
            full_positive_timings=seconds.shape == (last,)
            and bool(np.isfinite(seconds).all() and (seconds > 0).all()),
            cuda_graph_captured=effective["cuda_graph_status"] == "captured",
            ten_substeps=effective["num_substeps"] == 10,
            actual_step_counter=(
                effective["step_index"] == last
                if platform == "embodichain"
                else effective["actual_substeps"] == last * 10
            ),
            frozen_solver_strategy=(
                effective["vbd_preset"] == "surface-fast"
                and effective["vbd"]["iterations"] == 16
                and not effective["vbd"]["multilevel_active"]
                and effective["vbd"]["particle_jacobi_polish_iterations"] == 1
                if platform == "embodichain"
                else effective["vbd_iterations"] == 20
            ),
        )
        manifest.update(
            effective=effective,
            checks=checks,
            summary=summarize_steps(seconds, WARMUP),
            readback_mean_ms=float(np.mean([r[4] for r in rows[WARMUP:]]) * 1000),
            actual_simulation_time_s=float(arrays["time"][-1]),
            episode_wall_s=time.perf_counter() - started,
            gpu_process_samples=process_samples,
            no_foreign_compute_sampled=not any(
                p["foreign_processes"] for p in process_samples
            ),
        )
        if repeat is not None:
            checks["no_foreign_compute_sampled"] = manifest[
                "no_foreign_compute_sampled"
            ]
        if not all(checks.values()):
            raise RuntimeError("Pilot failed clock, state or strategy checks")
        manifest["status"] = "completed"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if states:
            np.savez_compressed(
                output / "states.npz", **{k: np.stack(v) for k, v in states.items()}
            )
        with (output / "timing.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["frame", "actual_time_s", "steady", "step_s", "readback_s"]
            )
            writer.writerows(rows)
        manifest["gpu_memory"] = tracker.metrics()
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        tracker.close()
