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
"""Record authorized M06 episodes with completed-step timing and separate readback.

Run: use the M06 platform workers with --measure --repeat 1.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import datetime as dt
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    _environment,
    _memory_call,
    completed_step_seconds,
    file_hashes,
    summarize_steps,
    write_json,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m06 import (
    DEFAULT_OUTPUT as TASK,
    JOINTS,
    digest,
    observations,
)

WARMUP = 30
STEPS = 906
__all__ = ["measurement_directory", "performance_checks", "record_episode"]


def measurement_directory(platform: str, repeat: int) -> Path:
    """Reserve one immutable attempt after checking the frozen protocol."""
    if platform not in ("embodichain", "isaaclab") or repeat not in range(1, 6):
        raise ValueError("M06 requires the two platforms and repeats 1..5")
    protocol = json.loads((TASK / "measurement/protocol.json").read_text())
    if not protocol["authorized"]:
        raise ValueError("Formal M06 measurement is not authorized")
    if file_hashes(TASK / "inputs") != protocol["input_sha256"]:
        raise ValueError("Frozen M06 inputs changed")
    for path, expected in protocol["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for path, expected in protocol["preview_sha256"].items():
        if digest(TASK / path) != expected:
            raise ValueError(f"Frozen preview changed: {path}")
    output = TASK / "measurement" / platform / f"r{repeat:02d}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def performance_checks(
    manifest: dict[str, Any], states: dict[str, np.ndarray], seconds: np.ndarray
) -> dict[str, bool]:
    """Validate timing and executed inputs independently of final cloth geometry."""
    effective = manifest["effective"]
    expected = manifest["outer_steps"]
    step_dt = manifest["config"]["solver"]["dt"]
    return {
        "completed": manifest["status"] == "completed",
        "full_finite_state": all(
            key in states
            and len(states[key]) == expected + 1
            and np.isfinite(states[key]).all()
            for key in ("particles", "qpos", "target_q", "tcp", "time")
        ),
        "actual_time_grid": states["time"].shape == (expected + 1,)
        and np.allclose(
            states["time"], np.arange(expected + 1) * step_dt, rtol=0, atol=1e-9
        ),
        "joint_tracking": states["qpos"].shape == states["target_q"].shape
        and np.allclose(states["qpos"], states["target_q"], rtol=0, atol=1e-5),
        "full_positive_timings": seconds.shape == (expected,)
        and bool(np.isfinite(seconds).all() and (seconds > 0).all()),
        "no_cameras": manifest["camera_count"] == 0,
        "cuda_graph_captured": effective["cuda_graph_status"] == "captured",
        "solver_budget": effective["num_substeps"] == 10
        and effective["vbd_iterations"] == manifest["requested_iterations"],
        "actual_outer_steps": effective["outer_steps"] == expected
        and bool(
            np.isclose(
                effective["simulated_elapsed_s"], expected * step_dt, rtol=0, atol=1e-8
            )
        ),
    }


def record_episode(
    platform: str,
    output: Path,
    config: dict[str, Any],
    mesh: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray],
    advance: Callable[[], None],
    readings: Callable[[], dict[str, np.ndarray]],
    synchronize: Callable[[], None],
    diagnostics: Callable[[], dict[str, Any]],
    init_s: float,
) -> None:
    """Retain all 906 timings and 907 states, including failed task outcomes."""
    from scripts.benchmark.tech_reports.physics.platforms.dexuni._lib.benchlib.gpu_memory import (
        GpuProcessMemoryTracker,
    )

    tracker = GpuProcessMemoryTracker()
    tracker.start()
    manifest: dict[str, Any] = {
        "task": "M06",
        "platform": platform,
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "outer_steps": STEPS,
        "warmup_steps": WARMUP,
        "requested_iterations": 8 if platform == "embodichain" else 15,
        "camera_count": 0,
        "renderer": "no-render",
        "init_s": init_s,
        "environment": _environment(),
        "protocol_sha256": digest(TASK / "measurement/protocol.json"),
        "input_sha256": file_hashes(TASK / "inputs"),
        "timing_scope": "Warp device-synchronized completed platform outer step; EC sim.update(step=1); Isaac scene.write_data_to_sim/sim.step/scene.update; readback, checks, memory sampling and serialization excluded",
        "motion_scope": "graph-resident kinematic trajectory including per-substep FK; no dynamic articulated feedback",
    }
    write_json(output / "manifest.json", manifest)
    states: dict[str, list[np.ndarray]] = {
        k: [] for k in ("time", "particles", "qpos", "tcp", "target_q")
    }
    rows: list[tuple[int, float, bool, float, float]] = []
    episode_start = time.perf_counter()
    try:
        for frame in range(STEPS + 1):
            elapsed = completed_step_seconds(advance, synchronize) if frame else 0.0
            read_start = time.perf_counter()
            data = readings()
            data["target_q"] = trajectory["joint_targets"][frame]
            for key, value in data.items():
                states[key].append(np.array(value, copy=True))
            read_s = time.perf_counter() - read_start
            if not all(np.isfinite(value).all() for value in data.values()):
                raise RuntimeError(f"Nonfinite state at frame {frame}")
            if not np.isclose(
                data["time"], trajectory["time"][frame], rtol=0, atol=1e-9
            ):
                raise RuntimeError(f"Actual physics clock differs at frame {frame}")
            if not np.allclose(data["qpos"], data["target_q"], rtol=0, atol=1e-5):
                raise RuntimeError(f"Joint replay differs at frame {frame}")
            if frame:
                rows.append(
                    (frame, float(data["time"]), frame > WARMUP, elapsed, read_s)
                )
            if frame % 30 == 0:
                tracker.sample(f"frame_{frame}")
            if frame % 180 == 0 or frame == STEPS:
                print(f"{platform}: formal measurement {frame}/{STEPS}", flush=True)
        manifest.update(
            status="completed",
            effective=diagnostics(),
            episode_wall_s=time.perf_counter() - episode_start,
        )
        arrays = {key: np.stack(value) for key, value in states.items()}
        seconds = np.asarray([row[3] for row in rows])
        manifest["summary"] = summarize_steps(seconds)
        manifest["performance_checks"] = performance_checks(manifest, arrays, seconds)
        manifest["observations"] = observations(arrays, mesh)
        error = float(
            np.linalg.norm(arrays["tcp"] - trajectory["desired_tcp"], axis=-1).max()
        )
        manifest["max_tcp_error_m"] = error
        o = manifest["observations"]
        manifest["task_checks"] = {
            "tcp_tracking": error <= 0.0005,
            "sleeve_lift": o["left_sleeve_peak_lift_m"] >= 0.05,
            "sleeve_rightward_motion": o["left_sleeve_final_rightward_motion_m"]
            >= 0.15,
        }
        manifest["task_success"] = all(manifest["task_checks"].values())
        if platform == "embodichain":
            from scripts.benchmark.tech_reports.physics.experiments.dexuni.sdk_identity import (
                loaded_sdk_identity,
            )

            manifest["sdk_runtime"] = loaded_sdk_identity()
        # The shared memory helper takes one untimed final-target step; its episode label is overridden.
        manifest["memory_call"] = _memory_call(advance, readings, synchronize, tracker)
        manifest["memory_call"][
            "scope"
        ] = "one extra untimed warmed final-target outer step after the 906-step episode; retained readback; excluded from episode states and timings"
        if not all(manifest["performance_checks"].values()):
            raise RuntimeError("Completed episode violates the frozen timing protocol")
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        serialize_start = time.perf_counter()
        np.savez_compressed(
            output / "states.npz",
            joint_names=np.asarray(JOINTS),
            **{key: np.stack(value) for key, value in states.items() if value},
        )
        with (output / "timing.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["frame", "actual_time_s", "steady", "step_s", "readback_record_s"]
            )
            writer.writerows(rows)
        manifest["gpu_memory"] = tracker.metrics()
        manifest["serialization_s"] = time.perf_counter() - serialize_start
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        tracker.close()
