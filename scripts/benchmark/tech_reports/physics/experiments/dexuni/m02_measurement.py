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
"""Formal M02 measurement: frozen protocol, per-repeat folders, step timing.

Freeze the protocol once (records the user's authorization):
    python3 scripts/benchmark/tech_reports/physics/experiments/dexuni/m02_measurement.py
Then run each platform script with ``--measure --repeat N`` (N = 1..5).
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_common import (
    OUTPUT,
    fold_metrics,
)

WARMUP = 30
AUTHORIZATION = "可以测试了，看起来embodichain更稳更快，测一下吧"
PLATFORM_DIR = Path(__file__).resolve().parents[2] / "platforms/dexuni"
SOURCES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "m02_common.py",
    PLATFORM_DIR / "embodichain/m02.py",
    PLATFORM_DIR / "isaaclab/m02.py",
)
# Plain VBD sweeps per substep requested on each platform (EmbodiChain: set by
# the surface-fast preset; Isaac Lab: Newton's stock solver).
REQUESTED_ITERATIONS = {"embodichain": 8, "isaaclab": 10}
__all__ = ["measurement_directory", "record_episode", "write_protocol"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_hashes() -> dict[str, str]:
    root = OUTPUT / "inputs"
    return {
        str(p.relative_to(root)): _sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=lambda v: np.asarray(v).tolist()) + "\n"
    )


def write_protocol() -> None:
    """Freeze inputs, sources and the approved comparison video."""
    path = OUTPUT / "measurement/protocol.json"
    if path.exists():
        raise FileExistsError("The M02 protocol is already frozen")
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        path,
        {
            "authorized": True,
            "authorization": AUTHORIZATION,
            "authorized_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "platforms": list(REQUESTED_ITERATIONS),
            "repeats": 5,
            "warmup_steps": WARMUP,
            "order": "alternating: embodichain rNN, then isaaclab rNN",
            "timing_scope": "Warp device-synchronized completed outer step; readback, checks, memory sampling and serialization excluded",
            "requested_iterations": REQUESTED_ITERATIONS,
            "input_sha256": _input_hashes(),
            "source_sha256": {str(p): _sha(p) for p in SOURCES},
            "approved_preview_sha256": _sha(OUTPUT / "videos/m02_comparison.mp4"),
        },
    )
    print(f"Protocol frozen: {path}")


def measurement_directory(platform: str, repeat: int) -> Path:
    """Check the frozen protocol and reserve one new repeat directory."""
    protocol = json.loads((OUTPUT / "measurement/protocol.json").read_text())
    if not protocol["authorized"] or platform not in protocol["platforms"]:
        raise ValueError("M02 measurement is not authorized for this platform")
    if repeat not in range(1, protocol["repeats"] + 1):
        raise ValueError("Repeat must be 1..5")
    if _input_hashes() != protocol["input_sha256"]:
        raise ValueError("Frozen M02 inputs changed")
    for path, expected in protocol["source_sha256"].items():
        if _sha(Path(path)) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    if (
        _sha(OUTPUT / "videos/m02_comparison.mp4")
        != protocol["approved_preview_sha256"]
    ):
        raise ValueError("Approved comparison video changed")
    output = OUTPUT / "measurement" / platform / f"r{repeat:02d}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def observations(states: dict[str, np.ndarray], inputs: dict) -> dict[str, Any]:
    """Fold result and hang-phase smoothness from the recorded states."""
    particles, tcp = states["particles"], states["tcp"]
    dt_ = inputs["config"]["dt"]
    time_s = np.arange(len(particles)) * dt_
    final = fold_metrics(particles[-1], inputs["groups"])
    # Hang (7.2-8.0 s): grippers still; frame-to-frame second difference of
    # nodes more than 50 mm from both TCPs.
    middle = int(round(7.6 / dt_))
    far = np.all(
        np.linalg.norm(particles[middle][:, None] - tcp[middle][None], axis=2) > 0.05,
        axis=1,
    )
    hang = (time_s[1:-1] >= 7.2) & (time_s[1:-1] <= 8.0)
    accel = np.linalg.norm(
        particles[2:] - 2 * particles[1:-1] + particles[:-2], axis=2
    )[hang][:, far]
    return {
        "final_left_xy_error_mm": 1e3 * final["left_xy_error_m"],
        "final_right_xy_error_mm": 1e3 * final["right_xy_error_m"],
        "final_mid_xy_error_mm": 1e3 * final["mid_xy_error_m"],
        "final_highest_node_above_table_mm": 1e3 * (final["highest_z_m"] - 0.86),
        "final_lowest_node_above_table_mm": 1e3 * (final["lowest_z_m"] - 0.86),
        "hang_lowest_node_above_table_mm": 1e3
        * (particles[int(round(8.0 / dt_))][:, 2].min() - 0.86),
        "hang_jitter_p95_mm": 1e3 * float(np.percentile(accel, 95)),
        "hang_jitter_max_mm": 1e3 * float(accel.max()),
        "max_tcp_error_mm": 1e3
        * float(np.linalg.norm(tcp - inputs["tcp_baked"], axis=-1).max()),
    }


def record_episode(
    platform: str,
    output: Path,
    inputs: dict,
    advance: Callable[[], None],
    readings: Callable[[], dict[str, np.ndarray]],
    synchronize: Callable[[], None],
    diagnostics: Callable[[], dict[str, Any]],
    init_s: float,
) -> None:
    """Time every outer step of one camera-free episode and keep all states."""
    from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
        _environment,
        _memory_call,
        completed_step_seconds,
        summarize_steps,
    )
    from scripts.benchmark.tech_reports.physics.platforms.dexuni._lib.benchlib.gpu_memory import (
        GpuProcessMemoryTracker,
    )

    config = inputs["config"]
    steps = config["steps"]
    tracker = GpuProcessMemoryTracker()
    tracker.start()
    manifest: dict[str, Any] = {
        "task": "M02",
        "platform": platform,
        "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": config,
        "outer_steps": steps,
        "warmup_steps": WARMUP,
        "requested_iterations": REQUESTED_ITERATIONS[platform],
        "camera_count": 0,
        "renderer": "no-render",
        "init_s": init_s,
        "environment": _environment(),
        "protocol_sha256": _sha(OUTPUT / "measurement/protocol.json"),
        "coupling": "one_way; kinematic joint replay (plan lists two-way; one-way chosen by the user)",
    }
    _write_json(output / "manifest.json", manifest)
    states: dict[str, list[np.ndarray]] = {}
    rows: list[tuple[int, float, bool, float, float]] = []
    start = time.perf_counter()
    try:
        for frame in range(steps + 1):
            elapsed = completed_step_seconds(advance, synchronize) if frame else 0.0
            read_start = time.perf_counter()
            data = readings()
            data["target_q"] = inputs["targets"][frame]
            for key, value in data.items():
                states.setdefault(key, []).append(np.array(value, copy=True))
            read_s = time.perf_counter() - read_start
            if not all(np.isfinite(v).all() for v in data.values()):
                raise RuntimeError(f"Nonfinite state at frame {frame}")
            if not np.isclose(data["time"], frame * config["dt"], rtol=0, atol=1e-8):
                raise RuntimeError(f"Physics clock differs at frame {frame}")
            if not np.allclose(data["qpos"], data["target_q"], rtol=0, atol=1e-5):
                raise RuntimeError(f"Joint replay differs at frame {frame}")
            if frame:
                rows.append(
                    (frame, float(data["time"]), frame > WARMUP, elapsed, read_s)
                )
            if frame % 30 == 0:
                tracker.sample(f"frame_{frame}")
            if frame % 180 == 0 or frame == steps:
                print(f"{platform}: measurement {frame}/{steps}", flush=True)
        effective = diagnostics()
        manifest.update(
            status="completed",
            effective=effective,
            episode_wall_s=time.perf_counter() - start,
        )
        arrays = {k: np.stack(v) for k, v in states.items()}
        seconds = np.asarray([row[3] for row in rows])
        manifest["summary"] = summarize_steps(seconds, WARMUP)
        manifest["observations"] = observations(arrays, inputs)
        manifest["performance_checks"] = {
            "full_positive_timings": seconds.shape == (steps,)
            and bool(np.isfinite(seconds).all() and (seconds > 0).all()),
            "cuda_graph_captured": effective["cuda_graph_status"] == "captured",
            "solver_budget": effective["num_substeps"] == config["substeps"]
            and effective["vbd_iterations"] == REQUESTED_ITERATIONS[platform],
            "simulated_time": bool(
                np.isclose(
                    effective["simulated_elapsed_s"], steps * config["dt"], atol=1e-8
                )
            ),
        }
        if platform == "embodichain":
            from scripts.benchmark.tech_reports.physics.experiments.dexuni.sdk_identity import (
                loaded_sdk_identity,
            )

            manifest["sdk_runtime"] = loaded_sdk_identity()
        manifest["memory_call"] = _memory_call(advance, readings, synchronize, tracker)
        manifest["memory_call"]["scope"] = (
            "one extra untimed warmed final-target step after the episode; "
            "excluded from states and timings"
        )
        if not all(manifest["performance_checks"].values()):
            raise RuntimeError(
                f"Protocol check failed: {manifest['performance_checks']}"
            )
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        np.savez_compressed(
            output / "states.npz",
            joint_names=np.asarray(inputs["names"]),
            **{k: np.stack(v) for k, v in states.items() if v},
        )
        with (output / "timing.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["frame", "time_s", "steady", "step_s", "readback_record_s"]
            )
            writer.writerows(rows)
        manifest["gpu_memory"] = tracker.metrics()
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(output / "manifest.json", manifest)
        tracker.close()


if __name__ == "__main__":
    write_protocol()
