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
"""Measure the accepted M03 trajectory without changing its physics or media.

Run: platform m03.py --measure --repeat 1 (after run_m03.py freezes the batch).
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
import time
from typing import Any
import numpy as np

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_common import (
    OUTPUT,
    digest,
    write_json,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_measurement import (
    _environment,
    _memory_call,
    completed_step_seconds,
    file_hashes,
    summarize_steps,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m03 import (
    observations,
)

WARMUP = 30
__all__ = ["measurement_directory", "record_episode"]


def measurement_directory(platform: str, repeat: int) -> Path:
    """Reserve one flat result prefix after verifying the accepted frozen batch."""
    if platform not in ("embodichain", "isaaclab") or repeat not in range(1, 6):
        raise ValueError("M03 requires two platforms and repeats 1..5")
    protocol = json.loads((OUTPUT / "measurement/protocol.json").read_text())
    if (
        not protocol["authorized"]
        or file_hashes(OUTPUT / "inputs") != protocol["input_sha256"]
    ):
        raise ValueError("Unauthorized or changed M03 inputs")
    for path, expected in protocol["source_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for path, expected in protocol["preview_sha256"].items():
        if digest(OUTPUT / path) != expected:
            raise ValueError(f"Accepted preview changed: {path}")
    output = OUTPUT / "measurement" / platform
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"r{repeat:02d}_manifest.json").open("x") as stream:
        json.dump(dict(status="initializing", platform=platform, repeat=repeat), stream)
    return output


def record_episode(
    platform: str,
    repeat: int,
    inputs: dict[str, Any],
    output: Path,
    step: Callable[[int], None],
    readings: Callable[[], dict[str, np.ndarray]],
    synchronize: Callable[[], None],
    diagnostics: Callable[[], dict[str, Any]],
    audit: dict[str, Any],
    init_s: float,
) -> None:
    """Time completed steps, retain every state, and profile memory outside timing."""
    from scripts.benchmark.tech_reports.physics.platforms.dexuni._lib.benchlib.gpu_memory import (
        GpuProcessMemoryTracker,
    )

    tracker = GpuProcessMemoryTracker()
    tracker.start()
    prefix = output / f"r{repeat:02d}"
    manifest = dict(
        task="M03",
        platform=platform,
        repeat=repeat,
        status="running",
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        camera_count=0,
        renderer="no-render",
        config=inputs["config"],
        outer_steps=inputs["steps"],
        warmup_steps=WARMUP,
        init_s=init_s,
        init_scope="worker main through scene readiness; interpreter imports excluded",
        environment=_environment(),
        initial_model=audit,
        protocol_sha256=digest(OUTPUT / "measurement/protocol.json"),
        input_sha256=file_hashes(OUTPUT / "inputs"),
        timing_scope="Warp device-synchronized completed outer step; EC sim.update(physics_dt=1/60,step=1); IL scene.write_data_to_sim/sim.step/scene.update; per-step checks, readback, memory sampling and serialization excluded",
        control_scope="same graph-resident kinematic named replay including substep FK; one_way; no dynamic articulated feedback",
    )
    write_json(prefix.with_name(prefix.name + "_manifest.json"), manifest)
    states: dict[str, list[np.ndarray]] = {}
    rows = []
    foreign_compute = []
    started = time.perf_counter()
    try:
        for frame in range(inputs["steps"] + 1):
            elapsed = (
                completed_step_seconds(lambda: step(frame), synchronize)
                if frame
                else 0.0
            )
            before = time.perf_counter()
            data = readings()
            synchronize()
            read_s = time.perf_counter() - before
            for key, value in data.items():
                states.setdefault(key, []).append(np.array(value, copy=True))
            if not all(np.isfinite(v).all() for v in data.values()):
                raise RuntimeError(f"Nonfinite state at {frame}")
            if not np.isclose(
                data["time"], inputs["commands"]["time"][frame], rtol=0, atol=1e-8
            ):
                raise RuntimeError(f"Actual clock mismatch at {frame}")
            if not np.allclose(
                data["qpos"], inputs["targets"][frame], rtol=0, atol=1e-5
            ):
                raise RuntimeError(f"Actual joint replay mismatch at {frame}")
            if frame:
                rows.append(
                    (frame, float(data["time"]), frame > WARMUP, elapsed, read_s)
                )
            if frame % 30 == 0:
                tracker.sample(f"frame_{frame}")
            if frame % 300 == 0 or frame == inputs["steps"]:
                processes = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,process_name",
                        "--format=csv,noheader",
                    ],
                    text=True,
                )
                others = [
                    line
                    for line in processes.splitlines()
                    if line.strip() and int(line.split(",", 1)[0]) != os.getpid()
                ]
                if others:
                    foreign_compute.append(dict(frame=frame, processes=others))
                print(
                    f"{platform} r{repeat:02d}: {frame}/{inputs['steps']}", flush=True
                )
        arrays = {k: np.stack(v) for k, v in states.items()}
        seconds = np.array([r[3] for r in rows])
        effective = diagnostics()
        error = float(
            np.linalg.norm(
                arrays["tcp"] - inputs["commands"]["tcp_baked"], axis=-1
            ).max()
        )
        manifest.update(
            status="completed",
            effective=effective,
            episode_wall_s=time.perf_counter() - started,
            summary=summarize_steps(seconds),
            max_tcp_replay_error_m=error,
            observations=observations(arrays, inputs["config"]),
        )
        iterations = 8 if platform == "embodichain" else 20
        manifest["performance_checks"] = dict(
            full_finite_state=all(
                len(v) == inputs["steps"] + 1 and np.isfinite(v).all()
                for v in arrays.values()
            ),
            actual_time_grid=np.allclose(
                arrays["time"], inputs["commands"]["time"], rtol=0, atol=1e-8
            ),
            joint_replay=np.allclose(
                arrays["qpos"], inputs["targets"], rtol=0, atol=1e-5
            ),
            tcp_replay=error < 0.004,
            full_positive_timings=seconds.shape == (inputs["steps"],)
            and bool(np.isfinite(seconds).all() and (seconds > 0).all()),
            cuda_graph_captured=effective["cuda_graph_status"] == "captured",
            no_cameras=not effective["cameras"],
            frozen_budget=effective["num_substeps"] == 10
            and effective["vbd_iterations"] == iterations,
            actual_duration=abs(
                effective["simulated_elapsed_s"]
                - inputs["config"]["motion"]["duration"]
            )
            < 1e-8,
        )
        if platform == "isaaclab":
            manifest["performance_checks"]["substep_clock"] = (
                effective["actual_substeps"] == inputs["steps"] * 10
            )
        else:
            manifest["performance_checks"]["outer_step_clock"] = (
                effective["step_index"] == inputs["steps"]
            )
        manifest["readback_mean_ms"] = float(
            np.mean([r[4] for r in rows[WARMUP:]]) * 1000
        )
        manifest["foreign_gpu_process_samples"] = foreign_compute
        manifest["performance_checks"][
            "no_foreign_compute_sampled"
        ] = not foreign_compute
        if not all(manifest["performance_checks"].values()):
            raise RuntimeError("Episode violates the frozen performance protocol")
        if platform == "embodichain":
            from scripts.benchmark.tech_reports.physics.experiments.dexuni.sdk_identity import (
                loaded_sdk_identity,
            )

            manifest["sdk_runtime"] = loaded_sdk_identity()
        manifest["memory_call"] = _memory_call(
            lambda: step(inputs["steps"] + 1), readings, synchronize, tracker
        )
        manifest["memory_call"][
            "scope"
        ] = "one extra untimed warmed final-target step after the full1650-step episode; retained readback; excluded from episode states and timings"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        np.savez_compressed(
            prefix.with_name(prefix.name + "_states.npz"),
            **{k: np.stack(v) for k, v in states.items() if v},
        )
        with prefix.with_name(prefix.name + "_timing.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["frame", "actual_time_s", "steady", "step_s", "readback_s"]
            )
            writer.writerows(rows)
        manifest["gpu_memory"] = tracker.metrics()
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(prefix.with_name(prefix.name + "_manifest.json"), manifest)
        tracker.close()
