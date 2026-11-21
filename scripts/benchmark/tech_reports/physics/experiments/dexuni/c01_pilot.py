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
"""Record completed C01 steps for frozen pilot or formal protocols.

Run: platform c01.py / c01_bread.py --pilot-performance, after protocol freeze.
Different goods and episode lengths are reported separately, without speedups.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

import numpy as np
import psutil

from .c01_common import OUTPUT, packing_metrics
from .c01_replay import sha256, write_json
from .s04_measurement import _environment, completed_step_seconds, summarize_steps

__all__ = ["pilot_directory", "record_episode"]


def _gpu_processes() -> list[dict[str, Any]]:
    """Read compute PIDs and framebuffer use outside all timed intervals."""
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    result = []
    for row in csv.reader(value.splitlines()):
        if row:
            result.append(
                dict(pid=int(row[0]), name=row[1].strip(), gpu_mib=float(row[2]))
            )
    return result


def pilot_directory(platform: str) -> Path:
    """Require the frozen one-episode protocol, untouched previews and idle GPU."""
    root = OUTPUT / "pilot_performance"
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        platform not in ("embodichain", "isaaclab")
        or protocol["scope"] != "informal pilot only; one episode per platform"
    ):
        raise ValueError("Expected the current C01 pilot protocol")
    for p, h in protocol["protected_sha256"].items():
        if sha256(OUTPUT / p) != h:
            raise ValueError(f"Prior C01 artifact changed: {p}")
    for p, h in protocol["source_sha256"].items():
        if sha256(OUTPUT.parents[1] / p) != h:
            raise ValueError(f"Pilot source changed: {p}")
    foreign = [p for p in _gpu_processes() if p["pid"] != os.getpid()]
    if foreign:
        raise RuntimeError(f"Wait for foreign GPU compute before pilot: {foreign}")
    output = root / platform
    if output.exists():
        raise FileExistsError(f"Retain all attempts; do not overwrite {output}")
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
    *,
    protocol_path: Path | None = None,
    formal: bool = False,
    repeat: int = 1,
) -> None:
    """Time completed public steps, saving all actual states and warmup samples."""
    protocol_path = protocol_path or OUTPUT / "pilot_performance/protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["formal_measurement"] != formal:
        raise ValueError("Measurement mode differs from the frozen protocol")
    shutil.copy2(Path(__file__), output / "source/c01_pilot.py")
    shutil.copy2(
        Path(__file__).with_name("s04_measurement.py"),
        output / "source/s04_measurement.py",
    )
    if formal:
        shutil.copy2(
            Path(__file__).with_name("c01_measurement.py"),
            output / "source/c01_measurement.py",
        )
    joints, mesh = inputs["joints"], inputs["mesh"]
    scene = inputs.get("scene", inputs["manifest"]["scene"])
    total = len(joints["time"]) - 1
    manifest = dict(
        task="C01",
        platform=platform,
        status="running",
        scope=protocol["scope"],
        formal_measurement=formal,
        repeat=repeat,
        performance_measurements_started=True,
        measurement_authorized=protocol["authorized"],
        camera_count=0,
        renderer="disabled",
        user_scene_approved=False,
        scene=scene,
        joint_state_order=joints["joint_names"].tolist(),
        initial_model=audit,
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        environment=_environment(),
        warmup_steps=30,
        outer_steps=total,
        protocol_sha256=sha256(protocol_path),
        timing_scope="Warp device synchronization before and after the existing public outer-step closure; includes final synchronization; excludes state readback/checks/memory polling/serialization",
        memory_scope="Sampled nvidia-smi process framebuffer and process RSS outside timers; not transient peak or incremental allocation",
        input_sha256={
            str(p.relative_to(output / "inputs")): sha256(p)
            for p in (output / "inputs").rglob("*")
            if p.is_file()
        },
        execution_source_sha256={
            str(p.relative_to(output / "source")): sha256(p)
            for p in (output / "source").rglob("*")
            if p.is_file() and p.suffix != ".pyc"
        },
    )
    write_json(output / "manifest.json", manifest)
    states: dict[str, list[np.ndarray]] = {}
    metrics, timing, memory = [], [], []
    preview = output / "preview"
    preview.mkdir()
    foreign_seen = []
    try:
        for frame in range(total + 1):
            elapsed = completed_step_seconds(step, synchronize) if frame else 0.0
            before = time.perf_counter()
            sample = readings()
            synchronize()
            read_s = time.perf_counter() - before
            t = float(sample["time"])
            if abs(t - joints["time"][frame]) > 1e-7 or not all(
                np.isfinite(v).all() for v in sample.values()
            ):
                raise RuntimeError(f"Actual clock/finite state failure at {frame}: {t}")
            qe = float(abs(sample["qpos"] - joints["targets"][frame]).max())
            te = float(
                np.linalg.norm(
                    sample["tcp"] - joints["tcp_baked"][frame], axis=-1
                ).max()
            )
            if qe > 2e-5 or te > 2e-5:
                raise RuntimeError(f"Actual joint/TCP failure at {frame}: {qe}/{te}")
            for k, v in sample.items():
                states.setdefault(k, []).append(np.array(v, copy=True))
            row = packing_metrics(
                sample["bag"],
                {n: sample[n].mean(axis=0) for n in scene["goods"]},
                {"mouth": mesh["bag_mouth_group"], "bottom": mesh["bag_bottom_group"]},
            )
            row.update(time=t, joint_replay_error=qe, tcp_replay_error_m=te)
            metrics.append(row)
            if frame:
                timing.append((frame, t, frame > 30, elapsed, read_s))
            if frame % 120 == 0 or frame == total:
                processes = _gpu_processes()
                foreign = [p for p in processes if p["pid"] != os.getpid()]
                foreign_seen.extend(foreign)
                memory.append(
                    dict(
                        frame=frame,
                        time_s=t,
                        compute_processes=processes,
                        cpu_rss_mib=psutil.Process().memory_info().rss / 1024**2,
                    )
                )
                print(
                    f"C01 {platform} {'formal' if formal else 'pilot'} frame={frame}/{total} actual={t:.3f}s foreign_compute={len(foreign)}",
                    flush=True,
                )
        effective = diagnostics()
        manifest["effective"] = effective
        iterations = effective.get(
            "iterations", effective.get("vbd", {}).get("iterations")
        )
        if (
            effective["cuda_graph_status"] != "captured"
            or effective["step_index"] != total
            or effective["num_substeps"] != 16
            or iterations != 24
        ):
            raise RuntimeError(
                f"Actual solver cadence/budget/graph differs: {effective}"
            )
        manifest["status"] = "completed"
        seconds = np.asarray([r[3] for r in timing])
        summary = summarize_steps(seconds, 30)
        summary.update(
            step_sd_ms=float(seconds[30:].std(ddof=1) * 1000),
            readback_mean_ms=float(np.mean([r[4] for r in timing[30:]]) * 1000),
            realtime_factor=float((1 / 60) / (summary["cost_time_ms"] / 1000)),
            sampled_process_gpu_max_mib=max(
                (
                    p["gpu_mib"]
                    for s in memory
                    for p in s["compute_processes"]
                    if p["pid"] == os.getpid()
                ),
                default=None,
            ),
            sampled_cpu_rss_max_mib=max(s["cpu_rss_mib"] for s in memory),
            performance_eligible=not foreign_seen,
            foreign_compute_seen=foreign_seen,
            repeat_count=1,
            formal_measurement=formal,
        )
        write_json(output / "summary.json", summary)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        with (output / "steps.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["frame", "actual_time_s", "steady", "completed_step_s", "readback_s"]
            )
            writer.writerows(timing)
        np.savez_compressed(
            preview / "states.npz", **{k: np.asarray(v) for k, v in states.items()}
        )
        write_json(preview / "metrics.json", metrics)
        write_json(output / "memory_samples.json", memory)
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
