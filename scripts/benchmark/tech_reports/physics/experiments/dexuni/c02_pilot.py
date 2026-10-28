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
"""Measure one informal complete C02 episode per platform without cameras.

Run: .venv/bin/python -m scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_pilot --launch
The accepted workers stay byte-identical. Runtime instrumentation separates
completed public physics advancement, the full feedback/IK/command step, and
readback. All physical samples and attempts are retained, without quality filtering.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

import numpy as np

__all__ = ["main"]
TECH = Path(__file__).resolve().parents[3]
REPO = TECH.parents[2]
ROOT = TECH.parents[2] / "outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing"
OUTPUT = ROOT / "pilot_performance"
SDK_PATH = "/home/oem/code/repos/dexsimforembodychain/dexsim/modules/dexsim/build_Release/lib/python_package:/home/oem/miniconda3/envs/dexsim1/lib/python3.11/site-packages"
WARMUP = 30


def _sha(path: Path) -> str:
    """Hash one artifact without modifying it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    """Read one structured pilot artifact."""
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    """Write small metadata with native scalar values."""
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _gpu() -> list[dict[str, Any]]:
    """Sample process framebuffer residency outside every timed interval."""
    value = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        {"pid": int(r[0]), "name": r[1].strip(), "gpu_mib": float(r[2])}
        for r in csv.reader(value.splitlines())
        if r
    ]


def _activity(samples: int = 1) -> tuple[dict[int, float], str]:
    """Read sampled per-process SM activity, preserving the original monitor text."""
    raw = subprocess.check_output(
        ["nvidia-smi", "pmon", "-c", str(samples), "-s", "um"], text=True
    )
    result: dict[int, float] = {}
    for line in raw.splitlines():
        row = line.split()
        if len(row) > 4 and row[0].isdigit() and row[1].isdigit():
            pid = int(row[1])
            sm = float(row[3]) if row[3] != "-" else 0.0
            result[pid] = max(result.get(pid, 0.0), sm)
    return result, raw


def _verify(protocol: dict[str, Any]) -> None:
    """Check every protected input, recording, runtime and benchmark source."""
    for rel, digest in protocol["protected_sha256"].items():
        assert _sha(ROOT / rel) == digest, rel
    for rel, digest in protocol["source_sha256"].items():
        assert _sha(TECH / rel) == digest, rel
    assert (
        _sha(OUTPUT / "source/task_sources.zip") == protocol["task_sources_zip_sha256"]
    )


def _prepare() -> dict[str, Any]:
    """Freeze the single-episode preliminary protocol before any timing starts."""
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "source").mkdir()
    sources = sorted(
        set((TECH / "physics/experiments/dexuni").glob("*.py"))
        | set((TECH / "physics/platforms/dexuni/embodichain").glob("*.py"))
        | set((TECH / "physics/platforms/dexuni/isaaclab").glob("*.py"))
    )
    source_hashes = {str(f.relative_to(TECH)): _sha(f) for f in sources}
    with zipfile.ZipFile(
        OUTPUT / "source/task_sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for f in sources:
            archive.write(f, str(f.relative_to(TECH)))
    protected = _read(ROOT / "analysis/platform_port/accepted_native_sha256.json")
    mutable = {"REPORT.md", "README.md", "report_provenance.json"}
    for f in (ROOT / "platforms").rglob("*"):
        if f.is_file() and "__pycache__" not in f.parts and f.name not in mutable:
            protected[str(f.relative_to(ROOT))] = _sha(f)
    with zipfile.ZipFile(OUTPUT / "source/preview_reports.zip", "w") as archive:
        for name in mutable:
            archive.write(ROOT / "platforms" / name, name)
    protocol = dict(
        authorized_user_request="Informal timing after complete comparison recordings",
        scope="Informal pilot only; one fresh complete64s process per platform",
        formal_measurement=False,
        episodes_per_platform=1,
        camera_count=0,
        outer_steps=3840,
        outer_dt_s=1 / 60,
        warmup_steps=WARMUP,
        primary_metric="Completed public physics advancement plus state publication, synchronized before/after; target upload and IK excluded",
        secondary_metric="Instrumented complete existing feedback/IK/command/physics step; includes timing-hook synchronization overhead",
        readback="Existing owned public state readback, synchronized, separate from both step scopes",
        memory="Sampled nvidia-smi main/IK process framebuffer and main RSS outside timers; not transient peak; recorder retains raw states",
        configurations={
            "embodichain": {"iterations": 12, "substeps": 6, "trajectory": "reference"},
            "isaaclab": {
                "iterations": 96,
                "substeps": 24,
                "trajectory": "accepted bimanual adaptation",
            },
        },
        strategy_scope="Original task materials/time grid; solver strategy and robot trajectory differ as recorded in accepted previews. No identical-control/budget claim, formal ranking or success-rate estimate.",
        background_policy="Wait for active external compute before launch; idle resident contexts and desktop graphics disclosed; retain all attempts and background samples",
        quality_policy="Original runtime shape/upright gates unchanged; all finite/clock/joint/raw checks; independent packing/lift and original geometry checks after each attempt; no geometry-based timing filtering",
        protected_sha256=protected,
        source_sha256=source_hashes,
        task_sources_zip_sha256=_sha(OUTPUT / "source/task_sources.zip"),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    _write(OUTPUT / "protocol.json", protocol)
    return protocol


class _Timers:
    """Instrument the existing worker closures without changing simulation logic."""

    def __init__(self, platform: str, step: Callable, readings: Callable) -> None:
        import psutil
        import warp as wp

        self.platform = platform
        self.step = step
        self.readings = readings
        self.synchronize = lambda: wp.synchronize_device("cuda:0")
        values = dict(
            zip(
                step.__code__.co_freevars,
                (c.cell_contents for c in step.__closure__),
                strict=True,
            )
        )
        self.sim = values["sim"]
        self.controller = values["controller"]
        self.owned_pids = {os.getpid(), self.controller.process.pid}
        self.process = psutil.Process()
        self.rows: list[dict[str, Any]] = []
        self.memory: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.originals: list[tuple[Any, str, Any]] = []
        self.frame = 0
        if platform == "embodichain":
            self._replace(self.sim, "update", self._ec_update)
        else:
            self.scene = values["scene"]
            self.physics_started: float | None = None
            self._replace(self.sim, "step", self._il_step)
            self._replace(self.scene, "update", self._il_update)

    def _replace(self, obj: Any, name: str, method: Callable) -> None:
        """Keep each bound method for exact delegation and later restoration."""
        self.originals.append((obj, name, getattr(obj, name)))
        setattr(obj, name, method)

    def restore(self) -> None:
        """Restore instance methods before normal platform teardown."""
        for obj, name, method in self.originals:
            setattr(obj, name, method)

    def _ec_update(self, *args: Any, **kwargs: Any) -> Any:
        """Time the existing EC public advancement and publication call."""
        self.synchronize()
        start = time.perf_counter()
        result = self.originals[0][2](*args, **kwargs)
        self.synchronize()
        assert self.current is not None
        self.current["physics_s"] = time.perf_counter() - start
        return result

    def _il_step(self, *args: Any, **kwargs: Any) -> Any:
        """Start timing after target submission and before Isaac advancement."""
        self.synchronize()
        assert self.physics_started is None
        self.physics_started = time.perf_counter()
        return self.originals[0][2](*args, **kwargs)

    def _il_update(self, *args: Any, **kwargs: Any) -> Any:
        """Finish after the existing Isaac scene-state publication call."""
        result = self.originals[1][2](*args, **kwargs)
        self.synchronize()
        assert self.current is not None and self.physics_started is not None
        self.current["physics_s"] = time.perf_counter() - self.physics_started
        self.physics_started = None
        return result

    def advance(self) -> None:
        """Time the complete original feedback/IK/command/physics closure."""
        self.frame += 1
        row = {"frame": self.frame, "steady": self.frame > WARMUP}
        self.rows.append(row)
        self.current = row
        self.synchronize()
        start = time.perf_counter()
        self.step()
        self.synchronize()
        row["outer_s"] = time.perf_counter() - start
        assert row["physics_s"] > 0

    def observe(self) -> dict[str, Any]:
        """Time readback separately, then sample memory outside that interval."""
        self.synchronize()
        start = time.perf_counter()
        result = self.readings()
        self.synchronize()
        elapsed = time.perf_counter() - start
        if self.current is not None:
            self.current.update(readback_s=elapsed, actual_time_s=float(result["time"]))
        if self.frame % 120 == 0 or self.frame == 3840:
            processes = _gpu()
            activity, raw = _activity()
            external = [p for p in processes if p["pid"] not in self.owned_pids]
            self.memory.append(
                dict(
                    frame=self.frame,
                    actual_time_s=float(result["time"]),
                    processes=processes,
                    external_compute=external,
                    active_external_compute=[
                        p for p in external if activity.get(p["pid"], 0) > 0
                    ],
                    pmon=raw,
                    cpu_rss_mib=self.process.memory_info().rss / 1024**2,
                )
            )
        return result

    def save(self, output: Path, environment: dict[str, Any]) -> None:
        """Retain every timing and background sample, including incomplete runs."""
        from .s04_measurement import summarize_steps

        columns = [
            "frame",
            "actual_time_s",
            "steady",
            "physics_s",
            "outer_s",
            "readback_s",
        ]
        with (output / "steps.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.rows)
        _write(output / "memory_samples.json", self.memory)
        complete = [r for r in self.rows if all(k in r for k in columns)]
        summaries = {}
        if len(complete) > WARMUP:
            for field in ("physics_s", "outer_s", "readback_s"):
                summaries[field] = summarize_steps(
                    np.array([r[field] for r in complete]), WARMUP
                )
        phases = []
        for a, b, name in [
            (0, 4.5, "approach"),
            (4.5, 10.5, "tip"),
            (10.5, 24, "handover"),
            (24, 36, "can"),
            (36, 48, "carton"),
            (48, 50, "settle"),
            (50, 56, "regrasp"),
            (56, 58, "lift"),
            (58, 60, "hold"),
            (60, 62, "lower"),
            (62, 64, "final"),
        ]:
            rows = [
                r
                for r in complete
                if r["steady"] and round(a * 60) < r["frame"] <= round(b * 60)
            ]
            if rows:
                phases.append(
                    {
                        "phase": name,
                        "samples": len(rows),
                        **{
                            field
                            + "_mean_ms": float(
                                np.mean([r[field] for r in rows]) * 1000
                            )
                            for field in ("physics_s", "outer_s", "readback_s")
                        },
                    }
                )
        gpu_max = {}
        for key, pid in [("main", os.getpid()), ("ik", self.controller.process.pid)]:
            values = [
                p["gpu_mib"]
                for s in self.memory
                for p in s["processes"]
                if p["pid"] == pid
            ]
            gpu_max[key] = max(values, default=None)
        manifest = _read(output / "manifest.json")
        manifest.update(
            performance_measurements=True,
            formal_measurement=False,
            camera_count=0,
            pilot_protocol_sha256=_sha(OUTPUT / "protocol.json"),
            timing_scope=_read(OUTPUT / "protocol.json")["primary_metric"],
            environment=environment,
        )
        _write(output / "manifest.json", manifest)
        _write(
            output / "summary.json",
            dict(
                scope="One informal complete-episode attempt; not a repeated formal estimate",
                repeat_count=1,
                formal_measurement=False,
                completed=manifest["status"] == "completed" and len(complete) == 3840,
                complete_timing_samples=len(complete),
                timings=summaries,
                phase_timings=phases,
                sampled_process_gpu_max_mib=gpu_max,
                sampled_cpu_rss_max_mib=max(s["cpu_rss_mib"] for s in self.memory),
                cpu_retained_delta_mib=self.memory[-1]["cpu_rss_mib"]
                - self.memory[0]["cpu_rss_mib"],
                transient_gpu_peak_mib=None,
                active_external_compute_seen=[
                    p for s in self.memory for p in s["active_external_compute"]
                ],
                background_memory_disclosed=True,
                no_physical_quality_filtering=True,
            ),
        )


def _worker(platform: str, output: Path) -> None:
    """Load an unchanged accepted worker and replace only its recording adapter."""
    sys.path.insert(0, str(REPO))
    from .s04_measurement import _environment

    protocol = _read(OUTPUT / "protocol.json")
    _verify(protocol)
    worker = importlib.import_module(
        "scripts.benchmark.tech_reports.physics.platforms.dexuni." + platform + ".c02"
    )
    original_record = worker.record

    def record(
        platform_name: str,
        inputs: dict,
        out: Path,
        step: Callable,
        readings: Callable,
        capture: Callable | None,
        diagnostics: Callable,
        audit: dict,
        until: float | None = None,
    ) -> None:
        """Delegate all original physical recording and acceptance checks unchanged."""
        assert capture is None and until is None
        shutil.copy2(Path(__file__), out / "source/c02_pilot.py")
        shutil.copy2(
            OUTPUT / "source/task_sources.zip", out / "source/task_sources.zip"
        )
        timers = _Timers(platform, step, readings)
        environment = _environment()
        governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        environment["cpu_governor"] = (
            governor.read_text().strip() if governor.exists() else None
        )
        try:
            original_record(
                platform_name,
                inputs,
                out,
                timers.advance,
                timers.observe,
                None,
                diagnostics,
                audit,
                None,
            )
        finally:
            timers.restore()
            timers.save(out, environment)

    worker.record = record
    sys.argv = [str(worker.__file__), "--out", str(output), "--no-video"]
    if platform == "isaaclab":
        sys.argv += [
            "--visualizer",
            "none",
            "--iterations",
            "96",
            "--substeps",
            "24",
            "--trajectory",
            str(ROOT / "platforms/isaaclab/inputs/controller_trajectory.json"),
        ]
    worker.main()


def _launch() -> None:
    """Run one fresh no-camera process for each platform, retaining every attempt."""
    protocol = _prepare()
    attempts: list[dict[str, Any]] = []
    for platform in ("embodichain", "isaaclab"):
        _verify(protocol)
        while True:
            processes = _gpu()
            activity, raw = _activity(3)
            active = [p for p in processes if activity.get(p["pid"], 0) > 0]
            _write(
                OUTPUT / (platform + "_background.json"),
                dict(processes=processes, pmon=raw, active=active),
            )
            if not active:
                break
            print("Waiting for active external GPU compute:", active, flush=True)
            time.sleep(30)
        output = OUTPUT / platform
        log = OUTPUT / (platform + ".log")
        env = os.environ.copy()
        if platform == "embodichain":
            env["PYTHONPATH"] = SDK_PATH
        else:
            env.pop("PYTHONPATH", None)
        executable = (
            REPO / ".venv/bin/python"
            if platform == "embodichain"
            else Path("/home/oem/app/IsaacLab/.venv/bin/python")
        )
        command = [
            str(executable),
            "-m",
            "scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_pilot",
            "--worker",
            platform,
            "--out",
            str(output),
        ]
        row = dict(platform=platform, command=command, status="running", log=log.name)
        attempts.append(row)
        _write(OUTPUT / "attempts.json", attempts)
        print("Starting C02 informal pilot:", platform, flush=True)
        with log.open("w") as stream:
            completed = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        row.update(
            exit_code=completed.returncode,
            status="completed" if completed.returncode == 0 else "failed",
        )
        _write(OUTPUT / "attempts.json", attempts)
        _verify(protocol)
        print("C02 informal pilot ended:", platform, completed.returncode, flush=True)
        if (output / "preview/states.npz").exists():
            commands = [
                [
                    str(REPO / ".venv/bin/python"),
                    "-m",
                    "scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_analysis",
                    "--run",
                    str(output),
                ],
                [
                    str(REPO / ".venv/bin/python"),
                    str(ROOT / "analysis/platform_port/verify_scene.py"),
                    "--run",
                    str(output),
                ],
                [
                    "/home/oem/code/engine/dexsim/.env-3.12/bin/python",
                    str(TECH / "physics/experiments/dexuni/c02_geometry.py"),
                    "--run",
                    str(output),
                ],
            ]
            row["analysis_exit_codes"] = []
            with (OUTPUT / (platform + "_analysis.log")).open("w") as stream:
                for command in commands:
                    result = subprocess.run(
                        command,
                        cwd=REPO,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    row["analysis_exit_codes"].append(result.returncode)
            _write(OUTPUT / "attempts.json", attempts)
    _verify(protocol)
    _report()


def _report() -> None:
    """Write exactly one three-table report from all retained preliminary attempts."""
    protocol = _read(OUTPUT / "protocol.json")
    _verify(protocol)
    results: dict[str, Any] = {}
    perf_rows, quality_rows, leaderboard = [], [], []
    for attempt in _read(OUTPUT / "attempts.json"):
        platform = attempt["platform"]
        out = OUTPUT / platform

        def read_optional(name: str) -> dict[str, Any]:
            """Retain an unavailable result explicitly after a setup or runtime failure."""
            path = out / name
            return _read(path) if path.exists() else {}

        summary = read_optional("summary.json")
        metrics = read_optional("analysis/current_metrics.json")
        scene = read_optional("analysis/scene_checks.json")
        geometry = read_optional("analysis/robot_geometry.json")
        manifest = read_optional("manifest.json")
        lift = metrics.get("lift", {})
        core = bool(
            summary.get("completed")
            and metrics.get("finite")
            and lift.get("completed")
            and lift.get("both_loaded_throughout_hold")
            and metrics.get("final_loaded") == [True, True]
            and lift.get("bag_bottom_min_clearance_m", 0) > 0.04
            and min(lift.get("payload_bottom_min_clearance_m", [0])) > 0.04
        )
        timers = summary.get("timings", {})
        physics = timers.get("physics_s", {})

        def fmt(value: Any) -> str:
            """Distinguish unavailable observations from numerical zero."""
            return "未测/未完成" if value is None else f"{value:.3f}"

        memory = summary.get("sampled_process_gpu_max_mib", {})
        perf_rows.append(
            f"| {platform} | {fmt(physics.get('cost_time_ms'))} | {fmt(physics.get('median_ms'))} | {fmt(physics.get('p95_ms'))} | {fmt(timers.get('outer_s', {}).get('cost_time_ms'))} | {fmt(timers.get('readback_s', {}).get('cost_time_ms'))} | {fmt(memory.get('main'))}/{fmt(memory.get('ik'))} | {fmt(summary.get('cpu_retained_delta_mib'))} | 未测 | 未测 |"
        )
        failures = [
            name for name, passed in scene.get("checks", {}).items() if not passed
        ]
        collisions = (
            f"{len(geometry.get('self_collision_samples', []))}/{len(geometry.get('table_collision_samples', []))}"
            if geometry
            else "未测"
        )
        quality_rows.append(
            f"| {platform} | {'1/1完成' if core else '0/1完整完成'} | {metrics.get('actual_seconds', '未完成')} | {summary.get('complete_timing_samples', 0)} /{physics.get('measured_steps', 0)} | {fmt(metrics.get('max_tcp_target_error_m') * 1000 if metrics.get('max_tcp_target_error_m') is not None else None)} | {fmt(lift.get('bag_bottom_min_clearance_m') * 1000 if lift.get('bag_bottom_min_clearance_m') is not None else None)} | {collisions} | {', '.join(failures) or ('全部通过' if scene else '未核验')} |"
        )
        leaderboard.append(
            f"| 不作统计排名 | {platform} | {'流程完成' if core else '本次未完整完成'} | {len(summary.get('active_external_compute_seen', []))} |"
        )
        results[platform] = dict(
            attempt=attempt,
            task_flow_completed=core,
            summary=summary,
            metrics=metrics,
            scene_checks=scene,
            robot_geometry=geometry,
            environment=manifest.get("environment", {}),
        )
    report = "\n".join(
        [
            "# C02 — 性能摸测（每平台一次完整无摄像机尝试）",
            "",
            "[64秒完整对比录像](../platforms/comparison.mp4) · [录像验证与几何限制](../platforms/REPORT.md)",
            "",
            "保持当前已录制两版的物理场景和各自机器人动作，完整推进64s/3840外步；前30步为暖机，其余3810步进入汇总。每平台仅一个新进程，不构成重复测量、成功率估计或正式排名。所有尝试均保留，不按物理质量筛选计时。",
            "",
            "## Time & Memory",
            "",
            "| 平台 | physics mean ms | median ms | p95 ms | 含反馈/IK/命令的外步mean ms | 读回mean ms | 采样GPU主/IK MiB | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
            "|---|---|---|---|---|---|---|---|---|---|",
            *perf_rows,
            "",
            "physics计时：EC为原sim.update；Isaac为原sim.step加scene.update，均在所属Warp设备同步后开始、完成后同步再停止。机器人目标上传和IK在此范围外。第二列外步耗时覆盖原反馈/IK/命令/物理闭包，包含嵌套计时钩子的同步开销；原公开状态读回另计。初始化、JIT/Graph捕获暖机、记录检查、显存采样和序列化均不进入稳态physics样本。",
            "",
            "显存为每120步在计时范围外采样的nvidia-smi进程驻留最大值，分别列主进程和独立IK进程；不代表分配增量或瞬时峰值。cpu_delta_mb为开始至末尾的采样RSS差，包含保留原始状态的记录器内存。未启用额外分配峰值分析。桌面图形和闲置计算上下文保留披露；外部活跃计算采样见memory_samples.json。",
            "",
            "## Success & Other Metrics",
            "",
            "| 平台 | 本次success_rate（描述性） | 实际秒数 | 全部计时/稳态步数 | max TCP mm | 悬停袋底最低mm | 自身/桌面碰撞样本 | 未通过的原例子严格判据 |",
            "|---|---|---|---|---|---|---|---|---|",
            *quality_rows,
            "",
            "流程完成指实际两件装袋、提袋、悬停、放回并最终保留商品；原袋口弯曲和机器人流程判据单列。Isaac采用明确记录的双手立袋/缓慢换抓，原right_wait_clear/right_relaxed_before_packing与此流程不同；lip_bow是实际未达原阈值，不能按流程差异忽略。离散FCL检查不证明连续时间无碰撞。",
            "",
            "## Leaderboard",
            "",
            "| 排名 | 平台 | 本次完整流程 | 外部活跃计算记录数 |",
            "|---|---|---|---|",
            *leaderboard,
            "",
            "本轮不做统计排名，列出全部平台。EC使用原DexUni12×6与DAT/多层/Chebyshev，Isaac使用普通VBD96×24及已录制的动作适配；这些策略/动作差异是当前可运行版本的一部分，不能称为相同迭代或相同轨迹对比。分阶段耗时在各summary.json的phase_timings，原始逐步计时在steps.csv。硬件、驱动、软件版本、线程和CPU调度配置见各manifest.json；此处不修改运行时、功耗配置或已有正式数据。",
            "",
            "输入/录像/执行源校验与范围在protocol.json；所有尝试在attempts.json。",
            "",
        ]
    )
    assert sum(line.startswith("|---") for line in report.splitlines()) == 3
    (OUTPUT / "REPORT.md").write_text(report)
    _write(
        OUTPUT / "results.json",
        dict(
            formal_measurement=False,
            protocol_sha256=_sha(OUTPUT / "protocol.json"),
            source_sha256=_sha(Path(__file__)),
            platforms=results,
            report_sha256=_sha(OUTPUT / "REPORT.md"),
        ),
    )
    print("Markdown report saved:", OUTPUT / "REPORT.md", flush=True)


def main() -> None:
    """Dispatch preliminary orchestration or one instrumented platform worker."""
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--launch", action="store_true")
    choice.add_argument("--report", action="store_true")
    choice.add_argument("--worker", choices=("embodichain", "isaaclab"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.launch:
        _launch()
    elif args.report:
        _report()
    else:
        if args.out is None:
            parser.error("--out is required for a fresh worker")
        _worker(args.worker, args.out)


if __name__ == "__main__":
    main()
