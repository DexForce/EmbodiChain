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
"""Profile the unchanged public L-02 loading path in a fresh process.

Run: python -m scripts.benchmark.tech_reports.loading.experiments.l02_profile \
    --config <existing-config.json> --output <new-directory> --profile
"""

from __future__ import annotations

import argparse
import cProfile
import faulthandler
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import pstats
import signal
import subprocess
import sys
import textwrap
import time

import psutil

__all__ = ["main"]


def main() -> None:
    """Save per-stage profiles, actual configuration and backend timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-render", action="store_true")
    parser.add_argument("--native-sample-interval", type=float, default=0)
    parser.add_argument("--omit-duplicate-material", action="store_true")
    parser.add_argument("--defer-camera", action="store_true")
    parser.add_argument("--frame", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "profiler-source.py").write_bytes(Path(__file__).read_bytes())
    cfg = json.loads(args.config.read_text())
    if args.count is not None:
        assert 0 < args.count <= cfg["object_count"]
        cfg["object_count"] = args.count
        cfg["objects"] = cfg["objects"][: args.count]
        cfg["positions"] = cfg["positions"][: args.count]
    cfg.update(output=str(out), purpose="profiling_diagnostic", record=False)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    gpu_command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv",
    ]
    meta = {
        "argv": sys.argv,
        "python": sys.executable,
        "pid": os.getpid(),
        "started_epoch": time.time(),
        "source_config": str(args.config.resolve()),
        "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "profile": args.profile,
        "trace_render": args.trace_render,
        "native_sample_interval": args.native_sample_interval,
        "omit_duplicate_material": args.omit_duplicate_material,
        "defer_camera": args.defer_camera,
        "scope": "Diagnostic; initialization, no cache purge; not an uncontended performance comparison",
        "environment": {
            key: os.environ.get(key)
            for key in [
                "PYTHONPATH",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "LD_PRELOAD",
                "L02_HYBRID_LIBRARY",
                "L02_NATIVE_COUNTS",
            ]
        },
        "gpu_before": subprocess.check_output(gpu_command, text=True),
        "stages": {},
        "status": "running",
    }
    for name in [
        "embodichain",
        "embodichain_robots",
        "embodichain_tasks",
        "dexsim",
        "newton",
        "warp",
    ]:
        spec = importlib.util.find_spec(name)
        meta.setdefault("module_resolution", {})[name] = spec.origin if spec else None
    meta["versions"] = {
        name: importlib.metadata.version(name)
        for name in ["dexsim-engine", "newton", "warp-lang", "torch", "numpy"]
    }
    meta["embodichain_commit"] = subprocess.check_output(
        ["git", "-C", "/home/oem/code/EmbodiChain", "rev-parse", "HEAD"], text=True
    ).strip()
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)
    from scripts.benchmark.tech_reports.loading.platforms.embodichain.l02_loading import (
        Simulation,
    )
    import numpy as np
    import torch
    import warp as wp

    if args.defer_camera:
        # Diagnostic only: include the same camera creation after prepare, still
        # before the prepared mark/first step/first frame. No render omission.
        from scripts.benchmark.tech_reports.loading.platforms.embodichain import (
            l02_loading,
        )

        original = textwrap.dedent(inspect.getsource(Simulation.__init__))
        start = original.index('    c = cfg["camera"]')
        end = original.index('    mark("declared")', start)
        camera = original[start:end]
        modified = original[:start] + original[end:]
        needle = "    self.sim.prepare()\n"
        assert modified.count(needle) == 1
        modified = modified.replace(needle, needle + camera)
        (out / "diagnostic_simulation_init.py").write_text(modified)
        namespace = dict(l02_loading.__dict__)
        exec(
            compile(modified, str(out / "diagnostic_simulation_init.py"), "exec"),
            namespace,
        )
        Simulation.__init__ = namespace["__init__"]

    if args.omit_duplicate_material:
        # Diagnostic only: generated meshes already receive this exact material
        # in add_mesh_to_render_body before attachment. Keep file imports intact.
        from dexsim.spawn.adapters import common, newton_rigid_adapter

        original = inspect.getsource(common.create_render_actor)
        needle = "if material is not None and not isinstance(material, MaterialDesc):"
        assert original.count(needle) == 1
        replacement = (
            "if (file_geom is not None and file_geom.file_path is not None "
            "and material is not None and not isinstance(material, MaterialDesc)):"
        )
        modified = original.replace(needle, replacement)
        (out / "diagnostic_create_render_actor.py").write_text(modified)
        exec(
            compile(modified, str(out / "diagnostic_create_render_actor.py"), "exec"),
            common.__dict__,
        )
        newton_rigid_adapter.create_render_actor = common.create_render_actor
    if args.native_sample_interval:
        signal.signal(signal.SIGALRM, lambda *_: None)
        signal.setitimer(
            signal.ITIMER_REAL, args.native_sample_interval, args.native_sample_interval
        )

    line_times: dict[int, list[float | int]] = {}
    render_calls: list[dict[str, object]] = []
    if args.trace_render:
        from dexsim.spawn.adapters.common import create_render_actor

        target = create_render_actor.__code__
        active: dict[int, tuple[int, float, float]] = {}

        def trace_render(frame, event, arg):
            """Time only create_render_actor lines, including native waits."""
            if frame.f_code is not target:
                return None
            now = time.perf_counter()
            key = id(frame)
            if event == "call":
                active[key] = (frame.f_lineno, now, now)
            else:
                line, previous_time, started = active[key]
                row = line_times.setdefault(line, [0, 0.0])
                row[0] += 1
                row[1] += now - previous_time
                active[key] = (frame.f_lineno, now, started)
                if event == "return":
                    render_calls.append(
                        {
                            "name": frame.f_locals["desc"].name,
                            "elapsed_s": now - started,
                        }
                    )
                    del active[key]
                    if len(render_calls) % 100 == 0:
                        (out / "render-lines.json").write_text(
                            json.dumps(line_times, indent=2) + "\n"
                        )
                        (out / "render-calls.json").write_text(
                            json.dumps(render_calls, indent=2) + "\n"
                        )
                        print(
                            "RENDER_PROGRESS",
                            len(render_calls),
                            sum(x["elapsed_s"] for x in render_calls[-100:]),
                            flush=True,
                        )
            return trace_render

        sys.settrace(trace_render)

    profiler = cProfile.Profile() if args.profile else None
    previous = time.perf_counter()

    def mark(stage: str) -> None:
        """Persist each completed stage, excluding profile serialization."""
        nonlocal previous
        elapsed = time.perf_counter() - previous
        if profiler:
            profiler.disable()
            profiler.dump_stats(str(out / f"{stage}.pstats"))
            with (out / f"{stage}.txt").open("w") as stream:
                stats = pstats.Stats(profiler, stream=stream)
                stats.sort_stats("cumulative").print_stats(65)
                stats.sort_stats("tottime").print_stats(35)
            profiler.clear()
        meta["stages"][stage] = {
            "elapsed_s": elapsed,
            "rss_mib": psutil.Process().memory_info().rss / 2**20,
            "epoch": time.time(),
        }
        (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("PROFILE_STAGE", stage, elapsed, flush=True)
        previous = time.perf_counter()
        if profiler:
            profiler.enable()

    if profiler:
        profiler.enable()
    try:
        simulation = Simulation(cfg, mark)
        state = simulation.step()
        wp.synchronize()
        torch.cuda.synchronize()
        mark("first_step")
        assert state.shape == (cfg["object_count"], 13)
        assert np.isfinite(state).all()
        np.save(out / "first_state.npy", state)
        meta["position_error_max_m"] = float(
            np.linalg.norm(state[:, :3] - np.asarray(cfg["positions"]), axis=1).max()
        )
        from dexsim.engine.newton_physics.backend_registry import get_newton_backend

        backend = get_newton_backend(simulation.sim._world)
        meta["backend_timings_ms"] = backend.initialization_timings_ms
        meta["body_count"] = int(backend.model.body_count)
        meta["shape_count"] = int(backend.model.shape_count)
        if args.frame:
            rgb = simulation.render()
            mark("first_frame")
            from PIL import Image

            Image.fromarray(rgb).save(out / "first_frame.png")
        if args.audit:
            arrays, audit = simulation.audit()
            np.savez(out / "model.npz", **arrays)
            (out / "model-audit.json").write_text(
                json.dumps(audit, indent=2, default=str) + "\n"
            )
        meta["status"] = "completed"
    except BaseException as error:
        meta["status"] = "failed"
        meta["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        sys.settrace(None)
        if args.trace_render:
            (out / "render-lines.json").write_text(
                json.dumps(line_times, indent=2) + "\n"
            )
            (out / "render-calls.json").write_text(
                json.dumps(render_calls, indent=2) + "\n"
            )
        if profiler:
            profiler.disable()
        faulthandler.cancel_dump_traceback_later()
        meta["gpu_after"] = subprocess.check_output(gpu_command, text=True)
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        rows = list(meta["stages"].items())
        report = [
            "# L-02 加载诊断摘要",
            "",
            meta["scope"],
            "",
            "## Time & Memory",
            "",
            "| stage | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | RSS MiB（边界采样） |",
            "| --- | ---: | --- | --- | --- | ---: |",
        ]
        report += [
            f"| {stage} | {value['elapsed_s'] * 1000:.3f} | 未独立测量 | 未独立测量 | 未独立测量 | {value['rss_mib']:.2f} |"
            for stage, value in rows
        ]
        report += [
            "",
            "## Success & Other Metrics",
            "",
            "| 配置 | success_rate | 状态 | 最大位置误差 m |",
            "| --- | --- | --- | ---: |",
            f"| {cfg['shape']} {cfg['asset_mode']} {cfg['object_count']} | 不适用：单次诊断 | {meta['status']} | {meta.get('position_error_max_m', '未测')} |",
            "",
            "## Leaderboard",
            "",
            "| 实现 | 排名 | 说明 |",
            "| --- | --- | --- |",
            "| EmbodiChain 公共 L-02 路径 | 不排名 | 含 profiling 或共享机器噪声，仅用于定位 |",
            "",
        ]
        (out / "benchmark-summary.md").write_text("\n".join(report))
        print("Diagnostic report:", out / "benchmark-summary.md", flush=True)


if __name__ == "__main__":
    main()
