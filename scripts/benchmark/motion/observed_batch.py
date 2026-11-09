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

"""Compare fresh serial observed queries with one native batch on the same GPU.

Run: python -m scripts.benchmark.motion.observed_batch --output batch.json
Timing includes snapshot copies, construction, planning, close and one group GC.
Frozen scene capture, result verification and memory probes are reported separately.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyPhysicsCfg
from embodichain.lab.sim.objects import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain_robots import FrankaPandaCfg
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import PlanState
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
    CuroboPlannerCfg,
    CuroboPlanOptions,
    CuroboWorldCfg,
)
from embodichain.lab.sim.motion.planners.curobo.payload import (
    CuroboQuerySnapshot,
    curobo_motion_generator,
)
from embodichain.lab.sim.scene.collision_sources import snapshot_collision_sources


def main() -> None:
    """Freeze one four-environment corpus and compare equal work per batch size."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("invalid repeat/warmup counts")
    torch.set_num_threads(1)
    b = args.batch_size
    report: dict[str, Any] = {
        "batch_size": b,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "nvidia-curobo", "warp-lang", "dexsim-engine")
        },
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "fresh planners, shared static world, no payload, joint targets, graphs off; equal group GC; no physical stepping",
        "seed": 949,
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sim = None
    serial_sim = None
    was_gc = gc.isenabled()

    def save() -> None:
        """Persist completed or failed evidence."""
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        setup = time.perf_counter()
        sim = SimulationManager(
            SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=b)
        )
        robot = sim.add_robot(
            cfg=FrankaPandaCfg.from_dict({"uid": "batch949", "robot_type": "panda"})
        )
        block = sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="obstacle",
                shape=CubeCfg(size=[0.18, 0.40, 0.36]),
                attrs=RigidBodyPhysicsCfg(),
                body_type="static",
                init_pos=[0.45, 0, 0.18],
                init_rot=[0, 0, 0],
            )
        )
        sim.prepare()
        torch.cuda.synchronize()
        report["setup_s"] = time.perf_counter() - setup
        capture = time.perf_counter()
        full = robot.get_qpos().clone()
        start = robot.get_qpos(name="arm").clone()
        target = start.clone()
        target[:, 0] += torch.linspace(0.04, 0.12, b, device=start.device)
        frames = torch.linalg.inv(
            robot.get_control_part_base_pose("arm", to_matrix=True)
        )
        world = snapshot_collision_sources(
            {"obstacle": block.get_collision_shapes()},
            {"obstacle": block.get_local_pose(to_matrix=True)},
            frames,
        )
        torch.cuda.synchronize()
        report["capture_ms"] = (time.perf_counter() - capture) * 1000
        h = hashlib.sha256()
        for value in (full, start, target, frames, world["obstacle"].pose):
            h.update(value.cpu().contiguous().numpy().tobytes())
        report["corpus_digest"] = h.hexdigest()
        report["start"] = start.cpu().tolist()
        report["target"] = target.cpu().tolist()
        cfg = CuroboPlannerCfg(
            robot_uid=robot.uid,
            world=CuroboWorldCfg(rigid_objects=world),
            use_cuda_graph=False,
            preserve_plan_samples=True,
            warmup_iterations=0,
            max_attempts=2,
        )

        # The generic planner contract requires B == robot.num_instances.
        # Keep it intact: serial B=1 uses an identical one-instance robot owner,
        # while both modes consume the same frozen numeric corpus and world.
        serial_setup = time.perf_counter()
        serial_sim = SimulationManager(
            SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=1)
        )
        serial_robot = serial_sim.add_robot(
            cfg=FrankaPandaCfg.from_dict({"uid": "batch949", "robot_type": "panda"})
        )
        serial_sim.prepare()
        assert serial_robot.joint_names == robot.joint_names
        torch.testing.assert_close(serial_robot.get_qpos(), full[:1], atol=0, rtol=0)
        serial_cfg = replace(cfg, sim_instance_id=serial_sim.instance_id)
        report["serial_owner_setup_s"] = time.perf_counter() - serial_setup

        def query(lo: int, hi: int, mode: str) -> Any:
            """Plan one owned slice, dropping native handles before group collection."""
            torch.manual_seed(949)
            np.random.seed(949)
            snap = CuroboQuerySnapshot(
                full_qpos=full[lo:hi], world_to_base=frames[lo:hi]
            )
            with curobo_motion_generator(
                serial_cfg if mode == "serial" else cfg, query_snapshot=snap
            ) as mg:
                result = mg.generate(
                    [PlanState.from_qpos(target[lo:hi])],
                    MotionGenOptions(
                        start_qpos=start[lo:hi],
                        control_part="arm",
                        plan_opts=CuroboPlanOptions(control_part="arm"),
                    ),
                )
                native = next(iter(mg.planner._backend_cache.values())).planner
                assert isinstance(
                    native,
                    (
                        mg.planner._bindings.BatchMotionPlanner
                        if hi - lo > 1
                        else mg.planner._bindings.MotionPlanner
                    ),
                )
                return result

        def group(mode: str) -> list[Any]:
            """Submit equal rows serially or through one native batch call."""
            return (
                [query(i, i + 1, mode) for i in range(b)]
                if mode == "serial"
                else [query(0, b, mode)]
            )

        gc.collect()
        gc.disable()
        for iteration in range(args.warmup + args.repeats + 1):
            memory = iteration == args.warmup + args.repeats
            modes = ("serial", "batch") if iteration % 2 == 0 else ("batch", "serial")
            for mode in modes:
                gc.collect()
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated()
                if memory:
                    torch.cuda.reset_peak_memory_stats()
                t = time.perf_counter()
                results = group(mode)
                torch.cuda.synchronize()
                plan_end = time.perf_counter()
                gc.collect()
                torch.cuda.synchronize()
                end = time.perf_counter()
                row = {
                    "iteration": iteration,
                    "mode": mode,
                    "phase": (
                        "memory"
                        if memory
                        else ("warmup" if iteration < args.warmup else "steady")
                    ),
                    "group_ms": (end - t) * 1000,
                    "pre_gc_ms": (plan_end - t) * 1000,
                    "gc_ms": (end - plan_end) * 1000,
                    "allocated_after_gc_bytes": torch.cuda.memory_allocated(),
                }
                if memory:
                    row.update(
                        allocated_before_bytes=before,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                masks = torch.cat([r.success for r in results])
                row["success"] = masks.cpu().tolist()
                cursor = 0
                for r in results:
                    n = len(r.success)
                    assert (
                        torch.isfinite(r.positions).all() and torch.isfinite(r.dt).all()
                    )
                    assert (r.dt >= 0).all()
                    torch.testing.assert_close(
                        r.positions[:, 0], start[cursor : cursor + n], atol=1e-3, rtol=0
                    )
                    if r.success.any():
                        torch.testing.assert_close(
                            r.positions[r.success, -1],
                            target[cursor : cursor + n][r.success],
                            atol=0.02,
                            rtol=0,
                        )
                        assert (r.duration[r.success] > 0).all()
                    cursor += n
                row["durations_s"] = (
                    torch.cat([r.duration for r in results]).cpu().tolist()
                )
                row["sample_counts"] = [r.positions.shape[1] for r in results]
                report["rows"].append(row)
                save()
                del results, r
        report["summary"] = {}
        for mode in ("serial", "batch"):
            rows = [
                r
                for r in report["rows"]
                if r["phase"] == "steady" and r["mode"] == mode
            ]
            values = [r["group_ms"] for r in rows]
            successes = sum(sum(r["success"]) for r in rows)
            report["summary"][mode] = {
                "p50_ms": float(np.percentile(values, 50)),
                "p95_ms": float(np.percentile(values, 95)),
                "successes": successes,
                "total_rows": len(rows) * b,
                "successful_rows_per_s": 1000 * successes / sum(values),
            }
        report["speedup"] = (
            report["summary"]["batch"]["successful_rows_per_s"]
            / report["summary"]["serial"]["successful_rows_per_s"]
        )
        report["status"] = "completed"
    except BaseException as exc:
        import traceback

        report.update(
            status="failed", error=repr(exc), traceback=traceback.format_exc()
        )
        raise
    finally:
        if was_gc:
            gc.enable()
        save()
        if serial_sim is not None:
            serial_sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()
        if sim is not None:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
