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

"""Measure fresh B=1 observed cuRobo queries on a frozen synthetic scene.

Run: python -m scripts.benchmark.motion.observed_queries --output report.json
Requires PR #935 and native cuRobo. No simulator stepping or physical certificate.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
from typing import Any

import numpy as np
import torch

from embodichain.compute.geometry.mesh_bounds import aabb_sphere_cover
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyPhysicsCfg
from embodichain.lab.sim.objects import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain_robots import FrankaPandaCfg
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import PlanState, MoveType
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
    CuroboPlannerCfg,
    CuroboWorldCfg,
    CuroboPlanOptions,
)
from embodichain.lab.sim.motion.planners.curobo.payload import (
    CuroboQuerySnapshot,
    CuroboPayload,
    curobo_motion_generator,
)
from embodichain.lab.sim.scene.collision_sources import snapshot_collision_sources


def measured(call: Callable[[], Any]) -> tuple[Any, float]:
    """Time completed work on the sole visible planning GPU."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def digest(*values: torch.Tensor) -> str:
    """Fingerprint tensor shape, dtype and bytes for frozen inputs/results."""
    h = hashlib.sha256()
    for value in values:
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    """Run serial fresh queries, check results, and checkpoint raw measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--sphere-count", type=int, choices=(1, 27), default=27)
    parser.add_argument("--payload", choices=("none", "both", "only"), default="both")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    torch.set_num_threads(1)
    torch.manual_seed(949)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    versions = {}
    for name in ("torch", "nvidia-curobo", "warp-lang", "dexsim-engine"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unknown"
    report: dict[str, Any] = {
        "versions": versions,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "seed": 949,
        "threads": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "fresh B=1, static box, joint target +0.12 rad, no graphs, native samples; no physical execution",
        "memory_scope": "separate warmed queries; Torch allocator only, excludes native allocations",
        "rows": [],
    }

    def save() -> None:
        """Keep partial evidence if a later native query fails."""
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    sim = None
    try:
        setup_start = time.perf_counter()
        sim = SimulationManager(
            SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=1)
        )
        robot = sim.add_robot(
            cfg=FrankaPandaCfg.from_dict({"uid": "baseline949", "robot_type": "panda"})
        )
        block = sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="obstacle949",
                shape=CubeCfg(size=[0.18, 0.40, 0.36]),
                attrs=RigidBodyPhysicsCfg(),
                body_type="static",
                init_pos=[0.45, 0.0, 0.18],
                init_rot=[0.0, 0.0, 0.0],
            )
        )
        sim.prepare()
        torch.cuda.synchronize()
        report["simulation_setup_ms"] = (time.perf_counter() - setup_start) * 1000
        full = robot.get_qpos().clone()
        start = robot.get_qpos(name="arm").clone()
        target = start.clone()
        target[:, 0] += 0.12
        frame = torch.linalg.inv(
            robot.get_control_part_base_pose("arm", to_matrix=True)
        )
        object_pose = robot.compute_fk(qpos=start, name="arm", to_matrix=True)
        spheres = torch.tensor([[0.0, 0.0, 0.0, 0.015]], device=start.device)
        if args.sphere_count == 27:
            axis = torch.tensor([-0.015, 0.015], device=start.device)
            spheres = aabb_sphere_cover(torch.cartesian_prod(axis, axis, axis))
        report["payload_sphere_count"] = args.sphere_count
        report["input_digest"] = digest(
            full,
            start,
            target,
            frame,
            object_pose,
            spheres,
            block.get_local_pose(to_matrix=True),
        )
        report["full_qpos"] = full.cpu().tolist()
        report["target_qpos"] = target.cpu().tolist()
        historical: list[tuple[torch.Tensor, torch.Tensor]] = []
        reference: dict[bool, tuple[torch.Tensor, torch.Tensor]] = {}
        # Alternate A/B each round, testing result lifetime across fresh backends.
        for round_id in range(args.warmup + args.repeats + 1):
            for payload_enabled in {
                "none": (False,),
                "both": (False, True),
                "only": (True,),
            }[args.payload]:
                memory_run = round_id == args.warmup + args.repeats
                row: dict[str, Any] = {
                    "round": round_id,
                    "payload": payload_enabled,
                    "phase": (
                        "memory"
                        if memory_run
                        else ("warmup" if round_id < args.warmup else "steady")
                    ),
                }
                report["rows"].append(row)
                torch.manual_seed(949)
                np.random.seed(949)

                def capture() -> tuple[dict, CuroboQuerySnapshot]:
                    """Capture owned inputs without stepping or changing the scene."""
                    world = snapshot_collision_sources(
                        {"obstacle": block.get_collision_shapes()},
                        {"obstacle": block.get_local_pose(to_matrix=True)},
                        frame,
                    )
                    payload = (
                        CuroboPayload(
                            object_id="carried",
                            control_part="arm",
                            spheres=spheres,
                            object_pose=object_pose,
                        )
                        if payload_enabled
                        else None
                    )
                    return world, CuroboQuerySnapshot(
                        full_qpos=full, world_to_base=frame, payload=payload
                    )

                if memory_run:
                    torch.cuda.reset_peak_memory_stats()
                    row["allocated_before_bytes"] = torch.cuda.memory_allocated()
                total_start = time.perf_counter()
                (world, snapshot), row["capture_ms"] = measured(capture)
                cfg = CuroboPlannerCfg(
                    robot_uid=robot.uid,
                    use_cuda_graph=False,
                    preserve_plan_samples=True,
                    warmup_iterations=0,
                    world=CuroboWorldCfg(rigid_objects=world),
                    interpolation_dt=0.04,
                )
                scope = curobo_motion_generator(cfg, query_snapshot=snapshot)
                mg = None
                try:
                    mg, row["facade_ms"] = measured(scope.__enter__)
                    _, row["backend_prepare_including_attach_ms"] = measured(
                        lambda: mg.planner.prepare_backend(
                            control_part="arm",
                            batch_size=1,
                            move_type=MoveType.JOINT_MOVE,
                        )
                    )
                    result, row["plan_ms"] = measured(
                        lambda: mg.generate(
                            [PlanState.from_qpos(target)],
                            MotionGenOptions(
                                start_qpos=start,
                                control_part="arm",
                                plan_opts=CuroboPlanOptions(control_part="arm"),
                            ),
                        )
                    )
                    row["success"] = bool(result.success.all())
                    row["samples"] = (
                        0 if result.positions is None else result.positions.shape[1]
                    )
                    if row["success"]:
                        assert result.positions is not None
                        assert (
                            torch.isfinite(result.positions).all()
                            and torch.isfinite(result.dt).all()
                        )
                        assert torch.all(result.dt >= 0) and torch.all(
                            result.duration > 0
                        )
                        row["start_error_rad"] = float(
                            (result.positions[:, 0] - start).abs().max()
                        )
                        row["goal_error_rad"] = float(
                            (result.positions[:, -1] - target).abs().max()
                        )
                        assert (
                            row["start_error_rad"] < 1e-3
                            and row["goal_error_rad"] < 0.02
                        )
                        row["trajectory_digest"] = digest(result.positions, result.dt)
                        if payload_enabled:
                            try:
                                mg.planner.validate_joint_trajectory(
                                    result.positions, control_part="arm"
                                )
                            except NotImplementedError:
                                row["arbitrary_checker"] = "unavailable_as_required"
                            else:
                                raise AssertionError("payload checker must fail closed")
                        else:
                            valid, row["validation_ms"] = measured(
                                lambda: mg.planner.validate_joint_trajectory(
                                    result.positions, control_part="arm"
                                )
                            )
                            row["all_samples_valid"] = bool(valid.all())
                            assert row["all_samples_valid"]
                        old = reference.get(payload_enabled)
                        if old is not None:
                            row["repeat_same_shape"] = (
                                old[0].shape == result.positions.shape
                            )
                            row["repeat_exact"] = torch.equal(
                                old[0], result.positions
                            ) and torch.equal(old[1], result.dt)
                        else:
                            reference[payload_enabled] = (
                                result.positions.clone(),
                                result.dt.clone(),
                            )
                        historical.append((result.positions, result.positions.clone()))
                finally:
                    if mg is not None:
                        _, row["close_ms"] = measured(
                            lambda: scope.__exit__(None, None, None)
                        )
                row["total_with_validation_ms"] = (
                    time.perf_counter() - total_start
                ) * 1000
                row["query_ms"] = sum(
                    row[key]
                    for key in (
                        "capture_ms",
                        "facade_ms",
                        "backend_prepare_including_attach_ms",
                        "plan_ms",
                        "close_ms",
                    )
                )
                if memory_run:
                    row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    row["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
                assert torch.equal(
                    robot.get_qpos(), full
                ), "planning changed observed robot state"
                assert all(
                    torch.equal(a, b) for a, b in historical
                ), "closed query result was modified"
                row["ownership_and_state_passed"] = True
                row["allocated_after_close_bytes"] = torch.cuda.memory_allocated()
                save()
                print("BASELINE949 " + json.dumps(row), flush=True)
        summary = {}
        for enabled in {"none": (False,), "both": (False, True), "only": (True,)}[
            args.payload
        ]:
            rows = [
                r
                for r in report["rows"]
                if r["phase"] == "steady" and r["payload"] == enabled
            ]
            summary[str(enabled)] = {
                "count": len(rows),
                "successes": sum(r["success"] for r in rows),
            }
            for key in (
                "capture_ms",
                "facade_ms",
                "backend_prepare_including_attach_ms",
                "plan_ms",
                "close_ms",
                "query_ms",
                "total_with_validation_ms",
            ):
                values = [r[key] for r in rows]
                summary[str(enabled)][key] = {
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                }
            summary[str(enabled)]["reciprocal_query_stage_sum_per_second"] = (
                1000 * len(rows) / sum(r["query_ms"] for r in rows)
            )
        report["summary"] = summary
        report["retained_trajectory_storage_bytes"] = sum(
            a.untyped_storage().nbytes() + b.untyped_storage().nbytes()
            for a, b in historical
        )
        report["allocated_before_gc_bytes"] = torch.cuda.memory_allocated()
        del mg, scope, result
        report["gc_collected_objects"] = gc.collect()
        torch.cuda.synchronize()
        report["allocated_after_gc_bytes"] = torch.cuda.memory_allocated()
        report["status"] = "completed"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save()
        if sim is not None:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
