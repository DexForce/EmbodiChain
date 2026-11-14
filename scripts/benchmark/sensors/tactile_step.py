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


"""Compare completed Newton steps with contact grids and indentation.

Run: python -m scripts.benchmark.sensors.tactile_step --device cuda:0
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import dexsim
import torch
import warp as wp

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg, RenderCfg, RigidObjectCfg
from embodichain.lab.sim.sensors import ContactGridCfg, SensorCfg, TactileIndentationCfg
from embodichain.lab.sim.shapes import CubeCfg


def _benchmark_variant(
    envs: int,
    fraction: float,
    variant: str,
    device: torch.device,
    repeats: int,
    warmup: int,
    resolution: int,
) -> tuple[dict[str, object], list[torch.Tensor]]:
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            startup_summary="off",
            num_envs=envs,
            arena_space=1.0,
            physics_cfg=NewtonPhysicsCfg(
                device=device,
                gravity=(0.0, 0.0, 0.0),
                num_substeps=1,
                use_cuda_graph=False,
                solver_cfg={"solver_type": "mujoco_warp"},
            ),
            render_cfg=RenderCfg(renderer="no-render"),
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="pad",
                shape=CubeCfg(size=(0.06, 0.06, 0.02)),
                body_type="static",
                init_pos=(0.0, 0.0, 0.49),
            )
        )
        tool = sim.add_rigid_object(
            RigidObjectCfg(
                uid="tool",
                shape=CubeCfg(size=(0.04, 0.04, 0.04)),
                init_pos=(0.0, 0.0, 0.515),
            )
        )
        sim.prepare()
        poses = tool.get_local_pose()
        active = round(envs * fraction)
        poses[active:, 0] = 0.4
        velocities = torch.zeros((envs, 3), device=device)
        alternate = poses.clone()
        alternate[:active, 0] += 0.001
        grid = None
        indent = None
        if variant in ("grid", "both-dense", "both-screened"):
            grid = sim.add_sensor(
                ContactGridCfg(
                    uid="contact_grid",
                    asset_uid="pad",
                    mount=SensorCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
                    size=(0.06, 0.06),
                    resolution=(8, 8),
                    max_plane_distance=0.015,
                )
            )
        if variant in ("dense", "screened", "both-dense", "both-screened"):
            indent = sim.add_sensor(
                TactileIndentationCfg(
                    uid="indentation",
                    target_uids=("tool",),
                    mount=SensorCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
                    size=(0.06, 0.06),
                    resolution=(resolution, resolution),
                    curvature_radius=0.10,
                    max_depth=0.03,
                    use_candidates=variant in ("screened", "both-screened"),
                )
            )

        def step(pose: torch.Tensor) -> None:
            tool.set_local_pose(pose)
            tool.set_velocity(velocities, velocities)
            sim.update(step=1, render_final_step=False)

        sync: Callable[[], None] = (
            (lambda: torch.cuda.synchronize(device))
            if device.type == "cuda"
            else (lambda: None)
        )
        trajectory = (poses, alternate)
        for i in range(warmup):
            step(trajectory[i % 2])
        sync()
        latencies = []
        for i in range(repeats):
            start = perf_counter()
            step(trajectory[i % 2])
            sync()
            latencies.append((perf_counter() - start) * 1000)
        # Measure sustained completed work separately from synchronized latency.
        start = perf_counter()
        for i in range(repeats):
            step(trajectory[i % 2])
        sync()
        throughput = envs * repeats / (perf_counter() - start)

        torch_peak = None
        if device.type == "cuda":
            before = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            for i in range(repeats):
                step(trajectory[i % 2])
            sync()
            torch_peak = (torch.cuda.max_memory_allocated(device) - before) / 2**20

        results = []
        observed = {}
        if grid is not None:
            rows = grid.get_data()
            counts = rows["count"].sum(dim=(1, 2))
            observed["contact_rows"] = int(counts.sum())
            observed["incomplete_contact_rows"] = int((~rows["complete"]).sum())
            if int((counts > 0).sum()) != active or observed["incomplete_contact_rows"]:
                raise RuntimeError(
                    "The contact grid fixture lost or misattributed rows."
                )
        if indent is not None:
            rows = indent.get_data()
            if int(rows["hit"].flatten(1).any(dim=-1).sum()) != active:
                raise RuntimeError("The indentation fixture found a wrong contact row.")
            results = [rows[name].cpu() for name in ("depth", "shape_id", "hit")]
            observed["active_depth_rows"] = active
        return (
            {
                "environments": envs,
                "active_fraction": active / envs,
                "path": variant,
                "depth_rays_per_environment": resolution * resolution,
                "p50_ms": round(median(latencies), 5),
                "p95_ms": round(sorted(latencies)[math.ceil(repeats * 0.95) - 1], 5),
                "environments_per_second": round(throughput, 1),
                "peak_torch_transient_mib": torch_peak,
                **observed,
            },
            results,
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Run the representative scene variants and print their qualified results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 512, 2048])
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.0, 0.05, 0.25, 1.0]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.envs, args.warmup, args.repeats, args.resolution) <= 0:
        parser.error("All sizes and repetition counts must be positive.")
    if not all(0 <= value <= 1 for value in args.fractions):
        parser.error("Active fractions must be in [0, 1].")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; tactile step benchmark skipped.")
        return
    rows = []
    for count in args.envs:
        # Fractions that round to the same scene do not add independent evidence.
        active_counts = sorted({round(count * fraction) for fraction in args.fractions})
        for active in active_counts:
            dense = None
            for variant in (
                "off",
                "grid",
                "dense",
                "screened",
                "both-dense",
                "both-screened",
            ):
                result, observed = _benchmark_variant(
                    count,
                    active / count,
                    variant,
                    device,
                    args.repeats,
                    args.warmup,
                    args.resolution,
                )
                if variant == "dense":
                    dense = observed
                if variant in ("screened", "both-dense", "both-screened"):
                    for first, second in zip(observed, dense, strict=True):
                        torch.testing.assert_close(first, second, atol=1e-6, rtol=0.0)
                    result["matches_dense"] = True
                rows.append(result)
                print(json.dumps(result), flush=True)

    report = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "dexsim": dexsim.__version__,
        "torch": torch.__version__,
        "warp": wp.__version__,
        "solver": "Newton MuJoCo Warp, no-render, one substep, graph disabled",
        "warmup": args.warmup,
        "repeats": args.repeats,
        "scope": "Pose and velocity writes, completed SimulationManager.update, and sensor acquisition. Torch transient peak excludes persistent outputs and native/driver allocations.",
        "results": rows,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
