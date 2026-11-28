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
"""Compare DexUni free-rigid piles at fixed scene and batch sizes.

Run: python -m scripts.benchmark.robotics.dexuni.piles mj 16 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import dexsim
import numpy as np
import warp as wp
from dexsim.engine.newton_physics import DexUniSolverCfg, NewtonCfg


def add_pusher(builder: dexsim.spawn.SceneBuilder, columns: int) -> None:
    """Add a force-limited prismatic blade at the near face of the pile."""
    spawn = dexsim.spawn
    origin = np.eye(4, dtype=np.float32)
    offset = origin.copy()
    offset[:3, 3] = (-0.07, (columns - 1) * 0.053 / 2, 0.04)
    links = [
        spawn.LinkDesc(
            "base", "", origin, rigid_body=spawn.RigidBodyPhysicsDesc.static()
        ),
        spawn.LinkDesc(
            "blade",
            "base",
            offset,
            rigid_body=spawn.RigidBodyPhysicsDesc.dynamic(mass=0.5),
            collisions=[
                spawn.CollisionDesc(
                    geometry_type=spawn.GeometryType.CUBE,
                    size=(0.015, columns * 0.053, 0.065),
                )
            ],
        ),
    ]
    joint = spawn.JointDesc(
        "slide",
        "base",
        "blade",
        dexsim.engine.JointType.PRISMATIC,
        origin_pose=offset,
        axis=np.array([1.0, 0.0, 0.0]),
        lower_limit=0.0,
        upper_limit=0.15,
        newton=spawn.NewtonJointDesc(
            target_ke=400.0, target_kd=20.0, armature=0.1, effort_limit=20.0
        ),
    )
    builder.add_articulation(
        spawn.ArticulationDesc(
            name="pusher",
            pose=origin,
            links=links,
            joints=[joint],
            fixed_base=True,
            per_env=True,
        )
    )


def run(
    mode: str,
    count: int,
    batch: int,
    out: Path,
    push: bool = False,
    iterations: int | None = None,
    cg_sparse: bool = False,
) -> None:
    """Drop four-layer columns, then measure drift and completed step times."""
    dt = 0.01
    cfg = dexsim.WorldConfig()
    cfg.open_windows = False
    cfg.init_ground = False
    cfg.renderer = dexsim.types.Renderer.NORENDER
    # Explicit empty ownership leaves all free trees on the VBD side.
    solver = DexUniSolverCfg(mujoco_articulations=() if mode == "vbd" else None)
    if cg_sparse:
        if mode != "mj":
            raise ValueError("The sparse CG profile uses the MuJoCo branch")
        solver.mujoco_options.update(solver="cg", jacobian="sparse")
    if iterations is not None:
        if mode != "vbd" or iterations < 1:
            raise ValueError(
                "An AVBD iteration sweep requires vbd and positive iterations"
            )
        solver.vbd_options["iterations"] = iterations
    if mode in ("one_way", "two_way"):
        if not push:
            raise ValueError("A coupling comparison needs a pusher")
        solver.coupling = mode
    cfg.newton_cfg = NewtonCfg(
        dt=dt,
        num_substeps=4,
        device="cuda:0",
        solver_cfg=solver,
        collision_pipeline_cfg=None,
        sync_to_dexsim=False,
    )
    cfg.newton_cfg.use_cuda_graph = True
    world = dexsim.World(cfg)
    world.set_manual_update(True)
    builder = dexsim.spawn.SceneBuilder(world)
    builder.replicate(batch, spacing=(2.0, 2.0, 0.0))
    builder.add_plane(
        "floor", 20.0, per_env=False, physics=dexsim.spawn.RigidBodyPhysicsDesc.static()
    )
    cols = math.ceil(math.sqrt(count / 4))
    rng = np.random.default_rng(19)
    for i in range(count):
        col, level = divmod(i, 4)
        x, y = divmod(col, cols)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = (
            x * 0.053 + rng.uniform(-0.0003, 0.0003),
            y * 0.053 + rng.uniform(-0.0003, 0.0003),
            0.025 + level * 0.052 + 0.025,
        )
        builder.add_cube(
            f"box{i}",
            0.05,
            pose=pose,
            physics=dexsim.spawn.RigidBodyPhysicsDesc.dynamic(mass=0.065),
        )
    if push:
        add_pusher(builder, cols)
    scene = builder.finalize()
    try:
        handles = [
            scene.get_object(f"env_{r}/box{i}")
            for r in range(batch)
            for i in range(count)
        ]

        def state():
            return np.asarray([h.get_world_pose() for h in handles]).reshape(
                batch, count, 4, 4
            )

        def advance(n):
            for _ in range(n):
                world.update(dt, sync_to_dexsim=False)
            wp.synchronize_device("cuda:0")

        advance(130)
        before = state()
        if push:
            blades = [scene.get_articulation(f"env_{r}/pusher") for r in range(batch)]
            for blade in blades:
                blade.set_target_qpos([0.12])
        times = []
        for _ in range(16):
            start = time.perf_counter()
            advance(10)
            times.append((time.perf_counter() - start) / 10)
        after = state()
        diag = scene.get_solver_diagnostics()
        assert diag["features"]["backend"] == (
            "pure_vbd"
            if mode == "vbd"
            else "coupled" if mode in ("one_way", "two_way") else "pure_mujoco"
        )
        assert diag["cuda_graph_status"] == "captured"
        extent = 0.025 * np.abs(after[:, :, 2, :3]).sum(axis=-1)
        z = after[:, :, 2, 3]
        bottom = z - extent
        top = z + extent
        # Nominal neighbors are axis-aligned. Vertical AABB overlap is a
        # conservative proxy when tilted; also record tilt and displacement.
        layers = np.arange(1, count) % 4 != 0
        penetration = np.maximum(0.0, top[:, :-1][:, layers] - bottom[:, 1:][:, layers])
        drift = np.linalg.norm(after[:, :, :3, 3] - before[:, :, :3, 3], axis=-1)
        tilt = np.degrees(np.arccos(np.clip(after[:, :, 2, 2], -1, 1)))
        quality = {
            "max_floor_penetration_mm": float(1000 * max(0.0, -bottom.min())),
            "max_adjacent_z_overlap_mm": float(1000 * penetration.max()),
            "max_drift_mm": float(1000 * drift.max()),
            "max_tilt_deg": float(tilt.max()),
            "finite": bool(np.isfinite(after).all()),
        }
        memory = None
        try:
            import pynvml as nv

            nv.nvmlInit()
            handle = nv.nvmlDeviceGetHandleByIndex(0)
            samples = []
            for _ in range(16):
                advance(1)
                procs = nv.nvmlDeviceGetComputeRunningProcesses(handle)
                samples.append(
                    next(
                        proc.usedGpuMemory / 2**20
                        for proc in procs
                        if proc.pid == os.getpid()
                    )
                )
            memory = max(samples)
            nv.nvmlShutdown()
        except (ImportError, StopIteration):
            pass
        query = scene.create_contact_query(
            handles, match="any", device="cuda:0", capacity=32 * count * batch
        )
        advance(1)
        count_active, dropped = query.fetch_solver_report_async().read_counts()
        result = dict(
            snapshot_contact_count=count_active,
            snapshot_contact_dropped=dropped,
            iterations=iterations,
            cg_sparse=cg_sparse,
            count=count,
            batch=batch,
            mode=mode,
            seconds=times,
            quality=quality,
            process_gpu_resident_MiB=memory,
            memory_scope="maximum NVML process residency over 16 completed warmed steps",
            diagnostics=diag,
        )
        if push:
            result["blade_travel_m"] = [
                float(blade.get_current_qpos()[0]) for blade in blades
            ]
            result["mean_payload_dx_m"] = float(
                (after[:, :, 0, 3] - before[:, :, 0, 3]).mean()
            )
        out.mkdir(parents=True, exist_ok=True)
        suffix = "" if iterations is None else f"_iter{iterations}"
        if cg_sparse:
            suffix = "_cg_sparse"
        (
            out / f"{'push' if push else 'pile'}_{mode}_{count}_{batch}{suffix}.json"
        ).write_text(json.dumps(result, default=str))
        print(result["quality"], flush=True)
    finally:
        scene.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("mj", "vbd", "one_way", "two_way"))
    parser.add_argument("count", type=int)
    parser.add_argument("batch", type=int)
    parser.add_argument("--out", type=Path, default=Path("outputs/dexuni"))
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--cg-sparse", action="store_true")
    opts = parser.parse_args()
    run(
        opts.mode,
        opts.count,
        opts.batch,
        opts.out,
        opts.push,
        opts.iterations,
        opts.cg_sparse,
    )
