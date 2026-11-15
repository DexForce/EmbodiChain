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


"""Measure completed Newton steps with hydroelastic export and pad projection.

Run: python -m scripts.benchmark.sensors.hydroelastic_step
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter

import dexsim
import torch
import trimesh
import warp as wp
from dexsim.scene import Scene
from newton.geometry import HydroelasticSDF

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    NewtonCollisionPipelineCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import SensorCfg, TactileIndentationCfg
from embodichain.lab.sim.shapes import MeshCfg, MeshCollisionCfg


def _measure(
    device: torch.device, envs: int, variant: str, warmup: int, repeats: int
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=envs,
            arena_space=0.5,
            startup_summary="off",
            render_cfg=RenderCfg(renderer="no-render"),
            physics_cfg=NewtonPhysicsCfg(
                device=device,
                gravity=(0.0, 0.0, 0.0),
                num_substeps=1,
                use_cuda_graph=False,
                solver_cfg={
                    "solver_type": "mujoco_warp",
                    "use_mujoco_contacts": False,
                    "nconmax": 256,
                },
                collision_cfg=NewtonCollisionPipelineCfg(
                    max_triangle_pairs=8192 * envs,
                    sdf_hydroelastic_config=HydroelasticSDF.Config(
                        output_contact_surface=variant != "geometric",
                        pre_prune_contacts=True,
                    ),
                ),
            ),
        )
    )
    try:
        geometry = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        for uid, kind, z in (("pad", "static", 0.5), ("tool", "dynamic", 0.58)):
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=uid,
                    body_type=kind,
                    init_pos=(0.0, 0.0, z),
                    shape=MeshCfg(
                        vertices=geometry.vertices,
                        triangles=geometry.faces,
                        collision=MeshCollisionCfg(
                            approximation="sdf", sdf_resolution=32, is_hydroelastic=True
                        ),
                    ),
                )
            )
        sim.prepare()
        tool = sim.get_rigid_object("tool")
        sensor = sim.add_sensor(
            TactileIndentationCfg(
                uid="tactile",
                target_uids=("tool",),
                size=(0.1, 0.1),
                resolution=(16, 16),
                mount=SensorCfg.OffsetCfg(parent="pad", pos=(0.0, 0.0, 0.05)),
                rgb=TactileIndentationCfg.RGBCfg(
                    x_response=(0.25, 0.0, 0.0), y_response=(0.0, 0.25, 0.0)
                ),
                shear=TactileIndentationCfg.ShearCfg(
                    normal_stiffness=100.0,
                    shear_stiffness=50.0,
                    friction=0.5,
                    marker_indices=(119, 120, 135, 136),
                ),
                surface=(
                    TactileIndentationCfg.SurfaceCfg(max_distance=0.03)
                    if variant == "projection"
                    else None
                ),
            )
        )
        poses = tool.get_local_pose()
        shifted = poses.clone()
        shifted[:, 0] += 0.001
        poses = (poses, shifted)
        zero = torch.zeros((envs, 3), device=device)

        def step(index: int) -> None:
            tool.set_local_pose(poses[index % 2])
            tool.set_velocity(zero, zero)
            sim.update(step=1, render_final_step=False)

        for index in range(warmup):
            step(index)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
        times = []
        for index in range(repeats):
            start = perf_counter()
            step(index)
            torch.cuda.synchronize(device)
            times.append(1e3 * (perf_counter() - start))
        transient = (torch.cuda.max_memory_allocated(device) - baseline_memory) / 2**20
        start = perf_counter()
        for index in range(repeats):
            step(index)
        torch.cuda.synchronize(device)
        throughput = envs * repeats / (perf_counter() - start)
        data = sensor.get_data()
        stats = {
            "environments": envs,
            "variant": variant,
            "p50_ms": median(times),
            "p95_ms": sorted(times)[math.ceil(0.95 * len(times)) - 1],
            "arenas_per_second": throughput,
            "peak_torch_transient_mib": transient,
        }
        if variant != "geometric":
            scene = sim.spawn_result
            snapshot = scene.get_hydroelastic_contact_surface()
            assert snapshot is not None and scene.get_simulation_step() == snapshot.step
            stats["exported_faces"] = int(snapshot.face_count.item())
            stats["face_capacity"] = snapshot.capacity
            if not 0 < stats["exported_faces"] <= snapshot.capacity:
                raise RuntimeError("Benchmark requires a complete nonempty surface.")
        if variant == "projection":
            projected = data["surface_face_count"].flatten(1).sum(dim=1)
            if not data["surface_complete"].all() or not (projected > 0).all():
                raise RuntimeError(
                    "Benchmark requires a complete surface in every arena."
                )
            stats["projected_faces"] = int(projected.sum())
        if not torch.isfinite(data["shear_force"]).all():
            raise RuntimeError("Benchmark observed a nonfinite shear force.")
        outputs = {
            name: data[name].cpu() for name in ("depth", "shape_id", "hit", "rgb")
        }
        return stats, outputs
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def main() -> None:
    """Compare identical sensor and collision workloads with qualified exports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(*args.envs, args.warmup, args.repeats) <= 0:
        parser.error("Environment counts and repetitions must be positive.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("Newton hydroelastic projection requires a CUDA device.")
    if not hasattr(Scene, "get_hydroelastic_contact_capabilities"):
        parser.error("Install a DexSim build with completed hydroelastic bindings.")
    variants = ("geometric", "export", "projection")
    if args.reverse:
        variants = variants[::-1]
    results = []
    for envs in args.envs:
        baseline = None
        for variant in variants:
            stats, data = _measure(device, envs, variant, args.warmup, args.repeats)
            if baseline is None:
                baseline = data
            else:
                for name, reading in data.items():
                    torch.testing.assert_close(
                        reading,
                        baseline[name],
                        atol=1.0e-6,
                        rtol=0.0,
                        msg=f"{variant} changed {name}",
                    )
            results.append(stats)
            print(json.dumps(stats), flush=True)
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(device),
                    "dexsim": dexsim.__version__,
                    "torch": torch.__version__,
                    "warp": wp.__version__,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
