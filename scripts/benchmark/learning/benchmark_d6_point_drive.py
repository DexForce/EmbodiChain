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
"""Measure native point-drive configuration and completed explicit steps.

Run one World per process::

    python -m scripts.benchmark.learning.benchmark_d6_point_drive --num-envs 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    CollisionPropertiesCfg,
    DefaultPhysicsCfg,
    DefaultRigidBodyPropertiesCfg,
    MassPropertiesCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidConstraintCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg

__all__ = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=25)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--point-anchors", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("counts must be positive")
    device = torch.device(args.device)
    sim = SimulationManager(
        SimulationManagerCfg(
            num_envs=args.num_envs,
            device=str(device),
            gpu_id=device.index,
            physics_cfg=DefaultPhysicsCfg(
                enable_tgs=False, cache_material=True, gravity=(0, 0, 0)
            ),
            physics_dt=0.002,
            headless=True,
            enable_default_plane=False,
            render_cfg=RenderCfg(renderer="no-render"),
            enable_entity_gizmo=False,
            robot_ik_gizmo=None,
            startup_summary="off",
        )
    )
    try:
        props = RigidBodyPhysicsCfg(
            mass_props=MassPropertiesCfg(mass=1),
            rigid_props=DefaultRigidBodyPropertiesCfg(
                has_gravity=False,
                linear_damping=0,
                angular_damping=0,
                sleep_threshold=0,
                min_position_iters=100,
                min_velocity_iters=1,
            ),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="anchor",
                body_type="static",
                init_pos=(0, 0, 0),
                shape=CubeCfg(size=[0.02] * 3),
                attrs=props,
            )
        )
        payload = sim.add_rigid_object(
            RigidObjectCfg(
                uid="payload",
                init_pos=(0, 0, 1),
                shape=CubeCfg(size=[0.02] * 3),
                attrs=props,
            )
        )
        sim.prepare()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        anchor_frame = torch.eye(4).numpy()
        anchor_frame[2, 3] = 1.0
        sim.create_rigid_constraint(
            RigidConstraintCfg(
                name="point",
                rigid_object_a_uid="anchor",
                rigid_object_b_uid="payload",
                local_frame_a=anchor_frame,
                local_frame_b=torch.eye(4).numpy(),
                constraint_type="spherical",
                linear_drive_stiffness=100.0,
                linear_drive_damping=20.0,
                linear_drive_velocity=[0, 0.2, 0],
                linear_drive_point_anchors=args.point_anchors,
            )
        )
        torch.cuda.synchronize(device)
        creation_ms = (time.perf_counter() - start) * 1000
        for _ in range(500):
            sim.update(0.002, step=1, render_final_step=False)
        blocks = []
        updates = []
        for _ in range(5):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.steps):
                updated = time.perf_counter()
                count = sim.update_rigid_constraint_linear_drive(
                    "point", 100.0, 20.0, [0, 0.2, 0]
                )
                updates.append((time.perf_counter() - updated) * 1000)
                assert count == args.num_envs
                sim.update(0.002, step=1, render_final_step=False)
            torch.cuda.synchronize(device)
            blocks.append((time.perf_counter() - start) * 1000 / args.steps)
        pose = payload.get_local_pose()
        # At equilibrium -K*y + D*target_v = 0, so y=20*.2/100=.04 m.
        error = float((pose[:, 1] - 0.04).abs().max())
        if error > 1e-4:
            raise AssertionError(f"point-drive equilibrium error {error}")
        report = {
            "scope": "creation separately; completed step includes drive update and native stepping, excludes rendering/IK",
            "num_envs": args.num_envs,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "solver": "PGS",
            "point_anchors": args.point_anchors,
            "dt_s": 0.002,
            "warmup_steps": 500,
            "blocks": 5,
            "steps_per_block": args.steps,
            "creation_ms": creation_ms,
            "mean_update_host_ms": statistics.mean(updates),
            "completed_step_mean_ms": statistics.mean(blocks),
            "completed_step_median_ms": statistics.median(blocks),
            "equilibrium_max_error_m": error,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
