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

"""Drop a procedural water block into a box using DexSim 0.5.1rc1 SPH.

Run ``python scripts/tutorials/sim/create_fluid.py`` for native particles, or
add ``--headless --renderer no-render`` for physics only. The scene uses five
primitive walls and needs no task assets, robot, Gym or Task Program.
"""

from __future__ import annotations

import argparse

from embodichain.cli.sim import add_sim_args_to_parser

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Return simulation options without loading or initializing the engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_sim_args_to_parser(parser)
    parser.set_defaults(device="cuda", physics="newton", renderer="fast-rt")
    parser.add_argument("--steps", type=int, default=480)
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    """Create, step, inspect and reset a single-world fluid.

    Args:
        args: Parsed options; None reads the command line.
    """
    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    if args.physics != "newton" or not str(args.device).startswith("cuda"):
        parser.error("Fluids require --physics newton and a CUDA device.")
    if args.num_envs != 1:
        parser.error("This DexSim 0.5.1rc1 example requires --num-envs 1.")
    if args.renderer not in {"fast-rt", "rt", "no-render"}:
        parser.error("Choose --renderer fast-rt, rt or no-render for fluids.")
    if args.steps <= 0:
        parser.error("--steps must be positive.")

    import torch
    import trimesh

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        FluidObjectCfg,
        FluidRenderCfg,
        FluidSpawnCfg,
        NewtonPhysicsCfg,
        RenderCfg,
        RigidObjectCfg,
    )
    from embodichain.lab.sim.shapes import MeshCfg, MeshCollisionCfg
    from embodichain.lab.visualization import visualization_cfg_from_args

    spacing = 0.01
    sim = SimulationManager(
        SimulationManagerCfg(
            device=args.device,
            headless=args.headless,
            gpu_id=args.gpu_id,
            physics_dt=1 / 240,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_cfg=NewtonPhysicsCfg(
                num_substeps=4,
                collision_cfg=None,
                solver_cfg={
                    "solver_type": "sph",
                    "sph_method": "dfsph",
                    "particle_diameter": spacing,
                    "particle_support_radius": 2.25 * spacing,
                },
            ),
            visualization=visualization_cfg_from_args(args),
        )
    )
    fluid = None
    try:
        # SPH queries texture SDFs; render boxes alone are not fluid boundaries.
        for uid, position, size in (
            ("floor", (0, 0, 0), (0.32, 0.32, 0.02)),
            ("left", (-0.16, 0, 0.15), (0.02, 0.32, 0.3)),
            ("right", (0.16, 0, 0.15), (0.02, 0.32, 0.3)),
            ("front", (0, -0.16, 0.15), (0.32, 0.02, 0.3)),
            ("back", (0, 0.16, 0.15), (0.32, 0.02, 0.3)),
        ):
            mesh = trimesh.creation.box(extents=size)
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=uid,
                    body_type="kinematic",
                    init_pos=position,
                    shape=MeshCfg(
                        vertices=mesh.vertices,
                        triangles=mesh.faces,
                        collision=MeshCollisionCfg(approximation="sdf"),
                    ),
                )
            )
        fluid = sim.add_fluid_object(
            FluidObjectCfg(
                uid="water",
                particle_spacing=spacing,
                init_pos=(0, 0, 0.16),
                spawn=FluidSpawnCfg(kind="box", size=(0.12, 0.12, 0.12)),
                render=FluidRenderCfg(
                    mode="none" if args.renderer == "no-render" else "particles"
                ),
            )
        )
        sim.prepare()
        initial = fluid.data.default_particle_state_w
        print(f"Created {fluid.num_particles} fluid particles.")
        if not args.headless:
            sim.open_window()
        for _ in range(args.steps):
            sim.update(step=1)
        state = fluid.data.particle_state_w
        if not bool(torch.isfinite(state).all()):
            raise RuntimeError("Fluid state is not finite.")
        print(
            f"Mean particle height after stepping: {state[..., 2].mean().item():.3f} m"
        )
        fluid.reset()
        torch.testing.assert_close(fluid.data.particle_state_w, initial)
        print("Restored the initial fluid state without advancing physics.")
    finally:
        fluid = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
