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

"""Create a cable fixed at one end and let it bend under gravity with DexUni.

Run ``python scripts/tutorials/sim/create_cable.py`` to open the native viewer.
Add ``--headless --renderer no-render`` for physics only, or ``--num-envs 2``
to replicate the cable. Geometry is procedural and needs no downloaded assets.
"""

from __future__ import annotations

import argparse

from embodichain.cli.sim import add_sim_args_to_parser

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Return simulation options without initializing the engine."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_sim_args_to_parser(parser)
    parser.set_defaults(device="cuda", physics="newton", renderer="fast-rt")
    parser.add_argument(
        "--steps",
        type=int,
        default=480,
        help="Simulation steps before resetting and exiting.",
    )
    return parser


def main(args: argparse.Namespace | None = None) -> None:
    """Create, prepare, step, inspect and reset a replicated cable.

    Args:
        args: Parsed options; None reads the command line.
    """
    parser = build_parser()
    if args is None:
        args = parser.parse_args()
    if args.physics != "newton" or not str(args.device).startswith("cuda"):
        parser.error("Cables require --physics newton and a CUDA device.")
    if args.num_envs < 1:
        parser.error("--num-envs must be positive.")
    if args.steps <= 0:
        parser.error("--steps must be positive.")

    import torch

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import (
        CableObjectCfg,
        CablePhysicsCfg,
        NewtonPhysicsCfg,
        RenderCfg,
    )
    from embodichain.lab.sim.material import VisualMaterialCfg
    from embodichain.lab.visualization import visualization_cfg_from_args

    sim = SimulationManager(
        SimulationManagerCfg(
            device=args.device,
            num_envs=args.num_envs,
            arena_space=args.arena_space,
            headless=args.headless,
            gpu_id=args.gpu_id,
            physics_dt=1 / 240,
            render_cfg=RenderCfg(renderer=args.renderer),
            physics_cfg=NewtonPhysicsCfg(
                num_substeps=4,
                collision_cfg=None,
                solver_cfg={"solver_type": "dexuni"},
            ),
            visualization=visualization_cfg_from_args(args),
        )
    )
    cable = None
    try:
        # N + 1 local points define N segments. Fix the first segment in place.
        cable = sim.add_cable_object(
            CableObjectCfg(
                uid="cable",
                points=[(0.03 * index, 0, 0) for index in range(21)],
                radius=0.006,
                init_pos=(0, 0, 0.8),
                fixed_segments=(0,),
                attrs=CablePhysicsCfg(
                    stretch_stiffness=2.0e4,
                    stretch_damping=10.0,
                    bend_stiffness=0.01,
                    bend_damping=0.001,
                ),
                visual_material=VisualMaterialCfg(base_color=(0.15, 0.45, 0.9, 1)),
            )
        )
        # Declaration becomes usable segment data at this boundary.
        sim.prepare()
        initial = cable.data.default_segment_state_w
        print(
            f"Created {cable.data.num_segments} cable segments per environment "
            f"in {cable.num_instances} environments."
        )
        if not args.headless:
            sim.open_window()
        for _ in range(args.steps):
            sim.update(step=1)
        # World XYZ, XYZW, COM linear velocity and angular velocity, (E, S, 13).
        state = cable.data.segment_state_w
        if not bool(torch.isfinite(state).all()):
            raise RuntimeError("Cable state is not finite.")
        print(f"Free segment height per environment: {state[:, -1, 2].tolist()} m")
        # An episode reset also restores drive targets, loads and solver history.
        sim.reset_objects_state()
        torch.testing.assert_close(cable.data.segment_state_w, initial)
        print("Restored the initial cable state without advancing physics.")
    finally:
        cable = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
