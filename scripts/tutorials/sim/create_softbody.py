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

"""
This script demonstrates how to create a simulation scene using SimulationManager.
It shows the basic setup of simulation context, adding objects, lighting, and sensors.
"""

from __future__ import annotations

import argparse

from dexsim.utility.path import get_resources_data_path

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import (
    SoftbodyPhysicalAttributesCfg,
    SoftbodyVoxelAttributesCfg,
)
from embodichain.lab.sim.objects import (
    SoftObject,
    SoftObjectCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    maybe_init_gpu_physics,
    maybe_open_window,
    resolve_demo_steps,
    run_simulation_loop,
    shutdown_sim,
)


def main():
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_demo_args(parser)
    # Soft-body simulation requires GPU physics.
    parser.set_defaults(device="cuda")
    args = parser.parse_args()

    sim = create_default_sim(
        args,
        num_envs=args.num_envs,
        add_default_light=False,
    )

    print("[INFO]: Scene setup complete!")

    # add softbody to the scene
    cow: SoftObject = sim.add_soft_object(
        cfg=SoftObjectCfg(
            uid="cow",
            shape=MeshCfg(
                fpath=get_resources_data_path("Model", "cow", "cow.obj"),
            ),
            init_pos=[0.0, 0.0, 3.0],
            voxel_attr=SoftbodyVoxelAttributesCfg(
                simulation_mesh_resolution=8,
                maximal_edge_length=0.5,
            ),
            physical_attr=SoftbodyPhysicalAttributesCfg(
                youngs=1e6,
                poissons=0.45,
                density=100,
                dynamic_friction=0.1,
                min_position_iters=30,
            ),
        ),
    )
    print("[INFO]: Add soft object complete!")

    # Open window when the scene has been set up
    maybe_init_gpu_physics(sim)
    maybe_open_window(sim, args)

    print(f"[INFO]: Running simulation with {args.num_envs} environment(s)")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Run the simulation
    run_simulation(sim, cow, args)


def run_simulation(
    sim: SimulationManager,
    soft_obj: SoftObject,
    args: argparse.Namespace,
) -> None:
    """Run the simulation loop.

    Args:
        sim: The SimulationManager instance to run
        soft_obj: soft object
    """

    try:
        with DemoRecording(sim, args, prefix="create_softbody"):
            run_simulation_loop(
                sim,
                max_steps=resolve_demo_steps(args),
                on_step=lambda step: soft_obj.reset() if step % 500 == 0 else None,
            )
    finally:
        shutdown_sim(sim)
        print("[INFO]: Simulation terminated successfully")


if __name__ == "__main__":
    main()
