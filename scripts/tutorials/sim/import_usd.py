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
This script demonstrates how to import USD files into the scene.
Currently, it supports importing USD files as rigid objects or articulations.
Multiple arenas are not supported when importing USD files.
"""

from __future__ import annotations

import argparse

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.objects import (
    RigidObject,
    RigidObjectCfg,
    RobotCfg,
    Robot,
)
from embodichain.data import get_data_path
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
    args = parser.parse_args()

    sim = create_default_sim(
        args,
        num_envs=1,
        arena_space=3.0,
        add_default_light=False,
    )

    cube: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=[0.1, 0.1, 0.1]),
            body_type="dynamic",
            attrs=RigidBodyAttributesCfg(
                mass=1.0,
                dynamic_friction=0.5,
                static_friction=0.5,
                restitution=0.1,
            ),
            init_pos=[0.0, 0.0, 1.0],
        )
    )

    sugar_box_path = get_data_path("SugarBox/sugar_box_usd/sugar_box.usda")
    print(f"Loading USD file from: {sugar_box_path}")
    sugar_box: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="sugar_box",
            shape=MeshCfg(fpath=sugar_box_path),
            body_type="dynamic",
            init_pos=[0.2, 0.2, 1.0],
            use_usd_properties=True,
        )
    )

    # Add objects to the scene
    h1_path = get_data_path("UnitreeH1Usd/H1_usd/h1.usd")
    print(f"Loading USD file from: {h1_path}")
    h1: Robot = sim.add_robot(
        cfg=RobotCfg(
            uid="h1",
            fpath=h1_path,
            build_pk_chain=False,
            init_pos=[-0.2, -0.2, 1.05],
            use_usd_properties=False,
        )
    )

    # Open window when the scene has been set up
    maybe_init_gpu_physics(sim)
    maybe_open_window(sim, args)

    print("[INFO]: Scene setup complete!")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Run the simulation
    run_simulation(sim, args)


def run_simulation(sim: SimulationManager, args: argparse.Namespace) -> None:
    """Run the simulation loop.

    Args:
        sim: The SimulationManager instance to run
    """

    try:
        with DemoRecording(sim, args, prefix="import_usd"):
            run_simulation_loop(
                sim,
                max_steps=resolve_demo_steps(args),
                sleep=0.03,
            )
    finally:
        shutdown_sim(sim)
        print("[INFO]: Simulation terminated successfully")


if __name__ == "__main__":
    main()
