# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import argparse
import time

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RenderCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.objects import (
    RigidObject,
    RigidObjectCfg,
    RobotCfg,
    Robot,
)
from embodichain.data import get_data_path


def main():
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        sim_device=args.device,
        render_cfg=RenderCfg(
            renderer=args.renderer,
        ),  # Enable ray tracing for better visuals
        num_envs=1,
        arena_space=3.0,
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

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
    if not args.headless:
        sim.open_window()

    print("[INFO]: Scene setup complete!")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Run the simulation
    run_simulation(sim)


def run_simulation(sim: SimulationManager):
    """Run the simulation loop.

    Args:
        sim: The SimulationManager instance to run
    """

    # Initialize GPU physics if using CUDA
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    step_count = 0

    try:
        last_time = time.time()
        last_step = 0
        while True:
            # Update physics simulation
            sim.update(step=1)
            time.sleep(0.03)  # Sleep to limit update rate (optional)
            step_count += 1

            # Print FPS every second
            if step_count % 100 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = (
                    sim.num_envs * (step_count - last_step) / elapsed
                    if elapsed > 0
                    else 0
                )
                # print(f"[INFO]: Simulation step: {step_count}, FPS: {fps:.2f}")
                last_time = current_time
                last_step = step_count

    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        # Clean up resources
        sim.destroy()
        print("[INFO]: Simulation terminated successfully")


if __name__ == "__main__":
    main()
