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
This script demonstrates how to create a simulation scene using SimulationManager.
It shows the basic setup of simulation context, adding objects, lighting, and sensors.
"""

import argparse
import time
from dexsim.utility.path import get_resources_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    SoftbodyVoxelAttributesCfg,
    SoftbodyPhysicalAttributesCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.objects import (
    SoftObject,
    SoftObjectCfg,
)


def main():
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--num_envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument(
        "--enable_rt",
        action="store_true",
        default=False,
        help="Enable ray tracing for better visuals",
    )
    args = parser.parse_args()

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        sim_device="cuda",  # soft simulation only supports cuda device
        enable_rt=args.enable_rt,  # Enable ray tracing for better visuals
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

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
    if not args.headless:
        sim.open_window()

    print(f"[INFO]: Running simulation with {args.num_envs} environment(s)")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Run the simulation
    run_simulation(sim, cow)


def run_simulation(sim: SimulationManager, soft_obj: SoftObject) -> None:
    """Run the simulation loop.

    Args:
        sim: The SimulationManager instance to run
        soft_obj: soft object
    """

    # Initialize GPU physics
    sim.init_gpu_physics()

    step_count = 0

    try:
        last_time = time.time()
        last_step = 0
        while True:
            # Update physics simulation
            sim.update(step=1)
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
                print(f"[INFO]: Simulation step: {step_count}, FPS: {fps:.2f}")
                last_time = current_time
                last_step = step_count
                if step_count % 500 == 0:
                    soft_obj.reset()

    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        # Clean up resources
        sim.destroy()
        print("[INFO]: Simulation terminated successfully")


if __name__ == "__main__":
    main()
