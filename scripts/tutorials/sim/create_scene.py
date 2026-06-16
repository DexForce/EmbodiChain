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
It shows the basic setup of simulation context, adding objects, and sensors.
"""

import argparse
import time

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    RigidBodyAttributesCfg,
    RenderCfg,
    physics_cfg_for_backend,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.data import get_data_path


def main():
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of simulation steps to run before exiting.",
    )
    args = parser.parse_args()

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        device=args.device,
        physics_cfg=physics_cfg_for_backend(args.physics),
        render_cfg=RenderCfg(
            renderer=args.renderer,
        ),
        num_envs=args.num_envs,
        arena_space=3.0,
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

    # Add cube object to the scene
    cube: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=[0.1, 0.1, 0.1]),
            body_type="dynamic",
            body_scale=[0.5, 0.5, 0.5],
            attrs=RigidBodyAttributesCfg(
                mass=0.1,
                dynamic_friction=0.5,
                static_friction=0.5,
                restitution=0.1,
            ),
            init_pos=[0, 0.0, 1.0],
        )
    )

    # Add chair object to the scene
    path = get_data_path("Chair/chair.glb")
    chair: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="chair",
            shape=MeshCfg(fpath=path),
            body_type="dynamic",
            attrs=RigidBodyAttributesCfg(
                mass=10.0,
            ),
            body_scale=[0.5, 0.5, 0.5],
            init_pos=[0.0, 0.0, 0.5],
            init_rot=[0.0, 0.0, 0.0],
            max_convex_hull_num=32,
        )
    )

    print("[INFO]: Scene setup complete!")
    print(f"[INFO]: Running simulation with {args.num_envs} environment(s)")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Open window when the scene has been set up
    if not args.headless:
        sim.open_window()

    # Run the simulation
    run_simulation(sim, max_steps=args.max_steps)


def run_simulation(sim: SimulationManager, max_steps: int | None = None):
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
            step_count += 1

            if max_steps is not None and step_count >= max_steps:
                break

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

    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        # Clean up resources
        sim.destroy()
        print("[INFO]: Simulation terminated successfully")


if __name__ == "__main__":
    main()
