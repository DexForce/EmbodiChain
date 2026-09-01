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
import time

from dexsim.utility.path import get_resources_data_path
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.sim.cfg import (
    NewtonCollisionPipelineCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    SoftObjectCfg,
    SoftbodyVoxelAttributesCfg,
    SoftbodyPhysicalAttributesCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.objects import SoftObject


def main() -> None:
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    parser.set_defaults(device="cuda", physics="newton")
    args = parser.parse_args()
    if args.physics != "newton":
        parser.error("Soft bodies require --physics newton.")
    if not str(args.device).startswith("cuda"):
        parser.error("Soft bodies require a CUDA device.")

    # Configure the simulation
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=args.headless,
        num_envs=args.num_envs,
        arena_space=args.arena_space,
        gpu_id=args.gpu_id,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        device=args.device,
        render_cfg=RenderCfg(
            renderer=args.renderer
        ),  # Enable ray tracing for better visuals
        physics_cfg=NewtonPhysicsCfg(
            num_substeps=6,
            solver_cfg={
                "solver_type": "vbd",
                "iterations": 8,
                "particle_enable_self_contact": False,
                "particle_self_contact_radius": 0.001,
                "particle_self_contact_margin": 0.001,
                "particle_topological_contact_filter_threshold": 3,
                "particle_enable_tile_solve": True,
                "soft_contact_ke": 5.0e4,
                "soft_contact_kd": 1.0e-3,
                "soft_contact_mu": 1.5,
            },
            collision_cfg=NewtonCollisionPipelineCfg(
                soft_contact_margin=0.002,
            ),
        ),
        visualization=visualization_cfg_from_args(args),
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
            init_pos=[0.0, 5.0, 3.0],
            particle_radius=0.01,
            voxel_attr=SoftbodyVoxelAttributesCfg(
                triangle_remesh_resolution=24,
                simulation_mesh_resolution=16,
                voxel_num_relaxation_iters=5,
            ),
            physical_attr=SoftbodyPhysicalAttributesCfg(
                # Equivalent to the DexSim demo's k_mu=1e4 and k_lambda=5e4.
                youngs=2.833333333e4,
                poissons=5.0 / 12.0,
                density=50.0,
                elasticity_damping=2.0e-3,
            ),
        ),
    )
    print("[INFO]: Add soft object complete!")

    sim.prepare()

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
