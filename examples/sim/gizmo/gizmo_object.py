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

"""Manipulate raycast-selected entities with dexsim's world-level gizmo."""

from __future__ import annotations

import argparse
import time

import dexsim

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RenderCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg
from embodichain.utils import logger


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
        headless=args.headless,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        sim_device=args.device,
        render_cfg=RenderCfg(
            renderer=args.renderer
        ),  # Enable ray tracing for better visuals
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

    # Add two cubes to the scene
    cube1: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube1",
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
    cube2: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube2",
            shape=CubeCfg(size=[0.1, 0.1, 0.1]),
            body_type="kinematic",
            attrs=RigidBodyAttributesCfg(
                mass=1.0,
                dynamic_friction=0.5,
                static_friction=0.5,
                restitution=0.1,
            ),
            init_pos=[0.3, 0.0, 1.0],
        )
    )

    # Opening a window enables the world-level controller by default. Passing
    # a config here reconfigures it for unlimited simultaneous bindings.
    if not args.headless:
        gizmo_config = dexsim.interaction.EntityGizmoConfig()
        gizmo_config.max_gizmos = 0
        sim.open_window(entity_gizmo_config=gizmo_config)

    logger.log_info("Scene setup complete!")
    logger.log_info(f"Running simulation with 1 environment(s)")
    if not args.headless:
        logger.log_info("Left-click an entity and press G to attach/detach its gizmo.")
        logger.log_info("Multiple selected entities can keep gizmos simultaneously.")
    logger.log_info("Press Ctrl+C to stop the simulation")

    # Run the simulation
    run_simulation(sim)


def run_simulation(sim: SimulationManager):
    """Run the simulation loop."""
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    step_count = 0
    gizmo_enabled = sim.has_entity_gizmo()
    try:
        last_time = time.time()
        last_step = 0
        while True:
            sim.update(step=1)

            step_count += 1

            # Demonstrate programmatic cancellation after 200000 steps.
            if step_count == 200000 and gizmo_enabled:
                logger.log_info("Disabling entity gizmo control at step 200000")
                sim.disable_entity_gizmo()
                gizmo_enabled = False

            # Print FPS every second
            if step_count % 1000 == 0:
                current_time = time.time()
                elapsed = current_time - last_time
                fps = (
                    sim.num_envs * (step_count - last_step) / elapsed
                    if elapsed > 0
                    else 0
                )
                logger.log_info(f"Simulation step: {step_count}, FPS: {fps:.2f}")
                last_time = current_time
                last_step = step_count
    except KeyboardInterrupt:
        logger.log_info("\nStopping simulation...")
    finally:
        if sim.has_entity_gizmo():
            sim.disable_entity_gizmo()
        sim.destroy()
        logger.log_info("Simulation terminated successfully")


if __name__ == "__main__":
    main()
