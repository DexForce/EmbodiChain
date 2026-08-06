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

"""Manipulate objects with native DexSim or browser-based Viser Gizmos."""

from __future__ import annotations

import argparse
import time

import dexsim

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import visualization_cfg_from_args
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
        headless=True,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        sim_device=args.device,
        render_cfg=RenderCfg(
            renderer=args.renderer
        ),  # Enable ray tracing for better visuals
        visualization=visualization_cfg_from_args(args),
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

    # Add two cubes to the scene
    cube1: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube1",
            shape=CubeCfg(size=[0.1, 0.1, 0.1]),
            body_type="kinematic",
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

    native_window_opened = False
    if not args.headless:
        entity_gizmo_config = dexsim.interaction.EntityGizmoConfig()
        entity_gizmo_config.max_gizmos = 0
        native_window_opened = sim.open_window(
            entity_gizmo_config=entity_gizmo_config,
        )

    # Native windows use DexSim's raycast-selected entity controller; headless
    # Viser uses one backend-neutral Gizmo per published object.
    if args.viser:
        sim.enable_gizmo(
            uid="cube1",
            enable_native=False,
        )
        sim.enable_gizmo(
            uid="cube2",
            enable_native=False,
        )
    elif native_window_opened:
        logger.log_info("Left-click an entity and press G to attach/detach its Gizmo.")
        logger.log_info("Multiple selected entities can keep Gizmos simultaneously.")
    else:
        logger.log_warning(
            "Gizmo interaction is disabled in headless mode without Viser."
        )

    logger.log_info("Scene setup complete!")
    logger.log_info(f"Running simulation with 1 environment(s)")
    if native_window_opened or args.viser:
        if args.viser and sim.has_gizmo("cube1"):
            logger.log_info("Gizmo enabled for cube1 - you can drag it around!")
        if args.viser and sim.has_gizmo("cube2"):
            logger.log_info("Gizmo enabled for cube2 - you can drag it around!")
    logger.log_info("Press Ctrl+C to stop the simulation")

    # Run the simulation
    run_simulation(sim)


def run_simulation(sim: SimulationManager):
    """Run the simulation loop."""
    if sim.is_use_gpu_physics:
        sim.init_gpu_physics()

    step_count = 0
    gizmo_enabled = True
    try:
        last_time = time.time()
        last_step = 0
        while True:
            sim.update(step=1)

            step_count += 1

            # Disable Gizmo control after 200000 steps (example).
            if step_count == 200000 and gizmo_enabled:
                logger.log_info("Disabling Gizmo control at step 200000")
                if sim.has_entity_gizmo():
                    sim.disable_entity_gizmo()
                else:
                    sim.disable_gizmo("cube1")
                    sim.disable_gizmo("cube2")
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
        sim.destroy()
        logger.log_info("Simulation terminated successfully")


if __name__ == "__main__":
    main()
