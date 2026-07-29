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

from __future__ import annotations

"""
This script demonstrates how to create a simulation scene using SimulationManager.
It shows the basic setup of simulation context, adding objects, and sensors.
"""

import argparse

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg
from embodichain.data import get_data_path
from embodichain.lab.sim.utility.demo_utils import (
    DemoRecording,
    add_demo_args,
    create_default_sim,
    maybe_init_gpu_physics,
    maybe_open_window,
    run_simulation_loop,
    shutdown_sim,
)

DEFAULT_HEADLESS_STEPS = 1000
RECORD_LOOK_AT = ((2.6, -2.2, 1.6), (0.0, 0.0, 0.45), (0.0, 0.0, 1.0))


def main():
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_demo_args(parser)
    parser.set_defaults(record_fps=20)
    args = parser.parse_args()
    if args.headless and args.record_steps is None:
        args.record_steps = DEFAULT_HEADLESS_STEPS

    sim = create_default_sim(
        args,
        num_envs=args.num_envs,
        arena_space=3.0,
        add_default_light=False,
    )

    # Add cube object to the scene
    sim.add_rigid_object(
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
            init_pos=[0, 0.0, 1.0],
        )
    )

    # Add chair object to the scene
    path = get_data_path("Chair/chair.glb")
    sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="chair",
            shape=MeshCfg(fpath=path),
            body_type="dynamic",
            attrs=RigidBodyAttributesCfg(
                mass=3.0,
            ),
            body_scale=[0.5, 0.5, 0.5],
            init_pos=[0.0, 0.0, 0.2],
            init_rot=[90.0, 0.0, 0.0],
        )
    )

    print("[INFO]: Scene setup complete!")
    print(f"[INFO]: Running simulation with {args.num_envs} environment(s)")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Open window when the scene has been set up
    maybe_init_gpu_physics(sim)
    maybe_open_window(sim, args)

    run_simulation(sim, args)


def run_simulation(
    sim: SimulationManager,
    args: argparse.Namespace,
) -> None:
    """Run the simulation loop.

    Args:
        sim: The SimulationManager instance to run.
        args: Parsed demo arguments.
    """

    try:
        with DemoRecording(
            sim,
            args,
            prefix="create_scene",
            look_at=RECORD_LOOK_AT,
        ):
            run_simulation_loop(sim, max_steps=args.record_steps)
    finally:
        shutdown_sim(sim)


if __name__ == "__main__":
    main()
