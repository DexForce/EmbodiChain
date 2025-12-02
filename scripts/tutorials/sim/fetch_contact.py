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
It shows the basic setup of simulation context, adding objects, and sensors.
"""

import argparse
import time
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, ContactFilterCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg, Robot, RobotCfg
from embodichain.data import get_data_path


def create_cube(
    sim: SimulationManager, uid: str, position: list = (0.0, 0.0, 0)
) -> RigidObject:
    """create cube

    Args:
        sim (SimulationManager): simulation manager
        uid (str): uid of the rigid object
        position (list, optional): init position. Defaults to (0., 0., 0).

    Returns:
        RigidObject: rigid object
    """
    cube_size = (0.02, 0.02, 0.02)
    cube: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=uid,
            shape=CubeCfg(size=cube_size),
            body_type="dynamic",
            attrs=RigidBodyAttributesCfg(
                mass=0.1,
                dynamic_friction=0.9,
                static_friction=0.95,
                restitution=0.01,
                sleep_threshold=0.0,
            ),
            init_pos=position,
        )
    )
    return cube


def create_robot(
    sim: SimulationManager, uid: str, position: list = (0.0, 0.0, 0)
) -> Robot:
    """create robot

    Args:
        sim (SimulationManager): _description_
        uid (str): _description_
        position (list, optional): _description_. Defaults to (0., 0., 0).

    Returns:
        Robot: _description_
    """
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    pgi_urdf_path = get_data_path("DH_PGC_140_50/DH_PGC_140_50.urdf")
    robot_cfg_dict = {
        "uid": "UR10_PGI",
        "urdf_cfg": {
            "components": [
                {
                    "component_type": "arm",
                    "urdf_path": ur10_urdf_path,
                    "transform": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                {
                    "component_type": "hand",
                    "urdf_path": pgi_urdf_path,
                    "transform": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            ],
        },
        "init_pos": position,
        "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.025, 0.025],
        "drive_pros": {
            "stiffness": {"JOINT[1-6]": 1e4, "FINGER[1-2]_JOINT": 1e2},
            "damping": {"JOINT[1-6]": 1e3, "FINGER[1-2]_JOINT": 1e1},
            "max_effort": {"JOINT[1-6]": 1e5, "FINGER[1-2]_JOINT": 1e3},
        },
        "solver_cfg": {
            "arm": {
                "class_type": "PytorchSolver",
                "end_link_name": "ee_link",
                "root_link_name": "base_link",
                "tcp": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.16],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
        "control_parts": {"arm": ["JOINT[1-6]"], "gripper": ["FINGER[1-2]_JOINT"]},
    }
    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(robot_cfg_dict))
    return robot


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
        "--num_envs", type=int, default=100, help="Number of parallel environments"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Simulation device (cuda or cpu)"
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
        sim_device=args.device,
        enable_rt=args.enable_rt,  # Enable ray tracing for better visuals
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

    # Build multiple arenas if requested
    if args.num_envs > 1:
        sim.build_multiple_arenas(args.num_envs, space=3.0)

    # Add objects to the scene
    cube0 = create_cube(sim, "cube0", position=[0.0, 0.0, 0.025])
    cube1 = create_cube(sim, "cube1", position=[0.0, 0.0, 0.05])
    cube2 = create_cube(sim, "cube2", position=[0.0, 0.0, 0.075])
    robot = create_robot(sim, "UR10_PGI", position=[0.5, 0.0, 0.0])

    print("[INFO]: Scene setup complete!")
    print(f"[INFO]: Running simulation with {args.num_envs} environment(s)")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Open window when the scene has been set up
    if not args.headless:
        sim.open_window()

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
    # contact filter config
    contact_filter_cfg = ContactFilterCfg()
    contact_filter_cfg.rigid_uid_list = ["cube0", "cube1", "cube2"]
    contact_filter_cfg.filter_need_both_actor = True
    try:
        accmulated_cost_time = 0.0
        while True:
            # Update physics simulation
            sim.update(step=1)
            start_time = time.time()
            contact_report = sim.get_contact(contact_filter_cfg)
            accmulated_cost_time += time.time() - start_time

            n_contact = contact_report.contact_data.shape[0]
            if n_contact > 0:
                contact_positions = contact_report.contact_data[:, 0:3]
                contact_normals = contact_report.contact_data[:, 3:6]
                contact_frictions = contact_report.contact_data[:, 6:9]
                contact_impluses = contact_report.contact_data[:, 9]
                contact_distances = contact_report.contact_data[:, 10]
                contact_user_ids = (
                    contact_report.contact_user_ids
                )  # user can use userid to identify which object the contact belongs to, using rigid_object.get_user_id()
                contact_env_ids = (
                    contact_report.contact_env_ids
                )  # contact belongs to which environment

                # filter contact report for specific rigid object
                cube1 = sim.get_rigid_object("cube1")
                filtered_contact_report = contact_report.filter_by_user_ids(
                    cube1.get_user_ids()
                )

            step_count += 1

            # # Print FPS every second
            if step_count % 100 == 0:
                average_cost_time = accmulated_cost_time / 100.0
                print(
                    f"[INFO]: Fetch contact cost time: {average_cost_time * 1000:.2f} ms, num_envs: {sim.num_envs}"
                )
                accmulated_cost_time = 0.0

    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        # Clean up resources
        sim.destroy()
        print("[INFO]: Simulation terminated successfully")


if __name__ == "__main__":
    main()
