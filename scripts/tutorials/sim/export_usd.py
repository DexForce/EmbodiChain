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
This script demonstrates how to export a simulation scene to a usd file using the SimulationManager. 
"""

import argparse
import numpy as np
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.cfg import (
    LightCfg,
    JointDrivePropertiesCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
    ArticulationCfg,
)
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.data import get_data_path
from embodichain.utils import logger

from embodichain.lab.sim.robots.dexforce_w1.cfg import DexforceW1Cfg


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including number of environments and rendering options.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )

    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering"
    )
    parser.add_argument("--headless", action="store_true", help="Enable headless mode")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run the environment on, e.g., 'cpu' or 'cuda'",
    )
    return parser.parse_args()


def initialize_simulation(args) -> SimulationManager:
    """
    Initialize the simulation environment based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        SimulationManager: Configured simulation manager instance.
    """
    config = SimulationManagerCfg(
        headless=True,
        sim_device=args.device,
        enable_rt=args.enable_rt,
        physics_dt=1.0 / 100.0,
        num_envs=1,
        arena_space=2.5,
    )
    sim = SimulationManager(config)

    if args.enable_rt:
        light = sim.add_light(
            cfg=LightCfg(
                uid="main_light",
                color=(0.6, 0.6, 0.6),
                intensity=30.0,
                init_pos=(1.0, 0, 3.0),
            )
        )

    return sim


def create_robot(sim: SimulationManager) -> Robot:
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    cfg = DexforceW1Cfg.from_dict(
        {
            "uid": "dexforce_w1",
            "init_pos": [0.4, -0.5, 0.0],
        }
    )
    cfg.solver_cfg["left_arm"].tcp = np.array(
        [
            [1.0, 0.0, 0.0, 0.012],
            [0.0, 1.0, 0.0, 0.04],
            [0.0, 0.0, 1.0, 0.11],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    cfg.solver_cfg["right_arm"].tcp = np.array(
        [
            [1.0, 0.0, 0.0, 0.012],
            [0.0, 1.0, 0.0, -0.04],
            [0.0, 0.0, 1.0, 0.11],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    cfg.init_qpos = [
        1.0000e00,
        -2.0000e00,
        1.0000e00,
        0.0000e00,
        -2.6921e-05,
        -2.6514e-03,
        -1.5708e00,
        1.4575e00,
        -7.8540e-01,
        1.2834e-01,
        1.5708e00,
        -2.2310e00,
        -7.8540e-01,
        1.4461e00,
        -1.5708e00,
        1.6716e00,
        7.8540e-01,
        7.6745e-01,
        0.0000e00,
        3.8108e-01,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        1.5000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        0.0000e00,
        1.5000e00,
        6.9974e-02,
        7.3950e-02,
        6.6574e-02,
        6.0923e-02,
        0.0000e00,
        6.7342e-02,
        7.0862e-02,
        6.3684e-02,
        5.7822e-02,
        0.0000e00,
    ]
    return sim.add_robot(cfg=cfg)


def create_table(sim: SimulationManager) -> RigidObject:
    """
    Create a table rigid object in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        RigidObject: The table object added to the simulation.
    """
    scoop_cfg = RigidObjectCfg(
        uid="table",
        shape=MeshCfg(
            fpath=get_data_path("MultiW1Data/table_a.obj"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.5,
        ),
        max_convex_hull_num=8,
        body_type="kinematic",
        init_pos=[1.1, -0.5, 0.08],
        init_rot=[0.0, 0.0, 0.0],
    )
    scoop = sim.add_rigid_object(cfg=scoop_cfg)
    return scoop


def create_caffe(sim: SimulationManager) -> Robot:
    """
    Create a caffe (container) articulated object in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The caffe object added to the simulation.
    """
    container_cfg = ArticulationCfg(
        uid="caffe",
        fpath=get_data_path("MultiW1Data/cafe/cafe.urdf"),
        init_pos=[1.05, -0.5, 0.79],
        init_rot=[0, 0, -30],
        attrs=RigidBodyAttributesCfg(
            mass=1.0,
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness=1.0, damping=0.1, max_effort=100.0, drive_type="force"
        ),
    )
    print(f"Loading URDF file from: {container_cfg.fpath}")
    container = sim.add_articulation(cfg=container_cfg)
    return container


def create_cup(sim: SimulationManager) -> RigidObject:
    """
    Create a cup rigid object in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        RigidObject: The cup object added to the simulation.
    """
    scoop_cfg = RigidObjectCfg(
        uid="cup",
        shape=MeshCfg(
            fpath=get_data_path("MultiW1Data/paper_cup_2.obj"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.3,
        ),
        max_convex_hull_num=1,
        body_type="dynamic",
        init_pos=[0.86, -0.76, 0.841],
        init_rot=[0.0, 0.0, 0.0],
    )
    scoop = sim.add_rigid_object(cfg=scoop_cfg)
    return scoop


def main():
    """
    Main function to create simulation scene.

    Initializes the simulation and creates the robot and objects in the scene.
    """
    args = parse_arguments()
    sim = initialize_simulation(args)

    robot = create_robot(sim)
    table = create_table(sim)
    caffe = create_caffe(sim)
    cup = create_cup(sim)

    sim.export_usd("w1_coffee_scene.usda")

    logger.log_info("Scene exported successfully.")

    if not args.headless:
        sim.open_window()
        logger.log_info("Press Ctrl+C to exit.")
        try:
            while True:
                sim.update(step=1)
        except KeyboardInterrupt:
            logger.log_info("\nExit")


if __name__ == "__main__":
    main()
