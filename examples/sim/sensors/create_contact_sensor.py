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

"""Create a grasping scene and report filtered robot/object contacts."""

from __future__ import annotations

import argparse
import time

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.sensors import (
    ContactSensorCfg,
    ArticulationContactFilterCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg, Robot, RobotCfg
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


def create_cube(
    sim: SimulationManager,
    uid: str,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RigidObject:
    """create cube

    Args:
        sim (SimulationManager): simulation manager
        uid (str): uid of the rigid object
        position (list, optional): init position. Defaults to (0., 0., 0).

    Returns:
        RigidObject: rigid object
    """
    cube_size = (0.025, 0.025, 0.025)
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


def robot_grasp_pose(
    robot: Robot,
    cube: RigidObject,
    sim: SimulationManager,
) -> None:
    sim.update(step=100)
    arm_ids = robot.get_joint_ids("arm")
    gripper_ids = robot.get_joint_ids("hand")
    rest_arm_qpos = robot.get_qpos()[:, arm_ids]
    ee_xpos = robot.compute_fk(qpos=rest_arm_qpos, name="arm", to_matrix=True)
    target_xpos = ee_xpos.clone()
    cube_xpos = cube.get_local_pose(to_matrix=True)
    cube_position = cube_xpos[:, :3, 3]

    target_xpos[:, :3, 3] = cube_position

    approach_xpos = target_xpos.clone()
    approach_xpos[:, 2, 3] += 0.1

    approach_success, approach_qpos = robot.compute_ik(
        pose=approach_xpos, joint_seed=rest_arm_qpos, name="arm"
    )
    target_success, target_qpos = robot.compute_ik(
        pose=target_xpos, joint_seed=approach_qpos, name="arm"
    )
    if not (approach_success.all() and target_success.all()):
        raise RuntimeError("Failed to solve the cube grasp pose.")
    robot.set_qpos(approach_qpos, joint_ids=arm_ids)
    sim.update(step=40)

    robot.set_qpos(target_qpos, joint_ids=arm_ids)
    sim.update(step=40)
    hand_close_qpos = (
        torch.tensor([0.025, 0.025], device=sim.device)
        .unsqueeze(0)
        .repeat(sim.num_envs, 1)
    )
    robot.set_qpos(hand_close_qpos, joint_ids=gripper_ids)
    sim.update(step=20)


def create_robot(
    sim: SimulationManager,
    uid: str,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
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
        "uid": uid,
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
        "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
        "drive_pros": {
            "stiffness": {"Joint[1-6]": 1e4, "finger[1-2]_joint": 1e2},
            "damping": {"Joint[1-6]": 1e3, "finger[1-2]_joint": 1e1},
            "max_effort": {"Joint[1-6]": 1e5, "finger[1-2]_joint": 1e3},
        },
        "solver_cfg": {
            "arm": {
                "class_type": "PytorchSolver",
                "end_link_name": "ee_link",
                "root_link_name": "base_link",
                "tcp": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.13],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        },
        "control_parts": {
            "arm": ["Joint[1-6]"],
            "hand": ["finger[1-2]_joint"],
        },
    }
    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(robot_cfg_dict))
    return robot


def main() -> None:
    """Main function to create and run the simulation scene."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a simulation scene with SimulationManager"
    )
    add_demo_args(parser)
    args = parser.parse_args()

    sim = create_default_sim(
        args,
        num_envs=args.num_envs,
        add_default_light=False,
    )

    try:
        create_cube(sim, "cube0", position=(0.0, 0.0, 0.03))
        create_cube(sim, "cube1", position=(0.0, 0.0, 0.06))
        cube2 = create_cube(sim, "cube2", position=(0.0, 0.0, 0.09))
        robot = create_robot(sim, "UR10_PGI", position=(0.5, 0.0, 0.0))

        print("[INFO]: Scene setup complete!")
        print(f"[INFO]: Running simulation with {args.num_envs} environment(s)")

        maybe_init_gpu_physics(sim)
        maybe_open_window(sim, args)
        robot_grasp_pose(robot, cube2, sim)
        run_simulation(sim, args)
    finally:
        shutdown_sim(sim)
        print("[INFO]: Simulation terminated successfully")


def run_simulation(sim: SimulationManager, args: argparse.Namespace) -> None:
    """Run the simulation loop.

    Args:
        sim: The SimulationManager instance to run
    """

    # contact filter config
    contact_filter_cfg = ContactSensorCfg()
    contact_filter_cfg.rigid_uid_list = ["cube0", "cube1", "cube2"]
    contact_filter_art_cfg = ArticulationContactFilterCfg()
    contact_filter_art_cfg.articulation_uid = "UR10_PGI"
    contact_filter_art_cfg.link_name_list = ["finger1_link", "finger2_link"]
    contact_filter_cfg.articulation_cfg_list = [contact_filter_art_cfg]
    contact_filter_cfg.filter_need_both_actor = True

    contact_sensor = sim.add_sensor(sensor_cfg=contact_filter_cfg)

    accumulated_cost_time = 0.0

    def update_contact(step: int) -> None:
        """Update the sensor and periodically report/filter contacts."""
        nonlocal accumulated_cost_time
        started_at = time.perf_counter()
        contact_sensor.update()
        contact_sensor.get_data()
        accumulated_cost_time += time.perf_counter() - started_at

        if step % 100 != 0:
            return
        print(
            "[INFO]: Fetch contact cost time: "
            f"{accumulated_cost_time * 10:.2f} ms, num_envs: {sim.num_envs}"
        )
        cube_ids = sim.get_rigid_object("cube2").get_user_ids()
        finger_ids = sim.get_robot("UR10_PGI").get_user_ids("finger1_link").reshape(-1)
        contact_sensor.filter_by_user_ids(torch.cat([cube_ids, finger_ids]))
        contact_sensor.set_contact_point_visibility(
            visible=True,
            rgba=(0.0, 0.0, 1.0, 1.0),
            point_size=6.0,
        )
        accumulated_cost_time = 0.0

    with DemoRecording(sim, args, prefix="contact_sensor"):
        run_simulation_loop(
            sim,
            max_steps=resolve_demo_steps(args),
            on_step=update_contact,
        )


if __name__ == "__main__":
    main()
