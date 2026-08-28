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

from __future__ import annotations

import argparse
import time
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.lab.sim.cfg import (
    DefaultRigidBodyPropertiesCfg,
    MassPropertiesCfg,
    RenderCfg,
    physics_cfg_for_backend,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
)
from embodichain.lab.sim.sensors import (
    ContactSensorCfg,
    ArticulationContactFilterCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.objects import RigidObject, RigidObjectCfg, Robot, RobotCfg
from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser


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
    cube_size = (0.025, 0.025, 0.025)
    cube: RigidObject = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=uid,
            shape=CubeCfg(size=cube_size),
            body_type="dynamic",
            attrs=RigidBodyPhysicsCfg(
                mass_props=MassPropertiesCfg(mass=0.1),
                rigid_props=DefaultRigidBodyPropertiesCfg(sleep_threshold=0.0),
                material_props=RigidBodyMaterialCfg(
                    dynamic_friction=0.9,
                    static_friction=0.95,
                    restitution=0.01,
                ),
            ),
            init_pos=position,
        )
    )
    return cube


def robot_grasp_pose(robot: Robot, cube: RigidObject, sim: SimulationManager):
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

    is_success, approach_qpos = robot.compute_ik(
        pose=approach_xpos, joint_seed=rest_arm_qpos, name="arm"
    )
    is_success, target_qpos = robot.compute_ik(
        pose=target_xpos, joint_seed=approach_qpos, name="arm"
    )
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
        "control_parts": {"arm": ["Joint[1-6]"], "hand": ["finger[1-2]_joint"]},
    }
    robot: Robot = sim.add_robot(cfg=RobotCfg.from_dict(robot_cfg_dict))
    return robot


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
        num_envs=args.num_envs,
        headless=True,
        physics_dt=1.0 / 100.0,  # Physics timestep (100 Hz)
        device=args.device,
        render_cfg=RenderCfg(
            renderer=args.renderer
        ),  # Enable ray tracing for better visuals
        physics_cfg=physics_cfg_for_backend(args.physics),
        visualization=visualization_cfg_from_args(args),
    )

    # Create the simulation instance
    sim = SimulationManager(sim_cfg)

    # Add objects to the scene
    cube0 = create_cube(sim, "cube0", position=[0.0, 0.0, 0.03])
    cube1 = create_cube(sim, "cube1", position=[0.0, 0.0, 0.06])
    cube2 = create_cube(sim, "cube2", position=[0.0, 0.0, 0.09])
    robot = create_robot(sim, "UR10_PGI", position=[0.5, 0.0, 0.0])
    sim.prepare()

    print("[INFO]: Scene setup complete!")
    print(f"[INFO]: Running simulation with {args.num_envs} environment(s)")
    print("[INFO]: Press Ctrl+C to stop the simulation")

    # Open window when the scene has been set up
    if not args.headless:
        sim.open_window()

    robot_grasp_pose(robot, cube2, sim)
    # Run the simulation
    run_simulation(sim)


def run_simulation(sim: SimulationManager):
    """Run the simulation loop.

    Args:
        sim: The SimulationManager instance to run
    """

    step_count = 0
    # contact filter config
    contact_filter_cfg = ContactSensorCfg()
    contact_filter_cfg.rigid_uid_list = ["cube0", "cube1", "cube2"]
    contact_filter_art_cfg = ArticulationContactFilterCfg()
    contact_filter_art_cfg.articulation_uid = "UR10_PGI"
    contact_filter_art_cfg.link_name_list = ["finger1_link", "finger2_link"]
    contact_filter_cfg.articulation_cfg_list = [contact_filter_art_cfg]
    contact_filter_cfg.filter_need_both_actor = True

    if sim.is_newton_backend:
        run_newton_contact_query(sim, contact_filter_cfg)
        return

    contact_sensor = sim.add_sensor(sensor_cfg=contact_filter_cfg)

    try:
        accmulated_cost_time = 0.0
        while True:
            # Update physics simulation
            sim.update(step=1)
            start_time = time.time()
            contact_sensor.update()
            contact_report = contact_sensor.get_data()
            accmulated_cost_time += time.time() - start_time
            step_count += 1

            # Print FPS every second
            if step_count % 100 == 0:
                average_cost_time = accmulated_cost_time / 100.0
                print(
                    f"[INFO]: Fetch contact cost time: {average_cost_time * 1000:.2f} ms, num_envs: {sim.num_envs}"
                )
                # filter contact report for a rigid object with a articulation link
                cube2_user_ids = sim.get_rigid_object("cube2").get_user_ids()
                finger1_user_ids = (
                    sim.get_robot("UR10_PGI").get_user_ids("finger1_link").reshape(-1)
                )
                filter_user_ids = torch.cat([cube2_user_ids, finger1_user_ids])
                filter_contact_report = contact_sensor.filter_by_user_ids(
                    filter_user_ids
                )
                # print("filter_contact_report", filter_contact_report)
                # visualize contact points
                contact_sensor.set_contact_point_visibility(
                    visible=True, rgba=(0.0, 0.0, 1.0, 1.0), point_size=6.0
                )
                accmulated_cost_time = 0.0

    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        # Clean up resources
        sim.destroy()
        print("[INFO]: Simulation terminated successfully")


def run_newton_contact_query(
    sim: SimulationManager, contact_filter_cfg: ContactSensorCfg
) -> None:
    """Run Newton's raw contact query for the configured collision shapes.

    The generic :class:`ContactSensor` currently consumes Default-backend
    ``PhysicsScene`` buffers. Newton's Spawn runtime instead owns the contact
    buffers directly, so this example queries those buffers without claiming
    that the generic sensor API is backend-neutral yet.

    Args:
        sim: Prepared simulation manager using the Newton backend.
        contact_filter_cfg: Rigid objects and articulation links to monitor.
    """
    import warp as wp
    from dexsim.engine.newton_physics.backend_registry import get_newton_backend

    result = sim.spawn_result
    if result is None:
        raise RuntimeError("Newton contact queries require a prepared Spawn scene.")
    backend = get_newton_backend(result.world)
    if backend is None:
        raise RuntimeError("Newton Spawn runtime is unavailable for contact queries.")
    if not callable(getattr(backend.solver, "update_contacts", None)):
        raise RuntimeError(
            "The active Newton solver does not expose contact-query support."
        )

    filter_shape_ids = _newton_filter_shape_ids(sim, contact_filter_cfg)
    step_count = 0
    accumulated_cost_time = 0.0

    try:
        while True:
            sim.update(step=1)
            start_time = time.time()
            backend.solver.update_contacts(backend.contacts, backend.state_0)
            total_contacts = int(
                wp.to_torch(backend.contacts.rigid_contact_count).reshape(-1)[0].item()
            )
            matched_contacts = 0
            if total_contacts > 0:
                shape0 = wp.to_torch(backend.contacts.rigid_contact_shape0)[
                    :total_contacts
                ]
                shape1 = wp.to_torch(backend.contacts.rigid_contact_shape1)[
                    :total_contacts
                ]
                shape0_matches = torch.isin(shape0, filter_shape_ids)
                shape1_matches = torch.isin(shape1, filter_shape_ids)
                if contact_filter_cfg.filter_need_both_actor:
                    matched_contacts = int(
                        torch.logical_and(shape0_matches, shape1_matches).sum().item()
                    )
                else:
                    matched_contacts = int(
                        torch.logical_or(shape0_matches, shape1_matches).sum().item()
                    )
            accumulated_cost_time += time.time() - start_time
            step_count += 1

            if step_count % 100 == 0:
                average_cost_time = accumulated_cost_time / 100.0
                print(
                    "[INFO]: Fetch Newton contact cost time: "
                    f"{average_cost_time * 1000:.2f} ms, "
                    f"contacts: {matched_contacts}, num_envs: {sim.num_envs}"
                )
                accumulated_cost_time = 0.0
    except KeyboardInterrupt:
        print("\n[INFO]: Stopping simulation...")
    finally:
        sim.destroy()
        print("[INFO]: Simulation terminated successfully")


def _newton_filter_shape_ids(
    sim: SimulationManager, contact_filter_cfg: ContactSensorCfg
) -> torch.Tensor:
    """Resolve a contact filter configuration to Newton Spawn shape IDs."""
    shape_ids: list[int] = []
    for rigid_uid in contact_filter_cfg.rigid_uid_list:
        rigid_object = sim.get_rigid_object(rigid_uid)
        if rigid_object is None:
            continue
        for entity in rigid_object._entities:
            physics_body = entity.physics_body
            if physics_body is not None:
                shape_ids.extend(int(shape_id) for shape_id in physics_body.shape_ids)

    for articulation_cfg in contact_filter_cfg.articulation_cfg_list:
        articulation = sim.get_robot(articulation_cfg.articulation_uid)
        if articulation is None:
            articulation = sim.get_articulation(articulation_cfg.articulation_uid)
        if articulation is None:
            continue
        for entity in articulation._entities:
            physics_articulation = entity.physics_articulation
            if physics_articulation is None:
                continue
            link_names = (
                set(articulation_cfg.link_name_list)
                if articulation_cfg.link_name_list
                else {link.name for link in physics_articulation.links}
            )
            for link in physics_articulation.links:
                if link.name in link_names:
                    shape_ids.extend(int(shape_id) for shape_id in link.shape_ids)

    if not shape_ids:
        raise ValueError(
            "The Newton contact filter did not resolve to any collision shapes."
        )
    return torch.tensor(
        sorted(set(shape_ids)),
        dtype=torch.int32,
        device=sim.device,
    )


if __name__ == "__main__":
    main()
