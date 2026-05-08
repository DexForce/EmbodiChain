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
Tutorial: Atomic Actions for Robot Motion Generation
=====================================================

This script shows how to use the atomic action system to plan and execute
a pick-and-place task with a robot arm.

Key concepts covered:
  1. Setting up a MotionGenerator and AtomicActionEngine
  2. Describing what to pick using ObjectSemantics and AntipodalAffordance
  3. Running a pick → place → move sequence with execute_static()

Run with:
    python atomic_actions.py [--num_envs N] [--enable_rt]
"""

import argparse
import numpy as np
import time
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RobotCfg,
    RigidObjectCfg,
    RigidBodyAttributesCfg,
    LightCfg,
    URDFCfg,
)
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    GraspGenerator,
    GraspGeneratorCfg,
    AntipodalSamplerCfg,
)

# Import everything from the public atomic_actions API
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ObjectSemantics,
    AntipodalAffordance,
    PickUpActionCfg,
    PlaceActionCfg,
    MoveActionCfg,
)


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including number of environments, device, and rendering options.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    parser.add_argument(
        "--enable_rt", action="store_true", help="Enable ray tracing rendering"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
    )
    return parser.parse_args()


def initialize_simulation(args):
    """
    Initialize the simulation environment based on the provided arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        SimulationManager: Configured simulation manager instance.
    """
    config = SimulationManagerCfg(
        headless=True,
        sim_device="cuda",
        enable_rt=args.enable_rt,
        physics_dt=1.0 / 100.0,
        num_envs=args.num_envs,
    )
    sim = SimulationManager(config)

    light = sim.add_light(
        cfg=LightCfg(uid="main_light", intensity=50.0, init_pos=(0, 0, 2.0))
    )

    return sim


def create_robot(sim: SimulationManager, position=[0.0, 0.0, 0.0]):
    """
    Create and configure a robot with an arm and a dexterous hand in the simulation.

    Args:
        sim (SimulationManager): The simulation manager instance.

    Returns:
        Robot: The configured robot instance added to the simulation.
    """
    # Retrieve URDF paths for the robot arm and hand
    ur10_urdf_path = get_data_path("UniversalRobots/UR10/UR10.urdf")
    gripper_urdf_path = get_data_path("DH_PGC_140_50_M/DH_PGC_140_50_M.urdf")
    # Configure the robot with its components and control properties
    cfg = RobotCfg(
        uid="UR10",
        urdf_cfg=URDFCfg(
            components=[
                {"component_type": "arm", "urdf_path": ur10_urdf_path},
                {"component_type": "hand", "urdf_path": gripper_urdf_path},
            ]
        ),
        drive_pros=JointDrivePropertiesCfg(
            stiffness={"JOINT[0-9]": 1e4, "FINGER[1-2]": 1e2},
            damping={"JOINT[0-9]": 1e3, "FINGER[1-2]": 1e1},
            max_effort={"JOINT[0-9]": 1e5, "FINGER[1-2]": 1e3},
            drive_type="force",
        ),
        control_parts={
            "arm": ["JOINT[0-9]"],
            "hand": ["FINGER[1-2]"],
        },
        solver_cfg={
            "arm": PytorchSolverCfg(
                end_link_name="ee_link",
                root_link_name="base_link",
                tcp=[
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.12],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            )
        },
        init_qpos=[0.0, -np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, 0.0, 0.0, 0.0],
        init_pos=position,
    )
    return sim.add_robot(cfg=cfg)


def create_mug(sim: SimulationManager) -> RigidObject:
    mug_cfg = RigidObjectCfg(
        uid="mug",
        shape=MeshCfg(
            fpath=get_data_path("CoffeeCup/cup.ply"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.01,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[0.55, 0.0, 0.01],
        init_rot=[0.0, 0.0, -90],
        body_scale=(4, 4, 4),
    )
    mug = sim.add_rigid_object(cfg=mug_cfg)
    return mug


def main():
    """Pick up a mug and place it at a new location using atomic actions."""
    args = parse_arguments()

    # ------------------------------------------------------------------ #
    # Step 1: Set up simulation, robot, and object                        #
    # ------------------------------------------------------------------ #
    sim: SimulationManager = initialize_simulation(args)
    robot = create_robot(sim)
    mug = create_mug(sim)

    # ------------------------------------------------------------------ #
    # Step 2: Create a MotionGenerator for the robot                      #
    # MotionGenerator handles trajectory planning (IK + TOPPRA smoothing) #
    # ------------------------------------------------------------------ #
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    # ------------------------------------------------------------------ #
    # Step 3: Configure the three atomic actions                          #
    #                                                                     #
    #  PickUpAction  — approach → close gripper → lift                   #
    #  PlaceAction   — lower → open gripper → retract                    #
    #  MoveAction    — free-space move to a target EEF pose               #
    # ------------------------------------------------------------------ #
    # Gripper joint values for this robot (DH_PGC_140):
    #   open  = [0.00, 0.00]   (fully open)
    #   close = [0.025, 0.025] (grasping width)
    hand_open = torch.tensor([0.00, 0.00], dtype=torch.float32, device=sim.device)
    hand_close = torch.tensor([0.025, 0.025], dtype=torch.float32, device=sim.device)

    pickup_cfg = PickUpActionCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        # Approach the object from directly above (negative world-Z)
        approach_direction=torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=sim.device
        ),
        pre_grasp_distance=0.15,  # hover 15 cm above before descending
        lift_height=0.15,  # lift 15 cm after grasping
    )

    place_cfg = PlaceActionCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        lift_height=0.15,
    )

    move_cfg = MoveActionCfg(
        control_part="arm",
    )

    # ------------------------------------------------------------------ #
    # Step 4: Build the AtomicActionEngine                                #
    #                                                                     #
    # actions_cfg_list defines the ORDER of actions that execute_static() #
    # will run. Each entry is matched positionally to target_list.        #
    # ------------------------------------------------------------------ #
    atomic_engine = AtomicActionEngine(
        motion_generator=motion_gen,
        actions_cfg_list=[pickup_cfg, place_cfg, move_cfg],
    )

    sim.init_gpu_physics()
    sim.open_window()

    # ------------------------------------------------------------------ #
    # Step 5: Describe the mug with ObjectSemantics                       #
    #                                                                     #
    # ObjectSemantics bundles together:                                   #
    #   - geometry (mesh vertices/triangles for grasp annotation)         #
    #   - affordance (how to grasp the object — here antipodal grasps)   #
    #   - entity reference (so the action can read the live object pose)  #
    # ------------------------------------------------------------------ #
    mug_grasp_affordance = AntipodalAffordance(
        object_label="mug",
        force_reannotate=False,
        custom_config={
            "gripper_collision_cfg": GripperCollisionCfg(
                max_open_length=0.088, finger_length=0.078, point_sample_dense=0.012
            ),
            "generator_cfg": GraspGeneratorCfg(
                viser_port=11801,
                antipodal_sampler_cfg=AntipodalSamplerCfg(
                    n_sample=20000, max_length=0.088, min_length=0.003
                ),
            ),
        },
    )
    mug_semantics = ObjectSemantics(
        label="mug",
        geometry={
            "mesh_vertices": mug.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": mug.get_triangles(env_ids=[0])[0],
        },
        affordance=mug_grasp_affordance,
        entity=mug,  # needed so PickUpAction can read the mug's live pose
    )

    # ------------------------------------------------------------------ #
    # Step 6: Define target poses for place and final rest                #
    #                                                                     #
    # Poses are 4×4 homogeneous transforms (rotation | translation).     #
    # For PickUpAction the target is mug_semantics — the action computes  #
    # the grasp pose automatically from the affordance.                   #
    # ------------------------------------------------------------------ #
    # Place the mug 20 cm to the left and 40 cm forward from its pickup pose
    place_xpos = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022, 0.2489],
            [-0.9977, 0.0540, -0.0401, 0.3970],
            [0.0401, 0.0000, -0.9992, 0.2400],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        dtype=torch.float32,
        device=sim.device,
    )

    # Move the arm to a safe resting pose after placing
    rest_xpos = torch.tensor(
        [
            [-0.0539, -0.9985, -0.0022, 0.5000],
            [-0.9977, 0.0540, -0.0401, 0.0000],
            [0.0401, 0.0000, -0.9992, 0.5000],
            [0.0000, 0.0000, 0.0000, 1.0000],
        ],
        dtype=torch.float32,
        device=sim.device,
    )

    # ------------------------------------------------------------------ #
    # Step 7: Plan and execute the full sequence                          #
    #                                                                     #
    # execute_static() plans all three actions in order and returns a     #
    # single concatenated joint trajectory (n_envs, n_waypoints, dof).   #
    # We then replay it frame-by-frame in the simulator.                 #
    # ------------------------------------------------------------------ #
    print("Planning pick → place → move trajectory...")
    is_success, traj = atomic_engine.execute_static(
        target_list=[mug_semantics, place_xpos, rest_xpos]
    )

    if not is_success:
        print("Planning failed. Check that the target poses are reachable.")
        return

    print(f"Success! Replaying {traj.shape[1]} waypoints...")
    for i in range(traj.shape[1]):
        robot.set_qpos(traj[:, i])
        sim.update(step=4)
        time.sleep(1e-2)

    input("Press Enter to exit...")


if __name__ == "__main__":
    main()
