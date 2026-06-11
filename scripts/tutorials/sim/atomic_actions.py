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
either a pick-and-place task or an upright demo for a fallen bottle/cup.

Key concepts covered:
  1. Setting up a MotionGenerator and AtomicActionEngine
  2. Describing what to pick using ObjectSemantics and AntipodalAffordance
  3. Running a pick → place → move sequence with execute_static()

Run with:
    python atomic_actions.py [--num_envs N] [--renderer hybrid|fast-rt|rt]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import time
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot, RigidObject
from embodichain.lab.sim.objects.articulation import Articulation
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.sim.solvers import PytorchSolverCfg
from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RenderCfg,
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
    register_action,
    unregister_action,
)
from embodichain.lab.sim.atomic_actions.actions import UprightAction, UprightActionCfg


def parse_arguments():
    """
    Parse command-line arguments to configure the simulation.

    Returns:
        argparse.Namespace: Parsed arguments including number of environments, device, and rendering options.
    """
    parser = argparse.ArgumentParser(
        description="Create and simulate a robot in SimulationManager"
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--demo_mode",
        type=str,
        choices=["pick_place", "upright", "pick_mug"],
        default="pick_place",
        help="Select the tutorial scenario to run.",
    )
    parser.add_argument(
        "--object_kind",
        type=str,
        choices=["cup", "bottle"],
        default="bottle",
        help="Object to use in upright mode.",
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
    sim_cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=True,
        sim_device="cuda",
        physics_dt=1.0 / 100.0,
        num_envs=args.num_envs,
        render_cfg=RenderCfg(renderer=args.renderer),
    )
    original_set_default_background = SimulationManager.set_default_background
    SimulationManager.set_default_background = lambda self: None
    try:
        sim = SimulationManager(sim_cfg)
    finally:
        SimulationManager.set_default_background = original_set_default_background

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
    cfg.drive_pros.armature = None
    original_set_joint_drive = Articulation.set_joint_drive

    def _set_joint_drive_without_armature(self, *args, **kwargs):
        kwargs.pop("armature", None)
        return original_set_joint_drive(self, *args, **kwargs)

    Articulation.set_joint_drive = _set_joint_drive_without_armature
    try:
        return sim.add_robot(cfg=cfg)
    finally:
        Articulation.set_joint_drive = original_set_joint_drive


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


def create_fallen_cup(sim: SimulationManager) -> RigidObject:
    cup_cfg = RigidObjectCfg(
        uid="cup",
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
        init_rot=[90.0, 0.0, -90.0],
        body_scale=(4, 4, 4),
    )
    cup = sim.add_rigid_object(cfg=cup_cfg)
    return cup


def create_fallen_bottle(sim: SimulationManager) -> RigidObject:
    bottle_cfg = RigidObjectCfg(
        uid="bottle",
        shape=MeshCfg(
            fpath=get_data_path("ScannedBottle/yibao.ply"),
        ),
        attrs=RigidBodyAttributesCfg(
            mass=0.02,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[0.55, 0.0, 0.01],
        init_rot=[90.0, 0.0, 0.0],
        body_scale=(0.001, 0.001, 0.001),
    )
    bottle = sim.add_rigid_object(cfg=bottle_cfg)
    return bottle


def build_grasp_affordance(object_label: str) -> AntipodalAffordance:
    return AntipodalAffordance(
        object_label=object_label,
        force_reannotate=True,
        custom_config={
            "gripper_collision_cfg": GripperCollisionCfg(
                max_open_length=0.088, finger_length=0.078, point_sample_dense=0.012
            ),
            "generator_cfg": GraspGeneratorCfg(
                viser_port=11801,
                antipodal_sampler_cfg=AntipodalSamplerCfg(
                    n_sample=2000, max_length=0.088, min_length=0.003
                ),
            ),
        },
    )


def create_object_semantics(
    obj: RigidObject, *, label: str, object_label: str
) -> ObjectSemantics:
    return ObjectSemantics(
        label=label,
        geometry={
            "mesh_vertices": obj.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": obj.get_triangles(env_ids=[0])[0],
        },
        affordance=build_grasp_affordance(object_label),
        entity=obj,
    )


def main():
    """Run the pick/place demo or the upright bottle/cup demo."""
    args = parse_arguments()

    # ------------------------------------------------------------------ #
    # Step 1: Set up simulation and robot                                 #
    # ------------------------------------------------------------------ #
    sim: SimulationManager = initialize_simulation(args)
    robot = create_robot(sim)

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

    base_action_kwargs = dict(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
    )
    grasp_action_kwargs = dict(
        **base_action_kwargs,
        pre_grasp_distance=0.15,
        lift_height=0.15,
    )
    place_action_kwargs = dict(
        **base_action_kwargs,
        lift_height=0.15,
    )

    # ------------------------------------------------------------------ #
    # Step 3: Define target poses for place and final rest                #
    #                                                                     #
    # Poses are 4×4 homogeneous transforms (rotation | translation).     #
    # For PickUpAction/UprightAction the target is object semantics —     #
    # the action computes the grasp pose automatically from the affordance.
    # ------------------------------------------------------------------ #
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

    if args.demo_mode == "pick_place":
        demo_object = create_mug(sim)
        demo_semantics = create_object_semantics(
            demo_object, label="mug", object_label="mug"
        )
        action_cfgs = [
            PickUpActionCfg(**grasp_action_kwargs),
            PlaceActionCfg(**place_action_kwargs),
            MoveActionCfg(control_part="arm"),
        ]
        target_list = [demo_semantics, place_xpos, rest_xpos]
        planning_label = "pick → place → move"
        action_registry_name = None
    elif args.demo_mode == "pick_mug":
        demo_object = create_mug(sim)
        demo_semantics = create_object_semantics(
            demo_object, label="mug", object_label="mug"
        )
        action_cfgs = [
            PickUpActionCfg(**grasp_action_kwargs),
        ]
        target_list = [demo_semantics]
        planning_label = "pick mug"
        action_registry_name = None
    else:
        if args.object_kind == "cup":
            demo_object = create_fallen_cup(sim)
        else:
            demo_object = create_fallen_bottle(sim)

        demo_semantics = create_object_semantics(
            demo_object, label=args.object_kind, object_label=args.object_kind
        )
        action_cfgs = [
            UprightActionCfg(**grasp_action_kwargs),
            MoveActionCfg(control_part="arm"),
        ]
        target_list = [demo_semantics, rest_xpos]
        planning_label = f"upright {args.object_kind} → move"
        action_registry_name = "upright"

    # ------------------------------------------------------------------ #
    # Step 4: Build the AtomicActionEngine                                #
    #                                                                     #
    # actions_cfg_list defines the ORDER of actions that execute_static() #
    # will run. Each entry is matched positionally to target_list.        #
    # ------------------------------------------------------------------ #
    if action_registry_name is not None:
        register_action(action_registry_name, UprightAction)

    try:
        atomic_engine = AtomicActionEngine(
            motion_generator=motion_gen,
            actions_cfg_list=action_cfgs,
        )

        sim.init_gpu_physics()
        if not args.headless:
            sim.open_window()
            if args.demo_mode == "upright":
                input(
                    f"Inspect the fallen {args.object_kind} in the window, then press Enter to start..."
                )
            elif args.demo_mode == "pick_mug":
                input("Inspect the mug in the window, then press Enter to start...")

        # ------------------------------------------------------------------ #
        # Step 5: Plan and execute the full sequence                          #
        #                                                                     #
        # execute_static() plans all three actions in order and returns a     #
        # single concatenated joint trajectory (n_envs, n_waypoints, dof).   #
        # We then replay it frame-by-frame in the simulator.                 #
        # ------------------------------------------------------------------ #
        print(f"Planning {planning_label} trajectory...")
        is_success, traj = atomic_engine.execute_static(target_list=target_list)

        if not is_success:
            print("Planning failed. Check that the target poses are reachable.")
            return

        print(f"Success! Replaying {traj.shape[1]} waypoints...")
        for i in range(traj.shape[1]):
            robot.set_qpos(traj[:, i])
            sim.update(step=4)
            time.sleep(1e-2)

        if not args.headless:
            input("Press Enter to exit...")
    finally:
        if action_registry_name is not None:
            unregister_action(action_registry_name)


if __name__ == "__main__":
    main()
