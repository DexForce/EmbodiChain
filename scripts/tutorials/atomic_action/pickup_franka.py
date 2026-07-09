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

"""Demonstrate PickUp on an upright object using a Franka Panda robot."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    GraspTarget,
    ObjectSemantics,
    PickUp,
    PickUpCfg,
)
from embodichain.lab.sim.cfg import (
    LightCfg,
    RenderCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
from embodichain.lab.sim.robots import FrankaPandaCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    clone_local_pose_from_first_env,
    draw_axis_marker,
    get_tutorial_window_size,
    start_auto_play_recording,
    stop_auto_play_recording,
)

# Franka Panda hand geometry (parallel-jaw gripper with two prismatic fingers).
GRIPPER_MAX_OPEN_WIDTH = 0.080
GRIPPER_FINGER_LENGTH = 0.050
GRIPPER_ROOT_Z_WIDTH = 0.050
GRIPPER_Y_THICKNESS = 0.030
GRIPPER_X_THICKNESS = 0.020

# Cap the closing qpos so the fingers do not close tighter than this value.
OBJECT_MIN_HAND_CLOSE_QPOS = 0.00
# Fully-open finger joint value for the Panda hand.
FRANKA_FINGER_OPEN_QPOS = 0.04

OBJECT_XY = (0.42, 0.08)

OBJECT_PRESETS = {
    "cube": {
        "label": "cube",
        "cube_size": (0.05, 0.05, 0.05),
        "init_z": 0.05,
        "init_rot": (0.0, 0.0, 0.0),
        "body_scale": (1.0, 1.0, 1.0),
        "mass": 0.05,
        "use_usd_properties": False,
    },
}

PICK_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240

APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "side": (0.0, 1.0, 0.0),
    "side_y": (0.0, -1.0, 0.0),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate PickUp on an upright object with a Franka Panda."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--object",
        choices=sorted(OBJECT_PRESETS.keys()),
        default="cube",
        help="Object preset to pick.",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=10000,
        help="Number of samples for antipodal grasp generation.",
    )
    parser.add_argument(
        "--force_reannotate",
        action="store_true",
        help="Force grasp region re-annotation instead of using cached data.",
    )
    parser.add_argument(
        "--auto_play",
        action="store_true",
        help="Run the viewer demo without waiting for keyboard input.",
    )
    parser.add_argument(
        "--approach",
        choices=["top", "side", "side_y", "custom"],
        default="top",
        help="Pick approach direction preset.",
    )
    parser.add_argument(
        "--custom_approach_direction",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="World-frame approach direction used when --approach custom.",
    )
    parser.add_argument(
        "--no_vis_eef_axis",
        action="store_true",
        help="Do not draw the current end-effector/TCP coordinate frame before planning.",
    )
    return parser.parse_args()


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    width, height = get_tutorial_window_size(args)
    cfg = SimulationManagerCfg(
        width=width,
        height=height,
        headless=True,
        num_envs=args.num_envs,
        sim_device=args.device,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_dt=1.0 / 100.0,
        arena_space=2.5,
    )
    sim = SimulationManager(cfg)
    sim.add_light(
        cfg=LightCfg(
            uid="main_light",
            color=(0.6, 0.6, 0.6),
            intensity=30.0,
            init_pos=(1.0, 0.0, 3.0),
        )
    )
    return sim


def create_robot(sim: SimulationManager) -> Robot:
    cfg = FrankaPandaCfg.from_dict({"robot_type": "panda"})
    robot = sim.add_robot(cfg=cfg)
    return robot


def create_pick_object(sim: SimulationManager, object_name: str) -> RigidObject:
    preset = OBJECT_PRESETS[object_name]
    cfg = RigidObjectCfg(
        uid=preset["label"],
        shape=CubeCfg(size=list(preset["cube_size"])),
        attrs=RigidBodyAttributesCfg(
            mass=preset["mass"],
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[OBJECT_XY[0], OBJECT_XY[1], preset["init_z"]],
        init_rot=preset["init_rot"],
        body_scale=preset["body_scale"],
        use_usd_properties=preset["use_usd_properties"],
    )
    obj = sim.add_rigid_object(cfg=cfg)

    # Settle the object to ensure it is resting on the ground before planning
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def build_grasp_generator_cfg(args: argparse.Namespace) -> GraspGeneratorCfg:
    return GraspGeneratorCfg(
        viser_port=11801,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=args.n_sample,
            max_length=GRIPPER_MAX_OPEN_WIDTH,
            min_length=0.003,
        ),
        is_partial_annotate=False,
        is_filter_ground_collision=False,
    )


def build_gripper_collision_cfg() -> GripperCollisionCfg:
    return GripperCollisionCfg(
        max_open_length=GRIPPER_MAX_OPEN_WIDTH,
        finger_length=GRIPPER_FINGER_LENGTH,
        y_thickness=GRIPPER_Y_THICKNESS,
        x_thickness=GRIPPER_X_THICKNESS,
        root_z_width=GRIPPER_ROOT_Z_WIDTH,
        open_check_margin=0.002,
        point_sample_dense=0.012,
    )


def create_object_semantics(
    obj: RigidObject, args: argparse.Namespace
) -> ObjectSemantics:
    label = OBJECT_PRESETS[args.object]["label"]
    # All environments share the same object geometry and pose (see
    # clone_local_pose_from_first_env), so reading env 0 is sufficient.
    return ObjectSemantics(
        label=label,
        geometry={
            "mesh_vertices": obj.get_vertices(env_ids=[0], scale=True)[0],
            "mesh_triangles": obj.get_triangles(env_ids=[0])[0],
        },
        affordance=AntipodalAffordance(
            mesh_vertices=obj.get_vertices(env_ids=[0], scale=True)[0],
            mesh_triangles=obj.get_triangles(env_ids=[0])[0],
            gripper_collision_cfg=build_gripper_collision_cfg(),
            generator_cfg=build_grasp_generator_cfg(args),
            force_reannotate=args.force_reannotate,
        ),
        entity=obj,
    )


def get_hand_open_close_qpos(
    robot: Robot, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    hand_limits = robot.get_qpos_limits(name="hand")[0].to(
        device=device, dtype=torch.float32
    )
    hand_open = torch.full_like(hand_limits[:, 0], FRANKA_FINGER_OPEN_QPOS)
    hand_open = torch.clamp(hand_open, min=hand_limits[:, 0], max=hand_limits[:, 1])
    hand_close = torch.clamp(
        torch.full_like(hand_limits[:, 1], OBJECT_MIN_HAND_CLOSE_QPOS),
        min=hand_limits[:, 0],
        max=hand_limits[:, 1],
    )
    return hand_open, hand_close


def resolve_approach_direction(
    args: argparse.Namespace, device: torch.device
) -> torch.Tensor:
    if args.approach == "custom":
        if args.custom_approach_direction is None:
            raise ValueError(
                "--custom_approach_direction is required when --approach custom."
            )
        direction = args.custom_approach_direction
    else:
        direction = APPROACH_DIRECTIONS[args.approach]

    approach_direction = torch.tensor(direction, dtype=torch.float32, device=device)
    norm = torch.linalg.norm(approach_direction)
    if norm < 1e-6:
        raise ValueError("approach_direction must be non-zero.")
    return approach_direction / norm


def make_pre_pick_eef_pose(robot: Robot, position: torch.Tensor) -> torch.Tensor:
    pose = robot.compute_fk(
        qpos=robot.get_qpos(name="arm"),
        name="arm",
        to_matrix=True,
    ).clone()
    pose[:, :3, 3] = position
    return pose


def initialize_pre_pick_robot_pose(
    robot: Robot,
    obj: RigidObject,
    hand_open: torch.Tensor,
) -> None:
    obj_pose = obj.get_local_pose(to_matrix=True)
    move_position = obj_pose[:, :3, 3].clone()
    move_position[:, 2] = 0.36
    pre_pick_pose = make_pre_pick_eef_pose(robot, move_position)
    ik_success, arm_qpos = robot.compute_ik(
        pose=pre_pick_pose,
        joint_seed=robot.get_qpos(name="arm"),
        name="arm",
    )
    if not torch.all(ik_success):
        raise RuntimeError("Failed to initialize the robot at the pre-pick pose.")

    n_envs = robot.get_qpos().shape[0]
    hand_qpos = hand_open.unsqueeze(0).repeat(n_envs, 1)
    for target in (False, True):
        robot.set_qpos(arm_qpos, name="arm", target=target)
        robot.set_qpos(hand_qpos, name="hand", target=target)
    robot.clear_dynamics()


def compute_pick_close_end_step() -> int:
    motion_waypoints = PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS
    n_approach = int(round(motion_waypoints) * 0.6)
    return n_approach + HAND_INTERP_STEPS


def format_tensor(tensor: torch.Tensor) -> str:
    rounded = (tensor.detach().cpu() * 10000.0).round() / 10000.0
    return str(rounded.tolist())


def draw_pick_object_axis(sim: SimulationManager, obj: RigidObject) -> None:
    # Visualize the object frame only for the first environment to keep the
    # viewer uncluttered when running with num_envs > 1.
    draw_axis_marker(
        sim,
        "pickup_object_axis",
        obj.get_local_pose(to_matrix=True)[:1],
    )


def main() -> None:
    """Pick up an object using an antipodal grasp affordance on Franka Panda."""
    args = parse_arguments()

    # ------------------------------------------------------------------ #
    # Step 1: Set up simulation, robot, and object                       #
    # ------------------------------------------------------------------ #
    sim = initialize_simulation(args)
    robot = create_robot(sim)
    print("**********tcp xpos:\n", robot._solvers["arm"].tcp_xpos)
    obj = create_pick_object(sim, args.object)

    # ------------------------------------------------------------------ #
    # Step 2: Create a MotionGenerator for the robot                     #
    # ------------------------------------------------------------------ #
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    # ------------------------------------------------------------------ #
    # Step 3: Configure the PickUp atomic action                         #
    # ------------------------------------------------------------------ #
    hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)
    approach_direction = resolve_approach_direction(args, sim.device)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    pickup_cfg = PickUpCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        approach_direction=approach_direction,
        pre_grasp_distance=0.15,
        lift_height=0.16,
        sample_interval=PICK_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Build the AtomicActionEngine                                #
    # ------------------------------------------------------------------ #
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(PickUp(motion_gen, cfg=pickup_cfg))

    # ------------------------------------------------------------------ #
    # Step 5: Describe the object with ObjectSemantics                    #
    # ------------------------------------------------------------------ #
    semantics = create_object_semantics(obj, args)

    if not args.headless:
        sim.open_window()
    if not args.no_vis_eef_axis:
        draw_pick_object_axis(sim, obj)
    if not args.auto_play:
        input(f"Inspect the upright {args.object}, then press Enter to plan...")

    # ------------------------------------------------------------------ #
    # Step 6: Plan the declared (name, typed_target) sequence             #
    # ------------------------------------------------------------------ #
    logger.log_info(
        f"Planning pick_up for {args.object} with "
        f"approach_direction={format_tensor(approach_direction)}"
    )
    start_time = time.time()
    is_success, traj, _ = atomic_engine.run(
        steps=[("pick_up", GraspTarget(semantics=semantics))]
    )
    cost_time = time.time() - start_time
    logger.log_info(f"Plan trajectory cost time: {cost_time:.2f} seconds")
    if not is_success.all():
        logger.log_warning("Failed to plan pickup demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the pickup demo...")

    # ------------------------------------------------------------------ #
    # Step 7: Replay the planned trajectory                               #
    # ------------------------------------------------------------------ #
    recording_started = start_auto_play_recording(
        sim, args, video_prefix=f"pickup_franka_{args.object}_auto_play"
    )
    try:
        post_grasp_clear_step = compute_pick_close_end_step()
        should_clear_object_dynamics = True
        for i in range(traj.shape[1]):
            robot.set_qpos(traj[:, i, :])
            sim.update(step=4)
            if should_clear_object_dynamics and i + 1 >= post_grasp_clear_step:
                obj.clear_dynamics()
                should_clear_object_dynamics = False
                logger.log_info(f"Object dynamics cleared after grasp at step={i}")
            time.sleep(1e-2)

        logger.log_info(
            f"PickUp keeps the upright {args.object} suspended in the gripper."
        )

        final_qpos = traj[:, -1, :]
        for i in range(POST_TRAJECTORY_STEPS):
            robot.set_qpos(final_qpos)
            sim.update(step=2)
            time.sleep(1e-2)
    finally:
        stop_auto_play_recording(sim, recording_started)

    if not args.auto_play:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
