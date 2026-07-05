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

"""Demonstrate Place after a PickUp precondition has created held-object state."""

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
    EndEffectorPoseTarget,
    GraspTarget,
    ObjectSemantics,
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
)
from embodichain.lab.sim.cfg import (
    LightCfg,
    RenderCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.planners import MotionGenerator, MotionGenCfg, ToppraPlannerCfg
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
    broadcast_waypoint_pose_batch,
    clone_local_pose_from_first_env,
    create_ur5_gripper_robot_cfg,
    draw_axis_marker,
    get_tutorial_window_size,
    start_auto_play_recording,
    stop_auto_play_recording,
)

GRIPPER_MAX_OPEN_WIDTH = 0.080
GRIPPER_FINGER_LENGTH = 0.088
GRIPPER_ROOT_Z_WIDTH = 0.096
GRIPPER_Y_THICKNESS = 0.040

OBJECT_LABEL = "cube"
OBJECT_SIZE = (0.05, 0.05, 0.05)
OBJECT_XY = (-0.42, -0.08)
OBJECT_MIN_HAND_CLOSE_QPOS = 0.024
OBJECT_APPROACH_DIRECTION = (0.0, 0.0, -1.0)
OBJECT_INIT_ROT = (0.0, 0.0, 0.0)
OBJECT_BODY_SCALE = (1.0, 1.0, 1.0)
OBJECT_MASS = 0.05
OBJECT_USE_USD_PROPERTIES = False

PICK_SAMPLE_INTERVAL = 120
PLACE_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
PLACE_LIFT_HEIGHT = 0.14


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate Place by picking an object and releasing it at a target pose."
    )
    add_env_launcher_args_to_parser(parser)
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


def create_robot(sim: SimulationManager, position=(0.0, 0.0, 0.0)) -> Robot:
    cfg = create_ur5_gripper_robot_cfg(init_pos=position)
    return sim.add_robot(cfg=cfg)


def create_pick_object(sim: SimulationManager) -> RigidObject:
    cfg = RigidObjectCfg(
        uid=OBJECT_LABEL,
        shape=CubeCfg(size=list(OBJECT_SIZE)),
        attrs=RigidBodyAttributesCfg(
            mass=OBJECT_MASS,
            dynamic_friction=0.97,
            static_friction=0.99,
            enable_ccd=True,
        ),
        max_convex_hull_num=16,
        init_pos=[OBJECT_XY[0], OBJECT_XY[1], 0.5 * OBJECT_SIZE[2]],
        init_rot=OBJECT_INIT_ROT,
        body_scale=OBJECT_BODY_SCALE,
        use_usd_properties=OBJECT_USE_USD_PROPERTIES,
    )
    obj = sim.add_rigid_object(cfg=cfg)

    # Set the object to a stable pose on the ground by simulating a few steps.
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def get_hand_open_close_qpos(
    robot: Robot, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    hand_limits = robot.get_qpos_limits(name="hand")[0].to(
        device=device, dtype=torch.float32
    )
    hand_open = hand_limits[:, 0]
    hand_close_limit = hand_limits[:, 1]
    hand_close = torch.minimum(
        hand_close_limit,
        torch.full_like(hand_close_limit, OBJECT_MIN_HAND_CLOSE_QPOS),
    )
    return hand_open, hand_close


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
        root_z_width=GRIPPER_ROOT_Z_WIDTH,
        open_check_margin=0.002,
        point_sample_dense=0.012,
    )


def create_object_semantics(
    obj: RigidObject, args: argparse.Namespace
) -> ObjectSemantics:
    return ObjectSemantics(
        label=OBJECT_LABEL,
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


def make_place_eef_poses(device: torch.device) -> torch.Tensor:
    """Build a multi-waypoint place trajectory ``(n_waypoint, 4, 4)``.

    Two waypoints are returned: a higher hover pose and the final release pose.
    ``Place`` approaches from above the first waypoint, descends through each
    waypoint in order, opens the gripper at the last, and retracts — so this
    exercises the multi-waypoint descent path.
    """
    rotation = torch.tensor(
        [
            [0.0539, 0.9985, -0.0022],
            [0.9977, -0.0540, -0.0401],
            [-0.0401, -0.0000, -0.9992],
        ],
        dtype=torch.float32,
        device=device,
    )
    hover_pose = torch.eye(4, dtype=torch.float32, device=device)
    hover_pose[:3, :3] = rotation
    hover_pose[:3, 3] = torch.tensor(
        [-0.40, 0.48, 0.20], dtype=torch.float32, device=device
    )
    place_pose = torch.eye(4, dtype=torch.float32, device=device)
    place_pose[:3, :3] = rotation
    place_pose[:3, 3] = torch.tensor(
        [-0.40, 0.48, 0.10], dtype=torch.float32, device=device
    )
    return torch.stack([hover_pose, place_pose], dim=0)


def compute_pick_close_end_step() -> int:
    motion_waypoints = PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS
    n_approach = int(round(motion_waypoints) * 0.6)
    return n_approach + HAND_INTERP_STEPS


def main() -> None:
    """Pick up an object and place it at a target end-effector pose."""
    args = parse_arguments()

    # ------------------------------------------------------------------ #
    # Step 1: Set up simulation, robot, and object                 #
    # ------------------------------------------------------------------ #
    sim = initialize_simulation(args)
    robot = create_robot(sim)
    obj = create_pick_object(sim)

    if not args.headless:
        sim.open_window()

    # ------------------------------------------------------------------ #
    # Step 2: Create a MotionGenerator for the robot                      #
    # ------------------------------------------------------------------ #
    motion_gen = MotionGenerator(
        cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=robot.uid))
    )

    # ------------------------------------------------------------------ #
    # Step 3: Configure the PickUp and Place atomic actions               #
    # ------------------------------------------------------------------ #
    hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    pickup_cfg = PickUpCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        approach_direction=torch.tensor(
            OBJECT_APPROACH_DIRECTION, dtype=torch.float32, device=sim.device
        ),
        pre_grasp_distance=0.15,
        lift_height=0.16,
        sample_interval=PICK_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )
    place_cfg = PlaceCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_open_qpos=hand_open,
        hand_close_qpos=hand_close,
        lift_height=PLACE_LIFT_HEIGHT,
        sample_interval=PLACE_SAMPLE_INTERVAL,
        hand_interp_steps=HAND_INTERP_STEPS,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Build the AtomicActionEngine                                #
    # ------------------------------------------------------------------ #
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(PickUp(motion_gen, cfg=pickup_cfg))
    atomic_engine.register(Place(motion_gen, cfg=place_cfg))

    # ------------------------------------------------------------------ #
    # Step 5: Describe the object and define the place target             #
    # ------------------------------------------------------------------ #
    semantics = create_object_semantics(obj, args)
    place_eef_poses = make_place_eef_poses(sim.device)

    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "place_target_axis", place_eef_poses[-1])
    if not args.auto_play:
        input("Inspect the object, then press Enter to plan PickUp -> Place...")

    # ------------------------------------------------------------------ #
    # Step 6: Plan the declared (name, typed_target) sequence             #
    # ------------------------------------------------------------------ #
    # Pass a multi-waypoint trajectory (n_envs, n_waypoint, 4, 4): Place
    # approaches from above the first waypoint, descends through each
    # waypoint in order, opens the gripper at the last, and retracts.
    n_envs = robot.get_qpos().shape[0]
    multi_waypoint_xpos = broadcast_waypoint_pose_batch(
        place_eef_poses, num_envs=n_envs
    )
    place_target = EndEffectorPoseTarget(xpos=multi_waypoint_xpos)
    logger.log_info("Planning PickUp precondition -> Place release trajectory")
    is_success, traj, _ = atomic_engine.run(
        steps=[
            ("pick_up", GraspTarget(semantics=semantics)),
            ("place", place_target),
        ]
    )
    if not is_success.all():
        logger.log_warning("Failed to plan Place demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the Place demo...")

    # ------------------------------------------------------------------ #
    # Step 7: Replay the planned trajectory                               #
    # ------------------------------------------------------------------ #
    recording_started = start_auto_play_recording(
        sim, args, video_prefix="place_auto_play"
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

        logger.log_info("Place opens the gripper and clears WorldState.held_object.")
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
