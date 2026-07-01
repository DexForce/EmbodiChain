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

"""Demonstrate moving a held object to an object-centric target pose."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    EndEffectorPoseTarget,
    GraspTarget,
    HeldObjectPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
    MoveHeldObject,
    MoveHeldObjectCfg,
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
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.toolkits.graspkit.pg_grasp.antipodal_generator import (
    AntipodalSamplerCfg,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
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

OBJECT_LABEL = "sugar_box"
OBJECT_MESH_PATH = "SugarBox/sugar_box_usd/sugar_box.usda"
OBJECT_XY = (-0.42, -0.08)
OBJECT_APPROACH_DIRECTION = (0.0, 0.0, -1.0)
OBJECT_MIN_HAND_CLOSE_QPOS = 0.024
OBJECT_INIT_ROT = (0.0, 0.0, 0.0)
OBJECT_BODY_SCALE = (0.8, 0.8, 0.8)
OBJECT_MASS = 0.05
OBJECT_USE_USD_PROPERTIES = False

MOVE_SAMPLE_INTERVAL = 60
PICK_SAMPLE_INTERVAL = 120
MOVE_HELD_OBJECT_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveHeldObject holding a sugar box in the gripper."
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
        device=args.device,
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
        shape=MeshCfg(fpath=get_data_path(OBJECT_MESH_PATH)),
        attrs=RigidBodyAttributesCfg(
            mass=OBJECT_MASS,
            dynamic_friction=0.97,
            static_friction=0.99,
        ),
        max_convex_hull_num=16,
        init_pos=[OBJECT_XY[0], OBJECT_XY[1], 0.0],
        init_rot=OBJECT_INIT_ROT,
        body_scale=OBJECT_BODY_SCALE,
        use_usd_properties=OBJECT_USE_USD_PROPERTIES,
    )
    obj = sim.add_rigid_object(cfg=cfg)

    sim.update(
        step=10
    )  # Settle the object to ensure it is resting on the ground before planning
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


def make_pre_pick_eef_pose(robot: Robot, position: torch.Tensor) -> torch.Tensor:
    pose = robot.compute_fk(
        qpos=robot.get_qpos(name="arm"),
        name="arm",
        to_matrix=True,
    )[0].clone()
    pose[:3, 3] = position
    return pose


def make_object_target_pose(device: torch.device) -> torch.Tensor:
    pose = torch.eye(4, dtype=torch.float32, device=device)
    pose[:3, :3] = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    pose[:3, 3] = torch.tensor([-0.3, -0.3, 0.5], dtype=torch.float32, device=device)
    return pose


def compute_pick_close_end_step() -> int:
    motion_waypoints = PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS
    n_approach = int(round(motion_waypoints) * 0.6)
    return MOVE_SAMPLE_INTERVAL + n_approach + HAND_INTERP_STEPS


def main() -> None:
    """Pick up an object and move the held object to an object-frame target."""
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
    # Step 3: Configure the three atomic actions                          #
    # ------------------------------------------------------------------ #
    hand_open, hand_close = get_hand_open_close_qpos(robot, sim.device)
    move_cfg = MoveEndEffectorCfg(
        control_part="arm",
        sample_interval=MOVE_SAMPLE_INTERVAL,
    )
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
    move_held_object_cfg = MoveHeldObjectCfg(
        control_part="arm",
        hand_control_part="hand",
        hand_close_qpos=hand_close,
        sample_interval=MOVE_HELD_OBJECT_SAMPLE_INTERVAL,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Build the AtomicActionEngine                                #
    # ------------------------------------------------------------------ #
    atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
    atomic_engine.register(MoveEndEffector(motion_gen, cfg=move_cfg))
    atomic_engine.register(PickUp(motion_gen, cfg=pickup_cfg))
    atomic_engine.register(MoveHeldObject(motion_gen, cfg=move_held_object_cfg))

    # ------------------------------------------------------------------ #
    # Step 5: Describe the object and define the motion targets           #
    # ------------------------------------------------------------------ #
    semantics = create_object_semantics(obj, args)
    obj_pose = obj.get_local_pose(to_matrix=True)
    move_position = obj_pose[0, :3, 3].clone()
    move_position[2] = 0.36
    move_target = make_pre_pick_eef_pose(robot, move_position)
    object_target_pose = make_object_target_pose(sim.device)
    move_held_object_target = HeldObjectPoseTarget(
        object_target_pose=object_target_pose
    )

    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "move_held_object_target_axis", object_target_pose)
    if not args.auto_play:
        input("Inspect the sugar box, then press Enter to plan...")

    # ------------------------------------------------------------------ #
    # Step 6: Plan the declared (name, typed_target) sequence             #
    # ------------------------------------------------------------------ #
    logger.log_info("Planning move_end_effector -> pick_up -> move_held_object")
    start_time = time.time()
    is_success, traj, _ = atomic_engine.run(
        steps=[
            ("move_end_effector", EndEffectorPoseTarget(xpos=move_target)),
            ("pick_up", GraspTarget(semantics=semantics)),
            ("move_held_object", move_held_object_target),
        ]
    )
    cost_time = time.time() - start_time
    logger.log_info(f"Plan trajectory cost time: {cost_time:.2f} seconds")
    if not is_success:
        logger.log_warning("Failed to plan move_held_object demo trajectory.")
        return

    if not args.auto_play:
        input("Press Enter to replay the move_held_object demo...")

    # ------------------------------------------------------------------ #
    # Step 7: Replay the planned trajectory                               #
    # ------------------------------------------------------------------ #
    recording_started = start_auto_play_recording(
        sim, args, video_prefix="move_held_object_auto_play"
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

        logger.log_info("MoveHeldObject keeps the sugar box suspended in the gripper.")

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
