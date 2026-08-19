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
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionInvocation,
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    MotionPolicy,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    make_clear_dynamics_callback,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

OBJECT_SIZE = (0.05, 0.05, 0.05)
OBJECT_XY = (-0.42, -0.08)
PICK_SAMPLE_INTERVAL = 120
PLACE_SAMPLE_INTERVAL = 120
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
PLACE_LIFT_HEIGHT = 0.14


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Place tutorial."""
    parser = create_tutorial_argument_parser(
        "Pick up a cube and place it at a target pose.",
        features=("grasp_sampling", "visualize_axes"),
    )
    return parser.parse_args()


def create_pick_object(sim) -> RigidObject:
    """Create a settled cube for the PickUp and Place sequence."""
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=list(OBJECT_SIZE)),
            attrs=RigidBodyAttributesCfg(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
                enable_ccd=True,
            ),
            max_convex_hull_num=16,
            init_pos=[*OBJECT_XY, 0.5 * OBJECT_SIZE[2]],
        )
    )
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def make_place_eef_poses(device: torch.device) -> torch.Tensor:
    """Build hover and release waypoints for the multi-waypoint Place target."""
    rotation = torch.tensor(
        [
            [0.0539, 0.9985, -0.0022],
            [0.9977, -0.0540, -0.0401],
            [-0.0401, 0.0, -0.9992],
        ],
        dtype=torch.float32,
        device=device,
    )
    poses = []
    for position in ((-0.40, 0.48, 0.20), (-0.40, 0.48, 0.10)):
        pose = torch.eye(4, dtype=torch.float32, device=device)
        pose[:3, :3], pose[:3, 3] = rotation, torch.tensor(position, device=device)
        poses.append(pose)
    return torch.stack(poses)


def main() -> None:
    """Plan and replay PickUp followed by a multi-waypoint Place."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    obj = create_pick_object(sim)
    motion_gen = create_curobo_motion_generator(robot)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)

    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
    )
    semantics = create_antipodal_semantics(
        obj,
        label="cube",
        n_sample=args.n_sample,
        force_reannotate=args.force_reannotate,
    )
    place_poses = make_place_eef_poses(sim.device)
    if not args.no_vis_eef_axis:
        draw_axis_marker(
            sim,
            "place_target_axis",
            broadcast_pose_batch(place_poses[-1], robot.get_qpos().shape[0]),
        )
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the cube, then press Enter to plan PickUp -> Place..."
    )

    endpoint_mapping = {"primary": {"motion": "arm", "grasp": "hand"}}
    pick_binding = engine.bind_control_parts(
        "pick_up",
        endpoint_mapping,
    )
    place_binding = engine.bind_control_parts(
        "place",
        endpoint_mapping,
    )
    compiled = engine.compile(
        (
            ActionInvocation(
                "pick_up",
                GraspGoal(semantics),
                pick_binding,
                MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PICK_SAMPLE_INTERVAL,
                ),
                skill_options=PickUpOptions(
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
            ActionInvocation(
                "place",
                PlaceGoal(
                    broadcast_waypoint_pose_batch(
                        place_poses, robot.get_qpos().shape[0]
                    )
                ),
                place_binding,
                MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PLACE_SAMPLE_INTERVAL,
                ),
                skill_options=PlaceOptions(
                    lift_height=PLACE_LIFT_HEIGHT,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
        )
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan Place demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the Place demo...")
    clear_after_step = compiled.segment(0, "lift").start
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="place_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=make_clear_dynamics_callback(obj, clear_after_step),
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
