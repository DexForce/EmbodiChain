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

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    GraspTarget,
    PickUp,
    PickUpCfg,
    Place,
    PlaceCfg,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    clone_local_pose_from_first_env,
    create_antipodal_semantics,
    create_toppra_motion_generator,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    prepare_tutorial_scene,
    replay_trajectory,
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
    parser = argparse.ArgumentParser(
        description="Pick up a cube and place it at a target pose."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--n_sample", type=int, default=10000)
    parser.add_argument("--force_reannotate", action="store_true")
    parser.add_argument("--auto_play", action="store_true")
    parser.add_argument("--no_vis_eef_axis", action="store_true")
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
    robot = add_ur5_gripper_robot(sim)
    obj = create_pick_object(sim)
    motion_gen = create_toppra_motion_generator(robot)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)

    engine = AtomicActionEngine(motion_generator=motion_gen)
    engine.register(
        PickUp(
            motion_gen,
            cfg=PickUpCfg(
                hand_open_qpos=hand_open,
                hand_close_qpos=hand_close,
                pre_grasp_distance=0.15,
                lift_height=0.16,
                sample_interval=PICK_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
    )
    engine.register(
        Place(
            motion_gen,
            cfg=PlaceCfg(
                hand_open_qpos=hand_open,
                hand_close_qpos=hand_close,
                lift_height=PLACE_LIFT_HEIGHT,
                sample_interval=PLACE_SAMPLE_INTERVAL,
                hand_interp_steps=HAND_INTERP_STEPS,
            ),
        )
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

    success, trajectory, _ = engine.run(
        [
            ("pick_up", GraspTarget(semantics)),
            (
                "place",
                EndEffectorPoseTarget(
                    broadcast_waypoint_pose_batch(
                        place_poses, robot.get_qpos().shape[0]
                    )
                ),
            ),
        ]
    )
    if not success.all():
        logger.log_warning("Failed to plan Place demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the Place demo...")
    clear_after_step = (
        round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6) + HAND_INTERP_STEPS
    )
    dynamics_cleared = False

    def clear_object_dynamics(step_idx: int, _: int) -> None:
        nonlocal dynamics_cleared
        if not dynamics_cleared and step_idx + 1 >= clear_after_step:
            obj.clear_dynamics()
            dynamics_cleared = True

    replay_trajectory(
        sim,
        robot,
        trajectory,
        args,
        video_prefix="place_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=clear_object_dynamics,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
