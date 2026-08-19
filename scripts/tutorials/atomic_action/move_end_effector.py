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

"""Demonstrate MoveEndEffector with a multi-waypoint pose trajectory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AtomicActionEngine,
    EndEffectorPoseGoal,
    MotionPolicy,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    create_curobo_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

MOVE_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the MoveEndEffector tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate MoveEndEffector with a multi-waypoint pose trajectory.",
        features=("visualize_axes",),
    )
    return parser.parse_args()


def main() -> None:
    """Move the robot end effector through two pose waypoints."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    motion_gen = create_curobo_motion_generator(robot)

    engine = AtomicActionEngine(motion_generator=motion_gen)

    start_pose = robot.compute_fk(
        robot.get_qpos(name="arm"), name="arm", to_matrix=True
    )[0]
    poses = start_pose.unsqueeze(0).repeat(2, 1, 1)
    poses[:, :3, 3] += torch.tensor(
        [[-0.08, -0.08, 0.08], [0.04, 0.12, 0.04]],
        dtype=poses.dtype,
        device=poses.device,
    )
    num_envs = robot.get_qpos().shape[0]
    if not args.no_vis_eef_axis:
        for name, pose in zip(("target", "side"), poses, strict=True):
            draw_axis_marker(
                sim,
                f"move_end_effector_{name}_axis",
                broadcast_pose_batch(pose, num_envs=num_envs),
            )
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the robot, then press Enter to plan MoveEndEffector..."
    )

    compiled = engine.compile(
        (
            ActionInvocation(
                skill_id="move_end_effector",
                goal=EndEffectorPoseGoal(
                    broadcast_waypoint_pose_batch(poses, num_envs)
                ),
                binding=ActionBinding(manipulators={"primary": "arm"}),
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=MOVE_SAMPLE_INTERVAL,
                ),
            ),
        )
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan MoveEndEffector demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the MoveEndEffector demo...")
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="move_end_effector_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
