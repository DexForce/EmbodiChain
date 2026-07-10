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

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    broadcast_pose_batch,
    broadcast_waypoint_pose_batch,
    create_toppra_motion_generator,
    create_tutorial_simulation,
    draw_axis_marker,
    make_top_down_eef_pose,
    prepare_tutorial_scene,
    replay_trajectory,
)

MOVE_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the MoveEndEffector tutorial."""
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveEndEffector with a multi-waypoint pose trajectory."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--auto_play", action="store_true")
    parser.add_argument("--no_vis_eef_axis", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Move the robot end effector through two pose waypoints."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim)
    motion_gen = create_toppra_motion_generator(robot)

    engine = AtomicActionEngine(motion_generator=motion_gen)
    engine.register(
        MoveEndEffector(
            motion_gen,
            cfg=MoveEndEffectorCfg(sample_interval=MOVE_SAMPLE_INTERVAL),
        )
    )

    poses = torch.stack(
        [
            make_top_down_eef_pose(
                torch.tensor([0.30, -0.20, 0.36], device=sim.device)
            ),
            make_top_down_eef_pose(torch.tensor([0.45, 0.10, 0.30], device=sim.device)),
        ]
    )
    n_envs = robot.get_qpos().shape[0]
    if not args.no_vis_eef_axis:
        for name, pose in zip(("target", "side"), poses, strict=True):
            draw_axis_marker(
                sim,
                f"move_end_effector_{name}_axis",
                broadcast_pose_batch(pose, num_envs=n_envs),
            )
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the robot, then press Enter to plan MoveEndEffector..."
    )

    success, trajectory, _ = engine.run(
        [
            (
                "move_end_effector",
                EndEffectorPoseTarget(broadcast_waypoint_pose_batch(poses, n_envs)),
            )
        ]
    )
    if not success.all():
        logger.log_warning("Failed to plan MoveEndEffector demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the MoveEndEffector demo...")
    replay_trajectory(
        sim,
        robot,
        trajectory,
        args,
        video_prefix="move_end_effector_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
