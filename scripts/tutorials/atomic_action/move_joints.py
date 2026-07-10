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

"""Demonstrate MoveJoints with named and explicit joint-space targets."""

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
    JointPositionTarget,
    MoveJoints,
    MoveJointsCfg,
    NamedJointPositionTarget,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_toppra_motion_generator,
    create_tutorial_simulation,
    draw_axis_marker,
    prepare_tutorial_scene,
    replay_trajectory,
)

MOVE_JOINTS_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the MoveJoints tutorial."""
    parser = argparse.ArgumentParser(
        description="Demonstrate MoveJoints with named and explicit qpos targets."
    )
    add_env_launcher_args_to_parser(parser)
    parser.add_argument("--auto_play", action="store_true")
    parser.add_argument("--no_vis_eef_axis", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Move the robot arm through a named target and two explicit waypoints."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim)
    motion_gen = create_toppra_motion_generator(robot)

    ready, mid, home = (
        torch.tensor(qpos, dtype=torch.float32, device=sim.device)
        for qpos in (
            [0.35, -1.20, 1.30, -1.65, -1.57, 0.20],
            [0.15, -1.40, 1.45, -1.60, -1.57, 0.10],
            [0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        )
    )
    engine = AtomicActionEngine(motion_generator=motion_gen)
    engine.register(
        MoveJoints(
            motion_gen,
            cfg=MoveJointsCfg(
                sample_interval=MOVE_JOINTS_SAMPLE_INTERVAL,
                named_joint_positions={"ready": ready},
            ),
        )
    )

    if not args.no_vis_eef_axis:
        draw_axis_marker(
            sim,
            "move_joints_start_eef_axis",
            robot.compute_fk(robot.get_qpos(name="arm"), name="arm", to_matrix=True),
        )
    wait_for_user = prepare_tutorial_scene(
        sim, args, "Inspect the robot, then press Enter to plan MoveJoints..."
    )

    waypoints = (
        torch.stack([mid, home]).unsqueeze(0).repeat(robot.get_qpos().shape[0], 1, 1)
    )
    success, trajectory, _ = engine.run(
        [
            ("move_joints", NamedJointPositionTarget("ready")),
            ("move_joints", JointPositionTarget(waypoints)),
        ]
    )
    if not success.all():
        logger.log_warning("Failed to plan MoveJoints demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the MoveJoints demo...")
    replay_trajectory(
        sim,
        robot,
        trajectory,
        args,
        video_prefix="move_joints_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    main()
