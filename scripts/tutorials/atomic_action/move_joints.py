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

from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AtomicActionEngine,
    ControlPartCommandProfile,
    JointPositionGoal,
    MotionPolicy,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    create_curobo_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

MOVE_JOINTS_SAMPLE_INTERVAL = 80
POST_TRAJECTORY_STEPS = 120


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the MoveJoints tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate MoveJoints with named and explicit qpos targets.",
        features=("visualize_axes",),
    )
    return parser.parse_args()


def main() -> None:
    """Move the robot arm through a named target and two explicit waypoints."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    motion_gen = create_curobo_motion_generator(robot)

    home = robot.get_qpos(name="arm")[0].clone()
    limits = robot.get_qpos_limits(name="arm")[0]

    def offset_from_home(offsets: tuple[float, ...]) -> torch.Tensor:
        target = home.clone()
        count = min(target.numel(), len(offsets))
        target[:count] += torch.tensor(
            offsets[:count], dtype=target.dtype, device=target.device
        )
        return torch.minimum(torch.maximum(target, limits[:, 0]), limits[:, 1])

    ready = offset_from_home((0.35, 0.37, -0.27, -0.08, 0.0, 0.20))
    mid = offset_from_home((0.15, 0.17, -0.12, -0.03, 0.0, 0.10))
    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "arm": ControlPartCommandProfile.joint_positions(ready=ready),
        },
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
    binding = ActionBinding(manipulators={"primary": "arm"})
    policy = MotionPolicy(
        strategy="motion_gen",
        sample_count=MOVE_JOINTS_SAMPLE_INTERVAL,
    )
    compiled = engine.compile(
        (
            ActionInvocation(
                "move_joints", JointPositionGoal("ready"), binding, policy
            ),
            ActionInvocation(
                "move_joints", JointPositionGoal(waypoints), binding, policy
            ),
        ),
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan MoveJoints demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the MoveJoints demo...")
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="move_joints_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
