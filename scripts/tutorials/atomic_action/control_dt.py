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

"""Compare one interpolated action at two explicit control periods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    JointPositionGoal,
    MotionPolicy,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_tutorial_robot,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

SAMPLE_COUNT = 40
FAST_CONTROL_STEPS = 2
SLOW_CONTROL_STEPS = 8
RESET_STEPS = 20
POST_TRAJECTORY_STEPS = 40


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the control-period tutorial."""
    parser = create_tutorial_argument_parser(
        "Compare identical joint interpolation at two explicit control periods."
    )
    return parser.parse_args()


def main() -> None:
    """Replay the same geometric path with fast and slow waypoint timing."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    sim.prepare()
    engine = AtomicActionEngine(motion_generator=create_toppra_motion_generator(robot))

    initial_qpos = robot.get_qpos().clone()
    start_arm_qpos = robot.get_qpos(name="arm")[0]
    arm_limits = robot.get_qpos_limits(name="arm")[0]
    offsets = torch.zeros_like(start_arm_qpos)
    offsets[: min(4, offsets.numel())] = torch.tensor(
        [0.30, 0.25, -0.20, -0.10][: min(4, offsets.numel())],
        dtype=offsets.dtype,
        device=offsets.device,
    )
    target_arm_qpos = torch.minimum(
        torch.maximum(start_arm_qpos + offsets, arm_limits[:, 0]),
        arm_limits[:, 1],
    )

    invocation = engine.make_invocation(
        "move_joints",
        JointPositionGoal(target_arm_qpos),
        control_parts={"primary": {"motion": "arm"}},
        motion_policy=MotionPolicy(
            strategy="ik_interp",
            sample_count=SAMPLE_COUNT,
        ),
    )
    physics_dt = float(sim.sim_config.physics_dt)
    fast_control_dt = FAST_CONTROL_STEPS * physics_dt
    slow_control_dt = SLOW_CONTROL_STEPS * physics_dt
    fast = engine.compile(
        (invocation,),
        engine.initial_context(control_dt=fast_control_dt),
    )
    slow = engine.compile(
        (invocation,),
        engine.initial_context(control_dt=slow_control_dt),
    )
    if not fast.plan_success.all() or not slow.plan_success.all():
        logger.log_warning("Failed to compile one of the control-period plans.")
        return
    if not torch.allclose(fast.trajectory.positions, slow.trajectory.positions):
        raise RuntimeError("control_dt unexpectedly changed the geometric path.")

    expected_ratio = slow_control_dt / fast_control_dt
    actual_ratio = slow.trajectory.duration / fast.trajectory.duration
    if not torch.allclose(
        actual_ratio,
        torch.full_like(actual_ratio, expected_ratio),
    ):
        raise RuntimeError("Trajectory duration does not scale with control_dt.")

    logger.log_info(
        f"Both plans contain the same {fast.trajectory.waypoint_count} waypoints."
    )
    logger.log_info(
        f"Fast: control_dt={fast_control_dt:.3f}s, "
        f"duration={fast.trajectory.duration.max().item():.3f}s."
    )
    logger.log_info(
        f"Slow: control_dt={slow_control_dt:.3f}s, "
        f"duration={slow.trajectory.duration.max().item():.3f}s "
        f"({expected_ratio:.1f}x slower)."
    )

    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "The two plans have identical positions. Press Enter to replay the fast one...",
    )
    replay_trajectory(
        sim,
        robot,
        fast.trajectory,
        args,
        video_prefix="control_dt_fast_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )

    robot.set_qpos(initial_qpos, target=False)
    robot.set_qpos(initial_qpos, target=True)
    zero_qvel = torch.zeros_like(robot.get_qvel())
    robot.set_qvel(zero_qvel, target=False)
    robot.set_qvel(zero_qvel, target=True)
    sim.update(step=RESET_STEPS)

    if wait_for_user:
        input("Robot reset. Press Enter to replay the slow trajectory...")
    replay_trajectory(
        sim,
        robot,
        slow.trajectory,
        args,
        video_prefix="control_dt_slow_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
