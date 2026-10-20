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

import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    JointPositionGoal,
    MotionPolicy,
)
from embodichain.utils import logger
from embodichain_examples.atomic_action.support import (
    create_affordance_sampling_context,
    add_tutorial_robot,
    create_curobo_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    prepare_tutorial_scene,
    replay_trajectory,
    _compile_demo,
    run_tutorial,
)
from embodichain_examples.atomic_action.support import (
    initialize_benchmark_simulation,
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


def run(args: argparse.Namespace) -> None:
    """Move the robot arm through a named target and two explicit waypoints.

    Args:
        args: Resolved standalone or configured host arguments.
    """
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot)
    sim.prepare()
    motion_gen = create_curobo_motion_generator(
        robot,
        planner=getattr(args, "planner", "trapezoidal"),
    )

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

    control_parts = {"primary": {"motion": "arm"}}
    policy = MotionPolicy(
        strategy="motion_gen",
        sample_count=MOVE_JOINTS_SAMPLE_INTERVAL,
    )
    compiled = _compile_demo(
        engine,
        args,
        # Required targets must be action boundaries before control-grid
        # retiming and spatial expansion, which preserve phase endpoints.
        tuple(
            engine.make_invocation(
                "move_joints",
                JointPositionGoal(target),
                control_parts=control_parts,
                motion_policy=policy,
            )
            for target in ("ready", mid, home)
        ),
        engine.initial_context(
            control_dt=sim.sim_config.physics_dt,
            **(
                {"affordance_sampling": create_affordance_sampling_context(args)}
                if getattr(args, "run", None) is not None
                else {}
            ),
        ),
    )
    if not compiled.plan_success.all():
        if getattr(args, "run", None) is not None:
            raise RuntimeError(
                f"Atomic planning failed: {compiled.plan_success.tolist()}"
            )
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
        compiled=compiled,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


def main() -> None:
    """Run the standalone example with its original arguments."""
    run(parse_arguments())


if __name__ == "__main__":
    run_tutorial(main)


def initialize_simulation(args) -> "SimulationManager":
    """Create the tutorial simulation for interactive or benchmark runs."""
    return initialize_benchmark_simulation(args)


def create_robot(sim: "SimulationManager") -> "Robot":
    """Add the default MoveJoints tutorial robot."""
    robot = add_tutorial_robot(sim, "ur5")
    # DexSim binds body_data (and therefore get_qpos) only after prepare().
    sim.prepare()
    return robot


__all__: list[str] = []
