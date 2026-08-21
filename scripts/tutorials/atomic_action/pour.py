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

"""Pick up a cube from the side, then rotate it with the Pour action."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    MotionPolicy,
    PickUpOptions,
    PourGoal,
    PourOptions,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.axis_align import (
    create_align_object,
    create_axis_align_semantics,
)
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_toppra_motion_generator,
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

POUR_INTERNAL_AXIS = (1.0, 0.0, 0.0)
APPROACH_DIRECTION = (-0.707, 0, -0.707)
PICK_SAMPLE_INTERVAL = 120
POUR_SAMPLE_INTERVAL = 80
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
OBJ_POSITION = (-0.5, 0.0, 0.0)
RECORD_LOOK_AT = (
    (-1.5, 0.2, 1.2),
    (-0.4, 0.0, 0.4),
    (0.0, 0.0, 1.0),
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Pour tutorial."""
    parser = create_tutorial_argument_parser(
        "Pick up a cube horizontally, then pour it about a local axis.",
        features=("grasp_sampling", "visualize_axes"),
    )
    parser.add_argument(
        "--rotate_angle",
        type=float,
        default=math.pi / 4.0,
        help="Signed pouring rotation in radians.",
    )
    return parser.parse_args()


def main() -> None:
    """Plan and replay PickUp followed by Pour."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim, tcp_z=0.15)
    obj = create_align_object(
        sim,
        obj_position=OBJ_POSITION,
    )
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    motion_gen = create_toppra_motion_generator(robot)

    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
    )
    semantics = create_axis_align_semantics(obj, args, POUR_INTERNAL_AXIS)
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "pour_object_axis", obj.get_local_pose(to_matrix=True))
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the cube, then press Enter to plan PickUp followed by Pour...",
    )

    control_parts = {"primary": {"motion": "arm", "grasp": "hand"}}
    compiled = engine.compile(
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts=control_parts,
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=PICK_SAMPLE_INTERVAL,
                ),
                skill_options=PickUpOptions(
                    approach_direction=torch.tensor(
                        APPROACH_DIRECTION,
                        dtype=torch.float32,
                        device=sim.device,
                    ),
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=HAND_INTERP_STEPS,
                ),
            ),
            engine.make_invocation(
                "pour",
                PourGoal(),
                control_parts=control_parts,
                motion_policy=MotionPolicy(
                    strategy="motion_gen",
                    sample_count=POUR_SAMPLE_INTERVAL,
                ),
                skill_options=PourOptions(rotate_angle=args.rotate_angle),
            ),
        ),
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan PickUp followed by Pour.")
        return

    if wait_for_user:
        input("Press Enter to replay the PickUp + Pour trajectory...")
    clear_after_step = compiled.segment(0, "lift").start
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="pour_cube_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        on_trajectory_step=make_clear_dynamics_callback(obj, clear_after_step),
        look_at=RECORD_LOOK_AT,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
