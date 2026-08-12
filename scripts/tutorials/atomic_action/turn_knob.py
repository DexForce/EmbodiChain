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

"""Demonstrate TurnKnob on a microwave articulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.atomic_actions import (
    ActionBinding,
    ActionInvocation,
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    ObjectSemantics,
    TurnAffordance,
    TurnKnobGoal,
    TurnKnobOptions,
)
from embodichain.lab.sim.cfg import ArticulationCfg
from embodichain.lab.sim.objects import Articulation, Robot
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    make_eef_pose_at,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

MICROWAVE_ASSET = "MicrowaveOven/microwave_oven.urdf"
KNOB_LINK_NAME = "cap_1"
MICROWAVE_POSITION = (-1.0, -0.30, 0.4)
MICROWAVE_ORIENTATION = (0.0, 0.0, 90)  # degrees
TURN_SAMPLE_INTERVAL = 140
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the TurnKnob tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate TurnKnob on a microwave power knob.",
        features=("visualize_axes",),
    )
    parser.add_argument("--turn_angle", type=float, default=-0.7853981634)
    return parser.parse_args()


def create_microwave(sim) -> Articulation:
    """Create the fixed-base microwave articulation used by the demo."""
    microwave = sim.add_articulation(
        cfg=ArticulationCfg(
            uid="microwave",
            fpath=get_data_path(MICROWAVE_ASSET),
            init_pos=MICROWAVE_POSITION,
            init_rot=MICROWAVE_ORIENTATION,
            fix_base=True,
        )
    )
    sim.update(step=10)
    return microwave


def create_knob_semantics(microwave: Articulation) -> ObjectSemantics:
    """Create turn semantics for the microwave power knob."""
    return ObjectSemantics(
        label="microwave_power_knob",
        geometry={},
        entity=microwave,
        affordance=TurnAffordance(
            articulation=microwave,
            link_name=KNOB_LINK_NAME,
            turn_axis=torch.tensor([0.0, 0.0, -1.0], device=microwave.device),
        ),
    )


def initialize_robot_near_knob(
    robot: Robot,
    microwave: Articulation,
    hand_open: torch.Tensor,
) -> None:
    """Place the open gripper near the knob; values are intentionally coarse."""
    knob_position = microwave.get_link_pose(KNOB_LINK_NAME, to_matrix=True)[:, :3, 3]
    start_position = knob_position.clone()
    start_position[:, 1] += 0.25
    start_position[:, 2] += 0.05
    success, arm_qpos = robot.compute_ik(
        pose=make_eef_pose_at(robot, start_position),
        joint_seed=robot.get_qpos(name="arm"),
        name="arm",
    )
    if not torch.all(success):
        logger.log_warning(
            "The coarse microwave pre-turn pose is not reachable; keeping the "
            "robot's configured initial arm pose."
        )
        arm_qpos = robot.get_qpos(name="arm")
    hand_qpos = hand_open.unsqueeze(0).expand(robot.get_qpos().shape[0], -1)
    for target in (False, True):
        robot.set_qpos(arm_qpos, name="arm", target=target)
        robot.set_qpos(hand_qpos, name="hand", target=target)
    robot.clear_dynamics()


def main() -> None:
    """Plan and replay the microwave power-knob TurnKnob trajectory."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(
        sim, init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0]
    )
    microwave = create_microwave(sim)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    # initialize_robot_near_knob(robot, microwave, hand_open)
    motion_gen = create_toppra_motion_generator(robot)
    semantics = create_knob_semantics(microwave)

    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
    )
    if not args.no_vis_eef_axis:
        draw_axis_marker(
            sim,
            "microwave_power_knob_axis",
            microwave.get_link_pose(KNOB_LINK_NAME, to_matrix=True),
        )
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the microwave, then press Enter to plan TurnKnob...",
    )

    compiled = engine.compile(
        (
            ActionInvocation(
                skill_id="turn_knob",
                goal=TurnKnobGoal(semantics),
                binding=ActionBinding(
                    manipulators={"primary": "arm"},
                    end_effectors={"primary": "hand"},
                ),
                motion_policy=MotionPolicy(sample_count=TURN_SAMPLE_INTERVAL),
                skill_options=TurnKnobOptions(
                    hand_interp_steps=HAND_INTERP_STEPS,
                    pre_grasp_distance=0.12,
                    turn_angle=args.turn_angle,
                ),
            ),
        )
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan the TurnKnob demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the TurnKnob demo...")
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="turn_microwave_knob_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
