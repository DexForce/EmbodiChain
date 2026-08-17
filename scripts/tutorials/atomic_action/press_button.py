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

"""Demonstrate PressButton on a microwave start button."""

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
    PressButtonAffordance,
    PressButtonGoal,
    PressButtonOptions,
)
from embodichain.lab.sim.cfg import ArticulationCfg, JointDrivePropertiesCfg
from embodichain.lab.sim.objects import Articulation
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
    draw_axis_marker,
    get_hand_open_close_qpos,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)

MICROWAVE_ASSET = "MicrowaveOven/microwave_oven_with_inertials.urdf"
BUTTON_LINK_NAME = "button_cap"
MICROWAVE_POSITION = (-1.0, -0.30, 0.4)
MICROWAVE_ORIENTATION = (0.0, 0.0, 90)  # degrees
PRESS_SAMPLE_INTERVAL = 140
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the PressButton tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate PressButton on a microwave start button.",
        features=("visualize_axes",),
    )
    parser.add_argument("--press_distance", type=float, default=0.03)
    return parser.parse_args()


def create_microwave(sim) -> Articulation:
    """Create the fixed-base microwave articulation used by the demo."""
    microwave = sim.add_articulation(
        cfg=ArticulationCfg(
            uid="microwave",
            fpath=get_data_path(MICROWAVE_ASSET),
            init_pos=MICROWAVE_POSITION,
            init_qpos=(0, 0, 0, 0),
            init_rot=MICROWAVE_ORIENTATION,
            drive_pros=JointDrivePropertiesCfg(
                stiffness=1e-3, damping=1e2, max_effort=1e-2
            ),
            fix_base=True,
        )
    )
    sim.update(step=10)
    return microwave


def create_button_semantics(microwave: Articulation) -> ObjectSemantics:
    """Create press semantics for the microwave start button."""
    return ObjectSemantics(
        label="microwave_start_button",
        geometry={},
        entity=microwave,
        affordance=PressButtonAffordance(
            articulation=microwave,
            link_name=BUTTON_LINK_NAME,
            # button_cap's local -z direction matches the prismatic joint's
            # inward press direction in this asset.
            press_axis=torch.tensor([0.0, 0.0, -1.0], device=microwave.device),
        ),
    )


def main() -> None:
    """Plan and replay the microwave start-button PressButton trajectory."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(
        sim, init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0]
    )
    microwave = create_microwave(sim)
    hand_open, hand_close = get_hand_open_close_qpos(robot, close_qpos=0.040)
    motion_gen = create_toppra_motion_generator(robot)
    semantics = create_button_semantics(microwave)
    affordance = semantics.affordance
    assert isinstance(affordance, PressButtonAffordance)

    engine = AtomicActionEngine(
        motion_generator=motion_gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
    )
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the microwave, then press Enter to plan PressButton...",
    )

    compiled = engine.compile(
        (
            ActionInvocation(
                skill_id="press_button",
                goal=PressButtonGoal(semantics),
                binding=ActionBinding(
                    manipulators={"primary": "arm"},
                    end_effectors={"primary": "hand"},
                ),
                motion_policy=MotionPolicy(sample_count=PRESS_SAMPLE_INTERVAL),
                skill_options=PressButtonOptions(
                    hand_interp_steps=HAND_INTERP_STEPS,
                    approach_distance=0.12,
                    press_distance=args.press_distance,
                ),
            ),
        )
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan the PressButton demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the PressButton demo...")
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="press_microwave_button_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
