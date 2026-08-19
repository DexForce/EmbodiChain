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

"""Demonstrate Press on an articulation link or rigid object."""

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
    PressAffordance,
    PressGoal,
    PressOptions,
)
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.objects import Articulation, RigidObject
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_simulation,
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
RIGID_BUTTON_POSITION = (-0.7, -0.00, 0.70)
RIGID_BUTTON_SIZE = (0.04, 0.02, 0.04)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Press tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate Press on an articulation-link or rigid button.",
        features=("visualize_axes",),
    )
    parser.add_argument("--press_distance", type=float, default=0.03)
    parser.add_argument(
        "--press_position",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional target-local press position overriding the affordance.",
    )
    parser.add_argument(
        "--rigid_object",
        action="store_true",
        help="Use a standalone rigid button instead of the microwave link.",
    )
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


def create_rigid_button(sim) -> RigidObject:
    """Create the standalone static rigid button used by the optional demo."""
    button = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="rigid_button",
            shape=CubeCfg(size=list(RIGID_BUTTON_SIZE)),
            body_type="static",
            init_pos=RIGID_BUTTON_POSITION,
        )
    )
    sim.update(step=10)
    return button


def create_button_semantics(
    target: Articulation | RigidObject,
) -> ObjectSemantics:
    """Create press semantics for an articulation-link or rigid button."""
    if isinstance(target, Articulation):
        affordance = PressAffordance(
            articulation=target,
            link_name=BUTTON_LINK_NAME,
            # button_cap's local -z direction matches the prismatic joint's
            # inward press direction in this asset.
            press_axis=torch.tensor([0.0, 0.0, -1.0], device=target.device),
        )
        label = "microwave_start_button"
    else:
        affordance = PressAffordance(
            rigid_object=target,
            press_axis=torch.tensor([-1.0, 0.0, 0.0], device=target.device),
        )
        label = "rigid_button"
    return ObjectSemantics(
        label=label,
        geometry={},
        entity=target,
        affordance=affordance,
    )


def main() -> None:
    """Plan and replay Press for the selected target object type."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(
        sim, init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0]
    )
    target = create_rigid_button(sim) if args.rigid_object else create_microwave(sim)
    hand_open, hand_close = get_hand_open_close_qpos(robot, close_qpos=0.040)
    motion_gen = create_toppra_motion_generator(robot)
    semantics = create_button_semantics(target)
    affordance = semantics.affordance
    assert isinstance(affordance, PressAffordance)

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
        "Inspect the button target, then press Enter to plan Press...",
    )

    compiled = engine.compile(
        (
            ActionInvocation(
                skill_id="press",
                goal=PressGoal(semantics),
                binding=ActionBinding(
                    manipulators={"primary": "arm"},
                    end_effectors={"primary": "hand"},
                ),
                motion_policy=MotionPolicy(sample_count=PRESS_SAMPLE_INTERVAL),
                skill_options=PressOptions(
                    hand_interp_steps=HAND_INTERP_STEPS,
                    approach_distance=0.12,
                    press_distance=args.press_distance,
                    press_position=(
                        None
                        if args.press_position is None
                        else tuple(args.press_position)
                    ),
                ),
            ),
        )
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan the Press demo trajectory.")
        return

    if wait_for_user:
        input("Press Enter to replay the Press demo...")
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix=(
            "press_rigid_button_auto_play"
            if args.rigid_object
            else "press_microwave_button_auto_play"
        ),
        hold_steps=POST_TRAJECTORY_STEPS,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
