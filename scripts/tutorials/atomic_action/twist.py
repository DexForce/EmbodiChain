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

"""Demonstrate Twist on an articulation link or rigid object."""

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
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    ObjectSemantics,
    TwistAffordance,
    TwistGoal,
    TwistOptions,
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
KNOB_LINK_NAME = "cap_1"
MICROWAVE_POSITION = (-1.0, -0.30, 0.4)
MICROWAVE_ORIENTATION = (0.0, 0.0, 90)  # degrees
TWIST_SAMPLE_INTERVAL = 140
HAND_INTERP_STEPS = 12
POST_TRAJECTORY_STEPS = 240
RIGID_KNOB_POSITION = (-0.7, -0.00, 0.70)
RIGID_KNOB_SIZE = (0.05, 0.05, 0.05)
KNOB_SCENE_ENTITY_ID = "twist-target"
KNOB_AXIS_ORIGIN = (0.0, 0.0, 0.0)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the Twist tutorial."""
    parser = create_tutorial_argument_parser(
        "Demonstrate Twist on an articulation-link or rigid knob.",
        features=("visualize_axes",),
    )
    parser.add_argument("--twist_angle", type=float, default=-0.7853981634)
    parser.add_argument(
        "--rigid_object",
        action="store_true",
        help="Use a standalone rigid knob instead of the microwave link.",
    )
    return parser.parse_args()


def create_microwave(sim) -> Articulation:
    """Create the fixed-base microwave articulation used by the demo."""
    microwave = sim.add_articulation(
        cfg=ArticulationCfg(
            uid="microwave",
            fpath=get_data_path(MICROWAVE_ASSET),
            asset_physics_mode="overlay",
            init_pos=MICROWAVE_POSITION,
            init_rot=MICROWAVE_ORIENTATION,
            drive_pros=JointDrivePropertiesCfg(
                drive_type="force",
                stiffness=1e-3,
                damping=1e2,
                max_effort=1e-2,
            ),
            fix_base=True,
        )
    )
    sim.update(step=10)
    return microwave


def create_rigid_knob(sim) -> RigidObject:
    """Create the standalone static rigid knob used by the optional demo."""
    knob = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="rigid_knob",
            shape=CubeCfg(size=list(RIGID_KNOB_SIZE)),
            body_type="static",
            init_pos=RIGID_KNOB_POSITION,
        )
    )
    sim.update(step=10)
    return knob


def create_knob_semantics(
    target: Articulation | RigidObject,
) -> tuple[ObjectSemantics, torch.Tensor]:
    """Create twist semantics for an articulation-link or rigid knob."""
    if isinstance(target, Articulation):
        vertices, _ = target.get_link_vert_face(KNOB_LINK_NAME)
        target_pose = target.get_link_pose(KNOB_LINK_NAME, to_matrix=True)
        affordance = TwistAffordance(
            grasp_position=_mesh_center(vertices),
            # The cap_1 revolute axis passes through its link-frame origin.
            axis_origin=KNOB_AXIS_ORIGIN,
            twist_axis=torch.tensor([0.0, 0.0, -1.0], device=target.device),
        )
        label = "microwave_power_knob"
    else:
        vertices = target.get_vertices(env_ids=[0], scale=True)[0]
        target_pose = target.get_local_pose(to_matrix=True)
        affordance = TwistAffordance(
            grasp_position=_mesh_center(vertices),
            axis_origin=KNOB_AXIS_ORIGIN,
            twist_axis=torch.tensor([-1.0, 0.0, 0.0], device=target.device),
        )
        label = "rigid_knob"
    return (
        ObjectSemantics(
            label=label,
            geometry={},
            entity_id=KNOB_SCENE_ENTITY_ID,
            affordance=affordance,
        ),
        target_pose,
    )


def _mesh_center(vertices: torch.Tensor) -> tuple[float, float, float]:
    """Return an explicit local gripper-center point for a knob mesh."""
    center = torch.as_tensor(vertices, dtype=torch.float32).mean(dim=0)
    return tuple(float(value) for value in center)


def main() -> None:
    """Plan and replay Twist for the selected target object type."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(
        sim, init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0]
    )
    target = create_rigid_knob(sim) if args.rigid_object else create_microwave(sim)
    sim.prepare()
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    motion_gen = create_toppra_motion_generator(robot)
    semantics, target_pose = create_knob_semantics(target)

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
        "Inspect the knob target, then press Enter to plan Twist...",
    )

    compiled = engine.compile(
        (
            engine.make_invocation(
                "twist",
                TwistGoal(
                    semantics,
                    target_pose,
                ),
                control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
                motion_policy=MotionPolicy(sample_count=TWIST_SAMPLE_INTERVAL),
                skill_options=TwistOptions(
                    hand_interp_steps=HAND_INTERP_STEPS,
                    pre_grasp_distance=0.12,
                    twist_angle=args.twist_angle,
                ),
            ),
        ),
        context=engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan the Twist demo trajectory.")
        return

    if isinstance(target, RigidObject):
        focus_pose = target.get_local_pose(to_matrix=True)
    elif isinstance(target, Articulation):
        focus_pose = target.get_link_pose(KNOB_LINK_NAME, to_matrix=True)
    else:
        raise ValueError("Unsupported target type for Press demo.")
    focus_position = [focus_pose[0, 0, 3], focus_pose[0, 1, 3], focus_pose[0, 2, 3]]
    camera_position = [
        focus_position[0] + 0.3,
        focus_position[1] + 0.3,
        focus_position[2] + 0.3,
    ]
    look_at = [camera_position, focus_position, [0, 0, 1]]
    if wait_for_user:
        input("Press Enter to replay the Twist demo...")
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix=(
            "twist_rigid_knob_auto_play"
            if args.rigid_object
            else "twist_microwave_knob_auto_play"
        ),
        hold_steps=POST_TRAJECTORY_STEPS,
        look_at=look_at,
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
