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

"""Demonstrate OpenDoor with automatic handle-to-hinge resolution."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    MotionPolicy,
    ObjectSemantics,
    ObservedArticulationJointState,
    OpenDoorAffordance,
    OpenDoorGoal,
    OpenDoorOptions,
    SceneSnapshot,
)
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.objects import Articulation
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    create_parallel_jaw_grasp_pose_generator,
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
HANDLE_LINK_NAME = "door_handle"
MICROWAVE_SCENE_ENTITY_ID = "microwave"
MICROWAVE_POSITION = (-1.0, 0.20, 0.4)
MICROWAVE_ORIENTATION = (0.0, 0.0, 90.0)  # degrees
HANDLE_SCENE_ENTITY_ID = "microwave-door-handle"
TRAJECTORY_SAMPLE_COUNT = 300
HAND_INTERP_STEPS = 30
POST_TRAJECTORY_STEPS = 240


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the door-opening tutorial."""
    parser = create_tutorial_argument_parser(
        "Grasp a microwave handle and open its door with OpenDoor.",
        features=("grasp_sampling", "visualize_axes"),
    )
    parser.add_argument(
        "--open_angle",
        type=float,
        default=math.radians(60.0),
        help="Desired absolute hinge opening in radians (default: 60 degrees).",
    )
    parser.add_argument("--approach_distance", type=float, default=0.10)
    parser.add_argument("--retract_distance", type=float, default=0.10)
    parser.add_argument("--door_waypoint_count", type=int, default=50)
    return parser.parse_args()


def create_microwave(sim: SimulationManager) -> Articulation:
    """Create the fixed-base microwave with an unactuated door hinge."""
    microwave = sim.add_articulation(
        cfg=ArticulationCfg(
            uid="microwave",
            fpath=get_data_path(MICROWAVE_ASSET),
            init_pos=MICROWAVE_POSITION,
            init_rot=MICROWAVE_ORIENTATION,
            drive_pros=JointDrivePropertiesCfg(drive_type="none"),
            attrs=RigidBodyAttributesCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            fix_base=True,
        )
    )
    sim.update(step=10)
    return microwave


def create_door_handle_semantics(microwave: Articulation) -> ObjectSemantics:
    """Resolve the first parent revolute joint from ``door_handle``.

    Only the handle link is configured. ``OpenDoorAffordance`` traverses the
    fixed ``door_to_door_handle_fixed`` joint and resolves ``door_hinge``.
    """
    affordance = OpenDoorAffordance.from_articulation(
        microwave,
        HANDLE_LINK_NAME,
    )
    logger.log_info(
        f"OpenDoor resolved parent revolute joint {affordance.joint_name!r} "
        f"from handle link {HANDLE_LINK_NAME!r}."
    )
    return ObjectSemantics(
        label="microwave_door_handle",
        geometry={},
        entity_id=HANDLE_SCENE_ENTITY_ID,
        affordance=affordance,
    )


def main() -> None:
    """Plan and replay a 60-degree microwave-door opening by default."""
    args = parse_arguments()
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(
        sim,
        init_qpos=[0.0, -1.57, 1.57, -3.14, -1.57, 0.0, 0.0, 0.0],
        tcp_z=0.15,
    )
    microwave = create_microwave(sim)
    hand_open, hand_close = get_hand_open_close_qpos(robot, close_qpos=0.04)
    semantics = create_door_handle_semantics(microwave)
    affordance = semantics.affordance
    assert isinstance(affordance, OpenDoorAffordance)
    assert affordance.joint_limits is not None
    lower_limit, upper_limit = affordance.joint_limits
    closed_position = lower_limit if affordance.opening_direction > 0 else upper_limit
    open_position = upper_limit if affordance.opening_direction > 0 else lower_limit
    open_fraction = (args.open_angle - closed_position) / (
        open_position - closed_position
    )
    handle_pose = microwave.get_link_pose(HANDLE_LINK_NAME, to_matrix=True)
    hinge_joint_index = microwave.joint_names.index(affordance.joint_name)
    hinge_position = microwave.get_qpos(target=False)[
        :, hinge_joint_index : hinge_joint_index + 1
    ]
    if not args.no_vis_eef_axis:
        draw_axis_marker(sim, "door_handle_link_pose", handle_pose)

    engine = AtomicActionEngine(
        motion_generator=create_toppra_motion_generator(robot),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
        grasp_pose_generators={
            "hand": create_parallel_jaw_grasp_pose_generator(
                n_sample=args.n_sample,
                force_refresh=args.force_reannotate,
            )
        },
    )
    wait_for_user = prepare_tutorial_scene(
        sim,
        args,
        "Inspect the closed microwave, then press Enter to plan OpenDoor...",
    )
    compiled = engine.compile(
        (
            engine.make_invocation(
                "open_door",
                OpenDoorGoal(
                    semantics,
                    handle_pose,
                    open_fraction=open_fraction,
                ),
                control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
                motion_policy=MotionPolicy(sample_count=TRAJECTORY_SAMPLE_COUNT),
                skill_options=OpenDoorOptions(
                    hand_interp_steps=HAND_INTERP_STEPS,
                    door_waypoint_count=args.door_waypoint_count,
                    approach_distance=args.approach_distance,
                    retract_distance=args.retract_distance,
                ),
            ),
        ),
        context=engine.initial_context(
            scene=SceneSnapshot(
                timestamp=0.0,
                version=0,
                articulation_joints={
                    (
                        MICROWAVE_SCENE_ENTITY_ID,
                        affordance.joint_name,
                    ): ObservedArticulationJointState(hinge_position)
                },
            ),
            control_dt=sim.sim_config.physics_dt,
        ),
    )
    if not compiled.plan_success.all():
        logger.log_warning("Failed to plan the OpenDoor tutorial trajectory.")
        return

    focus_position = handle_pose[0, :3, 3].tolist()
    camera_position = [
        focus_position[0] + 0.4,
        focus_position[1] + 0.3,
        focus_position[2] + 1.5,
    ]
    if wait_for_user:
        input("Press Enter to replay OpenDoor...")
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="open_door_auto_play",
        hold_steps=POST_TRAJECTORY_STEPS,
        look_at=[camera_position, focus_position, [0.0, 0.0, 1.0]],
    )
    if wait_for_user:
        input("Press Enter to exit the simulation...")


if __name__ == "__main__":
    run_tutorial(main)
