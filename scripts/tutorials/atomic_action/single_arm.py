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

"""Teach single-arm Atomic Skills by compiling PickUp followed by Place.

Scene geometry and robot calibration belong to examples/sim/atomic_action.
This walkthrough exposes the typed application code that composes those skills.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    GraspGoal,
    MotionPolicy,
    PickUpOptions,
    PlaceGoal,
    PlaceOptions,
    create_simulation_atomic_action_engine,
)
from examples.sim.atomic_action import place as scene
from examples.sim.atomic_action.support import (
    add_tutorial_robot,
    broadcast_waypoint_pose_batch,
    create_antipodal_semantics,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_argument_parser,
    create_tutorial_motion_generator,
    create_tutorial_simulation,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    make_clear_dynamics_callback,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)


def main() -> None:
    """Build the scene, bind one arm/hand, compile two goals, and play the result."""
    parser = create_tutorial_argument_parser(
        "Single-arm walkthrough: PickUp -> Place.", features=("grasp_sampling",)
    )
    args = parser.parse_args()

    # -- Scene setup --
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot, tcp_point="grasp")
    obj = scene.create_pick_object(sim)
    sim.prepare()
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    semantics = create_antipodal_semantics(obj, label="cube")

    # -- Resources --
    engine = create_simulation_atomic_action_engine(
        motion_generator=create_tutorial_motion_generator(robot, args.planner),
        scene_entities=(obj,),
        control_profiles={
            "eef": ControlPartCommandProfile.joint_positions(
                open=hand_open, grasp=hand_close
            )
        },
        grasp_pose_generators={
            "eef": create_parallel_jaw_grasp_pose_generator(
                robot=robot, n_sample=args.n_sample, force_refresh=args.force_reannotate
            )
        },
    )
    binding = {"primary": {"motion": "arm", "grasp": "eef"}}
    policy = MotionPolicy(strategy="motion_gen", sample_count=120)

    # -- Goals --
    pick = engine.make_invocation(
        "pick_up",
        GraspGoal(semantics),
        control_parts=binding,
        motion_policy=policy,
        skill_options=PickUpOptions(
            pre_grasp_distance=0.15, lift_height=0.16, hand_interp_steps=12
        ),
    )
    place = engine.make_invocation(
        "place",
        PlaceGoal(
            broadcast_waypoint_pose_batch(
                scene.make_place_eef_poses(sim.device), robot.get_qpos().shape[0]
            )
        ),
        control_parts=binding,
        motion_policy=policy,
        skill_options=PlaceOptions(lift_height=0.10, hand_interp_steps=12),
    )

    # -- Planning --
    wait = prepare_tutorial_scene(sim, args, "Inspect the scene, then press Enter...")
    context = engine.initial_context(control_dt=sim.sim_config.physics_dt)
    compiled = engine.compile((pick, place), context)
    if not bool(compiled.plan_success.all()):
        raise RuntimeError("The single-arm walkthrough could not plan both skills.")

    # -- Playback --
    # compile() projects the PickUp attachment for planning Place. Actual
    # physical grasp/release happens only when the host plays these commands.
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="single_arm_walkthrough",
        hold_steps=120,
        on_trajectory_step=make_clear_dynamics_callback(
            obj, compiled.segment(0, "lift").start
        ),
    )
    if wait:
        input("Press Enter to finish the walkthrough...")


if __name__ == "__main__":
    run_tutorial(main)
