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

"""Teach two-participant resource binding with a complete HandOver action.

Run with --is_horizontal for the PGC stick fixture, or omit it for the Robotiq
can fixture. The runnable scene catalog lives in examples/sim/atomic_skill.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from embodichain.lab.sim.atomic_skills import (
    ControlPartCommandProfile,
    HandOverGoal,
    HandOverOptions,
    MotionPolicy,
    create_simulation_atomic_skill_engine,
)
from examples.sim.atomic_skill import handover as scene
from examples.sim.atomic_skill.support import (
    clone_local_pose_from_first_env,
    create_affordance_sampling_context,
    create_antipodal_semantics,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_motion_generator,
    create_tutorial_simulation,
    get_hand_open_close_qpos,
    prepare_tutorial_scene,
    replay_trajectory,
    run_tutorial,
)


def main() -> None:
    """Bind two arm/hand participants and let one skill plan their transfer."""
    args = scene.parse_arguments()

    # -- Scene setup --
    sim = create_tutorial_simulation(args, arena_space=3.0)
    robot = scene.create_dual_robot(sim, args.robot, is_horizontal=args.is_horizontal)
    scene.create_support_surface(sim)
    obj = scene.create_handover_object(sim, args)
    sim.prepare()
    clone_local_pose_from_first_env(obj)
    for _ in range(50):
        sim.update(step=10)
    obj.clear_dynamics()
    semantics = create_antipodal_semantics(obj, label="handover")

    # -- Resources --
    # Asset, TCP, grasp geometry and closing commands are one fixture choice.
    close = (
        scene.HORIZONTAL_HAND_CLOSE_QPOS
        if args.is_horizontal
        else scene.HAND_CLOSE_QPOS
    )
    model = (
        scene.HORIZONTAL_GRIPPER_MODEL
        if args.is_horizontal
        else scene.HANDOVER_GRIPPER_MODEL
    )
    profiles = {}
    for part in ("left_eef", "right_eef"):
        opened, closed = get_hand_open_close_qpos(
            robot, hand_control_part=part, close_qpos=close
        )
        profiles[part] = ControlPartCommandProfile.joint_positions(
            open=opened, grasp=closed
        )
    generator = create_parallel_jaw_grasp_pose_generator(
        n_sample=args.n_sample,
        force_refresh=args.force_reannotate,
        approach_direction_samples=getattr(args, "grasp_approach_samples", 4),
        gripper_model=model,
    )
    engine = create_simulation_atomic_skill_engine(
        motion_generator=create_tutorial_motion_generator(robot, args.planner),
        scene_entities=(obj,),
        control_profiles=profiles,
        grasp_pose_generators={"left_eef": generator, "right_eef": generator},
    )

    # -- Goal and binding --
    target = torch.eye(4, dtype=torch.float32, device=sim.device)
    target[:3, 3] = torch.tensor(scene.FINAL_OBJECT_XYZ, device=sim.device)
    invocation = engine.make_invocation(
        "hand_over",
        HandOverGoal(semantics, target_pose=target),
        control_parts={
            "source": {"motion": "left_arm", "grasp": "left_eef"},
            "destination": {"motion": "right_arm", "grasp": "right_eef"},
        },
        motion_policy=MotionPolicy(strategy="motion_gen", sample_count=220),
        skill_options=HandOverOptions(
            pre_grasp_distance=0.08, lift_height=0.15, hand_interp_steps=10
        ),
    )

    # -- Planning --
    wait = prepare_tutorial_scene(sim, args, "Inspect both arms, then press Enter...")
    context = engine.initial_context(
        control_dt=sim.sim_config.physics_dt,
        affordance_sampling=create_affordance_sampling_context(args),
    )
    compiled = engine.compile((invocation,), context)
    if not bool(compiled.plan_success.all()):
        raise RuntimeError("The dual-arm walkthrough could not plan the transfer.")

    # -- Playback --
    # source/destination are action-local candidate participants. HandOver
    # chooses the nearer pickup arm and coordinates both hands inside its plan.
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="dual_arm_walkthrough",
        hold_steps=120,
        look_at=scene.HANDOVER_RECORD_LOOK_AT,
    )
    if wait:
        input("Press Enter to finish the walkthrough...")


if __name__ == "__main__":
    run_tutorial(main)
