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

"""Pick up the reference cube and transport it by an object-pose goal."""

from __future__ import annotations

import argparse
import torch

from embodichain.lab.sim.atomic_actions import (
    ControlPartCommandProfile,
    GraspGoal,
    HeldObjectPoseGoal,
    MotionPolicy,
    PickUpOptions,
    create_simulation_atomic_action_engine,
)
from .pickup import create_pick_object
from .support import (
    add_tutorial_robot,
    broadcast_pose_batch,
    _compile_demo,
    create_affordance_sampling_context,
    create_antipodal_semantics,
    create_curobo_motion_generator,
    create_parallel_jaw_grasp_pose_generator,
    create_tutorial_simulation,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    make_clear_dynamics_callback,
    prepare_tutorial_scene,
    replay_trajectory,
)

__all__: list[str] = []


def run(args: argparse.Namespace) -> None:
    """Execute the fixed-scene PickUp and MoveHeldObject sample."""
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot, tcp_point="grasp")
    obj = create_pick_object(sim)
    sim.prepare()
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    motion = create_curobo_motion_generator(
        robot, planner=args.planner, use_cuda_graph=args.physics != "newton"
    )
    engine = create_simulation_atomic_action_engine(
        motion_generator=motion,
        scene_entities=(obj,),
        control_profiles={
            "eef": ControlPartCommandProfile.joint_positions(
                open=hand_open, grasp=hand_close
            )
        },
        grasp_pose_generators={
            "eef": create_parallel_jaw_grasp_pose_generator(
                robot=robot,
                n_sample=args.n_sample,
                force_refresh=args.force_reannotate,
                approach_direction_samples=getattr(args, "grasp_approach_samples", 4),
            )
        },
    )
    semantics = create_antipodal_semantics(obj, label="cube")
    target = torch.eye(4, device=sim.device)
    target[:3, 3] = torch.tensor([-0.42, 0.10, 0.25], device=sim.device)
    mapping = {"primary": {"motion": "arm", "grasp": "eef"}}
    policy = MotionPolicy(strategy="motion_gen", sample_count=120)
    prepare_tutorial_scene(sim, args, "Inspect MoveHeldObject...")
    compiled = _compile_demo(
        engine,
        args,
        (
            engine.make_invocation(
                "pick_up",
                GraspGoal(semantics),
                control_parts=mapping,
                motion_policy=policy,
                skill_options=PickUpOptions(
                    pre_grasp_distance=0.15,
                    lift_height=0.16,
                    hand_interp_steps=12,
                    approach_direction=torch.tensor(
                        [0.0, 0.0, -1.0], device=sim.device
                    ),
                ),
            ),
            engine.make_invocation(
                "move_held_object",
                HeldObjectPoseGoal(broadcast_pose_batch(target, args.num_envs)),
                control_parts=mapping,
                motion_policy=policy,
            ),
        ),
        engine.initial_context(
            control_dt=sim.sim_config.physics_dt,
            affordance_sampling=create_affordance_sampling_context(args),
        ),
    )
    if not bool(compiled.plan_success.all()):
        raise RuntimeError(
            f"MoveHeldObject planning failed: {compiled.plan_success.tolist()}"
        )
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="move_held_object",
        hold_steps=120,
        compiled=compiled,
        on_trajectory_step=make_clear_dynamics_callback(
            obj, compiled.segment(0, "lift").start
        ),
    )
