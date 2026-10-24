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

"""Push a cube along a fixed ground-plane target."""

from __future__ import annotations

import argparse
import torch

from embodichain.lab.sim.atomic_actions import (
    Affordance,
    ControlPartCommandProfile,
    MotionPolicy,
    ObjectSemantics,
    PushObjectGoal,
    create_simulation_atomic_action_engine,
)
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from .support import (
    add_tutorial_robot,
    broadcast_pose_batch,
    _compile_demo,
    clone_local_pose_from_first_env,
    create_curobo_motion_generator,
    create_tutorial_rigid_body_physics,
    create_tutorial_simulation,
    get_hand_open_close_qpos,
    prepare_tutorial_scene,
    replay_trajectory,
)

__all__: list[str] = []


def run(args: argparse.Namespace) -> None:
    """Plan and replay the reference cube's constrained pushing path."""
    sim = create_tutorial_simulation(args)
    robot = add_tutorial_robot(sim, args.robot, tcp_point="grasp")
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="cube",
            shape=CubeCfg(size=[0.05, 0.05, 0.05]),
            init_pos=[-0.42, -0.08, 0.05],
            attrs=create_tutorial_rigid_body_physics(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
                newton_contact=sim.is_newton_backend,
            ),
        )
    )
    sim.prepare()
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    hand_open, hand_close = get_hand_open_close_qpos(robot)
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
    )
    semantics = ObjectSemantics(
        affordance=Affordance(), geometry={}, label="cube", entity_id=obj.uid
    )
    target = torch.eye(4, device=sim.device)
    target[:3, 3] = torch.tensor([-0.42, 0.10, 0.005], device=sim.device)
    prepare_tutorial_scene(sim, args, "Inspect PushObject...")
    compiled = _compile_demo(
        engine,
        args,
        (
            engine.make_invocation(
                "push_object",
                PushObjectGoal(semantics, broadcast_pose_batch(target, args.num_envs)),
                control_parts={"primary": {"motion": "arm", "grasp": "eef"}},
                motion_policy=MotionPolicy(strategy="ik_interp", sample_count=120),
            ),
        ),
        engine.initial_context(control_dt=sim.sim_config.physics_dt),
    )
    if not bool(compiled.plan_success.all()):
        raise RuntimeError(
            f"PushObject planning failed: {compiled.plan_success.tolist()}"
        )
    replay_trajectory(
        sim,
        robot,
        compiled.trajectory,
        args,
        video_prefix="push_object",
        hold_steps=120,
        compiled=compiled,
    )
