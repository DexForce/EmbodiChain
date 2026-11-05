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

"""Deterministic source inputs for complete nominal Slide segment comparisons."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import hashlib
import json
import random
import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    JointPositionCommand,
    EntityState,
    ObjectSemantics,
    ObservedArticulationJointState,
    SceneSnapshot,
    SceneEntityPose,
    SlideAffordance,
    SlideGoal,
    SlideJointTarget,
    SlideOptions,
    MotionPolicy,
    RecoveryPolicy,
    SlidePlanningRecovery,
    Slide,
)
from ._ordinary_pick_fixture import ordinary_fixture, OrdinaryCandidates


def articulation_plan_fixture(directory, mode, *, action_type=Slide):
    generator, context, semantics, calls, _ = ordinary_fixture(directory)
    original = generator._generate_once

    def generate(targets, *, options):
        start = options.start_qpos.detach().clone()
        draws = (random.random(), float(np.random.rand()), float(torch.rand(())))
        result = original(targets, options=options)
        calls[-1].update(start=start.tolist(), draws=list(draws))
        if mode == "exhausted" or mode == "dense_approach" and len(calls) == 1:
            result.success = torch.zeros_like(result.success)
        return result

    generator._generate_once = generate
    pose = context.scene.entities["object"].pose
    context = replace(
        context,
        scene=SceneSnapshot(
            0.0,
            0,
            {"handle": EntityState(pose), "drawer": EntityState(torch.eye(4)[None])},
            articulation_joints={
                ("drawer", "rail"): ObservedArticulationJointState(torch.zeros(1, 1))
            },
        ),
    )
    affordance = SlideAffordance(
        mesh_vertices=semantics.affordance.mesh_vertices,
        mesh_triangles=semantics.affordance.mesh_triangles,
        translation_axis=torch.tensor([0.0, 1.0, 0.0]),
        joint_name="rail",
        joint_limits=(0.0, 0.2),
    )
    goal = SlideGoal(
        ObjectSemantics(affordance, {}, entity_id="handle"),
        SceneEntityPose("handle"),
        joint_target=SlideJointTarget("drawer", "rail", 0.2, axis_sign=-1),
    )
    engine = AtomicActionEngine(
        generator,
        load_builtins=False,
        control_profiles={
            "hand": ControlPartCommandProfile(
                commands={
                    "open": JointPositionCommand(torch.tensor([0.04, 0.04])),
                    "grasp": JointPositionCommand(torch.zeros(2)),
                }
            )
        },
        grasp_pose_generators={"hand": OrdinaryCandidates()},
    )
    engine.register(action_type())
    invocation = engine.make_invocation(
        "slide",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=120,
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            dynamic_collision_mode="off",
            segment_sample_count_retries=(260, 320),
        ),
        skill_options=SlideOptions(
            hand_interp_steps=5,
            approach_along_grasp_axis=True,
            preshape_fraction=0.85,
            release_retreat_distance=0.04,
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=0,
            max_action_retries=0,
            planning_recovery=SlidePlanningRecovery(),
        ),
        invocation_id="source-slide",
    )
    return engine, generator, context, invocation, calls


def command_digest(plan):
    frames = [
        {
            "env_ids": frame.env_ids.tolist(),
            "active": frame.active_mask.tolist(),
            "hold": frame.hold_duration.tolist(),
            "commands": [
                {
                    "part": command.target.control_part,
                    "joints": list(command.target.joint_ids),
                    "positions": command.payload.positions.tolist(),
                    "velocities": (
                        None
                        if command.payload.velocities is None
                        else command.payload.velocities.tolist()
                    ),
                }
                for command in frame.commands
            ],
        }
        for frame in plan.commands.frames
    ]
    return hashlib.sha256(
        json.dumps(
            frames, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def articulation_plan_record(plan, calls):
    def digest(value):
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    trajectory = plan.joint_trajectory
    return {
        "commands": command_digest(plan),
        "success": plan.plan_success.tolist(),
        "positions": digest(trajectory.positions.tolist()),
        "dt": digest(trajectory.dt.tolist()),
        "shape": list(trajectory.positions.shape),
        "final": trajectory.positions[:, -1].tolist(),
        "segments": [[value.name, value.start, value.stop] for value in plan.segments],
        "calls": [
            {
                "count": value["count"],
                "preserve": value["preserve"],
                "start": value["start"],
                "targets": digest(value["targets"]),
                "draws": value["draws"],
            }
            for value in calls
        ],
        "next_random": [
            random.random(),
            float(np.random.rand()),
            float(torch.rand(())),
        ],
    }
