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

"""Complete Press inputs and observations for immutable source comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    JointPositionCommand,
    MotionPolicy,
    ObjectSemantics,
    PlanningContext,
    Press,
    PressAffordance,
    PressGoal,
    PressOptions,
    PressPlanningRecovery,
    RecoveryPolicy,
    SceneEntityPose,
    EntityState,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners.utils import PlanResult


def _press_runtime(directory):
    """Provide a Cartesian robot without depending on other migration layers."""
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = 5
    robot.num_instances = 1
    robot.control_parts = {"arm": object(), "hand": object()}
    robot.joint_names = ["x", "y", "z", "h0", "h1"]
    robot.get_joint_ids.side_effect = lambda name=None: {
        None: list(range(5)),
        "arm": [0, 1, 2],
        "hand": [3, 4],
    }[name]
    initial = torch.tensor([[0.0, 0.0, 0.4, 0.04, 0.04]])
    robot.get_qpos.return_value = initial
    robot.get_qvel.return_value = torch.zeros_like(initial)
    robot.get_qvel_limits.side_effect = lambda name=None, **kw: torch.ones(
        1, len(kw.get("joint_ids", robot.get_joint_ids(name=name)))
    )
    robot.get_qpos_limits.side_effect = (
        lambda name=None, **kw: torch.tensor([[[-2.0, 2.0]]])
        .expand(1, len(kw.get("joint_ids", robot.get_joint_ids(name=name))), 2)
        .clone()
    )
    robot.get_solver.return_value = NS(
        get_iteration_params=lambda: {"pos_eps": 1e-5, "rot_eps": 1e-5}
    )
    path = directory / "press_fixture.urdf"
    path.write_text(
        '<robot name="fixture">'
        + "".join(
            f'<joint name="{name}" type="prismatic"><limit velocity="1"/></joint>'
            for name in robot.joint_names
        )
        + "</robot>"
    )
    robot.cfg.fpath = str(path)
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = [], [], []

    def fk(qpos, **kwargs):
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., :3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
        poses[..., :3, 3] = qpos[..., :3]
        return poses

    robot.compute_fk.side_effect = fk
    robot.compute_batch_fk.side_effect = fk
    generator = object.__new__(MotionGenerator)
    generator.robot = robot
    generator.device = robot.device
    generator.planner = NS(cfg=NS(planner_type="oracle"), collision_world_info=None)
    pose = torch.eye(4)[None]
    pose[:, :3, 3] = torch.tensor([0.2, 0.0, 0.1])
    context = PlanningContext(
        RobotObservation(0.0, initial, torch.zeros_like(initial)),
        TaskState(1, "cpu"),
        SceneSnapshot(0.0, 0, {"object": EntityState(pose)}),
        torch.tensor([0]),
        control_dt=0.04,
    )
    return generator, context


def press_fixture(directory, mode, *, batch_size=1):
    generator, context = _press_runtime(directory)
    calls = []
    if batch_size != 1:
        qpos = context.robot.qpos.repeat(batch_size, 1)
        generator.robot.num_instances = batch_size
        generator.robot.get_qpos.return_value = qpos
        generator.robot.get_qvel.return_value = torch.zeros_like(qpos)
        context = replace(
            context,
            robot=RobotObservation(0.0, qpos, torch.zeros_like(qpos)),
            task=TaskState(batch_size, "cpu"),
            scene=SceneSnapshot(
                0.0,
                0,
                {
                    "object": EntityState(
                        context.scene.entities["object"].pose.repeat(batch_size, 1, 1)
                    )
                },
            ),
            env_ids=torch.arange(batch_size),
        )
    solves = []

    def generate(targets, *, options):
        draws = [random.random(), float(np.random.rand()), float(torch.rand(()))]
        seed = options.start_qpos
        path = (
            torch.cat(
                (
                    seed[:, None],
                    torch.stack([target.xpos[:, :3, 3] for target in targets], dim=1),
                ),
                dim=1,
            )
            if options.preserve_cartesian_samples
            else torch.lerp(
                seed[:, None],
                targets[-1].xpos[:, None, :3, 3],
                torch.linspace(0, 1, options.sample_count)[None, :, None],
            )
        )
        dt = path.new_full(path.shape[:2], options.interpolation_dt)
        dt[:, 0] = 0.0
        calls.append(
            {
                "draws": draws,
                "start": seed.tolist(),
                "count": options.sample_count,
                "preserve": options.preserve_cartesian_samples,
                "targets": [target.xpos.tolist() for target in targets],
            }
        )
        result = PlanResult(success=torch.tensor([True]), positions=path, dt=dt)
        if mode == "exhausted" or mode == "roll" and len(calls) <= 4:
            result.success = torch.zeros_like(result.success)
        return result

    def solve(targets, seed, options):
        solves.append(targets.shape[1])
        torch.rand(())
        return (
            torch.zeros(targets.shape[:2], dtype=torch.bool),
            seed[:, None].expand(-1, targets.shape[1], -1).clone(),
        )

    generator._generate_once = generate
    generator._solve_waypoints_for_options = solve
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
    )
    engine.register(Press())
    invocation = engine.make_invocation(
        "press",
        PressGoal(
            ObjectSemantics(
                PressAffordance(
                    press_axis=torch.tensor([0.0, 0.0, -1.0]),
                    press_position=(0.0, 0.0, 0.0),
                ),
                {},
                entity_id="object",
            ),
            SceneEntityPose("object"),
        ),
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=80,
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            dynamic_collision_mode="off",
        ),
        skill_options=PressOptions(
            hand_interp_steps=5,
            approach_distance=0.01,
            press_distance=0.005,
            contact_roll_alternatives=() if mode == "default" else (math.pi / 2,),
        ),
        recovery_policy=RecoveryPolicy(
            max_replans=0,
            max_action_retries=0,
            planning_recovery=None if mode == "default" else PressPlanningRecovery(),
        ),
        invocation_id="source-press",
    )
    return engine, context, invocation, calls, solves


def press_record(plan, calls, solves):
    def digest(value):
        return hashlib.sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    trajectory = plan.joint_trajectory
    return {
        "success": plan.plan_success.tolist(),
        "positions": (
            None if trajectory is None else digest(trajectory.positions.tolist())
        ),
        "dt": None if trajectory is None else digest(trajectory.dt.tolist()),
        "commands": digest(
            [
                {
                    "hold": frame.hold_duration.tolist(),
                    "active": frame.active_mask.tolist(),
                    "values": [
                        (
                            command.target.control_part,
                            command.payload.positions.tolist(),
                        )
                        for command in frame.commands
                    ],
                }
                for frame in plan.commands.frames
            ]
        ),
        "segments": [(value.name, value.start, value.stop) for value in plan.segments],
        "calls": digest(calls),
        "call_count": len(calls),
        "continuation_calls": len(solves),
        "metadata": (
            None if plan.diagnostics is None else dict(plan.diagnostics.metadata)
        ),
        "next_random": [
            random.random(),
            float(np.random.rand()),
            float(torch.rand(())),
        ],
    }
