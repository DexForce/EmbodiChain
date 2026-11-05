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

"""Fixed transform cases shared with complete B0 engine source oracles."""

from __future__ import annotations
from types import SimpleNamespace as NS
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
    GraspGoal,
    MotionPolicy,
    PickUpOptions,
    PickPlanningRecovery,
    RecoveryPolicy,
)
from embodichain.lab.sim.motion import mesh_collision
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
from embodichain.toolkits.graspkit.pg_grasp.constraints import (
    GraspGeometryConstraint,
    grasp_geometry_digest,
)
from ._ordinary_pick_fixture import ordinary_fixture, OrdinaryCandidates
from ._constrained_pick_fixture import ConstrainedCandidates
from .test_pick_native_closing import QueryProvider
from dataclasses import replace
import hashlib, json
import torch
from embodichain.compute.trajectory import differentiate_positions

VARIANTS = ("identity", "arm_jump", "scene_bulge", "close_change", "velocity_spike")


def digest(value):
    return (
        None
        if value is None
        else hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
    )


def transform(action, events, state, variant):
    def apply(request, context, plan):
        state["stage"] = "after"
        state["transforms"] += 1
        events.append(
            {
                "kind": "transform",
                "count": request.motion_policy.sample_count,
                "variant": variant,
            }
        )
        if variant == "identity":
            return plan
        trajectory = plan.joint_trajectory
        positions = trajectory.positions.clone()
        approach = next(s for s in plan.segments if s.name == "approach")
        close = next(s for s in plan.segments if s.name == "close")
        if variant == "arm_jump":
            positions[:, :, 0] += 3.0
        elif variant == "scene_bulge":
            count = min(30, approach.stop - 2)
            positions[:, 1 : count + 1, 1] += (
                0.04
                * torch.sin(
                    torch.linspace(0, torch.pi, count, device=positions.device)
                )[None]
            )
        elif variant == "close_change":
            positions[:, close.start + 3, 3] += 0.002
        elif variant == "velocity_spike":
            positions[:, min(10, approach.stop - 2), 0] += 0.1
        velocities = (
            None
            if trajectory.velocities is None
            else differentiate_positions(positions, trajectory.dt)
        )
        accelerations = (
            None
            if trajectory.accelerations is None
            else differentiate_positions(velocities, trajectory.dt)
        )
        trajectory = replace(
            trajectory,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
        )
        commands = action._joint_command_sequence(
            request, trajectory, active_mask=plan.plan_success
        )
        return replace(
            plan,
            joint_trajectory=trajectory,
            commands=commands,
            tracking=action._tracking_sequence(request, commands),
        )

    return apply


def ports(events, state):
    def robot(robot, positions, part):
        result = {
            "valid": True,
            "frames": len(positions),
            "scope": "discrete_robot_meshes",
        }
        events.append(
            {
                "kind": "robot",
                "stage": state["stage"],
                "frames": len(positions),
                "positions": digest(positions),
            }
        )
        return result

    def scene(robot, simulation, positions, part):
        blocked = bool((positions[:, 1] > 0.015).any())
        result = (
            {
                "valid": False,
                "frame": int((positions[:, 1] > 0.015).nonzero()[0]),
                "kind": "scene",
                "pairs": [["hand", "obstacle"]],
            }
            if blocked
            else {
                "valid": True,
                "frames": len(positions),
                "scope": "discrete_free_motion_meshes",
            }
        )
        events.append(
            {
                "kind": "scene",
                "stage": state["stage"],
                "frames": len(positions),
                "positions": digest(positions),
                "valid": not blocked,
            }
        )
        return result

    return robot, scene


def record(plan, calls, ik, events, state):
    trajectory = plan.joint_trajectory
    rejected = getattr(plan, "rejected_candidate", None)
    candidate = trajectory.positions if rejected is None else rejected.positions
    return {
        "success": plan.plan_success.tolist(),
        "failure": (
            None if plan.diagnostics.failure is None else plan.diagnostics.failure.code
        ),
        "positions": digest(trajectory.positions),
        "candidate_positions": digest(candidate),
        "dt": digest(trajectory.dt),
        "shape": list(trajectory.positions.shape),
        "commands_active": sum(bool(f.active_mask.any()) for f in plan.commands.frames),
        "events": events,
        "calls": calls,
        "ik": ik,
        "metadata": dict(plan.diagnostics.metadata),
        "rng": digest(torch.get_rng_state()),
        "transform_count": state["transforms"],
    }


def transform_inputs(directory, mode, monkeypatch):
    gen, context, sem, calls, ik = ordinary_fixture(directory)
    gen.robot.get_solver("arm").supports_continuous_batch_ik = False
    events = []
    state = {"stage": "before", "transforms": 0}
    robot_query, scene_query = ports(events, state)

    def inspect_scene(positions, **kwargs):
        return MotionCollisionReport(
            **{
                k: v
                for k, v in scene_query(gen.robot, None, positions, "arm").items()
                if k != "scope"
            }
        )

    def prefix(**kwargs):
        raise ValueError("controlled_prefix_unavailable")

    provider = None
    kwargs = {}
    goal = GraspGoal(semantics=sem)
    generator = OrdinaryCandidates()
    options = PickUpOptions(
        lift_height=0.16,
        pre_grasp_distance=0.15,
        approach_alignment_max_angle=0.1,
        hand_interp_steps=12,
        adaptive_grasp="ordinary",
        full_opening_clearance=True,
        require_qualified_approach=True,
    )
    recovery = RecoveryPolicy(
        action_timeout=90.0, planning_recovery=PickPlanningRecovery()
    )
    if mode == "constrained":
        constraint = GraspGeometryConstraint(
            grasp_geometry_digest(
                sem.affordance.mesh_vertices, sem.affordance.mesh_triangles
            ),
            (0.0, 0.0, 1.0),
            0.0,
            True,
        )
        goal = replace(goal, grasp_constraint=constraint)
        generator = ConstrainedCandidates()
        options = replace(
            options,
            adaptive_grasp="none",
            full_opening_clearance=False,
            require_qualified_approach=False,
            constrained_grasp=True,
        )
        recovery = replace(
            recovery, planning_recovery=PickPlanningRecovery(constrained_roll=True)
        )
    if mode in {"native", "fixed"}:
        provider = QueryProvider()
        kwargs = {
            "pick_closing_query_provider": provider,
            "hand_geometry_cfgs": {
                "hand": RobotHandGeometryCfg(
                    motion_part="arm",
                    hand_part="hand",
                    hand_root_link="hand_root",
                    pad_link_names=("pad_a", "pad_b"),
                    open_qpos=(0.04, 0.04),
                    closed_qpos=(0.0, 0.0),
                    sample_count=65,
                    body_max_edge=0.1,
                    pad_max_edge=0.01,
                    max_subdivisions=1,
                )
            },
        }
        options = replace(options, native_closing_revision=1 if mode == "fixed" else 4)
    if mode == "fixed":
        fixed = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
        fixed[2, 3] = 0.01
        options = replace(
            options,
            fixed_object_to_eef=fixed,
            qualify_fixed_grasp=True,
            adaptive_grasp="none",
            full_opening_clearance=False,
            require_qualified_approach=False,
        )
        recovery = replace(recovery, planning_recovery=None)
    engine = AtomicActionEngine(
        gen,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.full((2,), 0.04), grasp=torch.zeros(2)
            )
        },
        grasp_pose_generators={"hand": generator},
        free_motion_query=NS(robot=gen.robot, inspect=inspect_scene),
        joint_prefix_planner=NS(robot=gen.robot, plan=prefix),
        **kwargs,
    )
    action = engine.actions["pick_up"]
    original = action._plan

    def planned(request, current):
        state["stage"] = "before"
        events.append({"kind": "plan", "count": request.motion_policy.sample_count})
        return original(request, current)

    monkeypatch.setattr(action, "_plan", planned)
    invocation = engine.make_invocation(
        "pick_up",
        goal,
        control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
        motion_policy=MotionPolicy(
            sample_count=140,
            dynamic_collision_mode="off",
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            sample_count_retries=(260, 320),
        ),
        skill_options=options,
        recovery_policy=recovery,
        invocation_id="pick",
    )
    monkeypatch.setattr(
        mesh_collision,
        "_make_checker",
        lambda robot, simulation, part, **kw: lambda positions: robot_query(
            robot, positions, part
        ),
    )
    return engine, invocation, context, action, calls, ik, events, state, provider
