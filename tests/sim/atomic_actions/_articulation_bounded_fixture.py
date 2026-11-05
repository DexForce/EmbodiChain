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

"""Analytic redundant arm and scripted query facts for source recovery differentials."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import importlib.util
import hashlib
import json
import random
import xml.etree.ElementTree as ET
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
    PlanningContext,
    RobotObservation,
    TaskState,
    SceneEntityPose,
    SlideAffordance,
    SlideGoal,
    SlideJointTarget,
    SlideOptions,
    MotionPolicy,
    RecoveryPolicy,
    SlidePlanningRecovery,
    EndEffectorPoseGoal,
    MoveEndEffectorOptions,
    ArticulationWithdrawalPlanningRecovery,
)
from embodichain.lab.sim.atomic_actions.articulation_continuation import (
    ArticulationContinuationContext,
    ArticulationInstalledPlan,
    articulation_withdrawal_targets,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport
from ._ordinary_pick_fixture import OrdinaryCandidates
from ._articulation_plan_fixture import command_digest


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def bounded_fixture(directory: Path, mode: str, *, action_types=None):
    spec = importlib.util.spec_from_file_location(
        "source_ik_fixture", Path(__file__).parents[1] / "motion/_ik_branch_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    robot, solver, root, ik_calls, pose_reads = module.branch_fixture(
        directory, failed=mode == "pull_exhausted"
    )
    path = Path(robot.cfg.fpath)
    doc = ET.parse(path)
    ET.SubElement(doc.getroot(), "link", name="hand_link")
    hand_joint = ET.SubElement(
        doc.getroot(), "joint", name="hand_joint", type="prismatic"
    )
    ET.SubElement(hand_joint, "parent", link="tip")
    ET.SubElement(hand_joint, "child", link="hand_link")
    ET.SubElement(hand_joint, "axis", xyz="0 1 0")
    ET.SubElement(
        hand_joint, "limit", lower="0", upper="0.04", effort="10", velocity="10"
    )
    doc.write(path)
    solver.tcp_xpos[:3, :3] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    original_inverse = solver._compute_inverse_kinematics

    def inverse(self, pose, seeds):
        ok, q = original_inverse(pose, seeds)
        ok &= (pose[:, 1:3, 3].abs() <= 1e-6).all(-1)
        ok &= (pose[:, :3, :3] - torch.eye(3)).abs().amax(dim=(-1, -2)) <= 1e-6
        return ok, q

    solver._compute_inverse_kinematics = MethodType(inverse, solver)

    def get_ik(self, pose, qpos_seed):
        ok, q = self._compute_inverse_kinematics(
            pose @ torch.linalg.inv(self.tcp_xpos.to(pose)), qpos_seed
        )
        valid, q = self._qpos_map_to_limits(q)
        return ok & valid, q

    solver.get_ik = MethodType(get_ik, solver)
    robot.dof = 3
    robot.joint_names = ["q0", "q1", "hand_joint"]
    robot.control_parts = {"arm": ["q0", "q1"], "hand": ["hand_joint"]}
    ids = {None: [0, 1, 2], "arm": [0, 1], "hand": [2]}
    initial = torch.tensor([[0.0, 0.0, 0.04]])
    robot.get_qpos = lambda name=None, **kwargs: initial[:, ids[name]].clone()
    robot.get_qvel = lambda name=None, **kwargs: torch.zeros_like(initial[:, ids[name]])
    robot.get_joint_ids = lambda name=None: ids[name]
    robot.get_qvel_limits = lambda name=None: torch.full((1, len(ids[name])), 10.0)
    robot.get_qpos_limits = lambda name=None, **kwargs: torch.tensor(
        [[[-2.0, 2.0], [-2.0, 2.0], [0.0, 0.04]]]
    )[:, ids[name]].clone()
    robot.get_solver = lambda name: solver if name == "arm" else None
    robot.get_local_pose = lambda **kwargs: root.clone()

    def fk(qpos, **kwargs):
        pose = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        pose[..., 0, 3] = qpos[..., :2].sum(-1)
        return root @ pose @ solver.tcp_xpos

    robot.compute_fk = robot.compute_batch_fk = fk

    def native_link(name, **kwargs):
        pose_reads.append(name)
        return root.clone()

    robot.get_link_pose = native_link
    robot.compute_ik = lambda *, pose, joint_seed, name, **kwargs: solver.get_ik(
        torch.linalg.inv(root) @ pose, joint_seed
    )
    robot.mimic_ids, robot.mimic_parents, robot.mimic_multipliers = [], [], []
    generator = object.__new__(MotionGenerator)
    generator.robot, generator.device = robot, robot.device
    generator.planner = NS(
        cfg=NS(planner_type="analytic", sim_instance_id="fixture"),
        collision_world_info=None,
    )
    calls, queries, grasps = [], [], []
    core_generate = generator._generate_once

    def generate(targets, *, options):
        calls.append(
            {
                "count": options.sample_count,
                "preserve": options.preserve_cartesian_samples,
                "start": options.start_qpos.tolist(),
                "targets": _digest([value.xpos.tolist() for value in targets]),
            }
        )
        result = core_generate(targets, options=options)
        if mode in (
            "pull_recovered",
            "pull_second",
            "push_recovered",
            "push_prefix",
            "push_prefix_exhausted",
            "push_pregrasp",
        ):
            result.success = torch.zeros_like(result.success)
        return result

    generator._generate_once = generate

    def report(kind, positions):
        draws = [random.random(), float(np.random.rand()), float(torch.rand(()))]
        queries.append(
            {
                "kind": kind,
                "positions": _digest(positions.tolist()),
                "frames": len(positions),
                "draws": draws,
            }
        )
        count = sum(value["kind"] == "free" for value in queries)
        rejected = kind == "free" and (
            mode == "push_prefix_exhausted"
            or mode == "push_pregrasp"
            and float(positions[:, :2].sum(-1).min()) < -0.075
            or mode in ("pull_second", "push_prefix")
            and count == 1
        )
        if rejected:
            return MotionCollisionReport(
                False, frame=2, kind="scene", pairs=(("arm", "fixture_obstacle"),)
            )
        return MotionCollisionReport(
            True,
            frames=len(positions),
            scope=(
                "discrete_robot_meshes"
                if kind == "robot"
                else "discrete_free_motion_meshes"
            ),
        )

    class Free:
        def __init__(self):
            self.robot = robot

        def inspect(self, positions, *, control_part):
            return report("free", positions)

    class RobotQuery:
        def __init__(self, owner, *, control_part):
            assert owner is robot

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def inspect(self, positions):
            return report("robot", positions)

        def close(self):
            pass

    prefixes = []

    class Prefix:
        def __init__(self):
            self.robot = robot

        def plan(self, *, control_part, start_qpos, target_qpos, full_qpos, control_dt):
            prefixes.append(
                {
                    "part": control_part,
                    "start": start_qpos.tolist(),
                    "target": target_qpos.tolist(),
                    "full": full_qpos.tolist(),
                    "dt": control_dt,
                }
            )
            if mode in ("push_prefix_exhausted", "push_pregrasp"):
                return PlanResult(success=torch.tensor([False]))
            middle = (start_qpos + target_qpos) / 2
            middle[:, 1] += 0.05
            values = torch.stack((start_qpos, middle, target_qpos), 1)
            return PlanResult(
                success=torch.tensor([True]),
                positions=values,
                dt=torch.tensor([[0.0, control_dt, control_dt]]),
            )

    class Grasp(OrdinaryCandidates):
        def get_valid_grasp_poses(self, *, obj_poses, **kwargs):
            grasps.append(
                [random.random(), float(np.random.rand()), float(torch.rand(()))]
            )
            pose = obj_poses.clone()
            pose[:, :3, :3] = solver.tcp_xpos[:3, :3]
            return [(value[None], torch.zeros(1)) for value in pose]

    withdraw = mode.startswith("withdraw")
    push = mode.startswith("push")
    pose = root.clone()
    pose[:, 0, 3] += 0.0 if push else 0.2
    pose[:, 2, 3] += 0.2
    ctx = PlanningContext(
        RobotObservation(0.0, initial, torch.zeros_like(initial)),
        TaskState(1, "cpu"),
        SceneSnapshot(
            0.0,
            0,
            {"handle": EntityState(pose), "drawer": EntityState(root)},
            articulation_joints={
                ("drawer", "rail"): ObservedArticulationJointState(
                    torch.tensor([[0.2 if push else 0.0]])
                )
            },
        ),
        torch.tensor([0]),
        control_dt=0.04,
    )
    identity = "withdraw-source" if withdraw else "slide-source"

    class Provider:
        def __init__(self):
            self.robot = robot

        def prepare(self, request, context):
            facts = (
                (ArticulationInstalledPlan("slide", "previous", 0, True, True, True),)
                if mode == "withdraw_short"
                else ()
            )
            return ArticulationContinuationContext(
                "fixture",
                1 if facts else 0,
                "withdraw" if withdraw else "slide",
                identity,
                40 if mode == "paired_short" else None,
                facts,
            )

    prefix_service = Prefix()
    free_service = Free()
    grasp_generator = Grasp()
    engine = AtomicActionEngine(
        generator,
        load_builtins=action_types is None,
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=torch.tensor([0.04]), grasp=torch.tensor([0.0])
            )
        },
        grasp_pose_generators={"hand": grasp_generator},
        free_motion_query=free_service,
        joint_prefix_planner=prefix_service,
        articulation_continuation_provider=Provider(),
    )
    if action_types is not None:
        for action_type in action_types:
            engine.register(action_type())
    motion = MotionPolicy(
        sample_count=40 if withdraw else 120,
        interpolation_mode="fixed_grid",
        ik_seed_mode="sequential",
        acceleration_limit=None,
        dynamic_collision_mode="off",
        segment_sample_count_retries=(260, 320),
    )
    if withdraw:
        goal = EndEffectorPoseGoal(
            articulation_withdrawal_targets(
                fk(initial[:, :2]), torch.tensor([[-1.0, 0.0, 0.0]])
            )
        )
        invocation = engine.make_invocation(
            "move_end_effector",
            goal,
            control_parts={"primary": {"motion": "arm"}},
            motion_policy=motion,
            skill_options=MoveEndEffectorOptions(),
            recovery_policy=RecoveryPolicy(
                max_replans=0,
                max_action_retries=0,
                planning_recovery=ArticulationWithdrawalPlanningRecovery(),
            ),
            invocation_id=identity,
        )
    else:
        affordance = SlideAffordance(
            mesh_vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]
            ),
            mesh_triangles=torch.tensor([[0, 1, 2]]),
            translation_axis=torch.tensor([1.0, 0.0, 0.0]),
            joint_name="rail",
            joint_limits=(0.0, 0.2),
        )
        goal = SlideGoal(
            ObjectSemantics(affordance, {}, entity_id="handle"),
            SceneEntityPose("handle"),
            joint_target=SlideJointTarget(
                "drawer", "rail", 0.0 if push else 0.2, axis_sign=-1
            ),
        )
        invocation = engine.make_invocation(
            "slide",
            goal,
            control_parts={"primary": {"motion": "arm", "grasp": "hand"}},
            motion_policy=motion,
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
            invocation_id=identity,
        )
    return NS(
        engine=engine,
        generator=generator,
        context=ctx,
        invocation=invocation,
        calls=calls,
        queries=queries,
        prefixes=prefixes,
        prefix_service=prefix_service,
        free_service=free_service,
        grasp_generator=grasp_generator,
        grasps=grasps,
        ik=ik_calls,
        robot_query=RobotQuery,
        report=report,
        withdraw=withdraw,
        withdrawal_samples=40 if mode in ("paired_short", "withdraw_short") else None,
    )


def bounded_record(fixture, plan):
    path = plan.joint_trajectory
    ik = [
        {**value, "rng": hashlib.sha256(value["rng"].numpy().tobytes()).hexdigest()}
        for value in fixture.ik
    ]
    return {
        "commands": command_digest(plan),
        "success": plan.plan_success.tolist(),
        "positions": _digest(path.positions.tolist()),
        "dt": _digest(path.dt.tolist()),
        "shape": list(path.positions.shape),
        "final": path.positions[:, -1].tolist(),
        "segments": [[value.name, value.start, value.stop] for value in plan.segments],
        "metadata": {
            key: value
            for key, value in plan.diagnostics.metadata.items()
            if key.startswith("gen_sim_")
        },
        "calls": fixture.calls,
        "queries": fixture.queries,
        "prefixes": fixture.prefixes,
        "grasps": fixture.grasps,
        "ik_count": len(ik),
        "ik_trace": _digest(ik),
        "next_random": [
            random.random(),
            float(np.random.rand()),
            float(torch.rand(())),
        ],
    }


def park_fixture(directory: Path, mode: str, *, action_type=None):
    from ._handover_native_fixture import native_transfer_fixture
    from embodichain.lab.sim.atomic_actions import (
        JointPositionGoal,
        MoveJointsOptions,
        ArticulationParkPlanningRecovery,
    )

    generator, old, _, _, _, ik, _ = native_transfer_fixture(directory, "clear")
    robot = generator.robot
    initial = robot.get_qpos()
    neutral = mode in ("park_neutral", "park_neutral_exhausted")
    if neutral:
        initial[:, 1] = 0.2
        robot.get_qpos.side_effect = lambda name=None, **kwargs: initial[
            :, robot.get_joint_ids(name=name)
        ].clone()
    ready = initial[:, :6].clone()
    ready[:, 0] = 0.3
    context = replace(
        old,
        robot=RobotObservation(
            0.0, initial, torch.zeros_like(initial), root_pose=torch.eye(4)[None]
        ),
        observed_robot=None,
        task=TaskState(1, "cpu"),
    )
    original_ik = robot.compute_ik.side_effect

    def solve(**kwargs):
        valid, values = original_ik(**kwargs)
        if (
            mode == "park_exhausted"
            or mode == "park_second_via"
            and len(ik) == 1
            or mode == "park_over_ready"
            and len(ik) <= 2
        ):
            valid = torch.zeros_like(valid)
        return valid, values

    robot.compute_ik.side_effect = solve
    calls, queries = [], []
    original_generate = generator._generate_once

    def generate(targets, *, options):
        calls.append(
            {
                "count": options.sample_count,
                "start": options.start_qpos.tolist(),
                "targets": [value.qpos.tolist() for value in targets],
            }
        )
        return original_generate(targets, options=options)

    generator._generate_once = generate

    def report(positions):
        draws = [random.random(), float(np.random.rand()), float(torch.rand(()))]
        queries.append(
            {
                "positions": _digest(positions.tolist()),
                "frames": len(positions),
                "draws": draws,
            }
        )
        crossed = positions[:, 0].abs() < 0.04
        if neutral:
            crossed &= (
                torch.ones_like(crossed)
                if mode == "park_neutral_exhausted"
                else (positions[:, 1] - 0.2).abs() < 0.05
            )
        else:
            crossed &= positions[:, 2] < 0.44
        if crossed.any():
            return MotionCollisionReport(
                False,
                frame=int(crossed.nonzero()[0]),
                kind="self" if neutral else "scene",
                pairs=(("arm", "fixture_obstacle"),),
            )
        return MotionCollisionReport(True, frames=len(positions))

    class Query:
        def __init__(self):
            self.robot = robot

        def inspect(self, positions, *, control_part):
            return report(positions)

    class Provider:
        def __init__(self):
            self.robot = robot

        def prepare(self, request, current):
            predecessor = (
                ()
                if mode == "park_unowned"
                else (
                    ArticulationInstalledPlan(
                        "withdraw", "withdraw0", 0, True, False, True
                    ),
                )
            )
            return ArticulationContinuationContext(
                "park-fixture", 1, "park", "park-source", None, predecessor
            )

    engine = AtomicActionEngine(
        generator,
        load_builtins=action_type is None,
        control_profiles={
            "source_arm": ControlPartCommandProfile.joint_positions(park=ready)
        },
        free_motion_query=Query(),
        articulation_continuation_provider=Provider(),
    )
    if action_type is not None:
        engine.register(action_type())
    invocation = engine.make_invocation(
        "move_joints",
        JointPositionGoal("park"),
        control_parts={"primary": {"motion": "source_arm"}},
        motion_policy=MotionPolicy(
            sample_count=80,
            interpolation_mode="fixed_grid",
            ik_seed_mode="sequential",
            acceleration_limit=None,
            dynamic_collision_mode="off",
        ),
        skill_options=MoveJointsOptions(),
        recovery_policy=RecoveryPolicy(
            max_replans=0,
            max_action_retries=0,
            planning_recovery=ArticulationParkPlanningRecovery(),
        ),
        invocation_id="park-source",
    )
    return NS(
        engine=engine,
        generator=generator,
        context=context,
        invocation=invocation,
        calls=calls,
        queries=queries,
        ik=ik,
        report=report,
    )


def park_record(fixture, plan):
    path = plan.joint_trajectory
    return {
        "commands": command_digest(plan),
        "success": plan.plan_success.tolist(),
        "positions": _digest(path.positions.tolist()),
        "dt": _digest(path.dt.tolist()),
        "shape": list(path.positions.shape),
        "final": path.positions[:, -1].tolist(),
        "metadata": {
            key: value
            for key, value in plan.diagnostics.metadata.items()
            if key.startswith("gen_sim_")
        },
        "calls": fixture.calls,
        "queries": fixture.queries,
        "ik": fixture.ik,
        "next_random": [
            random.random(),
            float(np.random.rand()),
            float(torch.rand(())),
        ],
    }
