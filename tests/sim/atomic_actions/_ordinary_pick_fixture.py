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

"""Cartesian robot and collision oracles for complete ordinary Pick comparisons."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import torch
import trimesh

from embodichain.lab.sim.atomic_actions import (
    AxisAlignAffordance,
    EntityState,
    ObjectSemantics,
    PlanningContext,
    RobotObservation,
    SceneSnapshot,
    TaskState,
)
from embodichain.lab.sim.motion.motion_generator import MotionGenerator
from embodichain.lab.sim.motion.planners.utils import MoveType, PlanResult
from embodichain.toolkits.graspkit import (
    ParallelJawGraspPoseGenerator,
    ParallelJawGripperModelCfg,
)


class OrdinaryCandidates(ParallelJawGraspPoseGenerator):
    def __init__(self):
        super().__init__(ParallelJawGripperModelCfg(model_id="ordinary_fixture"))

    def get_valid_grasp_poses(self, *, obj_poses, **kwargs):
        result = []
        for pose in obj_poses:
            value = pose.clone()
            value[:3, :3] = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
            result.append((value[None], torch.zeros(1)))
        return result

    def get_best_grasp_poses(self, **kwargs):
        raise AssertionError("ordinary Pick uses the candidate protocol")

    def get_dual_arm_valid_grasp_poses(self, **kwargs):
        raise AssertionError("ordinary Pick has one hand")


def ordinary_fixture(directory: Path):
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
    path = directory / "ordinary_fixture.urdf"
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
    rotation = torch.diag(torch.tensor([1.0, -1.0, -1.0]))

    def fk(qpos, **kwargs):
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., :3, :3] = rotation
        poses[..., :3, 3] = qpos[..., :3]
        return poses

    robot.compute_fk.side_effect = fk
    robot.compute_batch_fk.side_effect = fk
    robot.compute_ik.side_effect = lambda pose, **kw: (
        torch.ones(pose.shape[0], dtype=torch.bool),
        pose[:, :3, 3].clone(),
    )
    ik_calls = []

    def ik(pose, joint_seed, **kw):
        torch.rand(())
        ik_calls.append({"poses": pose.tolist(), "seed": joint_seed.tolist()})
        return torch.ones(pose.shape[:2], dtype=torch.bool), pose[..., :3, 3].clone()

    robot.compute_batch_ik.side_effect = ik
    calls = []

    def generate(targets, *, options):
        options.__post_init__()
        start = options.start_qpos
        count = options.sample_count
        if targets[0].move_type is MoveType.JOINT_MOVE:
            path = torch.lerp(
                start[:, None],
                targets[-1].qpos[:, None],
                torch.linspace(0, 1, count)[None, :, None],
            )
        elif options.preserve_cartesian_samples:
            path = torch.cat(
                (
                    start[:, None],
                    torch.stack([value.xpos[:, :3, 3] for value in targets], 1),
                ),
                1,
            )
        else:
            points = torch.cat(
                (
                    start[:, None],
                    torch.stack([value.xpos[:, :3, 3] for value in targets], 1),
                ),
                1,
            )
            visits = (count // 3, 2 * count // 3, count - 1)
            path = torch.empty(start.shape[0], count, start.shape[1])
            previous = 0
            for target_index, stop in enumerate(visits):
                path[:, previous : stop + 1] = torch.lerp(
                    points[:, target_index : target_index + 1],
                    points[:, target_index + 1 : target_index + 2],
                    torch.linspace(0, 1, stop - previous + 1)[None, :, None],
                )
                previous = stop
            fraction = torch.linspace(0, 1, visits[1] - visits[0] + 1)
            path[:, visits[0] : visits[1] + 1, 0] += (
                torch.sin(fraction * torch.pi)[None] * 0.02
            )
        dt = torch.full(path.shape[:2], 0.04)
        dt[:, 0] = 0
        calls.append(
            {
                "joint": targets[0].move_type is MoveType.JOINT_MOVE,
                "count": count,
                "preserve": options.preserve_cartesian_samples,
                "targets": [
                    (v.qpos if v.move_type is MoveType.JOINT_MOVE else v.xpos).tolist()
                    for v in targets
                ],
            }
        )
        return PlanResult(
            success=torch.tensor([True]),
            positions=path,
            dt=dt,
            waypoint_indices=(
                torch.tensor([visits])
                if targets[0].move_type is MoveType.EEF_MOVE
                and not options.preserve_cartesian_samples
                else None
            ),
        )

    gen = object.__new__(MotionGenerator)
    gen.robot = robot
    gen.device = robot.device
    gen.planner = NS(cfg=NS(planner_type="oracle"), collision_world_info=None)
    gen._generate_once = generate
    mesh = trimesh.creation.box(extents=[0.03, 0.03, 0.06])
    sem = ObjectSemantics(
        AxisAlignAffordance(
            mesh_vertices=torch.tensor(mesh.vertices, dtype=torch.float32),
            mesh_triangles=torch.tensor(mesh.faces),
            internal_axis=torch.tensor([0.0, 0.0, 1.0]),
        ),
        {},
        "object",
    )
    pose = torch.eye(4)[None]
    pose[:, :3, 3] = torch.tensor([0.2, 0.0, 0.1])
    context = PlanningContext(
        RobotObservation(0.0, initial, torch.zeros_like(initial)),
        TaskState(1, "cpu"),
        SceneSnapshot(0.0, 0, {"object": EntityState(pose)}),
        torch.tensor([0]),
        control_dt=0.04,
    )
    return gen, context, sem, calls, ik_calls


def place_fixture(directory: Path):
    from dataclasses import replace
    from embodichain.lab.sim.atomic_actions import (
        HeldObjectState,
        PlaceGoal,
        SceneEntityPose,
    )

    generator, context, semantics, calls, ik = ordinary_fixture(directory)
    qpos = context.robot.qpos.clone()
    qpos[:, 3:] = 0.01
    generator.robot.get_qpos.return_value = qpos
    target = generator.robot.compute_fk(qpos[:, :3], name="arm", to_matrix=True)
    target[:, :3, 3] = torch.tensor([0.3, 0.15, 0.1])
    context = replace(
        context,
        robot=RobotObservation(0.0, qpos, torch.zeros_like(qpos)),
        task=TaskState(
            1,
            "cpu",
            held_objects={
                "arm": HeldObjectState(semantics, torch.eye(4)[None], target.clone())
            },
        ),
        scene=SceneSnapshot(0.0, 0, {"target": EntityState(target)}),
    )
    return generator, context, PlaceGoal(SceneEntityPose("target")), calls, ik


def assert_native_motion_geometry_source(value):
    """Check the declared shape authority of one real Motion report."""
    sources = {
        "discrete_robot_meshes": "native_cooked_robot_shapes",
        "discrete_free_motion_meshes": "native_cooked_robot_and_typed_scene_shapes",
        "discrete_precontact_meshes": "native_cooked_robot_and_typed_scene_shapes",
        "discrete_robot_articulation_meshes": (
            "native_cooked_robot_and_articulation_usd_collision_inputs"
        ),
    }
    if "backend" in value:
        assert value["backend"] == "newton.CollisionPipeline"
    assert type(value.get("valid")) is bool
    if value["valid"]:
        assert value.get("scope") in sources
        assert type(value.get("frames")) is int
        assert value["geometry_source"] == sources[value["scope"]]
    else:
        # Failed Motion reports retain frame/kind/pairs but no success scope.
        assert type(value.get("frame")) is int
        assert value.get("kind") in {"scene", "other_arm", "self"}
        assert value.get("pairs") and all(len(pair) == 2 for pair in value["pairs"])
        assert value["geometry_source"] in set(sources.values())


def source_record_after_verified_native_metadata(
    actual, reference, *, approach_frames=None, native_rest_culls=None
):
    """Assert new native facts before comparing the original behavior record.

    Only added Motion identity, successful native rest culling diagnostics and
    explicitly checked far-envelope receipts are projected. Every original
    field, tensor digest, candidate/call order, RNG and budget remains compared.
    """
    from collections.abc import Mapping
    from importlib.metadata import version
    import math

    native_rest_culls = native_rest_culls or {}
    motion_scopes = {
        "discrete_robot_meshes",
        "discrete_robot_articulation_meshes",
        "discrete_free_motion_meshes",
        "discrete_precontact_meshes",
    }

    def project(value, expected, path=()):
        if isinstance(value, Mapping):
            expected = expected if isinstance(expected, Mapping) else {}
            removed = set()
            if "backend" in value and "backend" not in expected:
                assert value["backend"] == "newton.CollisionPipeline"
                assert_native_motion_geometry_source(value)
                assert type(value.get("valid")) is bool
                if value["valid"]:
                    assert value.get("scope") in motion_scopes
                    assert type(value.get("frames")) is int
                else:
                    # MotionCollisionReport's original failure projection has
                    # frame/kind/pairs, retaining no success-scope field.
                    assert type(value.get("frame")) is int
                    assert value.get("kind") in {"scene", "other_arm", "self"}
                    assert value.get("pairs")
                    assert all(len(pair) == 2 for pair in value["pairs"])
                removed.add("backend")
            if "geometry_source" in value and "geometry_source" not in expected:
                assert_native_motion_geometry_source(value)
                removed.add("geometry_source")
            return {
                key: project(item, expected.get(key), path + (key,))
                for key, item in value.items()
                if key not in removed
            }
        if isinstance(value, (list, tuple)):
            if path and path[-1] == "convex_clearance" and expected == [] and value:
                assert approach_frames is not None
                for receipt in value:
                    assert set(receipt) == {
                        "entity",
                        "component",
                        "valid",
                        "scope",
                        "required_clearance_m",
                        "numerical_allowance_m",
                        "minimum_distance_m",
                        "reported_distance_semantics",
                        "checked_frames",
                        "native_queries",
                        "bound_only_frames",
                        "collision_backend",
                        "collision_backend_version",
                        "distance_semantics",
                        "geometry_representation",
                        "last_query_kind",
                        "last_distance_interval_m",
                    }
                    assert receipt["entity"] == receipt["component"] == "table"
                    assert receipt["valid"] is True
                    assert (
                        receipt["scope"]
                        == "discrete_source_convex_solids_not_native_contact"
                    )
                    assert receipt["collision_backend"] == "newton.CollisionPipeline"
                    assert receipt["collision_backend_version"] == version("newton")
                    assert (
                        receipt["geometry_representation"]
                        == "declared_input_convex_envelope"
                    )
                    assert (
                        receipt["reported_distance_semantics"]
                        == "conservative_clearance_lower_bound"
                    )
                    assert (
                        receipt["distance_semantics"] == "capped_conversion_interval/v1"
                    )
                    assert receipt["required_clearance_m"] == 0.002
                    assert receipt["numerical_allowance_m"] == 1e-6
                    assert (
                        receipt["checked_frames"]
                        == receipt["bound_only_frames"]
                        == approach_frames
                    )
                    assert receipt["native_queries"] == 0
                    assert receipt["last_query_kind"] == "canonical_bound"
                    assert "native_contact_result" not in receipt
                    lower, upper = receipt["last_distance_interval_m"]
                    threshold = (
                        receipt["required_clearance_m"]
                        + receipt["numerical_allowance_m"]
                    )
                    assert math.isfinite(lower) and lower > threshold and upper is None
                    assert math.isfinite(receipt["minimum_distance_m"])
                    assert threshold < receipt["minimum_distance_m"] <= lower
                return []
            expected = expected if isinstance(expected, (list, tuple)) else ()
            return type(value)(
                project(
                    item,
                    expected[index] if index < len(expected) else None,
                    path + (str(index),),
                )
                for index, item in enumerate(value)
            )
        return value

    return project(actual, reference)
