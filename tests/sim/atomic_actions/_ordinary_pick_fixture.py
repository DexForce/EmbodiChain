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


def closing_query_parts(part_type, *, blocked=False):
    import numpy as np

    parts = []
    for sign in (-1, 1):
        poses = np.repeat(np.eye(4)[None], 65, axis=0)
        poses[:, 0, 3] = sign * np.linspace(0.15, 0.08, 65)
        parts.append(
            part_type(
                f"pad_{sign}",
                (trimesh.creation.box((0.02, 0.02, 0.02)),),
                poses,
                0.0,
                True,
                0.002,
            )
        )
    if blocked:
        poses = np.repeat(np.eye(4)[None], 65, axis=0)
        parts.append(
            part_type(
                "housing",
                (trimesh.creation.box((0.02, 0.02, 0.02)),),
                poses,
                0.0,
                False,
                0.002,
            )
        )
    return tuple(parts), (trimesh.creation.box((0.2, 0.2, 0.2)),)


def stable_closing_report(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {
            {
                "clearance_lower_bound_m": "minimum_distance_m",
                "first_nonpad_contact_halo_clearance_lower_bound_m": "first_nonpad_contact_halo_minimum_distance_m",
                "mesh_clearance_lower_bound_m": "mesh_distance_m",
                "minimum_rest_clearance_lower_bound_m": "minimum_rest_clearance_m",
            }.get(key, key): stable_closing_report(item)
            for key, item in value.items()
            if key
            not in {
                "query_seconds",
                "collision_backend",
                "collision_backend_version",
                "distance_semantics",
                "build_seconds",
                "query_scope",
                "object_pose_at_capture",
            }
            # Native query identity is checked at the Motion boundary. The
            # frozen record here compares candidate/search/command behavior.
            and not (key == "backend" and item == "newton.CollisionPipeline")
        }
    if isinstance(value, (list, tuple)):
        return [stable_closing_report(item) for item in value]
    return value


def native_candidates():
    from embodichain.toolkits.graspkit.pg_grasp import (
        AntipodalGraspPoseGenerator,
        AntipodalGraspPoseGeneratorCfg,
        ParallelJawGraspCollisionCfg,
    )

    generator = object.__new__(AntipodalGraspPoseGenerator)
    generator._gripper_model = ParallelJawGripperModelCfg(
        model_id="native_proposal_fixture", max_opening_width=0.128
    )
    generator._algorithm_cfg = AntipodalGraspPoseGeneratorCfg(
        approach_direction_samples=1,
        approach_deviation_angle=0.35,
        max_candidates=80,
        center_mode="centroid",
    )
    generator._collision_cfg = ParallelJawGraspCollisionCfg(
        filter_ground_collision=False
    )
    y, z = torch.meshgrid(
        torch.linspace(-0.03, 0.03, 11), torch.linspace(-0.03, 0.03, 11), indexing="ij"
    )
    centers = torch.stack((torch.zeros_like(y), y, z), -1).reshape(-1, 3)
    offset = torch.tensor([0.015, 0.0, 0.0])
    pairs = torch.stack((centers - offset, centers + offset), 1)
    generator.prepare_mesh = lambda **kwargs: pairs
    return generator


def native_end_candidates():
    """Controlled contact population through the real supported-end selector."""
    generator = native_candidates()

    def candidates(
        *,
        obj_poses,
        obj_longest_axis=None,
        is_positive_part=True,
        axis_region_selector=None,
        **kwargs,
    ):
        rows = []
        for row, pose in enumerate(obj_poses):
            centers = pose.new_tensor(
                [[0.0, 0.0, -0.015], [0.0, 0.0, 0.0], [0.0, 0.0, 0.015]]
            )
            mask = torch.ones(3, dtype=torch.bool, device=pose.device)
            if obj_longest_axis is not None and axis_region_selector is not None:
                axis = pose[:3, :3].T @ obj_longest_axis[row]
                positive = (
                    bool(is_positive_part[row])
                    if isinstance(is_positive_part, torch.Tensor)
                    and is_positive_part.ndim
                    else bool(is_positive_part)
                )
                selected = axis_region_selector(
                    centers=centers,
                    pair_ids=torch.arange(3),
                    pair_centers=centers,
                    axis=axis,
                    positive=positive,
                    center_mode="centroid",
                )
                mask = torch.zeros_like(mask) if selected is None else selected[0]
            values = pose.repeat(int(mask.sum()), 1, 1)
            values[:, :3, :3] = torch.diag(pose.new_tensor([1.0, -1.0, -1.0]))
            values[:, :3, 3] += centers[mask] @ pose[:3, :3].T
            rows.append((values, torch.arange(3, dtype=pose.dtype)[mask]))
        return rows

    generator.get_valid_grasp_poses = candidates
    return generator
