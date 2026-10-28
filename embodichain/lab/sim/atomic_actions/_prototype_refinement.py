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

"""Ground skill goals and mesh/gripper evidence for prototype refinement."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from embodichain.lab.sim.motion.refinement import MotionRefinementCfg
from embodichain.toolkits.graspkit.pose_generator import (
    GraspPoseGenerator,
    ParallelJawGripperModelCfg,
)

from .affordance import AntipodalAffordance
from .bindings import JointPositionTarget
from .goals import resolve_pose_goal
from .invocation import ResolvedActionRequest
from .state import PlanningContext
from .trajectory_ops import resolve_pose_target

__all__: list[str] = []


def _world_goal(value: object, context: PlanningContext, name: str) -> np.ndarray:
    """Resolve a single exact semantic target using the ordinary goal resolver."""
    pose = resolve_pose_target(
        resolve_pose_goal(value, context, name=name),
        num_envs=1,
        device=context.robot.qpos.device,
    )
    if pose.ndim == 4:
        if pose.shape[1] != 1:
            raise ValueError("refinement currently requires one semantic target pose")
        pose = pose[:, 0]
    return pose[0].detach().cpu().numpy()


def _pick_refinement_anchor(
    request: ResolvedActionRequest,
    context: PlanningContext,
    object_pose: np.ndarray,
    source_anchor: np.ndarray,
    cfg: MotionRefinementCfg,
    generators: Mapping[str, GraspPoseGenerator],
    motion_generator: object,
) -> tuple[np.ndarray, str, dict[str, object]]:
    """Choose a bounded mesh-aware anchor without mutating source or task state."""
    goal = request.goal
    affordance = goal.semantics.affordance
    if not isinstance(affordance, AntipodalAffordance):
        raise ValueError("Pick refinement requires an antipodal grasp affordance")
    affordance.require_whole_object_mesh()
    if affordance.mesh_vertices is None or affordance.mesh_triangles is None:
        raise ValueError("Pick refinement requires explicit object mesh geometry")
    hand = request.binding.endpoint("primary", "grasp").require_target(
        JointPositionTarget
    )
    generator = generators.get(hand.control_part)
    model = getattr(generator, "gripper_model", None)
    if not isinstance(model, ParallelJawGripperModelCfg):
        raise ValueError("Pick refinement requires calibrated gripper geometry")
    vertices = affordance.mesh_vertices.detach().cpu().numpy().astype(np.float64)
    triangles = affordance.mesh_triangles.detach().cpu().numpy().astype(np.int64)
    digest = hashlib.sha256(
        vertices.astype("<f8").tobytes() + triangles.astype("<i8").tobytes()
    )
    digest.update(json.dumps(model.to_dict(), sort_keys=True).encode())
    if goal.grasp_xpos is not None:
        candidates = _world_goal(goal.grasp_xpos, context, "grasp_xpos")[None]
        origin = "explicit_goal"
    elif request.skill_options.fixed_object_to_eef is not None:
        fixed = request.skill_options.fixed_object_to_eef.detach().cpu().numpy()
        if fixed.shape == (1, 4, 4):
            fixed = fixed[0]
        candidates = (object_pose @ fixed)[None]
        origin = "configured_object_grasp"
    else:
        direction = torch.as_tensor(
            request.skill_options.approach_direction,
            dtype=context.robot.qpos.dtype,
            device=context.robot.qpos.device,
        )
        device = context.robot.qpos.device
        devices = [] if device.type == "cpu" else [device.index or 0]
        # The source generator predates explicit RNG arguments. Isolate its
        # torch stream instead of changing process-wide RNG state permanently.
        with torch.random.fork_rng(devices=devices):
            torch.default_generator.manual_seed(cfg.seed)
            if devices:
                torch.cuda.default_generators[devices[0]].manual_seed(cfg.seed)
            pool = affordance.get_grasp_candidates(
                generator,
                torch.as_tensor(object_pose, dtype=direction.dtype, device=device)[
                    None
                ],
                direction,
                obj_longest_axis=(
                    None
                    if request.skill_options.pick_object_part == "center"
                    else torch.tensor([0.0, 0.0, 1.0], device=device)
                ),
                is_positive_part=request.skill_options.pick_object_part != "bottom",
            )
        eligible = pool.valid[0] & torch.isfinite(pool.costs[0])
        candidates = pool.poses[0, eligible].detach().cpu().numpy()
        origin = "affordance_mesh"
    if not len(candidates):
        raise ValueError("refinement found no legal grasp candidates")
    translation = np.linalg.norm(candidates[:, :3, 3] - source_anchor[:3, 3], axis=1)
    angular = Rotation.from_matrix(
        candidates[:, :3, :3] @ source_anchor[:3, :3].T
    ).magnitude()
    valid = (translation <= cfg.max_translation_m) & (angular <= cfg.max_rotation_rad)
    order = np.argsort(
        translation / cfg.max_translation_m + angular / cfg.max_rotation_rad,
        kind="stable",
    )
    tested = 0
    for index in order[valid[order]][: cfg.max_candidates]:
        tested += 1
        candidate = candidates[index]
        local = np.linalg.inv(candidate) @ object_pose
        points = vertices @ local[:3, :3].T + local[:3, 3]
        width = float(np.ptp(points[:, 0]))
        # This is a necessary jaw-envelope check. Exact hand/self/scene mesh
        # collisions and intended-contact depth remain independently validated.
        if not model.min_opening_width <= width <= model.max_opening_width:
            continue
        if float(points[:, 0].min()) > 0 or float(points[:, 0].max()) < 0:
            continue
        if (
            abs(float((points[:, 2].max() + points[:, 2].min()) / 2))
            > model.finger_length / 2
        ):
            continue
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        solved = motion_generator.robot.compute_ik(
            torch.as_tensor(
                candidate,
                dtype=context.robot.qpos.dtype,
                device=context.robot.qpos.device,
            )[None],
            joint_seed=context.robot.qpos[:, list(arm.joint_ids)],
            name=arm.control_part,
            env_ids=context.env_ids.tolist(),
        )
        if (
            solved is None
            or not isinstance(solved[0], torch.Tensor)
            or not bool(solved[0].all())
        ):
            continue
        return (
            candidate.copy(),
            digest.hexdigest(),
            {
                "anchor_origin": origin,
                "gripper_model_id": model.model_id,
                "object_opening_extent_m": width,
                "candidate_index": int(index),
                "candidates_tested": tested,
                "anchor_ik_verified": True,
                "source_grasp_translation_error_m": float(translation[index]),
                "source_grasp_rotation_error_rad": float(angular[index]),
            },
        )
    raise ValueError("refinement has no grasp within geometry and correction budgets")
