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

"""Select physically labelled TCPs in bounded, row-local IK batches."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch

from .affordance_sampling import AffordanceSample, _pose_sampling_metadata

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.toolkits.graspkit.pg_grasp import GraspPoseGenerator
    from .affordance import AntipodalAffordance
    from .state import PlanningContext
    from .primitives.pick_up import PickUpOptions

__all__ = []


def pickup_adjustment(options: PickUpOptions) -> dict:
    """Describe optional TCP changes which must precede physical validation."""
    adjustment = {}
    if options.grasp_variant == "mirrored":
        adjustment["mirrored"] = True
    if options.rotate_upright:
        adjustment.update(
            rotate_upright=options.rotate_upright,
            upright_axis=(
                options.obj_upright_direction.tolist()
                if options.obj_upright_direction is not None
                else [0, 0, 1]
            ),
        )
    return adjustment


def select_ranked_grasp(
    labels: tuple[torch.Tensor, torch.Tensor, str],
    object_pose: torch.Tensor,
    env_ids: torch.Tensor,
    key: str,
    feasible: Callable[[torch.Tensor], torch.Tensor],
    *,
    calibration: torch.Tensor | None = None,
) -> AffordanceSample:
    """Select the first feasible TCP per row, preserving its original label ID."""
    local, order, cache_key = labels
    rows = torch.arange(len(object_pose), device=object_pose.device)
    chosen = torch.full_like(rows, -1)
    selected = torch.eye(4, device=object_pose.device, dtype=object_pose.dtype).repeat(
        len(rows), 1, 1
    )
    rank = torch.full_like(rows, -1)
    for offset in range(0, order.shape[1], 50):
        ids = order[:, offset : offset + 50]
        poses = object_pose[:, None] @ local[ids.clamp_min(0)]
        if calibration is not None:
            poses = poses @ calibration
        ok = feasible(poses) & (ids >= 0)
        first = ok.long().argmax(dim=1)
        accept = (chosen < 0) & ok.any(dim=1)
        chosen = torch.where(accept, ids[rows, first], chosen)
        rank = torch.where(accept, offset + first, rank)
        selected = torch.where(accept[:, None, None], poses[rows, first], selected)
        if (
            offset + 50 >= order.shape[1]
            or ((chosen >= 0) | (order[:, offset + 50] < 0)).all()
        ):
            break
    success = chosen >= 0
    return AffordanceSample(
        success,
        selected,
        {
            **_pose_sampling_metadata(selected, success, env_ids, object_pose),
            "key": key,
            "validated_cache": cache_key,
            "candidate_ranks": rank.tolist(),
            "candidate_ids": chosen.tolist(),
        },
    )


def reachable_path(
    robot: Robot,
    control_part: str,
    start_qpos: torch.Tensor,
    env_ids: torch.Tensor | None,
    waypoints: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Continue batch IK from approach to grasp and across skill-specific waypoints."""
    seed = start_qpos[:, None, :]
    feasible = torch.ones_like(seed[..., 0], dtype=torch.bool)
    for poses in waypoints:
        seed = seed.expand(-1, poses.shape[1], -1)
        ok, joints = robot.compute_batch_ik(
            pose=poses,
            name=control_part,
            joint_seed=seed,
            **({"env_ids": env_ids} if env_ids is not None else {}),
        )
        feasible = feasible & ok.reshape(poses.shape[:2]).bool()
        seed = torch.where(feasible[..., None], joints.reshape_as(seed), seed)
    return feasible


def offset_pose(poses: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Translate a batch of candidate poses without changing their rotations."""
    result = poses.clone()
    result[..., :3, 3] += delta
    return result


def sample_link_grasp(
    affordance: AntipodalAffordance,
    generator: GraspPoseGenerator,
    link_pose: torch.Tensor,
    direction: torch.Tensor,
    context: PlanningContext,
    key: str,
    feasible: Callable[[torch.Tensor], torch.Tensor],
) -> AffordanceSample:
    """Check physical handle labels on the live link; custom query services own their sampling."""
    labels = getattr(generator, "validated_labels", None)
    if labels is not None:
        pool = labels(
            affordance.mesh_vertices,
            affordance.mesh_triangles,
            affordance.validation_attrs,
            link_pose,
            direction,
            on_support=False,
        )
        return select_ranked_grasp(pool, link_pose, context.env_ids, key, feasible)
    return affordance.sample_candidates(
        affordance.get_grasp_candidates(generator, link_pose, direction),
        sampling=context.affordance_sampling,
        env_ids=context.env_ids,
        key=key,
        reference_poses=link_pose,
    )
