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

from embodichain.lab.sim.affordance.grasp.cache import validated_label_provider

from embodichain.lab.sim.affordance.sampling import (
    AffordancePoseCandidates,
    AffordanceSample,
    AffordanceSamplingContext,
    _sample_pose_candidates,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.toolkits.graspkit.pg_grasp import GraspPoseGenerator
    from embodichain.lab.sim.affordance.interactions import AntipodalAffordance
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
    calibration = options.grasp_frame_to_eef
    if not torch.equal(calibration, torch.eye(4).to(calibration)):
        adjustment["grasp_frame_to_eef"] = calibration.tolist()
    return adjustment


def select_ranked_grasp(
    labels: tuple[torch.Tensor, torch.Tensor, str],
    object_pose: torch.Tensor,
    env_ids: torch.Tensor,
    key: str,
    feasible: Callable[[torch.Tensor], torch.Tensor],
    *,
    sampling: AffordanceSamplingContext | None = None,
) -> AffordanceSample:
    """Apply Affordance branches to the best reachable, physically tested TCPs."""
    local, order, cache_key = labels
    # Affordance diversifies the first max(32, 4*count) feasible candidates.
    # A nominal request only needs the first reachable label.
    limit = max(32, 4 * sampling.count) if sampling and sampling.enabled else 1
    capacity = max(1, min(limit, order.shape[1]))
    counts = torch.zeros(len(object_pose), dtype=torch.long, device=order.device)
    ranks = order.new_full((len(counts), capacity), -1)
    ids = torch.full_like(ranks, -1)
    poses = torch.eye(4).to(object_pose).repeat(len(counts), capacity, 1, 1)
    for offset in range(0, order.shape[1], 50):
        block = order[:, offset : offset + 50]
        world = object_pose[:, None] @ local[block.clamp_min(0)]
        ok = (block >= 0) & feasible(world)
        slots = counts[:, None] + ok.long().cumsum(1) - 1
        rows, cols = torch.where(ok & (slots < capacity))
        dest = slots[rows, cols]
        ranks[rows, dest] = offset + cols
        ids[rows, dest] = block[rows, cols]
        poses[rows, dest] = world[rows, cols]
        counts += ok.sum(1)
        if (
            offset + 50 >= order.shape[1]
            or ((counts >= capacity) | (order[:, offset + 50] < 0)).all()
        ):
            break

    sample = _sample_pose_candidates(
        AffordancePoseCandidates(poses, ranks.to(poses.dtype), ids >= 0),
        sampling,
        env_ids=env_ids,
        key=key,
        reference_poses=object_pose,
    )
    rows = torch.arange(len(counts), device=order.device)
    selected = torch.tensor(
        sample.metadata["candidate_ids"], device=order.device
    ).clamp_min(0)
    sample.metadata.update(
        validated_cache=cache_key,
        candidate_ids=ids[rows, selected].masked_fill(~sample.success, -1).tolist(),
        candidate_ranks=ranks[rows, selected].masked_fill(~sample.success, -1).tolist(),
    )
    return sample


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
    hand_commands: tuple[torch.Tensor, torch.Tensor],
) -> AffordanceSample:
    """Check physical handle labels on the live link; custom query services own their sampling."""
    labels = validated_label_provider(generator, hand_commands)
    if labels is not None:
        pool = labels(
            affordance.mesh_vertices,
            affordance.mesh_triangles,
            affordance.validation_attrs,
            link_pose,
            direction,
            on_support=False,
        )
        return select_ranked_grasp(
            pool,
            link_pose,
            context.env_ids,
            key,
            feasible,
            sampling=context.affordance_sampling,
        )
    return affordance.sample_candidates(
        affordance.get_grasp_candidates(generator, link_pose, direction),
        sampling=context.affordance_sampling,
        env_ids=context.env_ids,
        key=key,
        reference_poses=link_pose,
    )
