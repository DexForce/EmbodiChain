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

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.affordance.grasp.cache import (
    ValidatedGraspCache,
    validated_label_provider,
)

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
    from .bindings import EndpointBinding
    from .invocation import ResolvedSkillRequest
    from .state import PlanningContext
    from .primitives.pick_up import PickUpOptions

__all__ = []

SAMPLED_GRASP_SKILLS = frozenset(
    {"pick_up", "axis_align", "hand_over", "slide", "open_door", "coordinated_pickment"}
)


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


def handover_adjustment(retract: float) -> dict | None:
    """Convert the handover approach-axis offset to a physical TCP adjustment."""
    return {"retract": retract} if retract else None


def grasp_commands(
    endpoint: EndpointBinding,
    *,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the open/grasp pair from an endpoint after invocation overrides."""
    return (
        endpoint.joint_positions("open", num_envs=num_envs, device=device, dtype=dtype),
        endpoint.joint_positions(
            "grasp", num_envs=num_envs, device=device, dtype=dtype
        ),
    )


def prepare_grasp_request(
    request: ResolvedSkillRequest,
    generators: Mapping[str, GraspPoseGenerator],
    context: PlanningContext,
) -> dict[str, tuple[Path, ...]]:
    """Prepare the physical labels read when planning this resolved invocation.

    Args:
        request: Snapshot returned by the owning engine's ``resolve`` method.
        generators: Mounted grasp services keyed by target ID. Hosts may select
            a subset of hands to prepare.
        context: Planning batch defining command row count, dtype and device.

    Returns:
        Cache paths per hand, one for each distinct effective open/grasp pair.
        Skills and goals that bypass sampled physical labels return an empty mapping.
    """
    from embodichain.lab.sim.affordance import AntipodalAffordance

    if request.skill_id not in SAMPLED_GRASP_SKILLS:
        return {}
    goal, options = request.goal, request.skill_options
    affordance = getattr(getattr(goal, "semantics", None), "affordance", None)
    if (
        not isinstance(affordance, AntipodalAffordance)
        or getattr(goal, "grasp_xpos", None) is not None
    ):
        return {}
    adjustment = None
    if request.skill_id == "pick_up":
        if options.fixed_object_to_eef is not None:
            return {}
        adjustment = pickup_adjustment(options)
    elif request.skill_id == "hand_over":
        adjustment = handover_adjustment(options.grasp_retract_distance)
    elif (
        request.skill_id == "coordinated_pickment"
        and options.paired_object_to_eef is not None
    ):
        return {}

    dtype = (
        torch.float32
        if request.skill_id in {"hand_over", "coordinated_pickment"}
        else context.robot.qpos.dtype
    )
    paths = {}
    for endpoint in request.binding.endpoints:
        if endpoint.endpoint_id != "grasp":
            continue
        part = endpoint.target.target_id
        cache = getattr(generators.get(part), "validated_labels", None)
        if isinstance(cache, ValidatedGraspCache):
            commands = grasp_commands(
                endpoint,
                num_envs=context.batch_size,
                device=context.robot.qpos.device,
                dtype=dtype,
            )
            pairs = torch.stack(commands, dim=1).detach().cpu().unique(dim=0)
            paths[part] = tuple(
                cache.precompute(
                    affordance.mesh_vertices,
                    affordance.mesh_triangles,
                    affordance.validation_attrs,
                    hand_commands=(opened, closed),
                    pose_adjustment=adjustment,
                )
                for opened, closed in pairs
            )
    return paths


def select_ranked_grasp(
    labels: tuple[torch.Tensor, torch.Tensor, str],
    object_pose: torch.Tensor,
    env_ids: torch.Tensor,
    key: str,
    feasible: Callable[[torch.Tensor], torch.Tensor],
    *,
    sampling: AffordanceSamplingContext | None = None,
    check_selection: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> AffordanceSample:
    """Apply Affordance branches to the best reachable, physically tested TCPs."""
    local, order, cache_key = labels
    # Affordance diversifies the first max(32, 4*count) feasible candidates.
    # A nominal request only needs the first reachable label.
    limit = max(32, 4 * sampling.count) if sampling and sampling.enabled else 1
    if check_selection is not None:
        limit = max(limit, 32)
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

    rows = torch.arange(len(counts), device=order.device)
    eligible = ids >= 0
    while True:
        sample = _sample_pose_candidates(
            AffordancePoseCandidates(poses, ranks.to(poses.dtype), eligible),
            sampling,
            env_ids=env_ids,
            key=key,
            reference_poses=object_pose,
        )
        selected = torch.tensor(
            sample.metadata["candidate_ids"], device=order.device
        ).clamp_min(0)
        if check_selection is None or not sample.success.any():
            break
        rejected = sample.success & ~check_selection(sample.poses, sample.success)
        if not rejected.any():
            break
        # Remove only colliding labels before assigning the requested branch
        # again. Never send a rejected command or credit an unavailable branch.
        eligible[rows[rejected], selected[rejected]] = False
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
