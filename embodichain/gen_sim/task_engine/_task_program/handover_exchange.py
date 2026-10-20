# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Bounded native-scene exchange qualification on an unchanged grasp pool."""

from __future__ import annotations

import math
from typing import Any

import torch

from embodichain.lab.sim.atomic_actions.affordance_sampling import (
    AffordancePoseCandidates,
)
from embodichain.lab.sim.atomic_actions.trajectory_ops import translate_pose_world

from .held_alignment_recovery import _RandomState, _cuda_devices

__all__: list[str] = []

HANDOVER_EXCHANGE_REVISION = 1
_UNAVAILABLE = (
    AttributeError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _endpoint(
    robot: Any,
    context: Any,
    arm: Any,
    target: torch.Tensor,
    seed: torch.Tensor,
    counter: dict[str, Any],
) -> tuple[bool, torch.Tensor, dict[str, Any]]:
    runtime = robot.get_qpos_limits(name=arm.control_part, env_ids=context.env_ids).to(
        seed
    )
    solver = robot.get_solver(arm.control_part)
    lower = solver.lower_qpos_limits.to(seed)
    upper = solver.upper_qpos_limits.to(seed)
    if (
        runtime.shape != (*seed.shape, 2)
        or lower.shape != seed.shape[1:]
        or upper.shape != seed.shape[1:]
        or not torch.isfinite(runtime).all()
        or not torch.isfinite(lower).all()
        or not torch.isfinite(upper).all()
    ):
        raise ValueError("handover_exchange_joint_limits_unavailable")
    lower = torch.maximum(runtime[..., 0], lower)
    upper = torch.minimum(runtime[..., 1], upper)
    if (lower > upper).any():
        raise ValueError("handover_exchange_joint_limits_inconsistent")
    counter["ik_queries"] += 1
    success, qpos = robot.compute_ik(
        pose=target,
        joint_seed=seed.detach().clone(),
        name=arm.control_part,
        env_ids=context.env_ids,
    )
    if (
        not isinstance(success, torch.Tensor)
        or success.dtype is not torch.bool
        or success.shape != (1,)
        or not isinstance(qpos, torch.Tensor)
        or qpos.shape != seed.shape
    ):
        raise ValueError("handover_exchange_endpoint_ik_shape_unavailable")
    finite = bool(torch.isfinite(qpos).all())
    within = finite and bool(((qpos >= lower) & (qpos <= upper)).all())
    accepted = bool(success.all()) and within
    return (
        accepted,
        qpos.detach().clone(),
        {
            "control_part": arm.control_part,
            "solver_success": bool(success.all()),
            "finite": finite,
            "within_runtime_and_solver_limits": within,
            "accepted": accepted,
        },
    )


def select_exchange(
    screen: Any,
    robot: Any,
    request: Any,
    context: Any,
    resources: Any,
    held: Any,
    exchange_pose: torch.Tensor,
    source_root: torch.Tensor,
    destination_root: torch.Tensor,
    original_pool: AffordancePoseCandidates,
    original_mask: torch.Tensor,
    approach_samples: int,
    pre_grasp_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Qualify at most three geometry-derived internal exchange stations.

    Args:
        screen: Same-tick native geometry screen with bounded station proposals.
        robot: Existing invocation-bound robot and its original IK solvers.
        request: Original resolved handover request, never modified.
        context: Original planning snapshot; no simulation is advanced.
        resources: Shared HandOver's bound source/destination participants.
        held: Existing verified source attachment and object-to-TCP transform.
        exchange_pose: Original internal object exchange pose.
        source_root: Original source arm root in the planning arena.
        destination_root: Original destination arm root in the planning arena.
        original_pool: Original poses, costs and validity, without resampling.
        original_mask: Original station's native approach qualification mask.
        approach_samples: Unchanged shared receiving-approach sample count.
        pre_grasp_distance: Unchanged receiving pre-grasp distance.

    Returns:
        Internal object pose, rigidly translated pool poses, qualification mask,
        and diagnostics. The complete shared plan and native command guard are
        still required; endpoint IK is not a trajectory or physical certificate.
    """
    poses, costs = original_pool.poses, original_pool.costs
    if (
        poses.ndim != 4
        or poses.shape[0] != 1
        or poses.shape[2:] != (4, 4)
        or costs.shape != poses.shape[:2]
        or original_pool.valid.shape != costs.shape
        or original_pool.valid.dtype is not torch.bool
        or original_mask.shape != costs.shape
        or original_mask.dtype is not torch.bool
    ):
        raise ValueError("handover_exchange_original_pool_unavailable")
    eligible = (
        original_pool.valid
        & torch.isfinite(costs)
        & torch.isfinite(poses).all(dim=(-2, -1))
    )
    if (original_mask & ~eligible).any():
        raise ValueError("handover_exchange_original_mask_identity_mismatch")
    evidence: dict[str, Any] = {
        "revision": HANDOVER_EXCHANGE_REVISION,
        "scope": "native_receiver_approach_and_both_arm_endpoint_ik_not_complete_plan",
        "accepted": False,
        "station_budget": 3,
        "ik_query_budget": 9,
        "budget_scope": "one_original_receive_pool",
        "ik_queries": 0,
        "station_attempts": [],
        "selected_pool_index": None,
        "selected_translation": None,
        "requires_complete_shared_plan_and_command_guard": True,
    }

    def unchanged(reason: str):
        evidence["reason"] = reason
        return exchange_pose, poses, original_mask, evidence

    if original_mask.any():
        return unchanged("original_station_qualified")
    if not eligible.any():
        return unchanged("original_pool_empty")
    if (
        context.batch_size != 1
        or context.env_ids.tolist() != [0]
        or context.affordance_sampling is not None
        or exchange_pose.shape != (1, 4, 4)
        or not torch.isfinite(exchange_pose).all()
        or not math.isfinite(pre_grasp_distance)
        or pre_grasp_distance < 0
    ):
        return unchanged("unsupported_exchange_scope")
    state = _RandomState.capture(_cuda_devices(robot.device, context.robot.qpos.device))
    try:
        proposals, geometry = screen.exchange_candidates(
            exchange_pose,
            poses,
            eligible,
            resources=resources,
            approach_samples=approach_samples,
            pre_grasp_distance=pre_grasp_distance,
            source_root=source_root,
            destination_root=destination_root,
        )
        evidence["proposal"] = geometry
        if len(proposals) > evidence["station_budget"]:
            raise ValueError("handover_exchange_station_budget_exceeded")
        proposals = sorted(
            proposals,
            key=lambda value: float(torch.linalg.vector_norm(value["translation"])),
        )
        source, destination = resources.first.arm, resources.second.arm
        source_seed = context.robot.qpos[:, list(source.joint_ids)].detach().clone()
        receiver_seed = (
            context.robot.qpos[:, list(destination.joint_ids)].detach().clone()
        )
        offset = held.object_to_eef.to(exchange_pose)
        if offset.ndim == 2:
            offset = offset.unsqueeze(0)
        if offset.shape != (1, 4, 4) or not torch.isfinite(offset).all():
            raise ValueError("handover_exchange_source_attachment_unavailable")
        for proposal in proposals:
            delta = proposal["translation"].to(exchange_pose)
            mask = proposal["mask"]
            if (
                delta.shape != (1, 3)
                or not torch.isfinite(delta).all()
                or mask.shape != eligible.shape
                or mask.dtype is not torch.bool
                or (mask & ~eligible).any()
            ):
                raise ValueError("handover_exchange_native_proposal_unavailable")
            note = {
                "axis": proposal["axis"],
                "translation": delta.detach().cpu().tolist(),
                "norm_m": float(torch.linalg.vector_norm(delta)),
                "native_approach": proposal["report"],
                "endpoint_ik": [],
                "accepted": False,
            }
            evidence["station_attempts"].append(note)
            if not mask.any():
                note["reason"] = "native_approach_rejected"
                continue
            selected = int(costs.masked_fill(~mask, float("inf")).argmin(1)[0])
            note["pool_index"] = selected
            station, translated = exchange_pose.clone(), poses.clone()
            station[:, :3, 3] += delta
            translated[..., :3, 3] += delta[:, None]
            grasp = translated[:, selected]
            pre = translate_pose_world(grasp, -grasp[:, :3, 2] * pre_grasp_distance)
            endpoints = (
                ("source_exchange", source, station @ offset, source_seed),
                ("receive_pregrasp", destination, pre, receiver_seed),
                ("receive_grasp", destination, grasp, receiver_seed),
            )
            pre_seed = receiver_seed
            for label, arm, target, seed in endpoints:
                accepted, solution, result = _endpoint(
                    robot,
                    context,
                    arm,
                    target,
                    pre_seed if label == "receive_grasp" else seed,
                    evidence,
                )
                note["endpoint_ik"].append({"target": label, **result})
                if not accepted:
                    note["reason"] = "endpoint_ik_rejected"
                    break
                if label == "receive_pregrasp":
                    pre_seed = solution
            else:
                note["accepted"] = evidence["accepted"] = True
                evidence["reason"] = "geometry_and_endpoint_ik_qualified"
                evidence["selected_pool_index"] = selected
                evidence["selected_translation"] = delta.detach().cpu().tolist()
                return station, translated, mask, evidence
        return unchanged("no_qualified_exchange_station")
    except _UNAVAILABLE as error:
        evidence["unavailable"] = str(error)
        return unchanged("exchange_qualification_unavailable")
    finally:
        state.restore()
