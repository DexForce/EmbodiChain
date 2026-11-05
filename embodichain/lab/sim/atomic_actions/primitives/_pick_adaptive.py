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

"""Per-plan adaptive Pick geometry and original candidate/path ordering."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, TYPE_CHECKING

import torch

from embodichain.compute.geometry.gripper import (
    adaptive_grasp_approaches,
)
from ..affordance_sampling import AffordancePoseCandidates, AffordanceSample
from embodichain.lab.sim.motion.search import PlanningRandomState
from embodichain.utils.math import pose_inv
from ..bindings import JointPositionTarget
from ..invocation import ResolvedActionRequest
from ..state import PlanningContext

if TYPE_CHECKING:
    from .pick_up import PickUp, PickUpOptions
    from embodichain.lab.sim.atomic_actions.source_native_contact import (
        NativeClosingQuery,
    )

from ._grasp_sampling import (
    observed_axis as _observed_axis,
    select_grasp as _select_grasp,
)

__all__: list[str] = []


@dataclass
class PickPlanningState:
    request: ResolvedActionRequest
    pour_inputs: Any = None
    ik_audit: dict[str, Any] = field(default_factory=dict)
    active_ik_audit: dict[str, Any] | None = None
    direction: torch.Tensor | None = None
    trajectory: tuple | None = None
    object_pose: torch.Tensor | None = None
    candidates: AffordancePoseCandidates | None = None
    candidate_routes: list[tuple[int, AffordancePoseCandidates, torch.Tensor]] = field(
        default_factory=list
    )
    raw_candidate_routes: list[tuple[int, AffordancePoseCandidates, torch.Tensor]] = (
        field(default_factory=list)
    )
    depth_recovery_attempts: int = 0
    closing_query: NativeClosingQuery | None = None
    depth_trace: list[dict[str, Any]] = field(default_factory=list)
    closing_scene_checks: list[dict[str, Any]] = field(default_factory=list)
    native_candidate_expanders: dict[int, Callable[[], AffordancePoseCandidates]] = (
        field(default_factory=dict)
    )


_PICK_STATE: ContextVar[PickPlanningState | None] = ContextVar(
    "atomic_pick_planning", default=None
)


def current_pick_state() -> PickPlanningState | None:
    return _PICK_STATE.get()


@contextmanager
def pick_planning_scope(request: ResolvedActionRequest) -> Iterator[PickPlanningState]:
    state = PickPlanningState(request)
    token = _PICK_STATE.set(state)
    try:
        yield state
    finally:
        _PICK_STATE.reset(token)


@contextmanager
def _depth_random_scope(checkpoint: PlanningRandomState | None, device: torch.device):
    if checkpoint is None:
        yield
        return
    current = PlanningRandomState.capture((device,))
    checkpoint.restore()
    try:
        yield
    finally:
        current.restore()


def adaptive_pick_sample(
    action: PickUp,
    semantics: Any,
    object_pose: torch.Tensor,
    start_qpos: torch.Tensor,
    manipulator: JointPositionTarget,
    generator: Any,
    options: PickUpOptions,
    context: PlanningContext,
    *,
    sample_key: str,
    choices_override: Any = None,
) -> AffordanceSample:
    state = current_pick_state()
    if state is None:
        raise RuntimeError("Adaptive Pick requires one active planning scope.")
    state.object_pose = object_pose
    affordance = semantics.affordance
    purpose = options.adaptive_grasp
    if choices_override is None:
        axis = _observed_axis(affordance, object_pose)
        tcp = action.robot.compute_fk(
            qpos=start_qpos, name=manipulator.control_part, to_matrix=True
        )
        choices = adaptive_grasp_approaches(
            axis, object_pose, tcp, ends=purpose == "handover_source", purpose=purpose
        )
    else:
        choices = choices_override
    grasp = state.request.binding.endpoint("primary", "grasp")
    hand = grasp.require_target(JointPositionTarget)
    commands = [
        grasp.joint_positions(
            name,
            num_envs=context.batch_size,
            device=action.device,
            dtype=context.robot.qpos.dtype,
        )
        for name in ("open", "grasp")
    ]
    success = torch.zeros(context.batch_size, dtype=torch.bool, device=action.device)
    poses = object_pose.clone()
    directions = torch.zeros_like(object_pose[:, :3, 3])
    strategies = torch.full_like(success, -1, dtype=torch.long)
    candidate_ids = torch.full_like(strategies, -1)
    full, lengths = None, None
    attempts = []
    depth_enabled = (
        state.closing_query is not None
        and state.closing_query.revision in (6, 7, 8, 9)
        and not options.constrained_grasp
    )
    depths = state.closing_query.depth_offsets() if depth_enabled else (0.0,)
    pools: dict[int, AffordancePoseCandidates] = {}
    checkpoints: dict[int, PlanningRandomState] = {}
    selected_depth = object_pose.new_zeros(context.batch_size)
    budget = (
        state.request.recovery_policy.planning_recovery.max_candidate_paths
        if depth_enabled
        else 0
    )
    for index, (choice, depth) in enumerate(
        (choice, depth) for depth in depths for choice in choices
    ):
        choice_index = index % len(choices)
        if depth and state.depth_recovery_attempts >= budget:
            break
        effective = options
        canonical_shift = None
        pool_costs = None
        if depth:
            transform = options.grasp_frame_to_eef.clone()
            transform[:3, 3] -= depth * transform[:3, 2]
            effective = replace(options, grasp_frame_to_eef=transform)
            canonical_shift = transform @ pose_inv(options.grasp_frame_to_eef)

        def observe_pool(pool):
            state.candidates = pool
            if purpose == "ordinary":
                state.candidate_routes.append((index, pool, choice.direction.clone()))

        def observe_raw_pool(pool):
            nonlocal pool_costs
            pool_costs = pool.costs
            if depth_enabled and not depth:
                pools[choice_index] = pool
                checkpoints[choice_index] = PlanningRandomState.capture(
                    (object_pose.device,)
                )
            if canonical_shift is not None:
                pool = replace(pool, poses=pool.poses @ canonical_shift.to(pool.poses))
            state.raw_candidate_routes.append((index, pool, choice.direction.clone()))

        def feasible(values, approach):
            limit = (
                getattr(generator, "native_proposal_limit", None)
                if choice.axis is None
                else None
            )
            return action._select_feasible_grasp_variants(
                values,
                start_qpos,
                object_pose,
                manipulator,
                effective,
                approach[:, None, None, :],
                **(
                    {
                        "candidate_costs": pool_costs,
                        "candidate_limit": limit,
                        "candidate_route": index,
                        "first_feasible": purpose == "ordinary"
                        and not getattr(context.affordance_sampling, "enabled", False),
                    }
                    if limit is not None
                    else {}
                ),
            )

        with _depth_random_scope(
            checkpoints[choice_index] if depth else None, object_pose.device
        ):
            sample, direction = _select_grasp(
                affordance,
                generator,
                object_pose,
                (choice,),
                context,
                f"{sample_key}:strategy_{choice_index}",
                feasible=feasible,
                candidate_observer=observe_pool,
                raw_candidate_observer=observe_raw_pool,
                candidate_pool=pools[choice_index] if depth else None,
                axis_region_policy=(
                    "supported_end" if purpose == "handover_source" else "mesh_extent"
                ),
            )
            path_rng = (
                PlanningRandomState.capture((object_pose.device,)) if depth else None
            )
        viable = sample.success & ~success
        attempt = {"strategy": index, "grasp_feasible": sample.success.tolist()}
        if depth_enabled:
            attempt["retreat_m"] = depth
            state.depth_trace.append(attempt)
        attempts.append(attempt)
        if not viable.any():
            continue
        if depth:
            state.depth_recovery_attempts += 1
        with _depth_random_scope(path_rng, object_pose.device):
            path_ok, candidate_path, candidate_lengths = (
                action._build_pickup_trajectory(
                    sample.poses,
                    start_qpos,
                    context.last_qpos,
                    state.request.motion_policy,
                    effective,
                    direction,
                    manipulator,
                    hand,
                    *commands,
                    context.require_control_dt(),
                )
            )
        accepted = viable & path_ok
        attempt["path_feasible"] = path_ok.tolist()
        if (
            depth_enabled
            and accepted.any()
            and (
                purpose == "ordinary"
                or (purpose == "handover_source" and state.closing_query.revision == 8)
            )
        ):
            from ._pick_path import _free_motion
            from ._pick_closing import inspect_precontact_scene

            stop = candidate_lengths["approach"]
            opening = _free_motion(
                action, candidate_path[0, stop - 1 : stop], manipulator.control_part
            )
            clear = bool(opening["valid"])
            if state.closing_query.revision >= 7:
                state.closing_scene_checks.append({"stage": "open_endpoint", **opening})
                if clear:
                    closing = inspect_precontact_scene(
                        action,
                        state.request,
                        context,
                        candidate_path[0, stop : stop + candidate_lengths["close"]],
                        state.closing_query,
                    )
                    state.closing_scene_checks.append({"stage": "candidate", **closing})
                    clear = bool(closing["valid"])
            key = (
                "open_and_precontact_scene_valid"
                if state.closing_query.revision >= 7
                else "open_endpoint_scene_valid"
            )
            attempt[key] = clear
            accepted &= clear
        if not accepted.any():
            action._expand_native_pick_candidates(state)
        if full is None:
            full = context.last_qpos[:, None].expand_as(candidate_path).clone()
            lengths = candidate_lengths
        if candidate_path.shape != full.shape or candidate_lengths != lengths:
            raise ValueError("Adaptive Pick strategies must preserve phase timing.")
        full = torch.where(accepted[:, None, None], candidate_path, full)
        poses = torch.where(accepted[:, None, None], sample.poses, poses)
        directions = torch.where(accepted[:, None], direction, directions)
        strategies = torch.where(accepted, index, strategies)
        candidate_ids = torch.where(
            accepted,
            candidate_ids.new_tensor(sample.metadata["candidate_ids"]),
            candidate_ids,
        )
        success |= accepted
        selected_depth = torch.where(
            accepted,
            selected_depth.new_full(selected_depth.shape, depth),
            selected_depth,
        )
        if success.all():
            break
    if full is not None:
        state.trajectory = (success, full, lengths)
    state.direction = directions
    return AffordanceSample(
        success,
        poses,
        {
            "key": sample_key,
            "candidate_ids": candidate_ids.tolist(),
            "selected_poses": poses.detach().cpu().tolist(),
            "adaptive_grasp": {
                **(
                    {"selected_retreat_m": selected_depth.tolist()}
                    if depth_enabled
                    else {}
                ),
                **(
                    {"axis_region_policy": "supported_end"}
                    if purpose == "handover_source"
                    else {}
                ),
                "strategy_indices": strategies.tolist(),
                "approach_directions": directions.tolist(),
                "attempts": attempts,
            },
        },
    )
