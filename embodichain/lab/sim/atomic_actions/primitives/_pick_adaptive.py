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
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

from embodichain.compute.geometry.gripper import (
    adaptive_grasp_approaches,
)
from ..affordance_sampling import AffordancePoseCandidates, AffordanceSample
from ..bindings import JointPositionTarget
from ..invocation import ResolvedActionRequest
from ..state import PlanningContext

if TYPE_CHECKING:
    from .pick_up import PickUp, PickUpOptions

from ._grasp_sampling import (
    observed_axis as _observed_axis,
    select_grasp as _select_grasp,
)

__all__: list[str] = []


@dataclass
class PickPlanningState:
    request: ResolvedActionRequest
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


_PICK_STATE: ContextVar[PickPlanningState | None] = ContextVar(
    "pick_planning_state", default=None
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
) -> AffordanceSample:
    state = current_pick_state()
    if state is None:
        raise RuntimeError("Adaptive Pick requires one active planning scope.")
    state.object_pose = object_pose
    affordance = semantics.affordance
    purpose = options.adaptive_grasp
    axis = _observed_axis(affordance, object_pose)
    tcp = action.robot.compute_fk(
        qpos=start_qpos, name=manipulator.control_part, to_matrix=True
    )
    choices = adaptive_grasp_approaches(
        axis, object_pose, tcp, ends=False, purpose=purpose
    )
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
    for index, choice in enumerate(choices):

        def observe_pool(pool):
            state.candidates = pool
            state.candidate_routes.append((index, pool, choice.direction.clone()))

        def observe_raw_pool(pool):
            state.raw_candidate_routes.append((index, pool, choice.direction.clone()))

        def feasible(values, approach):
            return action._select_feasible_grasp_variants(
                values,
                start_qpos,
                object_pose,
                manipulator,
                options,
                approach[:, None, None, :],
            )

        sample, direction = _select_grasp(
            affordance,
            generator,
            object_pose,
            (choice,),
            context,
            f"{sample_key}:strategy_{index}",
            feasible=feasible,
            candidate_observer=observe_pool,
            raw_candidate_observer=observe_raw_pool,
        )
        viable = sample.success & ~success
        attempt = {"strategy": index, "grasp_feasible": sample.success.tolist()}
        attempts.append(attempt)
        if not viable.any():
            continue
        path_ok, candidate_path, candidate_lengths = action._build_pickup_trajectory(
            sample.poses,
            start_qpos,
            context.last_qpos,
            state.request.motion_policy,
            options,
            direction,
            manipulator,
            hand,
            *commands,
            context.require_control_dt(),
        )
        accepted = viable & path_ok
        attempt["path_feasible"] = path_ok.tolist()
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
                "strategy_indices": strategies.tolist(),
                "approach_directions": directions.tolist(),
                "attempts": attempts,
            },
        },
    )
