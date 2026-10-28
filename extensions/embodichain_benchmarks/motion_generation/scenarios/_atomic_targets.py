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

"""Independently freeze feasible held-object targets before planner evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from embodichain.compute.kinematics import pose_error
from ._atomic_reference import _recover_reference_ik
from .atomic_case_utils import (
    apply_target_rotation_randomization,
    canonical_case_qpos,
    float_vector,
    randomized_vector_batch,
)

if TYPE_CHECKING:
    from .atomic_skill import AtomicSkillScenario

__all__: list[str] = []

_TARGET_STREAM_STRIDE = 1_000_003


def _reference_chain_rows(
    scenario: AtomicSkillScenario,
    start: torch.Tensor,
    targets: torch.Tensor,
    env_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return qualified endpoint chains; a failed row stays failed."""
    if scenario.robot is None or scenario.suite is None:
        raise RuntimeError("Atomic Skill runtime is not configured.")
    robot = scenario.robot
    protocol = scenario.suite.protocol
    count, waypoints = targets.shape[:2]
    feasible = torch.ones(count, dtype=torch.bool, device=start.device)
    references = start[:, None].expand(-1, waypoints, -1).clone()
    previous = start.clone()
    for index in range(waypoints):
        local = feasible.nonzero(as_tuple=False).flatten()
        if local.numel() == 0:
            break
        selected = [env_ids[i] for i in local.cpu().tolist()]
        goal = targets[local, index]
        seed = previous[local]
        flags, solved = robot.compute_ik(
            pose=goal, joint_seed=seed, name=scenario.control_part, env_ids=selected
        )
        solved = canonical_case_qpos(solved)
        lower, upper = (
            robot.get_qpos_limits(name=scenario.control_part, env_ids=selected)
            .to(solved)
            .unbind(-1)
        )
        violation = torch.maximum(lower - solved, solved - upper).clamp_min(0).amax(-1)
        valid = torch.as_tensor(flags, device=start.device, dtype=torch.bool).reshape(
            -1
        )
        valid = (
            valid
            & torch.isfinite(solved).all(-1)
            & (violation <= protocol.joint_limit_tolerance_rad)
        )
        check = valid.nonzero(as_tuple=False).flatten()
        if check.numel():
            actual = robot.compute_fk(
                qpos=solved[check],
                name=scenario.control_part,
                env_ids=[selected[i] for i in check.cpu().tolist()],
                to_matrix=True,
            )
            dp, dr = pose_error(actual, goal[check])
            valid[check] &= (dp.norm(dim=-1) <= protocol.position_threshold_m) & (
                dr.norm(dim=-1) <= protocol.rotation_threshold_rad
            )
        if not bool(valid.all().item()):
            valid, solved = _recover_reference_ik(
                robot,
                goal,
                seed,
                valid,
                solved,
                control_part=scenario.control_part,
                protocol=protocol,
                env_ids=selected,
            )
        feasible[local] &= valid
        accepted = local[valid]
        previous[accepted] = solved[valid]
        references[accepted, index] = solved[valid]
    return feasible, references


def _held_target_reference(
    scenario: AtomicSkillScenario,
    config: Mapping[str, object],
    *,
    seed: int,
    start: torch.Tensor,
    table_object_pose: torch.Tensor,
    object_to_eef: torch.Tensor,
    translation_stream: int,
    rotation_stream: int,
    default_offset: Sequence[float],
    height: float,
    waypoint_heights: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Use original targets by default; optionally condition on reference evidence.

    Candidate zero is the original target law. Further candidates change only
    goal-local random streams, keeping the original seed, task frame, proposal
    amplitudes and shell row radii. Freeze each row at its first qualified full
    endpoint chain. This opt-in selection changes the benchmark population and
    must be reviewed and frozen before evaluating any planner.
    """
    count = config.get("target_candidate_count", 1)
    if type(count) is not int or not 1 <= count <= 32:
        raise ValueError("target_candidate_count must be an integer in [1,32].")
    batch = start.shape[0]
    objects = table_object_pose.clone()
    targets = objects[:, None].expand(-1, len(waypoint_heights), -1, -1).clone()
    references = start[:, None].expand(-1, len(waypoint_heights), -1).clone()
    selected_attempts = torch.full((batch,), -1, dtype=torch.long, device=start.device)
    found = torch.zeros(batch, dtype=torch.bool, device=start.device)
    attempts: list[dict[str, object]] = []
    for attempt in range(count):
        candidate = table_object_pose.clone()
        stream_offset = attempt * _TARGET_STREAM_STRIDE
        candidate[:, :3, 3] += randomized_vector_batch(
            float_vector(
                config.get("target_object_offset_m"),
                name="target_object_offset_m",
                length=3,
                default=default_offset,
            ),
            config,
            jitter_name="target_object_offset_jitter_m",
            seed=seed,
            stream=translation_stream + stream_offset,
            batch_size=batch,
            dtype=candidate.dtype,
            device=candidate.device,
        )
        apply_target_rotation_randomization(
            candidate, config, seed=seed, stream=rotation_stream + stream_offset
        )
        candidate[:, 2, 3] += height
        eef = candidate @ object_to_eef
        poses = eef[:, None].expand(-1, len(waypoint_heights), -1, -1).clone()
        poses[:, :, 2, 3] += torch.as_tensor(
            waypoint_heights, dtype=poses.dtype, device=poses.device
        )[None]
        if count == 1:
            return candidate, poses, scenario.solve_reference_qpos(start, poses), {}
        ids = (~found).nonzero(as_tuple=False).flatten()
        env_ids = ids.cpu().tolist()
        valid, qpos = _reference_chain_rows(scenario, start[ids], poses[ids], env_ids)
        accepted = ids[valid]
        objects[accepted] = candidate[accepted]
        targets[accepted] = poses[accepted]
        references[accepted] = qpos[valid]
        selected_attempts[accepted] = attempt
        found[accepted] = True
        attempts.append(
            {
                "attempt": attempt,
                "environment_rows": env_ids,
                "qualified": valid.detach().cpu().tolist(),
                "target_object_poses": candidate[ids].detach().cpu().tolist(),
            }
        )
        if bool(found.all().item()):
            return (
                objects,
                targets,
                references,
                {
                    "target_selection": {
                        "rule": "first_reference_feasible_target_v1",
                        "candidate_count": count,
                        "goal_stream_stride": _TARGET_STREAM_STRIDE,
                        "selected_attempts": selected_attempts.cpu().tolist(),
                        "attempts": attempts,
                        "scope": "endpoint_reference_only_not_path_or_physical_success",
                    }
                },
            )
    missing = (~found).nonzero(as_tuple=False).flatten().cpu().tolist()
    raise RuntimeError(
        f"Independent reference search exhausted {count} target candidates for rows {missing}."
    )
