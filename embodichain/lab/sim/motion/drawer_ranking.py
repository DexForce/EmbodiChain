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

"""Sequential endpoint IK ranking before bounded complete drawer planning."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import torch

__all__ = ["prefilter_drawer_entry_paths"]


def prefilter_drawer_entry_paths(
    candidates: tuple[torch.Tensor, ...],
    seed: torch.Tensor,
    solve: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None
    ],
    *,
    endpoint_collision: (
        Callable[[str, torch.Tensor], Sequence[dict[str, Any]]] | None
    ) = None,
    max_candidates: int = 2,
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Rank outside/rim/release IK paths without treating ranking as certification.

    Each candidate restarts the original seed, then carries each solved endpoint
    forward. The optional collision callback receives the stage and controlled
    joint state, and may append an explicit release-open-hand check. If no screen
    survives, the first ranked path still receives a complete-planner attempt.
    """
    if seed.ndim != 2 or seed.shape[0] != 1:
        raise ValueError("Drawer candidate planning requires batch_size=1.")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be positive.")
    stages = ("outside", "rim", "release")
    records: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        candidate_seed = seed.clone()
        solved = 0
        failure_stage: str | None = None
        failure_reason: str | None = None
        failure_detail: str | None = None
        joint_distance = 0.0
        endpoint_collisions = []
        target = candidate
        for stage_index, stage_name in enumerate(stages):
            try:
                result = solve(target[:, stage_index], candidate_seed)
                if result is None:
                    raise RuntimeError("IK solver is unavailable")
                success, qpos = result
                success = torch.as_tensor(
                    success, dtype=torch.bool, device=candidate_seed.device
                ).reshape(-1)
                qpos = torch.as_tensor(
                    qpos, dtype=candidate_seed.dtype, device=candidate_seed.device
                )
                if qpos.ndim == 3:
                    qpos = qpos[:, 0]
                if qpos.shape != candidate_seed.shape:
                    raise ValueError(
                        "IK prefilter returned qpos with shape "
                        f"{tuple(qpos.shape)}, expected {tuple(candidate_seed.shape)}"
                    )
                if not bool(success[0]) or not torch.isfinite(qpos).all():
                    failure_stage = stage_name
                    failure_reason = "ik_failed"
                    break
                joint_distance += float(
                    torch.linalg.vector_norm(qpos - candidate_seed).item()
                )
                candidate_seed = qpos
                solved += 1
                if endpoint_collision is not None:
                    endpoint_collisions.extend(endpoint_collision(stage_name, qpos))
            except Exception as exc:  # pragma: no cover - backend-specific failure
                failure_stage = stage_name
                failure_reason = f"ik_error:{type(exc).__name__}"
                failure_detail = str(exc)
                break
        records.append(
            {
                "candidate_id": f"drawer_candidate_{index}",
                "index": index,
                "prefilter": "endpoint_ik",
                "solved_stages": solved,
                "accepted": solved == len(stages),
                "failure_stage": failure_stage,
                "failure_reason": failure_reason,
                "failure_detail": failure_detail,
                "joint_distance": joint_distance,
                **(
                    {"articulation_endpoints": endpoint_collisions}
                    if endpoint_collision is not None
                    else {}
                ),
            }
        )

    ranked = sorted(
        range(len(candidates)),
        key=lambda index: (
            not records[index]["accepted"],
            sum(
                not r["valid"] for r in records[index].get("articulation_endpoints", [])
            ),
            -records[index]["solved_stages"],
            records[index]["joint_distance"],
            index,
        ),
    )
    survivors = [index for index in ranked if records[index]["accepted"]]
    selected = survivors[:max_candidates]
    if not selected:
        # Preserve one complete-planner attempt even when the cheap screen has
        # no survivor; endpoint IK is intentionally not a hard rejection gate.
        selected = ranked[:1]
    return tuple(selected), records
