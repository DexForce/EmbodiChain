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

"""Simulation-independent fixed-time IK branch selection."""

from __future__ import annotations

from collections.abc import Callable
import math

import torch

__all__ = ["resolve_seeded_ik_path"]


def resolve_seeded_ik_path(
    poses: torch.Tensor,
    start: torch.Tensor,
    *,
    solve: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    solve_many: (
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
    pose_valid: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    limits: torch.Tensor,
    dt: float,
    candidate_predicate: Callable[[torch.Tensor], bool] | None = None,
    record_operation: Callable[[str, int], None] | None = None,
) -> torch.Tensor | None:
    """Resolve one fixed-time Cartesian path with bounded local IK seed search.

    The previous accepted posture seeds the next waypoint. Near a joint limit,
    candidates retain the original serial or batched perturbation order and
    maximize the minimum joint-limit margin. This operation does not draw random
    values itself; callbacks retain ownership of their random stream.

    Args:
        poses: Cartesian waypoints with shape ``(T, 4, 4)``.
        start: Initial posture with shape ``(1, D)``; only one path is supported.
        solve: Solve one pose and seed, returning a validity mask and joint values.
        solve_many: Optional multi-seed solver for the same target pose.
        pose_valid: Optional final FK check for joint rows and requested poses.
        lower: Lower joint bounds broadcastable to ``(1, D)``.
        upper: Upper joint bounds broadcastable to ``(1, D)``.
        limits: Positive joint velocities broadcastable to ``(1, D)``.
        dt: Unchanged interval between consecutive output samples.
        candidate_predicate: Additional predicate for the serial solver path.
        record_operation: Optional operation/count observer owned by the caller.

    Returns:
        Joint rows with shape ``(T + 1, D)``, including the start, or ``None``
        when the bounded search fails. Failure does not prove unreachability.

    Raises:
        ValueError: If shapes, timing, or velocity limits are invalid, or a
            serial-only predicate is combined with the multi-seed solver.
    """
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("Cartesian path poses must have shape (T, 4, 4).")
    if solve_many is not None and candidate_predicate is not None:
        raise ValueError("candidate_predicate requires the serial solver path.")
    if start.ndim != 2 or start.shape[0] != 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError(
            "Continuous IK requires one environment and a positive timestep."
        )
    if not torch.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Continuous IK requires finite positive velocity limits.")
    seed = start.clone()
    points = [seed]
    allowed = limits * dt

    def record_search(operation: str, *, count: int = 1) -> None:
        if record_operation is not None:
            record_operation(operation, count)

    def feasible(ok: torch.Tensor, q: torch.Tensor) -> bool:
        return bool(
            ok.all()
            and torch.isfinite(q).all()
            and (q >= lower).all()
            and (q <= upper).all()
            and ((q - seed).abs() <= allowed + 1e-5).all()
            and (candidate_predicate is None or candidate_predicate(q))
        )

    for pose in poses:
        target = pose[None] if isinstance(pose, torch.Tensor) else pose
        record_search("ik_solve")
        ok, q = solve(target, seed)
        q = q.reshape_as(seed)
        valid = feasible(ok, q)
        margin = float(torch.minimum(q - lower, upper - q).min()) if valid else -1.0
        if margin < 0.02:
            if solve_many is None:
                best = q if valid else None
                for joint in range(seed.shape[-1]):
                    for offset in (-0.15, 0.15, -0.05, 0.05):
                        trial = seed.clone()
                        trial[:, joint] += offset
                        record_search("ik_solve")
                        good, candidate = solve(target, trial.clamp(lower, upper))
                        candidate = candidate.reshape_as(seed)
                        if feasible(good, candidate):
                            score = float(
                                torch.minimum(
                                    candidate - lower, upper - candidate
                                ).min()
                            )
                            if score > margin:
                                best, margin = candidate, score
                if best is None:
                    return None
                q = best
                seed = q
                points.append(seed)
                continue
            trials = [seed[0]]
            for offset in (-0.05, 0.05, -0.15, 0.15):
                for joint in range(seed.shape[-1]):
                    trial = seed[0].clone()
                    trial[joint] += offset
                    trials.append(trial.clamp(lower, upper).reshape(-1))
            record_search("ik_batch_solve")
            record_search("ik_seed", count=len(trials))
            good, candidates = solve_many(target, torch.stack(trials))
            if valid:
                candidates = torch.cat((q, candidates))
                good = torch.cat(
                    (torch.ones(1, dtype=torch.bool, device=good.device), good)
                )
            good &= torch.isfinite(candidates).all(-1)
            good &= ((candidates >= lower) & (candidates <= upper)).all(-1)
            good &= ((candidates - seed).abs() <= allowed + 1e-5).all(-1)
            if not good.any():
                return None
            margins = torch.minimum(candidates - lower, upper - candidates).amin(-1)
            index = torch.argmax(torch.where(good, margins, -torch.inf))
            q = candidates[index : index + 1]
        seed = q
        points.append(seed)
    result = torch.cat(points)
    if pose_valid is not None and not pose_valid(result[1:], poses).all():
        return None
    return result
