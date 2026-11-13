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
"""Bounded vertical residuals for unintended pre-grasp hand contact."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np
import open3d as o3d
from ._contact_fit import _body, _mesh, _transform
from ._kinematics import Robot
from ._hand_geometry import hand_collision_surfaces

__all__: list[str] = []


def _idle_windows(
    phases: list[dict], grounding: dict, count: int, lead: int
) -> list[dict]:
    """Join consecutive source closures that did not transport an object."""
    windows = []
    for side in sorted({event["side"] for event in phases}):
        arm = [event for event in phases if event["side"] == side]
        i = 0
        while i < len(arm):
            event = arm[i]
            if (
                grounding.get((side, event["grasp"]), {}).get("object_id", "unknown")
                is not None
            ):
                i += 1
                continue
            first = i
            while (
                i + 1 < len(arm)
                and grounding.get((side, arm[i + 1]["grasp"]), {}).get(
                    "object_id", "unknown"
                )
                is None
            ):
                i += 1
            windows.append(
                dict(
                    side=side,
                    start=max(
                        arm[first - 1]["open_end"] if first else 0,
                        arm[first]["close_start"] - lead,
                    ),
                    checked_end=(
                        arm[i + 1]["close_start"] if i + 1 < len(arm) else count - 1
                    ),
                    fade_end=arm[i + 1]["grasp"] if i + 1 < len(arm) else count - 1,
                    grasps=[e["grasp"] for e in arm[first : i + 1]],
                )
            )
            i += 1
    return windows


def _lift_envelope(required: np.ndarray, radius: int = 15) -> np.ndarray:
    """Spread positive corrections without pinning an artificial window edge."""
    result = required.copy()
    for i in np.flatnonzero(required > 0):
        left, right = min(radius, i), min(radius, len(required) - 1 - i)
        ids = np.arange(i - left, i + right + 1)
        distance = (ids - i) / np.where(ids <= i, max(1, left), max(1, right))
        result[ids] = np.maximum(
            result[ids], required[i] * np.cos(distance * np.pi / 2) ** 2
        )
    return result


def clear_approach(
    robot: Robot,
    qpos: np.ndarray,
    side: int,
    scene: dict,
    deltas: dict[str, np.ndarray],
    visuals: tuple[dict, ...],
    start: int,
    close_start: int,
    solve_ik: Callable,
    *,
    grasp: int | None = None,
    clearance_m: float = 0.003,
) -> tuple[np.ndarray, dict]:
    """Lift only the original approach samples whose hand sweep hits geometry.

    The pre-closure lift fades to zero at the grasp anchor. This preserves
    grasp, transport and release anchors while allowing the closing transition
    to adapt. It is a
    local clearance residual, not a task planner or a full-arm collision check.
    Exhausting this local search returns its finite candidate and diagnostics;
    it does not reject a recording or establish physical task failure.
    """
    if not np.isfinite(clearance_m) or not 0 < clearance_m < 0.15:
        raise ValueError("Approach clearance must be between zero and 0.15 metres")
    end = close_start if grasp is None else grasp
    if end < close_start or end >= len(qpos):
        raise ValueError("Invalid grasp anchor for approach clearance")
    indices = np.arange(max(0, start), end + 1)
    checked_count = close_start - int(indices[0]) + 1
    result = qpos.copy()
    fingers = hand_collision_surfaces(robot, side)
    bodies = []
    for obj in scene["objects"]:
        mesh = _mesh(Path(obj["path"]))
        mesh.apply_transform(
            deltas.get(obj["id"], np.eye(4)) @ np.asarray(obj["transform"])
        )
        bodies.append(_body(obj["id"], mesh, np.eye(4)[None], 16))

    def distance(body, points):
        tensor = o3d.core.Tensor(points.astype(np.float32))
        if body.closed:
            return (
                body.bvh.compute_signed_distance(tensor, nthreads=2, nsamples=3)
                .numpy()
                .min()
            )
        return body.bvh.compute_distance(tensor, nthreads=2).numpy().min()

    total = np.zeros(len(indices))
    hit_frames = set()
    maximum_ik_error = 0.0

    def finish(status, reason=None):
        return result, dict(
            method="local upward TCP residual with a fade toward the grasp",
            status=status,
            reason=reason,
            corrected_frames=sorted(hit_frames),
            collision_links=list(fingers),
            geometry="authored URDF hand collision surfaces",
            max_lift_m=float(total.max()),
            minimum_checked_gap_m=None if not np.isfinite(minimum) else minimum,
            preferred_clearance_m=clearance_m,
            max_ik_error_m=maximum_ik_error,
            iterations=iteration,
            fade_end_frame=end,
            physical_success_verified=False,
        )

    # The caller includes its physical contact envelope and tracking allowance.
    # Aim slightly above that minimum to accommodate the IK residual.
    margin = clearance_m + 0.001
    # Smoothing a lift can bring a neighboring sample into contact. Recheck the
    # resulting sweep after every correction, including the final allowed one.
    max_corrections = 8
    for iteration in range(max_corrections + 1):
        fk = robot.links(result[indices])
        required = np.zeros(len(indices))
        minimum = float("inf")
        for link, local in fingers.items():
            world = _transform(local, fk[link])
            for i, points in enumerate(world[:checked_count]):
                lo, hi = points.min(0) - margin, points.max(0) + margin
                for body in bodies:
                    if np.any(hi < body.mesh.bounds[0]) or np.any(
                        lo > body.mesh.bounds[1]
                    ):
                        continue
                    gap = float(distance(body, points))
                    minimum = min(minimum, gap)
                    if gap >= clearance_m:
                        continue
                    hit_frames.add(int(indices[i]))
                    lower = 0.0
                    upper = max(
                        0.0, float(body.mesh.bounds[1, 2] - points[:, 2].min()) + margin
                    )
                    for _ in range(12):
                        middle = (lower + upper) / 2
                        shifted = points + [0, 0, middle]
                        if distance(body, shifted) >= margin:
                            upper = middle
                        else:
                            lower = middle
                    required[i] = max(required[i], upper)
        if not required.any():
            return finish("adjusted")
        if iteration == max_corrections:
            break
        lift = _lift_envelope(required)
        if (total + lift).max() > 0.15:
            return finish("review_needed", "Local vertical search budget exhausted")
        tcps = robot.tcp(result[indices], side)
        candidate = result.copy()
        for i in np.flatnonzero(lift > 1e-6):
            target = tcps[i].copy()
            target[2, 3] += lift[i]
            corrected, fit = solve_ik(robot, result[indices[i]], side, target)
            if not np.isfinite(corrected).all():
                return finish("review_needed", "IK returned nonfinite candidate")
            maximum_ik_error = max(maximum_ik_error, float(fit["tcp_error_m"]))
            candidate[indices[i]] = corrected
        result = candidate
        total += lift
    return finish("review_needed", "Local correction iteration budget exhausted")
