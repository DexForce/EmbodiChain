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
"""Bounded vertical residuals for unintended pre-grasp finger contact."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np
import open3d as o3d
import trimesh
from ._contact_fit import _body, _mesh, _transform
from ._kinematics import Robot

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
                    checked_end=arm[i]["open_end"],
                    fade_end=arm[i + 1]["grasp"] if i + 1 < len(arm) else count - 1,
                    grasps=[e["grasp"] for e in arm[first : i + 1]],
                )
            )
            i += 1
    return windows


def _lift_envelope(required: np.ndarray, radius: int = 15) -> np.ndarray:
    """Spread positive corrections smoothly, keeping both phase endpoints fixed."""
    result = required.copy()
    for i in np.flatnonzero(required > 0):
        left, right = min(radius, i), min(radius, len(required) - 1 - i)
        if not left or not right:
            raise ValueError(
                "Approach endpoint collides; cannot preserve grasp anchors"
            )
        ids = np.arange(i - left, i + right + 1)
        distance = (ids - i) / np.where(ids <= i, left, right)
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
) -> tuple[np.ndarray, dict]:
    """Lift only the original approach samples whose finger sweep hits geometry.

    The pre-closure lift fades to zero at the grasp anchor. This preserves
    grasp, transport and release anchors while allowing the closing transition
    to adapt. It is a
    local clearance residual, not a task planner or a full-arm collision check.
    """
    end = close_start if grasp is None else grasp
    if end < close_start or end >= len(qpos):
        raise ValueError("Invalid grasp anchor for approach clearance")
    indices = np.arange(max(0, start), end + 1)
    checked_count = close_start - int(indices[0]) + 1
    result = qpos.copy()
    fingers = {}
    for link in robot.finger_surfaces(side)[0]:
        parts = []
        for visual in visuals:
            if visual["link"] == link:
                part = _mesh(Path(visual["path"]))
                part.apply_transform(np.asarray(visual["origin"]))
                parts.append(part)
        if not parts:
            raise ValueError("Approach clearance requires actual finger geometry")
        mesh = trimesh.util.concatenate(parts)
        points, _ = trimesh.sample.sample_surface(mesh, 1024, seed=42)
        vertices = np.asarray(mesh.vertices)
        if len(vertices) > 1024:
            vertices = vertices[np.linspace(0, len(vertices) - 1, 1024).astype(int)]
        fingers[link] = np.concatenate([vertices, points])
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
    margin = 0.003
    for iteration in range(3):
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
                    if gap >= (0.001 if iteration else margin):
                        continue
                    hit_frames.add(int(indices[i]))
                    lower = 0.0
                    upper = max(
                        0.0, float(body.mesh.bounds[1, 2] - points[:, 2].min()) + margin
                    )
                    if upper + total[i] > 0.15:
                        raise ValueError(
                            "Approach clearance needs more than 15 cm vertical correction"
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
            return result, dict(
                method="local upward TCP residual with fixed boundary anchors",
                corrected_frames=sorted(hit_frames),
                max_lift_m=float(total.max()),
                minimum_checked_gap_m=None if not np.isfinite(minimum) else minimum,
                iterations=iteration,
                fade_end_frame=end,
            )
        lift = _lift_envelope(required)
        total += lift
        if total.max() > 0.15:
            raise ValueError("Smoothed approach correction exceeds 15 cm")
        tcps = robot.tcp(result[indices], side)
        for i in np.flatnonzero(lift > 1e-6):
            target = tcps[i].copy()
            target[2, 3] += lift[i]
            corrected, fit = solve_ik(robot, result[indices[i]], side, target)
            if fit["tcp_error_m"] > 0.002:
                raise ValueError("IK could not realize the required approach clearance")
            result[indices[i]] = corrected
    raise ValueError("Approach remains in contact after bounded local correction")
