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

"""Measured-state success rules for independent Atomic skill benchmarks.

Run: embodichain benchmark atomic-action --skill all --smoke
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = ["evaluate_skill"]

_POSITION_TOLERANCE = 0.01
_PLACE_XY_TOLERANCE = 0.015
_ROTATION_TOLERANCE = math.radians(5)
_MIN_LIFT = 0.04
_DROP_MARGIN = 0.15


def _rotation_error(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return the unsigned angle between batched rotation matrices."""
    relative = first.transpose(-1, -2) @ second
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(cosine.clamp(-1, 1))


def _signed_rotation(
    first: torch.Tensor, rotations: torch.Tensor, axis: torch.Tensor
) -> torch.Tensor:
    """Measure signed world-axis rotation, including intermediate tilt extrema."""
    relative = rotations @ first.transpose(-1, -2)
    sine_vector = (
        torch.stack(
            (
                relative[..., 2, 1] - relative[..., 1, 2],
                relative[..., 0, 2] - relative[..., 2, 0],
                relative[..., 1, 0] - relative[..., 0, 1],
            ),
            -1,
        )
        / 2
    )
    sine = (sine_vector * axis).sum(-1)
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.atan2(sine, cosine.clamp(-1, 1))


def evaluate_skill(
    skill: str,
    trace: dict[str, torch.Tensor],
    phases: list[dict[str, Any]],
    metadata: dict[str, Any],
    planned: list[bool],
) -> list[dict[str, Any]]:
    """Score actual scene measurements with ordered, skill-specific stages.

    Trace sample zero precedes playback, sample ``i + 1`` follows waypoint
    ``i``, and the last sample follows the terminal settling hold. Grasp checks
    use measured jaw closure/proximity plus subsequent lift/holding state; they
    do not claim a force-closure certificate.

    Args:
        skill: Canonical benchmark skill selector.
        trace: Actual joint, TCP, object and articulated-joint measurements.
        phases: Absolute waypoint ranges from the compiled Atomic plans.
        metadata: Scene goals, joint indices, gripper presets and axes.
        planned: Per-row Atomic planning success, independently of scene checks.

    Returns:
        Per-row success, first failed stage/reason, and numeric measurements.
    """
    batch = len(planned)
    if not any(planned):
        return [
            {
                "env_index": row,
                "planning_success": False,
                "success": False,
                "failure_stage": "planning",
                "failure_reason": "planner_reported_failure",
                "stages": {},
                "metrics": {},
            }
            for row in range(batch)
        ]
    checks: list[tuple[str, torch.Tensor, str]] = []
    metrics: dict[str, torch.Tensor] = {}
    true = torch.ones(batch, dtype=torch.bool)
    final = len(trace["qpos"]) - 1

    def phase(name: str, action: str | None = None) -> dict[str, Any]:
        matches = [
            item
            for item in phases
            if item["name"] == name and (action is None or item["skill"] == action)
        ]
        if not matches:
            raise ValueError(f"Missing measured phase {action or skill}:{name}")
        return matches[0]

    def hand_open(index: int, name: str = "eef") -> torch.Tensor:
        ids = metadata["hand_joint_ids"][name]
        expected = torch.tensor(metadata["hand_open"][name], dtype=trace["qpos"].dtype)
        closed = torch.tensor(metadata["hand_close"][name], dtype=trace["qpos"].dtype)
        error = (trace["qpos"][index, :, ids] - expected).abs().amax(-1)
        metrics[f"hand_{name}_{index}_open_error"] = error
        # A PGI command is in metres; Robotiq's coupled finger commands are in
        # radians. Require at least 90% reopening in each configured joint's
        # own command span instead of applying a shared dimensional tolerance.
        span = (closed - expected).abs().clamp_min(1e-6)
        remaining = ((trace["qpos"][index, :, ids] - expected).abs() / span).amax(-1)
        metrics[f"hand_{name}_{index}_remaining_closure_ratio"] = remaining
        return remaining <= 0.1

    def near(index: int, part: str = "arm") -> torch.Tensor:
        target = trace["object"] if "object" in trace else trace["link"]
        return (
            torch.linalg.vector_norm(
                trace[part][index, :, :3, 3] - target[index, :, :3, 3], dim=-1
            )
            <= 0.15
        )

    def held(index: int, part: str = "arm", hand: str = "eef") -> torch.Tensor:
        """Check current grasp state without requiring an invariant relative pose."""
        return ~hand_open(index, hand) & near(index, part) & no_drop

    no_drop = true
    if "object" in trace:
        no_drop = (
            trace["object"][:, :, 2, 3].amin(0)
            >= metadata.get("support_z", 0.0) - _DROP_MARGIN
        )

    if skill == "move_joint":
        errors = []
        for item in phases:
            endpoint = item["stop"]
            target = torch.tensor(item["target_qpos"], dtype=trace["qpos"].dtype)
            errors.append(
                (
                    trace["qpos"][endpoint][:, metadata["arm_joint_ids"]]
                    - target[:, metadata["arm_joint_ids"]]
                )
                .abs()
                .amax(-1)
            )
        metrics["joint_target_error_rad"] = torch.stack(errors).amax(0)
        checks.append(
            (
                "reached",
                metrics["joint_target_error_rad"] <= math.radians(0.3),
                "task_goal_miss",
            )
        )
    elif skill in {"pickup", "axis_align", "place", "pour"}:
        pickup_id = "axis_align" if skill == "axis_align" else "pick_up"
        closed = phase("close", pickup_id)["stop"]
        lifted = phase("manipulate" if skill == "axis_align" else "lift", pickup_id)[
            "stop"
        ]
        initial_z = trace["object"][0, :, 2, 3]
        lift_delta = trace["object"][lifted, :, 2, 3] - initial_z
        metrics["lift_m"] = lift_delta
        grasped = ~hand_open(closed) & near(closed)
        lifted_ok = (lift_delta >= _MIN_LIFT) & held(lifted)
        if skill in {"pickup", "axis_align"}:
            checks.append(("grasped", grasped, "object_not_grasped"))
            if skill == "axis_align":
                axis = torch.tensor(
                    metadata["internal_axis"], dtype=trace["qpos"].dtype
                )
                target = torch.tensor(
                    metadata["target_axis"], dtype=trace["qpos"].dtype
                )
                current = trace["object"][lifted, :, :3, :3] @ axis
                metrics["axis_error_rad"] = torch.acos(
                    (
                        (current * target).sum(-1)
                        / (current.norm(dim=-1) * target.norm())
                    ).clamp(-1, 1)
                )
                checks.append(
                    (
                        "aligned",
                        (metrics["axis_error_rad"] <= _ROTATION_TOLERANCE) & lifted_ok,
                        "task_goal_miss",
                    )
                )
            else:
                checks.append(("lifted", lifted_ok, "object_dropped"))
            checks.append(
                (
                    "held",
                    held(final)
                    & (trace["object"][final, :, 2, 3] - initial_z >= _MIN_LIFT)
                    & no_drop,
                    "object_dropped",
                )
            )
        else:
            prerequisite = grasped & lifted_ok
            metrics["prerequisite_success"] = prerequisite
            checks.append(("prerequisite_pickup", prerequisite, "prerequisite_failed"))
            if skill == "place":
                release = phase("release", "place")
                before_release = release["start"]
                released = hand_open(release["stop"]) & held(before_release)
                target = torch.tensor(
                    metadata["object_target_pose"], dtype=trace["qpos"].dtype
                )
                metrics["placement_xy_error_m"] = torch.linalg.vector_norm(
                    trace["object"][final, :, :2, 3] - target[:, :2, 3], dim=-1
                )
                metrics["placement_xy_tolerance_m"] = torch.full(
                    (batch,), _PLACE_XY_TOLERANCE
                )
                # Compare the settled landing with a prior settled sample,
                # excluding the expected gravity-driven fall after release.
                settled = len(trace["qpos"]) - 2
                metrics["landing_drift_m"] = torch.linalg.vector_norm(
                    trace["object"][final, :, :3, 3]
                    - trace["object"][settled, :, :3, 3],
                    dim=-1,
                )
                stable = (metrics["landing_drift_m"] <= _POSITION_TOLERANCE) & (
                    _rotation_error(
                        trace["object"][settled, :, :3, :3],
                        trace["object"][final, :, :3, :3],
                    )
                    <= _ROTATION_TOLERANCE
                )
                checks.extend(
                    (
                        ("released", released, "release_failure"),
                        (
                            "placed",
                            metrics["placement_xy_error_m"] <= _PLACE_XY_TOLERANCE,
                            "task_goal_miss",
                        ),
                        ("stable", stable & no_drop, "object_dropped"),
                    )
                )
            else:
                pouring = [item for item in phases if item["skill"] == "pour"]
                start, stop = pouring[0]["start"], pouring[-1]["stop"]
                internal = torch.tensor(
                    metadata["internal_axis"], dtype=trace["qpos"].dtype
                )
                axis = trace["object"][start, :, :3, :3] @ internal
                angles = _signed_rotation(
                    trace["object"][start, :, :3, :3],
                    trace["object"][start : stop + 1, :, :3, :3],
                    axis,
                )
                target_angle = metadata["pour_angle"]
                sign = 1 if target_angle >= 0 else -1
                metrics["peak_pour_angle_rad"] = (angles * sign).amax(0) * sign
                poured = (
                    metrics["peak_pour_angle_rad"] - target_angle
                ).abs() <= _ROTATION_TOLERANCE
                checks.append(
                    (
                        "poured",
                        poured & held(start) & held(stop),
                        "task_goal_miss",
                    )
                )
    elif skill == "press":
        peak = (trace["joint"] - trace["joint"][0]).amax(0)
        stroke = min(abs(metadata["joint_target"]), metadata["press_distance"])
        metrics.update(peak_press_m=peak, stroke_ratio=peak / max(stroke, 1e-9))
        checks.extend(
            (
                ("contacted", peak > 1e-5, "task_goal_miss"),
                ("pressed", metrics["stroke_ratio"] >= 0.8, "task_goal_miss"),
            )
        )
    elif skill in {"slide", "open_door", "twist"}:
        closed = phase("close")["stop"]
        checks.append(
            ("grasped", ~hand_open(closed) & near(closed), "object_not_grasped")
        )
        motion_name = {"slide": "pull", "open_door": "open", "twist": "twist"}[skill]
        motion = phase(motion_name)
        if skill == "twist":
            axis = torch.tensor(metadata["axis_world"], dtype=trace["qpos"].dtype)
            angles = _signed_rotation(
                trace["arm"][motion["start"], :, :3, :3],
                trace["arm"][motion["stop"], :, :3, :3],
                axis,
            )
            metrics["twist_angle_rad"] = angles
            reached = (angles - metadata["twist_angle"]).abs() <= _ROTATION_TOLERANCE
            stage = "twisted"
        else:
            metrics["joint_goal_error"] = (
                trace["joint"][motion["stop"]] - metadata["joint_target"]
            ).abs()
            reached = metrics["joint_goal_error"] <= (
                _POSITION_TOLERANCE if skill == "slide" else _ROTATION_TOLERANCE
            )
            stage = "slid" if skill == "slide" else "opened"
        checks.extend(
            (
                (stage, reached, "task_goal_miss"),
                ("released", hand_open(final), "release_failure"),
            )
        )
    elif skill == "handover":
        grasp = phase("pickup_close")["stop"]
        receive = phase("receive_close")["stop"]
        handed = phase("handover_release")["stop"]
        placed = phase("place")["stop"]
        source_grasp = ~hand_open(grasp, "left_eef") & near(grasp, "left_arm")
        lift = (
            trace["object"][phase("pickup_transport")["stop"], :, 2, 3]
            - trace["object"][0, :, 2, 3]
        )
        metrics["lift_m"] = lift
        transferred = (
            ~hand_open(receive, "left_eef")
            & ~hand_open(receive, "right_eef")
            & near(receive, "left_arm")
            & near(receive, "right_arm")
            & (lift >= _MIN_LIFT)
        )
        handed_over = (
            hand_open(handed, "left_eef")
            & held(handed, "right_arm", "right_eef")
            & (
                trace["object"][handed, :, 2, 3] - trace["object"][0, :, 2, 3]
                >= _MIN_LIFT
            )
        )
        target = torch.tensor(metadata["target_position"], dtype=trace["qpos"].dtype)
        metrics["delivery_error_m"] = torch.linalg.vector_norm(
            trace["object"][placed, :, :3, 3] - target, dim=-1
        )
        delivered = (
            (metrics["delivery_error_m"] <= 3 * _POSITION_TOLERANCE)
            & hand_open(final, "right_eef")
            & no_drop
        )
        checks.extend(
            (
                ("grasped", source_grasp, "object_not_grasped"),
                ("transferred", transferred, "object_dropped"),
                ("handed_over", handed_over, "task_goal_miss"),
                ("placed", delivered, "task_goal_miss"),
            )
        )
    else:
        raise ValueError(f"Unknown benchmark skill: {skill}")

    rows = []
    for row in range(batch):
        stages: dict[str, bool] = {}
        failed_stage = None if planned[row] else "planning"
        reason = None if planned[row] else "planner_reported_failure"
        for name, passed, failure_reason in checks:
            if failed_stage is not None:
                break
            stages[name] = bool(passed[row])
            if not stages[name]:
                failed_stage, reason = name, failure_reason
        rows.append(
            {
                "env_index": row,
                "planning_success": planned[row],
                "success": failed_stage is None,
                "failure_stage": failed_stage,
                "failure_reason": reason,
                "stages": stages,
                "metrics": {key: value[row].item() for key, value in metrics.items()},
            }
        )
    return rows
