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

"""Bounded robot-relative scene proposals and raw commanded-joint motion costs."""

from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.tools.assemble._json_io import pose_matrix
from scripts.tools.grasp_assemble._config import layout_settings
from scripts.tools.grasp_assemble._path import _joint_cost_weights

__all__ = ["resolve_layout", "layout_record", "motion_metrics", "motion_rejection"]


def resolve_layout(config: dict, proposal: dict | None) -> dict:
    """Resolve a bounded proposal into world poses for the fixed-origin UR5.

    Args:
        config: Original job, including seed poses and workspace constraints.
        proposal: Absolute robot-root XY positions and yaw offsets from seed poses.

    Returns:
        Independent config with the proposed world poses. Seed heights are retained.
    """
    settings = layout_settings(config.get("layout"))
    resolved = deepcopy(config)
    if settings["mode"] == "fixed":
        if proposal is not None:
            raise ValueError("layout must be null in fixed mode")
        return resolved
    keys = {
        "base_xy",
        "assemble_xy",
        "base_yaw_offset_degrees",
        "assemble_yaw_offset_degrees",
    }
    if not isinstance(proposal, dict) or set(proposal) != keys:
        raise ValueError("Optimize mode requires both XY positions and yaw offsets")
    positions = []
    for role, key in (
        ("base", "T_world_base"),
        ("assemble", "T_world_assemble_initial"),
    ):
        xy, yaw = proposal[f"{role}_xy"], proposal[f"{role}_yaw_offset_degrees"]
        if (
            not isinstance(xy, list)
            or len(xy) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in xy)
        ):
            raise ValueError(f"{role}_xy must contain two finite numbers")
        if type(yaw) not in (int, float) or not math.isfinite(yaw):
            raise ValueError(f"{role} yaw must be finite")
        for value, bound in zip(xy, ("x_bounds", "y_bounds")):
            if not settings[bound][0] <= value <= settings[bound][1]:
                raise ValueError(f"{role}_xy is outside layout.{bound}")
        if (
            not settings["yaw_offset_bounds_degrees"][0]
            <= yaw
            <= settings["yaw_offset_bounds_degrees"][1]
        ):
            raise ValueError(f"{role} yaw is outside the allowed interval")
        radius = float(np.linalg.norm(xy))
        if (
            not settings[f"{role}_min_robot_distance"]
            <= radius
            <= settings["max_robot_distance"]
        ):
            raise ValueError(
                f"{role} origin at radius {radius:.3f} m is outside robot distance bounds"
            )
        transform = pose_matrix(config[key]).copy()
        transform[:2, 3] = xy
        transform[:3, :3] = (
            Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ transform[:3, :3]
        )
        resolved[key] = transform.tolist()
        positions.append(np.asarray(xy))
    if np.linalg.norm(positions[0] - positions[1]) < settings["min_object_distance"]:
        raise ValueError("Initial object origins are too close together")
    return resolved


def layout_record(config: dict, proposal: dict | None) -> dict:
    """Record the layout's explicit reference frames and selected proposal.

    Args:
        config: Resolved scene config.
        proposal: Original proposal, or null for fixed mode.

    Returns:
        Both object poses relative to the robot root, plus its world transform.
    """
    return {
        "proposal": deepcopy(proposal),
        "T_world_robot": np.eye(4).tolist(),
        "T_robot_base": deepcopy(config["T_world_base"]),
        "T_robot_assemble_initial": deepcopy(config["T_world_assemble_initial"]),
    }


def motion_metrics(
    positions: np.ndarray,
    joint_names: list[str],
    *,
    joint_cost_weights: list[float] | np.ndarray | None = None,
) -> dict:
    """Measure actual commanded arm travel without concealing full revolutions.

    Args:
        positions: Arm joint samples, shape (N, J), in radians.
        joint_names: Names in the same column order.
        joint_cost_weights: Nonnegative ranking penalties in joint order; omitted
            weights are uniform. Raw motion metrics remain unchanged.

    Returns:
        Per-joint motion, peak step, and a lower-is-better layout score in degrees.
    """
    q = np.asarray(positions, dtype=float)
    if (
        q.ndim != 2
        or q.shape[0] < 2
        or q.shape[1] == 0
        or q.shape[1] != len(joint_names)
        or not np.isfinite(q).all()
    ):
        raise ValueError("Expected finite arm positions of shape (N >= 2, J)")
    weights = _joint_cost_weights(joint_cost_weights, q.shape[1])
    steps = np.abs(np.diff(q, axis=0))
    travel = np.rad2deg(steps.sum(axis=0))
    ranges = np.rad2deg(np.ptp(q, axis=0))
    peak_step = float(np.rad2deg(steps.max()))
    weighted_travel = float(np.dot(weights, travel))
    weighted_range = float(np.max(weights * ranges))
    return {
        "joint_names": list(joint_names),
        "joint_travel_degrees": travel.tolist(),
        "joint_range_degrees": ranges.tolist(),
        "total_joint_travel_degrees": float(travel.sum()),
        "max_joint_travel_degrees": float(travel.max()),
        "max_joint_range_degrees": float(ranges.max()),
        "max_joint_step_degrees": peak_step,
        "joint_cost_weights": weights.tolist(),
        "weighted_total_joint_travel_degrees": weighted_travel,
        "weighted_max_joint_range_degrees": weighted_range,
        "score": weighted_travel + 2 * weighted_range,
        "score_definition": (
            "sum_j(weight_j * joint_travel_degrees_j) + "
            "2 * max_j(weight_j * joint_range_degrees_j); lower is better"
        ),
    }


def motion_rejection(metrics: dict, config: dict) -> str | None:
    """Return an excessive-motion rejection in optimize mode, if any.

    Args:
        metrics: Measured arm trajectory metrics.
        config: Job containing layout acceptance limits.

    Returns:
        Rejection reason or null. Fixed mode only reports these metrics.
    """
    settings = layout_settings(config.get("layout"))
    if settings["mode"] == "optimize":
        violations = [
            f"{key}={metrics[key]:.2f} exceeds {settings[key]:.2f}"
            for key in (
                "max_joint_step_degrees",
                "max_joint_range_degrees",
            )
            if metrics[key] > settings[key]
        ]
        if violations:
            return "Excessive arm motion: " + "; ".join(violations)
    return None
