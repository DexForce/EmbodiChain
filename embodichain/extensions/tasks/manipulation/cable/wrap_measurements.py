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
"""Physical measurements and episode acceptance for wrapping a cable around a post."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.lab.evaluation._cable_checks import WRAP_CHECKS as WRAP_CHECKS

__all__ = ["evaluate_wrap"]


def _capsule_pad_gap(
    state: np.ndarray,
    lengths: np.ndarray,
    finger_pose: np.ndarray,
    config: dict[str, Any],
) -> float:
    """Exact capsule-axis / finger-box distance, less the physical rope radius."""
    axes = Rotation.from_quat(state[:, 3:7]).as_matrix()[:, :, 2]
    starts = state[:, :3] - axes * lengths[:, None] / 2
    ends = state[:, :3] + axes * lengths[:, None] / 2
    half = np.asarray(config["finger_pad_size_m"]) / 2
    a = (starts - finger_pose[:3, 3]) @ finger_pose[:3, :3] - config[
        "finger_pad_center_m"
    ]
    b = (ends - finger_pose[:3, 3]) @ finger_pose[:3, :3] - config[
        "finger_pad_center_m"
    ]
    best = np.inf
    for origin, delta in zip(a, b - a):
        breaks = [0.0, 1.0]
        for k in range(3):
            if abs(delta[k]) > 1e-12:
                breaks.extend(
                    float(v)
                    for v in (
                        (-half[k] - origin[k]) / delta[k],
                        (half[k] - origin[k]) / delta[k],
                    )
                    if 0 < v < 1
                )
        breaks = sorted(breaks)
        for left, right in zip(breaks[:-1], breaks[1:]):
            midpoint = origin + delta * ((left + right) / 2)
            active = np.abs(midpoint) > half
            offset = origin[active] - np.sign(midpoint[active]) * half[active]
            velocity = delta[active]
            denominator = float(velocity @ velocity)
            t = (
                float(np.clip(-(offset @ velocity) / denominator, left, right))
                if denominator > 1e-20
                else left
            )
            distance = np.linalg.norm(np.maximum(np.abs(origin + t * delta) - half, 0))
            best = min(best, float(distance))
    return best - config["cable_radius_m"]


def _rope_geometry(
    state: np.ndarray, lengths: np.ndarray, config: dict[str, Any]
) -> dict[str, Any]:
    """Measure actual capsule centerline geometry, independently of commanded orbit."""
    axis = Rotation.from_quat(state[:, 3:7]).as_matrix()[:, :, 2]
    starts = state[:, :3] - axis * lengths[:, None] / 2
    ends = state[:, :3] + axis * lengths[:, None] / 2
    nodes = np.vstack([starts[:1], (ends[:-1] + starts[1:]) / 2, ends[-1:]])
    center = np.asarray(config["post_center_xy_m"])
    theta = np.unwrap(np.arctan2(nodes[:, 1] - center[1], nodes[:, 0] - center[0]))
    samples = (
        starts[:, None]
        + (ends - starts)[:, None] * np.linspace(0, 1, 11)[None, :, None]
    )
    radial = np.linalg.norm(samples[:, :, :2] - center, axis=-1)
    gaps = radial - config["post_radius_m"] - config["cable_radius_m"]
    in_height = (
        samples[:, :, 2] > config["floor_height_m"] - config["cable_radius_m"]
    ) & (
        samples[:, :, 2]
        < config["floor_height_m"] + config["post_height_m"] + config["cable_radius_m"]
    )
    valid_gaps = gaps[in_height]
    near = (
        np.min(np.where(in_height, gaps, np.inf), axis=1)
        <= config["validation"]["contact_band_m"]
    )
    # Union on the circle: revisiting one sector must not inflate coverage.
    intervals = []
    for low, high, include in zip(
        np.minimum(theta[:-1], theta[1:]), np.maximum(theta[:-1], theta[1:]), near
    ):
        if not include:
            continue
        width = high - low
        start = low % (2 * np.pi)
        end = start + width
        intervals.append((start, min(end, 2 * np.pi)))
        if end > 2 * np.pi:
            intervals.append((0.0, end - 2 * np.pi))
    coverage, right = 0.0, 0.0
    for low, high in sorted(intervals):
        coverage += max(0.0, high - max(low, right))
        right = max(right, high)
    joint_gap = np.linalg.norm(ends[:-1] - starts[1:], axis=1)
    anchor_direction = np.r_[center - starts[0, :2], 0.0]
    anchor_direction /= np.linalg.norm(anchor_direction)
    anchor_angle = np.rad2deg(np.arccos(np.clip(axis[0] @ anchor_direction, -1.0, 1.0)))
    anchor_bends = np.rad2deg(
        np.arccos(np.clip(np.sum(axis[:4] * axis[1:5], axis=-1), -1.0, 1.0))
    )
    return {
        "nodes": nodes,
        "anchor_direction_error_deg": float(anchor_angle),
        "anchor_exit_bend_max_deg": float(anchor_bends.max()),
        "winding_deg": float(np.rad2deg(theta[-1] - theta[0])),
        "contact_arc_deg": float(np.rad2deg(coverage)),
        "post_contact_segments": int(near.sum()),
        "post_penetration_m": (
            float(max(0.0, -valid_gaps.min())) if valid_gaps.size else 0.0
        ),
        "joint_gap_max_m": float(joint_gap.max()),
        "centerline_length_m": float(
            np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()
        ),
        "free_end_m": ends[-1].tolist(),
        "rope_min_z_m": float(samples[:, :, 2].min() - config["cable_radius_m"]),
        "rope_max_z_m": float(samples[:, :, 2].max() + config["cable_radius_m"]),
    }


def evaluate_wrap(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["validation"]
    hold = [r for r in rows if r["time_s"] >= config["duration_s"] - 2 - 1e-8]
    grasped = [r for r in rows if r["time_s"] >= config["grasp_complete_s"]]
    lifted = [
        r
        for r in rows
        if config["lift_complete_s"] <= r["time_s"] <= config["lift_complete_s"] + 0.5
    ]
    checks = {
        "completed": bool(
            rows
            and abs(rows[-1]["time_s"] - config["duration_s"]) < 1e-8
            and len(hold) >= round(2 * config["control_hz"]) + 1
        ),
        "finite": bool(rows and all(r["finite"] for r in rows)),
        "joint_limits": bool(
            rows
            and max(r["arm_joint_limit_violation_rad"] for r in rows) < 1e-4
            and max(r["finger_joint_limit_violation_m"] for r in rows) < 1e-4
        ),
        "direct_floor_pickup": bool(
            lifted
            and rows[0]["free_end_m"][2]
            <= config["floor_height_m"] + 2 * config["cable_radius_m"]
            and min(r["free_end_m"][2] for r in lifted)
            > config["floor_height_m"] + 0.035
        ),
        "grasp_retained": bool(
            grasped
            and all(r["grasp_confirmed"] for r in grasped)
            and max(r["tail_tcp_distance_m"] for r in grasped)
            <= thresholds["tail_tcp_max_m"]
        ),
        "material_grasp_slip": bool(
            grasped
            and max(r["material_slip_m"] for r in grasped)
            <= thresholds.get("grasp_slip_max_m", 0.008)
        ),
        "bilateral_finger_contact": bool(
            grasped and np.mean([r["bilateral_rope_contact"] for r in grasped]) >= 0.95
        ),
        "actual_winding": bool(
            hold
            and min(abs(r["winding_deg"]) for r in hold)
            >= thresholds["winding_min_deg"]
        ),
        "post_contact_arc": bool(
            hold
            and min(r["contact_arc_deg"] for r in hold)
            >= thresholds["contact_arc_min_deg"]
        ),
        "hold_stable": bool(
            hold
            and np.ptp([r["winding_deg"] for r in hold])
            <= thresholds["hold_winding_drift_max_deg"]
            and np.max(
                np.linalg.norm(
                    np.asarray([r["free_end_m"] for r in hold]) - hold[0]["free_end_m"],
                    axis=1,
                )
            )
            <= thresholds["hold_end_drift_max_m"]
        ),
        "rope_below_post_top": bool(
            rows
            and max(r["rope_max_z_m"] for r in rows)
            < config["floor_height_m"] + config["post_height_m"]
        ),
        "post_nonpenetration": bool(
            rows
            and max(r["post_penetration_m"] for r in rows)
            <= thresholds["post_penetration_max_m"]
        ),
        "robot_post_clearance": bool(
            rows
            and max(r["robot_post_penetration_m"] for r in rows)
            <= thresholds["robot_post_penetration_max_m"]
        ),
        "anchor_points_toward_post": bool(
            rows and max(r["anchor_direction_error_deg"] for r in rows) < 0.1
        ),
        "smooth_anchor_exit": bool(
            hold
            and max(r["anchor_exit_bend_max_deg"] for r in hold)
            <= thresholds.get("anchor_exit_bend_max_deg", 20.0)
        ),
        "anchor_fixed": bool(rows and max(r["anchor_error_m"] for r in rows) < 1e-5),
        "length_preserved": bool(
            rows
            and max(r["joint_gap_max_m"] for r in rows) <= thresholds["joint_gap_max_m"]
        ),
        "contacts_not_dropped": bool(
            rows and max(r["dropped_contacts"] for r in rows) == 0
        ),
        "floor_nonpenetration": bool(
            rows
            and min(r["rope_min_z_m"] for r in rows)
            >= config["floor_height_m"] - thresholds["floor_penetration_max_m"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": {
            "hold_anchor_exit_bend_max_deg": max(
                [r["anchor_exit_bend_max_deg"] for r in hold], default=None
            ),
            "material_grasp_slip_max_m": max(
                [r["material_slip_m"] for r in grasped], default=None
            ),
            "final_winding_deg": rows[-1]["winding_deg"] if rows else None,
            "hold_winding_min_deg": min(
                [abs(r["winding_deg"]) for r in hold], default=None
            ),
            "hold_contact_arc_min_deg": min(
                [r["contact_arc_deg"] for r in hold], default=None
            ),
            "tail_tcp_max_m": max(
                [r["tail_tcp_distance_m"] for r in grasped], default=None
            ),
            "post_penetration_max_m": max(
                [r["post_penetration_m"] for r in rows], default=None
            ),
            "robot_post_penetration_max_m": max(
                [r["robot_post_penetration_m"] for r in rows], default=None
            ),
            "hold_winding_drift_deg": (
                float(np.ptp([r["winding_deg"] for r in hold])) if hold else None
            ),
            "hold_end_drift_m": (
                float(
                    np.max(
                        np.linalg.norm(
                            np.asarray([r["free_end_m"] for r in hold])
                            - hold[0]["free_end_m"],
                            axis=1,
                        )
                    )
                )
                if hold
                else None
            ),
            "joint_gap_max_m": max([r["joint_gap_max_m"] for r in rows], default=None),
            "bilateral_contact_fraction": (
                float(np.mean([r["bilateral_rope_contact"] for r in grasped]))
                if grasped
                else None
            ),
        },
        "thresholds": thresholds,
    }
