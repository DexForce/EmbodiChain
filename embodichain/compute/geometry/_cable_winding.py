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
"""Pure cable centerline, cylinder contact and finger clearance measurements."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

__all__: list[str] = []


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
    near = np.min(np.where(in_height, gaps, np.inf), axis=1) <= config["contact_band_m"]
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
