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
"""Engine-independent observation frames for connector scenes."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

__all__: list[str] = []


def _connector_frame(
    config: dict[str, Any],
    *,
    frame: int,
    plug_pose: np.ndarray,
    tcp_pose: np.ndarray,
    robot_qpos: np.ndarray,
    latch_angle_rad: float,
    cable_states: np.ndarray,
    cable_lengths: np.ndarray,
    anchor_baseline: np.ndarray,
    finger_latch_contact: bool | None,
    latch_socket_contact: bool | None,
    latch_limit_violation_rad: float | None,
    penetration_m: float | None,
    dropped_contacts: int | None,
    press_samples: list[dict[str, float]],
) -> dict[str, Any]:
    """Measure a control frame with geometry common to both Newton scenes.

    Poses are world transforms. Cable states contain segment centers and
    xyzw rotations, with each capsule aligned with its local positive z axis.
    Contact inputs describe the substep samples in the control interval.
    A missing contact measurement is represented by None.
    """
    insertion_rotation = np.asarray(config["insertion_rotation"])
    plug_tip = plug_pose[:3, 3]
    socket_delta = insertion_rotation.T @ (plug_tip - config["socket_mouth_m"])
    cable_end = cable_states[-1]
    end_axis = Rotation.from_quat(cable_end[3:7]).as_matrix()[:, 2]
    exit_point = (plug_pose @ np.r_[config["attachment_point_m"], 1.0])[:3]
    actual_end = cable_end[:3] + end_axis * cable_lengths[-1] / 2
    return {
        "frame": frame,
        "tip_z_m": float(plug_tip[2]),
        "tool_axis_z_abs": float(abs(tcp_pose[2, config["vertical_tool_axis"]])),
        "latch_angle_abs_deg": float(abs(np.rad2deg(latch_angle_rad))),
        "press_samples": press_samples,
        "time_s": frame / config["control_hz"],
        "tip_m": plug_tip.tolist(),
        "plug_pose": plug_pose.tolist(),
        "actual_tcp_pose": tcp_pose.tolist(),
        "depth_m": float(socket_delta[2]),
        "lateral_m": float(np.linalg.norm(socket_delta[:2])),
        "orientation_error_deg": float(
            np.rad2deg(
                Rotation.from_matrix(
                    insertion_rotation.T @ plug_pose[:3, :3]
                ).magnitude()
            )
        ),
        "latch_angle_deg": float(np.rad2deg(latch_angle_rad)),
        "finger_latch_contact": finger_latch_contact,
        "latch_socket_contact": latch_socket_contact,
        "latch_limit_violation_rad": latch_limit_violation_rad,
        "return_error_m": float(np.linalg.norm(plug_tip - config["return_tip_m"])),
        "return_orientation_error_deg": float(
            np.rad2deg(
                Rotation.from_matrix(
                    np.asarray(config["return_rotation"]).T @ plug_pose[:3, :3]
                ).magnitude()
            )
        ),
        "home_error_rad": float(
            np.max(np.abs(robot_qpos[: len(config["home_qpos"])] - config["home_qpos"]))
        ),
        "finger_gap_m": float(robot_qpos[len(config["home_qpos"]) :].sum()),
        "anchor_error_m": float(np.linalg.norm(cable_states[0, :3] - anchor_baseline)),
        "attachment_error_m": float(np.linalg.norm(actual_end - exit_point)),
        "penetration_m": penetration_m,
        "dropped_contacts": dropped_contacts,
        "finite": bool(
            np.isfinite(plug_pose).all()
            and np.isfinite(tcp_pose).all()
            and np.isfinite(cable_states).all()
            and np.isfinite(robot_qpos).all()
            and np.isfinite(latch_angle_rad)
        ),
    }
