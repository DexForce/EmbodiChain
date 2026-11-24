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
"""Measure the original Franka RJ45 physical acceptance on either simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from embodichain.lab.evaluation._cable_checks import RJ45_CHECKS

__all__ = ["RJ45_CHECKS", "evaluate_rj45", "measure_rj45_frame"]

_PLUG_ROTATION = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_RETURN_ROTATION = (
    _PLUG_ROTATION @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
)
_PLUG_ATTACHMENT = np.array([0, 0, -0.031, 1])


def __getattr__(name: str) -> tuple[str, ...]:
    """Keep the check-name export available to package consumers."""
    if name == "RJ45_CHECKS":
        # Isaac Lab loads this file directly for the measurement functions.
        from embodichain.lab.evaluation._cable_checks import RJ45_CHECKS

        return RJ45_CHECKS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def measure_rj45_frame(
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
) -> dict[str, Any]:
    """Measure a control frame with geometry common to both Newton scenes.

    Poses are world transforms. Cable states contain segment centers and
    xyzw rotations, with each capsule aligned with its local positive z axis.
    Contact inputs describe the four substep samples in the control interval.
    A missing contact measurement is represented by None.
    """
    plug_tip = plug_pose[:3, 3]
    socket_delta = _PLUG_ROTATION.T @ (plug_tip - config["socket_mouth_m"])
    cable_end = cable_states[-1]
    end_axis = Rotation.from_quat(cable_end[3:7]).as_matrix()[:, 2]
    exit_point = (plug_pose @ _PLUG_ATTACHMENT)[:3]
    actual_end = cable_end[:3] + end_axis * cable_lengths[-1] / 2
    return {
        "time_s": frame / config["control_hz"],
        "tip_m": plug_tip.tolist(),
        "plug_pose": plug_pose.tolist(),
        "actual_tcp_pose": tcp_pose.tolist(),
        "depth_m": float(socket_delta[2]),
        "lateral_m": float(np.linalg.norm(socket_delta[:2])),
        "orientation_error_deg": float(
            np.rad2deg(
                Rotation.from_matrix(_PLUG_ROTATION.T @ plug_pose[:3, :3]).magnitude()
            )
        ),
        "latch_angle_deg": float(np.rad2deg(latch_angle_rad)),
        "finger_latch_contact": finger_latch_contact,
        "latch_socket_contact": latch_socket_contact,
        "latch_limit_violation_rad": latch_limit_violation_rad,
        "return_error_m": float(np.linalg.norm(plug_tip - config["return_tip_m"])),
        "return_orientation_error_deg": float(
            np.rad2deg(
                Rotation.from_matrix(_RETURN_ROTATION.T @ plug_pose[:3, :3]).magnitude()
            )
        ),
        "home_error_rad": float(np.max(np.abs(robot_qpos[:7] - config["home_qpos"]))),
        "finger_gap_m": float(robot_qpos[7] + robot_qpos[8]),
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


def evaluate_rj45(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the same 17 gates and windows as the published 40-second run."""
    limits = config["validation"]
    retained = [r for r in rows if 18 <= r["time_s"] <= 19]
    final = [r for r in rows if r["time_s"] >= 39]
    grab = [r for r in rows if 6 <= r["time_s"] <= 9]
    pressed = [r for r in rows if 24 <= r["time_s"] <= 25]
    insertion = [r for r in rows if 5 <= r["time_s"] <= 14]
    penetration = [r["penetration_m"] for r in rows if r["penetration_m"] is not None]
    dropped = [r["dropped_contacts"] for r in rows if r["dropped_contacts"] is not None]
    limit_violations = [
        r["latch_limit_violation_rad"]
        for r in rows
        if r["latch_limit_violation_rad"] is not None
    ]
    checks = {
        "starts_on_floor": bool(rows and 0 < rows[0]["tip_m"][2] < 0.007),
        "insertion_without_finger_latch_press": bool(
            insertion and all(r["finger_latch_contact"] is False for r in insertion)
        ),
        "vertical_withdrawal_grasp": bool(
            pressed
            and all(
                abs(np.asarray(r["actual_tcp_pose"])[2, 1]) > np.cos(np.deg2rad(10))
                for r in pressed
            )
        ),
        "complete": bool(rows and rows[-1]["time_s"] >= config["duration_s"]),
        "pickup_and_lift": bool(
            grab
            and min((r["tip_m"][2] for r in grab)) > config["pickup_tip_m"][2] + 0.045
        ),
        "seated_while_robot_home": bool(
            retained
            and all(
                (
                    limits["seat_min_m"] < r["depth_m"] < limits["seat_max_m"]
                    and r["lateral_m"] < limits["lateral_max_m"]
                    and (r["orientation_error_deg"] < limits["orientation_max_deg"])
                    and (r["home_error_rad"] < 0.0001)
                    for r in retained
                )
            )
        ),
        "latch_locked_while_robot_home": bool(
            retained
            and any(r["latch_socket_contact"] for r in retained)
            and all(r["finger_latch_contact"] is False for r in retained)
        ),
        "physical_latch_press_before_withdrawal": any(
            r["finger_latch_contact"]
            and r["latch_angle_deg"] <= -limits["latch_pressed_min_deg"]
            and r["depth_m"] >= limits["seat_min_m"]
            for r in pressed
        ),
        "latch_limits": bool(
            rows
            and len(limit_violations) == len(rows)
            and max(limit_violations)
            < np.deg2rad(limits["latch_limit_violation_max_deg"])
        ),
        "latch_returns_after_release": bool(
            final
            and all(
                abs(r["latch_angle_deg"]) < limits["latch_return_max_deg"]
                for r in final
            )
        ),
        "unplugged": any(
            (27 <= r["time_s"] <= 28 and r["depth_m"] < -0.02 for r in rows)
        ),
        "placed_on_floor": bool(
            final
            and all(
                (
                    r["return_error_m"] < limits["return_position_max_m"]
                    and r["return_orientation_error_deg"]
                    < limits["return_orientation_max_deg"]
                    for r in final
                )
            )
        ),
        "final_home": bool(
            final
            and all(
                (
                    r["home_error_rad"] < 0.0001 and r["finger_gap_m"] > 0.07
                    for r in final
                )
            )
        ),
        "fixed_end": bool(rows and max((r["anchor_error_m"] for r in rows)) < 1e-05),
        "cable_attachment": bool(
            rows
            and max((r["attachment_error_m"] for r in rows))
            < limits["attachment_max_m"]
        ),
        "contacts": bool(
            rows
            and len(penetration) == len(dropped) == len(rows)
            and max(penetration) < limits["contact_penetration_max_m"]
            and sum(dropped) == 0
        ),
        "finite_state": bool(rows and all((r["finite"] for r in rows))),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "quality_status": (
            "floor_cycle_verified" if all(checks.values()) else "demo_candidate"
        ),
        "diagnostic_rate_hz": config["control_hz"] * 4,
        "peak_penetration_m": max(penetration, default=None),
        "peak_attachment_error_m": max(
            (r["attachment_error_m"] for r in rows), default=0
        ),
        "peak_latch_limit_violation_deg": (
            float(np.rad2deg(max(limit_violations))) if limit_violations else None
        ),
        "pressed_latch_min_deg": min(
            (r["latch_angle_deg"] for r in pressed), default=None
        ),
        "press_contact_frames": sum(r["finger_latch_contact"] is True for r in pressed),
        "final_return_error_m": final[-1]["return_error_m"] if final else None,
        "retained_depth_range_m": (
            [
                min((r["depth_m"] for r in retained)),
                max((r["depth_m"] for r in retained)),
            ]
            if retained
            else None
        ),
    }
