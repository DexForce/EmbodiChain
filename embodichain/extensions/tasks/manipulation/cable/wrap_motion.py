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
"""Plan a nominal joint-space reference for the continuous cable wrap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import torch

__all__ = ["PHASE_END_S", "WrapMotion", "plan_wrap_motion"]

_JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8)) + (
    "fr3_finger_joint1",
    "fr3_finger_joint2",
)

PHASE_END_S = {
    "home": 1.0,
    "approach": 3.0,
    "descend": 4.0,
    "close": 5.0,
    "lift": 7.0,
    "approach_post": 10.0,
    "orbit": 28.0,
    "tighten": 32.0,
    "settle": 34.0,
    "hold_closed": 36.0,
}


@dataclass(frozen=True)
class WrapMotion:
    """Time samples including the home reference at t=0, before the first action."""

    positions: np.ndarray
    poses: np.ndarray
    labels: list[str]
    times: np.ndarray
    report: dict[str, Any]
    joint_names: tuple[str, ...]

    def phase_slice(self, phase: str) -> slice:
        """Select commands (start, stop] exactly once on the global control grid."""
        scale = self.report["duration_s"] / PHASE_END_S["hold_closed"]
        previous_end = 0
        for name, nominal_end in PHASE_END_S.items():
            end = int(
                np.searchsorted(self.times, nominal_end * scale + 1e-9, side="right")
            )
            if phase == name:
                return slice(max(1, previous_end), end)
            previous_end = end
        raise ValueError(f"Unknown wrap phase {phase!r}.")


def plan_wrap_motion(config: dict[str, Any], robot_path: Path) -> WrapMotion:
    """Solve and check the wrap reference from the calibrated pickup posture."""
    import pytorch_kinematics as pk

    arm_names = list(_JOINT_NAMES[:7])
    floor = float(config.get("floor_height_m", 0.002))
    fps = int(config.get("control_hz", 60))
    duration = float(config.get("duration_s", 36.0))
    if duration < 36.0:
        raise ValueError(
            "At least 36 seconds are required, including a two-second hold"
        )
    scale = duration / PHASE_END_S["hold_closed"]
    radius = float(config.get("orbit_radius_m", 0.12))
    wrap_degrees = float(config.get("orbit_degrees", 450.0))
    direction = int(config.get("orbit_direction", 1))
    if direction not in (-1, 1):
        raise ValueError("orbit_direction must be +1 (CCW) or -1 (CW)")
    theta_start = np.deg2rad(float(config.get("orbit_start_deg", -90.0)))
    theta_end = theta_start + direction * np.deg2rad(wrap_degrees)
    center = np.asarray(config["post_center_xy_m"], dtype=float)
    pickup_end = np.asarray(config["cable_waypoints_m"][-1], dtype=float)
    grasp_offset = np.asarray(
        config.get("grasp_offset_world_m", [-0.020, 0, 0]), dtype=float
    )
    z_start = float(config.get("orbit_start_z_m", floor + 0.055))
    z_end = float(config.get("orbit_end_z_m", floor + 0.075))
    lift_height = float(config.get("pickup_lift_m", 0.09))
    radial_pull = float(config.get("tighten_radial_m", 0.018))
    tangent_pull = float(config.get("tighten_tangent_m", 0.020))
    open_gap = float(config.get("open_half_gap_m", 0.04))
    closed_gap = float(config.get("grip_half_gap_m", config["cable_radius_m"] - 0.0005))
    if not 0 < closed_gap < open_gap <= 0.04:
        raise ValueError("Invalid Franka gripper half gaps")
    hand_r = Rotation.from_euler(
        "z", float(config.get("grasp_yaw_deg", 0.0)), degrees=True
    ).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    chain = pk.build_serial_chain_from_urdf(robot_path.read_bytes(), "fr3_hand_tcp").to(
        dtype=torch.float64, device="cpu"
    )
    if chain.get_joint_parameter_names() != arm_names:
        raise ValueError("Unexpected Franka arm joint order")
    limits = np.asarray(chain.get_joint_limits(), dtype=float).T
    lo, hi = limits[:, 0] + 1e-5, limits[:, 1] - 1e-5
    home = np.asarray(config["home_qpos"], dtype=float)

    def pose(p: np.ndarray, rotation: np.ndarray = hand_r) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = rotation, p
        return result

    def fk(q: np.ndarray) -> np.ndarray:
        result = (
            chain.forward_kinematics(torch.tensor(q, dtype=torch.float64))
            .get_matrix()[0]
            .numpy()
        )
        result[2, 3] += floor
        return result

    def ik(target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        q = seed.copy()
        for _ in range(45):
            current = fk(q)
            dp = target[:3, 3] - current[:3, 3]
            dr = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            if np.linalg.norm(dp) < 2e-7 and np.linalg.norm(dr) < 2e-6:
                return q
            jac = chain.jacobian(torch.tensor(q, dtype=torch.float64))[0].numpy()
            jac[3:] *= 0.15
            delta = jac.T @ np.linalg.solve(
                jac @ jac.T + np.eye(6) * 1e-8, np.r_[dp, 0.15 * dr]
            )
            q = np.clip(
                q + delta * min(1.0, 0.1 / max(np.max(np.abs(delta)), 1e-9)), lo, hi
            )

        def residual(candidate: np.ndarray) -> np.ndarray:
            current = fk(candidate)
            return np.r_[
                current[:3, 3] - target[:3, 3],
                0.15
                * Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
            ]

        solved = least_squares(
            residual,
            q,
            bounds=(lo, hi),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=160,
        )
        if np.linalg.norm(residual(solved.x)) > 1e-5:
            raise RuntimeError(
                f"Unreachable wrap TCP {target[:3, 3]}: {residual(solved.x)}"
            )
        return solved.x

    def smooth(u: float) -> float:
        u = float(np.clip(u, 0, 1))
        return u**3 * (10 - 15 * u + 6 * u * u)

    def end_at(theta: float, height: float) -> np.ndarray:
        return np.r_[center + radius * np.array([np.cos(theta), np.sin(theta)]), height]

    home_pose = fk(home)
    pickup = pose(pickup_end + grasp_offset)
    above = pickup.copy()
    above[2, 3] += lift_height
    orbit_start = pose(end_at(theta_start, z_start) + grasp_offset)
    orbit_end = pose(end_at(theta_end, z_end) + grasp_offset)
    radial = np.array([np.cos(theta_end), np.sin(theta_end), 0.0])
    tangent = direction * np.array([-np.sin(theta_end), np.cos(theta_end), 0.0])
    tightened = orbit_end.copy()
    tightened[:3, 3] += radial_pull * radial + tangent_pull * tangent
    # The nominal pickup only seeds the IK branch for the authored wrap.
    # The Task Program commands the live cable pickup instead.
    pickup_keys = [
        (0.0, home_pose, open_gap, "HOME"),
        (PHASE_END_S["home"], home_pose, open_gap, "HOME"),
        (PHASE_END_S["approach"], above, open_gap, "APPROACH FLOOR END"),
        (PHASE_END_S["descend"], pickup, open_gap, "LOWER TO ROPE"),
        (PHASE_END_S["close"], pickup, closed_gap, "GRASP ROPE"),
        (PHASE_END_S["lift"], above, closed_gap, "LIFT AND ORGANIZE SLACK"),
        (PHASE_END_S["approach_post"], orbit_start, closed_gap, "TANGENTIAL APPROACH"),
    ]
    times = np.arange(round(duration * fps) + 1) / fps
    targets, labels, gaps = [], [], []
    orbit_start_clock = PHASE_END_S["approach_post"]
    orbit_end_clock = PHASE_END_S["orbit"]
    for t in times:
        clock = t / scale
        if clock <= PHASE_END_S["approach_post"]:
            index = min(
                max(
                    int(
                        np.searchsorted(
                            [k[0] for k in pickup_keys], clock, side="right"
                        )
                    )
                    - 1,
                    0,
                ),
                len(pickup_keys) - 2,
            )
            a, b = pickup_keys[index : index + 2]
            u = smooth((clock - a[0]) / (b[0] - a[0]))
            target = pose(
                (1 - u) * a[1][:3, 3] + u * b[1][:3, 3],
                Slerp([0, 1], Rotation.from_matrix([a[1][:3, :3], b[1][:3, :3]]))(
                    [u]
                ).as_matrix()[0],
            )
            gap, label = (1 - u) * a[2] + u * b[2], b[3]
        elif clock <= PHASE_END_S["orbit"]:
            u = smooth(
                (clock - orbit_start_clock) / (orbit_end_clock - orbit_start_clock)
            )
            theta = theta_start + (theta_end - theta_start) * u
            target = pose(end_at(theta, (1 - u) * z_start + u * z_end) + grasp_offset)
            gap, label = closed_gap, "WRAP ROPE AROUND PILLAR"
        elif clock <= PHASE_END_S["tighten"]:
            u = smooth(
                (clock - orbit_end_clock) / (PHASE_END_S["tighten"] - orbit_end_clock)
            )
            target = pose((1 - u) * orbit_end[:3, 3] + u * tightened[:3, 3])
            gap, label = closed_gap, "GENTLE TANGENTIAL TIGHTEN"
        elif clock < PHASE_END_S["settle"]:
            target, gap, label = tightened.copy(), closed_gap, "SETTLE WRAP"
        else:
            target, gap, label = tightened.copy(), closed_gap, "HOLD COMPLETED WRAP"
        targets.append(target)
        gaps.append(gap)
        labels.append(label)
    targets = np.asarray(targets)
    qpath = np.zeros((len(times), 9), dtype=np.float32)
    q = home.copy()
    for i, target in enumerate(targets):
        q = ik(target, q)
        qpath[i, :7], qpath[i, 7:] = q, gaps[i]
    actual = (
        chain.forward_kinematics(torch.tensor(qpath[:, :7], dtype=torch.float64))
        .get_matrix()
        .numpy()
    )
    actual[:, 2, 3] += floor
    pe = np.linalg.norm(actual[:, :3, 3] - targets[:, :3, 3], axis=1)
    re = Rotation.from_matrix(
        targets[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()
    vmax = np.max(np.abs(np.diff(qpath[:, :7], axis=0) * fps), axis=0)
    root = ET.parse(robot_path).getroot()
    velocity_limits = np.array(
        [
            float(root.find(f"joint[@name='{name}']/limit").get("velocity"))
            for name in arm_names
        ]
    )
    joint_limits_ok = bool(
        np.all(qpath[:, :7] >= lo - 1e-6) and np.all(qpath[:, :7] <= hi + 1e-6)
    )
    passed = bool(
        pe.max() < 1e-5
        and re.max() < 1e-4
        and np.all(vmax <= velocity_limits)
        and joint_limits_ok
    )
    report = {
        "passed": passed,
        "ik_position_max_m": float(pe.max()),
        "ik_rotation_max_rad": float(re.max()),
        "joint_velocity_max_rad_s": vmax.tolist(),
        "joint_velocity_limits_rad_s": velocity_limits.tolist(),
        "joint_limits_ok": joint_limits_ok,
        "minimum_joint_limit_margin_rad": float(
            np.min(np.minimum(qpath[:, :7] - lo, hi - qpath[:, :7]))
        ),
        "duration_s": duration,
        "hold_start_s": PHASE_END_S["settle"] * scale,
        "hold_duration_s": (PHASE_END_S["hold_closed"] - PHASE_END_S["settle"]) * scale,
        "orbit_start_s": orbit_start_clock * scale,
        "orbit_end_s": orbit_end_clock * scale,
        "orbit_degrees": wrap_degrees,
        "orbit_direction": direction,
        "orbit_radius_m": radius,
        "orbit_start_z_m": z_start,
        "orbit_end_z_m": z_end,
        "grasp_offset_world_m": grasp_offset.tolist(),
        "grasp_yaw_deg": float(config.get("grasp_yaw_deg", 0.0)),
        "grip_half_gap_m": closed_gap,
        "final_end_target_m": (tightened[:3, 3] - grasp_offset).tolist(),
        "physical_success_requires_actual_centerline_and_contacts": True,
    }
    if not passed:
        raise RuntimeError(f"Wrap motion validation failed: {report}")
    return WrapMotion(
        qpath,
        targets,
        labels,
        times,
        report,
        _JOINT_NAMES,
    )
