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
"""Calibrated nominal motion and measured-pose feedback for the RJ45 cycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
import torch

__all__ = ["Rj45Feedback", "Rj45Motion", "plan_rj45_motion"]

_PLUG_R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
_HAND_IN_PLUG = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
_WITHDRAW_IN_PLUG = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
_ARM_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
_JOINT_NAMES = _ARM_NAMES + [
    "fr3_finger_joint1",
    "fr3_finger_joint2",
]
_CARRY_CAPTURE_START_S, _CARRY_CAPTURE_END_S = 9, 10
_REGRASP_START_S, _REGRASP_CAPTURE_S = 19, 24.8
_PLACEMENT_CAPTURE_START_S, _PLACEMENT_CAPTURE_END_S = 31, 33


def _feedback_observation_steps(hz: int) -> tuple[int, ...]:
    """Return command arrivals that require observing the preceding control step."""
    return (
        *range(_CARRY_CAPTURE_START_S * hz, _CARRY_CAPTURE_END_S * hz + 1),
        _REGRASP_START_S * hz + 1,
        round(_REGRASP_CAPTURE_S * hz),
        *range(_PLACEMENT_CAPTURE_START_S * hz, _PLACEMENT_CAPTURE_END_S * hz + 1),
    )


def _pose(p: Any, r: np.ndarray = _PLUG_R) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = (r, p)
    return t


def _return_rotation() -> np.ndarray:
    """Rest the plug on its side so horizontal fingers clear the floor."""
    return _PLUG_R @ Rotation.from_euler("z", 90, degrees=True).as_matrix()


@dataclass(frozen=True)
class Rj45Motion:
    """Calibrated positions and tool poses, including the frame-zero reference."""

    positions: np.ndarray
    poses: np.ndarray
    labels: tuple[str, ...]
    times: np.ndarray


class _Rj45Kinematics:
    def __init__(self, robot_path: Path, floor_height_m: float) -> None:
        import pytorch_kinematics as pk

        self.chain = pk.build_serial_chain_from_urdf(
            robot_path.read_bytes(), "fr3_hand_tcp"
        ).to(dtype=torch.float64, device="cpu")
        if self.chain.get_joint_parameter_names() != _ARM_NAMES:
            raise ValueError("Franka kinematic joint order differs from the trajectory")
        limits = np.array(self.chain.get_joint_limits()).T
        self.lo = limits[:, 0] + 1e-5
        self.hi = limits[:, 1] - 1e-5
        self.floor_height_m = floor_height_m

    def fk(self, q: np.ndarray) -> np.ndarray:
        pose = (
            self.chain.forward_kinematics(torch.as_tensor(q, dtype=torch.float64))
            .get_matrix()[0]
            .numpy()
        )
        pose[2, 3] += self.floor_height_m
        return pose

    def _delta(self, q: np.ndarray, err: np.ndarray) -> np.ndarray:
        jacobian = self.chain.jacobian(torch.as_tensor(q, dtype=torch.float64))[
            0
        ].numpy()
        jacobian[3:] *= 0.15
        return jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + np.eye(6) * 1e-8, err
        )

    def solve_nominal(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        q = seed.copy()
        for _ in range(45):
            current = self.fk(q)
            dp = target[:3, 3] - current[:3, 3]
            dr = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            if np.linalg.norm(dp) < 2e-7 and np.linalg.norm(dr) < 2e-6:
                return q
            dq = self._delta(q, np.r_[dp, 0.15 * dr])
            scale = min(1.0, 0.1 / max(np.max(np.abs(dq)), 1e-9))
            q = np.clip(q + dq * scale, self.lo, self.hi)

        def residual(candidate: np.ndarray) -> np.ndarray:
            current = self.fk(candidate)
            return np.r_[
                current[:3, 3] - target[:3, 3],
                0.15
                * Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
            ]

        solved = least_squares(
            residual,
            q,
            bounds=(self.lo, self.hi),
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=120,
        )
        if np.linalg.norm(residual(solved.x)) > 1e-5:
            raise RuntimeError(
                f"Unreachable TCP target {target[:3, 3]}: residual {residual(solved.x)}"
            )
        return solved.x

    def solve_feedback(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        q = seed.astype(float)
        for _ in range(20):
            current = self.fk(q)
            error = np.r_[
                target[:3, 3] - current[:3, 3],
                0.15
                * Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec(),
            ]
            if np.linalg.norm(error) < 1e-7:
                break
            q = np.clip(q + self._delta(q, error), self.lo, self.hi)
        return q.astype(np.float32)


def plan_rj45_motion(config: dict[str, Any], robot_path: Path) -> Rj45Motion:
    """Build the calibrated nominal path and verify TCP tracking and joint limits.

    Args:
        config: Physical RJ45 scenario settings.
        robot_path: The exact robot URDF used by the scene.

    Returns:
        Nominal full-joint commands and tool poses on the control grid.
    """
    kinematics = _Rj45Kinematics(robot_path, config["floor_height_m"])
    fk = kinematics.fk
    home = np.asarray(config["home_qpos"], float)
    home_pose = fk(home)
    grasp = _pose(config["grasp_offset_m"], _HAND_IN_PLUG)
    vertical = _pose(config["withdraw_grasp_offset_m"], _WITHDRAW_IN_PLUG)
    pickup = _pose(config["pickup_tip_m"]) @ grasp
    seat_plug = _pose(
        np.asarray(config["socket_mouth_m"])
        + _PLUG_R @ np.array([0, 0, config["seat_depth_m"]])
    )
    seated = seat_plug @ grasp
    withdraw = seat_plug @ vertical
    return_plug = _pose(config["return_tip_m"], _return_rotation())
    place = return_plug @ vertical
    lift = np.array([0, 0, config["carry_height_m"]])

    def offset(pose: np.ndarray, delta: Any) -> np.ndarray:
        moved = pose.copy()
        moved[:3, 3] += delta
        return moved

    pre = offset(seated, [-0.0605, 0, 0])
    above_pick = offset(pickup, lift)
    above_pre = offset(pre, lift)
    above_seat = offset(seated, lift)
    withdraw_pre = offset(withdraw, [-0.0605, 0, 0])
    withdraw_approach = offset(withdraw, [0, 0.10, 0.06])
    withdraw_clear = offset(withdraw_pre, lift)
    above_place = offset(place, lift)
    place_retreat = offset(place, [-0.10, 0, 0.08])
    open_gap, closed = (0.04, config["grip_half_gap_m"])
    vertical_closed = config["withdraw_half_gap_m"]
    keys = [
        (0, home_pose, open_gap, "HOME", False),
        (1, home_pose, open_gap, "HOME", False),
        (3, above_pick, open_gap, "APPROACH FLOOR PLUG", False),
        (4, pickup, open_gap, "LOWER TO FLOOR", False),
        (5, pickup, closed, "GRASP BODY", False),
        (6, above_pick, closed, "LIFT FROM FLOOR", False),
        (9, above_pre, closed, "CARRY TO SOCKET", False),
        (10, pre, closed, "ALIGN", False),
        (12, seated, closed, "INSERT", False),
        (13, seated, closed, "HOLD INSERTION", False),
        (14, seated, open_gap, "RELEASE PLUG", False),
        (15, above_seat, open_gap, "RETREAT", False),
        (17.5, home_pose, open_gap, "RETURN HOME", True),
        (18, home_pose, open_gap, "SETTLE AT HOME", True),
        (19, home_pose, open_gap, "HOME / PLUG SEATED", True),
        (22, withdraw_approach, open_gap, "APPROACH VERTICAL GRASP", False),
        (23, withdraw, open_gap, "ALIGN ABOVE AND BELOW LATCH", False),
        (24, withdraw, vertical_closed, "CLOSE FINGERS ON LATCH", False),
        (25, withdraw, vertical_closed, "HOLD VERTICAL GRASP", False),
        (27, withdraw_pre, vertical_closed, "UNPLUG", False),
        (28, withdraw_clear, vertical_closed, "LIFT CLEAR", False),
        (31, above_place, vertical_closed, "TURN AND RETURN PLUG", False),
        (32, place, vertical_closed, "PLACE ON FLOOR", False),
        (33, place, vertical_closed, "SETTLE ON FLOOR", False),
        (34, place, open_gap, "OPEN GRIPPER", False),
        (35, place_retreat, open_gap, "RETREAT FROM FLOOR", False),
        (38, home_pose, open_gap, "FINAL HOME", True),
        (40, home_pose, open_gap, "COMPLETE", True),
    ]
    fps = config["control_hz"]
    times = np.arange(round(config["duration_s"] * fps) + 1) / fps
    qpath = np.zeros((len(times), len(_JOINT_NAMES)), np.float32)
    targets, labels = ([], [])
    q = home.copy()
    segment = 0
    joint_start = None
    for i, t in enumerate(times):
        while segment < len(keys) - 2 and t > keys[segment + 1][0] + 1e-10:
            segment += 1
            joint_start = q.copy()
        a, b = (keys[segment], keys[segment + 1])
        u = float(np.clip((t - a[0]) / (b[0] - a[0]), 0, 1))
        u = u**3 * (10 - 15 * u + 6 * u * u)
        target = np.eye(4)
        target[:3, 3] = (1 - u) * a[1][:3, 3] + u * b[1][:3, 3]
        target[:3, :3] = Slerp(
            [0, 1], Rotation.from_matrix([a[1][:3, :3], b[1][:3, :3]])
        )([u]).as_matrix()[0]
        if b[4]:
            if joint_start is None:
                joint_start = q.copy()
            q = (1 - u) * joint_start + u * home
            target = fk(q)
        else:
            q = kinematics.solve_nominal(target, q)
        qpath[i, :7] = q
        qpath[i, 7:9] = (1 - u) * a[2] + u * b[2]
        targets.append(target)
        labels.append(b[3])
    actual = (
        kinematics.chain.forward_kinematics(
            torch.tensor(qpath[:, :7], dtype=torch.float64)
        )
        .get_matrix()
        .numpy()
    )
    actual[:, 2, 3] += config.get("floor_height_m", 0.0)
    targets = np.asarray(targets)
    pe = np.linalg.norm(actual[:, :3, 3] - targets[:, :3, 3], axis=1)
    re = Rotation.from_matrix(
        targets[:, :3, :3] @ actual[:, :3, :3].transpose(0, 2, 1)
    ).magnitude()
    vmax = np.max(np.abs(np.diff(qpath[:, :7], axis=0) * fps), axis=0)
    velocity_limits = np.array(
        [
            float(
                ET.parse(robot_path)
                .getroot()
                .find(f"joint[@name='{name}']/limit")
                .get("velocity")
            )
            for name in _ARM_NAMES
        ]
    )
    if pe.max() > 1e-05 or re.max() > 0.0001 or np.any(vmax > velocity_limits):
        raise RuntimeError(
            f"Motion validation failed: position {pe.max()}, rotation {re.max()}, velocity {vmax}"
        )
    if np.any(qpath[:, :7] < kinematics.lo - 1e-6) or np.any(
        qpath[:, :7] > kinematics.hi + 1e-6
    ):
        raise RuntimeError("Nominal Franka RJ45 motion violates position limits.")
    return Rj45Motion(qpath, targets, tuple(labels), times)


def _smooth_weight(value: float) -> float:
    return value**3 * (10 - 15 * value + 6 * value * value)


def _interpolate_rotation(
    start: np.ndarray, end: np.ndarray, weight: float
) -> np.ndarray:
    return Slerp([0, 1], Rotation.from_matrix([start, end]))([weight]).as_matrix()[0]


class Rj45Feedback:
    """Carry measured grasp transforms across the calibrated pickup and insertion."""

    def __init__(
        self,
        config: dict[str, Any],
        robot_path: Path,
        settled_plug: np.ndarray,
    ) -> None:
        self._kinematics = _Rj45Kinematics(robot_path, config["floor_height_m"])
        self._settled_plug = settled_plug.copy()
        self._pickup_origin = np.asarray(config["pickup_tip_m"])
        self._pickup_turn = settled_plug[:3, :3] @ _PLUG_R.T
        self._carry_grasp = _pose(config["grasp_offset_m"], _HAND_IN_PLUG)
        self._vertical_grasp = _pose(
            config["withdraw_grasp_offset_m"], _WITHDRAW_IN_PLUG
        )
        self._ideal_seat = _pose(
            np.asarray(config["socket_mouth_m"])
            + _PLUG_R @ np.array([0, 0, config["seat_depth_m"]])
        )
        self._grasp: np.ndarray | None = None
        self._retained_plug: np.ndarray | None = None
        self._observed_withdraw_grasp = False

    def correct(
        self,
        time_s: float,
        nominal_q: np.ndarray,
        nominal_tcp: np.ndarray,
        observe: Callable[[], tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Correct one command from the preceding live state.

        Args:
            time_s: Time at which this command arrives on the control grid.
            nominal_q: Full-joint reference at the current tick.
            nominal_tcp: Nominal tool pose at the current tick.
            observe: Read the current tool and plug poses, in that order.

        Returns:
            Full-joint command with the measured-pose correction applied.
        """
        target = nominal_tcp.copy()
        if 1 < time_s < _CARRY_CAPTURE_START_S:
            weight = _smooth_weight(
                min((time_s - 1) / 2, 1.0)
                if time_s < 6
                else (_CARRY_CAPTURE_START_S - time_s) / 3
            )
            turn = _interpolate_rotation(np.eye(3), self._pickup_turn, weight)
            origin = self._pickup_origin
            target[:3, :3] = turn @ target[:3, :3]
            target[:3, 3] = (
                origin
                + turn @ (target[:3, 3] - origin)
                + weight * (self._settled_plug[:3, 3] - origin)
            )
        elif _CARRY_CAPTURE_START_S <= time_s <= 14 or _REGRASP_START_S < time_s <= 34:
            target = self._plug_target(time_s, nominal_tcp, observe)
        else:
            return nominal_q.copy()
        command = nominal_q.copy()
        command[:7] = self._kinematics.solve_feedback(target, nominal_q[:7])
        return command

    def _plug_target(
        self,
        time_s: float,
        nominal_tcp: np.ndarray,
        observe: Callable[[], tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        extracting = time_s > _REGRASP_START_S
        nominal_grasp = self._vertical_grasp if extracting else self._carry_grasp
        if _REGRASP_START_S < time_s < _REGRASP_CAPTURE_S:
            self._grasp = nominal_grasp
        elif (
            time_s <= _CARRY_CAPTURE_END_S
            or (extracting and not self._observed_withdraw_grasp)
            or self._grasp is None
        ):
            hand, plug = observe()
            self._grasp = np.linalg.inv(plug) @ hand
            if extracting:
                self._observed_withdraw_grasp = True
                self._retained_plug = plug.copy()

        desired_plug = nominal_tcp @ np.linalg.inv(nominal_grasp)
        if extracting:
            if self._retained_plug is None:
                _, plug = observe()
                self._retained_plug = plug.copy()
            weight = _smooth_weight(
                float(
                    np.clip(
                        (
                            (time_s - _REGRASP_START_S) / 3
                            if time_s <= 28
                            else (_PLACEMENT_CAPTURE_START_S - time_s) / 3
                        ),
                        0,
                        1,
                    )
                )
            )
            ideal = self._ideal_seat
            retained = self._retained_plug
            delta = retained @ np.linalg.inv(ideal)
            # Turn about the socket rather than the world origin.
            turn = _interpolate_rotation(np.eye(3), delta[:3, :3], weight)
            desired_plug[:3, 3] = (
                ideal[:3, 3]
                + turn @ (desired_plug[:3, 3] - ideal[:3, 3])
                + weight * (retained[:3, 3] - ideal[:3, 3])
            )
            desired_plug[:3, :3] = turn @ desired_plug[:3, :3]

        grasp = self._grasp
        assert grasp is not None
        if _PLACEMENT_CAPTURE_START_S <= time_s <= _PLACEMENT_CAPTURE_END_S:
            # Update before floor release, then freeze the estimate while opening.
            hand, plug = observe()
            measured = np.linalg.inv(plug) @ hand
            grasp[:3, 3] = 0.9 * grasp[:3, 3] + 0.1 * measured[:3, 3]
            grasp[:3, :3] = _interpolate_rotation(grasp[:3, :3], measured[:3, :3], 0.1)
        return desired_plug @ grasp
