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

"""Explicit native command-target observation, independent of task authority."""

from __future__ import annotations
from collections.abc import Callable
from typing import TYPE_CHECKING
import torch
from embodichain.lab.sim.evidence import (
    PhysicalObservationStamp,
    RobotCommandObservation,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["capture_robot_commands", "verify_robot_commands"]


def _stamp(
    expected: PhysicalObservationStamp, read: Callable[[], PhysicalObservationStamp]
) -> None:
    if type(expected) is not PhysicalObservationStamp or read() != expected:
        raise ValueError("robot_command_observation_stamp_changed")


def _arrays(robot: Robot, shape: torch.Size) -> tuple[torch.Tensor, ...]:
    values = []
    for reader, target in (
        (robot.get_qpos, False),
        (robot.get_qvel, False),
        (robot.get_qpos, True),
        (robot.get_qvel, True),
    ):
        value = reader(target=target)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError("retained_native_command_snapshot_unavailable")
        values.append(value.detach().clone())
    return tuple(values)


def capture_robot_commands(
    robot: Robot,
    *,
    expected_qpos: torch.Tensor,
    expected_qvel: torch.Tensor,
    timestamp: float,
    source_stamp: PhysicalObservationStamp,
    read_source_stamp: Callable[[], PhysicalObservationStamp],
) -> RobotCommandObservation:
    """Read actual positions/velocities then position/velocity drive targets.

    Args:
        robot: Borrowed exact host-owned robot; no scene lookup or singleton.
        expected_qpos: Actual B1 observation positions to compare exactly.
        expected_qvel: Actual observation velocities; projected values are invalid.
        timestamp: Host time of that observation.
        source_stamp: Complete same-observation native stamp.
        read_source_stamp: Host reader for the exact native source/lifetime.

    Returns:
        Owned command observation. This function is opt-in and never advances
        physics or supplies retained-command authority.

    Raises:
        ValueError: Observation, native source, or command arrays are unavailable
            or change during capture. There is no later-state fallback.
    """
    if (
        not isinstance(expected_qpos, torch.Tensor)
        or not isinstance(expected_qvel, torch.Tensor)
        or expected_qpos.ndim != 2
        or expected_qpos.shape[0] != 1
        or expected_qvel.shape != expected_qpos.shape
    ):
        raise ValueError("Robot command capture requires the explicit B1 observation.")
    _stamp(source_stamp, read_source_stamp)
    actual, velocity, desired, desired_velocity = _arrays(robot, expected_qpos.shape)
    if not torch.equal(actual, expected_qpos) or not torch.equal(
        velocity, expected_qvel
    ):
        raise ValueError("retained_observation_changed")
    _stamp(source_stamp, read_source_stamp)
    return RobotCommandObservation(
        robot_uid=robot.uid,
        joint_names=tuple(robot.joint_names),
        timestamp=timestamp,
        source_stamp=source_stamp,
        qpos=actual,
        qvel=velocity,
        target_qpos=desired,
        target_qvel=desired_velocity,
    )


def verify_robot_commands(
    robot: Robot,
    observation: RobotCommandObservation,
    *,
    read_source_stamp: Callable[[], PhysicalObservationStamp],
) -> None:
    """Verify the captured arrays and complete source identity after qualification.

    Args:
        robot: The same borrowed host-owned robot used for capture.
        observation: Original owned actual-state/drive-target values.
        read_source_stamp: Reader for the same native source/lifetime.

    Raises:
        ValueError: Any identity or array changed. This reads again only to reject
            change; fresh values never replace the original observation.
    """
    if (
        robot.uid != observation.robot_uid
        or tuple(robot.joint_names) != observation.joint_names
    ):
        raise ValueError("retained_robot_identity_changed")
    _stamp(observation.source_stamp, read_source_stamp)
    current = _arrays(robot, observation.qpos.shape)
    _stamp(observation.source_stamp, read_source_stamp)
    if any(
        not torch.equal(before, after)
        for before, after in zip(
            (
                observation.qpos,
                observation.qvel,
                observation.target_qpos,
                observation.target_qvel,
            ),
            current,
            strict=True,
        )
    ):
        raise ValueError("retained_command_changed_during_check")
