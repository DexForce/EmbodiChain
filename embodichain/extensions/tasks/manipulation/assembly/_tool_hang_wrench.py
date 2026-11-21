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

"""Grasp, carry, and release the wrench over the assembled Tool Hang frame."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.extensions.tasks.manipulation.assembly._tool_hang_expert import (
    ToolHangMotion,
    _limit,
    _UP,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.types import ControllerAction

__all__: list[str] = []

_GRIP = np.array([-0.026, 0.0, 0.0])
_HOLE = np.array([-0.0935328, 0.0, 0.0])
_FREE_END = np.array([-0.0475, 0.0, 0.08625])
_ELBOW = np.array([0.04375, 0.0, 0.08625])


def _pick_wrench(motion: ToolHangMotion) -> Iterator[ControllerAction]:
    closing = motion.pose("tool")[:3, 1].copy()
    closing[2] = 0.0
    closing /= np.linalg.norm(closing)
    down = np.column_stack((np.cross(closing, -_UP), closing, -_UP))
    overhead = motion.point("tool", np.array([-0.004, 0.0, 0.0])) + 0.14 * _UP
    yield from motion.move(overhead, down, 0.04, rate=0.7, settle=9)
    yield from motion.hold(0.04, 4)
    yield from motion.move(motion.point("tool", _GRIP) + 0.07 * _UP, down, 0.04)
    yield from motion.move(
        motion.point("tool", _GRIP) - 0.006 * _UP, down, 0.04, rate=0.45
    )
    yield from motion.hold(0.0, 16)
    start = motion.hand()[:3, 3].copy()
    for rise in (0.018, 0.060, 0.14):
        yield from motion.move(start + rise * _UP, down, 0.0, rate=0.55)
        if motion.point("tool", _GRIP)[2] < start[2] + 0.5 * rise:
            raise RuntimeError("The wrench did not rise with the fingers.")


def _place_site(
    motion: ToolHangMotion,
    site: np.ndarray,
    goal: np.ndarray,
    orientation: np.ndarray,
) -> Iterator[ControllerAction]:
    hand = motion.hand()
    if np.linalg.norm(hand[:3, 3] - motion.point("tool", _GRIP)) > 0.05:
        raise RuntimeError("The wrench slipped during transport.")
    rotation = Rotation.from_matrix(
        orientation @ motion.pose("tool")[:3, :3].T
    ).as_matrix()
    xyz = goal - rotation @ (motion.point("tool", site) - hand[:3, 3])
    yield from motion.move(xyz, rotation @ hand[:3, :3], 0.0, rate=1.5, settle=3)


def _bar(motion: ToolHangMotion) -> tuple[np.ndarray, np.ndarray]:
    tip = motion.point("frame", _FREE_END)
    along = motion.point("frame", _ELBOW) - tip
    return tip, along / np.linalg.norm(along)


def _ring_target(
    motion: ToolHangMotion, x: float, height: float, lean: float, spin: float
) -> tuple[np.ndarray, np.ndarray]:
    tip, along = _bar(motion)
    horizontal = along.copy()
    horizontal[2] = 0.0
    horizontal /= np.linalg.norm(horizontal)
    upright = np.column_stack((-_UP, np.cross(-horizontal, -_UP), -horizontal))
    rotation = Rotation.from_rotvec(np.cross(_UP, horizontal) * np.deg2rad(lean))
    side = Rotation.from_euler("z", spin, degrees=True).as_matrix()
    return tip + x * along + height * _UP, rotation.as_matrix() @ upright @ side


def _guide_ring(
    motion: ToolHangMotion, x: float, height: float, lean: float, spin: float
) -> Iterator[ControllerAction]:
    """Track the live bar and wrench without pushing a wedged ring through the lip."""
    for _ in range(6):
        goal, orientation = _ring_target(motion, x, height, lean, spin)
        hand = motion.hand()
        current = motion.pose("tool")[:3, :3]
        site = motion.point("tool", _HOLE)
        turn = Rotation.from_matrix(orientation @ current.T).as_rotvec()
        if np.linalg.norm(goal - site) < 0.0016 and np.linalg.norm(turn) < 0.10:
            break
        rotation = Rotation.from_rotvec(_limit(turn, 0.14)).as_matrix()
        target = goal - rotation @ (site - hand[:3, 3])
        target = hand[:3, 3] + _limit(target - hand[:3, 3], 0.012)
        yield from motion.move(
            target, rotation @ hand[:3, :3], 0.0, rate=0.60, settle=4
        )


def hang_wrench(motion: ToolHangMotion) -> Iterator[ControllerAction]:
    """Guide the ring over the lip and seat it on the bar before opening the fingers."""
    motion.grip_effort = 19.0
    yield from _pick_wrench(motion)
    # Turn the long handle toward the open work area so its swing clears the
    # socket. Keep the chosen side as small forces rotate the loose frame.
    _, axis = _bar(motion)
    spin = 40.0 if axis[0] >= 0.0 else -40.0
    above, orientation = _ring_target(motion, 0.004, 0.065, 62.0, spin)
    current = motion.pose("tool")[:3, :3]
    swing = Rotation.from_matrix(orientation @ current.T).as_rotvec()
    halfway = Rotation.from_rotvec(0.58 * swing).as_matrix() @ current
    yield from _place_site(motion, _GRIP, np.array([-0.07, -0.15, 1.07]), halfway)
    yield from _place_site(motion, _HOLE, above, orientation)
    # Each position is (distance along the bar, height above it, ring tilt).
    waypoints = (
        (0.004, 0.029, 62.0),
        (0.004, 0.013, 62.0),
        (0.005, 0.008, 55.0),
        (0.009, 0.005, 43.0),
        (0.016, 0.002, 26.0),
        (0.027, -0.002, 6.0),
    )
    motions = (_guide_ring(motion, *waypoint, spin) for waypoint in waypoints)
    for action in chain.from_iterable(motions):
        yield action
        metrics = motion.env.compute_task_state()[2]
        if (
            metrics["success.hanging.frame_assembled"].item()
            and metrics["success.hanging.hole_threaded"].item()
            and metrics["success.hanging.ring_bar_contact"].item()
            and metrics["success.hanging.along_fraction"].item() >= 0.065
        ):
            break
    else:
        # The ring has not reached the bar. Let the normal evaluator report
        # the incomplete attempt without opening the grasp in free space.
        return
    hand = motion.hand().copy()
    yield from motion.hold(0.04, 12, target=True)
    yield from motion.move(
        hand[:3, 3] - 0.055 * hand[:3, 2] + 0.025 * _UP, hand[:3, :3], 0.04, rate=0.6
    )
    yield from motion.hold(0.04, 24, target=True)
