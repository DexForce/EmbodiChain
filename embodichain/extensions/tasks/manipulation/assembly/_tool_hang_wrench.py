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
    *,
    refine: bool = False,
) -> Iterator[ControllerAction]:
    for _ in range(3 if refine else 1):
        hand = motion.hand()
        if np.linalg.norm(hand[:3, 3] - motion.point("tool", _GRIP)) > 0.05:
            raise RuntimeError("The wrench slipped during transport.")
        position = motion.point("tool", site)
        difference = Rotation.from_matrix(orientation @ motion.pose("tool")[:3, :3].T)
        if refine:
            if (
                np.linalg.norm(goal - position) < 0.0028
                and difference.magnitude() < 0.38
            ):
                break
            difference = Rotation.from_rotvec(_limit(difference.as_rotvec(), 0.12))
        rotation = difference.as_matrix()
        xyz = goal - rotation @ (position - hand[:3, 3])
        if refine:
            xyz = hand[:3, 3] + _limit(xyz - hand[:3, 3], 0.03)
        yield from motion.move(
            xyz,
            rotation @ hand[:3, :3],
            0.0,
            rate=1.1 if refine else 1.5,
            settle=6 if refine else 3,
        )


def hang_wrench(motion: ToolHangMotion) -> Iterator[ControllerAction]:
    """Lift the large ring above the retaining lip and release it onto the bar."""
    motion.grip_effort = 19.0
    yield from _pick_wrench(motion)
    free_end = motion.point("frame", _FREE_END)
    along = motion.point("frame", _ELBOW) - free_end
    along[2] = 0.0
    along /= np.linalg.norm(along)
    lean = np.deg2rad(62.0)
    tilt = Rotation.from_rotvec(np.cross(_UP, along) * lean)
    hanging = np.column_stack((-_UP, np.cross(-along, -_UP), -along))
    orientation = tilt.as_matrix() @ hanging
    ring_up = np.cos(lean) * _UP + np.sin(lean) * along
    # The upper segment passes the lip while the lower segment stays outside.
    toward_base = motion.robot.get_local_pose()[0, :3].cpu().numpy() - free_end
    far_side = np.dot(toward_base, along) > 0
    into_bar, drop = (0.016, 0.038) if far_side else (0.014, 0.040)
    release = free_end + into_bar * along + drop * _UP - 0.012934 * ring_up
    current = motion.pose("tool")[:3, :3]
    swing = Rotation.from_matrix(orientation @ current.T).as_rotvec()
    halfway = Rotation.from_rotvec(0.58 * swing).as_matrix() @ current
    yield from _place_site(motion, _GRIP, np.array([-0.07, -0.15, 1.07]), halfway)
    above = release + 0.03 * _UP
    yield from _place_site(motion, _HOLE, above, orientation)
    yield from _place_site(motion, _HOLE, above, orientation, refine=True)
    yield from _place_site(motion, _HOLE, release, orientation, refine=True)
    hand = motion.hand().copy()
    yield from motion.move(hand[:3, 3], hand[:3, :3], 0.04, steps=7, settle=3)
    # On the far side of the fixture, withdrawing along the wrist can sweep the
    # shaft. Back out along the bar's free end before raising the open jaws.
    retreat = (
        -0.060 * along + 0.025 * _UP if far_side else -0.060 * hand[:3, 2] + 0.040 * _UP
    )
    yield from motion.move(hand[:3, 3] + retreat, hand[:3, :3], 0.04, rate=0.8)
    yield from motion.hold(0.04, 24)
