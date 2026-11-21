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

"""A contact-based DexUni expert for the drawer storage task."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import product
from typing import Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils.registration import register_env

from ._drawer_motion import _DrawerMotion

__all__ = ["DrawerStorageDexUniEnv"]


@register_env("DrawerStorage-UR5-DexUni-v1", max_episode_steps=4800)
class DrawerStorageDexUniEnv(EmbodiedEnv):
    """Store a cube inside a drawer and leave the rail closed."""

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Generate measured opening, transfer and closure actions for the UR5.

        Args:
            **kwargs: Additional demo launcher arguments.

        Yields:
            Action segments with checks for rail travel, grasp and cube containment.

        Raises:
            ValueError: If the environment contains more than one arena.
        """
        if self.num_envs != 1:
            raise ValueError("The drawer storage expert requires one arena.")
        expert = _DrawerExpert(self)
        yield DemoSegment(
            name="open_drawer",
            target_uid="drawer",
            instruction="Grip the handle and open the drawer.",
            actions=expert.slide(0.26),
            validator=lambda: 0.25 <= expert.rail <= 0.28,
        )
        yield DemoSegment(
            name="pick_cube",
            target_uid="cube",
            instruction="Pick up and lift the cube from the side table.",
            actions=expert.pick(),
            validator=lambda: expert.cube_height - expert.initial_cube_height > 0.08,
        )
        yield DemoSegment(
            name="store_cube",
            target_uid="cube",
            instruction="Release the cube inside the open drawer.",
            actions=expert.store(),
            validator=expert.cube_stored,
        )
        yield DemoSegment(
            name="close_drawer",
            target_uid="drawer",
            instruction="Close the drawer and withdraw the gripper.",
            actions=expert.slide(0.0),
            validator=lambda: expert.rail <= 0.01 and expert.cube_stored(),
        )


class _DrawerExpert(_DrawerMotion):
    """Plan bounded arm and gripper commands from the observed scene."""

    def __init__(self, env: DrawerStorageDexUniEnv) -> None:
        super().__init__(env)
        self.drawer = env.sim.get_articulation("drawer")
        self.cube = env.sim.get_rigid_object("cube")
        self.initial_cube_height = self.cube_height
        self.top = self._tensor(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
        self.handle_orientation = self._tensor(((0, 0, -1), (0, 1, 0), (1, 0, 0)))
        self.corners = self._tensor(list(product((-0.025, 0.025), repeat=3)))

    @property
    def rail(self) -> float:
        return float(self.drawer.get_qpos()[0, 0])

    @property
    def cube_height(self) -> float:
        return float(self.cube.get_local_pose()[0, 2])

    def slide(self, target: float) -> Iterator[ControllerAction]:
        tolerance = 0.007 if target == 0.0 else 0.01
        for _ in range(5):
            if abs(self.rail - target) < tolerance:
                break
            handle = self.drawer.get_link_pose("large_handle_bar")[0, :3]
            contact = handle + self._tensor((-0.009, 0, 0))
            approach = contact + self._tensor((0.09, 0, 0))
            # The calibrated horizontal grasp leaves the wrist outside its
            # folded singularity when the drawer is fully open.
            approach[0] = approach[0].clamp_max(-0.542)
            yield from self._approach(approach, self.handle_orientation)
            yield from self._move(contact, self.handle_orientation, 40)
            yield from self._grip(0.04)
            yield from self._hold(12)
            # Regrasp between short pulls so the rail is moved by frictional contact.
            stroke = max(-0.095, min(0.095, target - self.rail))
            end = contact + self._tensor((stroke, 0, 0))
            yield from self._move(end, self.handle_orientation, 65)
            yield from self._grip(0.0)
            retreat = self._tool_position() + self._tensor((0.075, 0, 0))
            retreat[0] = retreat[0].clamp_max(-0.542)
            yield from self._move(retreat, self.handle_orientation, 35)
        if target == 0.0:
            yield from self._hold(40)

    def pick(self) -> Iterator[ControllerAction]:
        origin = self.cube.get_local_pose()[0, :3]
        above = origin + self._tensor((0, 0, 0.15))
        clear_handle = self._tool_position().clone()
        clear_handle[2] = 0.63
        yield from self._move(clear_handle, self.handle_orientation, 45)
        yield from self._approach(above, self.top)
        yield from self._move(origin, self.top, 45)
        yield from self._grip(0.034)
        yield from self._hold(15)
        yield from self._move(above, self.top, 48)

    def store(self) -> Iterator[ControllerAction]:
        tray = self.drawer.get_link_pose("drawer", to_matrix=True)[0]
        drop = tray[:3, 3] + tray[:3, :3] @ self._tensor((0, 0.068, 0.014))
        above = drop + self._tensor((0, 0, 0.115))
        yield from self._move(self._tensor((-0.47, 0.20, 0.574)), self.top, 50)
        yield from self._move(above, self.top, 75)
        yield from self._move(drop, self.top, 60)
        yield from self._grip(0.0)
        yield from self._hold(25)
        yield from self._move(above, self.top, 45)

    def cube_stored(self) -> bool:
        tray = self.drawer.get_link_pose("drawer", to_matrix=True)[0]
        cube = self.cube.get_local_pose(to_matrix=True)[0]
        local = torch.linalg.solve(tray, cube)
        corners = self.corners @ local[:3, :3].T + local[:3, 3]
        lower = self._tensor((-0.1975, 0.009, -0.058))
        upper = self._tensor((0.1975, 0.310, 0.0435))
        contained = bool(((corners >= lower) & (corners <= upper)).all())
        released = bool((self.robot.get_qpos(name="eef").abs() <= 0.002).all())
        clear = all(
            float(
                torch.linalg.vector_norm(
                    self.robot.get_link_pose(link)[0, :3] - cube[:3, 3]
                )
            )
            > 0.08
            for link in ("gripper_finger1_link_1", "gripper_finger2_link_1")
        )
        return contained and released and clear
