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

from __future__ import annotations

from collections.abc import Iterable, Iterator
from importlib.resources import files
from typing import Any

import torch


from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.sim.cfg import ArticulationCfg
from embodichain.utils import logger

from .._drawer_motion import _DrawerMotion

__all__ = ["PlaceObjectDrawerEnv", "PlaceObjectDrawerDexUniEnv"]


@register_env("PlaceObjectDrawer-DualUR5-v1", max_episode_steps=600)
@register_env("PlaceObjectDrawer-DexforceW1V030Pika-v1", max_episode_steps=600)
@register_env("PlaceObjectDrawer-TianjiMarvin-v1", max_episode_steps=600)
@register_env("PlaceObjectDrawer-v1", max_episode_steps=600)
class PlaceObjectDrawerEnv(EmbodiedEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

    def is_task_success(self, **kwargs) -> torch.Tensor:
        """Determine if the task is successfully completed.

        The task is successful if:
        1. Object is within drawer inner_box area
        2. Drawer has been closed

        Args:
            **kwargs: Additional arguments for task-specific success criteria.

        Returns:
            torch.Tensor: A boolean tensor indicating success for each environment in the batch.
        """
        try:
            object_obj = self.sim.get_rigid_object("object")
            drawer = self.sim.get_articulation("drawer")
        except Exception as e:
            logger.log_warning(f"Object or drawer not found: {e}, returning False.")
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Get poses
        object_pose = object_obj.get_local_pose(to_matrix=True)
        # Get drawer inner_box (drawer interior) pose, not outer_box
        inner_box_pose = drawer.get_link_pose("inner_box", to_matrix=True)

        # Extract positions
        object_pos = object_pose[:, :3, 3]  # (num_envs, 3)
        inner_box_pos = inner_box_pose[:, :3, 3]  # (num_envs, 3)

        # Get drawer joint position
        drawer_qpos = drawer.get_qpos()  # (num_envs, num_joints)
        drawer_joint_pos = drawer_qpos[:, 0]

        # Check if drawer has been closed
        drawer_closed = drawer_joint_pos < 0.05

        # Check if object is within drawer inner_box area
        xy_diff = torch.abs(object_pos[:, :2] - inner_box_pos[:, :2])  # (num_envs, 2)
        xy_tolerance = torch.tensor(
            [0.03, 0.03], dtype=torch.float32, device=self.device
        )
        object_in_drawer_xy = torch.all(xy_diff < xy_tolerance.unsqueeze(0), dim=1)

        # Object must be in drawer and drawer must be closed
        success = drawer_closed & object_in_drawer_xy

        return success


def prepare_tableware_drawer(
    cfg: EmbodiedEnvCfg, articulation: dict[str, Any]
) -> EmbodiedEnvCfg:
    """Load the packaged drawer model and its authored scene pose."""
    model = files("embodichain.extensions.tasks.configs").joinpath(
        "tasks/manipulation/tableware/place_object_drawer/tabletop_drawer.urdf"
    )
    cfg.articulation.append(
        ArticulationCfg.from_dict({**articulation, "fpath": str(model)})
    )
    return cfg


@register_env("PlaceObjectDrawer-UR5-DexUni-v1", max_episode_steps=3600)
class PlaceObjectDrawerDexUniEnv(EmbodiedEnv):
    """Put the toy in the open tabletop drawer, then push the drawer closed."""

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Generate the physical pickup, placement and closing actions.

        Args:
            **kwargs: Additional demo launcher arguments.

        Yields:
            Action segments that check the lift, placement and final drawer position.

        Raises:
            ValueError: If the environment contains more than one arena.
        """
        if self.num_envs != 1:
            raise ValueError("The tabletop drawer expert requires one arena.")
        expert = _TablewareDrawerExpert(self)
        yield DemoSegment(
            name="pick_duck",
            target_uid="object",
            instruction="Grip the toy duck and lift it clear of the table.",
            actions=expert.pick(),
            validator=lambda: expert.lift_height() >= 0.045,
        )
        yield DemoSegment(
            name="store_duck",
            target_uid="object",
            instruction="Place the duck on the liner inside the open drawer.",
            actions=expert.store(),
            validator=expert.object_on_liner,
        )
        yield DemoSegment(
            name="close_drawer",
            target_uid="drawer",
            instruction="Push the drawer closed while the duck stays inside.",
            actions=expert.close(),
            validator=lambda: expert.rail < 0.04 and expert.object_on_liner(),
        )


class _TablewareDrawerExpert(_DrawerMotion):
    """Carry the duck around the cabinet and physically push the rail shut."""

    def __init__(self, env: EmbodiedEnv) -> None:
        super().__init__(env)
        self.drawer = env.sim.get_articulation("drawer")
        self.duck = env.sim.get_rigid_object("object")
        self.initial_height = float(self.duck.get_local_pose()[0, 2])
        self.top = self._tensor(((-1, 0, 0), (0, 1, 0), (0, 0, -1)))
        self.push = self._tensor(((0, 0, 1), (0, -1, 0), (1, 0, 0)))

    @property
    def rail(self) -> float:
        return float(self.drawer.get_qpos()[0, 0])

    def lift_height(self) -> float:
        return float(self.duck.get_local_pose()[0, 2]) - self.initial_height

    def pick(self) -> Iterator[ControllerAction]:
        center = self.duck.get_local_pose()[0, :3]
        grasp = center + self._tensor((0, 0.043, -0.016))
        above = grasp + self._tensor((0, 0, 0.14))
        yield from self._approach(above, self.top)
        yield from self._move(grasp, self.top, 45)
        yield from self._grip(0.038)
        yield from self._hold(16)
        yield from self._move(above, self.top, 48)

    def store(self) -> Iterator[ControllerAction]:
        tray = self.drawer.get_link_pose("inner_box", to_matrix=True)[0]
        tool_offset = self._tool_position() - self.duck.get_local_pose()[0, :3]
        drop = tray[:3, 3] + self._tensor((-0.009, -0.016, 0.096)) + tool_offset
        above = drop + self._tensor((0, 0, 0.085))
        # Approach the open tray along its exposed side and stay clear of the cabinet.
        yield from self._move(self._tensor((0.52, -0.12, 1.14)), self.top, 80)
        yield from self._move(self._tensor((0.52, 0.145, 1.14)), self.top, 90)
        yield from self._move(above, self.top, 65)
        yield from self._move(drop, self.top, 50)
        yield from self._grip(0.0)
        yield from self._hold(48)
        yield from self._move(above, self.top, 48)
        high = self._tool_position().clone()
        high[2] = 1.16
        yield from self._move(high, self.top, 45)
        yield from self._move(self._tensor((0.39, 0.11, 1.16)), self.top, 55)

    def close(self) -> Iterator[ControllerAction]:
        tray = self.drawer.get_link_pose("inner_box", to_matrix=True)[0]
        touch = tray[:3, 3] + tray[:3, :3] @ self._tensor((0.091, 0, 0.085))
        travel = self.rail + 0.019
        yield from self._approach(touch + self._tensor((-0.058, 0, 0)), self.push)
        yield from self._grip(0.032)
        yield from self._move(touch, self.push, 34)
        yield from self._move(touch + self._tensor((travel, 0, 0)), self.push, 230)
        retreat = self._tool_position() + self._tensor((-0.075, 0, 0))
        yield from self._move(retreat, self.push, 36)
        yield from self._grip(0.0)
        yield from self._hold(40)

    def object_on_liner(self) -> bool:
        tray = self.drawer.get_link_pose("inner_box", to_matrix=True)[0]
        displacement = self.duck.get_local_pose()[0, :3] - tray[:3, 3]
        relative = tray[:3, :3].T @ displacement
        centered = bool((relative[:2].abs() < 0.03).all())
        above_liner = 0.062 < float(relative[2]) < 0.18
        released = bool((self.robot.get_qpos(name="eef").abs() <= 0.002).all())
        return centered and above_liner and released
