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

"""Joint and Cartesian commands shared by the two drawer experts."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.envs.types import ControllerAction

__all__: list[str] = []


class _DrawerMotion:
    """Turn single-arena UR5 pose and gripper goals into controller actions."""

    def __init__(self, env: EmbodiedEnv) -> None:
        self.env = env
        self.robot = env.robot
        self.arm = self.robot.get_joint_ids("arm")
        self.fingers = self.robot.get_joint_ids("eef")
        self.command = self.robot.get_qpos().clone()

    def _tensor(self, values: Any) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.env.device)

    def _tool_position(self) -> torch.Tensor:
        pose = self.robot.compute_fk(
            self.robot.get_qpos(name="arm"), name="arm", to_matrix=True
        )
        return pose[0, :3, 3]

    def _hold(self, steps: int) -> Iterator[ControllerAction]:
        for _ in range(steps):
            yield ControllerAction(self.command)

    def _solve(self, position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
        pose = torch.eye(4, device=self.env.device).unsqueeze(0)
        pose[0, :3, :3] = rotation
        pose[0, :3, 3] = position
        success, joints = self.robot.compute_ik(
            pose, joint_seed=self.command[:, self.arm], name="arm"
        )
        if not bool(success[0]):
            raise RuntimeError("The drawer expert could not reach its target pose.")
        return joints[0]

    def _approach(
        self, position: torch.Tensor, rotation: torch.Tensor
    ) -> Iterator[ControllerAction]:
        start = self.command[0, self.arm].clone()
        target = self._solve(position, rotation)
        steps = 70
        for index in range(1, steps + 1):
            s = index / steps
            self.command[0, self.arm] = start.lerp(target, s * s * (3 - 2 * s))
            yield ControllerAction(self.command)
        yield from self._hold(15)

    def _move(
        self, position: torch.Tensor, rotation: torch.Tensor, steps: int
    ) -> Iterator[ControllerAction]:
        start = self._tool_position()
        for index in range(1, steps + 1):
            s = index / steps
            joints = self._solve(start.lerp(position, s * s * (3 - 2 * s)), rotation)
            if float((joints - self.command[0, self.arm]).abs().max()) > 0.45:
                raise RuntimeError(
                    "The drawer expert changed IK branches during a move."
                )
            self.command[0, self.arm] = joints
            yield ControllerAction(self.command)
        yield from self._hold(8)

    def _grip(self, position: float) -> Iterator[ControllerAction]:
        start = self.command[:, self.fingers].clone()
        steps = 25
        for index in range(1, steps + 1):
            self.command[:, self.fingers] = start + (position - start) * (index / steps)
            yield ControllerAction(self.command)
