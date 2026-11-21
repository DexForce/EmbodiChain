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

"""Franka opening the hinged door from RoboCasa365 OpenMicrowave episode 86."""

from __future__ import annotations

from collections.abc import Iterable
from functools import cached_property
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Articulation

__all__ = ["RoboCasa365OpenMicrowaveEnv"]


@register_env(
    "RoboCasa365OpenMicrowave-Franka-v1",
    max_episode_steps=300,
    supports_handwritten_demo=True,
)
class RoboCasa365OpenMicrowaveEnv(EmbodiedEnv):
    """Evaluate door travel, independently of the gripper and microwave buttons."""

    @cached_property
    def _door(self) -> tuple[Articulation, int]:
        asset = self.sim.get_articulation("microwave")
        name = "microwave_right_group_1_microjoint"
        if name not in asset.joint_names:
            raise ValueError(
                f"Missing microwave joint {name!r}. Available: {asset.joint_names}."
            )
        return asset, asset.joint_names.index(name)

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Measure opening from the closed upper limit towards the negative lower limit."""
        asset, index = self._door
        position = asset.get_qpos()[:, index]
        lower, upper = asset.get_qpos_limits(joint_ids=[index])[:, 0].unbind(-1)
        fraction = (upper - position) / (upper - lower)
        valid = torch.isfinite(fraction) & (upper > lower)
        return valid & (fraction >= 0.90), ~valid, {"door_open_fraction": fraction}

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Return the measured predicate for demonstrations and external evaluation."""
        return self.compute_task_state(**kwargs)[0]

    def _grasp_pose(self, angle: torch.Tensor, approach: float) -> torch.Tensor:
        """Transform the fixture's measured hinge and handle into the arena frame."""
        fixture, _ = self._door
        root = fixture.get_local_pose(to_matrix=True)
        pose = torch.zeros_like(root)
        c, s = angle.cos(), angle.sin()
        # The source fixture.xml locates the vertical handle 551 mm from the hinge.
        # Aim 10 mm inside its front surface to keep both pads on the bar.
        radial = 0.194508 + 0.356796
        transverse = -0.264068 + 0.203346 + 0.010 + approach
        pose[:, 0, 3] = -0.356796 + c * radial - s * transverse
        pose[:, 1, 3] = -0.203346 + s * radial + c * transverse
        pose[:, 2, 3] = -0.009596
        # Once the door passes 37 degrees, rotate the wrist more slowly than
        # the vertical bar so the Franka can reach the far end of the swing.
        yaw = angle - 0.7 * (angle + 0.65).clamp(max=0)
        pose[:, 0, 1] = yaw.cos()
        pose[:, 1, 1] = yaw.sin()
        pose[:, 0, 2] = -yaw.sin()
        pose[:, 1, 2] = yaw.cos()
        pose[:, 2, 0] = pose[:, 3, 3] = 1
        return root @ pose

    def _move_hand(
        self, pose: torch.Tensor, command: float, steps: int
    ) -> Iterable[torch.Tensor]:
        arm = self.robot.get_joint_ids("arm")
        current = self.robot.get_qpos()[:, arm]
        result = self.robot.compute_ik(pose, joint_seed=current, name="arm")
        if result is None or not bool(result[0].all()):
            raise RuntimeError(
                "Microwave door handle is outside the Franka's IK workspace"
            )
        target = result[1].reshape_as(current)
        hand = current.new_full((self.num_envs, 1), command)
        scale = self.action_manager.get_term("arm").cfg.params["scale"]
        for _ in range(steps):
            current = self.robot.get_qpos()[:, arm]
            relative = ((target - current) / scale).clamp(-2, 2)
            yield torch.cat((relative, hand), dim=-1)

    def _pull_handle(self) -> Iterable[torch.Tensor]:
        fixture, index = self._door
        for _ in range(self.cfg.max_episode_steps // 2):
            if bool(self.is_task_success().all()):
                break
            angle = fixture.get_qpos()[:, index]
            # Small look-ahead keeps the fingers in contact. The final arc
            # needs additional travel as the wrist swings out beside the door.
            lead = torch.where(angle < -1.12, 0.05, 0.02)
            goal = self._grasp_pose((angle - lead).clamp(min=-1.56), 0.0)
            yield from self._move_hand(goal, -1.0, 2)

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Approach, grasp and pull the actual door with closed-loop joint actions."""
        fixture, index = self._door
        closed = fixture.get_qpos()[:, index]
        for name, instruction, offset, hand, steps in (
            (
                "approach",
                "Approach the microwave handle with an open gripper.",
                -0.12,
                1.0,
                38,
            ),
            ("reach", "Position the fingers around the handle.", 0.0, 1.0, 24),
            ("grasp", "Close the gripper on the handle.", 0.0, -1.0, 10),
        ):
            yield DemoSegment(
                name=name,
                target_uid="microwave",
                instruction=instruction,
                actions=self._move_hand(self._grasp_pose(closed, offset), hand, steps),
            )
        yield DemoSegment(
            name="open_door",
            target_uid="microwave",
            instruction="Pull the door open along its hinge arc.",
            actions=self._pull_handle(),
            validator=self.is_task_success,
        )
