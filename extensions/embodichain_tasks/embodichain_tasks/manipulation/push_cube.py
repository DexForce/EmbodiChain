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

from collections.abc import Iterable
from dataclasses import MISSING
from typing import Any

import torch

from embodichain.lab.gym.utils.registration import register_env
from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.types import EnvObs
from embodichain.utils import configclass
from embodichain.utils.math import matrix_from_quat

__all__ = ["PushCubeEnv"]


@configclass
class _PushExpertCfg:
    """Configured geometry and timing for the task's handwritten expert."""

    arm_part: str = MISSING
    eef_command: float = MISSING
    orientation_xyzw: tuple[float, ...] = MISSING
    approach_offset: tuple[float, ...] = MISSING
    approach_height: float = MISSING
    descend_height: float = MISSING
    push_displacement: tuple[float, ...] = MISSING
    steps: tuple[int, ...] = MISSING


@configclass
class _PushCubeCfg:
    """Task parameters independent of cameras and policy models."""

    success_height_range: tuple[float, float] | None = None
    failure_height: float | None = None
    truncation_height: float | None = -0.1


@register_env(
    "PushCube-Franka-Visual-v1", max_episode_steps=300, supports_handwritten_demo=True
)
@register_env(
    "PushCubeRL",
    max_episode_steps=50,
    override=True,
    supports_rl=True,
    supports_handwritten_demo=False,
)
class PushCubeEnv(EmbodiedEnv):
    """Push a cube toward a configured target, with optional expert segments.

    Scene entities, reset events, sensors and actions belong to the deployment.
    ``cfg.extensions['push_cube']`` selects height bounds and optional expert
    geometry. Both deployments use the existing ``success_threshold`` extension
    and ``distance_to_goal`` metric.

    Args:
        cfg: Physical scene, managers and task parameters.
        **kwargs: Arguments forwarded to the common environment.
    """

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        cfg = cfg if cfg is not None else EmbodiedEnvCfg()
        extensions = cfg.extensions or {}
        parameters = dict(extensions.get("push_cube", {}))
        expert = parameters.pop("expert", None)
        self._push_cfg = _PushCubeCfg(**parameters)
        self.success_threshold = extensions.get("success_threshold", 0.1)
        if self.success_threshold <= 0:
            raise ValueError("PushCube success_threshold must be positive")
        heights = self._push_cfg.success_height_range
        if heights is not None and (len(heights) != 2 or heights[0] > heights[1]):
            raise ValueError(
                "PushCube success_height_range must contain ordered bounds"
            )
        self._push_expert = None if expert is None else _PushExpertCfg(**expert)
        if self._push_expert is not None:
            self._push_expert.validate()
            if len(self._push_expert.steps) != 3 or any(
                type(steps) is not int or steps < 1 for steps in self._push_expert.steps
            ):
                raise ValueError(
                    "PushCube expert requires three positive segment lengths"
                )
        super().__init__(cfg, **kwargs)

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Measure configured XY target distance and object height bounds.

        Args:
            **kwargs: Compatibility evaluation keywords.

        Returns:
            Success and failure masks, and the configured distance metric.
        """
        c = self._push_cfg
        pose = self.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
        position = pose[:, :3, 3]
        goal = getattr(self, "goal_pose", None)
        success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        distance = torch.zeros(self.num_envs, device=self.device)
        if goal is not None:
            distance = torch.linalg.vector_norm(
                position[:, :2] - goal[:, :2, 3], dim=-1
            )
            success = distance < self.success_threshold
            if c.success_height_range is not None:
                low, high = c.success_height_range
                success &= (position[:, 2] >= low) & (position[:, 2] <= high)
        fail = (
            position[:, 2] < c.failure_height
            if c.failure_height is not None
            else torch.zeros_like(success)
        )
        return success, fail, {"distance_to_goal": distance}

    def check_truncated(self, obs: EnvObs, info: dict[str, Any]) -> torch.Tensor:
        """Apply the configured fall truncation independently of task failure.

        Args:
            obs: Current observations.
            info: Measured task results.

        Returns:
            Per-environment fall truncation mask.
        """
        c = self._push_cfg
        if c.truncation_height is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        pose = self.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
        return pose[:, 2, 3] < c.truncation_height

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        """Use measured success to accept recorded expert demonstrations.

        Args:
            **kwargs: Compatibility evaluation keywords.

        Returns:
            Measured per-environment success mask.
        """
        return self.compute_task_state(**kwargs)[0]

    def _expert_waypoints(
        self, targets: torch.Tensor, steps: int
    ) -> Iterable[torch.Tensor]:
        c = self._push_expert
        if c is None:
            raise ValueError("PushCube expert geometry must be configured")
        arm_ids = self.robot.get_joint_ids(c.arm_part)
        start_qpos = self.robot.get_qpos()[:, arm_ids]
        start_pose = self.robot.compute_fk(start_qpos, name=c.arm_part, to_matrix=True)
        for index in range(steps):
            fraction = float(index + 1) / steps
            pose = targets.clone()
            pose[:, :3, 3] = (
                start_pose[:, :3, 3] * (1 - fraction) + targets[:, :3, 3] * fraction
            )
            seed = self.robot.get_qpos()[:, arm_ids]
            result = self.robot.compute_ik(pose, joint_seed=seed, name=c.arm_part)
            if result is None or not bool(torch.as_tensor(result[0]).all()):
                raise RuntimeError(f"PushCube expert IK failed at waypoint {index}")
            qpos = result[1].reshape(self.num_envs, -1)
            yield torch.cat(
                (
                    qpos,
                    torch.full((self.num_envs, 1), c.eef_command, device=self.device),
                ),
                dim=-1,
            )

    def create_demo_segments(self, **kwargs: Any) -> Iterable[DemoSegment]:
        """Yield configured approach, descent and push segments.

        Args:
            **kwargs: Compatibility demonstration keywords.

        Returns:
            Lazy expert segments with measured final acceptance.

        Raises:
            ValueError: This deployment does not configure an expert.
        """
        c = self._push_expert
        if c is None:
            raise ValueError("This PushCube deployment does not configure an expert")
        cube = self.sim.get_rigid_object("cube").get_local_pose()
        target = torch.eye(4, device=self.device).repeat(self.num_envs, 1, 1)
        target[:, :3, :3] = matrix_from_quat(
            torch.tensor(c.orientation_xyzw, device=self.device)
        )
        target[:, :3, 3] = cube[:, :3]
        target[:, :2, 3] += torch.tensor(c.approach_offset, device=self.device)
        target[:, 2, 3] = c.approach_height
        for name, steps, instruction in zip(
            ("approach", "descend", "push"),
            c.steps,
            (
                "Move behind the cube.",
                "Lower the gripper.",
                "Push the cube into the target.",
            ),
        ):
            target = target.clone()
            if name == "descend":
                target[:, 2, 3] = c.descend_height
            elif name == "push":
                target[:, :3, 3] += torch.tensor(
                    c.push_displacement, device=self.device
                )
            yield DemoSegment(
                name=name,
                target_uid="cube",
                instruction=instruction,
                actions=self._expert_waypoints(target, steps),
                validator=self.is_task_success if name == "push" else None,
            )
