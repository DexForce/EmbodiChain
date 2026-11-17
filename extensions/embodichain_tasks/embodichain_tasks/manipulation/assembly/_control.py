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

"""Encoder gravity feedforward and policy command application."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from embodichain.compute.kinematics import URDFGravityCompensation
from embodichain.lab.gym.envs.managers.actions import RelativeJointPositionAction

__all__ = ["AssemblyJointPositionAction"]


class EncoderGravityCompensation:
    """Adapt configured robot joint order and root frame to the shared model."""

    def __init__(self, env: Any, arm_part: str) -> None:
        self.env = env
        gravity = np.asarray(self.env.sim_cfg.physics_cfg.gravity, dtype=float)
        if gravity.ndim == 0:
            gravity = np.array([0, 0, float(gravity)])
        local_pose = getattr(self.env.cfg.robot, "init_local_pose", None)
        root_rotation = (
            np.asarray(local_pose, dtype=float)[:3, :3]
            if local_pose is not None
            else Rotation.from_euler(
                "xyz", self.env.cfg.robot.init_rot, degrees=True
            ).as_matrix()
        )
        root_transform = (
            np.asarray(local_pose, dtype=float).copy()
            if local_pose is not None
            else np.eye(4)
        )
        if local_pose is None:
            root_transform[:3, :3] = root_rotation
            root_transform[:3, 3] = getattr(self.env.cfg.robot, "init_pos", [0, 0, 0])
        self.robot_base = torch.tensor(
            root_transform, dtype=torch.float32, device=self.env.device
        )

        self._model = URDFGravityCompensation(
            Path(env.cfg.robot._pk_urdf_path).read_bytes(),
            root_rotation.T @ gravity,
            device=env.device,
        )
        names = list(self._model.joint_names)
        self._gravity_joint_ids = torch.tensor(
            [env.robot.joint_names.index(name) for name in names], device=env.device
        )
        self._gravity_arm_ids = torch.tensor(
            [
                names.index(env.robot.joint_names[index])
                for index in env.robot.get_joint_ids(name=arm_part, remove_mimic=True)
            ],
            device=env.device,
        )

    def torques(self) -> torch.Tensor:
        """Read encoders and select configured arm torques without simulator truth."""
        q = self.env.robot.get_qpos()[:, self._gravity_joint_ids]
        return self._model.torques(q)[:, self._gravity_arm_ids]


class AssemblyJointPositionAction(RelativeJointPositionAction):
    """Own policy inputs, arm drive and held gripper command.

    The configuration selects this implementation and its arm control part.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        """Apply the per-row warmup mask before measured-position delta scaling.

        Args:
            actions: Batched configured-arm policy requests; the input is unchanged.
        """
        masked = actions.clone()
        masked[
            self._env._elapsed_steps < self._env.assembly_state.cfg.grasp_warmup_steps
        ] = 0
        super().process_actions(masked)

    def apply_actions(self) -> None:
        """Apply encoder feedforward, held grasp and the processed arm target."""
        state = self._env.assembly_state
        self._env.robot.set_qf(state.controller.torques(), name=state.cfg.arm_part)
        self._env.robot.set_qpos(
            torch.full(
                (
                    self._env.num_envs,
                    len(self._env.robot.get_joint_ids(name=state.cfg.gripper_part)),
                ),
                state.scene._grasp_target_joint,
                device=self._env.device,
            ),
            name=state.cfg.gripper_part,
        )
        super().apply_actions()
