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
from embodichain.lab.gym.envs.managers.actions import RelativeJointPositionAction

__all__ = ["AssemblyJointPositionAction"]


class EncoderGravityCompensation:
    """Compute encoder-only feedforward from the prepared robot's URDF."""

    def __init__(self, env: Any) -> None:
        self.env = env
        import xml.etree.ElementTree as ET
        import pytorch_kinematics as pk

        xml = Path(self.env.cfg.robot._pk_urdf_path).read_bytes()
        self._gravity_chain = pk.build_chain_from_urdf(xml).to(device=self.env.device)
        names = self._gravity_chain.get_joint_parameter_names()
        self._gravity_joint_ids = torch.tensor(
            [self.env.robot.joint_names.index(name) for name in names],
            device=self.env.device,
        )
        self._gravity_arm_ids = torch.tensor(
            [names.index(f"joint{i}") for i in range(1, 8)], device=self.env.device
        )
        self._gravity_links = []
        for link in ET.fromstring(xml).findall("link"):
            inertia = link.find("inertial")
            if inertia is not None:
                origin = inertia.find("origin")
                com = (
                    np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
                    if origin is not None
                    else np.zeros(3)
                )
                self._gravity_links.append(
                    (
                        link.attrib["name"],
                        float(inertia.find("mass").attrib["value"]),
                        torch.tensor(com, dtype=torch.float32, device=self.env.device),
                    )
                )
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
        self._gravity_acceleration = torch.tensor(
            root_rotation.T @ gravity, dtype=torch.float32, device=self.env.device
        )

    def torques(self) -> torch.Tensor:
        # One batched differentiation per control step. The chain and COMs
        # are prepared once; no host transfers or simulator truth enter it.
        with torch.enable_grad():
            q = (
                self.env.robot.get_qpos()[:, self._gravity_joint_ids]
                .detach()
                .requires_grad_(True)
            )
            transforms = self._gravity_chain.forward_kinematics(q)
            potential = torch.zeros(self.env.num_envs, device=self.env.device)
            for name, mass, com in self._gravity_links:
                transform = transforms[name].get_matrix()
                position = transform[:, :3, :3] @ com + transform[:, :3, 3]
                potential = potential - mass * (
                    position * self._gravity_acceleration
                ).sum(dim=-1)
            torque = torch.autograd.grad(potential.sum(), q)[0]
        return torque[:, self._gravity_arm_ids].detach()


class AssemblyJointPositionAction(RelativeJointPositionAction):
    """Own policy inputs, arm drive and held gripper command.

    Preparation rejects additional terms that could write these resources.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        masked = actions.clone()
        masked[
            self._env._elapsed_steps < int(getattr(self._env, "grasp_warmup_steps", 10))
        ] = 0
        super().process_actions(masked)

    def apply_actions(self) -> None:
        state = self._env.assembly_state
        self._env.robot.set_qf(state.controller.torques(), name="arm")
        self._env.robot.set_qpos(
            torch.full(
                (self._env.num_envs, len(self._env.robot.get_joint_ids(name="eef"))),
                state.scene._grasp_target_joint,
                device=self._env.device,
            ),
            name="eef",
        )
        super().apply_actions()
