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

from itertools import product

import torch

from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env_function
from embodichain.lab.task_program.integrations.composition import (
    load_configured_task_program_deployment,
)
from embodichain.utils.utility import load_config
from embodichain_tasks.configs import get_config_path

__all__ = ["DrawerStorageEnv"]


_CONFIG = get_config_path("tasks/manipulation/drawer_storage")


class DrawerStorageEnv(EmbodiedEnv):
    """Task Program execution with independent measured storage acceptance.

    The 50 mm cube and drawer interior are fixed by this task's environment.
    Drawer bounds come from Drawer/model_split_links_with_inertials.urdf:
    side inner faces +/-0.1965, front/back 0.01/0.309 and floor/top -0.057/0.0425.
    One millimetre is allowed for contact resolution at the boundary.
    """

    _hand_joints = ("gripper_finger1_joint_1",)
    _open_hand = (0.0,)
    _finger_links = ("gripper_finger1_link_1", "gripper_finger2_link_1")

    def is_task_success(self, **kwargs: object) -> torch.Tensor:
        """Require all cube corners inside, a closed rail and an empty hand.

        Args:
            **kwargs: Forwarded to the shared Task Program completion gate.

        Returns:
            One measured acceptance value per environment. Task Program segment
            completion and its configured stability windows remain separate gates.
        """
        program_success = super().is_task_success(**kwargs)
        cube = self.sim.get_rigid_object("cube")
        drawer = self.sim.get_articulation("drawer")
        pose = cube.get_local_pose(to_matrix=True)
        tray = drawer.get_link_pose("drawer", to_matrix=True)
        corners = pose.new_tensor(tuple(product((-0.025, 0.025), repeat=3)))
        world = (
            torch.einsum("bij,kj->bki", pose[:, :3, :3], corners) + pose[:, None, :3, 3]
        )
        local = torch.einsum(
            "bji,bkj->bki", tray[:, :3, :3], world - tray[:, None, :3, 3]
        )
        lower = pose.new_tensor((-0.1965, 0.010, -0.057)) - 0.001
        upper = pose.new_tensor((0.1965, 0.309, 0.0425)) + 0.001
        contained = torch.isfinite(local).all(dim=(1, 2)) & (
            (local >= lower) & (local <= upper)
        ).all(dim=(1, 2))
        rail = drawer.get_qpos(target=False)[
            :, drawer.joint_names.index("cabinet_to_drawer")
        ]
        hand_indices = [
            self.robot.joint_names.index(name) for name in self._hand_joints
        ]
        hand = self.robot.get_qpos(target=False)[:, hand_indices]
        released = torch.isfinite(hand).all(dim=1) & (
            (hand - hand.new_tensor(self._open_hand)).abs() <= 0.002
        ).all(dim=1)
        clear = torch.ones_like(contained)
        for name in self._finger_links:
            finger = self.robot.get_link_pose(name, to_matrix=True)
            clear &= torch.isfinite(finger[:, :3, 3]).all(dim=1) & (
                torch.linalg.vector_norm(finger[:, :3, 3] - pose[:, :3, 3], dim=1)
                > 0.08
            )
        return (
            program_success
            & contained
            & torch.isfinite(rail)
            & (rail >= -0.001)
            & (rail <= 0.01)
            & released
            & clear
        )


class _FrankaDrawerStorageEnv(DrawerStorageEnv):
    _hand_joints = ("fr3_finger_joint1", "fr3_finger_joint2")
    _open_hand = (0.04, 0.04)
    _finger_links = ("fr3_leftfinger", "fr3_rightfinger")


def _register_deployment(filename: str, env_cls: type[DrawerStorageEnv]) -> None:
    declaration = load_config(_CONFIG / filename)
    profile = load_config((_CONFIG / declaration["embodiment"]["component"]).resolve())[
        "skill_profile"
    ]
    deployment = load_configured_task_program_deployment(
        task_program=declaration["task_program"],
        skill_profile=profile,
        base_dir=_CONFIG,
    )
    environment = load_config(_CONFIG / declaration["environment"]["component"])
    register_env_function(
        env_cls,
        declaration["id"],
        max_episode_steps=environment["max_episode_steps"],
        task_program_adapter_factory=deployment.integration.adapter_factory,
    )


_register_deployment("task.ur5.yaml", DrawerStorageEnv)
_register_deployment("task.franka.yaml", _FrankaDrawerStorageEnv)
