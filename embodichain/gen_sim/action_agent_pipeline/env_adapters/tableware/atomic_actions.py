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

from typing import Any

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.base_agent_env import (
    BaseAgentEnv,
)
from embodichain.lab.gym.envs.tasks.tableware.configurable_success import (
    evaluate_configured_success,
)
from embodichain.lab.gym.utils.registration import register_env

__all__ = ["AtomicActionsAgentEnv"]


@register_env("AtomicActionsAgent-v3", max_episode_steps=600)
class AtomicActionsAgentEnv(BaseAgentEnv, EmbodiedEnv):
    """Config-driven agent environment for atomic-action tasks."""

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        if bool(getattr(self, "ignore_terminations_during_agent", False)):
            self.cfg.ignore_terminations = True
        super()._init_agents(**kwargs)

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info

    def update_obj_info(self) -> None:
        super().update_obj_info()
        self._apply_grasp_pose_overrides()

    def is_task_success(self, **kwargs) -> torch.Tensor:
        return evaluate_configured_success(self)

    def compute_task_state(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, dict]:
        success = self.is_task_success()
        fail = torch.zeros_like(success)
        return success, fail, {}

    def _apply_grasp_pose_overrides(self) -> None:
        overrides = list(getattr(self, "agent_grasp_pose_overrides", []) or [])
        if bool(getattr(self, "force_apple_top_down_grasp_pose", False)):
            overrides.append(
                {
                    "type": "top_down",
                    "object": getattr(self, "agent_success_object", "apple"),
                    "height_offset": getattr(
                        self, "apple_top_down_grasp_height_offset", 0.035
                    ),
                }
            )

        for override in overrides:
            if str(override.get("type", "")).lower() != "top_down":
                raise ValueError(f"Unsupported grasp pose override: {override!r}.")
            self._apply_top_down_grasp_pose_override(override)

    def _apply_top_down_grasp_pose_override(self, override: dict[str, Any]) -> None:
        object_name = str(override.get("object", override.get("object_uid")))
        if not hasattr(self, "obj_info") or object_name not in self.obj_info:
            return

        obj = self.sim.get_rigid_object(object_name)
        object_pose = obj.get_local_pose(to_matrix=True)
        if object_pose.ndim == 3:
            object_pose = object_pose.squeeze(0)
        object_pose = object_pose.to(
            dtype=self.init_qpos.dtype, device=self.robot.device
        )

        grasp_pose_world = self._make_top_down_grasp_pose_world(
            object_pose=object_pose,
            height_offset=float(override.get("height_offset", 0.035)),
            side=str(override.get("side", "right")),
        )
        grasp_pose_object = torch.linalg.inv(object_pose) @ grasp_pose_world
        self.obj_info[object_name]["grasp_pose_obj"] = grasp_pose_object
        self.affordance_datas[f"{object_name}_grasp_pose_object"] = (
            grasp_pose_object.unsqueeze(0)
        )

    def _make_top_down_grasp_pose_world(
        self,
        object_pose: torch.Tensor,
        height_offset: float,
        side: str,
    ) -> torch.Tensor:
        object_position = object_pose[:3, 3]
        z_axis = torch.tensor(
            [0.0, 0.0, -1.0], dtype=object_pose.dtype, device=object_pose.device
        )
        x_axis = self._top_down_grasp_x_axis(object_position, side=side)
        y_axis = torch.cross(z_axis, x_axis, dim=0)
        y_axis = y_axis / torch.linalg.norm(y_axis).clamp_min(1e-6)

        grasp_pose = torch.eye(4, dtype=object_pose.dtype, device=object_pose.device)
        grasp_pose[:3, 0] = x_axis
        grasp_pose[:3, 1] = y_axis
        grasp_pose[:3, 2] = z_axis
        grasp_pose[:3, 3] = object_position + torch.tensor(
            [0.0, 0.0, height_offset],
            dtype=object_pose.dtype,
            device=object_pose.device,
        )
        return grasp_pose

    def _top_down_grasp_x_axis(
        self, object_position: torch.Tensor, side: str
    ) -> torch.Tensor:
        base_pose = getattr(self, f"{side}_arm_base_pose", None)
        if base_pose is None:
            base_pose = getattr(self, "right_arm_base_pose", None)
        if base_pose is None:
            base_pose = getattr(self, "left_arm_base_pose", None)

        if base_pose is not None:
            base_pose = torch.as_tensor(
                base_pose, dtype=object_position.dtype, device=object_position.device
            )
            base_position = base_pose[:3, 3] if base_pose.ndim == 2 else base_pose[:3]
            x_axis = object_position - base_position
            x_axis = x_axis.clone()
            x_axis[2] = 0.0
        else:
            x_axis = torch.tensor(
                [1.0, 0.0, 0.0],
                dtype=object_position.dtype,
                device=object_position.device,
            )

        if torch.linalg.norm(x_axis) < 1e-6:
            x_axis = torch.tensor(
                [1.0, 0.0, 0.0],
                dtype=object_position.dtype,
                device=object_position.device,
            )
        return x_axis / torch.linalg.norm(x_axis).clamp_min(1e-6)
