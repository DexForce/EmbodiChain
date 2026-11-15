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

"""Reference-relative position commands for motion-tracking tasks."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from embodichain.lab.gym.envs.managers.actions import DefaultJointPositionAction

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.managers.cfg import ActionTermCfg
    from embodichain_tasks.locomotion.tracking.g1_motion import UnitreeG1MotionEnv
__all__ = ["ReferenceJointPositionAction"]


class ReferenceJointPositionAction(DefaultJointPositionAction):
    """Combine policy residuals with each joint's configured reference weight."""

    contract_id = None
    representation = "reference_joint_position"
    normalization = "none"

    def __init__(self, cfg: ActionTermCfg, env: UnitreeG1MotionEnv) -> None:
        super().__init__(cfg, env)
        self.setpoint_actions = torch.zeros_like(self.raw_actions)
        self._reference_offset = torch.ones(1, dtype=torch.long, device=env.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._store_raw(actions)
        env = self._env
        reference = env.motion.future(env.motion_frame, self._reference_offset)
        setpoint, target = env.task.control(
            self.raw_actions, reference["q"][:, 0], self.position_bias
        )
        self.setpoint_actions.copy_(setpoint)
        self._processed_actions.copy_(target)

    def apply_actions(self) -> None:
        """Write residual PD targets with the same range used by the model."""
        self._env.robot.set_qpos(
            self._processed_actions,
            joint_ids=list(self._joint_ids),
            clamp_target_qpos=False,
        )

    def reset(self, env_ids: list[int] | torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.setpoint_actions[slice(None) if env_ids is None else env_ids] = 0
