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

"""Original Sharpa physical task boundary for subsequent APG integration.

The pinned upstream environment owns every task term. This collector only
copies measured transitions before selected-row resets; it does not substitute
the approximate Newton scene, privileged actor or reward implementation.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from sharpa_rl_unilab.tasks.sharpa_inhand.teacher_env import SharpaTeacherEnv
from sharpa_rl_unilab.training.supermimic_mdp import (
    SharpaSuperMimicCapture,
    SharpaSuperMimicMDP,
)
from unilab.base.torch_env import TorchEnvState

__all__: list[str] = []


class _AlignedTask:
    """Collect original observations and physical transitions without autoreset."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.env = SharpaTeacherEnv(cfg, int(cfg.training.num_envs), auto_reset=False)
        self.capture = SharpaSuperMimicCapture(self.env)
        self.mdp = SharpaSuperMimicMDP(self.env, cfg.training.device)
        self.transition: dict[str, torch.Tensor] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(
        self, env_indices: torch.Tensor | None = None, *, seed: int | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        observation, info = self.env.reset(env_indices, seed=seed)
        self.capture.refresh(env_indices)
        self.transition = None
        return observation, info

    def step(self, action: torch.Tensor) -> TorchEnvState:
        before = {key: value.detach().clone() for key, value in self.capture().items()}
        state = self.env.step(action)
        following = self.capture()
        # Reset overwrites manager-owned buffers. Copy the old episode now.
        self.transition = {
            "s": before["s"],
            "s1": following["s"].detach().clone(),
            "xi": before["xi"],
            "a": action.detach().clone(),
            "r": state.reward.detach().clone(),
            "terminated": state.terminated.detach().clone(),
            "truncated": state.truncated.detach().clone(),
        }
        return state

    def reset_done(self, state: TorchEnvState) -> dict[str, torch.Tensor]:
        ids = (state.terminated | state.truncated).nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.env.reset(ids)
            self.capture.refresh(ids)
        assert self.env.state is not None
        return self.env.state.obs

    def close(self) -> None:
        self.env.close()
