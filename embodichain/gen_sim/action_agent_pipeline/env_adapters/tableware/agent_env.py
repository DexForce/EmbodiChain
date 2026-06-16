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

import torch

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.base_agent_env import (
    BaseAgentEnv,
)
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.success import (
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

    def is_task_success(self, **kwargs) -> torch.Tensor:
        return evaluate_configured_success(self)

    def compute_task_state(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, dict]:
        success = self.is_task_success()
        fail = torch.zeros_like(success)
        return success, fail, {}
