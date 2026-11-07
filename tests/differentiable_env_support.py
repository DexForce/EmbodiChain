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

"""Linear CPU environment shared by differentiable protocol and collector tests."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from gymnasium.spaces import Box


class LinearDifferentiableEnv:
    """Minimal CPU environment with differentiable linear dynamics."""

    def __init__(self, num_envs: int = 2) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.single_observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(1,),
            dtype=float,
        )
        self.single_action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=float,
        )
        self._state = torch.ones((num_envs, 1), device=self.device)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del seed, options
        self._state = torch.ones((self.num_envs, 1), device=self.device)
        return self._state, {}

    def step(self, action: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        self._state = self._state + action
        reward = -self._state.square().sum(dim=-1)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        truncated = torch.zeros_like(terminated)
        return self._state, reward, terminated, truncated, {}

    def detach_state(self) -> torch.Tensor:
        self._state = self._state.detach()
        return self._state

    def close(self) -> None:
        return None
