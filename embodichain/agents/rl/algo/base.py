# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

from typing import Dict, Any, Callable
import torch
from tensordict import TensorDict


class BaseAlgorithm:
    """Base class for RL algorithms following TorchRL conventions.

    Algorithms implement rollout collection and policy update using TensorDict.
    No custom buffer classes - use TensorDict operations directly.
    """

    device: torch.device

    def collect_rollout(
        self,
        env,
        policy,
        tensordict: TensorDict,
        buffer_size: int,
        on_step_callback: Callable | None = None,
    ) -> TensorDict:
        """Collect rollout and return TensorDict with batch_size=[T, N]."""
        raise NotImplementedError

    def update(self, rollout: TensorDict) -> Dict[str, float]:
        """Update policy using collected data and return training losses."""
        raise NotImplementedError
