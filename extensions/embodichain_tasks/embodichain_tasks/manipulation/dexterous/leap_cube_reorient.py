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
from embodichain.learning.rl.env import register_learning_env
from ._runtime import DexSimManipulationEnv

__all__ = ["make_env"]


@register_learning_env("LeapCubeReorientDexSim-v1")
def make_env(
    *, num_envs: int, device: torch.device | str, **cfg: Any
) -> DexSimManipulationEnv:
    """Create the leap cube reorient task with the selected DexSim solver."""
    return DexSimManipulationEnv(task_id="M4", num_envs=num_envs, device=device, **cfg)
