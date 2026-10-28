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
"""Reset and reward functors for blind assembly tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .observations import (
    AssemblyProprioception,
    assembly_contact_labels,
    assembly_privileged_state,
)

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.assembly_env import BlindAssemblyEnv

__all__ = [
    "AssemblyProprioception",
    "assembly_privileged_state",
    "assembly_contact_labels",
    "reset_assembly",
    "assembly_milestone_reward",
]


def reset_assembly(env: BlindAssemblyEnv, env_ids: torch.Tensor) -> None:
    """Initialize selected physical grasps without stepping other arenas."""
    env.reset_assembly_rows(env_ids)


def assembly_milestone_reward(
    env: BlindAssemblyEnv, obs: Any, action: torch.Tensor, info: dict[str, Any]
) -> torch.Tensor:
    """Pay each capture/mesh/seat milestone once in the current episode."""
    env.refresh_assembly_state()
    weights = (
        [1.0, 0.0, 1.0, 0.0] if env.assembly_kind == "peg" else [1.0, 1.0, 1.0, 0.0]
    )
    return (
        env._assembly_new_milestones.float() * torch.tensor(weights, device=env.device)
    ).sum(dim=-1)
