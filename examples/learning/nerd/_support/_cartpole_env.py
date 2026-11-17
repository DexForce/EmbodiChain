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

"""Cartpole factory for the existing lightweight learning-environment route."""

from __future__ import annotations

from collections.abc import Callable
import torch

from embodichain.learning.rl.env import register_learning_env
from ._model_env import NeRDModelEnv

__all__ = ["make_cartpole_dynamics_env"]


@register_learning_env("CartpoleDynamics-v1")
def make_cartpole_dynamics_env(
    num_envs: int,
    device: torch.device | str,
    *,
    checkpoint: str,
    max_episode_steps: int = 500,
    random_reset: bool = True,
    seed: int | None = None,
    physics_profile: str | None = None,
    reset_source: (
        Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
) -> NeRDModelEnv:
    """Construct a Cartpole model task without allocating a World or solver.

    Args:
        num_envs: Number of independent canonical state/history rows.
        device: Policy and transition device.
        checkpoint: Portable NeRD export directory.
        max_episode_steps: Control-step episode horizon.
        random_reset: Whether to perturb initial reference states.
        seed: Private reset-generator seed.
        physics_profile: Optional confirmation of the export's physical profile.
        reset_source: Optional selected-row initial-state provider.

    Returns:
        A DifferentiableEnvProtocol implementation using the shared RL trainers.
    """
    from ._model_factory import make_reference_env

    return make_reference_env(
        "cartpole",
        num_envs,
        device,
        checkpoint=checkpoint,
        max_episode_steps=max_episode_steps,
        random_reset=random_reset,
        seed=seed,
        physics_profile=physics_profile,
        reset_source=reset_source,
    )
