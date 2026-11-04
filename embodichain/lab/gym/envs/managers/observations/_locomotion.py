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

"""Observation functors for native locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.types import EnvObs

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = ["velocity_locomotion_observation"]


def velocity_locomotion_observation(
    env: EmbodiedEnv,
    obs: EnvObs,
    privileged: bool = False,
    enable_corruption: bool = False,
) -> torch.Tensor:
    """Return the task-owned actor or privileged critic observation.

    Args:
        env: Environment providing the task state and robot.
        obs: Observation-manager input; the task builds its own observation tensors.
        privileged: Return the critic observation when true, otherwise the actor observation.
        enable_corruption: Request the task-defined observation noise.

    Returns:
        The selected actor or privileged critic tensor.
    """
    build = getattr(env, "build_velocity_locomotion_observations", None)
    if not callable(build):
        raise TypeError(
            "velocity_locomotion_observation requires an environment that "
            "implements build_velocity_locomotion_observations()."
        )
    actor, critic = build(enable_corruption=enable_corruption)
    return critic if privileged else actor
