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

"""Reward functors for native locomotion environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.sim.types import EnvAction, EnvObs

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

__all__ = [
    "velocity_locomotion_reward",
    "velocity_locomotion_total_reward",
]


def velocity_locomotion_reward(
    env: EmbodiedEnv,
    obs: EnvObs,
    action: EnvAction,
    info: dict[str, Any],
    term: str,
) -> torch.Tensor:
    """Select one named raw reward from the task-owned locomotion MDP.

    Args:
        env: Environment providing the task state and robot.
        obs: Observation-manager input; the task builds its own observation tensors.
        action: Policy actions with one row per environment and one column per controlled joint.
        info: Step information forwarded to the task reward computation.
        term: Raw reward term name to select from the task result.

    Returns:
        The selected raw reward tensor with one value per environment.
    """
    compute = getattr(env, "get_velocity_locomotion_reward_terms", None)
    if not callable(compute):
        raise TypeError(
            "velocity_locomotion_reward requires an environment that "
            "implements get_velocity_locomotion_reward_terms()."
        )
    terms = compute(info)
    try:
        return terms[term]
    except KeyError as exc:
        raise KeyError(
            f"Unknown velocity locomotion reward term {term!r}; "
            f"available terms: {sorted(terms)}"
        ) from exc


def velocity_locomotion_total_reward(
    env: EmbodiedEnv,
    obs: EnvObs,
    action: EnvAction,
    info: dict[str, Any],
) -> torch.Tensor:
    """Return the task-defined weighted reward for one control step.

    Args:
        env: Environment providing the task state and robot.
        obs: Observation-manager input; the task builds its own observation tensors.
        action: Policy actions with one row per environment and one column per controlled joint.
        info: Step information forwarded to the task reward computation.

    Returns:
        The task total reward with one value per environment.
    """
    del obs, action
    compute = getattr(env, "get_velocity_locomotion_reward", None)
    if not callable(compute):
        raise TypeError(
            "velocity_locomotion_total_reward requires an environment that "
            "implements get_velocity_locomotion_reward()."
        )
    return compute(info).total
