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

"""Discount-consistent reward transforms independent of simulation and policies."""

from __future__ import annotations

import math
import torch

__all__ = ["potential_shaping", "aggregate_reward_sequence"]


def _discount(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("discount must be finite and between zero and one")


def potential_shaping(
    previous: torch.Tensor,
    current: torch.Tensor,
    terminated: torch.Tensor,
    *,
    discount: float,
) -> torch.Tensor:
    """Compute shaping at the learner's actual terminal boundary.

    Args:
        previous: Previous potential, shaped [batch].
        current: Pre-reset successor potential with matching dtype/device.
        terminated: True absorbing terminal mask, not a time-limit mask.
        discount: Discount per executed control step.

    Returns:
        Owned shaping rewards. Skill completion inside a continuing task must
        not be supplied as an episode termination.
    """
    _discount(discount)
    if previous.ndim != 1 or not previous.is_floating_point():
        raise ValueError("potentials must be floating point batch vectors")
    if (
        current.shape != previous.shape
        or current.dtype != previous.dtype
        or current.device != previous.device
    ):
        raise ValueError("potentials must share shape, dtype and device")
    if (
        terminated.shape != previous.shape
        or terminated.dtype != torch.bool
        or terminated.device != previous.device
    ):
        raise ValueError("terminated must be a boolean vector on the potential device")
    return discount * torch.where(terminated, 0.0, current) - previous


def aggregate_reward_sequence(
    rewards: torch.Tensor,
    valid: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    discount: float,
    episode_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce the first contiguous executed episode prefix of each action chunk.

    Args:
        rewards: Control-step rewards, shaped [batch, time].
        valid: Executed-step mask. A false slot closes the prefix for that row.
        terminated: Absorbing terminal flags per control step.
        truncated: Administrative time-limit flags per control step.
        discount: Discount per control step, matching the consuming learner.
        episode_ids: Optional integer episode identities, shaped [batch, time].
            Selected prefixes crossing an episode boundary reject.

    Returns:
        Discounted returns, continuation discounts and executed lengths. Both
        end flags close a prefix, but only termination suppresses bootstrap.
        Empty prefixes return zero reward, unit continuation and zero length.
        The caller retains the final pre-reset observation for bootstrap.
    """
    _discount(discount)
    if rewards.ndim != 2 or not rewards.is_floating_point():
        raise ValueError("rewards must be a floating point [batch, time] tensor")
    for mask in (valid, terminated, truncated):
        if (
            mask.shape != rewards.shape
            or mask.dtype != torch.bool
            or mask.device != rewards.device
        ):
            raise ValueError("masks must match reward shape/device and be boolean")
    batch, steps = rewards.shape
    if steps == 0:
        return (
            rewards.new_zeros(batch),
            rewards.new_ones(batch),
            torch.zeros(batch, dtype=torch.long, device=rewards.device),
        )
    stops = ~valid | terminated | truncated
    prior_stop = torch.cat((torch.zeros_like(stops[:, :1]), stops[:, :-1]), dim=1)
    selected = valid & (prior_stop.long().cumsum(dim=1) == 0)
    if episode_ids is not None:
        if (
            episode_ids.shape != rewards.shape
            or episode_ids.device != rewards.device
            or episode_ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "episode_ids must match reward shape/device and be integer"
            )
        if bool((selected & (episode_ids != episode_ids[:, :1])).any()):
            raise ValueError("An executed reward prefix cannot cross episodes")
    lengths = selected.sum(dim=1)
    powers = rewards.new_tensor(discount).pow(
        torch.arange(steps, device=rewards.device)
    )
    total = (torch.where(selected, rewards, 0.0) * powers).sum(dim=1)
    terminal = (terminated & selected).any(dim=1)
    continuation = torch.where(terminal, 0.0, rewards.new_tensor(discount).pow(lengths))
    return total, continuation, lengths
