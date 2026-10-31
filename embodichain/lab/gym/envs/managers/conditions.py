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

"""Configuration-ready manipulation condition functors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .cfg import SceneEntityCfg
from ..objectives.conditions import (
    ConditionResult,
    evaluate_below_height,
    evaluate_near_target,
)

if TYPE_CHECKING:
    from ..embodied_env import EmbodiedEnv

__all__ = ["object_near_target", "object_below_height"]


def _position(env: EmbodiedEnv, entity_cfg: SceneEntityCfg) -> torch.Tensor:
    entity = env.sim.get_rigid_object(entity_cfg.uid)
    if entity is None:
        raise ValueError(f"Condition requires rigid object {entity_cfg.uid!r}.")
    return entity.get_local_pose(to_matrix=True)[:, :3, 3]


def object_near_target(
    env: EmbodiedEnv,
    entity_cfg: SceneEntityCfg,
    target_pose_key: str,
    tolerance: float,
    axes: Sequence[int] = (0, 1, 2),
    inclusive: bool = False,
) -> ConditionResult:
    """Compare an object with a reset-owned virtual target.

    Args:
        env: Live environment owning the target attribute.
        entity_cfg: Bound source rigid object.
        target_pose_key: Attribute published by a reset event.
        tolerance: Positive distance threshold in metres.
        axes: Position axes used for the distance.
        inclusive: Whether equality passes; defaults to a strict bound.

    Returns:
        Measured condition and distance diagnostics.
    """
    target = getattr(env, target_pose_key, None)
    if not isinstance(target, torch.Tensor):
        raise ValueError(
            f"Missing reset target {target_pose_key!r}; configure a reset event before evaluating conditions."
        )
    if target.shape == (env.num_envs, 4, 4):
        target = target[:, :3, 3]
    return evaluate_near_target(
        _position(env, entity_cfg), target, tolerance, axes=axes, inclusive=inclusive
    )


def object_below_height(
    env: EmbodiedEnv, entity_cfg: SceneEntityCfg, height: float
) -> ConditionResult:
    """Check a bound rigid object's measured local height.

    Args:
        env: Live environment.
        entity_cfg: Bound rigid object.
        height: Local Z threshold in metres.

    Returns:
        Measured condition and height diagnostics.
    """
    return evaluate_below_height(_position(env, entity_cfg), height)
