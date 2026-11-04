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

from .cfg import ConditionCfg, SceneEntityCfg
from .manager_base import Functor
from ..objectives.conditions import (
    ConditionResult,
    evaluate_below_height,
    evaluate_near_target,
)

if TYPE_CHECKING:
    from ..embodied_env import EmbodiedEnv
    from embodichain.lab.sim.objects import RigidObject

__all__ = ["object_near_target", "object_below_height"]


def _rigid_object(env: EmbodiedEnv, entity_cfg: SceneEntityCfg) -> RigidObject:
    entity = env.sim.get_rigid_object(entity_cfg.uid)
    if entity is None:
        raise ValueError(f"Condition requires rigid object {entity_cfg.uid!r}.")
    return entity


class object_near_target(Functor):
    """Compare a bound rigid object with a reset-owned virtual target.

    Entity and geometric parameter validation happens at construction, before
    reset publishes the target. Calls only measure the current physical state.

    Args:
        cfg: Condition parameters, including the entity and target attribute.
        env: Owning environment whose reset event publishes the target.
    """

    def __init__(self, cfg: ConditionCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._entity = _rigid_object(env, cfg.params["entity_cfg"])
        key = cfg.params["target_pose_key"]
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("target_pose_key must be a nonempty string.")
        sample = torch.zeros(1, 3)
        evaluate_near_target(
            sample,
            sample,
            cfg.params["tolerance"],
            axes=cfg.params.get("axes", (0, 1, 2)),
            inclusive=cfg.params.get("inclusive", False),
        )

    def __call__(
        self,
        env: EmbodiedEnv,
        entity_cfg: SceneEntityCfg,
        target_pose_key: str,
        tolerance: float,
        axes: Sequence[int] = (0, 1, 2),
        inclusive: bool = False,
    ) -> ConditionResult:
        """Measure distance using the configured axes and comparison boundary.

        Args:
            env: Live environment owning the target attribute.
            entity_cfg: Source rigid object bound at construction.
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
        position = self._entity.get_local_pose(to_matrix=True)[:, :3, 3]
        return evaluate_near_target(
            position, target, tolerance, axes=axes, inclusive=inclusive
        )


class object_below_height(Functor):
    """Check a bound rigid object's measured local height.

    Entity and height validation happens during construction.

    Args:
        cfg: Condition parameters containing the entity and height threshold.
        env: Owning environment.
    """

    def __init__(self, cfg: ConditionCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self._entity = _rigid_object(env, cfg.params["entity_cfg"])
        evaluate_below_height(torch.zeros(1, 3), cfg.params["height"])

    def __call__(
        self, env: EmbodiedEnv, entity_cfg: SceneEntityCfg, height: float
    ) -> ConditionResult:
        """Measure the strict local Z threshold.

        Args:
            env: Live environment.
            entity_cfg: Rigid object bound at construction.
            height: Local Z threshold in metres.

        Returns:
            Measured condition and height diagnostics.
        """
        position = self._entity.get_local_pose(to_matrix=True)[:, :3, 3]
        return evaluate_below_height(position, height)
