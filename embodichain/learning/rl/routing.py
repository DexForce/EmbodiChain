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

"""Trainer routing based on an algorithm's rollout semantics."""

from __future__ import annotations

from typing import Any, TypeAlias

from embodichain.learning.rl.algo import RolloutKind
from embodichain.learning.rl.differentiable_trainer import DifferentiableTrainer
from embodichain.learning.rl.utils.trainer import Trainer

__all__ = ["TrainerType", "get_trainer_class"]

TrainerType: TypeAlias = type[Trainer] | type[DifferentiableTrainer]


def get_trainer_class(algorithm: Any) -> TrainerType:
    """Return the trainer compatible with ``algorithm``."""
    rollout_kind = getattr(algorithm, "rollout_kind", RolloutKind.STANDARD)
    if rollout_kind == RolloutKind.DIFFERENTIABLE:
        return DifferentiableTrainer
    if rollout_kind == RolloutKind.STANDARD:
        return Trainer
    raise ValueError(f"Unsupported rollout kind: {rollout_kind!r}.")
