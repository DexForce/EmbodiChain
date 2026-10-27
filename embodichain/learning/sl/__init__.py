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

"""Supervised learning (SL) for policies and learned state transitions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .types import LossResult, SupervisedBatch, SupervisedObjective
from .distributed import distributed_run
from .learner import SupervisedLearner
from .sources import SupervisedSource, TensorBatchSource
from .trainer import SupervisedTrainer, SupervisedTrainerCfg

if TYPE_CHECKING:
    from .sequence import SequenceBatchSource

__all__ = [
    "LossResult",
    "SupervisedBatch",
    "SupervisedObjective",
    "SupervisedLearner",
    "SupervisedSource",
    "TensorBatchSource",
    "SupervisedTrainer",
    "SupervisedTrainerCfg",
    "distributed_run",
    "SequenceBatchSource",
]


def __getattr__(name: str) -> Any:
    if name != "SequenceBatchSource":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.sequence"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
