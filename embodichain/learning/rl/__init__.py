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

"""On-policy reinforcement learning pipeline.

Algorithms (PPO/GRPO), rollout buffers, collectors, policy/model builders, and the training entry point; rollout data flows as ``TensorDict`` objects.
"""

from . import algo
from . import buffer
from . import models
from . import utils
from .differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from .env import DifferentiableObservation, DifferentiableVecEnv

__all__ = [
    "DifferentiableObservation",
    "DifferentiableTrainer",
    "DifferentiableTrainerCfg",
    "DifferentiableVecEnv",
    "algo",
    "buffer",
    "models",
    "utils",
]
