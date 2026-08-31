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

from __future__ import annotations

from . import algo
from . import buffer
from . import models
from . import utils
from .differentiable_trainer import (
    DifferentiableTrainer,
    DifferentiableTrainerCfg,
)
from .env import (
    DifferentiableObservation,
    DifferentiableRolloutSpec,
    DifferentiableVecEnv,
    LearningVecEnv,
    build_learning_env,
    get_registered_learning_env_names,
    register_learning_env,
    ScheduledDifferentiableVecEnv,
    stratified_rollout_value,
)
from .evaluation import evaluate_episodes
from .gradients import BatchedGradientNormStats, clip_batched_gradient_norm
from .normalization import RunningObservationNormalizer
from .routing import get_trainer_class

__all__ = [
    "DifferentiableObservation",
    "DifferentiableTrainer",
    "DifferentiableTrainerCfg",
    "DifferentiableRolloutSpec",
    "DifferentiableVecEnv",
    "BatchedGradientNormStats",
    "LearningVecEnv",
    "RunningObservationNormalizer",
    "ScheduledDifferentiableVecEnv",
    "build_learning_env",
    "clip_batched_gradient_norm",
    "evaluate_episodes",
    "get_registered_learning_env_names",
    "get_trainer_class",
    "register_learning_env",
    "stratified_rollout_value",
    "algo",
    "buffer",
    "models",
    "utils",
]
