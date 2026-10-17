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

from importlib import import_module
from typing import Any

# Keep lightweight APG imports independent of the simulator-backed Trainer.
_LAZY_EXPORTS = {
    "algo": ("algo", None),
    "buffer": ("buffer", None),
    "models": ("models", None),
    "utils": ("utils", None),
    "DifferentiableTrainer": ("differentiable_trainer", "DifferentiableTrainer"),
    "DifferentiableTrainerCfg": ("differentiable_trainer", "DifferentiableTrainerCfg"),
    "DifferentiableObservation": ("env", "DifferentiableObservation"),
    "DifferentiableRolloutSpec": ("env", "DifferentiableRolloutSpec"),
    "DifferentiableVecEnv": ("env", "DifferentiableVecEnv"),
    "LearningVecEnv": ("env", "LearningVecEnv"),
    "build_learning_env": ("env", "build_learning_env"),
    "get_registered_learning_env_names": ("env", "get_registered_learning_env_names"),
    "register_learning_env": ("env", "register_learning_env"),
    "ScheduledDifferentiableVecEnv": ("env", "ScheduledDifferentiableVecEnv"),
    "stratified_rollout_value": ("env", "stratified_rollout_value"),
    "evaluate_episodes": ("evaluation", "evaluate_episodes"),
    "BatchedGradientNormStats": ("gradients", "BatchedGradientNormStats"),
    "clip_batched_gradient_norm": ("gradients", "clip_batched_gradient_norm"),
    "RunningObservationNormalizer": ("normalization", "RunningObservationNormalizer"),
    "get_trainer_class": ("routing", "get_trainer_class"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        path, symbol = _LAZY_EXPORTS[name]
        module = import_module(f"{__name__}.{path}")
        value = module if symbol is None else getattr(module, symbol)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
