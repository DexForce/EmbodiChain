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

"""RL helper utilities: algorithm config, optimizers, and observation helpers."""

from .config import AlgorithmCfg, LRSchedulerCfg, OptimizerCfg
from .helper import dict_to_tensordict, flatten_dict_observation
from .optimizer import (
    bind_scheduler_horizon,
    build_lr_scheduler,
    build_optimizer,
    coerce_lr_scheduler_cfg,
    coerce_optimizer_cfg,
    get_registered_lr_scheduler_names,
    get_registered_optimizer_names,
    scheduler_needs_horizon,
)

__all__ = [
    "AlgorithmCfg",
    "LRSchedulerCfg",
    "OptimizerCfg",
    "bind_scheduler_horizon",
    "build_lr_scheduler",
    "build_optimizer",
    "coerce_lr_scheduler_cfg",
    "coerce_optimizer_cfg",
    "dict_to_tensordict",
    "flatten_dict_observation",
    "get_registered_lr_scheduler_names",
    "get_registered_optimizer_names",
    "scheduler_needs_horizon",
]
