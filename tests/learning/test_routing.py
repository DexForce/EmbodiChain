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

from __future__ import annotations

import torch
import torch.nn as nn

from embodichain.learning.rl.algo import APG, APGCfg, PPO, PPOCfg, RolloutKind
from embodichain.learning.rl.differentiable_trainer import DifferentiableTrainer
from embodichain.learning.rl.models import ActorCritic, ActorOnly
from embodichain.learning.rl.routing import get_trainer_class
from embodichain.learning.rl.utils.trainer import Trainer


def test_get_trainer_class_routes_by_rollout_kind() -> None:
    actor = nn.Linear(2, 1)
    apg_policy = ActorOnly(2, 1, torch.device("cpu"), actor=actor)
    ppo_policy = ActorCritic(
        2,
        1,
        torch.device("cpu"),
        actor=nn.Linear(2, 1),
        critic=nn.Linear(2, 1),
    )
    apg = APG(APGCfg(device="cpu"), apg_policy)
    ppo = PPO(PPOCfg(device="cpu"), ppo_policy)

    assert apg.rollout_kind is RolloutKind.DIFFERENTIABLE
    assert ppo.rollout_kind is RolloutKind.STANDARD
    assert get_trainer_class(apg) is DifferentiableTrainer
    assert get_trainer_class(ppo) is Trainer
    assert get_trainer_class(apg) is get_trainer_class(
        type("Algo", (), {"rollout_kind": "differentiable"})()
    )
