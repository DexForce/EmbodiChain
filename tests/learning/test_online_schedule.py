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

"""Fractional update ratios, model warmup and schedule restoration."""

from __future__ import annotations

import pytest
from embodichain.learning.dynamics import OnlineLearningCfg, OnlineLearningSchedule


def test_prefill_warmup_and_resumed_fractional_update_credit():
    schedule = OnlineLearningSchedule(
        OnlineLearningCfg(
            warmup_updates=2, updates_per_collect=0.4, model_updates_per_cycle=3
        )
    )
    events = []

    def collect():
        events.append("collect")
        return 8

    def model():
        events.append("model")
        return {"loss": 2.0}

    def policy():
        events.append("policy")
        return {"loss": 3.0}

    schedule.collect(collect, prefill=True)
    with pytest.raises(RuntimeError, match="warmup"):
        schedule.step(collect, model, policy)
    schedule.warmup(model)
    schedule.warmup(model)
    schedule.step(collect, model, policy)
    schedule.step(collect, model, policy)
    assert events == ["collect", "model", "model", "collect", "collect"]
    restored = OnlineLearningSchedule(schedule.cfg)
    restored.load_state_dict(schedule.state_dict())
    events.clear()
    for _ in range(3):
        restored.step(collect, model, policy)
    assert events == [
        "collect",
        "model",
        "model",
        "model",
        "policy",
        "collect",
        "collect",
        "model",
        "model",
        "model",
        "policy",
    ]
    assert (
        restored.iterations == 5
        and restored.policy_updates == 2
        and restored.model_updates == 8
    )
    assert restored.prefill_steps == 1 and restored.physical_transitions == 48
    assert restored.credit == pytest.approx(0.0)
