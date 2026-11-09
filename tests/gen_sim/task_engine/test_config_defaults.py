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

import pytest
from embodichain.gen_sim.task_engine.config import (
    TaskEngineExecutionCfg,
    TaskEnginePlanningCfg,
    load_task_engine_config,
)


def test_profile_and_acceptance_defaults_remain_legacy():
    assert TaskEngineExecutionCfg().acceptance_mode == "strict"
    assert TaskEnginePlanningCfg().runtime_profile == "default"
    with pytest.raises(ValueError):
        TaskEngineExecutionCfg(acceptance_mode="anything")
    with pytest.raises(ValueError):
        TaskEngineExecutionCfg(acceptance_mode="experiment", num_envs=2)
    with pytest.raises(ValueError):
        TaskEnginePlanningCfg(runtime_profile="task_special_case")


def test_packaged_defaults_enable_automatic_qualified_policy():
    workflow, planning, execution = load_task_engine_config()
    assert workflow.max_parallel_workers == 1
    assert workflow.max_action_attempts == 1
    assert planning.runtime_profile == "auto"
    assert planning.experimental_open_release is None
    assert execution.acceptance_mode == "experiment"
