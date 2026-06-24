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

from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.agent_env import (
    AgenticGenSimEnv,
)


def test_agentic_gen_sim_env_rejects_reserved_agent_config_keys() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)

    with pytest.raises(ValueError, match="reserved keys: task_name"):
        env._validate_agent_config_keys("TaskAgent", {"task_name": "bad"})


def test_agentic_gen_sim_env_rejects_reserved_common_agent_config_keys() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)
    agent_config = {
        "Agent": {"prompt_kwargs": {}, "task_name": "bad"},
        "TaskAgent": {},
        "CompileAgent": {},
    }

    with pytest.raises(
        ValueError, match="Agent config contains reserved keys: task_name"
    ):
        env._init_agents(agent_config, task_name="UnitTask")
