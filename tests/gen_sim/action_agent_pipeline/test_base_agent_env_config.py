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

from unittest.mock import Mock

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


def test_agentic_gen_sim_env_rejects_missing_agent_sections() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)

    with pytest.raises(ValueError, match="missing required sections: CompileAgent"):
        env._init_agents(
            {
                "Agent": {"prompt_kwargs": {}},
                "TaskAgent": {},
            },
            task_name="UnitTask",
        )


def test_agentic_gen_sim_env_rejects_batched_state_init() -> None:
    class BatchedAgenticGenSimEnv(AgenticGenSimEnv):
        @property
        def num_envs(self):
            return 2

    env = BatchedAgenticGenSimEnv.__new__(BatchedAgenticGenSimEnv)

    with pytest.raises(ValueError, match="supports num_envs=1 only"):
        env.get_states()


def test_agentic_gen_sim_env_draws_arrangement_target_and_high_markers() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)
    env.sim = Mock()
    env.arrangement_debug = {
        "slots": [
            {
                "target": [0.0, 0.2, 0.5],
                "high": [0.0, 0.2, 0.7],
            }
        ]
    }

    env._draw_arrangement_debug_markers()

    assert env.sim.draw_marker.call_count == 2
    target_cfg = env.sim.draw_marker.call_args_list[0].args[0]
    high_cfg = env.sim.draw_marker.call_args_list[1].args[0]
    assert target_cfg.axis_xpos[0, :3, 3].tolist() == pytest.approx([0.0, 0.2, 0.5])
    assert high_cfg.axis_xpos[0, :3, 3].tolist() == pytest.approx([0.0, 0.2, 0.7])
