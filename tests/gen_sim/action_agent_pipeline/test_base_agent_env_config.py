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
import torch

from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.agent_env import (
    AgenticGenSimEnv,
)
from embodichain.lab.gym.envs import EmbodiedEnv


class _SingleEnvAgenticGenSimEnv(AgenticGenSimEnv):
    @property
    def num_envs(self):
        return 1

    @property
    def device(self):
        return torch.device("cpu")


def test_agentic_gen_sim_env_rejects_reserved_agent_config_keys() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)

    with pytest.raises(ValueError, match="reserved keys: task_name"):
        env._validate_agent_config_keys("TaskAgent", {"task_name": "bad"})


def test_agentic_gen_sim_env_returns_false_before_runtime_state_is_ready() -> None:
    env = _SingleEnvAgenticGenSimEnv.__new__(_SingleEnvAgenticGenSimEnv)
    env._agent_runtime_state_ready = False
    env.agent_success = {
        "type": "object_lifted",
        "object": "tray",
        "min_height": 0.08,
    }

    success = env.is_task_success()

    assert success.tolist() == [False]


def test_agentic_gen_sim_env_preserves_success_validation_after_runtime_ready() -> None:
    env = _SingleEnvAgenticGenSimEnv.__new__(_SingleEnvAgenticGenSimEnv)
    env._agent_runtime_state_ready = True
    env.agent_success = {
        "type": "object_lifted",
        "object": "tray",
        "min_height": 0.08,
    }
    tray = Mock()
    tray.get_local_pose.return_value = torch.eye(4).unsqueeze(0)
    env.sim = Mock()
    env.sim.get_rigid_object.return_value = tray
    env.obj_info = {}

    with pytest.raises(ValueError, match="requires an initial height"):
        env.is_task_success()


def test_agentic_gen_sim_env_scopes_arm_ik_to_requested_env_ids() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)
    env.robot = Mock()
    env.robot.compute_ik.return_value = (
        torch.tensor([True]),
        torch.tensor([[0.1, 0.2]]),
    )
    env.get_agent_arm_control_part = Mock(return_value="left_arm")
    target_xpos = torch.eye(4)
    qpos_seed = torch.tensor([[0.0, 0.0]])

    success, qpos = env.get_arm_ik(
        target_xpos,
        is_left=True,
        qpos_seed=qpos_seed,
        env_ids=[0],
    )

    assert success is True
    assert qpos.tolist() == pytest.approx([0.1, 0.2])
    call_kwargs = env.robot.compute_ik.call_args.kwargs
    assert call_kwargs["name"] == "left_arm"
    assert call_kwargs["pose"] is target_xpos
    assert call_kwargs["joint_seed"] is qpos_seed
    assert call_kwargs["env_ids"] == [0]


def test_agentic_gen_sim_env_reset_latches_success_before_invalidating_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _SingleEnvAgenticGenSimEnv.__new__(_SingleEnvAgenticGenSimEnv)
    env._agent_runtime_state_ready = True
    env.episode_success_status = torch.zeros(1, dtype=torch.bool)
    env.is_task_success = Mock(return_value=torch.tensor([True]))
    env._draw_arrangement_debug_markers = Mock()
    env.get_states = Mock()
    runtime_ready_during_parent_reset = None

    def fake_parent_reset(self, seed=None, options=None):
        nonlocal runtime_ready_during_parent_reset
        runtime_ready_during_parent_reset = self._agent_runtime_state_ready
        return "obs", {}

    monkeypatch.setattr(EmbodiedEnv, "reset", fake_parent_reset)

    obs, info = env.reset()

    assert env.episode_success_status.tolist() == [True]
    assert runtime_ready_during_parent_reset is False
    assert env._agent_runtime_state_ready is True
    assert obs == "obs"
    assert info == {}


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


def test_agentic_gen_sim_env_initializes_batched_state() -> None:
    class BatchedAgenticGenSimEnv(AgenticGenSimEnv):
        @property
        def num_envs(self):
            return 2

    env = BatchedAgenticGenSimEnv.__new__(BatchedAgenticGenSimEnv)
    env.robot = Mock()
    env.robot.control_parts = {
        "left_arm": [0, 1],
        "left_eef": [2],
        "right_arm": [3, 4],
        "right_eef": [5],
    }
    env.robot.get_qpos.return_value = torch.zeros(2, 6)
    env.robot.get_joint_ids.side_effect = lambda name: env.robot.control_parts[name]
    env.robot.compute_fk.side_effect = lambda qpos, **kwargs: (
        torch.eye(4).unsqueeze(0).repeat(qpos.shape[0], 1, 1)
    )
    env.robot.get_control_part_base_pose.return_value = (
        torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    )
    env.agent_open_state = [0.05]
    env.agent_close_state = [0.0]
    env.update_obj_info = Mock()

    env.get_states()

    assert env.init_qpos.shape == (2, 6)
    assert env.left_arm_current_qpos.shape == (2, 2)
    assert env.right_arm_current_qpos.shape == (2, 2)
    assert env.left_arm_current_gripper_state.shape == (2, 1)
    assert env.right_arm_current_gripper_state.shape == (2, 1)


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
