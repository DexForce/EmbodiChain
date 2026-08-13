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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from embodichain.learning.rl.evaluation import infer_policy_action
from embodichain.learning.rl.motion_policy_evaluation.native_task import (
    EmbodiChainTaskPolicyAdapter,
    _policy_context_from_env,
    evaluate_native_task,
)
from embodichain.learning.rl.runtime import PolicyRuntime
from dexsim.kit.motion_policy import EvaluationFrame, PolicyContext


class Policy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[2.0], [-1.0]]))

    def get_action(
        self,
        tensordict: TensorDict,
        deterministic: bool = False,
    ) -> TensorDict:
        assert deterministic
        tensordict["action"] = tensordict["obs"] @ self.weight
        return tensordict


class ActionManager:
    def __init__(self) -> None:
        self.calls = 0

    def convert_policy_action_to_env_action(self, action: torch.Tensor):
        self.calls += 1
        return action + 0.5


class Environment:
    num_envs = 1
    physics_dt = 0.005
    step_dt = 0.02

    def __init__(self) -> None:
        self.unwrapped = self
        self.cfg = SimpleNamespace(sim_steps_per_control=4)
        self.action_manager = ActionManager()
        self.actions = []
        self.episode_step = 0
        self.reset_seeds = []
        self.closed = 0

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.episode_step = 0
        return self._observation(), {}

    def step(self, action):
        self.actions.append(action.clone())
        self.episode_step += 1
        done = self.episode_step == 2
        return (
            self._observation(),
            torch.tensor([1.25]),
            torch.tensor([done]),
            torch.tensor([False]),
            {"success": torch.tensor([done])},
        )

    def close(self):
        self.closed += 1

    def _observation(self):
        return {
            "policy": torch.tensor(
                [[float(self.episode_step), 1.0]],
                dtype=torch.float32,
            )
        }


class SimulatorEnvironment(Environment):
    def __init__(self) -> None:
        super().__init__()
        self.sim = SimpleNamespace(get_world=lambda: None)
        self.exit_process_values = []

    def close(self, *, exit_process=None):
        self.closed += 1
        self.exit_process_values.append(exit_process)


class ViewerWindow:
    def __init__(self) -> None:
        self.titles = []

    def set_window_title(self, title):
        self.titles.append(title)

    def native(self):
        return self

    def key_state(self, _key):
        return False


class ViewerWorld:
    def __init__(self) -> None:
        self.window = ViewerWindow()
        self.open = True
        self.update_dts = []

    def is_window_initialized(self):
        return self.open

    def get_windows(self):
        return self.window

    def update(self, dt):
        self.update_dts.append(dt)
        self.open = False


class ViewerSimulatorEnvironment(SimulatorEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.world = ViewerWorld()
        self.sim = SimpleNamespace(get_world=lambda: self.world)


def test_native_adapter_uses_shared_deterministic_inference_chain():
    policy = Policy()
    observation = {"policy": torch.tensor([[0.25, 0.75]])}
    expected = infer_policy_action(
        policy,
        observation,
        device="cpu",
        num_envs=1,
    )
    adapter = EmbodiChainTaskPolicyAdapter(policy, torch.device("cpu"), 1)
    adapter.setup(PolicyContext(None, 0.005, 4, 0.02))

    output = adapter.infer(
        EvaluationFrame(
            control_step=0,
            policy_time=0.0,
            simulation_step=0,
            simulation_time=0.0,
            observation=observation,
        )
    )

    assert torch.equal(output.action, expected)
    adapter.close()


def test_lightweight_task_timing_uses_one_environment_step():
    context = _policy_context_from_env(SimpleNamespace(dt=0.125))

    assert context.physics_dt == pytest.approx(0.125)
    assert context.sim_steps_per_control == 1
    assert context.policy_dt == pytest.approx(0.125)


def test_native_task_runs_original_action_conversion_once_per_step():
    env = Environment()
    runtime = PolicyRuntime(
        env=env,
        policy=Policy(),
        device=torch.device("cpu"),
        env_id="ExampleTask",
    )

    result = evaluate_native_task(
        runtime,
        seed=17,
        viewer=False,
        episodes=2,
        control_steps=None,
        duration=None,
    )

    assert result.reason == "episode target reached"
    assert result.control_steps == 4
    assert result.simulation_steps == 16
    assert result.effective_duration == pytest.approx(0.08)
    assert len(result.episodes) == 2
    assert result.metrics == pytest.approx(
        {
            "eval/avg_reward": 2.5,
            "eval/avg_length": 2.0,
            "eval/success_rate": 1.0,
        }
    )
    assert env.action_manager.calls == 4
    assert len(env.actions) == 4
    assert env.reset_seeds == [17, None]
    assert env.closed == 1


def test_native_task_control_limit_can_stop_before_episode_end():
    env = Environment()
    runtime = PolicyRuntime(
        env=env,
        policy=Policy(),
        device=torch.device("cpu"),
        env_id="ExampleTask",
    )

    result = evaluate_native_task(
        runtime,
        seed=1,
        viewer=False,
        episodes=None,
        control_steps=1,
        duration=None,
    )

    assert result.reason == "control steps reached"
    assert result.control_steps == 1
    assert result.episodes == ()


def test_native_task_closes_simulator_without_exiting_process():
    env = SimulatorEnvironment()
    runtime = PolicyRuntime(
        env=env,
        policy=Policy(),
        device=torch.device("cpu"),
        env_id="ExampleSimulatorTask",
    )

    result = evaluate_native_task(
        runtime,
        seed=1,
        viewer=False,
        episodes=None,
        control_steps=1,
        duration=None,
    )

    assert result.reason == "control steps reached"
    assert env.closed == 1
    assert env.exit_process_values == [False]


def test_native_task_pause_waits_for_viewer_close_after_termination():
    env = ViewerSimulatorEnvironment()
    runtime = PolicyRuntime(
        env=env,
        policy=Policy(),
        device=torch.device("cpu"),
        env_id="ExampleViewerTask",
    )

    result = evaluate_native_task(
        runtime,
        seed=1,
        viewer=True,
        episodes=None,
        control_steps=None,
        duration=None,
        termination_behavior="pause",
    )

    assert result.reason == "viewer closed"
    assert result.control_steps == 2
    assert env.world.update_dts == [0.0]


def test_native_task_viewer_requires_a_simulator_environment():
    env = Environment()
    runtime = PolicyRuntime(
        env=env,
        policy=Policy(),
        device=torch.device("cpu"),
        env_id="ExampleTask",
    )

    with pytest.raises(
        ValueError,
        match="requires an EmbodiChain simulator task",
    ):
        evaluate_native_task(
            runtime,
            seed=1,
            viewer=True,
            episodes=None,
            control_steps=None,
            duration=None,
        )

    assert env.closed == 1
