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

pytest.importorskip("dexsim.kit.motion_policy.evaluator")

from dexsim.kit.motion_policy import EvaluationFrame, PolicyContext

from embodichain.learning.rl.evaluation import infer_policy_action
from embodichain.learning.rl.policy_evaluation.viewer import (
    EmbodiChainTaskEnvironment,
    EmbodiChainTaskPolicyAdapter,
    evaluate_native_viewer,
)
from embodichain.learning.rl.runtime import PolicyRuntime


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


class Window:
    def __init__(self) -> None:
        self.titles = []
        self.keys = set()

    def set_window_title(self, title):
        self.titles.append(title)

    def native(self):
        return self

    def key_state(self, key):
        return key in self.keys


class World:
    def __init__(self) -> None:
        self.window = Window()
        self.open = True

    def is_window_initialized(self):
        return self.open

    def get_windows(self):
        return self.window


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
        self.world = World()
        self.sim = SimpleNamespace(get_world=lambda: self.world)
        self.actions = []
        self.episode_step = 0
        self.reset_seeds = []
        self.exit_process_values = []

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
            {
                "success": torch.tensor([done]),
                "metrics": {"task_progress": torch.tensor([self.episode_step])},
            },
        )

    def close(self, *, exit_process=None):
        self.exit_process_values.append(exit_process)

    def _observation(self):
        return {
            "policy": torch.tensor(
                [[float(self.episode_step), 1.0]],
                dtype=torch.float32,
            )
        }


def _runtime(env: Environment, policy: Policy | None = None) -> PolicyRuntime:
    return PolicyRuntime(
        env=env,
        policy=policy or Policy(),
        device=torch.device("cpu"),
        env_id="ExampleTask",
    )


def test_viewer_adapter_uses_the_shared_deterministic_inference_chain():
    policy = Policy()
    observation = {"policy": torch.tensor([[0.25, 0.75]])}
    expected = infer_policy_action(policy, observation, device="cpu", num_envs=1)
    adapter = EmbodiChainTaskPolicyAdapter(policy, torch.device("cpu"))
    adapter.setup(PolicyContext(None, 0.005, 4, 0.02))

    output = adapter.infer(EvaluationFrame(0, 0.0, 0.0, 0, observation=observation))

    assert torch.equal(output.action, expected)
    adapter.close()


def test_viewer_reuses_task_actions_resets_and_metrics():
    env = Environment()

    result = evaluate_native_viewer(
        _runtime(env),
        seed=17,
        episodes=2,
        control_steps=None,
        duration=None,
    )

    assert result.control_steps == 4
    assert result.simulation_steps == 16
    assert len(result.episodes) == 2
    assert result.metrics == pytest.approx(
        {
            "reward": 1.25,
            "task_progress": 2.0,
            "eval/avg_reward": 2.5,
            "eval/avg_length": 2.0,
            "eval/success_rate": 1.0,
        }
    )
    assert env.action_manager.calls == 4
    assert env.reset_seeds == [17, None]
    assert env.exit_process_values == [False]


def test_backspace_requests_one_reset_per_key_press():
    from dexsim.types import InputKey

    env = Environment()
    task = EmbodiChainTaskEnvironment(env, seed=1)
    env.world.window.keys.add(InputKey.SCANCODE_BACKSPACE)

    assert task.poll() == "manual reset"
    assert task.poll() is None
    env.world.window.keys.clear()
    assert task.poll() is None
    env.world.window.keys.add(InputKey.SCANCODE_BACKSPACE)
    assert task.poll() == "manual reset"
    task.close()


def test_viewer_closes_resources_when_evaluator_creation_fails(monkeypatch):
    env = Environment()
    policy = Policy()
    monkeypatch.setattr(
        "embodichain.learning.rl.policy_evaluation.viewer.create_motion_policy_evaluator",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("setup failed")),
    )

    with pytest.raises(RuntimeError, match="setup failed"):
        evaluate_native_viewer(
            _runtime(env, policy),
            seed=1,
            episodes=1,
            control_steps=None,
            duration=None,
        )

    assert env.exit_process_values == [False]
    assert policy.training is True
