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

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("dexsim.kit.motion_policy.evaluator")

from dexsim.kit.motion_policy import EvaluationFrame, PolicyContext

from embodichain.learning.rl.evaluation import infer_policy_action
from embodichain.learning.rl.policy_evaluation import PolicyViewerCameraCfg
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
        self.camera_pose = np.eye(4)
        self.look_at_calls = []
        self.pan_enabled = True

    def set_look_at(self, eye, target, up):
        self.look_at_calls.append((eye.copy(), target.copy(), up.copy()))
        self.camera_pose[:3, 3] = eye

    def set_camera_orbit_pan_enabled(self, enabled):
        self.pan_enabled = enabled

    def translate_camera_orbit(self, delta):
        self.camera_pose[:3, 3] += delta

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
    assert env.action_manager.calls == 0
    torch.testing.assert_close(env.actions[0], torch.tensor([[-1.0]]))
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


def test_viewer_applies_clamped_native_velocity_command():
    class VelocityEnvironment(Environment):
        def __init__(self) -> None:
            super().__init__()
            self.velocity_commands = []

        def velocity_command_bounds(self):
            return np.array([-1.0, -2.0, -0.3]), np.array([1.0, 2.0, 0.3])

        def set_velocity_command(self, command):
            self.velocity_commands.append(np.asarray(command).copy())

        def get_obs(self):
            return self._observation()

    env = VelocityEnvironment()
    task = EmbodiChainTaskEnvironment(
        env,
        seed=1,
        command=(4.0, -4.0, 0.5),
        keymap="arrows",
    )

    frame = task.reset()

    assert env.velocity_commands[-1].tolist() == pytest.approx([1.0, -2.0, 0.3])
    assert frame.controls["command"].tolist() == pytest.approx([1.0, -2.0, 0.3])
    task.close()


def test_viewer_rejects_commands_for_tasks_without_velocity_interface():
    with pytest.raises(ValueError, match="does not expose a velocity command"):
        EmbodiChainTaskEnvironment(Environment(), seed=1, command=(0.1, 0.0, 0.0))


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


class CameraEnvironment(Environment):
    policy_viewer_camera_cfg = PolicyViewerCameraCfg(
        eye_offset=(-2.0, -1.0, 1.0), target_height=0.5
    )

    def __init__(self):
        super().__init__()
        self.root_pose = np.array([1.0, 2.0, 0.5, 0.0, 0.0, 0.0, 1.0])

    def get_policy_viewer_target_pose(self):
        return self.root_pose.copy()


def _camera_task():
    env = CameraEnvironment()
    task = EmbodiChainTaskEnvironment(env, seed=1)
    task.open_viewer("Tracking test")
    task.reset()
    return env, task, env.world.window


def test_camera_uses_initial_heading_and_fixed_world_height():
    env = CameraEnvironment()
    # A 90-degree initial yaw rotates the rear-quarter (-2, -1) offset to (1, -2).
    env.root_pose[2:] = [3.0, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
    task = EmbodiChainTaskEnvironment(env, seed=1)
    task.open_viewer("Tracking test")
    task.reset()
    eye, target, up = env.world.window.look_at_calls[-1]
    np.testing.assert_allclose(eye, [2.0, 0.0, 1.5], atol=1e-14)
    np.testing.assert_allclose(target, [1.0, 2.0, 0.5])
    np.testing.assert_array_equal(up, [0.0, 0.0, 1.0])
    assert not env.world.window.pan_enabled
    task.close()


def test_camera_follows_xy_and_preserves_manual_orbit_and_zoom():
    env, task, window = _camera_task()
    # Stand in for a user orbit/zoom adjustment after initialization.
    window.camera_pose[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    window.camera_pose[:3, 3] = [8, -3, 7]
    before = window.camera_pose.copy()
    env.root_pose = np.array([1.5, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0])

    task.step(torch.zeros((1, 1)))

    np.testing.assert_array_equal(window.camera_pose[:3, :3], before[:3, :3])
    np.testing.assert_allclose(window.camera_pose[:3, 3] - before[:3, 3], [0.5, -1, 0])
    assert len(window.look_at_calls) == 1
    assert env.action_manager.calls == 0
    task.close()


def test_camera_t_toggle_is_edge_triggered_and_recenters_on_resume():
    from dexsim.types import InputKey

    env, task, window = _camera_task()
    before = window.camera_pose.copy()
    window.keys.add(InputKey.SCANCODE_T)
    assert task.poll() is None
    assert task.poll() is None
    assert window.pan_enabled
    env.root_pose[:3] = [6, 8, 4]
    task.step(torch.zeros((1, 1)))
    np.testing.assert_array_equal(window.camera_pose, before)
    window.keys.clear()
    task.poll()
    window.keys.add(InputKey.SCANCODE_T)
    task.poll()
    assert not window.pan_enabled
    np.testing.assert_allclose(window.look_at_calls[-1][1], [6, 8, 0.5])
    assert len(window.look_at_calls) == 2
    task.close()


def test_camera_reset_restores_tracking_and_new_heading():
    from dexsim.types import InputKey

    env, task, window = _camera_task()
    window.keys.add(InputKey.SCANCODE_T)
    task.poll()
    env.root_pose[:] = [4, 5, 2, 0, 0, 1, 0]
    window.keys = {InputKey.SCANCODE_BACKSPACE}
    assert task.poll() == "manual reset"
    task.reset()
    assert not window.pan_enabled
    np.testing.assert_allclose(window.look_at_calls[-1][0], [6, 6, 1.5])
    env.root_pose[0] += 0.25
    task.step(torch.zeros((1, 1)))
    np.testing.assert_allclose(window.camera_pose[:3, 3], [6.25, 6, 1.5])
    task.close()


def test_camera_is_optional_for_existing_tasks():
    from dexsim.types import InputKey

    env = Environment()
    task = EmbodiChainTaskEnvironment(env, seed=1)
    task.open_viewer("Existing task")
    task.reset()
    env.world.window.keys.add(InputKey.SCANCODE_T)
    task.poll()
    task.step(torch.zeros((1, 1)))
    assert env.world.window.look_at_calls == []
    task.close()


def test_camera_requires_explicit_pose_provider():
    env = Environment()
    env.policy_viewer_camera_cfg = PolicyViewerCameraCfg()
    task = EmbodiChainTaskEnvironment(env, seed=1)
    with pytest.raises(ValueError, match="Environment.*get_policy_viewer_target_pose"):
        task.open_viewer("Invalid target")
    task.close()


@pytest.mark.parametrize("offset", [(0, 0, 1), (float("nan"), 1, 1), (1, 2)])
def test_camera_rejects_invalid_presets(offset):
    env = CameraEnvironment()
    env.policy_viewer_camera_cfg = PolicyViewerCameraCfg(eye_offset=offset)
    task = EmbodiChainTaskEnvironment(env, seed=1)
    with pytest.raises(ValueError, match="eye_offset"):
        task.open_viewer("Invalid preset")
    task.close()


def test_camera_initialization_failure_closes_original_environment():
    env = CameraEnvironment()
    env.root_pose[3:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        evaluate_native_viewer(
            _runtime(env), seed=1, episodes=1, control_steps=None, duration=None
        )
    assert env.exit_process_values == [False]
