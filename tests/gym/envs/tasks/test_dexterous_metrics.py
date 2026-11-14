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

from contextlib import nullcontext
import math
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import Mock, call

import gymnasium as gym
import numpy as np
import pytest
from tensordict import TensorDict
import torch

from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.learning.rl.evaluation import evaluate_episodes
from embodichain_tasks.manipulation.dexterous._runtime import (
    DexterousManipulationEnv,
    TASK_SPECS,
)
from embodichain_tasks.manipulation.dexterous._shadow_task import ShadowState


class _Policy(torch.nn.Module):
    obs_dim = 1
    action_dim = 1
    actor_obs_groups = ("policy",)

    def __init__(self, action_dim: int = 1) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def get_action(self, data: Any, deterministic: bool = False) -> Any:
        del deterministic
        data["action"] = torch.zeros(data.batch_size[0], self.action_dim)
        return data


@pytest.fixture
def env_factory() -> Callable[[str], DexterousManipulationEnv]:
    """Build a lightweight CPU DexterousManipulationEnv exercising real step/reset hooks."""

    def make(task_id: str) -> DexterousManipulationEnv:
        spec = TASK_SPECS[task_id]
        env = DexterousManipulationEnv.__new__(DexterousManipulationEnv)
        env.task_id = task_id
        env._spec = spec
        env._device = torch.device("cpu")
        env._num_envs = 2
        env.max_episode_steps = spec.max_episode_steps
        env.action_dim = spec.action_dim
        env.policy_dim = spec.policy_dim
        env.critic_dim = spec.critic_dim
        env.single_action_space = gym.spaces.Box(
            -1.0, 1.0, (spec.action_dim,), dtype=np.float32
        )
        env.single_observation_space = gym.spaces.Dict(
            {
                "policy": gym.spaces.Box(
                    -np.inf, np.inf, (spec.policy_dim,), dtype=np.float32
                ),
                "critic": gym.spaces.Box(
                    -np.inf, np.inf, (spec.critic_dim,), dtype=np.float32
                ),
            }
        )
        env.sim_cfg = SimpleNamespace(physics_dt=spec.sim_dt)
        env.cfg = SimpleNamespace(
            seed=42,
            ignore_terminations=False,
            episode_context_in_info=False,
            sim_steps_per_control=round(spec.ctrl_dt / spec.sim_dt),
            events=None,
            observations=None,
            rewards=None,
            actions=None,
            dataset=None,
        )
        env.episode_context = EpisodeContext(2)
        env.episode_success_status = torch.zeros(2, dtype=torch.bool)
        env._profiler = SimpleNamespace(section=lambda *args, **kwargs: nullcontext())
        env._elapsed_steps = torch.zeros(2, dtype=torch.long)
        env._detached_uids_for_reset = ["hand", "obj"]
        env.sensors = {}
        env.event_manager = None
        env.observation_manager = None
        env.action_manager = None
        env.reward_manager = None
        env.dataset_manager = None
        env.rollout_buffer = None
        env._traj_buffer = None
        env._is_Keyboard = False

        env.sim = SimpleNamespace(
            device=torch.device("cpu"),
            is_rt_enabled=False,
            reset_objects_state=Mock(),
            clear_articulation_dynamics=Mock(),
            update=Mock(),
            render_frame=lambda **kwargs: nullcontext(),
            arena_offsets=torch.zeros(2, 3),
        )
        hand_dof = 24 if env.is_shadow else 16
        act_count = 20 if env.is_shadow else 16
        env._actuated_joint_ids = list(range(act_count))
        env._actuated_joint_ids_tensor = torch.arange(act_count, dtype=torch.long)
        env._kp = torch.ones(act_count)
        env._kd = torch.zeros(act_count)
        env._ctrl = torch.zeros(2, act_count)
        env._actuator_force = torch.zeros(2, act_count)
        env._raw_action = torch.zeros(2, spec.action_dim)
        env._policy_obs = torch.zeros(2, spec.policy_dim)
        env._critic_obs = torch.zeros(2, spec.critic_dim)
        env._step_reward = torch.zeros(2)
        env._step_terminated = torch.zeros(2, dtype=torch.bool)
        env._step_truncated = torch.zeros(2, dtype=torch.bool)
        env._episode_success = torch.zeros(2, dtype=torch.bool)
        env._step_metrics = {}
        env._reset_generator = torch.Generator().manual_seed(0)

        env.robot = SimpleNamespace(
            uid="hand",
            get_qpos=lambda: torch.zeros(2, hand_dof),
            get_qvel=lambda: torch.zeros(2, hand_dof),
            get_qf=lambda: torch.zeros(2, hand_dof),
            set_qpos=Mock(),
            set_qvel=Mock(),
            set_qf=Mock(),
        )
        env._object = SimpleNamespace(
            set_local_pose=Mock(),
            set_velocity=Mock(),
            get_local_pose=lambda: torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2
            ),
        )
        env._goal_obj = None
        env._cube_art = (
            SimpleNamespace(
                set_local_pose=Mock(),
                set_root_velocity=Mock(),
                set_qpos=Mock(),
                set_qvel=Mock(),
                set_qf=Mock(),
                get_qpos=lambda: torch.zeros(2, 1),
            )
            if task_id == "M3"
            else None
        )
        env._face_joint_id = 0

        if env.is_shadow:
            env._read_physical_state = lambda: ShadowState(
                torch.zeros(2, 31), torch.zeros(2, 30), torch.zeros(2, 3)
            )
        else:
            env._read_physical_state = lambda: {"qpos": torch.zeros(2, 24)}

        def task_reset(ids, *, generator=None):
            n = len(ids)
            if env.is_shadow:
                assert generator is env._reset_generator
                qpos = torch.zeros(n, 31)
                qpos[:, 27] = 1.0
                return qpos, torch.zeros(n, 30), torch.zeros(n, act_count)
            q_len = 24 if task_id == "M3" else 23
            qpos = torch.zeros(n, q_len)
            qpos[:, 19] = 1.0
            return {
                "qpos": qpos,
                "qvel": torch.zeros(n, q_len - 1),
                "ctrl": torch.zeros(n, act_count),
            }

        def observe(state, env_ids=None):
            del state
            if env.is_shadow:
                return torch.zeros(2, spec.policy_dim)
            n = 2 if env_ids is None else len(env_ids)
            return {
                "state": torch.zeros(n, spec.policy_dim),
                "privileged_state": torch.zeros(n, spec.critic_dim),
            }

        def step_fn(state, actions):
            del state, actions
            # Environment 0 succeeds on step 0 and terminates after 3 steps;
            # environment 1 never succeeds and terminates after 4 steps.
            success = (env._elapsed_steps == 0) & torch.tensor([True, False])
            done = env._elapsed_steps >= torch.tensor([2, 3])
            metrics = {"rotation": env._elapsed_steps.float()}
            if task_id != "M5":
                metrics["success"] = success.float()
            if env.is_shadow:
                return (
                    torch.zeros(2, spec.policy_dim),
                    torch.ones(2),
                    done,
                    metrics,
                )
            return (
                {
                    "state": torch.zeros(2, spec.policy_dim),
                    "privileged_state": torch.zeros(2, spec.critic_dim),
                },
                torch.ones(2),
                done,
                torch.zeros(2, dtype=torch.bool),
                metrics,
            )

        env.task = SimpleNamespace(
            reset=task_reset,
            observe=observe,
            action_targets=lambda x: torch.zeros(x.shape[0], act_count),
            control_targets=lambda x: torch.zeros(x.shape[0], act_count),
            face_detent=lambda a: torch.zeros_like(a),
            step=step_fn,
            after_step=step_fn,
        )
        env.reset(seed=42)
        return env

    return make


@pytest.mark.no_sim
@pytest.mark.parametrize("task_id", ["M1", "M2", "M3", "M4"])
def test_success_survives_until_autoreset_and_reaches_evaluation(
    env_factory: Callable[[str], DexterousManipulationEnv], task_id: str
) -> None:
    """An early success pulse persists across the episode until reset."""
    env = env_factory(task_id)
    result = evaluate_episodes(
        policy=_Policy(env.action_dim),
        env=env,
        num_episodes=2,
        device="cpu",
    )
    assert result["eval/success_rate"] == 0.5
    assert result["eval/metrics/success"] == 0.0
    assert result["eval/avg_length"] == 3.5


@pytest.mark.no_sim
@pytest.mark.parametrize("task_id", ["M1", "M3"])
def test_partial_reset_clears_only_selected_success_and_keeps_terminal_info(
    env_factory: Callable[[str], DexterousManipulationEnv], task_id: str
) -> None:
    """Selective reset clears only target rows and rewinds the reset generator on explicit seed."""
    env = env_factory(task_id)
    env._episode_success[:] = True
    trainer_rng = torch.random.get_rng_state()
    env.reset(seed=19, options={"reset_ids": torch.tensor([0])})
    assert env._reset_generator.initial_seed() == 19
    assert torch.equal(torch.random.get_rng_state(), trainer_rng)
    assert env._episode_success.tolist() == [False, True]
    env._elapsed_steps[:] = torch.tensor([2, 1])
    _, _, terminated, _, info = env.step(torch.zeros(2, env.action_dim))
    assert terminated.tolist() == [True, False]
    assert info["success"].tolist() == [False, True]
    env.reset()
    assert not env._episode_success.any()
    assert info["success"].tolist() == [False, True]


@pytest.mark.no_sim
def test_continuous_rotation_has_metrics_without_invented_binary_success(
    env_factory: Callable[[str], DexterousManipulationEnv],
) -> None:
    """M5 omits binary episode success while reporting continuous rotation metrics."""
    env = env_factory("M5")
    result = evaluate_episodes(
        policy=_Policy(env.action_dim),
        env=env,
        num_episodes=2,
        device="cpu",
    )
    assert math.isnan(result["eval/success_rate"])
    assert math.isfinite(result["eval/metrics/rotation"])


@pytest.mark.no_sim
@pytest.mark.parametrize("task_id", list(TASK_SPECS))
@pytest.mark.parametrize("ignore_terminations", [False, True])
def test_time_limit_respects_ignored_terminations(
    env_factory: Callable[[str], DexterousManipulationEnv],
    task_id: str,
    ignore_terminations: bool,
) -> None:
    """Only effective failures suppress time-limit resets and bootstrapping."""
    from embodichain.learning.rl.algo.common import compute_gae

    env = env_factory(task_id)
    env.cfg.ignore_terminations = ignore_terminations
    env._elapsed_steps.fill_(env.max_episode_steps - 1)
    env.task.step = lambda state, actions: (
        torch.zeros(2, env.policy_dim),
        torch.ones(2),
        torch.tensor([True, False]),
        {},
    )
    env.task.after_step = lambda state, actions: (
        {
            "state": torch.zeros(2, env.policy_dim),
            "privileged_state": torch.zeros(2, env.critic_dim),
        },
        torch.ones(2),
        torch.tensor([True, False]),
        torch.zeros(2, dtype=torch.bool),
        {},
    )
    _, reward, terminated, truncated, _ = env.step(torch.zeros(2, env.action_dim))
    assert terminated.tolist() == [not ignore_terminations, False]
    assert truncated.tolist() == [ignore_terminations, True]
    assert env._elapsed_steps.tolist() == [0, 0]
    rollout = TensorDict(
        {
            "value": torch.full((2, 2), 10.0),
            "reward": torch.stack((reward, torch.zeros(2)), dim=1),
            "done": torch.stack(
                (terminated | truncated, torch.zeros(2, dtype=torch.bool)), dim=1
            ),
            "truncated": torch.stack(
                (truncated, torch.zeros(2, dtype=torch.bool)), dim=1
            ),
        },
        batch_size=[2, 2],
    )
    _, returns = compute_gae(rollout, gamma=0.9, gae_lambda=0.95)
    expected = torch.tensor([10.0 if ignore_terminations else 1.0, 10.0])
    torch.testing.assert_close(returns[:, 0], expected)


@pytest.mark.no_sim
@pytest.mark.parametrize("task_id", list(TASK_SPECS))
def test_physics_samples_opted_in_sensors(
    env_factory: Callable[[str], DexterousManipulationEnv], task_id: str
) -> None:
    """Custom stepping samples each physics interval, including batched updates."""
    env = env_factory(task_id)
    sensor = Mock(requires_substep_update=True)
    passive = Mock(requires_substep_update=False)
    env.sensors = {"contacts": sensor, "camera": passive}

    def advance(dt, count, *, after_substep=None, **kwargs):
        for _ in range(count):
            if after_substep is not None:
                after_substep(dt)

    env.sim.update = advance
    env._advance_physics()
    sensor.begin_control_step.assert_called_once_with()
    assert (
        sensor.update_physics_step.call_args_list
        == [call(env.physics_dt)] * env.cfg.sim_steps_per_control
    )
    passive.begin_control_step.assert_not_called()
    passive.update_physics_step.assert_not_called()


@pytest.mark.no_sim
def test_face_turn_freejoint_velocity_uses_body_origin(
    env_factory: Callable[[str], DexterousManipulationEnv],
) -> None:
    """Convert COM velocity using each root's orientation without mutating state."""
    env = env_factory("M3")
    env._cube_root_com = torch.tensor([[0.0, 0.0, -0.01]] * 2)
    half_angle = math.sqrt(0.5)
    # Row 1 turns the local -Z COM offset into world -X by rotating about Y.
    env._cube_art.get_local_pose = lambda: torch.tensor(
        [[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, half_angle, 0, half_angle]]
    )
    com_velocity = torch.tensor([[0.0, 0.01, 0.0], [0.0, 0.02, 0.0]])
    env._cube_art.body_data = SimpleNamespace(
        root_lin_vel=com_velocity.clone(),
        root_ang_vel=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    env._cube_art.get_qvel = lambda: torch.full((2, 1), 0.5)

    state = DexterousManipulationEnv._read_physical_state(env)

    # v_origin = v_com - omega x r; the second cross product is -0.01 Y.
    torch.testing.assert_close(
        state["qvel"][:, 16:19], torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.03, 0.0]])
    )
    torch.testing.assert_close(
        state["qvel"][:, 19:22], torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    )
    torch.testing.assert_close(state["qvel"][:, 22], torch.full((2,), 0.5))
    torch.testing.assert_close(env._cube_art.body_data.root_lin_vel, com_velocity)
