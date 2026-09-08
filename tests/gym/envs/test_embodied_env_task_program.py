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

"""Tests for explicit Task Program environment integration hooks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs.demo import DemoSegment
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.task_program import CompiledTaskProgram


class _FakeBridge:
    """Minimal bridge protocol used by the environment adapter test."""

    def __init__(self, segment: DemoSegment) -> None:
        self._segment = segment
        self.iteration_count = 0

    def iter_segments(self):
        """Yield the configured segment lazily."""
        self.iteration_count += 1
        yield self._segment


class _DeclarativeEnv(EmbodiedEnv):
    """Environment stub with explicit compiler and bridge factories."""

    def compile_task_program(self, program):
        self.compiled_input = program
        return self.compiled_program

    def create_task_program_bridge(self, program):
        self.bridge_input = program
        return self.bridge


def _uninitialized_env(cls: type[EmbodiedEnv], task_program: object) -> EmbodiedEnv:
    """Create an environment instance without starting simulation."""
    env = object.__new__(cls)
    env.cfg = SimpleNamespace(task_program=task_program)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env._num_envs = 2
    env._task_program_adapter = None
    env._active_task_program_bridge = None
    return env


def test_embodied_env_cfg_disables_task_program_by_default() -> None:
    """Declarative execution remains an explicit opt-in configuration."""
    cfg = EmbodiedEnvCfg()

    assert cfg.task_program is None


def test_create_demo_segments_uses_explicit_compiler_and_bridge_hooks() -> None:
    """Configured programs flow through provider and runtime factories lazily."""
    program = object()
    compiled_program = object()
    expected_segment = DemoSegment(actions=(), name="declarative")
    bridge = _FakeBridge(expected_segment)
    env = _uninitialized_env(_DeclarativeEnv, program)
    env.compiled_program = compiled_program
    env.bridge = bridge

    segments = env.create_demo_segments(debug_mode=True)

    assert bridge.iteration_count == 0
    assert tuple(segments) == (expected_segment,)
    assert bridge.iteration_count == 1
    assert env.compiled_input is program
    assert env.bridge_input is compiled_program
    assert env._active_task_program_bridge is bridge


def test_episode_program_takes_precedence_over_static_configuration() -> None:
    """Callers can select a fresh declarative program for one episode."""
    configured_program = object()
    episode_program = object()
    compiled_program = object()
    bridge = _FakeBridge(DemoSegment(actions=(), name="episode"))
    env = _uninitialized_env(_DeclarativeEnv, configured_program)
    env.compiled_program = compiled_program
    env.bridge = bridge

    segments = env.create_demo_segments(task_program=episode_program)

    assert tuple(segments)[0].name == "episode"
    assert env.compiled_input is episode_program
    assert env.bridge_input is compiled_program


def test_episode_compiled_program_bypasses_recompilation() -> None:
    """A trusted MLLM frontend can pass its canonical compiled snapshot."""
    compiled_program = object.__new__(CompiledTaskProgram)
    bridge = _FakeBridge(DemoSegment(actions=(), name="model"))
    env = _uninitialized_env(_DeclarativeEnv, object())
    env.bridge = bridge

    segments = env.create_demo_segments(task_program=compiled_program)

    assert tuple(segments)[0].name == "model"
    assert not hasattr(env, "compiled_input")
    assert env.bridge_input is compiled_program


def test_configured_program_requires_explicit_adapter() -> None:
    """The base environment never guesses a live scene or runtime provider."""
    env = _uninitialized_env(EmbodiedEnv, object())

    with pytest.raises(NotImplementedError, match="task_program_adapter"):
        env.create_demo_segments()


def test_task_program_success_is_false_before_normal_bridge_completion() -> None:
    """A configured program never inherits the base environment's true default."""
    env = _uninitialized_env(EmbodiedEnv, object())

    assert env.is_task_success().tolist() == [False, False]

    env._active_task_program_bridge = SimpleNamespace(program_completed=False)
    assert env.is_task_success().tolist() == [False, False]


def test_task_program_success_uses_the_completed_bridge_mask() -> None:
    """Normal bridge completion publishes its row-local validator acceptance."""
    env = _uninitialized_env(EmbodiedEnv, object())
    env._active_task_program_bridge = SimpleNamespace(
        program_completed=True,
        completion_mask=torch.tensor([True, False]),
    )

    success = env.is_task_success()

    assert success.dtype == torch.bool
    assert success.device == env.device
    assert success.tolist() == [True, False]


def test_dynamic_task_program_success_does_not_require_static_config() -> None:
    """Episode-injected programs publish the same completed bridge result."""
    env = _uninitialized_env(EmbodiedEnv, None)
    env._active_task_program_bridge = SimpleNamespace(
        program_completed=True,
        completion_mask=torch.tensor([False, True]),
    )

    assert env.is_task_success().tolist() == [False, True]
