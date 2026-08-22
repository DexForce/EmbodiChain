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

"""Tests for explicit Expert Program environment integration hooks."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from embodichain.lab.gym.envs.demo import DemoSegment
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg


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

    def compile_expert_program(self, program):
        self.compiled_input = program
        return self.compiled_program

    def create_expert_program_bridge(self, program):
        self.bridge_input = program
        return self.bridge


def _uninitialized_env(cls: type[EmbodiedEnv], expert_program: object) -> EmbodiedEnv:
    """Create an environment instance without starting simulation."""
    env = object.__new__(cls)
    env.cfg = SimpleNamespace(expert_program=expert_program)
    return env


def test_embodied_env_cfg_disables_expert_program_by_default() -> None:
    """Declarative execution remains an explicit opt-in configuration."""
    cfg = EmbodiedEnvCfg()

    assert cfg.expert_program is None


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


def test_configured_program_requires_explicit_scene_provider_hook() -> None:
    """The base environment never guesses a live scene provider."""
    env = _uninitialized_env(EmbodiedEnv, object())

    with pytest.raises(NotImplementedError, match="explicit scene resolver"):
        env.create_demo_segments()
