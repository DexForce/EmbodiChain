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

"""Tests for task-environment seed reset behavior."""

from __future__ import annotations

from embodichain.lab.gym.envs.episode_context import EpisodeContext

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import embodichain.lab.gym.envs.base_env as base_env_module
from embodichain.lab.gym.envs import BaseEnv, EnvCfg, EmbodiedEnv
from embodichain.lab.sim import SimulationManager

# This module calls one unbound manager method on registry stubs; no World is
# constructed. Keep these reset-contract cases in the fast CPU CI lane.
pytestmark = pytest.mark.no_sim


class _ProfilerStub:
    @contextmanager
    def section(self, *args, **kwargs):
        del args, kwargs
        yield


class _ResetEnv(BaseEnv):
    """BaseEnv reset surface without constructing a simulator."""

    def __init__(self, num_envs: int = 1) -> None:
        self.reset_events: list[str] = []
        self.cfg = EnvCfg(seed=None)
        self._num_envs = num_envs
        self.episode_context = EpisodeContext(num_envs)
        self.sim = SimpleNamespace(
            device=torch.device("cpu"),
            is_window_opened=False,
            reset_objects_state=MagicMock(
                side_effect=lambda **_: self.reset_events.append("reset_objects")
            ),
            sync_render_state=MagicMock(
                side_effect=lambda: self.reset_events.append("sync_render")
            ),
            capture_visualization_safely=MagicMock(
                side_effect=lambda **_: self.reset_events.append("capture")
            ),
        )
        self.sim.render_frame = self._render_frame
        self._profiler = _ProfilerStub()
        self._task_success = torch.zeros(num_envs, dtype=torch.bool)
        self._detached_uids_for_reset: list[str] = []
        self._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)
        self.event_manager = MagicMock()
        self.initial_random_value = 0.0

    @contextmanager
    def _render_frame(self, *, force_visualization: bool = False):
        if self.sim.is_window_opened:
            self.sim.sync_render_state()
        yield
        self.sim.capture_visualization_safely(force=force_visualization)

    def is_task_success(self, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.zeros(self.num_envs, dtype=torch.bool)

    def _initialize_episode(self, env_ids: torch.Tensor, **kwargs) -> None:
        del env_ids, kwargs
        self.reset_events.append("initialize_episode")
        self.initial_random_value = float(self.np_random.random())

    def get_obs(self, **kwargs) -> dict[str, float]:
        del kwargs
        self.reset_events.append("get_obs")
        return {"random": self.initial_random_value}

    def get_info(self, **kwargs) -> dict:
        del kwargs
        return {}


def test_reset_seed_replays_gym_rng_and_reseeds_event_manager(monkeypatch) -> None:
    """A Gym reset seed becomes the effective task and event-manager seed."""
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)

    def _set_seed(seed: int) -> int:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        return seed + 1

    monkeypatch.setattr(base_env_module, "set_seed", _set_seed)
    env = _ResetEnv()

    first_obs, _ = env.reset(seed=2026)
    second_obs, _ = env.reset(seed=2026)

    assert first_obs == second_obs
    assert env.cfg.seed == 2027
    assert env.event_manager.set_seed.call_count == 2
    env.event_manager.set_seed.assert_called_with(2027)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True


def test_reset_publishes_episode_state_before_visual_observation() -> None:
    """Reset state must reach render consumers before capture and observations."""
    env = _ResetEnv()
    env.sim.is_window_opened = True

    env.reset()

    assert env.reset_events == [
        "reset_objects",
        "initialize_episode",
        "sync_render",
        "get_obs",
        "capture",
    ]


def test_headless_reset_leaves_publication_to_visual_consumers() -> None:
    """State-only training resets must not publish the entire render scene."""
    env = _ResetEnv()

    env.reset()

    env.sim.sync_render_state.assert_not_called()
    assert env.reset_events == [
        "reset_objects",
        "initialize_episode",
        "get_obs",
        "capture",
    ]


@pytest.mark.parametrize(
    ("reset_ids", "selected_rows"),
    [
        pytest.param(None, [0, 1, 2, 3], id="all"),
        pytest.param([3, 1], [3, 1], id="partial-list"),
        pytest.param(
            torch.tensor([3, 1], dtype=torch.int32), [3, 1], id="partial-int32"
        ),
        pytest.param(
            torch.tensor([3, 1], dtype=torch.int64), [3, 1], id="partial-int64"
        ),
        pytest.param([], [], id="empty"),
    ],
)
def test_reset_keeps_sensor_uid_exclusions_out_of_physical_registries(
    reset_ids: list[int] | torch.Tensor | None, selected_rows: list[int]
) -> None:
    """Shared UIDs preserve both resets with canonical row IDs and row isolation."""
    env = _ResetEnv(num_envs=4)
    prior_steps = torch.tensor([7, 8, 9, 10], dtype=torch.int32)
    env._elapsed_steps.copy_(prior_steps)
    robot = MagicMock()
    sensor = MagicMock(uid="shared")
    manager_sensor = MagicMock(uid="manager_only")
    env.sensors = {"observation": sensor}
    env.sim.physics = SimpleNamespace(name="default")
    env.sim._robots = {"shared": robot}
    for registry in (
        "_articulations",
        "_rigid_objects",
        "_rigid_object_groups",
        "_deformable_objects",
        "_lights",
    ):
        setattr(env.sim, registry, {})
    env.sim._sensors = {"shared": sensor, "manager_only": manager_sensor}
    env.sim.reset_objects_state = (
        lambda **kwargs: SimulationManager.reset_objects_state(env.sim, **kwargs)
    )

    _, info = env.reset(options={"reset_ids": reset_ids})

    assert robot.reset.call_count == 1
    assert sensor.reset.call_count == 1
    assert manager_sensor.reset.call_count == 1
    # EpisodeContext owns the normalized selection, independent of input dtype.
    expected_ids = torch.tensor(selected_rows, dtype=torch.long)
    torch.testing.assert_close(robot.reset.call_args.args[0], expected_ids)
    torch.testing.assert_close(sensor.reset.call_args.kwargs["env_ids"], expected_ids)
    torch.testing.assert_close(manager_sensor.reset.call_args.args[0], expected_ids)
    expected_steps = prior_steps.clone()
    expected_steps[expected_ids] = 0
    torch.testing.assert_close(env._elapsed_steps, expected_steps)
    assert [row["episode"] for row in info["episode_context"]] == [
        0 if row in selected_rows else -1 for row in range(env.num_envs)
    ]


def test_named_component_stream_rewinds_on_explicit_seed_and_is_independent():
    env = _ResetEnv()
    command = env.get_generator("commands")
    noise = env.get_generator("noise")
    env.reset(seed=42)
    first = torch.rand(6, generator=command)
    torch.rand(17, generator=noise)
    env.reset(seed=42)
    assert env.get_generator("commands") is command
    assert torch.equal(first, torch.rand(6, generator=command))
    env.reset()
    continued = torch.rand(6, generator=command)
    env.reset(seed=42)
    torch.rand(6, generator=command)
    assert torch.equal(continued, torch.rand(6, generator=command))


def test_unseeded_component_streams_do_not_share_a_fixed_seed():
    first, second = _ResetEnv(), _ResetEnv()
    a, b = first.get_generator("commands"), second.get_generator("commands")
    assert a.initial_seed() != b.initial_seed()
    assert not torch.equal(torch.rand(8, generator=a), torch.rand(8, generator=b))
    assert first.get_generator("commands") is a


def test_explicit_zero_seed_is_reproducible():
    first, second = _ResetEnv(), _ResetEnv()
    first.cfg.seed = second.cfg.seed = 0
    a, b = first.get_generator("commands"), second.get_generator("commands")
    assert torch.equal(torch.rand(8, generator=a), torch.rand(8, generator=b))


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("info_owner", [BaseEnv, EmbodiedEnv])
def test_episode_context_info_is_optional_without_changing_lifecycle(
    enabled: bool, info_owner: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opting out avoids snapshots while partial reset and provenance stay valid."""
    env = _ResetEnv(num_envs=2)
    assert env.cfg.episode_context_in_info is True
    env.cfg.episode_context_in_info = enabled
    env.compute_task_state = lambda **_: (
        torch.zeros(2, dtype=torch.bool),
        torch.zeros(2, dtype=torch.bool),
        {},
    )
    env.get_info = lambda **kwargs: info_owner.get_info(env, **kwargs)
    snapshot = MagicMock(wraps=env.episode_context.snapshot)
    monkeypatch.setattr(env.episode_context, "snapshot", snapshot)
    _, info = env.reset()
    assert ("episode_context" in info) is enabled
    assert snapshot.call_count == int(enabled)
    # Recording remains eligible and can still request an owned snapshot.
    assert env.episode_context.filter_recordable([0, 1]).tolist() == [0, 1]
    saved = snapshot()
    env.get_info()
    _, partial_info = env.reset(options={"reset_ids": [1]})
    assert ("episode_context" in partial_info) is enabled
    assert snapshot.call_count == 1 + 3 * int(enabled)
    assert [row["episode"] for row in saved] == [0, 0]
    assert [row["episode"] for row in snapshot()] == [0, 1]
    assert env.episode_context.filter_recordable([0, 1]).tolist() == [0, 1]
