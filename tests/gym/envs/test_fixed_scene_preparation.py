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

"""Controlled Gym preparation without a simulator or implicit reset events."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.managers.record import record_camera_data
from embodichain.lab.gym.utils.profiler import EnvProfiler
from embodichain.lab.sim.motion.expansion import (
    SceneCase,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.initial_state import (
    FixedSceneHost,
    InitialStateProfile,
)


class _PreparationEnv(EmbodiedEnv):
    """Exercise real preparation, reset, and step with CPU host ports."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._num_envs = 2
        self.cfg = SimpleNamespace(
            events=True,
            observations=True,
            rewards=True,
            dataset=True,
            trajectory_auto_save=False,
            ignore_terminations=False,
            sim_steps_per_control=1,
            seed=None,
        )
        self.sim_cfg = SimpleNamespace(physics_dt=0.01)
        self.sim = SimpleNamespace(
            device=torch.device("cpu"),
            num_envs=2,
            update=Mock(),
            reset_objects_state=Mock(),
            capture_visualization_safely=Mock(),
        )
        self.robot = object()
        self._profiler = EnvProfiler(None, torch.device("cpu"))
        self._elapsed_steps = torch.full((2,), 7, dtype=torch.long)
        self._task_success = torch.ones(2, dtype=torch.bool)
        self.episode_success_status = torch.ones(2, dtype=torch.bool)
        self._detached_uids_for_reset = []
        self.max_episode_steps = 1
        self.physical_value = torch.tensor([[90.0], [91.0]])
        self.camera = object.__new__(record_camera_data)
        self.camera.discard_and_clear = Mock(
            side_effect=lambda **kwargs: self.calls.append("camera_discard")
        )
        self.camera.save_and_clear = Mock()
        self.camera.finalize = Mock()
        self.event_manager = SimpleNamespace(
            _mode_functor_cfgs={"interval": [SimpleNamespace(func=self.camera)]},
            available_modes=["reset"],
            active_functors={},
            apply=Mock(side_effect=lambda **kwargs: self.calls.append("event_reset")),
            set_seed=Mock(),
            reset=Mock(),
        )
        self.observation_manager = SimpleNamespace(
            reset=Mock(side_effect=lambda **kwargs: self.calls.append("obs_reset"))
        )
        self.reward_manager = SimpleNamespace(
            reset=Mock(side_effect=lambda **kwargs: self.calls.append("reward_reset"))
        )
        self.dataset_manager = SimpleNamespace(
            reset=Mock(side_effect=lambda **kwargs: self.calls.append("dataset_reset")),
            apply=Mock(),
            available_modes=["save"],
            save_failed_episodes=True,
        )
        self.rollout_buffer = TensorDict(
            {
                "obs": TensorDict({"state": torch.full((2, 3, 1), -1.0)}, [2, 3]),
                "valid": torch.ones(2, 3, dtype=torch.bool),
            },
            [2, 3],
            device="cpu",
        )
        self._rollout_buffer_mode = "expert"
        self._max_rollout_steps = 3
        self.rollout_steps = torch.full((2,), 2, dtype=torch.long)
        self.current_rollout_step = 2
        self._traj_steps = torch.full((2,), 2, dtype=torch.long)
        self._traj_buffer = None
        self._traj_raw_action = object()
        self._demo_active_segment_ids = torch.full((2,), 3, dtype=torch.long)
        self._demo_active_mask = torch.zeros(2, dtype=torch.bool)
        self._demo_segment_participants = torch.ones(2, dtype=torch.bool)
        self._demo_active_segment_start_steps = torch.full((2,), 2, dtype=torch.long)
        self._demo_active_rollout_start_steps = torch.full((2,), 2, dtype=torch.long)
        self._demo_steps = torch.full((2,), 2, dtype=torch.long)
        self._demo_episode_metadata = [{"completed": True}, {"completed": True}]
        self._active_task_program_bridge = object()
        self._demo_no_auto_reset = False
        self._replay_no_auto_reset = False
        self._np_random = np.random.default_rng(17)

    def get_obs(self, **kwargs):
        self.calls.append("get_obs")
        return TensorDict({"state": self.physical_value.clone()}, [2], device="cpu")

    def compute_task_state(self, **kwargs):
        self.calls.append("task_state")
        return torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

    def _seed_recording_state(self, obs, env_ids):
        self.calls.append("seed_recording")
        super()._seed_recording_state(obs, env_ids)

    def _preprocess_action(self, action):
        return action

    def _step_action(self, action):
        return action

    def _postprocess_action(self, action):
        return action

    def get_reward(self, **kwargs):
        return torch.zeros(2)

    def _extend_reward(self, rewards, **kwargs):
        return rewards

    def _hook_after_sim_step(self, **kwargs):
        pass

    def check_truncated(self, **kwargs):
        return torch.zeros(2, dtype=torch.bool)

    def is_task_success(self, **kwargs):
        return torch.zeros(2, dtype=torch.bool)


def _prepare(env: _PreparationEnv, owner: object, *, verification=None):
    def prepare():
        env.calls.append("prepare")
        assert env._active_task_program_bridge is None
        assert env._elapsed_steps.tolist() == [0, 0]
        assert not env.rollout_buffer["valid"].any()

    def restore():
        env.calls.append("restore")
        env.physical_value = torch.tensor([[1.0], [2.0]])

    def settle():
        env.calls.append("settle")
        env.physical_value += 1.0

    def verify():
        env.calls.append("verify")
        if verification is not None:
            return verification
        return ValidationResult((ValidationCheck("initial_state", "passed"),))

    return env.prepare_generation_episode(
        owner, prepare=prepare, restore=restore, settle=settle, verify=verify
    )


def test_preparation_orders_cleanup_restore_verification_and_first_frame() -> None:
    env = _PreparationEnv()
    owner = object()
    env.acquire_generation_lease(owner)
    rng_state = env.np_random.bit_generator.state
    torch_rng_state = torch.random.get_rng_state().clone()

    obs, info = _prepare(env, owner)

    assert env.calls == [
        "camera_discard",
        "prepare",
        "restore",
        "settle",
        "obs_reset",
        "reward_reset",
        "dataset_reset",
        "verify",
        "get_obs",
        "task_state",
        "seed_recording",
    ]
    assert obs["state"].tolist() == [[2.0], [3.0]]
    assert torch.equal(env.rollout_buffer["obs", "state"][:, 0], obs["state"])
    assert info["elapsed_steps"].tolist() == [0, 0]
    for counter in (env.rollout_steps, env._traj_steps, env._demo_steps):
        assert counter.tolist() == [0, 0]
    assert env.current_rollout_step == 0
    assert not env.episode_success_status.any()
    assert not env._task_success.any()
    assert not env.rollout_buffer["valid"].any()
    assert env._demo_active_mask.all()
    assert not env._demo_segment_participants.any()
    assert all(not metadata["completed"] for metadata in env._demo_episode_metadata)
    assert env._traj_raw_action is None
    assert env.np_random.bit_generator.state == rng_state
    assert torch.equal(torch.random.get_rng_state(), torch_rng_state)
    env.event_manager.apply.assert_not_called()
    env.event_manager.reset.assert_not_called()
    env.event_manager.set_seed.assert_not_called()
    env.dataset_manager.apply.assert_not_called()
    env.camera.save_and_clear.assert_not_called()
    env.camera.finalize.assert_not_called()
    env.sim.update.assert_not_called()
    env.sim.reset_objects_state.assert_not_called()
    assert env.observation_manager.reset.call_args.kwargs["env_ids"].tolist() == [0, 1]


def test_lease_identity_blocks_foreign_calls_without_side_effects() -> None:
    env = _PreparationEnv()
    owner = object()
    env.acquire_generation_lease(owner)
    epoch = env.generation_epoch
    env.acquire_generation_lease(owner)
    assert env.generation_epoch == epoch
    assert env.sim._trajectory_generation_owner is owner
    for operation in (
        lambda: env.acquire_generation_lease(object()),
        lambda: env.release_generation_lease(object()),
        lambda: _prepare(env, object()),
        lambda: env.reset(seed=9),
        lambda: env.step(torch.zeros(2, 1)),
    ):
        with pytest.raises(RuntimeError):
            operation()
    assert env.generation_epoch == epoch
    assert env.sim._trajectory_generation_owner is owner
    assert env.calls == []
    env.event_manager.set_seed.assert_not_called()
    env.sim.update.assert_not_called()


@pytest.mark.parametrize(
    "status", ["failed", "unavailable", "not_run", "empty", "wrong"]
)
def test_unverified_preparation_does_not_publish_first_frame_or_allow_step(
    status,
) -> None:
    env = _PreparationEnv()
    owner = object()
    env.acquire_generation_lease(owner)
    previous_epoch = env.generation_epoch
    validation = (
        object()
        if status == "wrong"
        else (
            ValidationResult(())
            if status == "empty"
            else ValidationResult((ValidationCheck("initial_state", status),))
        )
    )
    with pytest.raises((RuntimeError, TypeError), match="verification"):
        _prepare(env, owner, verification=validation)
    assert env.generation_epoch > previous_epoch
    assert "get_obs" not in env.calls
    assert "seed_recording" not in env.calls
    assert not env._generation_preparing
    assert not env.rollout_buffer["valid"].any()
    with pytest.raises(RuntimeError, match="before stepping"):
        env.step(torch.zeros(2, 1))
    _prepare(env, owner)
    assert env._generation_prepared


def test_failed_restore_invalidates_previous_prepared_epoch() -> None:
    env = _PreparationEnv()
    owner = object()
    env.acquire_generation_lease(owner)
    _prepare(env, owner)
    old_epoch = env.generation_epoch
    env.calls.clear()
    restore = Mock(side_effect=ValueError("restore failed"))
    settle, verify = Mock(), Mock()
    with pytest.raises(ValueError, match="restore failed"):
        env.prepare_generation_episode(
            owner, prepare=Mock(), restore=restore, settle=settle, verify=verify
        )
    assert env.generation_epoch > old_epoch
    assert not env._generation_prepared
    assert "seed_recording" not in env.calls
    settle.assert_not_called()
    verify.assert_not_called()
    with pytest.raises(RuntimeError, match="before stepping"):
        env.step(torch.zeros(2, 1))


def test_generation_disables_auto_reset_until_release_without_touching_demo_flags() -> (
    None
):
    env = _PreparationEnv()
    owner = object()
    env.acquire_generation_lease(owner)
    _prepare(env, owner)
    env.step(torch.zeros(2, 1))
    assert env._elapsed_steps.tolist() == [1, 1]
    env.sim.reset_objects_state.assert_not_called()
    old_epoch = env.generation_epoch
    env.calls.clear()
    env._demo_no_auto_reset = True
    env.release_generation_lease(owner)
    assert env.generation_epoch > old_epoch
    assert env._demo_no_auto_reset
    assert not env._generation_no_auto_reset
    assert env.calls == []
    env.sim.reset_objects_state.assert_not_called()
    env._demo_no_auto_reset = False
    env.step(torch.zeros(2, 1))
    env.sim.reset_objects_state.assert_called_once()
    assert env._elapsed_steps.tolist() == [0, 0]


def test_generation_lease_rejects_none_owner() -> None:
    env = _PreparationEnv()
    with pytest.raises(ValueError, match="None"):
        env.acquire_generation_lease(None)
    with pytest.raises(RuntimeError, match="lease"):
        env.release_generation_lease(None)


class _PhysicalAdapter:
    """Fake physical port connected to the real Gym and host lifecycle."""

    def __init__(self, env: _PreparationEnv) -> None:
        self.env = env
        self.sim = env.sim
        self.robot = env.robot
        self.accept_state = True

    def signature(self) -> str:
        return "fixed-two-row-robot"

    def capture(self) -> torch.Tensor:
        return self.env.physical_value.clone()

    def restore(self, state: torch.Tensor) -> None:
        self.env.physical_value.copy_(state)

    def verify(self, state: torch.Tensor) -> ValidationResult:
        accepted = self.accept_state and torch.equal(self.env.physical_value, state)
        return ValidationResult(
            (ValidationCheck("physical_initial", "passed" if accepted else "failed"),)
        )


def _host_fixture(env: _PreparationEnv):
    adapter = _PhysicalAdapter(env)
    profile = InitialStateProfile(
        profile_id="fixed-test",
        prepare=lambda: None,
        signature=lambda: "fixed-conditions",
        verify=lambda cases: ValidationResult(
            (ValidationCheck("task_initial", "passed"),)
        ),
        settling_steps=1,
    )
    cases = tuple(
        SceneCase(f"scene-{row}", f"initial-{row}", f"signature-{row}", "task", "robot")
        for row in range(2)
    )
    return adapter, profile, cases


def test_host_and_gym_share_settled_first_frame_and_current_epoch() -> None:
    env = _PreparationEnv()
    adapter, profile, cases = _host_fixture(env)
    env.sim.update.side_effect = lambda dt, steps: env.physical_value.add_(0.5)
    with FixedSceneHost(adapter, profile, env=env) as host:
        first = host.acquire_case(cases)
        settled = env.physical_value.clone()
        assert settled.tolist() == [[90.5], [91.5]]
        assert torch.equal(env.rollout_buffer["obs", "state"][:, 0], settled)
        assert first.epoch == env.generation_epoch
        assert host.verify_initial(first).accepted
        calls_before_observation = list(env.calls)
        observation = host.initial_observation(first)
        assert torch.equal(observation["state"], settled)
        observation["state"].zero_()
        assert torch.equal(host.initial_observation(first)["state"], settled)
        assert env.calls == calls_before_observation
        env.sim.update.side_effect = None
        env.physical_value += 10.0
        restored = host.restore_initial()
        assert torch.equal(env.physical_value, settled)
        assert torch.equal(env.rollout_buffer["obs", "state"][:, 0], settled)
        assert restored.epoch == env.generation_epoch > first.epoch
        with pytest.raises(RuntimeError, match="obsolete"):
            host.assert_current(first)
        with pytest.raises(RuntimeError, match="obsolete"):
            host.initial_observation(first)
        host.assert_current(restored)
        env.sim.reset_objects_state.assert_not_called()
        env.event_manager.apply.assert_not_called()
    assert not env._generation_no_auto_reset


@pytest.mark.parametrize("operation", ["acquire", "restore"])
def test_host_physical_verification_failure_never_seeds_gym_episode(operation) -> None:
    env = _PreparationEnv()
    adapter, profile, cases = _host_fixture(env)
    with FixedSceneHost(adapter, profile, env=env) as host:
        previous = host.acquire_case(cases) if operation == "restore" else None
        adapter.accept_state = False
        env.calls.clear()
        with pytest.raises(RuntimeError, match="verification"):
            if operation == "restore":
                host.restore_initial()
            else:
                host.acquire_case(cases)
        assert "get_obs" not in env.calls
        assert "seed_recording" not in env.calls
        assert not env.rollout_buffer["valid"].any()
        with pytest.raises(RuntimeError, match="before stepping"):
            env.step(torch.zeros(2, 1))
        if previous is not None:
            with pytest.raises(RuntimeError, match="obsolete"):
                host.assert_current(previous)
        adapter.accept_state = True
        current = (
            host.restore_initial()
            if operation == "restore"
            else host.acquire_case(cases)
        )
        host.assert_current(current)


def test_host_verification_observes_reset_standard_manager_state() -> None:
    env = _PreparationEnv()
    adapter, profile, cases = _host_fixture(env)
    env.reward_manager.dirty = True
    env.reward_manager.reset.side_effect = lambda **kwargs: setattr(
        env.reward_manager, "dirty", False
    )
    profile.verify = lambda cases: ValidationResult(
        (
            ValidationCheck(
                "manager_initial", "failed" if env.reward_manager.dirty else "passed"
            ),
        )
    )
    with FixedSceneHost(adapter, profile, env=env) as host:
        initial = host.acquire_case(cases)
        assert host.verify_initial(initial).accepted
        env.reward_manager.dirty = True
        restored = host.restore_initial()
        assert host.verify_initial(restored).accepted
        assert env.reward_manager.reset.call_count == 2


def test_gym_lease_prevents_a_second_pure_sim_host() -> None:
    env = _PreparationEnv()
    adapter, profile, _ = _host_fixture(env)
    owner = object()
    env.acquire_generation_lease(owner)
    try:
        with pytest.raises(RuntimeError, match="already has"):
            FixedSceneHost(adapter, profile)
        assert env._generation_lease_owner is owner
        assert env.sim._trajectory_generation_owner is owner
    finally:
        env.release_generation_lease(owner)
    assert env.sim._trajectory_generation_owner is None
    with FixedSceneHost(adapter, profile):
        pass


def test_pure_sim_host_prevents_a_second_gym_lease() -> None:
    env = _PreparationEnv()
    adapter, profile, _ = _host_fixture(env)
    with FixedSceneHost(adapter, profile) as host:
        with pytest.raises(RuntimeError, match="already has"):
            env.acquire_generation_lease(object())
        assert env.sim._trajectory_generation_owner is host
        assert getattr(env, "_generation_lease_owner", None) is None
    owner = object()
    env.acquire_generation_lease(owner)
    env.release_generation_lease(owner)


def test_foreign_sim_owner_cannot_be_cleared_by_stale_gym_lease() -> None:
    env = _PreparationEnv()
    owner, foreign = object(), object()
    env.acquire_generation_lease(owner)
    old_epoch = env.generation_epoch
    env.sim._trajectory_generation_owner = foreign
    for operation in (
        lambda: env.acquire_generation_lease(owner),
        lambda: env.release_generation_lease(owner),
    ):
        with pytest.raises(RuntimeError, match="does not own"):
            operation()
    assert env._generation_lease_owner is owner
    assert env.sim._trajectory_generation_owner is foreign
    assert env.generation_epoch == old_epoch
