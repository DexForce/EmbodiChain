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

"""Raw history ownership, exact boundary timing and row-local lifecycle."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.gym.envs.managers.observation_types import (
    ObservationDescriptor,
    ObservationSample,
    ObservationFrame,
)
from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
from types import SimpleNamespace


def _recording(storage=None):
    context = EpisodeContext(2)
    context._begin_episode([0, 1])
    context._finish_initialization([0, 1])
    recording = ObservationManager({}, SimpleNamespace(episode_context=context))
    recording.enable_history(fields={}, actions=torch.zeros(2, 2, 1), storage=storage)
    return context, recording


def _frame(context, value, *, revision=0, valid=None, field="measurement"):
    values = torch.full((2, 1), float(value), dtype=torch.float64)
    descriptor = ObservationDescriptor(
        "probe", "pressure", "sensor", "scalar", ("Pa",), (1,), torch.float64
    )
    sample = ObservationSample(
        descriptor,
        values,
        torch.ones(2, dtype=torch.bool) if valid is None else valid,
        (None, "probe disconnected") if valid is not None else (None, None),
    )
    return ObservationFrame(
        None,
        {field: sample},
        context.episode_ids,
        revision,
        float(revision),
        "simulation_time",
        context,
    )


def test_history_has_initial_and_terminal_frames_and_preserves_owned_values():
    context, recording = _recording()
    initial = _frame(context, -10)
    recording.seed_history(initial, [0, 1])
    recording.append_history(
        _frame(context, 1, revision=1), torch.tensor([[11.0], [12.0]])
    )
    recording.append_history(
        _frame(context, 2, revision=2), torch.tensor([[21.0], [22.0]])
    )
    result = recording.read_history([1, 2], fields=["measurement"], episode_ids=[0, 0])
    assert result.values["measurement"].flatten().tolist() == [1.0, 2.0]
    assert result.values["measurement"].dtype == torch.float64
    assert result.frame_indices == (1, 2)
    assert result.revisions == (1, 2)
    assert result.timestamps == (1.0, 2.0)
    assert recording.history.steps.tolist() == [2, 2]
    archive = recording.snapshot_history()
    assert archive["fields"]["measurement"]["values"][:, :, 0].tolist() == [
        [-10.0, 1.0, 2.0],
        [-10.0, 1.0, 2.0],
    ]
    assert recording.schema["measurement"].dtype == torch.float64
    assert recording.history.actions[:, :, 0].tolist() == [[11.0, 21.0], [12.0, 22.0]]
    assert (
        recording.read_history(
            [0], env_ids=[1], fields=["measurement"], episode_ids=[0]
        )
        .values["measurement"]
        .item()
        == -10.0
    )
    # A later seed and caller edits cannot rewrite an exported snapshot.
    context._begin_episode([1])
    context._finish_initialization([1])
    recording.seed_history(_frame(context, 99, revision=3), [1])
    result.values["measurement"][0] = 500
    still_live = recording.read_history(
        [1], env_ids=[0], fields=["measurement"], episode_ids=[0]
    )
    assert still_live.values["measurement"].item() == 1.0
    assert recording.history.steps.tolist() == [2, 0]
    with pytest.raises(ValueError, match="another or uninitialized episode"):
        recording.read_history(
            [1], env_ids=[1], fields=["measurement"], episode_ids=[0]
        )


def test_queries_do_not_advance_history_and_missing_frames_never_fall_back():
    context, recording = _recording()
    recording.seed_history(_frame(context, 7), [0, 1])
    for _ in range(2):
        snapshot = recording.read_history(
            [0, 0], fields=["measurement"], episode_ids=[0, 0]
        )
        assert snapshot.values["measurement"].flatten().tolist() == [7.0, 7.0]
    assert recording.history.steps.tolist() == [0, 0]
    with pytest.raises(ValueError, match="not fully recorded"):
        recording.read_history([0, 1], fields=["measurement"], episode_ids=[0, 0])
    with pytest.raises(KeyError):
        recording.read_history([0, 0], fields=["unrecorded"], episode_ids=[0, 0])
    with pytest.raises(ValueError, match="new observation acquisition"):
        recording.append_history(_frame(context, 7), torch.zeros(2, 1))


def test_missing_acquisition_stays_unavailable_per_row():
    context, recording = _recording()
    recording.seed_history(_frame(context, 0), [0, 1])
    recording.append_history(
        _frame(context, 0, revision=1, valid=torch.tensor([True, False])),
        torch.zeros(2, 1),
    )
    result = recording.read_history([1, 1], fields=["measurement"], episode_ids=[0, 0])
    assert result.valid["measurement"].tolist() == [True, False]
    assert result.acquisition_errors["measurement"] == (
        None,
        "probe disconnected",
    )


def test_recording_rejects_foreign_frame_owner_and_declaration_drift():
    context, recording = _recording()
    other, _ = _recording()
    with pytest.raises(ValueError, match="another episode owner"):
        recording.seed_history(_frame(other, 1), [0, 1])
    recording.seed_history(_frame(context, 1), [0, 1])
    changed = _frame(context, 2, revision=1)
    sample = changed.fields["measurement"]
    changed = replace(
        changed,
        fields={
            "measurement": replace(
                sample, descriptor=replace(sample.descriptor, units=("kPa",))
            )
        },
    )
    with pytest.raises(ValueError, match="identity drift"):
        recording.append_history(changed, torch.zeros(2, 1))
    assert recording.history.steps.tolist() == [0, 0]
    # Field layout is configured at seed, rather than grown during execution.
    with pytest.raises(ValueError, match="field schema changed"):
        recording.append_history(
            replace(_frame(context, 2, revision=1), fields={"extra": sample}),
            torch.zeros(2, 1),
        )


@pytest.mark.parametrize("mutation", ["reset", "close"])
def test_history_checks_lifetime_after_copy(monkeypatch, mutation):
    context, recording = _recording()
    recording.seed_history(_frame(context, 1), [0, 1])
    original = context.snapshot
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            if mutation == "reset":
                context._begin_episode([1])
            else:
                recording.history.close()
        return result

    monkeypatch.setattr(context, "snapshot", interrupt)
    with pytest.raises(
        (ValueError, RuntimeError), match="another or uninitialized episode|after close"
    ):
        recording.read_history([0, 0], fields=["measurement"], episode_ids=[0, 0])


def test_control_boundary_is_recorded_before_reward_mutation_and_autoreset():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from tensordict import TensorDict
    from embodichain.lab.gym.envs.base_env import BaseEnv
    from embodichain.lab.gym.envs.managers import ObservationCfg, ObservationManager

    env = BaseEnv.__new__(BaseEnv)
    env._num_envs = 1
    env.episode_context = EpisodeContext(1)
    env.episode_context._begin_episode([0])
    env.episode_context._finish_initialization([0])

    live = torch.tensor([[0.0]], dtype=torch.float64)
    descriptor = ObservationDescriptor(
        "probe", "pressure", "sensor", "scalar", ("Pa",), (1,), torch.float64
    )

    def source(env, obs):
        return ObservationSample(
            descriptor, live, torch.ones(1, dtype=torch.bool), (None,)
        )

    env.observation_manager = ObservationManager(
        {
            "pressure": ObservationCfg(
                func=source,
                mode="add",
                name="pressure",
                consumers=("policy", "recording"),
            )
        },
        env,
    )
    env.observation_manager.enable_history(fields={}, actions=torch.zeros(1, 1, 1))
    env._profiler = SimpleNamespace(section=lambda *args, **kwargs: nullcontext())
    env.sim = SimpleNamespace(
        render_frame=lambda: nullcontext(),
        simulation_time=0.0,
        device=torch.device("cpu"),
    )
    env.get_obs = lambda **kwargs: env.observation_manager.compute(
        TensorDict({}, batch_size=[1])
    )
    env.observation_manager.seed_history(env._capture_observation(), [0])
    env._preprocess_action = env._step_action = env._postprocess_action = (
        lambda action: action
    )

    def advance():
        live.fill_(1.0)
        env.sim.simulation_time = 0.04

    env._advance_physics = advance
    env._update_sim_state = lambda **kwargs: None
    env._update_physical_objective = lambda: None
    env.get_info = lambda **kwargs: {}

    def reward(**kwargs):
        # Third-party reward hooks cannot rewrite the captured physical frame.
        live.fill_(99.0)
        return torch.zeros(1)

    env.get_reward = reward
    env._extend_reward = lambda rewards, **kwargs: rewards
    env._elapsed_steps = torch.zeros(1, dtype=torch.long)
    env._get_termination_flags = lambda *args: (
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    env._hook_after_sim_step = lambda observation_frame, action, **kwargs: env.observation_manager.append_history(
        observation_frame, action
    )
    terminal = []

    def reset(**kwargs):
        terminal.append(
            env.observation_manager.read_history(
                [1], fields=["pressure"], episode_ids=[0]
            )
        )
        env.episode_context._begin_episode([0])
        env.episode_context._finish_initialization([0])
        live.fill_(-5.0)
        frame = env._capture_observation()
        env.observation_manager.seed_history(frame, [0])
        return frame.policy, {}

    env.reset = reset
    result = env.step(torch.tensor([[7.0]]))
    assert terminal[0].values["pressure"].item() == 1.0
    assert terminal[0].timestamps == (0.04,)
    assert terminal[0].episode_ids == (0,)
    assert result[0]["pressure"].item() == -5.0
    assert env.episode_context.episode_ids == (1,)
    assert env.observation_manager.history.steps.tolist() == [0]


@pytest.mark.parametrize("mask", [torch.ones(2, dtype=torch.bool), torch.ones(1), True])
def test_invalid_recording_selection_rejects_before_control(mask):
    from embodichain.lab.gym.envs.base_env import BaseEnv

    env = BaseEnv.__new__(BaseEnv)
    env._num_envs = 1
    env.observation_manager = SimpleNamespace(history=object())
    with pytest.raises(ValueError, match="one bool per environment"):
        env.step(None, recording_active_mask=mask)


def test_alias_schema_drift_rejects_without_advancing_history():
    context, recording = _recording()
    frame = _frame(context, 1)
    sample = frame.fields["measurement"]
    recording.seed_history(
        replace(frame, fields={"measurement": sample, "alias": sample}), [0, 1]
    )
    changed = replace(
        _frame(context, 2, revision=1),
        fields={
            "measurement": sample,
            "alias": replace(
                sample, descriptor=replace(sample.descriptor, units=("kPa",))
            ),
        },
    )
    with pytest.raises(ValueError, match="identity drift for 'alias'"):
        recording.append_history(changed, torch.zeros(2, 1))
    assert recording.history.steps.tolist() == [0, 0]


def test_temporal_buffer_accepts_values_without_observation_descriptors():
    from embodichain.lab.gym.envs.managers._observation_history import (
        _ObservationHistory,
    )

    context = EpisodeContext(1)
    context._begin_episode([0])
    context._finish_initialization([0])
    frame = SimpleNamespace(
        fields={
            "field:17": SimpleNamespace(
                values=torch.tensor([[4.0]]),
                valid=torch.tensor([True]),
                acquisition_errors=(None,),
            )
        },
        episode_ids=context.episode_ids,
        revision=1,
        timestamp=0.04,
        clock_domain="simulation_time",
        episode_owner=context,
    )
    history = _ObservationHistory(
        episode_context=context, fields={}, actions=torch.zeros(1, 1, 1)
    )
    history.bind(frame, ("field:17",))
    history.seed(frame, [0])
    result = history.read([0], fields=["field:17"], episode_ids=[0])
    assert result.values["field:17"].item() == 4.0
    assert result.valid["field:17"].tolist() == [True]


def test_raw_history_is_a_view_of_host_rollout_storage():
    from tensordict import TensorDict

    storage = TensorDict({}, batch_size=[2], device="cpu")
    context, manager = _recording(storage)
    manager.seed_history(_frame(context, 4), [0, 1])
    manager.append_history(_frame(context, 7, revision=1), torch.ones(2, 1))
    snapshot = manager.read_history([1, 1], fields=["measurement"], episode_ids=[0, 0])
    torch.testing.assert_close(
        storage["raw_obs", "measurement"][:, 1], snapshot.values["measurement"]
    )
    assert storage["raw_steps"].data_ptr() == manager.history.steps.data_ptr()
    assert storage["raw_actions"].data_ptr() == manager.history.actions.data_ptr()
    storage["raw_obs", "measurement"][:, 1] = 9.0
    assert manager.read_history(
        [1, 1], fields=["measurement"], episode_ids=[0, 0]
    ).values["measurement"].flatten().tolist() == [9.0, 9.0]
    assert snapshot.values["measurement"].flatten().tolist() == [7.0, 7.0]


@pytest.mark.parametrize("training_device", ["cpu", "meta"])
def test_training_window_reuse_preserves_episode_observation_history(training_device):
    from tensordict import TensorDict
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

    storage = TensorDict(
        {"policy": TensorDict({"obs": torch.zeros(2, 4, 1)}, batch_size=[2, 4])},
        batch_size=[2],
    )
    context, manager = _recording(storage)
    manager.seed_history(_frame(context, 1), [0, 1])
    manager.append_history(_frame(context, 5, revision=1), torch.ones(2, 1))
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        _rollout_storage=storage,
        _expert_obs_buffer=storage["policy", "obs"],
        rollout_steps=torch.ones(2, dtype=torch.long),
    )
    env._infer_rollout_buffer_mode = (
        lambda value: EmbodiedEnv._infer_rollout_buffer_mode(env, value)
    )
    window = TensorDict(
        {
            name: torch.zeros(2, 2, 1, device=training_device)
            for name in (
                "obs",
                "action",
                "reward",
                "done",
                "value",
                "terminated",
                "truncated",
            )
        },
        batch_size=[2, 2],
        device=training_device,
    )
    EmbodiedEnv.set_rollout_buffer(env, window)
    assert env._rollout_buffer_mode == "rl"
    assert env._max_rollout_steps == 1
    assert "policy" not in storage.keys()
    assert env._expert_obs_buffer is None
    assert storage["training"].device == torch.device(training_device)
    if training_device == "cpu":
        window.zero_()
        storage["training", "obs"][0, 0] = 3.0
        assert window["obs"][0, 0].item() == 3.0
    else:
        assert storage["training", "obs"] is window["obs"]
    assert manager.history.steps.tolist() == [1, 1]
    assert manager.read_history(
        [1, 1], fields=["measurement"], episode_ids=[0, 0]
    ).values["measurement"].flatten().tolist() == [5.0, 5.0]
