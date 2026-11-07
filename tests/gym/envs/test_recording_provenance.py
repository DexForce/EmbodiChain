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

"""Training/replay identity contracts using existing trajectory buffers."""

from __future__ import annotations

from dataclasses import replace
from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
from embodichain.lab.gym.envs.managers.observation_types import (
    ObservationDescriptor,
    ObservationSample,
    ObservationFrame,
)

from embodichain.lab.gym.utils.gym_utils import _init_raw_rollout
from tensordict import TensorDict

from embodichain.lab.gym.envs.episode_context import EpisodeContext

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.managers.datasets import LeRobotRecorder
from embodichain.data_pipeline.recording import stable_config_hash
from embodichain.lab.sim.motion.expansion.contracts import (
    CandidateIdentity,
    CandidateSpec,
)


def make_env() -> SimpleNamespace:
    env = SimpleNamespace(
        cfg=SimpleNamespace(seed=12, task_program=None, sim_steps_per_control=2),
        step_dt=0.04,
        physics_dt=0.02,
        control_frequency=25,
        num_envs=1,
        device=torch.device("cpu"),
        active_joint_ids=[0, 1],
        robot=SimpleNamespace(uid="arm", dof=2),
        sim=SimpleNamespace(
            _articulations={}, _rigid_objects={}, physics_backend="default"
        ),
    )
    env._demo_episode_metadata = [EmbodiedEnv._new_demo_episode_metadata(env, 0)]
    env._demo_episode_metadata[0]["length"] = 3
    env._traj_buffer = TensorDict(
        {
            "states": {"robot": {"qpos": torch.arange(6).reshape(1, 3, 2)}},
            "actions": torch.zeros(1, 3, 2),
        },
        batch_size=[1, 3],
    )
    env._traj_steps = torch.tensor([3])
    env.get_demo_episode_metadata = lambda row: copy.deepcopy(
        env._demo_episode_metadata[row]
    )
    env._trajectory_action_metadata = lambda: {"action_kind": "expert"}
    env.episode_context = EpisodeContext(1)
    env.episode_context._begin_episode([0])
    env.episode_context._finish_initialization([0])
    return env


def test_environment_identity_is_unique_per_episode_and_shared_per_run() -> None:
    env = make_env()
    first = env._demo_episode_metadata[0]
    second = EmbodiedEnv._new_demo_episode_metadata(env, 0)

    assert UUID(first["episode_uuid"]) != UUID(second["episode_uuid"])
    assert first["run_uuid"] == second["run_uuid"]
    assert first["config_hash"] == second["config_hash"]


def test_episode_override_program_hash_uses_executed_program() -> None:
    env = make_env()
    env.cfg.task_program = {"program_id": "configured_default"}
    actual = {"program_id": "episode_override", "segments": ["pick", "place"]}
    env._active_task_program_bridge = SimpleNamespace(
        _program=actual, expansion_records=[]
    )
    result = SimpleNamespace(
        instruction=None,
        instruction_source="unknown",
        lengths=[3],
        completed_by_env=[True],
        terminal_reasons=["success"],
        episode_index=1,
        success=[True],
        terminated=[False],
        truncated=[False],
    )

    EmbodiedEnv._end_demo_episode_recording(env, result)

    recorded_hash = env._demo_episode_metadata[0]["program_hash"]
    assert recorded_hash == stable_config_hash(actual)
    assert recorded_hash != stable_config_hash(env.cfg.task_program)


def test_external_payloads_without_identity_do_not_share_worker_episode_index() -> None:
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = make_env()
    recorder.curr_episode = 0

    first = recorder._ensure_episode_identity(0, None)
    second = recorder._ensure_episode_identity(0, None)

    assert first["episode_uuid"] != second["episode_uuid"]
    assert first["run_uuid"] == second["run_uuid"]


def test_executed_expansion_records_preserve_selected_candidate_lineage() -> None:
    env = make_env()
    selected_identity = CandidateIdentity(
        scene_case_id="scene",
        initial_state_id="initial",
        candidate_id="selected",
        geometry_family_id="family",
        source_id="source",
        source_revision="revision",
        template_id="recipe:place:0:0",
        parent_id="reference",
    )
    unselected_identity = CandidateIdentity(
        scene_case_id="scene",
        initial_state_id="initial",
        candidate_id="other",
        geometry_family_id="other-family",
        source_id="other-source",
        source_revision="revision",
        template_id="other-recipe:place:0:0",
    )
    record = SimpleNamespace(
        env_id=0,
        selected_candidate_id="selected",
        candidates=(
            CandidateSpec(unselected_identity),
            CandidateSpec(selected_identity),
        ),
        to_metadata=lambda: {"selected_candidate_id": "selected"},
    )
    env._active_task_program_bridge = SimpleNamespace(
        _program=None, expansion_records=[record]
    )
    result = SimpleNamespace(
        instruction=None,
        instruction_source="unknown",
        lengths=[3],
        completed_by_env=[True],
        terminal_reasons=["success"],
        episode_index=1,
        success=[True],
        terminated=[False],
        truncated=[False],
    )

    EmbodiedEnv._end_demo_episode_recording(env, result)

    lineage = env._demo_episode_metadata[0]["expansion"][0][
        "selected_candidate_lineage"
    ]
    assert lineage == asdict(selected_identity)


def test_trajectory_metadata_links_source_identity_and_initial_state(
    tmp_path: Path,
) -> None:
    env = make_env()
    path = tmp_path / "source.pt"

    EmbodiedEnv.save_trajectory(env, str(path), env_ids=[0])

    trajectory = torch.load(path, weights_only=False)
    sidecar = env._demo_episode_metadata[0]
    assert trajectory["meta"]["episode_uuid"] == sidecar["episode_uuid"]
    assert trajectory["meta"]["run_uuid"] == sidecar["run_uuid"]
    assert trajectory["meta"]["config_hash"] == sidecar["config_hash"]
    assert sidecar["replay_artifact"]["initial_state_step"] == 0
    assert sidecar["replay_artifact"]["source_episode_uuid"] == sidecar["episode_uuid"]
    assert sidecar["trajectory_path"] == str(path.resolve())
    torch.testing.assert_close(
        trajectory["states"]["robot"]["qpos"][0, 0], torch.tensor([0, 1])
    )
    assert not list(tmp_path.glob(".source.pt.*"))


def test_fragment_retries_keep_stable_identity_and_source_replay(
    tmp_path: Path,
) -> None:
    env = make_env()
    EmbodiedEnv.save_trajectory(env, str(tmp_path / "source.pt"), env_ids=[0])
    metadata = copy.deepcopy(env._demo_episode_metadata[0])
    metadata.update(
        {
            "output_mode": "segment_fragments",
            "segments": [
                {"segment_id": 1, "start_step": 1, "end_step": 3, "success": True}
            ],
        }
    )
    recorder = LeRobotRecorder.__new__(LeRobotRecorder)
    recorder._env = env
    observations = TensorDict({"qpos": torch.zeros(3, 2)}, batch_size=[3])
    payload = list(
        recorder._episode_payloads(0, observations, torch.zeros(3, 2), {}, metadata)
    )[0]
    restarted = LeRobotRecorder.__new__(LeRobotRecorder)
    restarted._env = env
    retried = list(
        restarted._episode_payloads(0, observations, torch.zeros(3, 2), {}, metadata)
    )[0]
    fragment = payload[-1]

    assert fragment["episode_uuid"] == retried[-1]["episode_uuid"]
    assert fragment["episode_uuid"] != metadata["episode_uuid"]
    assert fragment["parent_episode_uuid"] == metadata["episode_uuid"]
    assert fragment["source_episode_uuid"] == metadata["source_episode_uuid"]
    assert fragment["source_start_step"] == 1
    assert fragment["replay_artifact"]["initial_state_step"] == 0
    assert fragment["source_trajectory_path"] == metadata["trajectory_path"]


def test_diagnostic_export_does_not_make_rejected_buffers_normal_recordings(tmp_path):
    env = make_env()
    with env.episode_context.check("task_spec", [0]) as finish:
        finish(accepted=False, reasons="unverified")
    path = tmp_path / "diagnostics.pt"
    EmbodiedEnv.save_trajectory(env, str(path), env_ids=[0], purpose="diagnostic")
    payload = torch.load(path, weights_only=False)
    assert payload["meta"]["artifact_purpose"] == "diagnostic"
    assert "trajectory_path" not in env._demo_episode_metadata[0]
    with pytest.raises(RuntimeError, match="rejected"):
        EmbodiedEnv.save_trajectory(env, str(tmp_path / "normal.pt"), env_ids=[0])
    assert not (tmp_path / "normal.pt").exists()


def test_saved_recording_preserves_terminal_state_and_described_raw_history(tmp_path):
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.gym.envs.managers.observation_types import (
        ObservationDescriptor,
        ObservationSample,
        ObservationFrame,
    )

    env = make_env()
    env._traj_buffer = TensorDict(
        {
            "states": {"robot": {"qpos": torch.zeros(1, 4, 2)}},
            "actions": torch.zeros(1, 4, 2),
        },
        batch_size=[1, 4],
    )
    env.episode_context = env.episode_context
    env.observation_manager = ObservationManager({}, env)
    env.observation_manager._env._rollout_storage = getattr(
        env.observation_manager._env, "_rollout_storage", None
    )
    if env.observation_manager._env._rollout_storage is None:
        env.observation_manager._env._rollout_storage = TensorDict(
            {},
            batch_size=[len(env.observation_manager._env.episode_context.episode_ids)],
        )
    _init_raw_rollout(
        env.observation_manager._env._rollout_storage,
        env.observation_manager._env.episode_context,
        **dict(
            fields={"robot/qpos": env._traj_buffer["states"]["robot"]["qpos"]},
            actions=env._traj_buffer["actions"][:, :3],
        ),
    )
    env.observation_manager.bind_rollout_storage(
        env.observation_manager._env._rollout_storage
    )
    env._traj_steps = env._rollout_storage["raw_steps"]
    descriptor = ObservationDescriptor(
        "arm", "qpos", "joint", "joint_order", ("joint_units",), (2,), torch.float32
    )
    for step in range(4):
        values = torch.tensor([[2.0 * step, 2.0 * step + 1.0]])
        frame = ObservationFrame(
            None,
            {
                "robot/qpos": ObservationSample(
                    descriptor, values, torch.ones(1, dtype=torch.bool), (None,)
                )
            },
            env.episode_context.episode_ids,
            step,
            step * 0.04,
            "simulation_time",
            env.episode_context,
        )
        if step == 0:
            EmbodiedEnv._record_initial_observation(env, frame, [0])
        else:
            env._traj_raw_action = torch.full((1, 2), float(step))
            EmbodiedEnv._write_trajectory_step(env, frame)
    path = tmp_path / "trajectory.pt"
    EmbodiedEnv.save_trajectory(env, str(path))
    payload = torch.load(path, weights_only=False)
    assert payload["states"]["robot"]["qpos"][0].tolist() == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
    ]
    assert payload["actions"].shape == (1, 3, 2)
    assert payload["terminal_states"]["robot"]["qpos"].tolist() == [[6.0, 7.0]]
    history = payload["raw_observations"]
    # Views in the artifact share one owned state snapshot, including after load.
    assert (
        payload["states"]["robot"]["qpos"].untyped_storage().data_ptr()
        == history["fields"]["robot/qpos"]["values"].untyped_storage().data_ptr()
    )
    assert history["fields"]["robot/qpos"]["values"][0].tolist() == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
        [6.0, 7.0],
    ]
    assert history["fields"]["robot/qpos"]["valid"].all()
    assert history["lengths"].tolist() == payload["meta"]["lengths"] == [3]


def _recording(storage=None):
    from tensordict import TensorDict
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.utils.gym_utils import (
        _init_raw_rollout,
        _close_raw_rollout,
    )

    context = EpisodeContext(2)
    context._begin_episode([0, 1])
    context._finish_initialization([0, 1])
    storage = (
        TensorDict({}, batch_size=[2], device="cpu") if storage is None else storage
    )
    env = SimpleNamespace(
        episode_context=context, _rollout_storage=storage, _rollout_buffer_mode="expert"
    )
    env.observation_manager = ObservationManager({}, env)
    _init_raw_rollout(storage, context, {}, torch.zeros(2, 2, 1))
    env.observation_manager.bind_rollout_storage(storage)

    def append(frame, action, **kwargs):
        env._traj_raw_action = action
        EmbodiedEnv._write_trajectory_step(env, frame, **kwargs)

    env.seed_history = lambda frame, ids: EmbodiedEnv._record_initial_observation(
        env, frame, ids
    )
    env.append_history = append
    env.read_history = env.observation_manager.read_history
    env.snapshot_history = env.observation_manager.snapshot_history
    env.close = lambda: _close_raw_rollout(storage)
    env.storage = storage
    return context, env


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
    EmbodiedEnv._record_initial_observation(recording, initial, [0, 1])
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
    assert recording.storage["raw_steps"].tolist() == [2, 2]
    torch.testing.assert_close(
        recording.storage["raw_obs", "measurement"][:, 2],
        torch.full((2, 1), 2.0, dtype=torch.float64),
    )
    archive = recording.snapshot_history()
    assert archive["fields"]["measurement"]["values"][:, :, 0].tolist() == [
        [-10.0, 1.0, 2.0],
        [-10.0, 1.0, 2.0],
    ]
    assert recording.observation_manager.schema["measurement"].dtype == torch.float64
    assert recording.storage["raw_actions"][:, :, 0].tolist() == [
        [11.0, 21.0],
        [12.0, 22.0],
    ]
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
    EmbodiedEnv._record_initial_observation(
        recording, _frame(context, 99, revision=3), [1]
    )
    result.values["measurement"][0] = 500
    still_live = recording.read_history(
        [1], env_ids=[0], fields=["measurement"], episode_ids=[0]
    )
    assert still_live.values["measurement"].item() == 1.0
    assert recording.storage["raw_steps"].tolist() == [2, 0]
    with pytest.raises(ValueError, match="another or uninitialized episode"):
        recording.read_history(
            [1], env_ids=[1], fields=["measurement"], episode_ids=[0]
        )


def test_queries_do_not_advance_history_and_missing_frames_never_fall_back():
    context, recording = _recording()
    EmbodiedEnv._record_initial_observation(recording, _frame(context, 7), [0, 1])
    for _ in range(2):
        snapshot = recording.read_history(
            [0, 0], fields=["measurement"], episode_ids=[0, 0]
        )
        assert snapshot.values["measurement"].flatten().tolist() == [7.0, 7.0]
    assert recording.storage["raw_steps"].tolist() == [0, 0]
    with pytest.raises(ValueError, match="not fully recorded"):
        recording.read_history([0, 1], fields=["measurement"], episode_ids=[0, 0])
    with pytest.raises(KeyError):
        recording.read_history([0, 0], fields=["unrecorded"], episode_ids=[0, 0])
    with pytest.raises(ValueError, match="new observation acquisition"):
        recording.append_history(_frame(context, 7), torch.zeros(2, 1))


def test_missing_acquisition_stays_unavailable_per_row():
    context, recording = _recording()
    EmbodiedEnv._record_initial_observation(recording, _frame(context, 0), [0, 1])
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


@pytest.mark.parametrize("alias", [False, True])
def test_recording_rejects_foreign_frame_owner_and_declaration_drift(alias):
    context, recording = _recording()
    other, _ = _recording()
    with pytest.raises(ValueError, match="another episode owner"):
        EmbodiedEnv._record_initial_observation(recording, _frame(other, 1), [0, 1])
    initial = _frame(context, 1)
    if alias:
        initial = replace(
            initial, fields={**initial.fields, "alias": initial.fields["measurement"]}
        )
    EmbodiedEnv._record_initial_observation(recording, initial, [0, 1])
    changed = _frame(context, 2, revision=1)
    sample = changed.fields["measurement"]
    changed = replace(
        changed,
        fields={
            **initial.fields,
            "alias" if alias else "measurement": replace(
                sample, descriptor=replace(sample.descriptor, units=("kPa",))
            ),
        },
    )
    with pytest.raises(ValueError, match="identity drift"):
        recording.append_history(changed, torch.zeros(2, 1))
    assert recording.storage["raw_steps"].tolist() == [0, 0]
    # Field layout is configured at seed, rather than grown during execution.
    with pytest.raises(ValueError, match="field schema changed"):
        recording.append_history(
            replace(_frame(context, 2, revision=1), fields={"extra": sample}),
            torch.zeros(2, 1),
        )


@pytest.mark.parametrize("mutation", ["reset", "close"])
def test_history_checks_lifetime_after_copy(monkeypatch, mutation):
    context, recording = _recording()
    EmbodiedEnv._record_initial_observation(recording, _frame(context, 1), [0, 1])
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
                recording.close()
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
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.utils.gym_utils import _init_raw_rollout

    env._rollout_storage = TensorDict({}, batch_size=[1], device="cpu")
    _init_raw_rollout(
        env._rollout_storage, env.episode_context, {}, torch.zeros(1, 1, 1)
    )
    env.observation_manager.bind_rollout_storage(env._rollout_storage)
    env._profiler = SimpleNamespace(section=lambda *args, **kwargs: nullcontext())
    env.sim = SimpleNamespace(
        render_frame=lambda: nullcontext(),
        simulation_time=0.0,
        device=torch.device("cpu"),
    )
    env.get_obs = lambda **kwargs: env.observation_manager.compute(
        TensorDict({}, batch_size=[1])
    )
    EmbodiedEnv._record_initial_observation(env, env._capture_observation(), [0])
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

    def record(observation_frame, action, **kwargs):
        env._traj_raw_action = action
        EmbodiedEnv._write_trajectory_step(env, observation_frame)

    env._hook_after_sim_step = record
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
        EmbodiedEnv._record_initial_observation(env, frame, [0])
        return frame.policy, {}

    env.reset = reset
    result = env.step(torch.tensor([[7.0]]))
    assert terminal[0].values["pressure"].item() == 1.0
    assert terminal[0].timestamps == (0.04,)
    assert terminal[0].episode_ids == (0,)
    assert result[0]["pressure"].item() == -5.0
    assert env.episode_context.episode_ids == (1,)
    assert env._rollout_storage["raw_steps"].tolist() == [0]


@pytest.mark.parametrize("mask", [torch.ones(2, dtype=torch.bool), torch.ones(1), True])
def test_invalid_recording_selection_rejects_before_control(mask):
    from embodichain.lab.gym.envs.base_env import BaseEnv

    env = BaseEnv.__new__(BaseEnv)
    env._num_envs = 1
    env.observation_manager = SimpleNamespace(has_history=True)
    with pytest.raises(ValueError, match="one bool per environment"):
        env.step(None, recording_active_mask=mask)


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
    assert manager.storage["raw_steps"].tolist() == [1, 1]
    assert manager.read_history(
        [1, 1], fields=["measurement"], episode_ids=[0, 0]
    ).values["measurement"].flatten().tolist() == [5.0, 5.0]
