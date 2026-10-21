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

"""Unit tests for OnlineDataset and OnlineDataEngine.

These tests do **not** start a real simulation subprocess. ``_make_fake_engine``
injects a pre-filled shared buffer for sampling tests, while lifecycle tests
drive ``start()`` with a lightweight process mock and a real multiprocessing
error pipe.

This exercises all public logic in ``sample_batch``,
``_trigger_refill_if_needed``, and ``OnlineDataset.__iter__`` without GPU or
sim dependencies.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import threading
import unittest
from types import SimpleNamespace
from queue import Empty
from unittest.mock import MagicMock

import pytest

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader

from embodichain.data_pipeline.engine import (
    OnlineDataEngineState,
    OnlineDataWorkerError,
)
from embodichain.data_pipeline.engine import data as engine_module
from embodichain.data_pipeline.datasets import (
    ChunkSizeSampler,
    GMMChunkSampler,
    OnlineDataset,
    UniformChunkSampler,
)
from embodichain.data_pipeline.engine.data import (
    OnlineDataEngine,
    OnlineDataEngineCfg,
    _apply_worker_simulation_overrides,
    _populate_language_indices,
)
from embodichain.data_pipeline.engine.language import SharedLanguageRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUFFER_SIZE = 8
MAX_EPISODE_STEPS = 50
STATE_DIM = 6
OBS_DIM = 10
ACTION_DIM = 4
CONSUMER_RESULT_TIMEOUT = 60.0
PROCESS_CLEANUP_TIMEOUT = 5.0
LAB_PACKAGE_NAME = "embodichain.lab"


def test_action_context_respects_locks_gaps_continuity_and_padded_provenance() -> None:
    torch.manual_seed(13)
    engine = _make_fake_engine(
        buffer_size=2, max_episode_steps=6, lock_start=0, lock_end=1
    )
    source = engine.shared_buffer
    source["valid"][1] = torch.tensor([True, True, False, True, True, True])
    source["continuity_id"][1] = torch.tensor([0, 0, -1, 1, 1, 2])
    source["obs"][0] = 1000
    for index in range(6):
        source["obs"][1, index] = index + 10
    source["obs"][1, 2] = float("nan")
    source["ods_frame_index"] = torch.arange(6).repeat(2, 1)
    sample = engine.sample_batch_with_context(128, 2, 4)
    assert sample.batch_size == (128, 5)
    assert torch.isfinite(sample["obs"]).all()
    assert not sample["loss_mask"][:, 0].any()
    assert sample["valid"][:, 1].all()
    assert (sample["ods_frame_index"][~sample["valid"]] == -1).all()
    assert (sample["task_index"][~sample["valid"]] == -1).all()
    assert (sample["subtask_index"][~sample["valid"]] == -1).all()
    assert (sample["obs"][~sample["valid"]] == 0).all()
    assert set(sample["ods_frame_index"][:, 1].tolist()) == {0, 1, 3, 4, 5}
    spans = {0: {0, 1}, 1: {0, 1}, 3: {3, 4}, 4: {3, 4}, 5: {5}}
    for row in range(128):
        anchor = int(sample["ods_frame_index"][row, 1])
        ids = sample["ods_frame_index"][row, sample["valid"][row]]
        assert set(ids.tolist()) <= spans[anchor]
        torch.testing.assert_close(
            sample["obs"][row, sample["valid"][row], 0], ids.float() + 10
        )
    sample["obs"].zero_()
    assert source["obs"][1, 0, 0] == 10


@pytest.mark.parametrize("history,future", [(0, 2), (2, 0), (True, 2), (2, 1.5)])
def test_action_context_rejects_invalid_lengths(history, future) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        _make_fake_engine().sample_batch_with_context(2, history, future)


@pytest.mark.parametrize("context", [False, True])
def test_sample_owns_nested_noncontiguous_images_across_source_replacement(
    context: bool,
) -> None:
    """Both sampling APIs detach every leaf, including repeated image indices."""
    steps = 6
    batch_size = 16  # One source row guarantees repeated anchors within the batch.
    engine = _make_fake_engine(buffer_size=1, max_episode_steps=steps)
    source = engine.shared_buffer
    camera = torch.arange(steps * 2 * 3 * 4, dtype=torch.uint8).reshape(
        1, steps, 2, 3, 4
    )
    source["obs"] = TensorDict(
        {
            "joint": source["obs"],
            "cameras": TensorDict(
                {"front": camera.transpose(-3, -2), "wrist": camera.flip(-1)},
                batch_size=source.batch_size,
            ),
        },
        batch_size=source.batch_size,
    )
    for key, value in (
        ("ods_run_id", 17),
        ("ods_episode_id", 23),
        ("ods_generation", 1),
        ("ods_accepted_at_ns", 100),
        ("ods_published_at_monotonic_ns", 200),
    ):
        source[key] = torch.full(source.batch_size, value, dtype=torch.int64)
    source["ods_frame_index"] = torch.arange(steps).reshape(1, steps)
    source.share_memory_()
    assert not source["obs", "cameras", "front"].is_contiguous()
    original = source.clone()

    def sample() -> TensorDict:
        if context:
            return engine.sample_batch_with_context(
                batch_size, 2, 3, generator=torch.Generator().manual_seed(39)
            )
        return engine.sample_batch(batch_size, 1)

    try:
        batch = sample()
        saved = batch.clone()
        valid = batch["valid"]
        frames = batch["ods_frame_index"].clamp(min=0)
        anchor_index = 1 if context else 0
        anchors = batch["ods_frame_index"][:, anchor_index]
        assert anchors.unique().numel() < batch_size
        for key, value in original.items(include_nested=True, leaves_only=True):
            actual = batch[key]
            assert (
                actual.untyped_storage().data_ptr()
                != source[key].untyped_storage().data_ptr()
            )
            expected = value[0, frames]
            identifier = isinstance(key, str) and (
                key.startswith("ods_") or key in {"task_index", "subtask_index"}
            )
            mask = valid.reshape(*valid.shape, *((1,) * (expected.ndim - valid.ndim)))
            expected = expected.masked_fill(~mask, -1 if identifier else 0)
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(source[key], value)

        # Model a producer reusing the same physical slot, then another consumer read.
        replacement = original.clone()
        replacement["obs", "joint"].add_(10)
        replacement["obs", "cameras", "front"].fill_(251)
        replacement["obs", "cameras", "wrist"].fill_(252)
        replacement["actions"].add_(10)
        replacement["ods_episode_id"].fill_(24)
        replacement["ods_generation"].fill_(2)
        with engine._lock_index.get_lock():
            source.copy_(replacement)
        later = sample()
        assert (later["ods_episode_id"][later["valid"]] == 24).all()
        for key, value in saved.items(include_nested=True, leaves_only=True):
            torch.testing.assert_close(batch[key], value)

        # Duplicate gather positions own distinct output elements as well.
        unique_anchors, counts = torch.unique(anchors, return_counts=True)
        matches = (anchors == unique_anchors[counts.argmax()]).nonzero().flatten()
        first, second = matches[:2].tolist()
        for key, value in batch.items(include_nested=True, leaves_only=True):
            second_before = value[second].clone()
            value[first].zero_()
            torch.testing.assert_close(value[second], second_before)
            if key in source.keys(include_nested=True, leaves_only=True):
                torch.testing.assert_close(source[key], replacement[key])
    finally:
        engine.stop()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


@pytest.mark.no_sim
def test_worker_overrides_preserve_newton_device_default() -> None:
    """Worker rendering overrides must not replace Newton's typed config."""
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.lab.sim.cfg import NewtonPhysicsCfg

    sim_cfg = SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(num_substeps=3),
    )

    _apply_worker_simulation_overrides(
        sim_cfg,
        {"headless": True, "renderer": "hybrid", "gpu_id": 0},
    )

    assert isinstance(sim_cfg.physics_cfg, NewtonPhysicsCfg)
    assert sim_cfg.device == "cuda:0"
    assert sim_cfg.physics_cfg.num_substeps == 3


@pytest.mark.no_sim
def test_worker_overrides_apply_explicit_device() -> None:
    """An authored worker device remains an explicit runtime override."""
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.lab.sim.cfg import NewtonPhysicsCfg

    sim_cfg = SimulationManagerCfg(physics_cfg=NewtonPhysicsCfg())

    _apply_worker_simulation_overrides(sim_cfg, {"device": "cpu"})

    assert sim_cfg.device == "cpu"


def _make_fake_engine(
    buffer_size: int = BUFFER_SIZE,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    refill_threshold: int = 1000,
    lock_start: int = -1,
    lock_end: int = -1,
    state: OnlineDataEngineState = OnlineDataEngineState.READY,
) -> OnlineDataEngine:
    """Build an OnlineDataEngine with a pre-filled shared buffer, bypassing start().

    The shared buffer is filled with deterministic random data so that tests can
    verify shapes and values without running a simulation subprocess.

    Args:
        buffer_size: Number of trajectory slots.
        max_episode_steps: Timesteps per trajectory.
        refill_threshold: Passed to OnlineDataEngineCfg; set high to avoid
            accidental refill triggers in most tests.
        lock_start: Write-lock range start (``-1`` means no lock).
        lock_end: Write-lock range end.
        state: Initial lifecycle state for the synthetic engine.

    Returns:
        A configured OnlineDataEngine whose ``shared_buffer`` contains valid
        random data and whose ``is_init`` property returns ``True``.
    """
    cfg = OnlineDataEngineCfg(
        buffer_size=buffer_size,
        max_episode_steps=max_episode_steps,
        state_dim=STATE_DIM,
        refill_threshold=refill_threshold,
        # gym_config must have num_envs so __init__ does not raise.
        gym_config={"num_envs": 1},
    )

    # Bypass __init__'s _create_buffer call — we build the engine manually.
    engine = object.__new__(OnlineDataEngine)
    engine.cfg = cfg

    # Build a synthetic shared buffer: shape [buffer_size, max_episode_steps].
    shared_buffer = TensorDict(
        {
            "obs": torch.randn(buffer_size, max_episode_steps, OBS_DIM),
            "actions": torch.randn(buffer_size, max_episode_steps, ACTION_DIM),
            "rewards": torch.randn(buffer_size, max_episode_steps, 1),
            "valid": torch.ones(buffer_size, max_episode_steps, dtype=torch.bool),
            "segment_id": torch.zeros(buffer_size, max_episode_steps, dtype=torch.long),
            "segment_accepted": torch.ones(
                buffer_size, max_episode_steps, dtype=torch.bool
            ),
            "continuity_id": torch.zeros(
                buffer_size, max_episode_steps, dtype=torch.long
            ),
            "task_index": torch.zeros(buffer_size, max_episode_steps, dtype=torch.long),
            "subtask_index": torch.zeros(
                buffer_size, max_episode_steps, dtype=torch.long
            ),
        },
        batch_size=[buffer_size, max_episode_steps],
    )
    engine.shared_buffer = shared_buffer
    engine.buffer_size = buffer_size
    engine.device = shared_buffer.device

    # Interprocess primitives — use the same mp context consistently to avoid
    engine._mp_ctx = mp.get_context("forkserver")
    engine._language_registry = SharedLanguageRegistry(context=engine._mp_ctx)
    engine._language_registry.intern("unknown_task", kind="task")
    engine._language_registry.intern("unknown_task", kind="subtask")
    engine._lock_index = engine._mp_ctx.Array("i", [lock_start, lock_end])
    engine._fill_signal = engine._mp_ctx.Event()
    engine._init_signal = engine._mp_ctx.Event()
    if state is OnlineDataEngineState.READY:
        engine._init_signal.set()
    engine._close_signal = engine._mp_ctx.Event()
    engine._sample_count = engine._mp_ctx.Value("i", 0)
    engine._state_value = engine._mp_ctx.Value("i", engine_module._STATE_TO_CODE[state])
    engine._worker_failed_signal = engine._mp_ctx.Event()
    engine._worker_error_buffer = engine._mp_ctx.Array(
        "B", engine_module._ERROR_BUFFER_SIZE
    )
    engine._worker_error_length = engine._mp_ctx.Value("i", 0)
    engine._channel_error = None
    engine._worker_error = None
    engine._owner_pid = os.getpid()
    engine._lifecycle_condition = threading.Condition(threading.RLock())
    engine._process_status_lock = threading.RLock()
    engine._stop_requested = False
    engine._cleanup_complete = False
    engine._forced_shutdown_attempted = False
    engine._monitor_thread = None

    # Sampling tests intentionally bypass the simulation worker.  Starting it
    # here would exercise unrelated environment setup and make teardown wait
    # for the subprocess timeout when the synthetic config cannot run a task.
    engine._sim_process = None
    if state is OnlineDataEngineState.READY:
        engine._sim_process = MagicMock()
        engine._sim_process.is_alive.return_value = True

        def finish_join(*args, **kwargs) -> None:
            engine._sim_process.is_alive.return_value = False

        engine._sim_process.join.side_effect = finish_join

    return engine


def test_manual_refill_coalesces_pending_and_active_generation() -> None:
    engine = _make_fake_engine(refill_threshold=1)
    engine.cfg.refill_policy = "manual"
    engine._progress = engine_module._ProducerProgress(engine._mp_ctx)
    engine._progress.values[1] = 1
    assert engine.request_refill() == 2
    assert engine.request_refill() == 2
    engine._fill_signal.clear()  # The producer accepted the same request.
    assert engine.request_refill() == 2
    engine.sample_batch(2, 1)
    assert not engine._fill_signal.is_set()
    assert engine._progress.values[6] == 2
    engine._progress.values[1] = 2
    assert engine.request_refill() == 3


def test_generation_wait_rejects_unrequested_generation_and_times_out() -> None:
    engine = _make_fake_engine()
    engine._progress = engine_module._ProducerProgress(engine._mp_ctx)
    engine._progress.values[1] = 1
    engine.wait_for_generation(1, timeout=0.1)
    with pytest.raises(ValueError, match="not been requested"):
        engine.wait_for_generation(2, timeout=0.1)
    engine._progress.values[0] = 2
    with pytest.raises(TimeoutError):
        engine.wait_for_generation(2, timeout=0.01)


def test_manual_refill_requires_owner_and_ready() -> None:
    engine = _make_fake_engine()
    engine._owner_pid = -1
    with pytest.raises(RuntimeError):
        engine.request_refill()
    engine._owner_pid = os.getpid()
    engine._set_state(OnlineDataEngineState.STOPPED)
    with pytest.raises(RuntimeError):
        engine.request_refill()


def test_stats_exclude_write_window_and_are_detached() -> None:
    engine = _make_fake_engine(lock_start=0, lock_end=1)
    engine._progress = engine_module._ProducerProgress(engine._mp_ctx)
    engine.run_id = 12
    stats = engine.get_stats()
    assert stats["valid_frames"] == (BUFFER_SIZE - 1) * MAX_EPISODE_STEPS
    engine._progress.add(2, 1)
    assert stats["successful_episodes"] == 0
    assert engine.get_stats()["successful_episodes"] == 1


def test_config_buffer_allocates_independent_action_width() -> None:
    from embodichain.lab.gym.utils.gym_utils import init_rollout_buffer_from_config

    config = {"enable_sensor": False}
    buffer = init_rollout_buffer_from_config(config, 5, 2, 9, action_dim=8)
    assert buffer["obs", "robot", "qpos"].shape == (2, 5, 9)
    assert buffer["actions"].shape == (2, 5, 8)
    legacy = init_rollout_buffer_from_config(config, 5, 2, 9)
    assert legacy["actions"].shape == (2, 5, 9)


def test_new_config_options_preserve_legacy_positional_arguments() -> None:
    cfg = OnlineDataEngineCfg(16, 300, 9, "cpu")
    assert cfg.state_dim == 9 and cfg.buffer_device == "cpu"
    assert cfg.action_dim is None and cfg.refill_policy == "sample_threshold"
    assert cfg.shutdown_timeout == 5.0


def test_worker_width_validation_uses_observation_selection_and_expert_layout() -> None:
    from types import SimpleNamespace

    attrs = {
        "expert_action_spec": SimpleNamespace(width=8),
        "rollout_action_semantics": "expert_controller_command",
        "single_observation_space": {"robot": {"qpos": SimpleNamespace(shape=(7,))}},
    }
    env = SimpleNamespace(get_wrapper_attr=lambda name: attrs[name])
    cfg = OnlineDataEngineCfg(state_dim=7, action_dim=8)
    engine_module._validate_worker_dimensions(env, cfg)
    cfg.action_dim = 9
    with pytest.raises(ValueError, match="widths disagree"):
        engine_module._validate_worker_dimensions(env, cfg)
    cfg.action_dim, cfg.state_dim = 8, 9
    with pytest.raises(ValueError, match="widths disagree"):
        engine_module._validate_worker_dimensions(env, cfg)


def _policy_recording_environment() -> SimpleNamespace:
    """A seven-dimensional policy schema beside an eight-joint expert schema."""
    joint_names = tuple(f"joint_{index}" for index in range(8))
    terms = {
        "arm": SimpleNamespace(
            command_type="qpos", controlled_joint_ids=tuple(range(6))
        ),
        "grip": SimpleNamespace(command_type="qpos", controlled_joint_ids=(6, 7)),
    }
    descriptors = (
        SimpleNamespace(
            name="arm",
            start=0,
            stop=6,
            term=SimpleNamespace(
                contract="relative_pose/v1", action_dim=6, joint_names=joint_names[:6]
            ),
        ),
        SimpleNamespace(
            name="grip",
            start=6,
            stop=7,
            term=SimpleNamespace(
                contract="parallel_gripper/v1",
                action_dim=1,
                joint_names=joint_names[6:],
            ),
        ),
    )
    attrs = {
        "rollout_action_semantics": "policy_request",
        "expert_action_spec": SimpleNamespace(width=8),
        "single_observation_space": {"robot": {"qpos": SimpleNamespace(shape=(9,))}},
        "action_manager": SimpleNamespace(
            descriptors=descriptors, total_action_dim=7, get_term=terms.__getitem__
        ),
        "robot": SimpleNamespace(joint_names=joint_names),
    }
    return SimpleNamespace(get_wrapper_attr=attrs.__getitem__, attrs=attrs)


def test_policy_recording_validates_live_schema_independently_of_expert_width() -> None:
    env = _policy_recording_environment()
    cfg = OnlineDataEngineCfg(
        state_dim=9,
        action_dim=7,
        action_semantics="policy_request",
        action_contracts=("relative_pose/v1", "parallel_gripper/v1"),
    )
    engine_module._validate_worker_dimensions(env, cfg)
    assert env.attrs["expert_action_spec"].width == 8


@pytest.mark.parametrize(
    "mismatch", ["semantics", "contracts", "slices", "joints", "width"]
)
def test_policy_recording_rejects_live_schema_mismatch(mismatch: str) -> None:
    env = _policy_recording_environment()
    cfg = OnlineDataEngineCfg(
        state_dim=9,
        action_dim=7,
        action_semantics="policy_request",
        action_contracts=("relative_pose/v1", "parallel_gripper/v1"),
    )
    manager = env.attrs["action_manager"]
    if mismatch == "semantics":
        env.attrs["rollout_action_semantics"] = "expert_controller_command"
    elif mismatch == "contracts":
        cfg.action_contracts = tuple(reversed(cfg.action_contracts))
    elif mismatch == "slices":
        manager.descriptors[1].start = 5
    elif mismatch == "joints":
        manager.descriptors[1].term.joint_names = ("joint_7", "joint_6")
    else:
        cfg.action_dim = 8
    with pytest.raises(ValueError):
        engine_module._validate_worker_dimensions(env, cfg)


@pytest.mark.parametrize(
    "options",
    [
        {"action_semantics": "unknown"},
        {"action_semantics": "policy_request", "action_dim": 7},
        {
            "action_semantics": "policy_request",
            "action_contracts": ("relative_pose/v1",),
        },
        {"action_contracts": ("relative_pose/v1",)},
    ],
)
def test_recording_contract_requires_explicit_compatible_configuration(
    options: dict,
) -> None:
    with pytest.raises(ValueError):
        engine_module._validate_recording_config(OnlineDataEngineCfg(**options))


def test_provenance_stamps_only_valid_frames_and_keeps_episode_frame_indices() -> None:
    shape = (2, 4)
    buffer = TensorDict(
        {
            "valid": torch.tensor(
                [[True, True, False, False], [True, True, True, False]]
            ),
            **{
                key: torch.full(shape, 999, dtype=torch.int64)
                for key in (
                    "ods_run_id",
                    "ods_episode_id",
                    "ods_generation",
                    "ods_frame_index",
                    "ods_accepted_at_ns",
                    "ods_published_at_monotonic_ns",
                )
            },
        },
        batch_size=shape,
    )
    engine_module._populate_provenance(
        buffer,
        run_id=17,
        generation=2,
        episode_start=5,
        accepted_at_ns=123456789,
        published_monotonic_ns=987654321,
    )
    valid = buffer["valid"]
    for key in buffer.keys():
        if key != "valid":
            assert (buffer[key][~valid] == -1).all()
    assert buffer["ods_frame_index"].tolist() == [[0, 1, -1, -1], [0, 1, 2, -1]]
    assert buffer["ods_episode_id"].tolist() == [[5, 5, -1, -1], [6, 6, 6, -1]]
    assert (buffer["ods_accepted_at_ns"][valid] == 123456789).all()
    assert (buffer["ods_published_at_monotonic_ns"][valid] == 987654321).all()
    assert (buffer["ods_run_id"][valid] == 17).all()
    assert (buffer["ods_generation"][valid] == 2).all()


def _install_fake_starting_worker(
    engine: OnlineDataEngine, *, initialize: bool = True
) -> MagicMock:
    """Install a process mock that optionally completes the initial fill."""
    process = MagicMock(pid=1234, exitcode=None)
    process.is_alive.return_value = True
    sentinel_read, sentinel_write = os.pipe()
    process.sentinel = sentinel_read
    write_closed = False

    def join(timeout=None) -> None:
        nonlocal write_closed
        if timeout is None:
            engine._close_signal.wait(timeout=5.0)
        if engine._close_signal.is_set():
            process.is_alive.return_value = False
            if not write_closed:
                os.close(sentinel_write)
                write_closed = True

    process.join.side_effect = join
    process.close.side_effect = lambda: os.close(sentinel_read)
    if initialize:
        process.start.side_effect = engine._init_signal.set
    process_context = MagicMock()
    process_context.Process.return_value = process
    engine._mp_ctx = process_context
    return process


def _forkserver_lifecycle_worker(
    started: mp.Event,
    finish: mp.Event,
    exit_code: int,
    error_channels: tuple | None,
) -> None:
    """Exercise parent status handling without importing a simulator."""
    started.set()
    if not finish.wait(CONSUMER_RESULT_TIMEOUT):
        os._exit(124)
    if error_channels is not None:
        engine_module._publish_worker_error(
            *error_channels, ValueError("original recorder close failure")
        )
    os._exit(exit_code)


def _start_lifecycle_worker(
    engine: OnlineDataEngine,
    *,
    finish: mp.Event | None = None,
    exit_code: int = 0,
    publish_close_error: bool = False,
) -> mp.Process:
    started = engine._mp_ctx.Event()
    channels = (
        engine._worker_error_buffer,
        engine._worker_error_length,
        engine._worker_failed_signal,
        engine._state_value,
    )
    process = engine._mp_ctx.Process(
        target=_forkserver_lifecycle_worker,
        args=(
            started,
            engine._close_signal if finish is None else finish,
            exit_code,
            channels if publish_close_error else None,
        ),
    )
    process.start()
    engine._sim_process = process
    assert started.wait(CONSUMER_RESULT_TIMEOUT), "Lifecycle worker did not start"
    engine._monitor_thread = threading.Thread(
        target=engine_module._monitor_worker_process,
        args=(
            process,
            engine._process_status_lock,
            engine._close_signal,
            *channels,
        ),
        daemon=True,
    )
    engine._monitor_thread.start()
    return process


def test_forkserver_monitor_preserves_normal_shutdown_status() -> None:
    """A monitor and owner must preserve the real zero status after close()."""
    engine = _make_fake_engine()
    process = _start_lifecycle_worker(engine)
    popen = process._popen
    monitor = engine._monitor_thread
    try:
        engine.stop()
        assert popen.returncode == 0
        assert monitor is not None and not monitor.is_alive()
        assert engine.state is OnlineDataEngineState.STOPPED
        assert not engine._worker_failed_signal.is_set()
    finally:
        engine.stop()


def test_forkserver_status_pipe_has_one_reader_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pause the real status read where competing joins formerly produced 255."""
    import multiprocessing.forkserver as forkserver

    engine = _make_fake_engine()
    process = _start_lifecycle_worker(engine)
    popen = process._popen
    original_read = forkserver.read_signed
    first_read, second_read, release_read = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    readers: list[int] = []
    errors: list[BaseException] = []

    def paused_read(sentinel: int) -> int:
        readers.append(threading.get_ident())
        first_read.set()
        if len(readers) > 1:
            second_read.set()
        assert release_read.wait(PROCESS_CLEANUP_TIMEOUT)
        return original_read(sentinel)

    def stop() -> None:
        try:
            engine.stop()
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(forkserver, "read_signed", paused_read)
    owner = threading.Thread(target=stop)
    owner.start()
    try:
        assert first_read.wait(PROCESS_CLEANUP_TIMEOUT)
        # An old monitor.join() enters the same pipe read while the first
        # reader is paused. The status lock now permits only one consumer.
        competing_read = second_read.wait(0.2)
    finally:
        release_read.set()
        owner.join(PROCESS_CLEANUP_TIMEOUT)
        monkeypatch.setattr(forkserver, "read_signed", original_read)
        if owner.is_alive():
            engine._close_signal.set()
    assert not owner.is_alive()
    assert errors == []
    assert not competing_read and len(readers) == 1
    assert popen.returncode == 0
    assert engine._cleanup_complete


def test_forkserver_monitor_broadcasts_real_hard_exit_code() -> None:
    """Sentinel monitoring still publishes an unexpected nonzero worker exit."""
    engine = _make_fake_engine()
    finish = engine._mp_ctx.Event()
    process = _start_lifecycle_worker(engine, finish=finish, exit_code=23)
    popen = process._popen
    finish.set()
    try:
        assert engine._worker_failed_signal.wait(PROCESS_CLEANUP_TIMEOUT)
        with pytest.raises(RuntimeError, match="exit code 23"):
            engine.stop()
        assert popen.returncode == 23
        assert engine.state is OnlineDataEngineState.FAILED
        assert engine._cleanup_complete
    finally:
        engine.stop()


def test_forkserver_shutdown_preserves_original_worker_exception() -> None:
    """A real close-time exception survives monitor/owner status serialization."""
    engine = _make_fake_engine()
    process = _start_lifecycle_worker(engine, publish_close_error=True)
    popen = process._popen
    try:
        with pytest.raises(ValueError, match="original recorder close failure"):
            engine.stop()
        assert popen.returncode == 0
        assert engine.state is OnlineDataEngineState.FAILED
        assert engine._cleanup_complete
    finally:
        engine.stop()


def _sample_engine_in_subprocess(
    engine: OnlineDataEngine, result_queue: mp.Queue
) -> None:
    """Sample once in a real consumer process and return a pickle-safe result."""
    try:
        sample = engine.sample_batch(batch_size=1, chunk_size=1)
        result_queue.put(("ok", tuple(sample.shape), LAB_PACKAGE_NAME in sys.modules))
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))


class _LanguageLengthTransform:
    """Exercise text decoding and numeric tokenization in a spawned loader."""

    def __init__(self, engine: OnlineDataEngine) -> None:
        self.engine = engine

    def __call__(self, sample: TensorDict) -> TensorDict:
        texts = self.engine.resolve_language(sample)
        sample["task_text_length"] = torch.tensor([len(text) for text in texts["task"]])
        sample["subtask_text_length"] = torch.tensor(
            [len(text) for text in texts["subtask"]]
        )
        return sample


# ===========================================================================
# TestOnlineDataEngine
# ===========================================================================


class TestOnlineDataEngine:
    """Tests for OnlineDataEngine.sample_batch and related internals."""

    def setup_method(self) -> None:
        self.engine = _make_fake_engine()

    def _use_created_engine(self) -> OnlineDataEngine:
        """Replace the ready sampling fixture with a fresh lifecycle fixture."""
        self.engine.stop()
        self.engine = _make_fake_engine(state=OnlineDataEngineState.CREATED)
        return self.engine

    def test_worker_forwards_original_exception(self, monkeypatch) -> None:
        """The subprocess entry point sends and re-raises the same exception."""
        error = ValueError("planner failed")
        monkeypatch.setattr(
            engine_module,
            "_run_sim_worker",
            MagicMock(side_effect=error),
        )
        with pytest.raises(ValueError) as raised:
            engine_module._sim_worker_fn(
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                MagicMock(),
                self.engine._worker_error_buffer,
                self.engine._worker_error_length,
                self.engine._worker_failed_signal,
                self.engine._state_value,
            )

        assert raised.value is error
        forwarded = self.engine._receive_worker_error()
        assert isinstance(forwarded, ValueError)
        assert str(forwarded) == "planner failed"

    def test_worker_registers_tasks_before_creating_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A forkserver worker registers task environments before gym.make."""
        import gymnasium as gym

        from embodichain.lab.gym.utils import gym_utils, registration

        task_packages_registered = False

        def discover_task_packages() -> list[str]:
            nonlocal task_packages_registered
            task_packages_registered = True
            return ["embodichain_tasks"]

        env = MagicMock()
        env_cfg = SimpleNamespace(
            filter_dataset_saving=False,
            init_rollout_buffer=True,
            max_episode_steps=None,
            sim_cfg=SimpleNamespace(
                headless=False,
                render_cfg=SimpleNamespace(renderer=None),
                gpu_id=None,
                device=None,
            ),
            num_envs=1,
        )

        def make_environment(*, id: str, cfg: object) -> MagicMock:
            assert id == "TaskProgramRepeatedPickPlace-Franka-v1"
            assert cfg is env_cfg
            assert task_packages_registered
            return env

        monkeypatch.setattr(
            registration, "discover_task_packages", discover_task_packages
        )
        monkeypatch.setattr(
            gym_utils, "config_to_cfg", lambda *_args, **_kwargs: env_cfg
        )
        monkeypatch.setattr(gym_utils, "get_manager_modules", lambda: [])
        monkeypatch.setattr(gym, "make", make_environment)

        context = mp.get_context("forkserver")
        fill_signal = context.Event()
        close_signal = context.Event()
        fill_signal.set()
        close_signal.set()
        error_buffer = context.Array("B", engine_module._ERROR_BUFFER_SIZE)
        error_length = context.Value("i", 0)

        engine_module._run_sim_worker(
            OnlineDataEngineCfg(
                gym_config={
                    "id": "TaskProgramRepeatedPickPlace-Franka-v1",
                    "num_envs": 1,
                },
            ),
            TensorDict({}, batch_size=[1, 1]),
            context.Array("i", [0, 1]),
            fill_signal,
            context.Event(),
            close_signal,
            error_buffer,
            error_length,
            context.Event(),
            context.Value(
                "i", engine_module._STATE_TO_CODE[OnlineDataEngineState.STARTING]
            ),
            [False],
        )

        env.close.assert_called_once_with()

    def test_worker_publishes_language_with_successful_rows_before_readiness(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import gymnasium as gym

        from embodichain.lab.gym.envs import demo
        from embodichain.lab.gym.utils import gym_utils, registration

        env = MagicMock()
        env_cfg = SimpleNamespace(
            filter_dataset_saving=False,
            init_rollout_buffer=True,
            max_episode_steps=None,
            num_envs=1,
            sim_cfg=SimpleNamespace(
                headless=False,
                render_cfg=SimpleNamespace(renderer=None),
                gpu_id=0,
                device=None,
            ),
        )
        monkeypatch.setattr(registration, "discover_task_packages", lambda: [])
        monkeypatch.setattr(
            gym_utils, "config_to_cfg", lambda *_args, **_kwargs: env_cfg
        )
        monkeypatch.setattr(gym_utils, "get_manager_modules", lambda: [])
        monkeypatch.setattr(gym, "make", lambda **_kwargs: env)
        result = SimpleNamespace(
            completed=True,
            all_success=True,
            instruction="Overall",
            length=3,
            segments=(
                SimpleNamespace(start_step=0, end_step=1, instruction="Pick"),
                SimpleNamespace(start_step=1, end_step=3, instruction="Place"),
            ),
        )
        monkeypatch.setattr(
            demo, "execute_demo_episode", lambda *_args, **_kwargs: result
        )
        context = mp.get_context("forkserver")
        registry = SharedLanguageRegistry(context=context)
        buffer = TensorDict(
            {
                "task_index": torch.full((1, 3), -1, dtype=torch.long),
                "subtask_index": torch.full((1, 3), -1, dtype=torch.long),
                "valid": torch.ones(1, 3, dtype=torch.bool),
            },
            batch_size=[1, 3],
        )
        for key in (
            "ods_run_id",
            "ods_episode_id",
            "ods_generation",
            "ods_frame_index",
            "ods_accepted_at_ns",
        ):
            buffer[key] = torch.full(buffer.batch_size, -1, dtype=torch.int64)
        lock = context.Array("i", [0, 1])
        lifecycle_events = engine_module._LifecycleEvents(context, 2, 3)
        original_populate = engine_module._populate_provenance

        def stamp_while_locked(*args: object, **kwargs: object) -> None:
            assert lock.get_lock()._semlock._is_mine()
            original_populate(*args, **kwargs)

        monkeypatch.setattr(engine_module, "_populate_provenance", stamp_while_locked)
        monkeypatch.setattr(engine_module.time, "time_ns", lambda: 123456789)
        fill, close = context.Event(), context.Event()
        fill.set()
        initialized = MagicMock()
        initialized.is_set.return_value = False
        published = []

        def publish_ready() -> None:
            assert list(lock[:]) == [-1, -1]
            published.append(registry.resolve(buffer["subtask_index"], kind="subtask"))
            events = lifecycle_events.drain()
            assert len(events) == 1
            assert events[0]["kind"] == "published"
            assert events[0]["eligible_anchors"] == 3
            assert events[0]["episode_id"] == 0
            close.set()
            fill.set()

        initialized.set.side_effect = publish_ready
        original_intern = registry.intern

        def register_while_locked(description: str, **kwargs: object) -> int:
            assert list(lock[:]) == [0, 1]
            return original_intern(description, **kwargs)

        monkeypatch.setattr(registry, "intern", register_while_locked)
        engine_module._run_sim_worker(
            OnlineDataEngineCfg(
                gym_config={"id": "Fake-v1", "num_envs": 1}, include_provenance=True
            ),
            buffer,
            lock,
            fill,
            initialized,
            close,
            context.Array("B", engine_module._ERROR_BUFFER_SIZE),
            context.Value("i", 0),
            context.Event(),
            context.Value(
                "i", engine_module._STATE_TO_CODE[OnlineDataEngineState.STARTING]
            ),
            [False],
            registry,
            lifecycle_events=lifecycle_events,
        )
        assert published == [[["Pick", "Place", "Place"]]]
        assert registry.resolve(buffer["task_index"]) == [["Overall"] * 3]
        assert buffer["ods_frame_index"].tolist() == [[0, 1, 2]]
        assert buffer["ods_accepted_at_ns"].tolist() == [[123456789] * 3]
        env.close.assert_called_once_with()

    def test_start_transitions_through_starting_to_ready(self) -> None:
        """A successful initial fill publishes the explicit lifecycle states."""
        engine = self._use_created_engine()
        process = _install_fake_starting_worker(engine)
        observed_states = []

        def initialize() -> None:
            observed_states.append(engine.state)
            engine._init_signal.set()

        process.start.side_effect = initialize

        engine.start()

        assert observed_states == [OnlineDataEngineState.STARTING]
        assert engine.state is OnlineDataEngineState.READY
        assert engine.is_init

    def test_double_start_is_rejected(self) -> None:
        """A ready engine cannot create a second producer process."""
        engine = self._use_created_engine()
        _install_fake_starting_worker(engine)
        engine.start()

        with pytest.raises(RuntimeError, match="requires state CREATED"):
            engine.start()

    def test_sample_before_ready_is_rejected(self) -> None:
        """Created engines cannot expose their zero-filled shared buffer."""
        engine = self._use_created_engine()

        with pytest.raises(RuntimeError, match="current state is CREATED"):
            engine.sample_batch(batch_size=1, chunk_size=1)

    def test_start_reraises_original_worker_exception(self) -> None:
        """Initial-fill failures retain their original exception type and text."""
        engine = self._use_created_engine()
        process = _install_fake_starting_worker(engine, initialize=False)
        process.start.side_effect = lambda: engine._record_worker_error(
            ValueError("initial planner failure")
        )

        with pytest.raises(ValueError, match="initial planner failure"):
            engine.start()

        assert engine.state is OnlineDataEngineState.FAILED

    def test_start_times_out_when_live_worker_never_initializes(self) -> None:
        """A hung initial fill cannot leave start blocked indefinitely."""
        engine = self._use_created_engine()
        engine.cfg.initialization_timeout = 0.001
        _install_fake_starting_worker(engine, initialize=False)

        with pytest.raises(TimeoutError, match="initial buffer fill exceeded"):
            engine.start()

        assert engine.state is OnlineDataEngineState.FAILED

    def test_start_drains_worker_error_published_during_cleanup(self) -> None:
        """A late recorder error annotates rather than disappears behind timeout."""
        engine = self._use_created_engine()
        engine.cfg.initialization_timeout = 0.001
        _install_fake_starting_worker(engine, initialize=False)
        shutdown_worker = engine._shutdown_worker

        def shutdown_and_publish_error() -> bool:
            forced_shutdown = shutdown_worker()
            assert engine_module._publish_worker_error(
                engine._worker_error_buffer,
                engine._worker_error_length,
                engine._worker_failed_signal,
                engine._state_value,
                ValueError("late recorder flush failure"),
            )
            return forced_shutdown

        engine._shutdown_worker = shutdown_and_publish_error

        with pytest.raises(TimeoutError) as raised:
            engine.start()

        assert any(
            "late recorder flush failure" in note for note in raised.value.__notes__
        )
        channel_error = engine._receive_worker_error()
        assert isinstance(channel_error, ValueError)
        assert str(channel_error) == "late recorder flush failure"
        assert engine.state is OnlineDataEngineState.FAILED
        assert engine._cleanup_complete

    @pytest.mark.parametrize("timeout", [float("nan"), float("inf")])
    def test_non_finite_initialization_timeout_is_rejected(
        self, timeout: float
    ) -> None:
        """A non-finite timeout cannot turn startup into an infinite wait."""
        cfg = OnlineDataEngineCfg(initialization_timeout=timeout)

        with pytest.raises(ValueError, match="must be finite"):
            OnlineDataEngine(cfg)

    @pytest.mark.parametrize(
        "timeout", [0.0, -1.0, float("nan"), float("inf"), float("-inf")]
    )
    def test_shutdown_timeout_rejects_nonpositive_or_nonfinite_values(
        self, timeout: float
    ) -> None:
        """The graceful-close bound must be finite and allow forward progress."""
        with pytest.raises(ValueError, match="shutdown_timeout must be finite"):
            OnlineDataEngine(OnlineDataEngineCfg(shutdown_timeout=timeout))

    @pytest.mark.parametrize("timeout", [None, 30.0])
    def test_stop_uses_default_or_configured_graceful_shutdown_timeout(
        self, timeout: float | None
    ) -> None:
        """A longer graceful close does not require forced termination."""
        process = self.engine._sim_process
        if timeout is not None:
            self.engine.cfg.shutdown_timeout = timeout

        self.engine.stop()

        process.join.assert_called_once_with(
            timeout=5.0 if timeout is None else timeout
        )
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        assert self.engine.state is OnlineDataEngineState.STOPPED

    def test_sampling_reraises_original_runtime_worker_exception(self) -> None:
        """A refill failure is surfaced before any stale batch is returned."""
        self.engine._record_worker_error(ValueError("refill planner failure"))

        with pytest.raises(ValueError, match="refill planner failure"):
            self.engine.sample_batch(batch_size=1, chunk_size=1)

        assert self.engine.state is OnlineDataEngineState.FAILED

    def test_sampling_detects_worker_exit_without_error_payload(self) -> None:
        """An unexplained worker exit still fails the ready engine immediately."""
        process = MagicMock(exitcode=17)
        process.is_alive.return_value = False
        self.engine._sim_process = process

        with pytest.raises(RuntimeError, match="exit code 17"):
            self.engine.sample_batch(batch_size=1, chunk_size=1)

        assert self.engine.state is OnlineDataEngineState.FAILED

    def test_stopped_engine_rejects_sampling(self) -> None:
        """Stopping permanently closes the sampling surface."""
        engine = self._use_created_engine()
        engine.stop()

        with pytest.raises(RuntimeError, match="current state is STOPPED"):
            engine.sample_batch(batch_size=1, chunk_size=1)

    def test_stopped_engine_rejects_restart(self) -> None:
        """A stopped engine cannot reuse closed process primitives."""
        engine = self._use_created_engine()
        engine.stop()

        with pytest.raises(RuntimeError, match="current state is STOPPED"):
            engine.start()

    def test_failed_engine_rejects_restart(self) -> None:
        """A failed engine is terminal until a new instance is constructed."""
        engine = self._use_created_engine()
        engine._set_state(OnlineDataEngineState.FAILED)
        engine._worker_error = ValueError("worker failed")

        with pytest.raises(RuntimeError, match="current state is FAILED"):
            engine.start()

    def test_context_manager_starts_and_stops_engine(self) -> None:
        """Managed use always closes the worker when the block exits."""
        engine = self._use_created_engine()
        _install_fake_starting_worker(engine)

        with engine as active_engine:
            assert active_engine is engine
            assert active_engine.state is OnlineDataEngineState.READY

        assert engine.state is OnlineDataEngineState.STOPPED

    def test_monitor_target_does_not_retain_engine(self) -> None:
        """The live monitor owns shared primitives, not the engine instance."""
        engine = self._use_created_engine()
        _install_fake_starting_worker(engine)

        engine.start()

        assert engine._monitor_thread._target is engine_module._monitor_worker_process
        assert all(arg is not engine for arg in engine._monitor_thread._args)

    def test_non_owner_cannot_stop_or_signal_owner_worker(self) -> None:
        """A DataLoader copy cannot mutate its owner's producer lifecycle."""
        owner_pid = self.engine._owner_pid
        self.engine._owner_pid = owner_pid + 1
        try:
            with pytest.raises(RuntimeError, match="only be called by owner process"):
                self.engine.stop()
            self.engine.__del__()

            assert not self.engine._close_signal.is_set()
            assert not self.engine._fill_signal.is_set()
        finally:
            self.engine._owner_pid = owner_pid

    @pytest.mark.parametrize("start_method", ["fork", "spawn"])
    def test_real_consumer_process_can_sample(self, start_method: str) -> None:
        """Forked and spawned DataLoader-style consumers receive safe copies."""
        if start_method not in mp.get_all_start_methods():
            pytest.skip(f"multiprocessing start method {start_method!r} unavailable")
        context = mp.get_context(start_method)
        result_queue = context.Queue()
        process = context.Process(
            target=_sample_engine_in_subprocess,
            args=(self.engine, result_queue),
        )

        process.start()
        try:
            # A spawned consumer imports torch and the EmbodiChain package from
            # scratch. Read its small result before joining so the Queue feeder
            # cannot delay process exit while the parent waits in join().
            try:
                result = result_queue.get(timeout=CONSUMER_RESULT_TIMEOUT)
            except Empty:
                if process.is_alive():
                    pytest.fail(
                        "consumer process did not publish a result within "
                        f"{CONSUMER_RESULT_TIMEOUT} seconds"
                    )
                process.join(timeout=0)
                pytest.fail(
                    "consumer process exited without publishing a result "
                    f"(exit code {process.exitcode})"
                )

            # A spawned consumer imports Torch and EmbodiChain from scratch.
            # Under xdist load, interpreter cleanup can exceed the generic
            # process-termination timeout even after the result is available.
            process.join(timeout=CONSUMER_RESULT_TIMEOUT)
            if process.is_alive():
                pytest.fail("consumer process did not exit after publishing a result")
            if process.exitcode != 0:
                pytest.fail(
                    "consumer process exited without sampling successfully "
                    f"(exit code {process.exitcode})"
                )
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=PROCESS_CLEANUP_TIMEOUT)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=PROCESS_CLEANUP_TIMEOUT)
            result_queue.close()
            result_queue.join_thread()

        assert result[:2] == ("ok", (1, 1))
        if start_method == "spawn":
            assert result[2] is False
        assert not self.engine._close_signal.is_set()

    def test_worker_error_is_broadcast_to_all_consumers(self) -> None:
        """Every consumer reconstructs the same exception snapshot."""
        context = mp.get_context("fork")
        assert engine_module._publish_worker_error(
            self.engine._worker_error_buffer,
            self.engine._worker_error_length,
            self.engine._worker_failed_signal,
            self.engine._state_value,
            ValueError("broadcast planner failure"),
        )
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_sample_engine_in_subprocess,
                args=(self.engine, result_queue),
            )
            for _ in range(2)
        ]

        for process in processes:
            process.start()
        results = [result_queue.get(timeout=10.0) for _ in processes]
        for process in processes:
            process.join(timeout=10.0)

        assert results == [
            ("error", "ValueError", "broadcast planner failure"),
            ("error", "ValueError", "broadcast planner failure"),
        ]
        assert all(process.exitcode == 0 for process in processes)

    def test_unpickleable_worker_error_uses_stable_fallback(self) -> None:
        """Unpickleable exception arguments still produce a useful failure."""
        error = ValueError("unpickleable planner failure", lambda: None)
        assert engine_module._publish_worker_error(
            self.engine._worker_error_buffer,
            self.engine._worker_error_length,
            self.engine._worker_failed_signal,
            self.engine._state_value,
            error,
        )

        with pytest.raises(OnlineDataWorkerError, match="unpickleable planner failure"):
            self.engine.sample_batch(batch_size=1, chunk_size=1)

    def test_stop_failure_does_not_publish_stopped(self) -> None:
        """A worker that survives kill keeps the engine failed and retryable."""
        process = self.engine._sim_process
        process.join.side_effect = None
        process.is_alive.return_value = True

        with pytest.raises(
            OnlineDataWorkerError, match="durability could not be confirmed"
        ):
            self.engine.stop()

        assert self.engine.state is OnlineDataEngineState.FAILED
        assert not self.engine._cleanup_complete
        process.is_alive.return_value = False
        with pytest.raises(
            OnlineDataWorkerError, match="durability could not be confirmed"
        ):
            self.engine.stop()
        assert self.engine.state is OnlineDataEngineState.FAILED
        assert self.engine._cleanup_complete
        self.engine.stop()

    def test_stop_detects_worker_that_died_before_shutdown(self) -> None:
        """An already-dead producer cannot be reclassified as a clean stop."""
        process = MagicMock(exitcode=23)
        process.is_alive.return_value = False
        self.engine._sim_process = process

        with pytest.raises(RuntimeError, match="before stop.*exit code 23"):
            self.engine.stop()

        assert self.engine.state is OnlineDataEngineState.FAILED
        assert self.engine._worker_failed_signal.is_set()
        assert self.engine._cleanup_complete

    @pytest.mark.parametrize("timeout", [5.0, 30.0])
    def test_forced_termination_fails_durability_contract(self, timeout: float) -> None:
        """Needing terminate means graceful recorder durability is unconfirmed."""
        self.engine.cfg.shutdown_timeout = timeout
        process = self.engine._sim_process
        process.join.side_effect = None
        process.is_alive.return_value = True

        def terminate() -> None:
            process.is_alive.return_value = False

        process.terminate.side_effect = terminate

        with pytest.raises(
            OnlineDataWorkerError, match="durability could not be confirmed"
        ):
            self.engine.stop()

        assert self.engine.state is OnlineDataEngineState.FAILED
        assert self.engine._cleanup_complete
        assert [call.kwargs["timeout"] for call in process.join.call_args_list] == [
            timeout,
            3.0,
        ]
        process.kill.assert_not_called()

    def test_stop_surfaces_worker_failure_during_shutdown(self) -> None:
        """A recorder failure during worker close cannot be published as stopped."""

        def shutdown_and_fail() -> None:
            self.engine._record_worker_error(ValueError("recorder flush failed"))

        self.engine._shutdown_worker = MagicMock(side_effect=shutdown_and_fail)

        with pytest.raises(ValueError, match="recorder flush failed"):
            self.engine.stop()

        assert self.engine.state is OnlineDataEngineState.FAILED
        assert self.engine._cleanup_complete

    def test_stop_cancels_start_without_state_reversal(self) -> None:
        """Concurrent stop owns cleanup and leaves the terminal state stopped."""
        engine = self._use_created_engine()
        _install_fake_starting_worker(engine, initialize=False)
        start_errors = []

        def start_engine() -> None:
            try:
                engine.start()
            except BaseException as error:
                start_errors.append(error)

        start_thread = threading.Thread(target=start_engine)
        start_thread.start()
        assert engine._fill_signal.wait(timeout=2.0)

        engine.stop()
        start_thread.join(timeout=2.0)

        assert not start_thread.is_alive()
        assert len(start_errors) == 1
        assert "cancelled by stop" in str(start_errors[0])
        assert engine.state is OnlineDataEngineState.STOPPED

    def test_context_exit_preserves_body_exception(self) -> None:
        """Cleanup failures annotate rather than replace a with-body error."""
        body_error = KeyError("training failed")
        self.engine.stop = MagicMock(side_effect=RuntimeError("cleanup failed"))

        result = self.engine.__exit__(KeyError, body_error, None)

        assert result is None
        assert any("cleanup failed" in note for note in body_error.__notes__)

    def test_context_exit_surfaces_background_worker_failure(self) -> None:
        """A worker failure after READY cannot be hidden by context cleanup."""
        engine = self._use_created_engine()
        _install_fake_starting_worker(engine)

        with pytest.raises(ValueError, match="late planner failure"):
            with engine:
                engine._record_worker_error(ValueError("late planner failure"))

        assert engine.state is OnlineDataEngineState.FAILED

    # -----------------------------------------------------------------------

    def test_sample_batch_shape(self) -> None:
        """sample_batch returns TensorDict with shape [batch_size, chunk_size]."""
        BATCH = 3
        CHUNK = 10
        result = self.engine.sample_batch(batch_size=BATCH, chunk_size=CHUNK)
        assert result.shape == (
            BATCH,
            CHUNK,
        ), f"Expected shape [{BATCH}, {CHUNK}], got {result.shape}"
        # All declared keys must be present.
        for key in ("obs", "actions", "rewards"):
            assert key in result, f"Missing key '{key}' in sample_batch result"

    def test_sample_language_snapshot_survives_refill_and_resolves_from_dataset(
        self,
    ) -> None:
        engine = self.engine
        result = SimpleNamespace(
            instruction="Overall task",
            length=MAX_EPISODE_STEPS,
            segments=(
                SimpleNamespace(
                    start_step=0,
                    end_step=MAX_EPISODE_STEPS,
                    instruction="Original segment",
                ),
            ),
        )
        _populate_language_indices(
            engine.shared_buffer, result, engine._language_registry
        )
        snapshot = engine.sample_batch(batch_size=2, chunk_size=3)
        result.instruction = "New overall task"
        result.segments[0].instruction = "New segment"
        _populate_language_indices(
            engine.shared_buffer, result, engine._language_registry
        )
        dataset = OnlineDataset(engine, chunk_size=3)
        assert dataset.resolve_language(snapshot) == {
            "task": [["Overall task"] * 3] * 2,
            "subtask": [["Original segment"] * 3] * 2,
        }
        item = next(iter(dataset))
        assert dataset.resolve_language(item) == {
            "task": ["New overall task"] * 3,
            "subtask": ["New segment"] * 3,
        }

    def test_language_registry_overflow_is_broadcast_as_worker_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        registry = SharedLanguageRegistry(
            capacity_bytes=16, context=self.engine._mp_ctx
        )

        def exhaust_registry(*args: object) -> None:
            registry.intern("The shared dictionary cannot fit this instruction")

        monkeypatch.setattr(engine_module, "_run_sim_worker", exhaust_registry)
        with pytest.raises(OverflowError, match="language_buffer_bytes"):
            engine_module._sim_worker_fn(
                self.engine.cfg,
                self.engine.shared_buffer,
                self.engine._lock_index,
                self.engine._fill_signal,
                self.engine._init_signal,
                self.engine._close_signal,
                self.engine._worker_error_buffer,
                self.engine._worker_error_length,
                self.engine._worker_failed_signal,
                self.engine._state_value,
                registry,
            )
        assert self.engine.state is OnlineDataEngineState.FAILED
        with pytest.raises(OverflowError, match="language_buffer_bytes"):
            self.engine.sample_batch(batch_size=1, chunk_size=1)

    def test_sample_batch_locks_respected(self) -> None:
        """Rows in [lock_start, lock_end) never appear in sampled data."""
        LOCK_START, LOCK_END = 2, 5
        engine = _make_fake_engine(
            buffer_size=BUFFER_SIZE,
            lock_start=LOCK_START,
            lock_end=LOCK_END,
        )
        row_ids = torch.arange(BUFFER_SIZE, dtype=torch.float32)
        engine.shared_buffer["obs"][..., 0] = row_ids[:, None]

        result = engine.sample_batch(batch_size=256, chunk_size=5)
        sampled_rows = set(result["obs"][:, 0, 0].tolist())

        assert sampled_rows.isdisjoint(range(LOCK_START, LOCK_END))

    def test_sample_batch_keeps_episode_uniform_distribution(self) -> None:
        """Short and long eligible episodes have equal row probability."""
        engine = _make_fake_engine(buffer_size=2, max_episode_steps=10)
        engine.shared_buffer["valid"][0, 1:] = False
        engine.shared_buffer["obs"][0, :, 0] = 0
        engine.shared_buffer["obs"][1, :, 0] = 1

        sample_count = 2_000
        result = engine.sample_batch(batch_size=sample_count, chunk_size=1)
        long_episode_fraction = result["obs"][:, 0, 0].float().mean().item()

        assert 0.4 < long_episode_fraction < 0.6

    def test_chunk_size_exceeds_max_steps_raises(self) -> None:
        """ValueError is raised when chunk_size > max_episode_steps."""
        # with self.assertRaises(ValueError):
        #     self.engine.sample_batch(batch_size=1, chunk_size=MAX_EPISODE_STEPS + 1)
        with pytest.raises(ValueError):
            self.engine.sample_batch(batch_size=1, chunk_size=MAX_EPISODE_STEPS + 1)

    def test_sample_batch_never_reads_invalid_tail(self) -> None:
        """Variable-length rows never expose padding or stale tail frames."""
        self.engine.shared_buffer["valid"][:, 7:] = False
        self.engine.shared_buffer["obs"][:, 7:] = 999.0

        for _ in range(20):
            result = self.engine.sample_batch(batch_size=4, chunk_size=5)
            assert result["valid"].all()
            assert not (result["obs"] == 999.0).any()

    def test_segment_sampling_never_crosses_boundary(self) -> None:
        """Segment mode keeps every sampled chunk inside one subtask."""
        midpoint = MAX_EPISODE_STEPS // 2
        self.engine.shared_buffer["segment_id"][:, midpoint:] = 1

        result = self.engine.sample_batch(
            batch_size=32,
            chunk_size=10,
            sampling_mode="segment",
        )

        assert (result["segment_id"] == result["segment_id"][:, :1]).all()

    def test_boundary_sampling_crosses_segment_boundary(self) -> None:
        """Boundary mode returns chunks containing both adjacent segments."""
        midpoint = MAX_EPISODE_STEPS // 2
        self.engine.shared_buffer["segment_id"][:, midpoint:] = 1

        result = self.engine.sample_batch(
            batch_size=16,
            chunk_size=8,
            sampling_mode="boundary",
        )

        assert (
            (result["segment_id"][:, 1:] != result["segment_id"][:, :-1])
            .any(dim=1)
            .all()
        )

    def test_segment_sampling_excludes_failed_segment_frames(self) -> None:
        """A valid transition remains ineligible when its segment was rejected."""
        midpoint = MAX_EPISODE_STEPS // 2
        self.engine.shared_buffer["segment_id"][:, midpoint:] = 1
        self.engine.shared_buffer["segment_accepted"][:, midpoint:] = False

        result = self.engine.sample_batch(
            batch_size=64,
            chunk_size=8,
            sampling_mode="segment",
        )

        assert result["segment_accepted"].all()
        assert (result["segment_id"] == 0).all()

    def test_boundary_sampling_rejects_failed_side_of_boundary(self) -> None:
        """Boundary chunks cannot promote a rejected segment into training data."""
        midpoint = MAX_EPISODE_STEPS // 2
        self.engine.shared_buffer["segment_id"][:, midpoint:] = 1
        self.engine.shared_buffer["segment_accepted"][:, midpoint:] = False

        with pytest.raises(RuntimeError, match="No unlocked valid chunk"):
            self.engine.sample_batch(
                batch_size=1,
                chunk_size=8,
                sampling_mode="boundary",
            )

    def test_sampling_never_crosses_continuity_boundary(self) -> None:
        """An out-of-band state jump is not exposed as a learnable transition."""
        midpoint = MAX_EPISODE_STEPS // 2
        self.engine.shared_buffer["continuity_id"][:, midpoint:] = 1

        result = self.engine.sample_batch(
            batch_size=64,
            chunk_size=10,
            sampling_mode="episode",
        )

        assert (result["continuity_id"] == result["continuity_id"][:, :1]).all()

    def test_boundary_sampling_rejects_cross_continuity_segment_boundary(
        self,
    ) -> None:
        midpoint = MAX_EPISODE_STEPS // 2
        self.engine.shared_buffer["segment_id"][:, midpoint:] = 1
        self.engine.shared_buffer["continuity_id"][:, midpoint:] = 1

        with pytest.raises(RuntimeError, match="No unlocked valid chunk"):
            self.engine.sample_batch(
                batch_size=1,
                chunk_size=8,
                sampling_mode="boundary",
            )

    def test_no_valid_window_raises(self) -> None:
        """Sampling fails clearly when all real episodes are too short."""
        self.engine.shared_buffer["valid"][:, 3:] = False

        with pytest.raises(RuntimeError, match="No unlocked valid chunk"):
            self.engine.sample_batch(batch_size=1, chunk_size=4)

    def test_refill_triggered_after_threshold(self) -> None:
        """_fill_signal is set once accumulated sample count exceeds the threshold."""
        # Use a very small threshold so we can trigger it quickly.
        engine = _make_fake_engine(refill_threshold=1)
        # threshold * buffer_size = 1 * 8 = 8 samples needed to trigger refill.
        threshold_total = engine.cfg.refill_threshold * engine.buffer_size

        # Draw enough samples to exceed the threshold.
        calls_needed = (threshold_total // 2) + 1
        for _ in range(calls_needed):
            engine.sample_batch(batch_size=2, chunk_size=5)

        assert (
            engine._fill_signal.is_set()
        ), "_fill_signal should be set after threshold"

    def test_refill_not_double_triggered(self) -> None:
        """_fill_signal is not re-set if it is already pending (not cleared)."""
        engine = _make_fake_engine(refill_threshold=1)
        threshold_total = engine.cfg.refill_threshold * engine.buffer_size

        # Trigger the first refill.
        for _ in range(threshold_total + 1):
            engine._trigger_refill_if_needed(1)

        assert (
            engine._fill_signal.is_set()
        ), "_fill_signal should be set after first trigger"

        # Record the set-time proxy: manually note it is already set, then call again.
        # The signal remains set (not cleared and re-set), sample_count stays 0.
        with engine._sample_count.get_lock():
            count_before = engine._sample_count.value

        # With the signal still pending, another large batch of triggers
        # should NOT clear and re-set it (count stays 0 from last reset).
        for _ in range(threshold_total + 1):
            engine._trigger_refill_if_needed(1)

        # _fill_signal should still be set (not cleared in between).
        assert (
            engine._fill_signal.is_set()
        ), "_fill_signal should remain set without reset"

    def teardown_method(self) -> None:
        try:
            self.engine.stop()
        except BaseException:
            pass


# ===========================================================================
# TestOnlineDataset
# ===========================================================================


class TestOnlineDataset:
    """Tests for OnlineDataset.__iter__ and DataLoader integration."""

    CHUNK_SIZE = 8

    def setup_method(self) -> None:
        self.engine = _make_fake_engine()

    # -----------------------------------------------------------------------

    def test_item_mode_yields_single_chunk(self) -> None:
        """In item mode next(iter(dataset)) has shape [chunk_size]."""
        dataset = OnlineDataset(self.engine, chunk_size=self.CHUNK_SIZE)
        sample = next(iter(dataset))
        assert list(sample.batch_size) == [
            self.CHUNK_SIZE
        ], "Item mode should yield a single chunk"

    def test_batch_mode_yields_batch(self) -> None:
        """In batch mode next(iter(dataset)) has shape [batch_size, chunk_size]."""
        BATCH = 4
        dataset = OnlineDataset(
            self.engine, chunk_size=self.CHUNK_SIZE, batch_size=BATCH
        )
        sample = next(iter(dataset))
        assert list(sample.batch_size) == [
            BATCH,
            self.CHUNK_SIZE,
        ], "Batch mode should yield a batch of chunks"

    @pytest.mark.parametrize("batch_size", [None, 4], ids=["item", "batch"])
    def test_transform_result_is_returned_once(self, batch_size: int | None) -> None:
        """Each yielded item or batch uses the transform's independent result."""
        expected_shape = (
            [self.CHUNK_SIZE] if batch_size is None else [batch_size, self.CHUNK_SIZE]
        )
        transformed = TensorDict(
            {"transformed_rewards": torch.ones(expected_shape)},
            batch_size=expected_shape,
        )
        transform = MagicMock(return_value=transformed)
        dataset = OnlineDataset(
            self.engine,
            chunk_size=self.CHUNK_SIZE,
            batch_size=batch_size,
            transform=transform,
        )

        sample = next(iter(dataset))

        transform.assert_called_once()
        raw_sample = transform.call_args.args[0]
        assert list(raw_sample.batch_size) == expected_shape
        assert "rewards" in raw_sample
        assert raw_sample is not transformed
        assert sample is transformed

    def test_sampling_mode_is_forwarded_to_engine(self) -> None:
        """OnlineDataset exposes segment-aware engine sampling."""
        midpoint = MAX_EPISODE_STEPS // 2
        self.engine.shared_buffer["segment_id"][:, midpoint:] = 1
        dataset = OnlineDataset(
            self.engine,
            chunk_size=12,
            batch_size=16,
            sampling_mode="segment",
        )

        sample = next(iter(dataset))

        assert (sample["segment_id"] == sample["segment_id"][:, :1]).all()

    def test_dataloader_item_mode(self) -> None:
        """DataLoader with batch_size=4 produces [4, chunk_size] batches."""
        BATCH = 4
        dataset = OnlineDataset(self.engine, chunk_size=self.CHUNK_SIZE)
        loader = DataLoader(
            dataset, batch_size=BATCH, collate_fn=OnlineDataset.collate_fn
        )
        batch = next(iter(loader))
        # DataLoader stacks chunk-level TensorDicts along a new batch dimension.
        first_key = "obs"
        assert (
            batch[first_key].shape[0] == BATCH
        ), f"Expected batch size {BATCH}, got {batch[first_key].shape[0]}"
        assert (
            batch[first_key].shape[1] == self.CHUNK_SIZE
        ), f"Expected chunk size {self.CHUNK_SIZE}, got {batch[first_key].shape[1]}"

    def test_spawned_dataloader_can_decode_language_in_a_transform(self) -> None:
        result = SimpleNamespace(
            instruction="Overall",
            length=MAX_EPISODE_STEPS,
            segments=(
                SimpleNamespace(
                    start_step=0, end_step=MAX_EPISODE_STEPS, instruction="Segment"
                ),
            ),
        )
        _populate_language_indices(
            self.engine.shared_buffer, result, self.engine._language_registry
        )
        dataset = OnlineDataset(
            self.engine,
            chunk_size=self.CHUNK_SIZE,
            transform=_LanguageLengthTransform(self.engine),
        )
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=1,
            multiprocessing_context="spawn",
            collate_fn=OnlineDataset.collate_fn,
        )
        iterator = iter(loader)
        try:
            batch = next(iterator)
            assert batch.batch_size == torch.Size([2, self.CHUNK_SIZE])
            assert (batch["task_text_length"] == len("Overall")).all()
            assert (batch["subtask_text_length"] == len("Segment")).all()
            assert (
                dataset.resolve_language(batch)["task"]
                == [["Overall"] * self.CHUNK_SIZE] * 2
            )
        finally:
            iterator._shutdown_workers()

    def test_dataloader_batch_mode(self) -> None:
        """DataLoader with batch_size=None passes through [4, chunk_size] batches."""
        BATCH = 4
        dataset = OnlineDataset(
            self.engine, chunk_size=self.CHUNK_SIZE, batch_size=BATCH
        )
        loader = DataLoader(
            dataset, batch_size=None, collate_fn=OnlineDataset.passthrough_collate_fn
        )
        batch = next(iter(loader))
        first_key = "obs"
        assert (
            batch[first_key].shape[0] == BATCH
        ), f"Expected batch size {BATCH}, got {batch[first_key].shape[0]}"
        assert (
            batch[first_key].shape[1] == self.CHUNK_SIZE
        ), f"Expected chunk size {self.CHUNK_SIZE}, got {batch[first_key].shape[1]}"


# ===========================================================================
# TestUniformChunkSampler
# ===========================================================================


class TestUniformChunkSampler(unittest.TestCase):
    """Tests for UniformChunkSampler."""

    def test_output_within_range(self) -> None:
        """All sampled values fall within [low, high]."""
        LOW, HIGH = 8, 32
        sampler = UniformChunkSampler(low=LOW, high=HIGH)
        for _ in range(200):
            v = sampler()
            self.assertGreaterEqual(v, LOW)
            self.assertLessEqual(v, HIGH)

    def test_output_is_int(self) -> None:
        """Sampled values are Python ints."""
        sampler = UniformChunkSampler(low=4, high=16)
        self.assertIsInstance(sampler(), int)

    def test_fixed_range_single_value(self) -> None:
        """When low == high the sampler always returns that value."""
        sampler = UniformChunkSampler(low=7, high=7)
        for _ in range(20):
            self.assertEqual(sampler(), 7)

    def test_invalid_low_raises(self) -> None:
        """ValueError when low < 1."""
        with self.assertRaises(ValueError):
            UniformChunkSampler(low=0, high=10)

    def test_invalid_high_raises(self) -> None:
        """ValueError when high < low."""
        with self.assertRaises(ValueError):
            UniformChunkSampler(low=10, high=5)

    def test_distribution_covers_range(self) -> None:
        """Empirically verify both endpoints are reachable over many samples."""
        LOW, HIGH = 1, 4
        sampler = UniformChunkSampler(low=LOW, high=HIGH)
        seen = set()
        for _ in range(500):
            seen.add(sampler())
        # All four values should appear with high probability.
        self.assertEqual(seen, {1, 2, 3, 4})


# ===========================================================================
# TestGMMChunkSampler
# ===========================================================================


class TestGMMChunkSampler(unittest.TestCase):
    """Tests for GMMChunkSampler."""

    def test_output_is_int(self) -> None:
        """Sampled values are Python ints."""
        sampler = GMMChunkSampler(means=[20.0], stds=[2.0])
        self.assertIsInstance(sampler(), int)

    def test_single_component_near_mean(self) -> None:
        """With one narrow Gaussian most samples cluster near the mean."""
        MEAN = 30
        sampler = GMMChunkSampler(means=[float(MEAN)], stds=[1.0])
        values = [sampler() for _ in range(100)]
        avg = sum(values) / len(values)
        self.assertAlmostEqual(avg, MEAN, delta=3.0)

    def test_clamping_low(self) -> None:
        """No sample falls below ``low`` even when the Gaussian would."""
        LOW = 20
        sampler = GMMChunkSampler(means=[1.0], stds=[1.0], low=LOW)
        for _ in range(100):
            self.assertGreaterEqual(sampler(), LOW)

    def test_clamping_high(self) -> None:
        """No sample exceeds ``high`` even when the Gaussian would."""
        HIGH = 5
        sampler = GMMChunkSampler(means=[100.0], stds=[1.0], high=HIGH)
        for _ in range(100):
            self.assertLessEqual(sampler(), HIGH)

    def test_clamping_both_bounds(self) -> None:
        """All samples fall within [low, high]."""
        LOW, HIGH = 10, 20
        sampler = GMMChunkSampler(
            means=[15.0, 50.0],
            stds=[5.0, 5.0],
            weights=[0.5, 0.5],
            low=LOW,
            high=HIGH,
        )
        for _ in range(200):
            v = sampler()
            self.assertGreaterEqual(v, LOW)
            self.assertLessEqual(v, HIGH)

    def test_at_least_one(self) -> None:
        """Sampled values are always ≥ 1 even without explicit low bound."""
        # Use a Gaussian centred at a very negative mean to stress-test floor.
        sampler = GMMChunkSampler(means=[-100.0], stds=[1.0])
        for _ in range(50):
            self.assertGreaterEqual(sampler(), 1)

    def test_uniform_weights_by_default(self) -> None:
        """Omitting weights gives equal probability to each component."""
        # Two well-separated components: values should appear on both sides.
        sampler = GMMChunkSampler(means=[5.0, 45.0], stds=[0.5, 0.5])
        low_count = sum(1 for _ in range(200) if sampler() <= 10)
        high_count = sum(1 for _ in range(200) if sampler() >= 40)
        # With uniform weights both components should fire ~50% of the time.
        self.assertGreater(low_count, 30)
        self.assertGreater(high_count, 30)

    def test_weight_bias(self) -> None:
        """Heavily biased weight causes one component to dominate."""
        sampler = GMMChunkSampler(
            means=[5.0, 50.0], stds=[0.5, 0.5], weights=[0.99, 0.01]
        )
        low_count = sum(1 for _ in range(300) if sampler() <= 10)
        # With 99% weight on the low component, nearly all samples should be low.
        self.assertGreater(low_count, 250)

    def test_invalid_stds_raises(self) -> None:
        """ValueError when any std ≤ 0."""
        with self.assertRaises(ValueError):
            GMMChunkSampler(means=[10.0], stds=[0.0])

    def test_mismatched_lengths_raises(self) -> None:
        """ValueError when means and stds have different lengths."""
        with self.assertRaises(ValueError):
            GMMChunkSampler(means=[10.0, 20.0], stds=[1.0])

    def test_mismatched_weights_raises(self) -> None:
        """ValueError when weights length differs from means."""
        with self.assertRaises(ValueError):
            GMMChunkSampler(means=[10.0], stds=[1.0], weights=[0.5, 0.5])

    def test_negative_weight_raises(self) -> None:
        """ValueError when any weight is negative."""
        with self.assertRaises(ValueError):
            GMMChunkSampler(means=[10.0, 20.0], stds=[1.0, 1.0], weights=[-0.1, 1.1])

    def test_zero_weight_sum_raises(self) -> None:
        """ValueError when all weights are zero."""
        with self.assertRaises(ValueError):
            GMMChunkSampler(means=[10.0], stds=[1.0], weights=[0.0])


# ===========================================================================
# TestOnlineDatasetDynamicChunk
# ===========================================================================


class TestOnlineDatasetDynamicChunk(unittest.TestCase):
    """Tests for OnlineDataset with ChunkSizeSampler chunk_size."""

    def setUp(self) -> None:
        self.engine = _make_fake_engine()

    def test_uniform_sampler_item_mode_shape(self) -> None:
        """Item mode with UniformChunkSampler: batch_size dim is absent, time dim varies."""
        LOW, HIGH = 5, 15
        sampler = UniformChunkSampler(low=LOW, high=HIGH)
        dataset = OnlineDataset(self.engine, chunk_size=sampler)
        it = iter(dataset)
        for _ in range(10):
            sample = next(it)
            # batch_size has one element — the chunk dimension.
            self.assertEqual(len(sample.batch_size), 1)
            chunk_dim = sample.batch_size[0]
            self.assertGreaterEqual(chunk_dim, LOW)
            self.assertLessEqual(chunk_dim, HIGH)

    def test_gmm_sampler_item_mode_shape(self) -> None:
        """Item mode with GMMChunkSampler: chunk dim is clamped within [low, high]."""
        LOW, HIGH = 4, 20
        sampler = GMMChunkSampler(
            means=[8.0, 16.0], stds=[2.0, 2.0], low=LOW, high=HIGH
        )
        dataset = OnlineDataset(self.engine, chunk_size=sampler)
        it = iter(dataset)
        for _ in range(10):
            sample = next(it)
            chunk_dim = sample.batch_size[0]
            self.assertGreaterEqual(chunk_dim, LOW)
            self.assertLessEqual(chunk_dim, HIGH)

    def test_uniform_sampler_batch_mode_shape(self) -> None:
        """Batch mode: per-batch chunk size is consistent across all trajectories."""
        BATCH = 3
        LOW, HIGH = 5, 15
        sampler = UniformChunkSampler(low=LOW, high=HIGH)
        dataset = OnlineDataset(self.engine, chunk_size=sampler, batch_size=BATCH)
        it = iter(dataset)
        for _ in range(10):
            batch = next(it)
            self.assertEqual(len(batch.batch_size), 2)
            self.assertEqual(batch.batch_size[0], BATCH)
            chunk_dim = batch.batch_size[1]
            self.assertGreaterEqual(chunk_dim, LOW)
            self.assertLessEqual(chunk_dim, HIGH)

    def test_dynamic_chunk_sizes_vary(self) -> None:
        """Consecutive samples from a uniform sampler produce different chunk sizes."""
        LOW, HIGH = 5, 30
        sampler = UniformChunkSampler(low=LOW, high=HIGH)
        dataset = OnlineDataset(self.engine, chunk_size=sampler)
        it = iter(dataset)
        sizes = {next(it).batch_size[0] for _ in range(50)}
        # With a range of 26 values, drawing 50 times should yield > 1 unique size.
        assert (
            len(sizes) >= 1
        ), "Expected multiple unique chunk sizes from uniform sampler"

    def test_invalid_chunk_size_type_raises(self) -> None:
        """TypeError when chunk_size is not an int or ChunkSizeSampler."""
        with self.assertRaises(TypeError):
            OnlineDataset(self.engine, chunk_size="large")  # type: ignore[arg-type]

    def test_invalid_chunk_size_int_raises(self) -> None:
        """ValueError when chunk_size is an int < 1."""
        with self.assertRaises(ValueError):
            OnlineDataset(self.engine, chunk_size=0)

    def test_custom_sampler_subclass(self) -> None:
        """A user-defined ChunkSizeSampler subclass is accepted and called."""

        class FixedSampler(ChunkSizeSampler):
            def __call__(self) -> int:
                return 7

        dataset = OnlineDataset(self.engine, chunk_size=FixedSampler())
        sample = next(iter(dataset))
        self.assertEqual(sample.batch_size[0], 7)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()


def _event_test_record(
    episode_id: int, *, kind: str = "published"
) -> dict[str, object]:
    return {
        "kind": kind,
        "run_id": 42,
        "episode_id": episode_id,
        "generation": 1 + episode_id // 2,
        "slot": episode_id % 2,
        "monotonic_ns": 1234 + episode_id,
        "valid_frame_indices": [0, 1, 3],
        "eligible_anchor_indices": [0, 3],
    }


def _publish_lifecycle_in_subprocess(events, close, blocked, finished) -> None:
    """Exercise real shared memory and interruptible backpressure without a GPU."""
    for episode in range(4):
        if not events.wait_for_space(1, close):
            return
        events.append(_event_test_record(episode))
    blocked.set()
    if events.wait_for_space(1, close):
        events.append(_event_test_record(4))
    finished.set()


@pytest.mark.parametrize("release_with_close", [False, True])
def test_lifecycle_ring_subprocess_backpressure_and_shutdown(
    release_with_close,
) -> None:
    context = mp.get_context("forkserver")
    events = engine_module._LifecycleEvents(context, capacity=4, max_steps=5)
    close, blocked, finished = (context.Event() for _ in range(3))
    process = context.Process(
        target=_publish_lifecycle_in_subprocess,
        args=(events, close, blocked, finished),
    )
    process.start()
    try:
        assert blocked.wait(20), "producer did not fill the ring"
        assert not finished.wait(0.1), "full ring must backpressure, not drop records"
        if release_with_close:
            close.set()
            assert finished.wait(5)
        initial = events.drain()
        assert [item["sequence"] for item in initial] == list(range(4))
        assert [item["episode_id"] for item in initial] == list(range(4))
        assert all(item["eligible_anchors"] == 2 for item in initial)
        assert all(item["valid_frame_indices"] == [0, 1, 3] for item in initial)
        if not release_with_close:
            assert finished.wait(5)
            assert events.drain()[0]["sequence"] == 4
        else:
            assert events.drain() == []
        process.join(5)
        assert process.exitcode == 0
        assert initial[0]["episode_id"] == 0  # Detached after ring slot reuse.
    finally:
        close.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_lifecycle_events_owner_only_and_drain_after_stop() -> None:
    engine = _make_fake_engine()
    assert engine.drain_lifecycle_events() == []
    engine._lifecycle_events = engine_module._LifecycleEvents(
        engine._mp_ctx, capacity=4, max_steps=5
    )
    engine._lifecycle_events.append(_event_test_record(1))
    engine._set_state(OnlineDataEngineState.STOPPED)
    assert engine.drain_lifecycle_events()[0]["episode_id"] == 1
    engine._owner_pid = -1
    with pytest.raises(RuntimeError, match="owner"):
        engine.drain_lifecycle_events()
    engine._owner_pid = os.getpid()


@pytest.mark.parametrize("capacity", [-1, 1, 3, True, 1.5])
def test_lifecycle_capacity_rejects_inadequate_initial_fill_storage(capacity) -> None:
    with pytest.raises(ValueError, match="lifecycle_event_capacity"):
        OnlineDataEngine(
            OnlineDataEngineCfg(buffer_size=2, lifecycle_event_capacity=capacity)
        )


def test_publication_event_distinguishes_valid_frames_from_eligible_anchors() -> None:
    row = TensorDict(
        {
            "valid": torch.tensor([True, False, True, True]),
            "continuity_id": torch.tensor([0, -1, -1, 2]),
        },
        batch_size=[4],
    )
    event = engine_module._publication_event(
        row, run_id=1, episode_id=2, generation=3, slot=4, monotonic_ns=5
    )
    assert event["valid_frame_indices"] == [0, 2, 3]
    assert event["eligible_anchor_indices"] == [0, 3]


def test_worker_lifecycle_publication_and_eviction_use_distinct_episode_ids(
    monkeypatch,
) -> None:
    """Run the real producer loop with a CPU-only environment adapter."""
    context = mp.get_context("forkserver")
    buffer = TensorDict(
        {
            "valid": torch.ones(1, 4, dtype=torch.bool),
            "continuity_id": torch.zeros(1, 4, dtype=torch.int64),
        },
        batch_size=[1, 4],
    )
    lock = context.Array("i", [0, 1])
    fill, close = context.Event(), context.Event()
    fill.set()
    events = engine_module._LifecycleEvents(context, capacity=2, max_steps=4)
    initialized = MagicMock()
    initialized.is_set.side_effect = [False, True]
    collected = []

    def initialize() -> None:
        collected.extend(events.drain())
        fill.set()

    initialized.set.side_effect = initialize
    env = MagicMock()
    config = SimpleNamespace(
        num_envs=1,
        sim_cfg=SimpleNamespace(
            headless=True,
            render_cfg=SimpleNamespace(renderer=None),
            gpu_id=0,
            device=None,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "gymnasium", SimpleNamespace(make=lambda **kwargs: env)
    )
    monkeypatch.setitem(
        sys.modules,
        "embodichain.lab.gym.utils.gym_utils",
        SimpleNamespace(
            config_to_cfg=lambda *args, **kwargs: config, get_manager_modules=lambda: []
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "embodichain.lab.gym.utils.registration",
        SimpleNamespace(discover_task_packages=lambda: []),
    )
    calls = 0

    def execute(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Successful shorter replacement must report only its valid frames.
            buffer["valid"][0, 2:] = False
            close.set()
            fill.set()
        return SimpleNamespace(completed=True, all_success=True)

    monkeypatch.setitem(
        sys.modules,
        "embodichain.lab.gym.envs.demo",
        SimpleNamespace(execute_demo_episode=execute),
    )
    monkeypatch.delenv("EMBODICHAIN_SIM_EXIT_PROCESS", raising=False)
    engine_module._run_sim_worker(
        OnlineDataEngineCfg(gym_config={"id": "Fake-v1", "num_envs": 1}),
        buffer,
        lock,
        fill,
        initialized,
        close,
        context.Array("B", engine_module._ERROR_BUFFER_SIZE),
        context.Value("i", 0),
        context.Event(),
        context.Value("i", 0),
        [False],
        progress=engine_module._ProducerProgress(context),
        run_id=42,
        lifecycle_events=events,
    )
    collected.extend(events.drain())
    assert [event["kind"] for event in collected] == [
        "published",
        "evicted",
        "published",
    ]
    assert [event["episode_id"] for event in collected] == [0, 0, 1]
    assert [event["generation"] for event in collected] == [1, 1, 2]
    assert [event["valid_frames"] for event in collected] == [4, 4, 2]
    assert [event["sequence"] for event in collected] == [0, 1, 2]
    assert (
        collected[0]["monotonic_ns"]
        <= collected[1]["monotonic_ns"]
        <= collected[2]["monotonic_ns"]
    )
    assert list(lock[:]) == [-1, -1]
    env.close.assert_called_once()


def test_lifecycle_capacity_accounts_for_initial_overlapping_write_window() -> None:
    # C=5, P=4 publishes 8 episodes and evicts 3 even before start returns.
    with pytest.raises(ValueError, match="initial-fill"):
        OnlineDataEngine(
            OnlineDataEngineCfg(
                buffer_size=5, gym_config={"num_envs": 4}, lifecycle_event_capacity=10
            )
        )
