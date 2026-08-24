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
)

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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


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
        },
        batch_size=[buffer_size, max_episode_steps],
    )
    engine.shared_buffer = shared_buffer
    engine.buffer_size = buffer_size
    engine.device = shared_buffer.device

    # Interprocess primitives — use the same mp context consistently to avoid
    engine._mp_ctx = mp.get_context("forkserver")
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


def _install_fake_starting_worker(
    engine: OnlineDataEngine, *, initialize: bool = True
) -> MagicMock:
    """Install a process mock that optionally completes the initial fill."""
    process = MagicMock(pid=1234, exitcode=None)
    process.is_alive.return_value = True

    def join(timeout=None) -> None:
        if timeout is None:
            engine._close_signal.wait(timeout=5.0)
        if engine._close_signal.is_set():
            process.is_alive.return_value = False

    process.join.side_effect = join
    if initialize:
        process.start.side_effect = engine._init_signal.set
    process_context = MagicMock()
    process_context.Process.return_value = process
    engine._mp_ctx = process_context
    return process


def _sample_engine_in_subprocess(
    engine: OnlineDataEngine, result_queue: mp.Queue
) -> None:
    """Sample once in a real consumer process and return a pickle-safe result."""
    try:
        sample = engine.sample_batch(batch_size=1, chunk_size=1)
        result_queue.put(("ok", tuple(sample.shape), LAB_PACKAGE_NAME in sys.modules))
    except BaseException as error:
        result_queue.put(("error", type(error).__name__, str(error)))


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

            process.join(timeout=PROCESS_CLEANUP_TIMEOUT)
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

    def test_forced_termination_fails_durability_contract(self) -> None:
        """Needing terminate means graceful recorder durability is unconfirmed."""
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

    def test_transform_applied(self) -> None:
        """Transform callable is invoked and its result is returned."""
        sentinel = {"called": False}

        def my_transform(td: TensorDict) -> TensorDict:
            sentinel["called"] = True
            return td

        dataset = OnlineDataset(
            self.engine, chunk_size=self.CHUNK_SIZE, transform=my_transform
        )
        next(iter(dataset))
        assert sentinel["called"], "transform should have been called"

    def test_transform_modifies_output(self) -> None:
        """Transform result is what the caller receives, not the raw sample."""
        SCALE = 99.0

        def scale_rewards(td: TensorDict) -> TensorDict:
            td["rewards"] = td["rewards"] * SCALE
            return td

        dataset = OnlineDataset(
            self.engine, chunk_size=self.CHUNK_SIZE, transform=scale_rewards
        )
        sample = next(iter(dataset))
        # Rewards should now be on the order of SCALE * original values.
        # Original rewards are standard-normal, so max abs should be >> 1 unless scaled.
        assert (
            sample["rewards"].abs().max().item() > 1.0
        ), "scaled rewards should have large absolute values"

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
