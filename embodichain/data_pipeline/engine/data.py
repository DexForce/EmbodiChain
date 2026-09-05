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

import os
import math
import pickle
import sys
import threading
import time
import multiprocessing as mp

from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, Literal
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from multiprocessing.synchronize import Event as MpEvent

import torch
from tensordict import TensorDict
from tqdm import tqdm

from embodichain.utils.logger import log_info, log_error
from embodichain.utils import configclass

__all__ = [
    "OnlineDataEngine",
    "OnlineDataEngineCfg",
    "OnlineDataEngineState",
    "OnlineDataWorkerError",
]

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManagerCfg

_ERROR_BUFFER_SIZE = 64 * 1024


class OnlineDataEngineState(str, Enum):
    """Lifecycle states for :class:`OnlineDataEngine`."""

    CREATED = "CREATED"
    STARTING = "STARTING"
    READY = "READY"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


_STATE_TO_CODE = {state: index for index, state in enumerate(OnlineDataEngineState)}
_CODE_TO_STATE = {code: state for state, code in _STATE_TO_CODE.items()}


class OnlineDataWorkerError(RuntimeError):
    """Fallback error for a worker exception that cannot be reconstructed."""


def _add_exception_note(error: BaseException, note: str) -> None:
    """Attach a PEP 678-style note on every supported Python version."""
    add_note = getattr(error, "add_note", None)
    if add_note is not None:
        add_note(note)
        return
    notes = getattr(error, "__notes__", None)
    if notes is None:
        notes = []
        error.__notes__ = notes
    notes.append(note)


def _forced_shutdown_error() -> OnlineDataWorkerError:
    """Build the error used when graceful worker durability is unknown."""
    return OnlineDataWorkerError(
        "OnlineDataEngine worker required terminate()/kill(); graceful close "
        "and recorder durability could not be confirmed."
    )


@dataclass(frozen=True)
class _WorkerErrorEnvelope:
    """Stable, pickle-safe description of a worker exception."""

    module: str
    qualname: str
    message: str
    representation: str
    exception_payload: bytes | None


def _make_worker_error_envelope(error: BaseException) -> _WorkerErrorEnvelope:
    """Create a stable envelope, retaining the exception when it is picklable."""
    exception_payload = None
    try:
        candidate = pickle.dumps(error, protocol=pickle.HIGHEST_PROTOCOL)
        if len(candidate) < _ERROR_BUFFER_SIZE // 2:
            exception_payload = candidate
    except BaseException:
        pass

    return _WorkerErrorEnvelope(
        module=type(error).__module__,
        qualname=type(error).__qualname__,
        message=str(error)[:4096],
        representation=repr(error)[:4096],
        exception_payload=exception_payload,
    )


def _error_from_envelope(envelope: _WorkerErrorEnvelope) -> BaseException:
    """Reconstruct an original exception or return the stable fallback type."""
    if envelope.exception_payload is not None:
        try:
            error = pickle.loads(envelope.exception_payload)
        except BaseException as decode_error:
            return OnlineDataWorkerError(
                f"Worker raised {envelope.module}.{envelope.qualname}: "
                f"{envelope.message} (exception reconstruction failed: "
                f"{type(decode_error).__name__}: {decode_error})"
            )
        if isinstance(error, BaseException):
            return error

    return OnlineDataWorkerError(
        f"Worker raised {envelope.module}.{envelope.qualname}: {envelope.message}"
    )


def _publish_worker_error(
    error_buffer: SynchronizedArray,
    error_length: Synchronized,
    failed_signal: MpEvent,
    state_value: Synchronized,
    error: BaseException,
) -> bool:
    """Publish a worker exception once to every process sharing the engine."""
    try:
        envelope = _make_worker_error_envelope(error)
        payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(error_buffer):
            envelope = _WorkerErrorEnvelope(
                module=envelope.module,
                qualname=envelope.qualname,
                message=envelope.message[:1024],
                representation=envelope.representation[:1024],
                exception_payload=None,
            )
            payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(error_buffer):
            return False

        with error_buffer.get_lock():
            error_buffer[: len(payload)] = payload
            error_length.value = len(payload)
        failed_signal.set()
        with state_value.get_lock():
            state_value.value = _STATE_TO_CODE[OnlineDataEngineState.FAILED]
    except BaseException:
        return False
    return True


def _monitor_worker_process(
    process: mp.Process,
    close_signal: MpEvent,
    error_buffer: SynchronizedArray,
    error_length: Synchronized,
    failed_signal: MpEvent,
    state_value: Synchronized,
) -> None:
    """Broadcast hard worker exits without retaining the engine instance."""
    try:
        process.join()
        if close_signal.is_set() or failed_signal.is_set():
            return
        error = RuntimeError(
            "OnlineDataEngine simulation worker exited unexpectedly "
            f"(exit code {process.exitcode})."
        )
    except BaseException as caught_error:
        if close_signal.is_set():
            return
        error = caught_error

    _publish_worker_error(
        error_buffer,
        error_length,
        failed_signal,
        state_value,
        error,
    )


@configclass
class OnlineDataEngineCfg:
    buffer_size: int = 16
    """Number of episodes (environment trajectories) that can be stored in the shared buffer at once.
    Must be ≥ num_envs and ideally a multiple of num_envs."""

    max_episode_steps: int = 300
    """Maximum number of timesteps per episode.  Must be ≥ chunk_size used by OnlineDataset."""

    # TODO: This param maybe changed to more general format.
    state_dim: int = 14
    """Dimensionality of the state space."""

    buffer_device: str = "cpu"
    """Device on which the shared buffer is allocated."""

    # TODO: We may support multiple envs in the future.
    gym_config: dict = dict()
    """Gym environment configuration dictionary (already loaded, not a file path).
    The contents depend on the specific environment being used. Default is None."""

    action_config: dict = dict()
    """Action configuration dictionary.  The contents depend on the specific environment and robot being used."""

    refill_threshold: int = 50
    """Total number of samples (refill_threshold * buffer_size) drawn from the shared buffer before a refill is triggered.
    Accumulates across all calls to :meth:`OnlineDataEngine.sample_batch`. When this threshold
    is exceeded the engine signals the simulation subprocess to regenerate the entire buffer,
    amortising the cost of environment simulation over many training steps.
    """

    max_generation_attempts: int = 3
    """Maximum planning/execution attempts for each buffer write transaction."""

    initialization_timeout: float = 300.0
    """Maximum seconds to wait for the worker's initial buffer fill."""


def _apply_worker_simulation_overrides(
    sim_cfg: "SimulationManagerCfg",
    gym_config: dict[str, object],
) -> None:
    """Apply worker-only overrides without replacing the typed physics config.

    ``config_to_cfg`` has already selected and decoded the backend before the
    worker starts.  Mutating that instance keeps backend-specific defaults
    (notably Newton's ``cuda:0`` device) and physics settings intact while
    still applying the worker's headless/render/GPU options.
    """
    sim_cfg.headless = bool(gym_config.get("headless", True))
    sim_cfg.render_cfg.renderer = str(gym_config.get("renderer", "hybrid"))
    sim_cfg.gpu_id = int(gym_config.get("gpu_id", 0))

    # ``None`` means that no runtime override was authored.  Leaving the
    # value untouched is what allows NewtonPhysicsCfg's CUDA default to win
    # over PhysicsBackendCfg's generic CPU default.
    device = gym_config.get("device")
    if device is not None:
        sim_cfg.device = device


# ---------------------------------------------------------------------------
# Subprocess entry point (module-level so it can be pickled by multiprocessing)
# ---------------------------------------------------------------------------


def _run_sim_worker(
    cfg: OnlineDataEngineCfg,
    shared_buffer: TensorDict,
    lock_index: SynchronizedArray,
    fill_signal: MpEvent,
    init_signal: MpEvent,
    close_signal: MpEvent,
    error_buffer: SynchronizedArray,
    error_length: Synchronized,
    failed_signal: MpEvent,
    state_value: Synchronized,
    error_reported: list[bool],
) -> None:
    """Simulation subprocess entry point.

    Builds the gym environment, then waits on *fill_signal*.  Each time the
    signal is raised the subprocess runs enough rollouts to overwrite every
    slot in *shared_buffer* with fresh demonstration data, and advances *lock_index*
    so the main process can avoid sampling from the slot currently being written.
    After the **first** fill completes *init_signal* is set exactly once so the
    main process knows the buffer contains valid data.

    Args:
        cfg: Engine configuration (picklable dataclass).
        shared_buffer: Shared-memory TensorDict of shape
            ``[buffer_size, max_episode_steps, ...]``.
        lock_index: Two-element shared integer array ``[write_start, write_end)``
            indicating which buffer rows are currently being overwritten.
        fill_signal: Event set by the main process to request a refill.
        init_signal: Event set by this worker after the first fill completes.
            Remains set permanently thereafter.
        close_signal: Event set by the main process to request a graceful shutdown.
    """
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import (
        config_to_cfg,
        get_manager_modules,
    )
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.utils.logger import log_info, log_warning

    gym_config: dict = cfg.gym_config
    action_config: dict = cfg.action_config

    # Build env config from the gym configuration dictionary.
    env_cfg = config_to_cfg(gym_config, manager_modules=get_manager_modules())
    env_cfg.filter_dataset_saving = True
    env_cfg.init_rollout_buffer = False
    # The environment must truncate at the exact capacity of the shared row.
    # Otherwise a longer successful plan is silently clipped by the writer and
    # published as if the complete episode had been stored.
    env_cfg.max_episode_steps = int(shared_buffer.batch_size[1])
    _apply_worker_simulation_overrides(env_cfg.sim_cfg, gym_config)

    num_envs: int = env_cfg.num_envs
    buffer_size: int = shared_buffer.batch_size[0]

    if buffer_size % num_envs != 0:
        log_warning(
            f"[Simulation Process] buffer_size ({buffer_size}) is not evenly divisible by "
            f"num_envs ({num_envs}). This may lead to inefficient buffer usage and should ideally be fixed by adjusting "
            "the OnlineDataEngineCfg.",
        )

    num_rollouts_per_fill: int = buffer_size // num_envs
    if buffer_size % num_envs != 0:
        num_rollouts_per_fill += (
            1  # Ensure we fill the entire buffer, even if the last slice is smaller.
        )

    # --- Build the environment and attach the initial tmp_buffer slice ------
    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    log_info("[Simulation Process] Environment created.", color="cyan")

    # --- Main loop: wait for fill signal, then fill the entire buffer -------
    try:
        while True:
            fill_signal.wait()
            fill_signal.clear()

            if close_signal.is_set():
                log_info(
                    "[Simulation Process] Close signal received. Shutting down.",
                    color="cyan",
                )
                break

            log_info(
                "[Simulation Process] Fill signal received. Starting full buffer fill.",
                color="cyan",
            )

            # Reset write cursor to the beginning of the buffer.
            with lock_index.get_lock():
                lock_index[0] = 0
                lock_index[1] = num_envs

            rollout_idx = 0
            while rollout_idx < num_rollouts_per_fill:
                if close_signal.is_set():
                    return

                with lock_index.get_lock():
                    write_start = lock_index[0]
                    write_end = lock_index[1]
                tmp_buffer = shared_buffer[write_start:write_end, :]
                result = None
                for attempt in range(1, cfg.max_generation_attempts + 1):
                    # set_rollout_buffer invalidates the locked rows before
                    # reuse, so stale tail frames can never become sampleable
                    # after a shorter replacement episode.
                    env.get_wrapper_attr("set_rollout_buffer")(tmp_buffer)
                    env.reset(options={"save_data": False})
                    result = execute_demo_episode(
                        env,
                        episode_index=rollout_idx,
                        should_stop=close_signal.is_set,
                        progress=lambda actions, description: tqdm(
                            actions,
                            desc=description,
                            unit="step",
                            leave=False,
                        ),
                    )
                    if result.completed and result.all_success:
                        break
                    if close_signal.is_set() or result.terminal_reason == "interrupted":
                        return
                    log_warning(
                        f"[Simulation Process] Rollout {rollout_idx + 1}/{num_rollouts_per_fill} "
                        f"attempt {attempt}/{cfg.max_generation_attempts} failed: "
                        f"{result.terminal_reason}."
                    )

                if result is None or not (result.completed and result.all_success):
                    raise RuntimeError(
                        f"Failed to generate rollout {rollout_idx + 1} after "
                        f"{cfg.max_generation_attempts} attempts."
                    )

                rollout_idx += 1

                log_info(
                    f"[Simulation Process] Rollout {rollout_idx}/{num_rollouts_per_fill} done. "
                    f"lock_index=[{write_start}, {write_end}], ",
                    color="cyan",
                )

                # Advance lock_index to the next write slice.
                next_start = write_start + num_envs
                next_end = write_end + num_envs
                if next_start >= buffer_size:
                    # Wrap around to the start of the buffer.
                    next_start = 0
                    next_end = num_envs
                elif next_end > buffer_size:
                    next_end = buffer_size
                    next_start = buffer_size - num_envs

                # Samplers hold this same lock until their selected data has
                # been copied. Publishing the next write window therefore
                # cannot race a sample that selected rows under the old mask.
                with lock_index.get_lock():
                    lock_index[0] = next_start
                    lock_index[1] = next_end

            # Unlock every row before publishing readiness to the parent.
            with lock_index.get_lock():
                lock_index[0] = -1
                lock_index[1] = -1

            # Signal that the buffer contains valid data for the first time.
            # is_set() is checked so subsequent refills do not redundantly set it.
            if not init_signal.is_set():
                init_signal.set()
                log_info(
                    "[Simulation Process] Initial buffer fill complete. Engine is ready.",
                    color="cyan",
                )
    finally:
        error = sys.exc_info()[1]
        if error is not None and not error_reported[0]:
            error_reported[0] = _publish_worker_error(
                error_buffer,
                error_length,
                failed_signal,
                state_value,
                error,
            )
        env.close()


def _sim_worker_fn(
    cfg: OnlineDataEngineCfg,
    shared_buffer: TensorDict,
    lock_index: SynchronizedArray,
    fill_signal: MpEvent,
    init_signal: MpEvent,
    close_signal: MpEvent,
    error_buffer: SynchronizedArray,
    error_length: Synchronized,
    failed_signal: MpEvent,
    state_value: Synchronized,
) -> None:
    """Run the simulation worker and publish a stable exception envelope.

    Picklable exceptions are reconstructed with their original type and
    message. Every consumer process reads the same shared snapshot; exceptions
    that cannot be reconstructed are represented by
    :class:`OnlineDataWorkerError`.
    """
    error_reported = [False]
    try:
        _run_sim_worker(
            cfg,
            shared_buffer,
            lock_index,
            fill_signal,
            init_signal,
            close_signal,
            error_buffer,
            error_length,
            failed_signal,
            state_value,
            error_reported,
        )
    except BaseException as error:
        if not error_reported[0]:
            _publish_worker_error(
                error_buffer,
                error_length,
                failed_signal,
                state_value,
                error,
            )
        raise


# ---------------------------------------------------------------------------
# OnlineDataEngine
# ---------------------------------------------------------------------------


class OnlineDataEngine:
    """Engine for managing Online Data Streaming (ODS) and environment rollouts.

    Creates a shared rollout buffer in CPU shared memory, spawns a dedicated
    simulation subprocess that fills the buffer with demonstration trajectories,
    and exposes a :meth:`sample_batch` method for the training process to draw
    batches of trajectory chunks.

    **Subprocess lifecycle**

    The simulation subprocess is started in :meth:`start` and immediately
    receives a fill signal so the buffer is populated before the first call to
    :meth:`sample_batch`.  The subprocess loops indefinitely: it waits for
    *fill_signal*, runs ``buffer_size // num_envs`` rollouts to overwrite every
    buffer slot, then goes back to waiting.

    **Concurrency and lock protection**

    :attr:`_lock_index` ``[write_start, write_end)`` is updated by the
    subprocess after each rollout so that :meth:`sample_batch` can skip the
    slot currently being written to, preventing partial reads.

    **Refill criterion**

    :meth:`sample_batch` accumulates the total number of individual trajectory
    samples drawn into :attr:`_sample_count`.  When this counter exceeds
    :attr:`~OnlineDataEngineCfg.refill_threshold` the fill signal is raised
    and the counter resets to zero.  This amortises the cost of GPU-accelerated
    simulation across many training iterations.

    **Lifecycle state**

    Every instance starts in :attr:`OnlineDataEngineState.CREATED`, passes
    through ``STARTING`` while the first fill is running, and only serves data
    in ``READY``. Worker failures transition to ``FAILED`` and explicit cleanup
    transitions to terminal ``STOPPED``; failed or stopped instances cannot be
    restarted.

    Args:
        cfg: Engine configuration.

    Attributes:
        shared_buffer: Shared-memory TensorDict of shape
            ``[buffer_size, max_episode_steps, ...]``.
        buffer_size: Total number of trajectory slots in the shared buffer.
        device: Device of the shared buffer.
        state: Current :class:`OnlineDataEngineState`.
        is_init: ``True`` only while the engine is ready to sample.
    """

    def __init__(self, cfg: OnlineDataEngineCfg) -> None:
        self._owner_pid = os.getpid()
        self._lifecycle_condition = threading.Condition(threading.RLock())
        self._stop_requested = False
        self._cleanup_complete = False
        self._forced_shutdown_attempted = False
        self.cfg = cfg

        if cfg.max_generation_attempts < 1:
            raise ValueError(
                "max_generation_attempts must be at least 1, "
                f"got {cfg.max_generation_attempts}."
            )
        if (
            not math.isfinite(cfg.initialization_timeout)
            or cfg.initialization_timeout <= 0
        ):
            raise ValueError(
                "initialization_timeout must be finite and greater than zero, "
                f"got {cfg.initialization_timeout}."
            )

        # Allocate the shared buffer (shape: [buffer_size, max_episode_steps, ...]).
        self.shared_buffer: TensorDict = self._create_buffer()
        self.buffer_size: int = self.shared_buffer.batch_size[0]
        self.device = self.shared_buffer.device

        num_envs: int = cfg.gym_config.get("num_envs", 1)

        if num_envs > self.buffer_size:
            log_error(
                f"num_envs ({num_envs}) exceeds buffer_size ({self.buffer_size}). "
                "Increase buffer_size in OnlineDataEngineCfg.",
                error_type=ValueError,
            )

        # -------------------------------------------------------------------
        # Shared interprocess state
        # -------------------------------------------------------------------

        # Use a spawn context to avoid forking unsafe runtime state.
        self._mp_ctx = mp.get_context("forkserver")

        # Current write window: subprocess updates these after each rollout.
        # Shape: [write_start, write_end)  (exclusive upper bound).
        self._lock_index: SynchronizedArray = self._mp_ctx.Array("i", [0, num_envs])

        # Raised by the main process to request a full buffer refill.
        self._fill_signal: MpEvent = self._mp_ctx.Event()

        # Set by the subprocess once the first complete buffer fill finishes.
        # Used by the :attr:`is_init` property to let callers wait for readiness.
        self._init_signal: MpEvent = self._mp_ctx.Event()

        # Set by the main process to request the simulation subprocess to stop.
        self._close_signal: MpEvent = self._mp_ctx.Event()

        # Accumulated sample count used by the refill criterion.
        self._sample_count: Synchronized = self._mp_ctx.Value("i", 0)

        # State and worker failures are shared so every DataLoader consumer
        # observes the same terminal state and exception snapshot.
        self._state_value: Synchronized = self._mp_ctx.Value(
            "i", _STATE_TO_CODE[OnlineDataEngineState.CREATED]
        )
        self._worker_failed_signal: MpEvent = self._mp_ctx.Event()
        self._worker_error_buffer: SynchronizedArray = self._mp_ctx.Array(
            "B", _ERROR_BUFFER_SIZE
        )
        self._worker_error_length: Synchronized = self._mp_ctx.Value("i", 0)

        # Handle to the simulation subprocess, set in start() and used in stop().
        self._sim_process: mp.Process | None = None
        self._monitor_thread: threading.Thread | None = None
        self._channel_error: BaseException | None = None
        self._worker_error: BaseException | None = None

    def start(self) -> None:
        """Start the worker and block until its first fill completes.

        Raises:
            RuntimeError: If the engine was already started or stopped.
            TimeoutError: If the first fill exceeds ``initialization_timeout``.
            BaseException: The original exception raised by the worker.
        """
        self._require_owner_process("start")
        with self._lifecycle_condition:
            self._require_state(OnlineDataEngineState.CREATED, "start")
            self._stop_requested = False
            self._set_state(OnlineDataEngineState.STARTING)

        try:
            with self._lifecycle_condition:
                self._sim_process = self._mp_ctx.Process(
                    target=_sim_worker_fn,
                    args=(
                        self.cfg,
                        self.shared_buffer,
                        self._lock_index,
                        self._fill_signal,
                        self._init_signal,
                        self._close_signal,
                        self._worker_error_buffer,
                        self._worker_error_length,
                        self._worker_failed_signal,
                        self._state_value,
                    ),
                    # Some planners create their own process pool. A daemonic
                    # producer would make those nested workers illegal.
                    daemon=False,
                )
                self._sim_process.start()
                process = self._sim_process
                self._monitor_thread = threading.Thread(
                    target=_monitor_worker_process,
                    args=(
                        process,
                        self._close_signal,
                        self._worker_error_buffer,
                        self._worker_error_length,
                        self._worker_failed_signal,
                        self._state_value,
                    ),
                    name="online-data-worker-monitor",
                    daemon=True,
                )
                self._monitor_thread.start()
            log_info(
                f"[OnlineDataEngine] Simulation subprocess started (PID={self._sim_process.pid}).",
                color="green",
            )

            # Trigger the initial fill so data is ready before the first sample.
            self._fill_signal.set()
            deadline = time.monotonic() + self.cfg.initialization_timeout

            while not self._init_signal.wait(timeout=0.1):
                with self._lifecycle_condition:
                    if self._stop_requested:
                        raise RuntimeError(
                            "OnlineDataEngine.start() was cancelled by stop()."
                        )
                self._ensure_worker_alive()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "OnlineDataEngine initial buffer fill exceeded "
                        f"{self.cfg.initialization_timeout} seconds."
                    )

            self._ensure_worker_alive()
            with self._lifecycle_condition:
                if self._stop_requested:
                    raise RuntimeError(
                        "OnlineDataEngine.start() was cancelled by stop()."
                    )
                self._set_state(OnlineDataEngineState.READY)
                self._lifecycle_condition.notify_all()
        except BaseException as error:
            cleanup_error = None
            forced_shutdown = False
            with self._lifecycle_condition:
                stop_requested = self._stop_requested
                if not self._worker_failed_signal.is_set():
                    self._worker_error = error
                try:
                    forced_shutdown = self._shutdown_worker()
                except BaseException as caught_cleanup_error:
                    cleanup_error = caught_cleanup_error
                    _add_exception_note(
                        error, f"Worker cleanup also failed: {caught_cleanup_error}"
                    )
                else:
                    self._cleanup_complete = True

                forced_shutdown = forced_shutdown or self._forced_shutdown_attempted

                # The worker may only publish an env.close()/recorder failure
                # while the join above is in progress. Keep the start error as
                # primary, but never lose that late durability error.
                channel_error = self._receive_worker_error()
                if channel_error is not None and channel_error is not error:
                    _add_exception_note(
                        error,
                        "Worker also failed during cleanup: "
                        f"{type(channel_error).__name__}: {channel_error}",
                    )

                if forced_shutdown:
                    durability_error = _forced_shutdown_error()
                    if channel_error is None:
                        self._record_worker_error(durability_error)
                        channel_error = durability_error
                    _add_exception_note(error, str(durability_error))

                if (
                    stop_requested
                    and channel_error is None
                    and cleanup_error is None
                    and not forced_shutdown
                ):
                    self._set_state(OnlineDataEngineState.STOPPED)
                else:
                    self._set_state(OnlineDataEngineState.FAILED)
                self._lifecycle_condition.notify_all()
            raise

    def _ensure_worker_alive(self) -> None:
        """Fail the engine immediately when its worker reports or exits."""
        worker_error = self._receive_worker_error()
        if worker_error is not None:
            self._set_state(OnlineDataEngineState.FAILED)
            raise worker_error

        if not self._is_owner_process():
            if self._close_signal.is_set():
                raise RuntimeError(
                    "OnlineDataEngine owner has stopped the simulation worker."
                )
            return

        if self._sim_process is None:
            error = RuntimeError("OnlineDataEngine simulation worker was not created.")
            self._record_worker_error(error)
            raise error
        if self._sim_process.is_alive():
            return

        self._sim_process.join(timeout=0)
        worker_error = self._receive_worker_error()
        if worker_error is None:
            worker_error = RuntimeError(
                "OnlineDataEngine simulation worker exited unexpectedly "
                f"(exit code {self._sim_process.exitcode})."
            )
            self._record_worker_error(worker_error)
        self._set_state(OnlineDataEngineState.FAILED)
        raise worker_error

    def _receive_worker_error(self) -> BaseException | None:
        """Return the broadcast worker exception when one has been published."""
        if not self._worker_failed_signal.is_set():
            return None
        if self._channel_error is not None:
            return self._channel_error

        try:
            with self._worker_error_buffer.get_lock():
                payload_length = self._worker_error_length.value
                payload = bytes(self._worker_error_buffer[:payload_length])
            envelope = pickle.loads(payload)
            if not isinstance(envelope, _WorkerErrorEnvelope):
                raise TypeError(f"invalid envelope type {type(envelope).__name__}")
            error = _error_from_envelope(envelope)
        except BaseException as decode_error:
            error = OnlineDataWorkerError(
                "OnlineDataEngine could not decode the worker error channel: "
                f"{type(decode_error).__name__}: {decode_error}"
            )
        self._channel_error = error
        if self._worker_error is None:
            self._worker_error = error
        self._set_state(OnlineDataEngineState.FAILED)
        return error

    def _record_worker_error(self, error: BaseException) -> None:
        """Publish an owner-detected worker failure to every consumer."""
        if self._worker_failed_signal.is_set():
            self._receive_worker_error()
            return

        self._channel_error = error
        if self._worker_error is None:
            self._worker_error = error
        published = _publish_worker_error(
            self._worker_error_buffer,
            self._worker_error_length,
            self._worker_failed_signal,
            self._state_value,
            error,
        )
        if not published:
            fallback = OnlineDataWorkerError(
                "OnlineDataEngine worker error serialization failed for "
                f"{type(error).__module__}.{type(error).__qualname__}."
            )
            _publish_worker_error(
                self._worker_error_buffer,
                self._worker_error_length,
                self._worker_failed_signal,
                self._state_value,
                fallback,
            )
        self._set_state(OnlineDataEngineState.FAILED)

    def _require_state(self, expected: OnlineDataEngineState, operation: str) -> None:
        """Require an exact lifecycle state for a public operation."""
        current_state = self.state
        if current_state is expected:
            return
        raise RuntimeError(
            f"OnlineDataEngine.{operation}() requires state {expected.value}; "
            f"current state is {current_state.value}."
        ) from self._worker_error

    def _is_owner_process(self) -> bool:
        """Whether the current process owns the producer lifecycle."""
        return os.getpid() == self._owner_pid

    def _require_owner_process(self, operation: str) -> None:
        """Reject lifecycle operations from forked or spawned consumers."""
        if self._is_owner_process():
            return
        raise RuntimeError(
            f"OnlineDataEngine.{operation}() may only be called by owner process "
            f"{self._owner_pid}; current process is {os.getpid()}."
        )

    def _set_state(self, state: OnlineDataEngineState) -> None:
        """Publish a lifecycle transition to every sharing process."""
        with self._state_value.get_lock():
            self._state_value.value = _STATE_TO_CODE[state]

    # -----------------------------------------------------------------------
    # Buffer initialisation
    # -----------------------------------------------------------------------

    def _create_buffer(self) -> TensorDict:
        """Allocate the shared rollout buffer.

        The buffer has shape ``[buffer_size, max_episode_steps, ...]`` and is
        placed in CPU shared memory so it can be safely accessed from both the
        main process and the simulation subprocess.

        Returns:
            TensorDict in shared memory.
        """
        from embodichain.lab.gym.utils.gym_utils import init_rollout_buffer_from_config

        gym_config: dict = self.cfg.gym_config
        max_episode_steps: int = gym_config.get(
            "max_episode_steps", self.cfg.max_episode_steps
        )

        shared_td = init_rollout_buffer_from_config(
            gym_config,
            device=self.cfg.buffer_device,
            batch_size=self.cfg.buffer_size,
            max_episode_steps=max_episode_steps,
            state_dim=self.cfg.state_dim,
        )

        if shared_td.device.type == "cpu":
            shared_td.share_memory_()

        return shared_td

    # -----------------------------------------------------------------------
    # Status
    # -----------------------------------------------------------------------

    @property
    def state(self) -> OnlineDataEngineState:
        """Return the engine's current lifecycle state."""
        with self._state_value.get_lock():
            state_code = self._state_value.value
        return _CODE_TO_STATE[state_code]

    @property
    def is_init(self) -> bool:
        """Whether the engine is ready to serve initialized data."""
        return self.state is OnlineDataEngineState.READY

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------

    def sample_batch(
        self,
        batch_size: int,
        chunk_size: int,
        sampling_mode: Literal["episode", "segment", "boundary"] = "episode",
    ) -> TensorDict:
        """Sample a batch of trajectory chunks from the shared rollout buffer.

        Only fully valid windows are candidates, so padding or stale tail
        frames are never returned. ``episode`` mode allows a window to cross
        segment boundaries, ``segment`` keeps every window inside one segment,
        and ``boundary`` deliberately samples windows crossing an internal
        segment boundary.

        After sampling the internal :attr:`_sample_count` is incremented by
        *batch_size*; if the count exceeds
        :attr:`~OnlineDataEngineCfg.refill_threshold` a buffer refill is
        triggered automatically.

        Args:
            batch_size: Number of trajectory chunks to include in the batch.
            chunk_size: Number of consecutive timesteps in each chunk.
            sampling_mode: Segment-boundary policy for candidate windows.

        Returns:
            TensorDict with batch size ``[batch_size, chunk_size]``.

        Raises:
            ValueError: If an argument is invalid.
            RuntimeError: If no unlocked valid window satisfies the policy.
        """
        with self._lifecycle_condition:
            worker_error = self._receive_worker_error()
            if worker_error is not None:
                raise worker_error
            self._require_state(OnlineDataEngineState.READY, "sample_batch")
            self._ensure_worker_alive()

        max_steps: int = self.shared_buffer.batch_size[1]
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
        if chunk_size > max_steps:
            log_error(
                f"chunk_size ({chunk_size}) exceeds max_episode_steps ({max_steps}).",
                error_type=ValueError,
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {chunk_size}.")
        if sampling_mode not in {"episode", "segment", "boundary"}:
            raise ValueError(
                "sampling_mode must be 'episode', 'segment', or 'boundary', "
                f"got {sampling_mode!r}."
            )
        if sampling_mode == "boundary" and chunk_size < 2:
            raise ValueError("boundary sampling requires chunk_size >= 2.")

        # Hold the producer's window lock through the final clone. The worker
        # may continue writing its already-advertised window, which is excluded
        # below, but cannot switch to one of our selected rows until the copy is
        # complete.
        with self._lock_index.get_lock():
            lock_start: int = self._lock_index[0]
            lock_end: int = self._lock_index[1]

            if "valid" in self.shared_buffer.keys():
                valid = self.shared_buffer["valid"].bool()
            else:
                # Schema-v1 buffers are one fully valid segment per row.
                valid = torch.ones(
                    self.buffer_size,
                    max_steps,
                    dtype=torch.bool,
                    device=self.shared_buffer.device,
                )

            all_rows = torch.arange(self.buffer_size, device=valid.device)
            is_locked = (all_rows >= lock_start) & (all_rows < lock_end)
            valid_windows = valid.unfold(1, chunk_size, 1).all(dim=-1)
            valid_windows[is_locked] = False

            segment_ids = self.shared_buffer.get("segment_id", None)
            if segment_ids is None:
                segment_ids = torch.zeros_like(valid, dtype=torch.int64)

            if sampling_mode == "segment":
                segment_windows = segment_ids.unfold(1, chunk_size, 1)
                same_segment = (segment_windows == segment_windows[..., :1]).all(
                    dim=-1
                ) & (segment_windows[..., 0] >= 0)
                valid_windows &= same_segment
            elif sampling_mode == "boundary":
                segment_windows = segment_ids.unfold(1, chunk_size, 1)
                crosses_boundary = (
                    segment_windows[..., 1:] != segment_windows[..., :-1]
                ).any(dim=-1)
                valid_windows &= crosses_boundary

            candidate_rows = (
                valid_windows.any(dim=1).nonzero(as_tuple=False).squeeze(-1)
            )
            if candidate_rows.numel() == 0:
                raise RuntimeError(
                    "[OnlineDataEngine] No unlocked valid chunk satisfies "
                    f"sampling_mode={sampling_mode!r} and chunk_size={chunk_size}."
                )

            # Preserve the historical episode-uniform sampling distribution:
            # choose an eligible row uniformly, then a valid offset within it.
            sampled_row_ids = torch.randint(
                0,
                candidate_rows.shape[0],
                (batch_size,),
                device=candidate_rows.device,
            )
            row_indices = candidate_rows[sampled_row_ids]
            start_indices = torch.multinomial(
                valid_windows[row_indices].to(dtype=torch.float32),
                num_samples=1,
                replacement=True,
            ).squeeze(-1)

            time_offsets = torch.arange(chunk_size, device=start_indices.device)
            time_indices = start_indices[:, None] + time_offsets[None, :]
            result = self.shared_buffer[row_indices[:, None], time_indices].clone()

        # Update sample count and conditionally trigger a refill.
        self._trigger_refill_if_needed(batch_size)

        return result

    # -----------------------------------------------------------------------
    # Refill criterion
    # -----------------------------------------------------------------------

    def _trigger_refill_if_needed(self, count: int = 1) -> None:
        """Accumulate sample count and trigger a buffer refill when the threshold is reached.

        This method is called by :meth:`sample_batch` after every batch.  The
        refill is only requested when the fill signal is not already pending
        (i.e. the subprocess has finished the previous refill).

        Args:
            count: Number of individual trajectory samples drawn in the latest
                call to :meth:`sample_batch` (typically equal to *batch_size*).
        """
        with self._sample_count.get_lock():
            self._sample_count.value += count
            should_refill = (
                self._sample_count.value >= self.cfg.refill_threshold * self.buffer_size
                and not self._fill_signal.is_set()
            )
            if should_refill:
                self._sample_count.value = 0

        if should_refill:
            self._fill_signal.set()
            log_info(
                f"[OnlineDataEngine] Sample count reached refill threshold (refill_threshold * buffer_size) "
                f"({self.cfg.refill_threshold * self.buffer_size}). Signalling subprocess to refill the buffer.",
                color="cyan",
            )

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def _detect_preexisting_worker_exit(self) -> BaseException | None:
        """Return and broadcast a worker exit observed before shutdown starts."""
        process = self._sim_process
        if process is None:
            error = RuntimeError("OnlineDataEngine simulation worker was not created.")
            self._record_worker_error(error)
            return error

        try:
            if process.is_alive():
                return None
            process.join(timeout=0)
            exit_code = process.exitcode
        except BaseException as inspection_error:
            error = RuntimeError(
                "OnlineDataEngine could not inspect its simulation worker before "
                f"shutdown: {type(inspection_error).__name__}: {inspection_error}"
            )
            self._record_worker_error(error)
            return error

        worker_error = self._receive_worker_error()
        if worker_error is not None:
            return worker_error

        error = RuntimeError(
            "OnlineDataEngine simulation worker exited unexpectedly "
            f"before stop() (exit code {exit_code})."
        )
        self._record_worker_error(error)
        return error

    def _shutdown_worker(self) -> bool:
        """Stop and reap the worker, returning whether force was required."""
        self._require_owner_process("stop")
        self._close_signal.set()
        self._fill_signal.set()

        process = self._sim_process
        if process is None:
            return False

        forced_shutdown = False

        try:
            is_alive = process.is_alive()
        except ValueError:
            is_alive = False

        try:
            process.join(timeout=5.0 if is_alive else 0)
        except (AssertionError, ValueError):
            pass

        try:
            is_alive = process.is_alive()
        except ValueError:
            is_alive = False
        if is_alive:
            forced_shutdown = True
            self._forced_shutdown_attempted = True
            process.terminate()
            process.join(timeout=3.0)

        try:
            is_alive = process.is_alive()
        except ValueError:
            is_alive = False
        if is_alive and hasattr(process, "kill"):
            forced_shutdown = True
            self._forced_shutdown_attempted = True
            process.kill()
            process.join(timeout=1.0)

        try:
            is_alive = process.is_alive()
        except ValueError:
            is_alive = False
        if is_alive:
            raise RuntimeError(
                "OnlineDataEngine simulation worker remained alive after "
                "graceful shutdown, terminate(), and kill()."
            )

        monitor_thread = self._monitor_thread
        if (
            monitor_thread is not None
            and monitor_thread is not threading.current_thread()
        ):
            monitor_thread.join(timeout=1.0)
            if monitor_thread.is_alive():
                raise RuntimeError(
                    "OnlineDataEngine worker monitor did not stop after the worker exited."
                )

        if hasattr(process, "close"):
            process.close()
        self._sim_process = None
        self._monitor_thread = None
        return forced_shutdown

    def stop(self) -> None:
        """Terminate the simulation subprocess and release resources.

        Sets the close signal and waits briefly for the subprocess to exit
        gracefully (it checks the signal between rollout steps).  If the
        subprocess is still alive after the grace period it is force-terminated.

        Safe to call multiple times — subsequent calls are no-ops if the
        subprocess has already been terminated.
        """
        self._require_owner_process("stop")
        with self._lifecycle_condition:
            if self.state is OnlineDataEngineState.STOPPED:
                return
            if self.state is OnlineDataEngineState.FAILED and self._cleanup_complete:
                # A successful join makes future publications impossible, but
                # still decode anything already visible before honoring the
                # idempotent stop contract.
                self._receive_worker_error()
                return

            if self.state is OnlineDataEngineState.STARTING:
                self._stop_requested = True
                self._close_signal.set()
                self._fill_signal.set()
                while self.state is OnlineDataEngineState.STARTING:
                    self._lifecycle_condition.wait(timeout=0.1)
                if self.state is OnlineDataEngineState.STOPPED:
                    return

            state_before_shutdown = self.state
            worker_error = self._receive_worker_error()
            if (
                worker_error is None
                and state_before_shutdown is OnlineDataEngineState.READY
            ):
                worker_error = self._detect_preexisting_worker_exit()

            forced_shutdown = False
            try:
                forced_shutdown = self._shutdown_worker()
            except BaseException as cleanup_error:
                # The worker can fail while handling the close signal (for
                # example, while flushing a recorder in ``env.close()``).
                # Re-read the shared channel after waiting for it so that a
                # durability failure is not hidden behind the cleanup path.
                worker_error = worker_error or self._receive_worker_error()
                if self._forced_shutdown_attempted:
                    durability_error = _forced_shutdown_error()
                    if worker_error is None:
                        self._record_worker_error(durability_error)
                        worker_error = durability_error
                    else:
                        _add_exception_note(worker_error, str(durability_error))
                self._set_state(OnlineDataEngineState.FAILED)
                self._lifecycle_condition.notify_all()
                if worker_error is not None:
                    _add_exception_note(
                        worker_error, f"Worker cleanup also failed: {cleanup_error}"
                    )
                    raise worker_error
                self._worker_error = cleanup_error
                raise

            # ``_shutdown_worker`` joins the producer, so any exception raised
            # during its final ``env.close()`` has now been published.
            worker_error = worker_error or self._receive_worker_error()
            forced_shutdown = forced_shutdown or self._forced_shutdown_attempted
            if forced_shutdown:
                durability_error = _forced_shutdown_error()
                if worker_error is None:
                    self._record_worker_error(durability_error)
                    worker_error = durability_error
                else:
                    _add_exception_note(worker_error, str(durability_error))

            self._cleanup_complete = True
            if worker_error is not None:
                self._set_state(OnlineDataEngineState.FAILED)
                self._lifecycle_condition.notify_all()
                raise worker_error

            self._set_state(OnlineDataEngineState.STOPPED)
            self._lifecycle_condition.notify_all()
        log_info("[OnlineDataEngine] Engine stopped.", color="green")

    def __enter__(self) -> "OnlineDataEngine":
        """Start the engine and return it for a managed lifecycle block."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """Stop the engine while preserving any exception from the block."""
        try:
            self.stop()
        except BaseException as cleanup_error:
            if exc_value is None:
                raise
            _add_exception_note(
                exc_value,
                "OnlineDataEngine cleanup also failed: "
                f"{type(cleanup_error).__name__}: {cleanup_error}",
            )
        return None

    def __getstate__(self) -> dict:
        """Serialize only consumer-safe fields for spawned DataLoader workers."""
        state = self.__dict__.copy()
        state["_sim_process"] = None
        state["_monitor_thread"] = None
        state["_lifecycle_condition"] = None
        state["_channel_error"] = None
        state["_worker_error"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore process-local synchronization after consumer deserialization."""
        self.__dict__.update(state)
        self._lifecycle_condition = threading.Condition(threading.RLock())

    def __del__(self) -> None:
        try:
            if getattr(self, "_owner_pid", None) != os.getpid():
                return
            if self.state is not OnlineDataEngineState.STOPPED:
                self.stop()
        except BaseException:
            # Destructors run during partially initialized objects and interpreter
            # shutdown, where cleanup must never mask the original exception.
            pass
