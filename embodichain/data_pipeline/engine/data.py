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
import uuid

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
from .language import SharedLanguageRegistry

__all__ = [
    "OnlineDataEngine",
    "OnlineDataEngineCfg",
    "OnlineDataEngineState",
    "OnlineDataWorkerError",
]

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.lab.gym.envs.demo import DemoEpisodeResult

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

    language_buffer_bytes: int = 1024 * 1024
    """Append-only task/subtask dictionary capacity in UTF-8 bytes. Descriptions
    remain available for copied samples after trajectory slots are reused.
    Exhaustion fails the worker explicitly; increase this for many unique tasks."""

    action_dim: int | None = None
    """Stored action width; None preserves the historical state_dim width."""

    refill_policy: Literal["sample_threshold", "manual"] = "sample_threshold"
    """Automatic sample-count refill or owner-requested refill."""

    include_provenance: bool = False
    """Include numeric run/episode/generation IDs, episode frame indices, and
    successful-publication Unix and host-monotonic timestamps in nanoseconds. Invalid frames use
    ``-1`` for every provenance field."""

    lifecycle_event_capacity: int = 0
    """Optional bounded publication/eviction log; zero disables logging.

    When enabled, capacity must fit twice ``buffer_size`` and all initial-fill
    records (including overlapping final-batch evictions). The owner
    must drain events while refilling; a full log backpressures the producer
    outside the trajectory-copy lock. Stop interrupts this wait.
    """

    action_semantics: Literal["expert_controller_command", "policy_request"] = (
        "expert_controller_command"
    )
    """Stored action meaning; the default preserves expert-controller commands."""

    action_contracts: tuple[str, ...] | None = None
    """Expected ActionManager contract IDs in flat policy order.

    Required with an explicit ``action_dim`` for ``policy_request`` recording.
    The worker rejects a different live contract sequence, width, or joint order.
    """

    shutdown_timeout: float = 5.0
    """Maximum seconds to wait for graceful worker close before forcing shutdown.

    Must be finite and positive. Forced termination still fails the engine's
    recorder-durability contract, regardless of this timeout.
    """


def _validate_recording_config(cfg: OnlineDataEngineCfg) -> None:
    """Require an explicit ordered contract when recording policy requests."""
    if cfg.action_semantics not in {"expert_controller_command", "policy_request"}:
        raise ValueError("Unknown ODS action_semantics.")
    if cfg.action_semantics == "policy_request":
        if cfg.action_dim is None or not cfg.action_contracts:
            raise ValueError(
                "policy_request ODS requires action_dim and ordered action_contracts."
            )
        if not isinstance(cfg.action_contracts, tuple) or any(
            not isinstance(contract, str) or not contract.strip()
            for contract in cfg.action_contracts
        ):
            raise ValueError(
                "action_contracts must be a tuple of nonempty contract IDs."
            )
    elif cfg.action_contracts is not None:
        raise ValueError(
            "action_contracts are only valid for policy_request recording."
        )


class _ProducerProgress:
    """Small shared progress record, independent of the trajectory write lock."""

    def __init__(self, context: object) -> None:
        # requested, completed, successes, failures, frames, seconds, samples
        self.values = context.Array("d", [1, 0, 0, 0, 0, 0, 0])

    def add(self, index: int, value: float) -> None:
        with self.values.get_lock():
            self.values[index] += value


class _LifecycleEvents:
    """Single-producer bounded shared log without multiprocessing feeder threads.

    Storage is O(capacity * max_episode_steps), with two byte masks per event.
    Condition ownership protects both the ring cursors and the complete record;
    the consumer never acquires the trajectory-copy lock.
    """

    _FIELDS = (
        "sequence",
        "kind_code",
        "run_id",
        "episode_id",
        "generation",
        "slot",
        "monotonic_ns",
    )

    def __init__(self, context: object, capacity: int, max_steps: int) -> None:
        self.capacity = capacity
        self.max_steps = max_steps
        self.condition = context.Condition()
        self.cursors = context.Array("q", [0, 0], lock=False)
        self.records = context.Array("q", capacity * len(self._FIELDS), lock=False)
        self.masks = context.Array("B", capacity * max_steps * 2, lock=False)

    def wait_for_space(self, count: int, close_signal: MpEvent) -> bool:
        """Wait outside the copy lock, returning False on cooperative shutdown."""
        with self.condition:
            while self.cursors[1] - self.cursors[0] + count > self.capacity:
                if close_signal.is_set():
                    return False
                self.condition.wait(timeout=0.05)
            return not close_signal.is_set()

    def append(self, event: dict[str, object]) -> None:
        """Publish one already-reserved event, failing rather than dropping it."""
        with self.condition:
            head, tail = self.cursors[:]
            if tail - head >= self.capacity:
                raise RuntimeError("ODS lifecycle event capacity exhausted")
            slot = tail % self.capacity
            record = [tail, int(event["kind"] == "published")]
            record.extend(int(event[key]) for key in self._FIELDS[2:])
            offset = slot * len(self._FIELDS)
            self.records[offset : offset + len(record)] = record
            offset = slot * self.max_steps * 2
            masks = bytearray(self.max_steps * 2)
            for index in event["valid_frame_indices"]:
                masks[index] = 1
            for index in event["eligible_anchor_indices"]:
                masks[self.max_steps + index] = 1
            self.masks[offset : offset + len(masks)] = masks
            self.cursors[1] = tail + 1
            self.condition.notify_all()

    def drain(self) -> list[dict[str, object]]:
        """Copy pending records and release their bounded slots atomically."""
        result = []
        with self.condition:
            head, tail = self.cursors[:]
            for sequence in range(head, tail):
                slot = sequence % self.capacity
                offset = slot * len(self._FIELDS)
                event = dict(
                    zip(self._FIELDS, self.records[offset : offset + len(self._FIELDS)])
                )
                event["kind"] = "published" if event.pop("kind_code") else "evicted"
                offset = slot * self.max_steps * 2
                masks = self.masks[offset : offset + self.max_steps * 2]
                event["valid_frame_indices"] = [
                    i for i, value in enumerate(masks[: self.max_steps]) if value
                ]
                event["eligible_anchor_indices"] = [
                    i for i, value in enumerate(masks[self.max_steps :]) if value
                ]
                event["valid_frames"] = len(event["valid_frame_indices"])
                event["eligible_anchors"] = len(event["eligible_anchor_indices"])
                result.append(event)
            self.cursors[0] = tail
            self.condition.notify_all()
        return result


def _publication_event(
    row: TensorDict,
    *,
    run_id: int,
    episode_id: int,
    generation: int,
    slot: int,
    monotonic_ns: int,
) -> dict[str, object]:
    """Describe every valid frame and every chunk-context eligible anchor."""
    valid = row["valid"].bool()
    continuity = row.get("continuity_id", torch.zeros_like(valid, dtype=torch.int64))
    return {
        "kind": "published",
        "run_id": run_id,
        "episode_id": episode_id,
        "generation": generation,
        "slot": slot,
        "monotonic_ns": monotonic_ns,
        "valid_frame_indices": valid.nonzero().flatten().tolist(),
        "eligible_anchor_indices": (valid & (continuity >= 0))
        .nonzero()
        .flatten()
        .tolist(),
    }


def _validate_worker_dimensions(env: object, cfg: OnlineDataEngineCfg) -> None:
    """Check the live recorded-action schema before generating any transition."""
    _validate_recording_config(cfg)
    if cfg.action_dim is None:
        return
    semantics = env.get_wrapper_attr("rollout_action_semantics")
    if semantics != cfg.action_semantics:
        raise ValueError(
            f"ODS action semantics {cfg.action_semantics!r} disagree with "
            f"the live rollout buffer {semantics!r}."
        )
    space = env.get_wrapper_attr("single_observation_space")
    if space["robot"]["qpos"].shape[-1] != cfg.state_dim:
        raise ValueError("ODS state/action widths disagree with the live environment")
    if semantics == "expert_controller_command":
        width = env.get_wrapper_attr("expert_action_spec").width
    else:
        manager = env.get_wrapper_attr("action_manager")
        if manager is None:
            raise ValueError("policy_request ODS requires a live ActionManager.")
        descriptors = manager.descriptors
        contracts = tuple(descriptor.term.contract for descriptor in descriptors)
        if contracts != cfg.action_contracts:
            raise ValueError(
                "ODS ordered action contracts disagree with the live ActionManager."
            )
        width = 0
        robot = env.get_wrapper_attr("robot")
        for descriptor in descriptors:
            if (
                descriptor.start != width
                or descriptor.stop - descriptor.start != descriptor.term.action_dim
            ):
                raise ValueError(
                    "ODS live action slices must be contiguous and ordered."
                )
            width = descriptor.stop
            term = manager.get_term(descriptor.name)
            if term.command_type in {"qpos", "qvel", "qf"}:
                joint_names = tuple(
                    robot.joint_names[index] for index in term.controlled_joint_ids
                )
                if tuple(descriptor.term.joint_names) != joint_names:
                    raise ValueError(
                        "ODS live action descriptor joint order disagrees."
                    )
        if width != manager.total_action_dim:
            raise ValueError("ODS descriptor and ActionManager widths disagree.")
    if width != cfg.action_dim:
        raise ValueError("ODS state/action widths disagree with the live environment")


def _populate_language_indices(
    buffer: TensorDict,
    result: "DemoEpisodeResult",
    registry: SharedLanguageRegistry,
) -> None:
    """Assign immutable text indices before publishing a successful write window."""
    rows, capacity = buffer.batch_size
    lengths = getattr(result, "lengths", ()) or (result.length,) * rows
    if len(lengths) != rows or any(
        type(length) is not int or not 0 <= length <= capacity for length in lengths
    ):
        raise ValueError("Online language row lengths must match the trajectory buffer")
    task = getattr(result, "instruction", None) or "unknown_task"
    task_id = registry.intern(task, kind="task")
    fallback_id = registry.intern(task, kind="subtask")
    task_indices, subtask_indices = buffer["task_index"], buffer["subtask_index"]
    task_indices.fill_(-1)
    subtask_indices.fill_(-1)
    valid = buffer.get("valid", None)
    occupied = torch.zeros(
        (rows, capacity), dtype=torch.bool, device=task_indices.device
    )
    for row, length in enumerate(lengths):
        if valid is not None and valid[row, length:].any():
            raise ValueError(
                f"Online row {row} has valid frames beyond result length {length}"
            )
        task_indices[row, :length] = task_id
        subtask_indices[row, :length] = fallback_id
    for segment in getattr(result, "segments", ()):
        active = getattr(segment, "active", ())
        starts, ends = getattr(segment, "start_steps", ()), getattr(
            segment, "end_steps", ()
        )
        instruction = getattr(segment, "instruction", None) or task
        subtask_id = None
        for row, length in enumerate(lengths):
            if active and row < len(active) and not active[row]:
                continue
            start = starts[row] if row < len(starts) else segment.start_step
            end = ends[row] if row < len(ends) else segment.end_step
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start <= end <= length
            ):
                raise ValueError(
                    f"Online language segment span [{start}, {end}) is outside row {row} length {length}"
                )
            if occupied[row, start:end].any():
                raise ValueError(f"Online language segment spans overlap in row {row}")
            if start == end:
                continue
            if subtask_id is None:
                subtask_id = registry.intern(instruction, kind="subtask")
            occupied[row, start:end] = True
            subtask_indices[row, start:end] = subtask_id
    if valid is not None:
        task_indices[~valid.bool()] = -1
        subtask_indices[~valid.bool()] = -1


def _populate_provenance(
    buffer: TensorDict,
    *,
    run_id: int,
    generation: int,
    episode_start: int,
    accepted_at_ns: int,
    published_monotonic_ns: int | None = None,
) -> None:
    """Stamp accepted rows while the producer still owns the publication lock."""
    valid = buffer["valid"].bool()
    rows, capacity = buffer.batch_size
    frame_indices = torch.arange(capacity, device=valid.device).expand(rows, -1)
    episode_indices = torch.arange(
        episode_start, episode_start + rows, device=valid.device
    )[:, None].expand(-1, capacity)
    for key, value in (
        ("ods_run_id", run_id),
        ("ods_generation", generation),
        ("ods_episode_id", episode_indices),
        ("ods_frame_index", frame_indices),
        ("ods_accepted_at_ns", accepted_at_ns),
    ):
        buffer[key].fill_(-1)
        buffer[key][valid] = value[valid] if isinstance(value, torch.Tensor) else value

    if "ods_published_at_monotonic_ns" in buffer.keys():
        buffer["ods_published_at_monotonic_ns"].fill_(-1)
        buffer["ods_published_at_monotonic_ns"][valid] = (
            time.monotonic_ns()
            if published_monotonic_ns is None
            else published_monotonic_ns
        )


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
    language_registry: SharedLanguageRegistry | None = None,
    progress: _ProducerProgress | None = None,
    run_id: int = 0,
    lifecycle_events: _LifecycleEvents | None = None,
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
        language_registry: Optional shared immutable text dictionaries. Real
            engines populate frame indices while the write window is excluded
            from sampling; omitted only by legacy lightweight worker callers.
    """
    import gymnasium as gym
    from embodichain.lab.gym.utils.gym_utils import (
        config_to_cfg,
        get_manager_modules,
    )
    from embodichain.lab.gym.utils.registration import discover_task_packages
    from embodichain.lab.gym.envs.demo import execute_demo_episode
    from embodichain.utils.logger import log_info, log_warning

    gym_config: dict = cfg.gym_config
    action_config: dict = cfg.action_config

    # Forkserver workers start with a fresh Gym registry. Import official task
    # packages here so the configured environment is registered before make().
    discover_task_packages()

    # Build env config from the gym configuration dictionary.
    env_cfg = config_to_cfg(gym_config, manager_modules=get_manager_modules())
    env_cfg.filter_dataset_saving = True
    env_cfg.init_rollout_buffer = False
    env_cfg.rollout_action_semantics = cfg.action_semantics
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
    accepted_episodes = 0
    slot_events: dict[int, dict[str, object]] = {}
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

            started_at = time.monotonic()
            generation = 1
            if progress is not None:
                with progress.values.get_lock():
                    generation = int(progress.values[1]) + 1

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
                if lifecycle_events is not None:
                    # Reserve for old-row eviction plus successful publication.
                    # Only this worker produces events, so space cannot shrink.
                    required = (
                        write_end
                        - write_start
                        + sum(
                            slot in slot_events
                            for slot in range(write_start, write_end)
                        )
                    )
                    if not lifecycle_events.wait_for_space(required, close_signal):
                        return
                    for slot in range(write_start, write_end):
                        old_event = slot_events.pop(slot, None)
                        if old_event is not None:
                            lifecycle_events.append(
                                {
                                    **old_event,
                                    "kind": "evicted",
                                    "monotonic_ns": time.monotonic_ns(),
                                }
                            )
                tmp_buffer = shared_buffer[write_start:write_end, :]
                result = None
                for attempt in range(1, cfg.max_generation_attempts + 1):
                    # set_rollout_buffer invalidates the locked rows before
                    # reuse, so stale tail frames can never become sampleable
                    # after a shorter replacement episode.
                    env.get_wrapper_attr("set_rollout_buffer")(tmp_buffer)
                    _validate_worker_dimensions(env, cfg)
                    env.reset(options={"save_data": False})
                    result = execute_demo_episode(
                        env,
                        episode_index=rollout_idx,
                        attempt_id=attempt - 1,
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
                    if progress is not None:
                        progress.add(3, num_envs)
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

                if language_registry is not None:
                    _populate_language_indices(tmp_buffer, result, language_registry)

                episode_start = accepted_episodes
                accepted_episodes += len(tmp_buffer)
                if progress is not None:
                    with progress.values.get_lock():
                        episode_start = int(progress.values[2])
                        progress.values[2] += len(tmp_buffer)
                        progress.values[4] += int(tmp_buffer["valid"].sum().item())

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
                    published_monotonic_ns = time.monotonic_ns()
                    if cfg.include_provenance:
                        _populate_provenance(
                            tmp_buffer,
                            run_id=run_id,
                            generation=generation,
                            episode_start=episode_start,
                            accepted_at_ns=time.time_ns(),
                            published_monotonic_ns=published_monotonic_ns,
                        )
                    if lifecycle_events is not None:
                        for row_index in range(len(tmp_buffer)):
                            event = _publication_event(
                                tmp_buffer[row_index],
                                run_id=run_id,
                                episode_id=episode_start + row_index,
                                generation=generation,
                                slot=write_start + row_index,
                                monotonic_ns=published_monotonic_ns,
                            )
                            lifecycle_events.append(event)
                            slot_events[write_start + row_index] = event
                    lock_index[0] = next_start
                    lock_index[1] = next_end

            # Unlock every row before publishing readiness to the parent.
            with lock_index.get_lock():
                lock_index[0] = -1
                lock_index[1] = -1

            if progress is not None:
                with progress.values.get_lock():
                    progress.values[1] = generation
                    progress.values[5] += time.monotonic() - started_at

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
        if os.environ.get("EMBODICHAIN_SIM_EXIT_PROCESS") == "0":
            from embodichain.lab.sim import SimulationManager

            SimulationManager.flush_cleanup_queue()


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
    language_registry: SharedLanguageRegistry | None = None,
    progress: _ProducerProgress | None = None,
    run_id: int = 0,
    lifecycle_events: _LifecycleEvents | None = None,
) -> None:
    """Run the simulation worker and publish a stable exception envelope.

    Picklable exceptions are reconstructed with their original type and
    message. Every consumer process reads the same shared snapshot; exceptions
    that cannot be reconstructed are represented by
    :class:`OnlineDataWorkerError`.
    """
    error_reported = [False]
    try:
        worker_args = (
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
        if lifecycle_events is not None:
            _run_sim_worker(
                *worker_args, language_registry, progress, run_id, lifecycle_events
            )
        elif progress is not None:
            _run_sim_worker(*worker_args, language_registry, progress, run_id)
        elif language_registry is None:
            _run_sim_worker(*worker_args)
        else:
            _run_sim_worker(*worker_args, language_registry)
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
        _validate_recording_config(cfg)
        if cfg.refill_policy not in {"sample_threshold", "manual"}:
            raise ValueError("refill_policy must be 'sample_threshold' or 'manual'")
        if cfg.action_dim is not None and (
            type(cfg.action_dim) is not int or cfg.action_dim < 1
        ):
            raise ValueError("action_dim must be a positive integer or None")
        if type(cfg.lifecycle_event_capacity) is not int or (
            cfg.lifecycle_event_capacity != 0
            and cfg.lifecycle_event_capacity < 2 * cfg.buffer_size
        ):
            raise ValueError(
                "lifecycle_event_capacity must be zero or at least 2 * buffer_size"
            )
        if cfg.lifecycle_event_capacity:
            parallel = cfg.gym_config.get("num_envs", 1)
            # A non-divisible final write window overlaps earlier rows. Its
            # otherwise invisible initial evictions must fit before start returns.
            initial_records = (
                2 * math.ceil(cfg.buffer_size / parallel) * parallel - cfg.buffer_size
            )
            if cfg.lifecycle_event_capacity < initial_records:
                raise ValueError(
                    "lifecycle_event_capacity cannot hold initial-fill events"
                )
        self.run_id = uuid.uuid4().int & ((1 << 63) - 1)

        if cfg.max_generation_attempts < 1:
            raise ValueError(
                "max_generation_attempts must be at least 1, "
                f"got {cfg.max_generation_attempts}."
            )
        if type(cfg.language_buffer_bytes) is not int or cfg.language_buffer_bytes < 1:
            raise ValueError("language_buffer_bytes must be a positive integer")
        if (
            not math.isfinite(cfg.initialization_timeout)
            or cfg.initialization_timeout <= 0
        ):
            raise ValueError(
                "initialization_timeout must be finite and greater than zero, "
                f"got {cfg.initialization_timeout}."
            )

        if not math.isfinite(cfg.shutdown_timeout) or cfg.shutdown_timeout <= 0:
            raise ValueError(
                "shutdown_timeout must be finite and greater than zero, "
                f"got {cfg.shutdown_timeout}."
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
        self._progress = _ProducerProgress(self._mp_ctx)
        self._lifecycle_events = (
            _LifecycleEvents(
                self._mp_ctx,
                cfg.lifecycle_event_capacity,
                self.shared_buffer.batch_size[1],
            )
            if cfg.lifecycle_event_capacity
            else None
        )
        self._language_registry = SharedLanguageRegistry(
            cfg.language_buffer_bytes, context=self._mp_ctx
        )

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
                        getattr(self, "_language_registry", None),
                        getattr(self, "_progress", None),
                        getattr(self, "run_id", 0),
                        getattr(self, "_lifecycle_events", None),
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
            **(
                {"action_dim": self.cfg.action_dim}
                if self.cfg.action_dim is not None
                else {}
            ),
        )

        for key in ("task_index", "subtask_index"):
            shared_td[key] = torch.full(
                shared_td.batch_size, -1, dtype=torch.int64, device=shared_td.device
            )

        if self.cfg.include_provenance:
            for key in (
                "ods_run_id",
                "ods_episode_id",
                "ods_generation",
                "ods_frame_index",
                "ods_accepted_at_ns",
                "ods_published_at_monotonic_ns",
            ):
                shared_td[key] = torch.full(
                    shared_td.batch_size, -1, dtype=torch.int64, device=shared_td.device
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
        segment boundaries within one causal-continuity region, ``segment``
        keeps every window inside one accepted segment, and ``boundary``
        deliberately samples windows crossing a boundary between accepted
        segments. No mode crosses a discontinuous state-restore boundary.

        After sampling the internal :attr:`_sample_count` is incremented by
        *batch_size*; if the count exceeds
        :attr:`~OnlineDataEngineCfg.refill_threshold` a buffer refill is
        triggered automatically.

        Args:
            batch_size: Number of trajectory chunks to include in the batch.
            chunk_size: Number of consecutive timesteps in each chunk.
            sampling_mode: Segment-boundary policy for candidate windows.

        Returns:
            TensorDict with batch size ``[batch_size, chunk_size]``. Integer
            ``task_index`` and ``subtask_index`` identify immutable language
            snapshots; :meth:`resolve_language` decodes their descriptions.

        Raises:
            ValueError: If an argument is invalid.
            RuntimeError: If no unlocked valid window satisfies the policy.
        """
        return self._sample_batch(batch_size, chunk_size, sampling_mode)

    def sample_batch_with_history(
        self,
        batch_size: int,
        chunk_size: int,
        *,
        start_probability: float = 0.0,
    ) -> TensorDict:
        """Sample supervised windows together with their complete causal prefix.

        Prefixes and windows are copied under the same producer lock. Each row
        starts at its uninterrupted validity/continuity region and is padded on
        the right. ``valid`` excludes padding and ``loss_mask`` selects exactly
        ``chunk_size`` supervised frames. Episode starts have an empty prefix,
        so their first actions can be supervised with a reset recurrent state.

        Args:
            batch_size: Number of sampled windows, with replacement.
            chunk_size: Number of supervised steps per window.
            start_probability: Probability of choosing the window at episode
                step zero when that window is eligible; otherwise use the
                usual episode-uniform, then offset-uniform sampling.

        Returns:
            An independent TensorDict with batch size ``[batch_size, time]``.
            ``time`` is the longest sampled prefix plus supervised window.
            Copying full prefixes also copies configured sensor payloads.

        Raises:
            ValueError: If counts or start_probability are invalid.
            RuntimeError: If the engine fails or no eligible window exists.
        """
        if not 0.0 <= start_probability <= 1.0:
            raise ValueError("start_probability must be between zero and one")
        return self._sample_batch(
            batch_size,
            chunk_size,
            "episode",
            with_history=True,
            start_probability=start_probability,
        )

    def sample_batch_with_context(
        self,
        batch_size: int,
        history_length: int,
        prediction_horizon: int,
    ) -> TensorDict:
        """Sample causal observations and future supervision around valid anchors.

        The current frame is at ``history_length - 1``. Every valid frame,
        including episode starts and ends, can be an anchor. Context never
        crosses invalid frames or continuity boundaries. Out-of-range numeric
        payloads are zero; language/provenance IDs are -1. ``valid`` marks real
        frames and ``loss_mask`` selects only valid current/future targets.
        Selection, provenance and payload copies share the producer lock.

        Args:
            batch_size: Number of contexts, with replacement.
            history_length: Number of causal observation frames including now.
            prediction_horizon: Number of target frames including now.

        Returns:
            TensorDict of shape ``[batch_size, history_length +
            prediction_horizon - 1]``. Consumers must mask padded supervision
            and explicitly handle missing initial observation history.

        Raises:
            ValueError: If context lengths are not positive integers.
            RuntimeError: If the worker fails or no unlocked valid anchor exists.
        """
        if any(
            type(n) is not int or n < 1 for n in (history_length, prediction_horizon)
        ):
            raise ValueError("Context lengths must be positive integers")
        return self._sample_batch(
            batch_size, 1, "episode", context=(history_length, prediction_horizon)
        )

    def _sample_batch(
        self,
        batch_size: int,
        chunk_size: int,
        sampling_mode: Literal["episode", "segment", "boundary"],
        *,
        with_history: bool = False,
        start_probability: float = 0.0,
        context: tuple[int, int] | None = None,
    ) -> TensorDict:
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

            continuity_ids = self.shared_buffer.get("continuity_id", None)
            if continuity_ids is None:
                # Schema-v2 and earlier buffers contain no out-of-band state
                # restore, so the complete row belongs to continuity region 0.
                continuity_ids = torch.zeros_like(valid, dtype=torch.int64)
            continuity_windows = continuity_ids.unfold(1, chunk_size, 1)
            same_continuity = (continuity_windows == continuity_windows[..., :1]).all(
                dim=-1
            ) & (continuity_windows[..., 0] >= 0)
            valid_windows &= same_continuity

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

            if sampling_mode in {"segment", "boundary"}:
                segment_accepted = self.shared_buffer.get("segment_accepted", None)
                if segment_accepted is None:
                    # Older successful-only online buffers predate explicit
                    # segment qualification and remain fully eligible.
                    segment_accepted = torch.ones_like(valid, dtype=torch.bool)
                accepted_windows = segment_accepted.bool().unfold(1, chunk_size, 1)
                valid_windows &= accepted_windows.all(dim=-1)

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
            if context is not None:
                history, future = context
                steps = torch.arange(max_steps, device=start_indices.device)
                compatible = valid[row_indices] & (
                    continuity_ids[row_indices]
                    == continuity_ids[row_indices, start_indices, None]
                )
                lower = torch.where(
                    (~compatible) & (steps[None] < start_indices[:, None]),
                    steps[None] + 1,
                    0,
                ).amax(-1)
                upper = torch.where(
                    (~compatible) & (steps[None] > start_indices[:, None]),
                    steps[None] - 1,
                    max_steps - 1,
                ).amin(-1)
                offsets = torch.arange(1 - history, future, device=steps.device)
                indices = start_indices[:, None] + offsets[None]
                context_valid = (indices >= lower[:, None]) & (
                    indices <= upper[:, None]
                )
                result = self.shared_buffer[
                    row_indices[:, None], indices.clamp(0, max_steps - 1)
                ].clone()
                for key, value in result.items(include_nested=True, leaves_only=True):
                    mask = context_valid.reshape(
                        *context_valid.shape, *((1,) * (value.ndim - 2))
                    )
                    invalid_value = (
                        -1
                        if key in {"task_index", "subtask_index"}
                        or (isinstance(key, str) and key.startswith("ods_"))
                        else 0
                    )
                    value.masked_fill_(~mask, invalid_value)
                result["valid"] = context_valid
                result["loss_mask"] = context_valid & (offsets[None] >= 0)
            elif with_history:
                use_start = (
                    torch.rand(batch_size, device=start_indices.device)
                    < start_probability
                ) & valid_windows[row_indices, 0]
                start_indices = torch.where(use_start, 0, start_indices)
                steps = torch.arange(max_steps, device=start_indices.device)
                compatible = valid[row_indices] & (
                    continuity_ids[row_indices]
                    == continuity_ids[row_indices, start_indices, None]
                )
                interrupted_prefix = (~compatible) & (
                    steps[None, :] < start_indices[:, None]
                )
                prefix_starts = torch.where(
                    interrupted_prefix, steps[None, :] + 1, 0
                ).amax(dim=-1)
                prefix_lengths = start_indices - prefix_starts
                lengths = prefix_lengths + chunk_size
                offsets = torch.arange(int(lengths.max()), device=steps.device)
                time_indices = (prefix_starts[:, None] + offsets[None, :]).clamp_max(
                    max_steps - 1
                )
                result = self.shared_buffer[row_indices[:, None], time_indices].clone()
                valid_history = offsets[None, :] < lengths[:, None]
                # Remove every padded payload, including stale tail observations.
                for key, value in result.items(include_nested=True, leaves_only=True):
                    mask = valid_history.reshape(
                        *valid_history.shape, *((1,) * (value.ndim - 2))
                    )
                    invalid_value = (
                        -1
                        if key in {"task_index", "subtask_index"}
                        or (isinstance(key, str) and key.startswith("ods_"))
                        else 0
                    )
                    value.masked_fill_(~mask, invalid_value)
                result["valid"] = valid_history
                result["loss_mask"] = valid_history & (
                    offsets[None, :] >= prefix_lengths[:, None]
                )
            else:
                time_offsets = torch.arange(chunk_size, device=start_indices.device)
                time_indices = start_indices[:, None] + time_offsets[None, :]
                result = self.shared_buffer[row_indices[:, None], time_indices].clone()

        # Update sample count and conditionally trigger a refill.
        self._trigger_refill_if_needed(batch_size)

        return result

    def resolve_language(self, batch: TensorDict) -> dict[str, object]:
        """Decode overall tasks and current subtasks for a copied sample batch.

        The lookup remains valid after a ring-buffer refill or engine shutdown.
        Returned strings are separate from the shared numeric TensorDict; callers
        can tokenize them in a training transform without changing sampling locks
        or collation behavior.

        Args:
            batch: Sampled TensorDict containing ``task_index``/``subtask_index``.

        Returns:
            ``task`` and ``subtask`` strings nested to match the batch dimensions
            (for example ``[batch_size][chunk_size]``).

        Raises:
            ValueError: If language indices are absent, unknown, or padding.
        """
        if "task_index" not in batch.keys() or "subtask_index" not in batch.keys():
            raise ValueError(
                "Online batch has no task_index/subtask_index language fields"
            )
        registry = getattr(self, "_language_registry", None)
        if registry is None:
            raise ValueError("Online engine has no shared language registry")
        return {
            "task": registry.resolve(batch["task_index"], kind="task"),
            "subtask": registry.resolve(batch["subtask_index"], kind="subtask"),
        }

    # -----------------------------------------------------------------------
    # Refill criterion
    # -----------------------------------------------------------------------

    def request_refill(self) -> int:
        """Request a fill, coalescing an outstanding manual request.

        Returns:
            Generation that will contain the requested fresh data.

        Raises:
            RuntimeError: If called outside READY or from a reader process.
        """
        self._require_owner_process("request_refill")
        error = self._receive_worker_error()
        if error is not None:
            raise error
        self._require_state(OnlineDataEngineState.READY, "request_refill")
        self._ensure_worker_alive()
        with self._progress.values.get_lock():
            values = self._progress.values
            if values[0] <= values[1]:
                values[0] = values[1] + 1
                self._fill_signal.set()
            return int(values[0])

    def wait_for_generation(self, generation: int, timeout: float = 300.0) -> None:
        """Wait for a requested generation while propagating worker failures.

        Args:
            generation: Positive generation returned by request_refill.
            timeout: Finite positive timeout in seconds.

        Raises:
            ValueError: If the generation was not requested or timeout is invalid.
            TimeoutError: If publication does not finish before the deadline.
        """
        if type(generation) is not int or generation < 1:
            raise ValueError("generation must be a positive integer")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        with self._progress.values.get_lock():
            if generation > self._progress.values[0]:
                raise ValueError("generation has not been requested")
        deadline = time.monotonic() + timeout
        while True:
            error = self._receive_worker_error()
            if error is not None:
                raise error
            self._require_state(OnlineDataEngineState.READY, "wait_for_generation")
            self._ensure_worker_alive()
            error = self._receive_worker_error()
            if error is not None:
                raise error
            with self._progress.values.get_lock():
                if self._progress.values[1] >= generation:
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ODS generation {generation} was not published in {timeout}s"
                )
            self._close_signal.wait(min(0.05, max(0.0, deadline - time.monotonic())))

    def drain_lifecycle_events(self) -> list[dict[str, object]]:
        """Return and remove pending publication/eviction records in order.

        Owner-only, including after stop or failure, so diagnostics are not lost.
        Disabled logging returns an empty list. Each record contains ``kind``,
        ``sequence``, ``run_id``, ``episode_id``, ``generation``, ``slot``,
        ``monotonic_ns``, ``valid_frame_indices``, ``eligible_anchor_indices``,
        ``valid_frames`` and ``eligible_anchors``. Eligible anchors refer to
        episode-mode context sampling (valid frames with nonnegative continuity).
        This is a host-local monotonic clock, not a wall-clock timestamp.

        Drain regularly while producing, including during owner-side waits.
        ``wait_for_generation`` does not consume these records and may time out
        if an enabled log is full. Records are detached from later row reuse.

        Returns:
            Detached lifecycle records in publication order, or an empty list
            when logging is disabled or no records are pending.

        Raises:
            RuntimeError: If called outside the engine's owner process.
        """
        self._require_owner_process("drain_lifecycle_events")
        events = getattr(self, "_lifecycle_events", None)
        return [] if events is None else events.drain()

    def get_stats(self) -> dict[str, int | float]:
        """Return a detached snapshot of production, consumption and valid rows.

        Returns:
            Engine identity, requested/completed generations, episode/frame
            counts, cumulative production seconds and trajectory sample count.
        """
        names = (
            "requested_generation",
            "completed_generation",
            "successful_episodes",
            "failed_episodes",
            "generated_frames",
            "production_seconds",
            "sampled_chunks",
        )
        with self._progress.values.get_lock():
            values = list(self._progress.values[:])
        stats = {
            name: (value if name == "production_seconds" else int(value))
            for name, value in zip(names, values)
        }
        stats["run_id"] = self.run_id
        with self._lock_index.get_lock():
            valid = self.shared_buffer["valid"].clone()
            start, end = self._lock_index[:]
            if start >= 0:
                valid[start:end] = False
            stats["valid_frames"] = int(valid.sum().item())
        return stats

    def _trigger_refill_if_needed(self, count: int = 1) -> None:
        """Accumulate sample count and trigger a buffer refill when the threshold is reached.

        This method is called by :meth:`sample_batch` after every batch.  The
        refill is only requested when the fill signal is not already pending
        (i.e. the subprocess has finished the previous refill).

        Args:
            count: Number of individual trajectory samples drawn in the latest
                call to :meth:`sample_batch` (typically equal to *batch_size*).
        """
        progress = getattr(self, "_progress", None)
        if progress is not None:
            progress.add(6, count)
        if self.cfg.refill_policy == "manual":
            return
        with self._sample_count.get_lock():
            self._sample_count.value += count
            should_refill = (
                self._sample_count.value >= self.cfg.refill_threshold * self.buffer_size
                and not self._fill_signal.is_set()
            )
            if should_refill:
                self._sample_count.value = 0
                if progress is not None:
                    progress.add(0, 1)
                self._fill_signal.set()

        if should_refill:
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
            process.join(timeout=self.cfg.shutdown_timeout if is_alive else 0)
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

        Sets the close signal and waits up to ``cfg.shutdown_timeout`` for the
        subprocess to exit gracefully (it checks the signal between rollout
        steps). If it remains alive after that wait, it is force-terminated.

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
