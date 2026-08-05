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

import time
import torch
import multiprocessing as mp

from typing import Literal
from multiprocessing.sharedctypes import Synchronized, SynchronizedArray
from multiprocessing.synchronize import Event as MpEvent
from tensordict import TensorDict
from tqdm import tqdm

from embodichain.lab.sim.cfg import RenderCfg
from embodichain.utils.logger import log_info, log_error
from embodichain.utils import configclass


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


# ---------------------------------------------------------------------------
# Subprocess entry point (module-level so it can be pickled by multiprocessing)
# ---------------------------------------------------------------------------


def _sim_worker_fn(
    cfg: OnlineDataEngineCfg,
    shared_buffer: TensorDict,
    lock_index: SynchronizedArray,
    fill_signal: MpEvent,
    init_signal: MpEvent,
    close_signal: MpEvent,
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
    from embodichain.lab.sim import SimulationManagerCfg
    from embodichain.utils.logger import log_info, log_warning, log_error

    gym_config: dict = cfg.gym_config
    action_config: dict = cfg.action_config

    # Build env config from the gym configuration dictionary.
    env_cfg = config_to_cfg(gym_config, manager_modules=get_manager_modules())
    env_cfg.filter_dataset_saving = True
    env_cfg.init_rollout_buffer = False
    env_cfg.sim_cfg = SimulationManagerCfg(
        headless=gym_config.get("headless", True),
        sim_device=gym_config.get("device", "cpu"),
        render_cfg=RenderCfg(renderer=gym_config.get("renderer", "hybrid")),
        gpu_id=gym_config.get("gpu_id", 0),
    )

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
            lock_index[0] = 0
            lock_index[1] = num_envs

            rollout_idx = 0
            while rollout_idx < num_rollouts_per_fill:
                if close_signal.is_set():
                    return

                tmp_buffer = shared_buffer[lock_index[0] : lock_index[1], :]
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
                    f"lock_index=[{lock_index[0]}, {lock_index[1]}], ",
                    color="cyan",
                )

                # Advance lock_index to the next write slice.
                next_start = lock_index[0] + num_envs
                next_end = lock_index[1] + num_envs
                if next_start >= buffer_size:
                    # Wrap around to the start of the buffer.
                    next_start = 0
                    next_end = num_envs
                elif next_end > buffer_size:
                    next_end = buffer_size
                    next_start = buffer_size - num_envs

                lock_index[0] = next_start
                lock_index[1] = next_end

            # # Signal that the buffer contains valid data for the first time.
            # # is_set() is checked so subsequent refills do not redundantly set it.
            if not init_signal.is_set():
                init_signal.set()
                log_info(
                    "[Simulation Process] Initial buffer fill complete. Engine is ready.",
                    color="cyan",
                )

            # # At this point the entire buffer has been filled with fresh data, and
            # # all the data in the buffer is valid and safe to sample from.
            lock_index[0] = -1
            lock_index[1] = -1

    except KeyboardInterrupt:
        log_warning("[Simulation Process] Stopping (KeyboardInterrupt).")
    except Exception as e:
        log_error(f"[Simulation Process] Unhandled error: {e}")
    finally:
        env.close()


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

    **Initialisation barrier**

    The :attr:`is_init` property returns ``False`` until the subprocess
    completes the very first full buffer fill, after which it becomes
    permanently ``True``.  Training code should wait on this flag before
    calling :meth:`sample_batch` to avoid drawing all-zero data.

    Args:
        cfg: Engine configuration.

    Attributes:
        shared_buffer: Shared-memory TensorDict of shape
            ``[buffer_size, max_episode_steps, ...]``.
        buffer_size: Total number of trajectory slots in the shared buffer.
        device: Device of the shared buffer.
        is_init: ``True`` once the buffer has been populated at least once.
    """

    def __init__(self, cfg: OnlineDataEngineCfg) -> None:
        self.cfg = cfg

        if cfg.max_generation_attempts < 1:
            raise ValueError(
                "max_generation_attempts must be at least 1, "
                f"got {cfg.max_generation_attempts}."
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

        # Handle to the simulation subprocess, set in start() and used in stop().
        self._sim_process: mp.Process | None = None

    def start(self) -> None:
        self._sim_process: mp.Process = self._mp_ctx.Process(
            target=_sim_worker_fn,
            args=(
                self.cfg,
                self.shared_buffer,
                self._lock_index,
                self._fill_signal,
                self._init_signal,
                self._close_signal,
            ),
            daemon=True,
        )
        self._sim_process.start()
        log_info(
            f"[OnlineDataEngine] Simulation subprocess started (PID={self._sim_process.pid}).",
            color="green",
        )

        # Trigger the initial fill so data is ready before the first sample.
        self._fill_signal.set()

        while not self.is_init:
            time.sleep(0.5)

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
    def is_init(self) -> bool:
        """Whether the shared buffer has been fully populated at least once.

        Returns ``True`` after the simulation subprocess completes its first
        full buffer fill, ``False`` while that initial fill is still in
        progress.  Callers that must not sample stale (all-zero) data can
        poll or block on this property before entering their training loop::

            while not engine.is_init:
                time.sleep(0.5)

        Returns:
            ``True`` once the buffer contains valid trajectory data.
        """
        return self._init_signal.is_set()

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

        # Build the set of rows that are safe to sample from: all valid rows
        # minus the slice currently being written by the subprocess.
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
            same_segment = (segment_windows == segment_windows[..., :1]).all(dim=-1) & (
                segment_windows[..., 0] >= 0
            )
            valid_windows &= same_segment
        elif sampling_mode == "boundary":
            segment_windows = segment_ids.unfold(1, chunk_size, 1)
            crosses_boundary = (
                segment_windows[..., 1:] != segment_windows[..., :-1]
            ).any(dim=-1)
            valid_windows &= crosses_boundary

        candidates = valid_windows.nonzero(as_tuple=False)
        if candidates.numel() == 0:
            raise RuntimeError(
                "[OnlineDataEngine] No unlocked valid chunk satisfies "
                f"sampling_mode={sampling_mode!r} and chunk_size={chunk_size}."
            )

        sampled_candidate_ids = torch.randint(
            0, candidates.shape[0], (batch_size,), device=candidates.device
        )
        selected = candidates[sampled_candidate_ids]
        row_indices = selected[:, 0]
        start_indices = selected[:, 1]

        time_offsets = torch.arange(chunk_size, device=start_indices.device)
        time_indices = start_indices[:, None] + time_offsets[None, :]

        result = self.shared_buffer[row_indices[:, None], time_indices]

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

    def stop(self) -> None:
        """Terminate the simulation subprocess and release resources.

        Sets the close signal and waits briefly for the subprocess to exit
        gracefully (it checks the signal between rollout steps).  If the
        subprocess is still alive after the grace period it is force-terminated.

        Safe to call multiple times — subsequent calls are no-ops if the
        subprocess has already been terminated.
        """
        if self._sim_process is None or not self._sim_process.is_alive():
            return

        # Ask the subprocess to stop and unblock it if it is waiting on fill_signal.
        self._close_signal.set()
        self._fill_signal.set()

        # Allow time for a graceful exit (close_signal is checked between steps).
        self._sim_process.join(timeout=5.0)

        if self._sim_process.is_alive():
            self._sim_process.terminate()
            self._sim_process.join(timeout=3.0)

        log_info("[OnlineDataEngine] Simulation subprocess terminated.", color="green")

    def __del__(self) -> None:
        self.stop()
