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

"""Asynchronous LeRobot recorder for parallel environments.

This module provides :class:`AsyncLeRobotRecorder`, which decouples episode
saving from the simulation loop. It is the recommended recorder when running
many parallel environments that complete episodes together: instead of
blocking ``env.reset()`` while episodes are converted and flushed to disk, the
completed episode buffers are cloned and handed to a background worker thread,
so the simulator can keep stepping.
"""

from __future__ import annotations

import copy
import queue
import threading
from typing import TYPE_CHECKING, Dict, Optional, Union

import torch

from embodichain.utils import logger
from embodichain.lab.gym.envs.demo import DEMO_ANNOTATION_KEYS
from .datasets import LeRobotRecorder

__all__ = ["AsyncLeRobotRecorder"]

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401

    LEROBOT_AVAILABLE = True
    __all__ = ["AsyncLeRobotRecorder"]
except ImportError:
    LEROBOT_AVAILABLE = False
    __all__ = []


class AsyncLeRobotRecorder(LeRobotRecorder):
    """LeRobot recorder that saves episodes on a background thread.

    Drop-in replacement for :class:`LeRobotRecorder` selected via the dataset
    config ``"func": "AsyncLeRobotRecorder"``. It shares the same on-disk
    format and feature-building logic; only the *timing* of the save differs.

    Why this helps parallel environments
    -------------------------------------
    In the synchronous recorder, :meth:`LeRobotRecorder.__call__` runs inside
    ``env.reset()`` (via ``DatasetManager.apply``) and blocks the simulator
    while it iterates ``add_frame`` + ``save_episode`` for every finished env.
    With ``num_envs=N`` all finishing at once, the sim stalls for the *sum* of
    all episodes' save time every reset.

    This recorder instead, on each ``apply``:

    1. Reads each env's rollout-buffer slice (``obs``/``actions``).
    2. **Clones** the slice to CPU (detached from the live buffer).
    3. Clones frame annotations and episode/segment metadata with the payload.
    4. Pushes the detached payload onto a queue.
    5. Returns immediately - the sim is free to reset and keep stepping.

    A single daemon worker thread drains the queue and runs the standard
    :meth:`LeRobotRecorder._save_single_episode` on each cloned payload.

    Correctness
    -----------
    * **No concurrent dataset access.** ``LeRobotDataset`` is not thread-safe;
      only the worker thread ever calls ``add_frame`` / ``save_episode`` /
      mutates ``curr_episode``. The main thread only enqueues and, at close,
      drains.
    * **No buffer race.** The slice is cloned in the caller thread *before*
      the buffer is cleared on reset, so the worker never reads memory that
      the sim is overwriting.
    * **Ordering.** A single worker preserves FIFO episode order, so
      ``episode_index`` assignment is deterministic.
    * **Drain on close.** :meth:`finalize` joins the worker before the parent
      flushes the image writer and finalizes the dataset.

    .. note::
        The clone copies each episode's camera frames into host RAM. Memory
        use is bounded by how far the worker falls behind (typically it keeps
        up, since per-frame PNG write is the only heavy step and can itself be
        offloaded via ``image_writer_threads``). For very high resolutions or
        many envs, monitor RSS.

    Args:
        cfg: :class:`~embodichain.lab.gym.envs.managers.cfg.DatasetFunctorCfg`
            with the same ``params`` as ``LeRobotRecorder``. The
            ``image_writer_threads`` / ``image_writer_processes`` params are
            honored and combine with the background worker (two levels of
            async: episode conversion off the sim thread, PNG writes off the
            worker thread).
        env: The environment instance.
    """

    def __init__(self, cfg, env: EmbodiedEnv):
        if not LEROBOT_AVAILABLE:
            logger.log_error(
                "LeRobot is not installed. Please install it with: pip install lerobot"
            )
        super().__init__(cfg, env)

        # Single-worker queue. A single worker guarantees LeRobotDataset is
        # only ever touched from one thread (it is not thread-safe) and keeps
        # episode ordering deterministic.
        self._save_queue: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._worker: threading.Thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncLeRobotRecorder-worker",
            daemon=True,
        )
        self._worker.start()
        logger.log_info(
            "[AsyncLeRobotRecorder] Background save worker started; "
            "episode saves will not block env.reset()."
        )

    def _worker_loop(self) -> None:
        """Consume cloned episodes from the queue and persist them."""
        while True:
            item = self._save_queue.get()
            if item is None:
                # Sentinel: finalize() is draining. Exit the worker.
                break
            env_id, obs_clone, action_clone, annotations, episode_metadata = item
            try:
                self._save_single_episode(
                    env_id,
                    obs_clone,
                    action_clone,
                    annotations=annotations,
                    episode_metadata=episode_metadata,
                )
            except Exception as e:  # noqa: BLE001 - worker must not die
                logger.log_error(
                    f"[AsyncLeRobotRecorder] Background worker failed on "
                    f"env {env_id}: {e}"
                )

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        save_path: Optional[str] = None,
        robot_meta: Optional[Dict] = None,
        instruction: Optional[str] = None,
        extra: Optional[Dict] = None,
        use_videos: bool = False,
        **kwargs,
    ) -> None:
        """Enqueue completed episodes for background saving.

        Reads each env's rollout-buffer slice, clones it to CPU (so the worker
        is immune to buffer reuse on reset), and pushes it onto the queue. The
        actual conversion and disk I/O happen asynchronously on the worker
        thread. Returns immediately so ``env.reset()`` is not blocked.

        Args:
            env: The environment instance.
            env_ids: Environment IDs to save. If None, enqueues all envs.
            save_path: Unused at call time (honored at construction).
            robot_meta: Unused at call time (honored at construction).
            instruction: Unused at call time (honored at construction).
            extra: Unused at call time (honored at construction).
            use_videos: Unused at call time (honored at construction).
            **kwargs: Construction-only params (e.g. ``image_writer_threads``)
                passed through by ``DatasetManager.apply``; honored in
                :meth:`__init__`, ignored here.
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        elif isinstance(env_ids, (list, range)):
            env_ids = torch.tensor(list(env_ids), device=env.device)

        if len(env_ids) == 0:
            return

        for env_id in env_ids.cpu().tolist():
            step = self._episode_length(env_id)
            obs_view = env.rollout_buffer["obs"][env_id, :step]
            action_view = env.rollout_buffer["actions"][env_id, :step]
            # Clone in the caller thread: the rollout buffer is cleared and
            # reused by the next episode on reset, so the worker must not hold
            # a view into it.
            obs_clone = obs_view.clone().cpu()
            action_clone = action_view.clone().cpu()
            annotations = {
                key: env.rollout_buffer[key][env_id, :step].clone().cpu()
                for key in DEMO_ANNOTATION_KEYS
                if key in env.rollout_buffer.keys()
            }
            metadata_getter = getattr(env, "get_demo_episode_metadata", None)
            episode_metadata = (
                copy.deepcopy(metadata_getter(env_id))
                if metadata_getter is not None
                else None
            )
            self._save_queue.put(
                (
                    env_id,
                    obs_clone,
                    action_clone,
                    annotations,
                    episode_metadata,
                )
            )

    def finalize(self) -> Optional[str]:
        """Drain the background worker, then finalize the dataset.

        Signals the worker to stop after finishing all queued episodes, waits
        for it to exit, and delegates to the parent to flush any remaining
        buffer, stop the async image writer, and finalize dataset metadata.
        """
        # Signal the worker to exit once the queue is drained.
        self._save_queue.put(None)
        self._worker.join()
        # Parent flushes leftover episodes, stops image writer, finalizes.
        return super().finalize()
