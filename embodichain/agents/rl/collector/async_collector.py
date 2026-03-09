# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import threading
from typing import Callable, Optional
import torch
from tensordict import TensorDict
from collections import deque

from ..utils.helper import dict_to_tensordict
from .base import BaseCollector


class AsyncCollector(BaseCollector):
    """Asynchronous data collector for VLA RL scenarios.

    Runs in a background thread to continuously collect transitions while
    the main thread performs model updates. Designed for scenarios where
    model inference is slow (e.g., VLA) but training is fast.

    Key features:
    - Background thread: Continuous data collection
    - Thread-safe buffer: Lock-protected writes
    - Step-level collection: Individual transitions added to buffer
    - Episode statistics tracking: Rewards and lengths

    Usage:
        collector = AsyncCollector(env, policy, buffer, device, ...)
        collector.start()  # Begin background collection
        # ... main thread does training ...
        collector.stop()   # Stop collection
    """

    def __init__(
        self,
        env,
        policy,
        buffer,
        device: torch.device,
        on_step_callback: Optional[Callable] = None,
    ):
        """Initialize async collector.

        Args:
            env: Environment to collect from
            policy: Policy for action selection
            buffer: VLABuffer instance (shared with Trainer)
            device: Device for tensor operations
            on_step_callback: Optional callback(transition, env_info) called after each step
        """
        super().__init__(env, policy, device, on_step_callback)
        self.buffer = buffer

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Episode statistics
        self._episode_count = 0
        self._step_count = 0

    def start(self):
        """Start background collection thread."""
        if self._running:
            raise RuntimeError("Collector is already running")

        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        print("[AsyncCollector] Background collection started")

    def collect(self, **kwargs) -> TensorDict:
        """For AsyncCollector, data is collected continuously in background.

        This method is just for interface compatibility with BaseCollector.
        Actual data retrieval happens through buffer.get().

        Returns:
            Empty TensorDict (not used in async mode)
        """
        raise NotImplementedError(
            "AsyncCollector collects data in background thread. "
            "Use buffer.get() to retrieve data instead."
        )

    def stop(self):
        """Stop background collection thread."""
        if not self._running:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                print("[AsyncCollector] Warning: Thread did not stop cleanly")

        print(
            f"[AsyncCollector] Stopped (collected {self._step_count} steps, {self._episode_count} episodes)"
        )

    def is_running(self) -> bool:
        """Check if collector is currently running."""
        return self._running

    def get_stats(self) -> dict:
        """Get collection statistics."""
        with self._lock:
            return {
                "steps_collected": self._step_count,
                "episodes_collected": self._episode_count,
            }

    def _collect_loop(self):
        """Background thread main loop: collect full rollout, then add to buffer.

        GAE requires sequential timesteps within the same trajectory. We accumulate
        T steps (one rollout) locally, then add the complete rollout to buffer.
        This ensures correct per-env trajectory ordering for GAE computation.
        """
        rollout_length = self.buffer.rollout_length
        current_td = self.obs_tensordict

        while self._running:
            try:
                rollout_list = []

                for t in range(rollout_length):
                    # Policy forward (no_grad for inference)
                    with torch.no_grad():
                        self.policy.train()
                        self.policy.forward(current_td)

                    action = (
                        current_td["env_action"]
                        if "env_action" in current_td.keys()
                        else current_td["action"]
                    )
                    env_action = self._format_env_action(action)

                    next_obs_dict, reward, terminated, truncated, env_info = (
                        self.env.step(env_action)
                    )

                    next_obs_td = dict_to_tensordict(next_obs_dict, self.device)
                    done = terminated | truncated
                    next_obs_for_td = next_obs_td["observation"]
                    if hasattr(self.policy, "reset_envs"):
                        self.policy.reset_envs(done, next_obs_for_td)
                    batch_size = next_obs_td.batch_size[0]

                    next_td = TensorDict(
                        {
                            "observation": next_obs_for_td,
                            "reward": (
                                reward.float().unsqueeze(-1)
                                if reward.dim() == 1
                                else reward.float()
                            ),
                            "done": (
                                done.bool().unsqueeze(-1)
                                if done.dim() == 1
                                else done.bool()
                            ),
                            "terminated": (
                                terminated.bool().unsqueeze(-1)
                                if terminated.dim() == 1
                                else terminated.bool()
                            ),
                            "truncated": (
                                truncated.bool().unsqueeze(-1)
                                if truncated.dim() == 1
                                else truncated.bool()
                            ),
                        },
                        batch_size=torch.Size([batch_size]),
                        device=self.device,
                    )

                    with torch.no_grad():
                        next_value_td = TensorDict(
                            {"observation": next_obs_for_td},
                            batch_size=next_td.batch_size,
                            device=self.device,
                        )
                        self.policy.get_value(next_value_td)
                        next_td["value"] = next_value_td["value"]

                    current_td["next"] = next_td
                    rollout_list.append(current_td.clone())

                    if self.on_step_callback is not None:
                        self.on_step_callback(current_td, env_info)

                    if done.any():
                        with self._lock:
                            self._episode_count += done.sum().item()

                    current_td = next_obs_td

                # Stack along dim=1: list of [N,...] -> [N, T, ...] (batch-first)
                rollout = torch.stack(rollout_list, dim=1)
                self.obs_tensordict = current_td

                with self._lock:
                    self.buffer.add_rollout(rollout)
                    self._step_count += rollout.batch_size[0] * rollout.batch_size[1]

            except Exception as e:
                print(f"[AsyncCollector] Error in collection loop: {e}")
                import traceback

                traceback.print_exc()
                break

        print("[AsyncCollector] Collection loop exited")


class AsyncCollectorStats:
    """Helper class to track async collection statistics safely."""

    def __init__(self, num_envs: int, device: torch.device):
        self.device = device
        self.num_envs = num_envs

        # Episode tracking (on device)
        self.curr_ret = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.curr_len = torch.zeros(num_envs, dtype=torch.int32, device=device)

        # Completed episodes (CPU)
        self.ret_window = deque(maxlen=100)
        self.len_window = deque(maxlen=100)
        self._lock = threading.Lock()

    def update(self, reward: torch.Tensor, done: torch.Tensor):
        """Update episode statistics (thread-safe).

        Args:
            reward: Reward tensor [N] or [N, 1]
            done: Done tensor [N] or [N, 1]
        """
        # Ensure correct shape
        if reward.dim() > 1:
            reward = reward.squeeze(-1)
        if done.dim() > 1:
            done = done.squeeze(-1)

        with self._lock:
            # Update cumulative stats
            self.curr_ret += reward
            self.curr_len += 1

            # Handle completed episodes
            done_idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
            if done_idx.numel() > 0:
                finished_ret = self.curr_ret[done_idx].detach().cpu().tolist()
                finished_len = self.curr_len[done_idx].detach().cpu().tolist()
                self.ret_window.extend(finished_ret)
                self.len_window.extend(finished_len)

                # Reset for finished episodes
                self.curr_ret[done_idx] = 0
                self.curr_len[done_idx] = 0

    def get_avg_stats(self) -> tuple[float, float]:
        """Get average episode return and length (thread-safe).

        Returns:
            (avg_return, avg_length) or (nan, nan) if no episodes completed
        """
        with self._lock:
            if len(self.ret_window) == 0:
                return float("nan"), float("nan")
            return float(sum(self.ret_window) / len(self.ret_window)), float(
                sum(self.len_window) / len(self.len_window)
            )
