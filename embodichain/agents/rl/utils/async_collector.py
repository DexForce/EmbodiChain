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
import time
from typing import Callable, Optional
import torch
from tensordict import TensorDict
from collections import deque

from .helper import dict_to_tensordict


class AsyncCollector:
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
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.device = device
        self.on_step_callback = on_step_callback

        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Episode statistics
        self._episode_count = 0
        self._step_count = 0

        # Initialize observation
        obs_dict, _ = self.env.reset()
        self.obs_tensordict = dict_to_tensordict(obs_dict, self.device)

    def start(self):
        """Start background collection thread."""
        if self._running:
            raise RuntimeError("Collector is already running")

        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        print("[AsyncCollector] Background collection started")

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
        """Background thread main loop: continuously collect transitions.

        This method runs in a separate thread and continuously:
        1. Gets action from policy
        2. Steps environment
        3. Constructs transition TensorDict
        4. Adds to buffer (thread-safe)
        5. Updates statistics
        """
        current_td = self.obs_tensordict

        while self._running:
            try:
                # Policy forward (no_grad for inference)
                with torch.no_grad():
                    self.policy.train()  # Use stochastic policy
                    self.policy.forward(current_td)

                # Extract action
                action = current_td["action"]
                action_type = getattr(self.env, "action_type", "delta_qpos")
                action_dict = {action_type: action}

                # Environment step
                next_obs_dict, reward, terminated, truncated, env_info = self.env.step(
                    action_dict
                )

                # Convert observation to TensorDict
                next_obs_td = dict_to_tensordict(next_obs_dict, self.device)
                done = terminated | truncated
                next_obs_for_td = next_obs_td["observation"]
                batch_size = next_obs_td.batch_size[0]

                # Build "next" TensorDict
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

                # Compute next value for bootstrapping (GAE computation)
                with torch.no_grad():
                    next_value_td = TensorDict(
                        {"observation": next_obs_for_td},
                        batch_size=next_td.batch_size,
                        device=self.device,
                    )
                    self.policy.get_value(next_value_td)
                    next_td["value"] = next_value_td["value"]

                # Add "next" to current transition
                current_td["next"] = next_td

                # Flatten transition for buffer (remove batch dimension for single-step storage)
                # Current buffer expects transitions without batch dimension
                # We need to add each parallel env's transition separately
                for env_idx in range(batch_size):
                    transition = current_td[env_idx]  # Extract single env's transition

                    # Thread-safe buffer write
                    with self._lock:
                        self.buffer.add(transition)
                        self._step_count += 1

                # Callback for statistics
                if self.on_step_callback is not None:
                    self.on_step_callback(current_td, env_info)

                # Handle episode termination
                if done.any():
                    with self._lock:
                        self._episode_count += done.sum().item()

                # Prepare next observation
                current_td = next_obs_td

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
