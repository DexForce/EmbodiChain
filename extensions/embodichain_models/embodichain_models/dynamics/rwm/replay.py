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

"""Owned, bounded device-resident transitions for world-model fitting.

Storage is time-major ``[capacity, num_envs, ...]``. Sampling gathers independent
tensors on the storage device. Append and sampling are single-owner operations;
concurrent writers or readers require synchronization supplied by the caller.
Physical next states must be captured before environment auto-reset.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

__all__ = ["TransitionReplay"]


class TransitionReplay:
    """Keep graph-free physical transitions in a fixed-size device ring.

    Args:
        capacity: Maximum vector steps retained, per environment.
        num_envs: Number of transitions supplied on each append.
        state_dim: Flat state width, independently of a particular robot model.
        action_dim: Flat control width.
        context_dim: Optional recorded physical-context width.
        device: Storage and sampling device.
        max_bytes: Optional allocation ceiling, checked before allocating.
        seed: Replay-owned sampling RNG seed.

    Float payloads are FP32. Terminal and truncated transitions are retained;
    their following episode is excluded from a sampled window using a validity
    mask and repeated final valid transition. Caller-owned simulator views are
    never retained, and previously returned samples survive ring replacement.
    """

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        state_dim: int,
        action_dim: int,
        *,
        context_dim: int = 0,
        device: str | torch.device = "cpu",
        max_bytes: int | None = None,
        seed: int = 0,
    ) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (capacity, num_envs, state_dim, action_dim)
        ):
            raise ValueError(
                "Replay capacity, environment count and state/action widths must be positive integers"
            )
        if type(context_dim) is not int or context_dim < 0:
            raise ValueError("context_dim must be a nonnegative integer")
        self.capacity, self.num_envs = capacity, num_envs
        self.state_dim, self.action_dim, self.context_dim = (
            state_dim,
            action_dim,
            context_dim,
        )
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._fields = {
            "state": (state_dim, torch.float32),
            "next_state": (state_dim, torch.float32),
            "action": (action_dim, torch.float32),
            "context": (context_dim, torch.float32),
            "terminated": (None, torch.bool),
            "truncated": (None, torch.bool),
            "episode_id": (None, torch.int64),
            "episode_step": (None, torch.int64),
        }
        scalar_bytes = 4 * (2 * state_dim + action_dim + context_dim) + 2 + 16
        self.allocated_bytes = capacity * num_envs * scalar_bytes + 2 * num_envs * 8
        if max_bytes is not None and (
            type(max_bytes) is not int or max_bytes < self.allocated_bytes
        ):
            raise ValueError(
                f"Replay needs {self.allocated_bytes} bytes, exceeding max_bytes"
            )
        self._data = {
            name: torch.zeros(
                (capacity, num_envs) + (() if width is None else (width,)),
                dtype=dtype,
                device=self.device,
            )
            for name, (width, dtype) in self._fields.items()
        }
        self._episode_id = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._episode_step = torch.zeros_like(self._episode_id)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.size, self.cursor = 0, 0

    def _check(self, name: str, value: torch.Tensor) -> None:
        width, dtype = self._fields[name]
        shape = (self.num_envs,) + (() if width is None else (width,))
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != dtype
            or value.device != self.device
        ):
            raise ValueError(
                f"{name} must have shape {shape}, dtype {dtype} and device {self.device}"
            )

    @torch.no_grad()
    def append(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        terminated: torch.Tensor | None = None,
        truncated: torch.Tensor | None = None,
    ) -> None:
        """Copy one vector step; flags end episodes after the stored transition.

        Args:
            state: Current FP32 states, ``[num_envs, state_dim]``.
            action: Executed FP32 controls.
            next_state: Physical next states captured before any reset.
            context: Recorded FP32 physical context; required for nonzero width.
            terminated: Hard terminal flags, default false.
            truncated: Time-limit flags, default false, retained separately.
        """
        if context is None:
            if self.context_dim:
                raise ValueError("Configured replay context must be supplied")
            context = state.new_empty((self.num_envs, 0))
        flags = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminated = flags if terminated is None else terminated
        truncated = flags if truncated is None else truncated
        payload = dict(
            state=state,
            action=action,
            next_state=next_state,
            context=context,
            terminated=terminated,
            truncated=truncated,
        )
        for name, value in payload.items():
            self._check(name, value)
        # Validate all fields before mutating a row. copy_ under no_grad owns
        # storage even when inputs are views or require gradients.
        for name, value in payload.items():
            self._data[name][self.cursor].copy_(value)
        self._data["episode_id"][self.cursor].copy_(self._episode_id)
        self._data["episode_step"][self.cursor].copy_(self._episode_step)
        done = terminated | truncated
        self._episode_id.add_(done.to(torch.int64))
        self._episode_step.copy_(torch.where(done, 0, self._episode_step + 1))
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _row(self, ages: torch.Tensor) -> torch.Tensor:
        return (self.cursor - self.size + ages) % self.capacity

    def _draw(
        self, count: int, high: int, generator: torch.Generator | None
    ) -> torch.Tensor:
        return torch.randint(
            high,
            (count,),
            device=self.device,
            generator=self.generator if generator is None else generator,
        )

    def sample_windows(
        self,
        batch_size: int,
        horizon: int,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Gather horizon transitions without carrying data across episodes.

        Args:
            batch_size: Number of sampled environment/window pairs.
            horizon: Requested transition count, no greater than retained steps.
            generator: Optional sampling RNG on the replay device.

        Returns:
            Owned field tensors with ``[batch, horizon]`` leading axes and a
            Boolean ``valid`` mask. The terminal transition itself is valid;
            positions after it repeat its data with ``valid=False``.
        """
        if (
            type(batch_size) is not int
            or batch_size < 1
            or type(horizon) is not int
            or not 1 <= horizon <= self.size
        ):
            raise ValueError(
                "Need a positive batch and a horizon within retained steps"
            )
        starts = self._draw(batch_size, self.size - horizon + 1, generator)
        envs = self._draw(batch_size, self.num_envs, generator)
        ages = starts[:, None] + torch.arange(horizon, device=self.device)
        rows = self._row(ages)
        episode = self._data["episode_id"][rows, envs[:, None]]
        valid = episode == episode[:, :1]
        # Preserve only the prefix in the starting episode, even if malformed
        # externally restored episode labels later repeat.
        valid = valid.to(torch.int64).cumprod(1).bool()
        positions = torch.arange(horizon, device=self.device).expand(batch_size, -1)
        final = valid.sum(1, keepdim=True) - 1
        safe_rows = self._row(starts[:, None] + torch.minimum(positions, final))
        result = {
            name: value[safe_rows, envs[:, None]] for name, value in self._data.items()
        }
        result["valid"] = valid
        return result

    def sample_starts(
        self,
        batch_size: int,
        history_length: int,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        """Gather imagination starts and oldest-first boundary-padded history.

        Args:
            batch_size: Number of starts.
            history_length: Number of states including the selected current one.
            generator: Optional sampling RNG on the replay device.

        Returns:
            ``start`` fields, ``history`` fields and Boolean ``history_valid``.
            Missing history repeats the earliest available state in that same
            episode. Requests never retry or wait for more data.
        """
        if (
            self.size == 0
            or type(batch_size) is not int
            or batch_size < 1
            or type(history_length) is not int
            or history_length < 1
        ):
            raise ValueError("Need nonempty replay and positive start/history sizes")
        ages = self._draw(batch_size, self.size, generator)
        envs = self._draw(batch_size, self.num_envs, generator)
        rows = self._row(ages)
        steps = self._data["episode_step"][rows, envs]
        back = torch.arange(history_length - 1, -1, -1, device=self.device)
        available = torch.minimum(steps, ages)[:, None]
        valid = back <= available
        history_rows = self._row(ages[:, None] - torch.minimum(back, available))
        return {
            "start": {name: value[rows, envs] for name, value in self._data.items()},
            "history": {
                name: value[history_rows, envs[:, None]]
                for name, value in self._data.items()
            },
            "history_valid": valid,
        }

    def state_dict(self) -> dict[str, Any]:
        """Snapshot owned replay content, boundaries and the sampling RNG.

        Returns:
            Independent tensors suitable for optional exact replay restoration.
            Full snapshots cost another replay-sized allocation; routine RWM
            recovery may instead rebuild and refill replay explicitly.
        """
        return {
            "format": "transition-replay-v1",
            "layout": (
                self.capacity,
                self.num_envs,
                self.state_dim,
                self.action_dim,
                self.context_dim,
            ),
            "data": {name: value.clone() for name, value in self._data.items()},
            "episode_id": self._episode_id.clone(),
            "episode_step": self._episode_step.clone(),
            "size": self.size,
            "cursor": self.cursor,
            "rng": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a matching replay after validating the complete layout.

        Args:
            state: Snapshot produced by :meth:`state_dict`. Payloads may reside
                on another device and are copied to this replay's device.
        """
        layout = (
            self.capacity,
            self.num_envs,
            self.state_dim,
            self.action_dim,
            self.context_dim,
        )
        if (
            state.get("format") != "transition-replay-v1"
            or tuple(state.get("layout", ())) != layout
        ):
            raise ValueError("Replay snapshot layout differs")
        if state["data"].keys() != self._data.keys():
            raise ValueError("Replay snapshot fields differ")
        size, cursor = state["size"], state["cursor"]
        if (
            type(size) is not int
            or not 0 <= size <= self.capacity
            or type(cursor) is not int
            or not 0 <= cursor < self.capacity
        ):
            raise ValueError("Invalid replay snapshot cursor")
        pairs = [(value, state["data"][name]) for name, value in self._data.items()]
        pairs.extend(
            (
                (self._episode_id, state["episode_id"]),
                (self._episode_step, state["episode_step"]),
            )
        )
        for current, saved in pairs:
            if (
                not isinstance(saved, torch.Tensor)
                or current.shape != saved.shape
                or current.dtype != saved.dtype
            ):
                raise ValueError("Replay snapshot tensor layout differs")
        candidate_rng = torch.Generator(device=self.device)
        candidate_rng.set_state(state["rng"].cpu())
        for current, saved in pairs:
            current.copy_(saved)
        self.generator.set_state(candidate_rng.get_state())
        self.size, self.cursor = size, cursor
