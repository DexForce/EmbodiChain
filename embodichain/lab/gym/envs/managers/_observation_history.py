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

"""Episode-owned raw observation history, independent of scene acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import torch
from tensordict import TensorDict, TensorDictBase

from .observation_types import ObservationHistorySnapshot

if TYPE_CHECKING:
    from ..episode_context import EpisodeContext

__all__: list[str] = []


class _FieldValue(Protocol):
    values: torch.Tensor
    valid: torch.Tensor
    acquisition_errors: tuple[str | None, ...]


class _HistoryInput(Protocol):
    fields: Mapping[str, _FieldValue]
    episode_ids: tuple[int, ...]
    revision: int
    timestamp: float | None
    clock_domain: str
    episode_owner: EpisodeContext | None


@dataclass(slots=True)
class _Channel:
    name: str
    values: torch.Tensor
    valid: torch.Tensor
    errors: dict[tuple[int, int], str] = field(default_factory=dict)


class _ObservationHistory:
    """Access raw observation history in the host's shared rollout storage.

    Args:
        episode_context: Exact host lifecycle owner, shared with consumers.
        fields: Optional preallocated fields with shape ``(B, T+1, ...)``.
            The recorder takes ownership; supplied buffers must not be reused.
        actions: Owned action buffer with shape ``(B, T, ...)``. Action semantics
            remain a host contract; this view does not reinterpret them.
        storage: Shared TensorDict with an environment-only batch axis. Raw
            values, validity, clocks, errors and counters are attached here;
            the view owns no independent dense history allocation.

    Only seed/append advance history. A query copies recorded values without
    touching simulation objects. Legacy trajectory exports may project the same
    buffers; policy rollout buffers remain consumer-owned projections.
    """

    def __init__(
        self,
        *,
        episode_context: EpisodeContext,
        fields: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
        storage: TensorDictBase | None = None,
    ) -> None:
        if not isinstance(actions, torch.Tensor) or actions.ndim < 2:
            raise ValueError(
                "Recording actions require environment and transition axes."
            )
        if len(episode_context.episode_ids) != actions.shape[0]:
            raise ValueError("Recording rows must match the episode lifecycle owner.")
        self._context = episode_context
        if storage is None:
            storage = TensorDict(
                {}, batch_size=[actions.shape[0]], device=actions.device
            )
        if tuple(storage.batch_size) != (actions.shape[0],) or storage.device not in (
            None,
            actions.device,
        ):
            raise ValueError(
                "Shared rollout storage must have the environment batch axis and recording device."
            )
        if "raw_obs" in storage.keys():
            raise ValueError("Shared rollout storage already has raw observations.")
        self.storage = storage
        self._buffers = dict(fields)
        self._channels: dict[str, _Channel] = {}
        self._layout: tuple[_Channel, ...] | None = None
        self.capacity = actions.shape[1]
        self.num_envs = actions.shape[0]
        storage["raw_actions"] = actions
        storage["raw_obs"] = TensorDict(
            dict(fields),
            batch_size=[self.num_envs, self.capacity + 1],
            device=actions.device,
        )
        storage["raw_valid"] = TensorDict(
            {}, batch_size=[self.num_envs, self.capacity + 1], device=actions.device
        )
        storage["raw_steps"] = torch.zeros(
            self.num_envs, dtype=torch.long, device=actions.device
        )
        self.actions = storage["raw_actions"]
        self.steps = storage["raw_steps"]
        self._episodes = (-1,) * self.num_envs
        storage["raw_revision"] = torch.full(
            (self.num_envs, self.capacity + 1),
            -1,
            dtype=torch.long,
            device=actions.device,
        )
        storage["raw_timestamp"] = torch.full(
            (self.num_envs, self.capacity + 1),
            float("nan"),
            dtype=torch.float64,
            device=actions.device,
        )
        self._revisions = storage["raw_revision"]
        self._timestamps = storage["raw_timestamp"]
        storage.set_non_tensor("raw_acquisition_errors", {})
        self._closed = False
        self._latest_revision = -1
        for values in self._buffers.values():
            if (
                tuple(values.shape[:2]) != (self.num_envs, self.capacity + 1)
                or values.device != actions.device
            ):
                raise ValueError(
                    "State fields require T+1 frames on the recording device."
                )

    def validate_episode_source(self, episode_context: EpisodeContext) -> None:
        """Require the consumer's actual lifecycle owner, not just numeric IDs.

        Args:
            episode_context: Consumer's expected recording lifecycle owner.
        """
        if self._context is not episode_context:
            raise ValueError("Recording belongs to another episode owner.")

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Recorded observations are unavailable after close.")

    def _check_frame(self, frame: _HistoryInput) -> None:
        self._check_open()
        if frame.episode_owner is not self._context:
            raise ValueError("Observation frame belongs to another episode owner.")
        if frame.revision < 0 or frame.clock_domain not in (
            "simulation_time",
            "unavailable",
        ):
            raise ValueError(
                "Recording requires an explicit simulation observation boundary."
            )
        if frame.episode_ids != self._context.episode_ids:
            raise ValueError("Observation frame belongs to another episode.")
        if self._layout is None:
            raise RuntimeError("Observation history storage has not been bound.")
        for channel in self._layout:
            values = frame.fields[channel.name].values
            if (
                values.shape[0] != self.num_envs
                or values.device != self.actions.device
                or tuple(values.shape[1:]) != tuple(channel.values.shape[2:])
                or values.dtype != channel.values.dtype
            ):
                raise ValueError(
                    f"History values differ from storage for {channel.name!r}."
                )

    def bind(self, frame: _HistoryInput, field_ids: Sequence[str]) -> None:
        """Allocate storage for IDs already resolved by the observation schema."""
        if self._layout is not None:
            return
        if self._buffers.keys() - set(field_ids):
            raise ValueError("Observation boundary is missing required history fields.")
        for name in field_ids:
            values = frame.fields[name].values
            storage = self._buffers.get(name)
            if storage is None:
                storage = torch.zeros(
                    (self.num_envs, self.capacity + 1, *values.shape[1:]),
                    dtype=values.dtype,
                    device=self.actions.device,
                )
            if (
                values.shape[0] != self.num_envs
                or values.device != self.actions.device
                or tuple(storage.shape[2:]) != tuple(values.shape[1:])
                or storage.dtype != values.dtype
            ):
                raise ValueError(f"History values differ from storage for {name!r}.")
            self.storage["raw_obs"][name] = storage
            self.storage["raw_valid"][name] = torch.zeros(
                (self.num_envs, self.capacity + 1),
                dtype=torch.bool,
                device=self.actions.device,
            )
            errors = self.storage.get_non_tensor("raw_acquisition_errors")
            errors[name] = {}
            self._channels[name] = _Channel(
                name,
                self.storage["raw_obs"][name],
                self.storage["raw_valid"][name],
                errors[name],
            )
        self._layout = tuple(self._channels.values())
        self._buffers.clear()

    def _write(
        self, frame: _HistoryInput, ids: torch.Tensor, slots: torch.Tensor
    ) -> None:
        rows = indices = None
        for channel in self._layout:
            sample = frame.fields[channel.name]
            channel.values[ids, slots] = sample.values[ids]
            channel.valid[ids, slots] = sample.valid[ids]
            if any(error is not None for error in sample.acquisition_errors):
                if rows is None:
                    rows, indices = ids.cpu().tolist(), slots.cpu().tolist()
                for row, slot in zip(rows, indices, strict=True):
                    if sample.acquisition_errors[row] is not None:
                        channel.errors[row, slot] = sample.acquisition_errors[row]
        self._revisions[ids, slots] = frame.revision
        self._timestamps[ids, slots] = (
            frame.timestamp if frame.timestamp is not None else float("nan")
        )

    def seed(self, frame: _HistoryInput, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Start selected rows from an authoritative post-reset state frame.

        Args:
            frame: Owned raw acquisition after reset preparation is complete.
            env_ids: Original rows being initialized; other rows stay intact.
        """
        self._check_frame(frame)
        ids = self._context._ids(env_ids).to(self.actions.device)
        reset_rows = set(ids.cpu().tolist())
        if not all(
            row["initialized"] for row in self._context.snapshot(sorted(reset_rows))
        ):
            raise ValueError("Recording requires initialized episode rows.")
        episodes = list(self._episodes)
        for row in reset_rows:
            episodes[row] = frame.episode_ids[row]
        self._episodes = tuple(episodes)
        self.steps[ids] = 0
        self._revisions[ids] = -1
        self._timestamps[ids] = float("nan")
        for channel in self._layout:
            for key in list(channel.errors):
                if key[0] in reset_rows:
                    del channel.errors[key]
            channel.valid[ids] = False
        self._write(frame, ids, self.steps[ids])
        self._latest_revision = max(self._latest_revision, frame.revision)

    def append(
        self,
        frame: _HistoryInput,
        action: torch.Tensor,
        *,
        active: torch.Tensor | None = None,
    ) -> None:
        """Commit actions and their post-transition states, including final ST.

        Args:
            frame: Exact observation captured before reward hooks and autoreset.
            action: Host-validated action in the fixed recording layout.
            active: Optional row mask for sticky vectorized demonstrations.
        """
        self._check_frame(frame)
        if tuple(action.shape) != (self.num_envs, *self.actions.shape[2:]):
            raise ValueError("Action does not match the recording layout.")
        mask = self.steps < self.capacity
        if active is not None:
            if active.shape != (self.num_envs,) or active.dtype != torch.bool:
                raise ValueError("Recording active mask must contain one bool per row.")
            mask &= active.to(mask.device)
        ids = mask.nonzero(as_tuple=False).flatten()
        if not ids.numel():
            return
        if self._episodes != frame.episode_ids:
            for row in ids.cpu().tolist():
                if self._episodes[row] < 0:
                    raise ValueError(
                        "Recording rows require an initial observation frame."
                    )
                if self._episodes[row] != frame.episode_ids[row]:
                    raise ValueError("Recording buffers belong to another episode.")
        slots = self.steps[ids]
        if frame.revision <= self._latest_revision and bool(
            (self._revisions[ids, slots] >= frame.revision).any()
        ):
            raise ValueError("Each transition requires a new observation acquisition.")
        self._write(frame, ids, slots + 1)
        self.actions[ids, slots] = action[ids].to(self.actions.device)
        self.steps[ids] += 1
        self._latest_revision = max(self._latest_revision, frame.revision)

    def read(
        self,
        frame_indices: Sequence[int],
        *,
        fields: Sequence[str],
        episode_ids: Sequence[int],
        env_ids: Sequence[int] | None = None,
    ) -> ObservationHistorySnapshot:
        """Copy exact selected raw fields; missing frames never use live state.

        Args:
            frame_indices: Exact row-local state index per requested row.
            fields: Bound field names, independent of entity type.
            episode_ids: Expected source episodes in requested row order.
            env_ids: Original rows, or all rows when omitted.

        Returns:
            Owned CPU tensors, validity, reasons and original sampling identity.

        Raises:
            RuntimeError: The recording has closed.
            ValueError: Rows, episodes, identities or completed frames differ.
            KeyError: A requested raw field was not recorded.
        """
        self._check_open()
        ids = self._context._ids(env_ids).to(self.actions.device)
        rows = tuple(ids.cpu().tolist())
        frames, episodes, names = (
            tuple(frame_indices),
            tuple(episode_ids),
            tuple(fields),
        )
        if (
            not rows
            or len(frames) != len(rows)
            or len(episodes) != len(rows)
            or any(
                type(frame) is not int or frame < 0 or frame > self.capacity
                for frame in frames
            )
            or any(type(episode) is not int or episode < 0 for episode in episodes)
            or not names
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Recorded boundaries require exact fields, frames and episodes per row."
            )

        def check_episode() -> None:
            self._check_open()
            current = self._context.snapshot(rows)
            if (
                tuple(row["episode"] for row in current) != episodes
                or not all(row["initialized"] for row in current)
                or tuple(self._episodes[row] for row in rows) != episodes
            ):
                raise ValueError(
                    "Recorded observations belong to another or uninitialized episode."
                )

        check_episode()
        slots = torch.tensor(frames, dtype=torch.long, device=self.actions.device)
        if bool((slots > self.steps[ids]).any()) or bool(
            (self._revisions[ids, slots] < 0).any()
        ):
            raise ValueError("The requested boundary is not fully recorded.")
        values, validity, errors = {}, {}, {}
        for name in names:
            channel = self._channels[name]
            valid = channel.valid[ids, slots].detach().cpu()
            values[name] = channel.values[ids, slots].detach().cpu()
            validity[name] = valid
            errors[name] = tuple(
                (
                    None
                    if ok
                    else channel.errors.get(
                        (row, slot),
                        "Raw observation acquisition is unavailable or nonfinite.",
                    )
                )
                for row, slot, ok in zip(rows, frames, valid.tolist(), strict=True)
            )
        revisions = tuple(self._revisions[ids, slots].cpu().tolist())
        times = tuple(
            None if value != value else value
            for value in self._timestamps[ids, slots].cpu().tolist()
        )
        check_episode()
        return ObservationHistorySnapshot(
            values,
            validity,
            errors,
            rows,
            episodes,
            frames,
            revisions,
            times,
            (
                "simulation_time"
                if all(time is not None for time in times)
                else "unavailable"
            ),
        )

    def snapshot_history(
        self,
        env_ids: Sequence[int] | None = None,
        *,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Snapshot canonical history once for archival and export projections.

        Args:
            env_ids: Original environment rows, or all rows.
            device: Snapshot tensor device. The default provides owned CPU
                data; a host may retain its device for a compatible file export.

        Returns:
            Owned numeric fields and temporal metadata. Field semantics and
            aliases are supplied exclusively by the observation manager.
        """
        self._check_open()
        ids = self._context._ids(env_ids).to(self.actions.device)
        rows = tuple(ids.cpu().tolist())
        if not rows:
            raise ValueError("A history snapshot requires at least one row.")
        episodes = tuple(self._episodes[row] for row in rows)

        def check_episode() -> None:
            self._check_open()
            current = self._context.snapshot(rows)
            if (
                tuple(row["episode"] for row in current) != episodes
                or not all(row["initialized"] for row in current)
                or min(episodes) < 0
            ):
                raise ValueError(
                    "Recorded observations belong to another or uninitialized episode."
                )

        check_episode()
        lengths = self.steps[ids].detach().cpu()
        size = int(lengths.max()) + 1
        # Advanced indexing already owns its result; to() copies only across devices.
        revisions = self._revisions[ids, :size].detach().to(device)
        timestamps = self._timestamps[ids, :size].detach().to(device)
        written = revisions >= 0
        fields = {}
        for channel in self._layout:
            values = channel.values[ids, :size].detach().to(device)
            values.masked_fill_(
                ~written.reshape(*written.shape, *((1,) * (values.ndim - 2))),
                0,
            )
            valid = channel.valid[ids, :size].detach().to(device) & written
            valid_rows = valid.cpu().tolist()
            fields[channel.name] = {
                "values": values,
                "valid": valid,
                "acquisition_errors": [
                    {
                        slot: channel.errors.get(
                            (row, slot),
                            "Raw observation acquisition is unavailable or nonfinite.",
                        )
                        for slot in range(int(lengths[index]) + 1)
                        if not valid_rows[index][slot]
                    }
                    for index, row in enumerate(rows)
                ],
            }
        check_episode()
        return {
            "schema_version": 1,
            "env_ids": rows,
            "episode_ids": episodes,
            "lengths": lengths,
            "revisions": revisions,
            "timestamps": timestamps,
            "clock_domain": (
                "simulation_time"
                if bool(torch.isfinite(timestamps[written]).all())
                else "unavailable"
            ),
            "timestamp_valid": torch.isfinite(timestamps) & written,
            "fields": fields,
        }

    def close(self) -> None:
        """End access to this live recording without publishing an episode."""
        self._closed = True
