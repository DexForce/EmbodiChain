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
from dataclasses import dataclass, asdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from .observations import ObservationDescriptor, ObservationFrame, ObservationSample

if TYPE_CHECKING:
    from .envs.episode_context import EpisodeContext

__all__ = ["EpisodeRecording", "RecordedObservationFrame"]


@dataclass(frozen=True, slots=True, eq=False)
class RecordedObservationFrame:
    """Owned selected fields at one exact boundary for each selected row.

    Args:
        fields: Raw values, descriptors, validity and acquisition errors.
        env_ids: Original environment row IDs in query order.
        episode_ids: Episode identities qualifying the selected buffers.
        frame_indices: Row-local state indices, including initial zero.
        revisions: Original acquisition identities per row.
        timestamps: Source simulation times, or None when unavailable.
        clock_domain: Source clock identity. Row indices are never timestamps.
    """

    fields: Mapping[str, ObservationSample]
    env_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]
    frame_indices: tuple[int, ...]
    revisions: tuple[int, ...]
    timestamps: tuple[float | None, ...]
    clock_domain: str = "simulation_time"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))


class EpisodeRecording:
    """Own raw state frames, transition actions and row-local recording cursors.

    Args:
        episode_context: Exact host lifecycle owner, shared with consumers.
        fields: Optional preallocated fields with shape ``(B, T+1, ...)``.
            The recorder takes ownership; supplied buffers must not be reused.
        actions: Owned action buffer with shape ``(B, T, ...)``. Action semantics
            remain a host contract; this component does not reinterpret them.

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
    ) -> None:
        if not isinstance(actions, torch.Tensor) or actions.ndim < 2:
            raise ValueError(
                "Recording actions require environment and transition axes."
            )
        if len(episode_context.episode_ids) != actions.shape[0]:
            raise ValueError("Recording rows must match the episode lifecycle owner.")
        self._context = episode_context
        self._fields = dict(fields)
        self.actions = actions
        self.capacity = actions.shape[1]
        self.num_envs = actions.shape[0]
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=actions.device)
        self._descriptors: dict[str, ObservationDescriptor] = {}
        self._valid: dict[str, torch.Tensor] = {}
        self._errors: dict[str, dict[tuple[int, int], str]] = {}
        self._episodes = [-1] * self.num_envs
        self._revisions = torch.full(
            (self.num_envs, self.capacity + 1),
            -1,
            dtype=torch.long,
            device=actions.device,
        )
        self._timestamps = torch.full(
            (self.num_envs, self.capacity + 1),
            float("nan"),
            dtype=torch.float64,
            device=actions.device,
        )
        self._closed = False
        self._seeded_rows: set[int] = set()
        self._latest_revision = -1
        for values in self._fields.values():
            if (
                tuple(values.shape[:2]) != (self.num_envs, self.capacity + 1)
                or values.device != actions.device
            ):
                raise ValueError(
                    "State fields require T+1 frames on the recording device."
                )

    @property
    def descriptors(self) -> Mapping[str, ObservationDescriptor]:
        """Return bound raw field identities without exposing recording buffers."""
        return MappingProxyType(self._descriptors)

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

    def _check_frame(self, frame: ObservationFrame) -> None:
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
        if set(self._fields) - set(frame.fields):
            raise ValueError(
                "Observation boundary is missing required recording fields."
            )
        for name, sample in frame.fields.items():
            if sample.values.shape[0] != self.num_envs:
                raise ValueError(
                    "Raw observation does not contain every environment row."
                )
            if sample.values.device != self.actions.device:
                raise ValueError(
                    "Raw observation and recording buffers must share a device."
                )
            if (
                name in self._descriptors
                and self._descriptors[name] != sample.descriptor
            ):
                raise ValueError(f"Raw observation identity drift for {name!r}.")
            if name in self._fields and (
                tuple(self._fields[name].shape[2:]) != sample.descriptor.shape
                or self._fields[name].dtype != sample.descriptor.dtype
            ):
                raise ValueError(f"Recording storage differs from {name!r} descriptor.")

    def _write(
        self, frame: ObservationFrame, ids: torch.Tensor, slots: torch.Tensor
    ) -> None:
        # Described aliases share one field allocation and one physical sample.
        storage_by_descriptor = {
            sample.descriptor: self._fields[name]
            for name, sample in frame.fields.items()
            if name in self._fields
        }
        written: set[ObservationDescriptor] = set()
        rows = indices = None
        for name, sample in frame.fields.items():
            descriptor = sample.descriptor
            if descriptor not in storage_by_descriptor:
                storage_by_descriptor[descriptor] = torch.zeros(
                    (self.num_envs, self.capacity + 1, *descriptor.shape),
                    dtype=descriptor.dtype,
                    device=self.actions.device,
                )
            self._fields[name] = storage_by_descriptor[descriptor]
            self._descriptors[name] = descriptor
            storage_by_descriptor.setdefault(descriptor, self._fields[name])
            if name not in self._valid:
                self._valid[name] = torch.zeros(
                    (self.num_envs, self.capacity + 1),
                    dtype=torch.bool,
                    device=self.actions.device,
                )
                self._errors[name] = {}
            if descriptor not in written:
                self._fields[name][ids, slots] = sample.values[ids].to(
                    self.actions.device
                )
                written.add(descriptor)
            self._valid[name][ids, slots] = sample.valid[ids].to(self.actions.device)
            if any(error is not None for error in sample.acquisition_errors):
                if rows is None:
                    rows, indices = (
                        ids.detach().cpu().tolist(),
                        slots.detach().cpu().tolist(),
                    )
                for row, slot in zip(rows, indices, strict=True):
                    error = sample.acquisition_errors[row]
                    if error is not None:
                        self._errors[name][row, slot] = error
        self._revisions[ids, slots] = frame.revision
        self._timestamps[ids, slots] = (
            frame.timestamp if frame.timestamp is not None else float("nan")
        )

    def seed(
        self, frame: ObservationFrame, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
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
        for row in reset_rows:
            self._episodes[row] = frame.episode_ids[row]
        self.steps[ids] = 0
        self._revisions[ids] = -1
        self._timestamps[ids] = float("nan")
        for errors in self._errors.values():
            for key in list(errors):
                if key[0] in reset_rows:
                    del errors[key]
        for valid in self._valid.values():
            valid[ids] = False
        self._write(frame, ids, self.steps[ids])
        self._seeded_rows.update(reset_rows)
        self._latest_revision = max(self._latest_revision, frame.revision)

    def append(
        self,
        frame: ObservationFrame,
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
        if (
            len(self._seeded_rows) != self.num_envs
            or tuple(self._episodes) != frame.episode_ids
        ):
            for row in ids.cpu().tolist():
                if row not in self._seeded_rows:
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
    ) -> RecordedObservationFrame:
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
        samples = {}
        for name in names:
            descriptor = self._descriptors[name]
            valid = self._valid[name][ids, slots].detach().cpu().clone()
            errors = tuple(
                (
                    None
                    if ok
                    else self._errors[name].get(
                        (row, slot),
                        "Raw observation acquisition is unavailable or nonfinite.",
                    )
                )
                for row, slot, ok in zip(rows, frames, valid.tolist(), strict=True)
            )
            samples[name] = ObservationSample(
                descriptor,
                self._fields[name][ids, slots].detach().cpu().clone(),
                valid,
                errors,
            )
        revisions = tuple(self._revisions[ids, slots].cpu().tolist())
        times = tuple(
            None if value != value else value
            for value in self._timestamps[ids, slots].cpu().tolist()
        )
        check_episode()
        return RecordedObservationFrame(
            samples,
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

    def snapshot_history(self, env_ids: Sequence[int] | None = None) -> dict[str, Any]:
        """Copy raw history for archival, retaining validity and T+1 boundaries.

        Args:
            env_ids: Original environment rows to export, or all rows.

        Returns:
            Versioned owned CPU tensors and serializable field descriptors.
            Rows may have different lengths; validity excludes unwritten slots.
            Aliases retain shared value storage in the serialized artifact.
        """
        self._check_open()
        ids = self._context._ids(env_ids).to(self.actions.device)
        rows = tuple(ids.cpu().tolist())
        if not rows:
            raise ValueError("A history snapshot requires at least one row.")
        episodes = tuple(self._episodes[row] for row in rows)
        current = self._context.snapshot(rows)
        if (
            tuple(row["episode"] for row in current) != episodes
            or not all(row["initialized"] for row in current)
            or any(row not in self._seeded_rows for row in rows)
        ):
            raise ValueError(
                "Recorded observations belong to another or uninitialized episode."
            )
        lengths = self.steps[ids].detach().cpu().clone()
        size = int(lengths.max()) + 1
        revisions = self._revisions[ids, :size].detach().cpu().clone()
        timestamps = self._timestamps[ids, :size].detach().cpu().clone()
        written = revisions >= 0
        fields, copied = {}, {}
        for name, descriptor in self._descriptors.items():
            if descriptor not in copied:
                values = self._fields[name][ids, :size].detach().cpu().clone()
                values.masked_fill_(
                    ~written.reshape(*written.shape, *((1,) * len(descriptor.shape))), 0
                )
                copied[descriptor] = values
            metadata = asdict(descriptor)
            metadata["dtype"] = str(descriptor.dtype)
            valid = self._valid[name][ids, :size].detach().cpu().clone() & written
            fields[name] = {
                "descriptor": metadata,
                "values": copied[descriptor],
                "valid": valid,
                "acquisition_errors": [
                    {
                        slot: self._errors[name].get(
                            (row, slot),
                            "Raw observation acquisition is unavailable or nonfinite.",
                        )
                        for slot in range(int(lengths[index]) + 1)
                        if not valid[index, slot]
                    }
                    for index, row in enumerate(rows)
                ],
            }
        self._check_open()
        if tuple(row["episode"] for row in self._context.snapshot(rows)) != episodes:
            raise ValueError("Recorded observations belong to another episode.")
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
