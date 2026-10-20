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

"""Row-local recording eligibility and owned component provenance."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import threading
from typing import Any

import torch

from ._json import json_safe_copy

__all__ = ["EpisodeContext"]


class EpisodeContext:
    """Keep recording eligibility attached to buffers rather than annotations.

    Args:
        num_envs: Number of independently reset environment rows.
        device: Device used by row masks and returned environment IDs.

    A successful reset must initialize a row before it can be recorded. Checks
    block publication until their synchronous scope closes. Rejection is sticky
    until the host consumes the old buffers and starts another episode. Context
    updates and snapshots belong to the creating thread; background writers
    consume owned snapshots instead of accessing this object.
    """

    def __init__(self, num_envs: int, device: str | torch.device = "cpu") -> None:
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        self._count = num_envs
        self._device = torch.device(device)
        self._thread = threading.get_ident()
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._rejected = torch.zeros_like(self._initialized)
        self._pending = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._episodes = [-1] * num_envs
        self._sources: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._reasons: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        self._active: dict[str, set[int]] = {}

    def _assert_thread(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("EpisodeContext belongs to the environment thread.")

    def _ids(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None
    ) -> torch.Tensor:
        self._assert_thread()
        if env_ids is None:
            return torch.arange(self._count, device=self._device)
        if isinstance(env_ids, slice):
            env_ids = list(range(self._count))[env_ids]
        if isinstance(env_ids, torch.Tensor):
            if env_ids.ndim != 1 or env_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError(
                    "Environment IDs must be a one-dimensional integer selection."
                )
            ids = env_ids.to(device=self._device, dtype=torch.long).clone()
        else:
            values = list(env_ids)
            if any(type(value) is not int for value in values):
                raise ValueError("Environment IDs must be integers.")
            ids = torch.tensor(values, device=self._device, dtype=torch.long)
        if len(torch.unique(ids)) != len(ids) or bool(
            ((ids < 0) | (ids >= self._count)).any()
        ):
            raise ValueError("Environment IDs must be unique valid rows.")
        return ids

    def filter_recordable(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return owned eligible row IDs in the caller's requested order.

        Args:
            env_ids: Requested rows, or all rows when omitted.

        Returns:
            Long tensor on the context device. Pending, uninitialized and
            previously rejected buffers are excluded independently of task success.
        """
        ids = self._ids(env_ids)
        return ids[
            self._initialized[ids] & ~self._rejected[ids] & (self._pending[ids] == 0)
        ]

    def snapshot(
        self, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> list[dict[str, Any]]:
        """Copy JSON provenance and eligibility without advancing any state.

        Args:
            env_ids: Selected rows, or all rows when omitted.

        Returns:
            Owned row dictionaries suitable for info, episode sidecars and
            asynchronous persistence. Their eligibility does not prove task success.
        """
        ids = self._ids(env_ids).cpu().tolist()
        recordable = set(self.filter_recordable(ids).cpu().tolist())
        initialized = self._initialized.cpu().tolist()
        return [
            {
                "schema_version": 1,
                "env_id": row,
                "episode": self._episodes[row],
                "initialized": initialized[row],
                "recordable": row in recordable,
                "pending_sources": sorted(
                    name for name, rows in self._active.items() if row in rows
                ),
                "rejections": deepcopy(self._reasons[row]),
                "sources": deepcopy(self._sources[row]),
            }
            for row in ids
        ]

    @contextmanager
    def check(
        self, source: str, env_ids: Sequence[int] | torch.Tensor | None = None
    ) -> Iterator[Callable[..., None]]:
        """Block selected buffers while a component performs a synchronous check.

        Args:
            source: Stable component name, e.g. ``events/scene_layout``.
            env_ids: Selected rows, or all rows when omitted.

        Yields:
            A callable that must be invoked exactly once as
            ``finish(accepted=bool_or_row_mask, metadata=..., reasons=...)``.
            Metadata may be one shared JSON mapping or a mapping from selected
            integer row IDs to JSON mappings. Reasons may be one string or a
            row-keyed mapping. Inputs are validated and copied before publication.

        Raises:
            RuntimeError: For overlapping scopes of the same source, a missing
                report, or a report attempted after closing the scope.

        Scopes survive internal resets. Exceptions, including cancellation,
        reject all selected current buffers even after a successful report.
        A later check cannot clear an earlier rejection in those buffers.
        """
        if type(source) is not str or not source or source != source.strip():
            raise ValueError("Check source must be a nonempty stripped string.")
        ids = self._ids(env_ids)
        rows = ids.cpu().tolist()
        if self._active.get(source, set()).intersection(rows):
            raise RuntimeError(f"Check {source!r} already covers a selected row.")
        self._active.setdefault(source, set()).update(rows)
        self._pending[ids] += 1
        report = None
        opened = True

        def finish(
            *,
            accepted: bool | Sequence[bool] | torch.Tensor,
            metadata: Mapping | None = None,
            reasons: str | Mapping[int, str] | None = None,
        ) -> None:
            nonlocal report
            self._assert_thread()
            if not opened or report is not None:
                raise RuntimeError(
                    "A check report is allowed exactly once inside its scope."
                )
            if type(accepted) is bool:
                mask = [accepted] * len(rows)
            elif isinstance(accepted, torch.Tensor):
                if accepted.dtype != torch.bool or accepted.shape != (len(rows),):
                    raise ValueError(
                        "Acceptance must be a bool mask matching selected rows."
                    )
                mask = accepted.detach().cpu().tolist()
            else:
                mask = list(accepted)
                if len(mask) != len(rows) or any(
                    type(value) is not bool for value in mask
                ):
                    raise ValueError(
                        "Acceptance must be a bool mask matching selected rows."
                    )
            if metadata is not None and not isinstance(metadata, Mapping):
                raise TypeError("Check metadata must be a mapping.")
            per_row = bool(metadata) and all(type(key) is int for key in metadata)
            if per_row and set(metadata) - set(rows):
                raise ValueError("Check metadata references unselected rows.")
            if reasons is not None and not isinstance(reasons, (str, Mapping)):
                raise TypeError("Check reasons must be a string or row mapping.")
            if isinstance(reasons, Mapping) and (
                any(type(key) is not int for key in reasons) or set(reasons) - set(rows)
            ):
                raise ValueError("Check reasons reference unselected rows.")
            values, errors = [], []
            for row in rows:
                value = metadata.get(row, {}) if per_row else (metadata or {})
                if not isinstance(value, Mapping):
                    raise TypeError("Each row's check metadata must be a mapping.")
                values.append(json_safe_copy(value, field_name="check metadata"))
                reason = reasons.get(row) if isinstance(reasons, Mapping) else reasons
                if reason is not None and (
                    type(reason) is not str or not reason.strip()
                ):
                    raise ValueError("Check reasons must be nonempty strings.")
                errors.append(reason)
            report = (mask, values, errors)

        def publish(
            mask: list[bool], values: list[dict], errors: list[str | None]
        ) -> None:
            for row, accepted, metadata, reason in zip(
                rows, mask, values, errors, strict=True
            ):
                entry = {"accepted": accepted, "metadata": metadata}
                self._sources[row][source] = deepcopy(entry)
                if not accepted:
                    self._rejected[row] = True
                    self._reasons[row].setdefault(
                        source,
                        {
                            "reason": reason or "rejected",
                            "metadata": deepcopy(metadata),
                        },
                    )

        try:
            yield finish
            if report is None:
                raise RuntimeError(f"Check {source!r} closed without a report.")
            publish(*report)
        except BaseException as error:
            values = report[1] if report is not None else [{} for _ in rows]
            publish(
                [False] * len(rows),
                values,
                [type(error).__name__ + ": " + str(error)] * len(rows),
            )
            raise
        finally:
            opened = False
            self._pending[ids] -= 1
            self._active[source].difference_update(rows)
            if not self._active[source]:
                del self._active[source]

    def _begin_episode(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = self._ids(env_ids)
        self._initialized[ids] = False
        self._rejected[ids] = False
        for row in ids.cpu().tolist():
            self._episodes[row] += 1
            self._sources[row] = {}
            self._reasons[row] = {}

    def _finish_initialization(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        self._initialized[self._ids(env_ids)] = True

    def _abort_initialization(
        self, env_ids: Sequence[int] | torch.Tensor, error: BaseException
    ) -> None:
        ids = self._ids(env_ids)
        self._initialized[ids] = False
        self._rejected[ids] = True
        for row in ids.cpu().tolist():
            self._reasons[row].setdefault(
                "reset", {"reason": type(error).__name__ + ": " + str(error)}
            )
