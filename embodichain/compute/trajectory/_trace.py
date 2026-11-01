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

"""Opt-in, call-scoped trajectory snapshots for host-owned diagnostics.

No I/O or simulator dependency belongs here. Device tensors are cloned only
when a host opens a capture scope; export and case identity stay with the host.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

import torch

__all__ = []

_CURRENT: ContextVar[list[dict[str, object]] | None] = ContextVar(
    "trajectory_capture", default=None
)
_IDENTITY: ContextVar[dict[str, object]] = ContextVar("trajectory_identity", default={})


def _snapshot(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported trajectory trace value: {type(value).__name__}")


@contextmanager
def _capture_trajectories(enabled: bool) -> Iterator[list[dict[str, object]]]:
    """Own snapshots until the enclosing planning call completes."""
    records: list[dict[str, object]] = []
    token = _CURRENT.set(records if enabled else None)
    try:
        yield records
    finally:
        _CURRENT.reset(token)


def _record_trajectory(stage: str, **fields: object) -> None:
    """Snapshot one stage without transfers; disabled capture retains nothing."""
    records = _CURRENT.get()
    if records is not None:
        records.append(
            {
                "stage": stage,
                **_IDENTITY.get(),
                **{key: _snapshot(value) for key, value in fields.items()},
            }
        )


@contextmanager
def _trajectory_scope(**identity: object) -> Iterator[None]:
    """Attach action/arm identity to nested stages while capture is enabled."""
    records = _CURRENT.get()
    if records is None:
        yield
        return
    token = _IDENTITY.set({**_IDENTITY.get(), "scope_id": len(records), **identity})
    _record_trajectory("scope_begin")
    try:
        yield
    finally:
        _record_trajectory("scope_end")
        _IDENTITY.reset(token)


def _tracing_enabled() -> bool:
    """Avoid preparing diagnostic-only values when capture is disabled."""
    return _CURRENT.get() is not None
