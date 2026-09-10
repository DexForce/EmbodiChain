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

"""Owned candidate values for physical-row atomic planning."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .invocation import ResolvedActionRequest
    from .state import PlanningContext


def _active_rows(mask: torch.Tensor | None, env_ids: torch.Tensor) -> torch.Tensor:
    """Validate a physical-row eligibility mask without broadcasting."""
    if mask is None:
        return torch.ones_like(env_ids, dtype=torch.bool)
    if (
        not isinstance(mask, torch.Tensor)
        or mask.dtype != torch.bool
        or mask.shape != env_ids.shape
        or mask.device != env_ids.device
    ):
        raise ValueError("active_mask must be bool with the env_ids shape/device.")
    return mask.detach().clone()


def _input_fingerprint(request: ResolvedActionRequest, context: PlanningContext) -> str:
    """Bind an evaluation to the current invocation, state, and scene snapshot."""
    return _value_fingerprint(request, context)


def _value_fingerprint(*values: object) -> str:
    """Fingerprint value objects and tensors without shortened tensor reprs."""
    digest = hashlib.sha256()

    def visit(value: object) -> None:
        digest.update(type(value).__qualname__.encode())
        if isinstance(value, torch.Tensor):
            cpu = value.detach().cpu().contiguous()
            digest.update(str((cpu.dtype, tuple(cpu.shape))).encode())
            # Singleton views can be contiguous while retaining a non-unit
            # stride (e.g. a one-joint command sliced from a multi-joint row).
            # The dtype-changing byte view needs an actual unit-stride copy.
            packed = cpu.reshape(-1).clone(memory_format=torch.contiguous_format)
            digest.update(packed.view(torch.uint8).numpy().tobytes())
        elif is_dataclass(value) and not isinstance(value, type):
            for item in fields(value):
                if item.init:
                    digest.update(item.name.encode())
                    visit(getattr(value, item.name))
        elif isinstance(value, Mapping):
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
        digest.update(b"\0")

    for value in values:
        visit(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class AtomicCandidateBatch:
    """Skill candidates evaluated at one physical-row planning context.

    Candidate columns are logical choices, not additional simulator rows.
    Concrete skills extend this value with their own poses and solved anchors.
    """

    skill_id: str
    env_ids: torch.Tensor
    valid_mask: torch.Tensor
    costs: torch.Tensor
    candidate_ids: tuple[tuple[str, ...], ...]
    failure_stages: tuple[tuple[str | None, ...], ...]
    input_fingerprint: str
    failure_reasons: tuple[tuple[str | None, ...], ...] = ()

    def __post_init__(self) -> None:
        if self.valid_mask.dtype != torch.bool or self.valid_mask.ndim != 2:
            raise ValueError("valid_mask must be bool with shape (E, K).")
        rows, columns = self.valid_mask.shape
        if self.env_ids.dtype != torch.long or self.env_ids.shape != (rows,):
            raise ValueError("env_ids must be int64 with shape (E,).")
        if torch.unique(self.env_ids).numel() != rows:
            raise ValueError("Candidate physical env_ids must be unique.")
        if self.costs.shape != (rows, columns):
            raise ValueError("costs must have shape (E, K).")
        for value in (self.env_ids, self.costs):
            if value.device != self.valid_mask.device:
                raise ValueError("Candidate tensors must share a device.")
        if not torch.isfinite(self.costs[self.valid_mask]).all():
            raise ValueError("Valid candidates must have finite costs.")
        for name in ("candidate_ids", "failure_stages"):
            values = getattr(self, name)
            if len(values) != rows or any(len(row) != columns for row in values):
                raise ValueError(f"{name} must have E rows and K columns.")
            object.__setattr__(self, name, tuple(tuple(row) for row in values))
        if not self.failure_reasons:
            object.__setattr__(
                self,
                "failure_reasons",
                tuple(
                    tuple(
                        None if value is None else value.rsplit(":", 1)[-1]
                        for value in row
                    )
                    for row in self.failure_stages
                ),
            )
            object.__setattr__(
                self,
                "failure_stages",
                tuple(
                    tuple(
                        (
                            None
                            if value is None or ":" not in value
                            else value.rsplit(":", 1)[0]
                        )
                        for value in row
                    )
                    for row in self.failure_stages
                ),
            )
        if len(self.failure_reasons) != rows or any(
            len(row) != columns for row in self.failure_reasons
        ):
            raise ValueError("failure_reasons must have E rows and K columns.")
        object.__setattr__(
            self, "failure_reasons", tuple(tuple(row) for row in self.failure_reasons)
        )
        for name in ("env_ids", "valid_mask", "costs"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())

    def select(
        self,
        indices: torch.Tensor,
        *,
        active_mask: torch.Tensor | None = None,
    ) -> AtomicCandidateSelection:
        """Select one column per physical row, using -1 for idle rows.

        Args:
            indices: Int64 column indices with shape ``(E,)``.
            active_mask: Optional eligible physical rows.

        Returns:
            Owned selection; failed candidates remain failed rather than falling
            back to another grasp.
        """
        return AtomicCandidateSelection(
            candidates=self, indices=indices, active_mask=active_mask
        )


@dataclass(frozen=True, slots=True, eq=False)
class AtomicCandidateSelection:
    """One candidate choice per physical row, with explicit idle slots."""

    candidates: AtomicCandidateBatch
    indices: torch.Tensor
    active_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, AtomicCandidateBatch):
            raise TypeError("candidates must be an AtomicCandidateBatch.")
        env_ids = self.candidates.env_ids
        if (
            self.indices.dtype != torch.long
            or self.indices.shape != env_ids.shape
            or self.indices.device != env_ids.device
        ):
            raise ValueError("indices must be int64 with the env_ids shape/device.")
        capacity = self.candidates.valid_mask.shape[1]
        if ((self.indices < -1) | (self.indices >= capacity)).any():
            raise ValueError("Selected candidate index is outside [-1, K).")
        active = _active_rows(self.active_mask, env_ids) & (self.indices >= 0)
        object.__setattr__(self, "indices", self.indices.detach().clone())
        object.__setattr__(self, "active_mask", active)
        # Concrete candidate __post_init__ methods clone their tensor payloads.
        object.__setattr__(self, "candidates", replace(self.candidates))


__all__ = ["AtomicCandidateBatch", "AtomicCandidateSelection"]
