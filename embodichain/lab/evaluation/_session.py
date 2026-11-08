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

"""Bounded batched temporal state for compiled evaluation graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import EvaluationStatus as Status
from ._graph import (
    BooleanRule,
    CompiledEvaluation,
    MilestoneRule,
    RuleResult,
    TemporalRule,
    _BoundPredicate,
    _combine,
    _empty_result,
    _qualified_metric,
    _validate_masks,
)

if TYPE_CHECKING:
    import torch

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class EvaluationSnapshot:
    """Owned rule results with row, episode, scope and acquisition identity.

    ``step`` is the last consumed slot overall. ``steps`` records the last
    slot of each row, or -1 before that row starts.
    """

    env_ids: tuple[int, ...]
    episode_ids: tuple[int, ...]
    scope: str
    step: int | None
    revisions: torch.Tensor
    rules: Mapping[str, RuleResult]
    steps: torch.Tensor


class _History:
    def __init__(self, starts):
        import torch

        self.status = torch.full_like(starts, Status.NOT_RUN)
        self.count = torch.zeros_like(starts)
        self.missing = torch.zeros_like(starts, dtype=torch.bool)
        self.start_revision = torch.full_like(starts, -1)
        self.witness = starts.new_full((len(starts), 2), -1)
        self.witness_revisions = torch.full_like(self.witness, -1)
        self.violation = torch.full_like(self.witness, -1)
        self.violation_revisions = torch.full_like(self.witness, -1)

    def reset(self, rows):
        self.status[rows] = Status.NOT_RUN
        self.count[rows] = 0
        self.missing[rows] = False
        for value in (
            self.start_revision,
            self.witness,
            self.witness_revisions,
            self.violation,
            self.violation_revisions,
        ):
            value[rows] = -1

    def result(self, *, closed=False, owned=True, milestone=False):
        if not owned:
            return RuleResult(
                self.status,
                {},
                self.witness,
                self.witness_revisions,
                self.violation,
                self.violation_revisions,
            )
        status = self.status.clone()
        if closed:
            pending = (status == Status.NOT_RUN) | (status == Status.UNKNOWN)
            status.masked_fill_(pending, Status.UNAVAILABLE)
            if milestone:
                status.masked_fill_(
                    pending & (self.count > 0) & ~self.missing, Status.FAILED
                )
        return RuleResult(
            status,
            {},
            self.witness.clone(),
            self.witness_revisions.clone(),
            self.violation.clone(),
            self.violation_revisions.clone(),
        )

    def record(
        self, rows, *, step, revision, offset=0, first_revision=None, violation=False
    ):
        boundary = self.violation if violation else self.witness
        revisions = self.violation_revisions if violation else self.witness_revisions
        boundary[:, 0].masked_fill_(rows, step - offset)
        boundary[:, 1].masked_fill_(rows, step)
        first = revision if first_revision is None else first_revision
        revisions[:, 0].copy_(first.where(rows, revisions[:, 0]))
        revisions[:, 1].copy_(revision.where(rows, revisions[:, 1]))


class RuleSession:
    """Advance one compiled graph for an isolated occurrence and episode cohort.

    The session consumes qualified acquisition revisions in strictly increasing
    control slots. Missing slots and stale or foreign rows cannot witness rules.
    Repeated immediate evaluations and snapshots leave all history unchanged.
    """

    def __init__(
        self,
        compiled: CompiledEvaluation,
        *,
        env_ids: Sequence[int],
        episode_ids: Sequence[int],
        scope: str,
        start_step: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        import torch

        self.compiled = compiled
        self.env_ids = tuple(env_ids)
        self.episode_ids = tuple(episode_ids)
        self.scope = scope
        if (
            not self.env_ids
            or len(set(self.env_ids)) != len(self.env_ids)
            or len(self.episode_ids) != len(self.env_ids)
            or any(
                type(value) is not int or value < 0
                for value in (*self.env_ids, *self.episode_ids, start_step)
            )
            or type(scope) is not str
            or not scope
        ):
            raise ValueError(
                "Expected a scope and unique rows with initialized episodes."
            )
        self._starts = torch.full(
            (len(self.env_ids),), start_step, device=device, dtype=torch.long
        )
        self.device = self._starts.device
        self._last_step = start_step - 1
        self._advanced = False
        self._max_revisions = torch.full_like(self._starts, -1)
        self._revisions = torch.full_like(self._starts, -1)
        self._steps = torch.full_like(self._starts, -1)
        self._history = {
            name: _History(self._starts)
            for name, rule in compiled._nodes
            if isinstance(rule, (TemporalRule, MilestoneRule))
        }
        self._latest = {}
        self._observation_owner = None
        self._observation_rows = None

    def evaluate(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        valid: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, RuleResult]:
        """Evaluate an instantaneous query without advancing history."""
        return self.compiled.evaluate(
            values, batch_size=len(self.env_ids), device=self.device, valid=valid
        )

    def advance(
        self,
        step: int,
        values: Mapping[str, torch.Tensor],
        *,
        revisions: torch.Tensor,
        episode_ids: Sequence[int],
        valid: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Consume one control slot, breaking continuity across unobserved slots."""
        self._advance(step, values, revisions, episode_ids, valid)

    def _advance(self, step, values, revisions, episode_ids, valid, *, active=None):
        import torch

        source_episodes = tuple(episode_ids)
        if (
            type(step) is not int
            or step <= self._last_step
            or revisions.dtype != torch.long
            or revisions.shape != self._starts.shape
            or revisions.device != self.device
            or len(source_episodes) != len(self.env_ids)
            or any(type(value) is not int for value in source_episodes)
        ):
            raise ValueError(
                "Control slots and acquisition identities must match the session."
            )
        owned = (
            torch.ones_like(revisions, dtype=torch.bool)
            if source_episodes == self.episode_ids
            else torch.tensor(
                [
                    actual == expected
                    for actual, expected in zip(
                        source_episodes, self.episode_ids, strict=True
                    )
                ],
                dtype=torch.bool,
                device=self.device,
            )
        )
        fresh = owned & (revisions > self._max_revisions)
        if active is not None:
            fresh &= active
        accepted_revision = torch.where(fresh, revisions, -1)
        valid = valid or {}
        _validate_masks(valid.values(), len(self.env_ids), self.device)
        qualified = {
            field: fresh & valid[field] if field in valid else fresh
            for field in self.compiled.required_fields
        }

        def advance_rule(name, rule, results):
            if isinstance(rule, TemporalRule):
                self._advance_window(
                    self._history[name],
                    rule,
                    results[rule.ref].status,
                    step,
                    accepted_revision,
                    active,
                )
            else:
                self._advance_milestone(
                    self._history[name], rule, results, step, accepted_revision, active
                )
            return self._history[name].result(owned=False)

        results = self.compiled._evaluate(
            values, len(self.env_ids), self.device, qualified, advance_rule
        )
        for name, node in self.compiled._nodes:
            if not isinstance(node, _BoundPredicate):
                continue
            result = results[name]
            previous = self._latest.get(name)
            if active is not None and previous is not None:
                result.status.copy_(torch.where(active, result.status, previous.status))
                for key, before in previous.metrics.items():
                    current = result.metrics.get(key)
                    if current is None:
                        current = _qualified_metric(before, ~active)
                    else:
                        shape = (len(active),) + (1,) * (current.ndim - 1)
                        current = torch.where(active.reshape(shape), current, before)
                    result.metrics[key] = current
            self._latest[name] = result
        self._max_revisions.copy_(torch.maximum(self._max_revisions, accepted_revision))
        if active is not None:
            accepted_revision = torch.where(active, accepted_revision, self._revisions)
            self._steps.masked_fill_(active, step)
        else:
            self._steps.fill_(step)
        self._revisions.copy_(accepted_revision)
        self._last_step = step
        self._advanced = True

    def _advance_window(self, history, rule, measured, step, revisions, active):
        import torch

        first, last = self._starts + rule.start, self._starts + rule.end
        unresolved = (history.status == Status.UNKNOWN) | (
            history.status == Status.NOT_RUN
        )
        if active is not None:
            unresolved &= active
        current = unresolved & (first <= step) & (step <= last)
        skipped = (
            unresolved
            & (first <= step - 1)
            & (last > self._last_step)
            & (step > self._last_step + 1)
        )
        available = ((measured == Status.SATISFIED) | (measured == Status.FAILED)) & (
            revisions >= 0
        )
        history.missing |= skipped | (current & ~available)
        history.status.masked_fill_(current, Status.UNKNOWN)
        passed = current & (measured == Status.SATISFIED) & (revisions >= 0)
        history.count.masked_fill_(skipped | (current & ~passed), 0)
        history.start_revision.copy_(
            torch.where(
                passed & (history.count == 0), revisions, history.start_revision
            )
        )
        history.count.add_(passed.long())
        if rule.op == "always":
            failed = current & available & (measured == Status.FAILED)
            reached = passed & (last == step) & ~history.missing
            history.record(failed, step=step, revision=revisions, violation=True)
            history.status.masked_fill_(failed, Status.FAILED)
            offset = rule.end - rule.start
        else:
            reached = passed & (history.count == rule.steps)
            offset = rule.steps - 1
        history.record(
            reached,
            step=step,
            revision=revisions,
            offset=offset,
            first_revision=history.start_revision,
        )
        history.status.masked_fill_(reached, Status.SATISFIED)
        ending = unresolved & (step >= last) & (history.status != Status.SATISFIED)
        if rule.op == "always":
            ending &= history.status != Status.FAILED
        history.status.masked_fill_(ending & ~history.missing, Status.FAILED)
        history.status.masked_fill_(ending & history.missing, Status.UNAVAILABLE)

    def _advance_milestone(self, history, rule, results, step, revisions, active):
        import torch

        pending = history.status != Status.SATISFIED
        if active is not None:
            pending &= active
        eligible = torch.ones_like(pending)
        for ref in rule.after:
            predecessor = results[ref]
            eligible &= (predecessor.status == Status.SATISFIED) & (
                predecessor.witness_steps[:, 1] < step
            )
        status = _combine("all", [results[ref].status for ref in rule.refs])
        qualified = (revisions >= 0) & (
            (status == Status.SATISFIED) | (status == Status.FAILED)
        )
        reached = pending & eligible & qualified & (status == Status.SATISFIED)
        history.count.add_((pending & eligible & qualified).long())
        history.missing |= pending & (
            (eligible & ~qualified) | (step > self._last_step + 1)
        )
        history.status.masked_fill_(pending, Status.UNKNOWN)
        history.status.masked_fill_(reached, Status.SATISFIED)
        history.record(reached, step=step, revision=revisions)

    def snapshot(self, *, closed: bool = False) -> EvaluationSnapshot:
        """Read an owned result without consuming an observation."""
        results = {}
        for name, node in self.compiled._nodes:
            if name in self._history:
                results[name] = self._history[name].result(
                    closed=closed, milestone=isinstance(node, MilestoneRule)
                )
            elif isinstance(node, BooleanRule):
                results[name] = RuleResult(
                    _combine(node.op, [results[ref].status for ref in node.refs]), {}
                )
            elif name in self._latest:
                result = self._latest[name]
                results[name] = RuleResult(
                    result.status.clone(),
                    {key: value.clone() for key, value in result.metrics.items()},
                    parameters=result.parameters,
                )
            else:
                results[name] = _empty_result(
                    len(self.env_ids), self.device, Status.NOT_RUN
                )
        return EvaluationSnapshot(
            self.env_ids,
            self.episode_ids,
            self.scope,
            self._last_step if self._advanced else None,
            self._revisions.clone(),
            results,
            self._steps.clone(),
        )

    def reset(self, env_ids: Sequence[int], episode_ids: Sequence[int]) -> None:
        """Replace selected rows' episodes and restart their windows at the next slot."""
        selected, replacements = tuple(env_ids), tuple(episode_ids)
        if (
            len(selected) != len(replacements)
            or len(set(selected)) != len(selected)
            or any(type(row) is not int or row not in self.env_ids for row in selected)
            or any(type(episode) is not int or episode < 0 for episode in replacements)
        ):
            raise ValueError(
                "Reset requires unique environment IDs and one new episode ID per row."
            )
        rows = [self.env_ids.index(row) for row in selected]
        if not rows:
            return
        identities = list(self.episode_ids)
        for row, episode in zip(rows, replacements, strict=True):
            identities[row] = episode
        self.episode_ids = tuple(identities)
        self._starts[rows] = self._last_step + 1
        self._revisions[rows] = -1
        self._steps[rows] = -1
        self._max_revisions[rows] = -1
        for history in self._history.values():
            history.reset(rows)
        for result in self._latest.values():
            result.status[rows] = Status.NOT_RUN
            for metric in result.metrics.values():
                metric[rows] = 0
