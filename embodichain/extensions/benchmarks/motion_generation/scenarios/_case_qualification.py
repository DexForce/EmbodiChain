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

"""Bounded, planner-independent qualification of complete input batches."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from typing import TYPE_CHECKING

from ..config import stable_hash

if TYPE_CHECKING:
    from ..models import BenchmarkCase

__all__: list[str] = []


class _CaseQualificationError(RuntimeError):
    """No independently qualified input was found; not an infeasibility proof."""

    def __init__(
        self, message: str, *, evidence: dict[str, object] | None = None
    ) -> None:
        super().__init__(message)
        self.qualification_diagnostics = deepcopy(evidence or {})


def _case_proposal_seed(seed: int, identity: str, attempt: int) -> int:
    """Keep the original first proposal and isolate later preparation streams."""
    if attempt == 0:
        return seed
    # High-range seeds cannot alias the small declared development/test banks.
    return (1 << 60) + int(
        stable_hash(("whole_batch_input_qualification_v1", identity, seed, attempt))[
            :15
        ],
        16,
    )


def _generate_qualified_case(
    factory: Callable[[int], BenchmarkCase],
    *,
    seed: int,
    identity: str,
    max_attempts: int,
    reset: Callable[[], None],
) -> BenchmarkCase:
    """Condition whole batches on input evidence before any planner is measured.

    Only typed qualification failures permit another proposal. Every retry
    regenerates one complete batch, preserving the provider's row strata and
    shared task geometry. The logical case/seed identities remain unchanged;
    proposal seeds, rejected evidence and the selection rule are retained.
    Ordinary runtime/backend errors always propagate without retry.
    """
    if type(max_attempts) is not int or not 1 <= max_attempts <= 128:
        raise ValueError("case_generation_attempts must be an integer in [1,128].")
    rejected: list[dict[str, object]] = []
    for attempt in range(max_attempts):
        proposal_seed = _case_proposal_seed(seed, identity, attempt)
        if attempt:
            reset()
        try:
            case = factory(proposal_seed)
        except _CaseQualificationError as error:
            if max_attempts == 1:
                raise
            rejected.append(
                {
                    "attempt": attempt,
                    "proposal_seed": proposal_seed,
                    "reason": str(error),
                    "evidence": error.qualification_diagnostics,
                }
            )
            if attempt + 1 == max_attempts:
                raise _CaseQualificationError(
                    f"Independent input qualification exhausted {max_attempts} "
                    f"whole-batch proposals for {identity!r}; no case was published. "
                    "This does not establish unreachability.",
                    evidence={
                        "case_id": identity,
                        "source_seed": seed,
                        "rejected": rejected,
                    },
                ) from error
            continue
        if max_attempts == 1:
            return case
        if case.seed != proposal_seed:
            raise ValueError("Atomic provider changed its input proposal seed.")
        return replace(
            case,
            case_id=identity,
            seed=seed,
            case_parameters={
                **case.case_parameters,
                "input_qualification": {
                    "rule": "first_qualified_whole_batch_v1",
                    "source_seed": seed,
                    "proposal_seed": proposal_seed,
                    "selected_attempt": attempt,
                    "max_attempts": max_attempts,
                    "rejected": rejected,
                    "scope": "endpoint_reference_only_not_path_or_physical_success",
                },
            },
        )
    raise AssertionError("Positive bounded input qualification loop did not terminate")
