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

"""Shared applicability rules for benchmark success denominators."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import TrialPhase, TrialRecord

__all__: list[str] = []


def _unsupported_case_ids(records: Iterable[TrialRecord]) -> frozenset[str]:
    """Return capability-gated cases excluded from success denominators."""
    by_case_source: dict[str, dict[str, list[TrialRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_case_source[record.case_id][record.algorithm_id].append(record)
    return frozenset(
        case_id
        for case_id, sources in by_case_source.items()
        if all(_source_unsupported(rows) for rows in sources.values())
    )


def _source_unsupported(records: list[TrialRecord]) -> bool:
    """Return whether a source was explicitly gated without measured evidence."""
    return not any(record.phase is TrialPhase.MEASURED for record in records) and any(
        record.phase is TrialPhase.AVAILABILITY
        and record.status == "unsupported"
        and record.failure_code in {"unsupported_capability", "unsupported_capacity"}
        for record in records
    )
