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

"""Offline distribution reports from confirmed native collection artifacts."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

__all__ = ["report_expansion_distribution"]


def report_expansion_distribution(
    before: Sequence[str | Path],
    after: Sequence[str | Path],
    output: str | Path,
    *,
    entity: str,
    cell_size_m: float = 0.01,
    minimum: int = 1,
) -> dict:
    """Compare matching-domain receipts without creating a simulation World.

    Args:
        before: Native jobs, expansion aggregates or workflow run directories.
        after: Independent or cumulative matching-domain collection directories.
        output: New destination for the complete HTML/JSON/JSONL report bundle.
        entity: Recorded rigid entity whose measured states are compared.
        cell_size_m: World-XY bin width, selected before examining results.
        minimum: Required episode count for observed-bin deficit diagnostics.

    Returns:
        Report directory, HTML entry, snapshot identity and confirmed counts.

    Raises:
        ValueError: For invalid inputs, mismatched domains or corrupt evidence.
        FileExistsError: If the destination already exists.
    """
    from ._distribution import _run
    from .preview import _inputs

    def jobs(paths):
        return _inputs(
            [
                Path(p) / "expand" if (Path(p) / "run.json").is_file() else Path(p)
                for p in paths
            ]
        )

    output = Path(output).expanduser().resolve()
    _run(jobs(before), jobs(after), output, entity, cell_size_m, minimum)
    report = json.loads((output / "summary.json").read_text())
    return {
        "directory": str(output),
        "html": str(output / "report.html"),
        "snapshot_id": report["snapshot_id"],
        "comparison_kind": report["comparison_kind"],
        "before": report["before"]["count"],
        "after": report["after"]["count"],
    }
