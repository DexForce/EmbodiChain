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

"""Merge successful CPU phase metrics into a scheduling hint cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["merge_history", "main"]


def merge_history(metrics: Path, history: Path) -> None:
    """Update observed CPU cases while retaining hints for unselected cases.

    Args:
        metrics: Runner metrics directory containing fast and slow-fast lanes.
        history: Destination JSON file, also accepted as duration-history input.
    """
    entries = {}
    if history.exists():
        entries = {
            row["nodeid"]: row for row in json.loads(history.read_text())["phases"]
        }
    observed = {}
    for lane in ("fast", "slow-fast"):
        for path in sorted((metrics / lane).glob("*.json")):
            report = json.loads(path.read_text())
            if report.get("exitstatus") != 0:
                continue
            for phase in report["phases"]:
                group = phase["group"]
                node = phase["nodeid"]
                suffix = "@" + group
                if node.endswith(suffix):
                    node = node[: -len(suffix)]
                row = observed.setdefault(
                    node, {"nodeid": node, "group": group, "seconds": 0.0}
                )
                row["seconds"] += phase["seconds"]
    if not observed:
        return
    entries.update(observed)
    history.parent.mkdir(parents=True, exist_ok=True)
    temporary = history.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"version": 1, "phases": list(entries.values())}, indent=2) + "\n"
    )
    temporary.replace(history)


def main() -> None:
    """Merge an explicitly selected metrics directory after a successful CI run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    args = parser.parse_args()
    merge_history(args.metrics, args.history)


if __name__ == "__main__":
    main()
