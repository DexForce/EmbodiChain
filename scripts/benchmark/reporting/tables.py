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

"""Dependency-free Markdown table rendering for technical reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

__all__ = ["render_markdown_table"]


def _display(value: object) -> str:
    """Format one cell without introducing invalid table separators."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.6f}"
    return str(value).replace("|", r"\|").replace("\n", " ")


def render_markdown_table(
    rows: Sequence[Mapping[str, object]], *, columns: Sequence[str] | None = None
) -> str:
    """Render stable headers and rows, including an empty-table schema."""
    if columns is None:
        column_names = tuple(sorted({key for row in rows for key in row}))
    else:
        column_names = tuple(columns)
    if not column_names:
        raise ValueError("a table requires at least one column")
    lines = [
        "| " + " | ".join(column_names) + " |",
        "| " + " | ".join("---" for _ in column_names) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_display(row.get(column)) for column in column_names)
            + " |"
        )
    return "\n".join(lines)
