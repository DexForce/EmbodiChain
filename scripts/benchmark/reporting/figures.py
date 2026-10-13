# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Small deterministic SVG/CSV figure outputs without plotting dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import html
import math
from pathlib import Path

__all__ = ["FigureStyle", "write_metric_csv", "write_svg_bar_chart"]


@dataclass(frozen=True)
class FigureStyle:
    """Shared visual defaults for report figures."""

    primary: str = "#2f6f9f"
    background: str = "white"
    foreground: str = "#333"
    font_family: str = "sans-serif"


def write_metric_csv(
    path: Path, rows: Sequence[Mapping[str, object]], *, columns: Sequence[str]
) -> Path:
    """Write plotting data beside a report so figures remain traceable."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            {column: row.get(column) for column in columns} for row in rows
        )
    return output


def write_svg_bar_chart(
    path: Path,
    *,
    title: str,
    values: Mapping[str, int | float],
    width: int = 640,
    height: int = 360,
    style: FigureStyle | None = None,
) -> Path:
    """Write a compact labelled bar chart with a stable default style."""
    if width < 200 or height < 160:
        raise ValueError("figure dimensions are too small")
    style = style or FigureStyle()
    finite = {
        str(label): float(value)
        for label, value in values.items()
        if type(value) in (int, float) and math.isfinite(float(value)) and value >= 0
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    margin_left, margin_bottom, chart_height = 70, 55, height - 95
    chart_width = width - margin_left - 25
    maximum = max(finite.values(), default=1.0) or 1.0
    bar_width = chart_width / max(1, len(finite)) * 0.65
    gap = chart_width / max(1, len(finite))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{html.escape(title)}</title>",
        f'<rect width="100%" height="100%" fill="{html.escape(style.background)}"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="{html.escape(style.font_family)}" font-size="16">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 25}" y2="{height - margin_bottom}" stroke="{html.escape(style.foreground)}"/>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{margin_left}" y2="{height - margin_bottom - chart_height}" stroke="{html.escape(style.foreground)}"/>',
    ]
    for index, (label, value) in enumerate(finite.items()):
        bar_height = value / maximum * chart_height
        x = margin_left + index * gap + (gap - bar_width) / 2
        y = height - margin_bottom - bar_height
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{html.escape(style.primary)}"/>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{height - margin_bottom + 20}" text-anchor="middle" font-family="{html.escape(style.font_family)}" font-size="11">{html.escape(label)}</text>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{max(42, y - 5):.2f}" text-anchor="middle" font-family="{html.escape(style.font_family)}" font-size="10">{value:.4g}</text>'
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
