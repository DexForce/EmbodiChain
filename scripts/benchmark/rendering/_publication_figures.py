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

"""Small offline vector charts for the rendering publication report."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
import textwrap
from typing import Any

__all__ = ["write_chart"]

_COLORS = (
    "#2166ac",
    "#b35806",
    "#1b7837",
    "#762a83",
    "#c51b7d",
    "#636363",
    "#4393c3",
    "#d6604d",
    "#5aae61",
    "#9970ab",
    "#e08214",
    "#018571",
    "#053061",
    "#67001f",
    "#00441b",
    "#40004b",
)


def _pdf_text(value: str) -> str:
    """Escape a literal string for the built-in PDF font."""
    return (
        value.encode("ascii", "replace")
        .decode("ascii")
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("\n", " ")
    )


def _write_pdf(path: Path, content: str, metadata: str, title: str) -> None:
    """Write one uncompressed vector page with inspectable provenance."""
    stream = content.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 980 600] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        (
            f"<< /Title ({_pdf_text(title)}) /Subject ({_pdf_text(metadata)}) "
            "/Creator (EmbodiChain offline publication report) >>"
        ).encode("ascii"),
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, value in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode() + value + b"\nendobj\n")
    offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for position in offsets[1:]:
        output.extend(f"{position:010d} 00000 n \n".encode())
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
            f"startxref\n{offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(output)


def write_chart(
    path: Path,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict[str, Any]],
    provenance: dict[str, Any],
    connect: bool = True,
    log_x: bool = False,
    x_ticks: dict[float, str] | None = None,
    y_ticks: dict[float, str] | None = None,
) -> tuple[Path, Path]:
    """Write matching SVG and PDF curves, including source IDs and run MAD.

    Args:
        path: Output stem; ``.svg`` and ``.pdf`` are appended.
        title: Scope-specific title, without a cross-workload ranking.
        x_label: Horizontal axis and unit.
        y_label: Vertical axis and unit.
        series: Labelled point lists with x/y and optional low/high bounds.
        provenance: Frozen input identities and the underlying plot records.
        connect: Connect points within each series when true.
        log_x: Use base-two spacing for environment-count curves.
        x_ticks: Optional categorical horizontal tick labels.
        y_ticks: Optional categorical vertical tick labels.

    Returns:
        The SVG and PDF paths.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(provenance, sort_keys=True, ensure_ascii=True)
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="980" height="600" viewBox="0 0 980 600">',
        f"<title>{html.escape(title)}</title>",
        f"<metadata>{html.escape(metadata)}</metadata>",
        '<rect width="980" height="600" fill="white"/>',
    ]
    pdf: list[str] = []

    def color(value: str) -> str:
        return " ".join(
            f"{int(value[index:index + 2], 16) / 255:.4f}" for index in (1, 3, 5)
        )

    def line(
        x1: float, y1: float, x2: float, y2: float, ink: str, width: float = 1
    ) -> None:
        svg.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{ink}" stroke-width="{width}"/>'
        )
        pdf.append(
            f"{color(ink)} RG {width} w {x1:.2f} {600-y1:.2f} m "
            f"{x2:.2f} {600-y2:.2f} l S"
        )

    def text(x: float, y: float, value: str, size: int = 11) -> None:
        svg.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="sans-serif" '
            f'font-size="{size}" fill="#222222">{html.escape(value)}</text>'
        )
        pdf.append(
            f"0.13 0.13 0.13 rg BT /F1 {size} Tf {x:.2f} {600-y:.2f} Td "
            f"({_pdf_text(value)}) Tj ET"
        )

    text(35, 30, title, 17)
    text(35, 52, y_label, 12)
    scope = provenance.get("scope", {})
    if scope:
        context = " | ".join(
            f"{key}={scope.get(key, 'missing')}"
            for key in ("scene_id", "profile", "delivery", "hardware_id")
            if key in scope
        )
        text(
            35,
            70,
            textwrap.shorten(
                context, width=170, placeholder="... (full scope in metadata)"
            ),
            9,
        )
    left, right, top, bottom = 90.0, 700.0, 90.0, 495.0
    points = [point for group in series for point in group["points"]]

    def transform(value: float) -> float:
        return math.log2(value) if log_x else value

    if points:
        xs = [transform(float(point["x"])) for point in points]
        xs += [transform(value) for value in x_ticks or {}]
        xs += [
            transform(point[key])
            for point in points
            for key in ("x_low", "x_high")
            if key in point
        ]
        ys = [float(point.get("high", point["y"])) for point in points]
        low_y = min(0.0, *(float(point.get("low", point["y"])) for point in points))
        high_y = max(ys + list(y_ticks or {}))
        low_x, high_x = min(xs), max(xs)
        if low_x == high_x:
            low_x, high_x = low_x - 0.5, high_x + 0.5
        high_y = high_y * 1.08 if high_y > 0 else 1.0
        if y_ticks:
            low_y = min(y_ticks) - 0.5
            high_y = max(y_ticks) + 0.5
    else:
        low_x, high_x, low_y, high_y = 0.0, 1.0, 0.0, 1.0
        if x_ticks:
            low_x, high_x = min(x_ticks), max(x_ticks)

    def px(value: float) -> float:
        return left + (transform(value) - low_x) / (high_x - low_x) * (right - left)

    def py(value: float) -> float:
        return bottom - (value - low_y) / (high_y - low_y) * (bottom - top)

    line(left, bottom, right, bottom, "#333333")
    line(left, top, left, bottom, "#333333")
    ticks_y = y_ticks or {
        low_y
        + (high_y - low_y) * index / 4: f"{low_y + (high_y - low_y) * index / 4:.3g}"
        for index in range(5)
    }
    for value, label in ticks_y.items():
        y = py(value)
        line(left, y, right, y, "#dddddd")
        text(12, y + 4, label, 10)
    if x_ticks is None:
        x_values = (
            [low_x + (high_x - low_x) * index / 4 for index in range(5)]
            if points and not log_x
            else sorted({float(point["x"]) for point in points})
        )
        if len(x_values) > 8:
            x_values = [x_values[round(i * (len(x_values) - 1) / 7)] for i in range(8)]
        x_ticks = {value: f"{value:.4g}" for value in x_values}
    for value, label in x_ticks.items():
        if log_x and value <= 0:
            continue
        x = px(value)
        line(x, bottom, x, bottom + 5, "#333333")
        text(x - 12, bottom + 22, label, 10)
    text(290, 545, x_label, 12)
    for index, group in enumerate(series):
        ink = _COLORS[index % len(_COLORS)]
        ordered = sorted(group["points"], key=lambda point: (point["x"], point["y"]))
        if connect:
            for previous, current in zip(ordered, ordered[1:]):
                line(
                    px(previous["x"]),
                    py(previous["y"]),
                    px(current["x"]),
                    py(current["y"]),
                    ink,
                    1.8,
                )
        for point in ordered:
            x, y = px(point["x"]), py(point["y"])
            if "low" in point and "high" in point:
                low, high = py(point["low"]), py(point["high"])
                line(x, low, x, high, ink, 1.4)
                line(x - 4, low, x + 4, low, ink, 1.4)
                line(x - 4, high, x + 4, high, ink, 1.4)
            if "x_low" in point and "x_high" in point:
                low, high = px(point["x_low"]), px(point["x_high"])
                line(low, y, high, y, ink, 1.4)
                line(low, y - 4, low, y + 4, ink, 1.4)
                line(high, y - 4, high, y + 4, ink, 1.4)
            svg.append(
                f'<rect x="{x-3:.2f}" y="{y-3:.2f}" width="6" height="6" fill="{ink}">'
                f"<title>{html.escape(json.dumps(point, sort_keys=True))}</title></rect>"
            )
            pdf.append(f"{color(ink)} rg {x-3:.2f} {600-y-3:.2f} 6 6 re f")
        line(725, 90 + index * 21, 743, 90 + index * 21, ink, 2)
        text(752, 94 + index * 21, str(group["label"]), 10)
    if not points:
        text(180, 275, "No measured values in this frozen artifact set.", 13)
    text(
        35,
        575,
        "Whiskers: median +/- MAD across independent process repeats; not a confidence interval.",
        10,
    )
    text(
        35,
        590,
        "Sources: figure_data.csv, plot_data.csv and embedded run IDs/config hashes. Missing cells stay in tables.",
        10,
    )
    svg.append("</svg>")
    svg_path, pdf_path = path.with_suffix(".svg"), path.with_suffix(".pdf")
    svg_path.write_text("\n".join(svg) + "\n", encoding="utf-8")
    _write_pdf(pdf_path, "\n".join(pdf), metadata, title)
    return svg_path, pdf_path
