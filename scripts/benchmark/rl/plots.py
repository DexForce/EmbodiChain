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

from __future__ import annotations

from collections import defaultdict
from math import isnan
from pathlib import Path
from statistics import mean
from typing import Any


COLORS = ["#1768ac", "#f26419", "#2a9134", "#c44536", "#6a4c93", "#1982c4"]


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8" />',
    ]


def _line_chart_svg(
    title: str,
    series: dict[str, list[tuple[float, float]]],
    width: int = 900,
    height: int = 420,
) -> str:
    margin_left = 70
    margin_right = 20
    margin_top = 40
    margin_bottom = 50
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_points = [point for points in series.values() for point in points]
    xs = [point[0] for point in all_points] or [0.0, 1.0]
    ys = [point[1] for point in all_points if not isnan(point[1])] or [0.0, 1.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    def tx(x: float) -> float:
        return margin_left + (x - x_min) / (x_max - x_min) * plot_width

    def ty(y: float) -> float:
        return margin_top + plot_height - (y - y_min) / (y_max - y_min) * plot_height

    lines = _svg_header(width, height)
    lines.extend(
        [
            f'<text x="{margin_left}" y="24" font-size="20" font-family="Arial" fill="#222">{title}</text>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1.5" />',
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#333" stroke-width="1.5" />',
        ]
    )
    for idx in range(5):
        y_val = y_min + (y_max - y_min) * idx / 4.0
        y_pos = ty(y_val)
        lines.append(
            f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{margin_left + plot_width}" y2="{y_pos:.2f}" stroke="#e8e1d6" stroke-width="1" />'
        )
        lines.append(
            f'<text x="10" y="{y_pos + 4:.2f}" font-size="12" font-family="Arial" fill="#555">{y_val:.3f}</text>'
        )

    for idx, (label, points) in enumerate(sorted(series.items())):
        color = COLORS[idx % len(COLORS)]
        polyline_points = " ".join(
            f"{tx(x):.2f},{ty(y):.2f}" for x, y in points if not isnan(y)
        )
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{polyline_points}" />'
        )
        legend_y = margin_top + 18 * idx
        lines.append(
            f'<line x1="{width - 180}" y1="{legend_y}" x2="{width - 150}" y2="{legend_y}" stroke="{color}" stroke-width="3" />'
        )
        lines.append(
            f'<text x="{width - 140}" y="{legend_y + 4}" font-size="12" font-family="Arial" fill="#333">{label}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _bar_chart_svg(
    title: str,
    items: list[tuple[str, float]],
    width: int = 900,
    height: int = 420,
) -> str:
    margin_left = 80
    margin_right = 20
    margin_top = 40
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    values = [value for _, value in items if not isnan(value)] or [1.0]
    value_max = max(values)
    if value_max <= 0:
        value_max = 1.0

    lines = _svg_header(width, height)
    lines.append(
        f'<text x="{margin_left}" y="24" font-size="20" font-family="Arial" fill="#222">{title}</text>'
    )
    bar_width = plot_width / max(len(items), 1)
    for idx, (label, value) in enumerate(items):
        color = COLORS[idx % len(COLORS)]
        bar_height = 0.0 if isnan(value) else (value / value_max) * plot_height
        x = margin_left + idx * bar_width + 10
        y = margin_top + plot_height - bar_height
        lines.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 20, 10):.2f}" height="{bar_height:.2f}" fill="{color}" rx="4" />'
        )
        lines.append(
            f'<text x="{x + max(bar_width - 20, 10) / 2:.2f}" y="{margin_top + plot_height + 18}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{label}</text>'
        )
        lines.append(
            f'<text x="{x + max(bar_width - 20, 10) / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{value:.3f}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines)


def build_plot_artifacts(
    run_results: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, str]:
    """Generate SVG plot artifacts and return named paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    grouped_histories: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    grouped_rewards: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in run_results:
        key = (result["task"], result["algorithm"])
        for item in result.get("eval_history", []):
            step = item.get("global_step")
            success = item.get("eval/success_rate")
            reward = item.get("eval/avg_reward")
            if isinstance(step, (int, float)) and isinstance(success, (int, float)):
                grouped_histories[key][float(step)].append(float(success))
            if isinstance(step, (int, float)) and isinstance(reward, (int, float)):
                grouped_rewards[key][float(step)].append(float(reward))

    tasks = sorted({result["task"] for result in run_results})
    for task in tasks:
        success_series = {}
        reward_series = {}
        for task_name, algorithm in sorted(grouped_histories.keys()):
            if task_name != task:
                continue
            success_series[algorithm] = sorted(
                (step, mean(values))
                for step, values in grouped_histories[(task_name, algorithm)].items()
            )
            reward_series[algorithm] = sorted(
                (step, mean(values))
                for step, values in grouped_rewards[(task_name, algorithm)].items()
            )
        if success_series:
            path = output / f"{task}_success_rate.svg"
            path.write_text(
                _line_chart_svg(f"{task} Success Rate", success_series),
                encoding="utf-8",
            )
            artifacts[f"{task}_success_rate"] = str(path)
        if reward_series:
            path = output / f"{task}_reward.svg"
            path.write_text(
                _line_chart_svg(f"{task} Evaluation Reward", reward_series),
                encoding="utf-8",
            )
            artifacts[f"{task}_reward"] = str(path)

    leaderboard_path = output / "leaderboard_score.svg"
    leaderboard_path.write_text(
        _bar_chart_svg(
            "Leaderboard Score",
            [(item["algorithm"], float(item["score"])) for item in leaderboard],
        ),
        encoding="utf-8",
    )
    artifacts["leaderboard_score"] = str(leaderboard_path)
    return artifacts


__all__ = ["build_plot_artifacts"]
