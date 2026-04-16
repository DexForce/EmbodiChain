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

from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def generate_markdown_report(
    run_results: list[dict[str, Any]],
    aggregate_results: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    plot_artifacts: dict[str, str],
    protocol: dict[str, Any] | None,
    output_path: str | Path,
) -> Path:
    """Write a benchmark markdown report with exactly two tables."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ordered_runs = sorted(
        run_results,
        key=lambda item: (
            str(item.get("task", "")),
            str(item.get("algorithm", "")),
            int(item.get("seed", 0)),
        ),
    )

    lines = [
        "# RL Benchmark Report",
        "",
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Benchmark Overview",
        "",
    ]
    if protocol:
        lines.extend(
            [
                f"- device: `{protocol.get('device')}`",
                f"- headless: `{protocol.get('headless')}`",
                f"- iterations: `{protocol.get('iterations')}`",
                f"- buffer_size: `{protocol.get('buffer_size')}`",
                f"- num_envs: `{protocol.get('num_envs')}`",
                f"- num_eval_envs: `{protocol.get('num_eval_envs')}`",
                f"- evaluation_interval: `{protocol.get('evaluation_interval')}`",
                f"- evaluation_episodes: `{protocol.get('evaluation_episodes')}`",
                f"- threshold_sustain_count: `{protocol.get('threshold_sustain_count', 3)}`",
                f"- final_eval_window: `{protocol.get('final_eval_window', 3)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Time & Memory",
            "",
            "| task | algorithm | seed | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | training_fps | env_fps |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in ordered_runs:
        train_steps = float(result.get("train_steps", float("nan")))
        training_fps = float(result.get("training_fps", float("nan")))
        cost_time_ms = _safe_divide(train_steps, training_fps) * 1000.0
        lines.append(
            "| {task} | {algorithm} | {seed} | {cost_time_ms} | {cpu_delta} | {gpu_delta} | {peak_gpu} | {train_fps} | {env_fps} |".format(
                task=result["task"],
                algorithm=result["algorithm"],
                seed=result["seed"],
                cost_time_ms=_fmt(cost_time_ms),
                cpu_delta=_fmt(result.get("cpu_delta_mb", "n/a")),
                gpu_delta=_fmt(result.get("gpu_delta_mb", "n/a")),
                peak_gpu=_fmt(result.get("peak_gpu_memory_mb", float("nan"))),
                train_fps=_fmt(result.get("training_fps", float("nan"))),
                env_fps=_fmt(result.get("environment_fps", float("nan")), digits=2),
            )
        )

    lines.extend(
        [
            "",
            "## Success & Other Metrics",
            "",
            "| task | algorithm | seed | success_rate | stable_success_rate | steps_to_threshold | first_hit | final_reward | final_episode_length |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in ordered_runs:
        lines.append(
            "| {task} | {algorithm} | {seed} | {success} | {stable_success} | {steps} | {first_hit} | {reward} | {episode_len} |".format(
                task=result["task"],
                algorithm=result["algorithm"],
                seed=result["seed"],
                success=_fmt(result.get("final_success_rate", float("nan"))),
                stable_success=_fmt(
                    result.get("final_success_rate_stable", float("nan"))
                ),
                steps=_fmt(result.get("steps_to_success_threshold", float("nan"))),
                first_hit=_fmt(
                    result.get("steps_to_success_threshold_first_hit", float("nan"))
                ),
                reward=_fmt(result.get("final_reward", float("nan"))),
                episode_len=_fmt(result.get("final_episode_length", float("nan"))),
            )
        )

    lines.extend(["", "## Notes", ""])
    if leaderboard:
        top = leaderboard[0]
        lines.append(
            "- Top algorithm by leaderboard score: "
            f"`{top.get('algorithm', 'n/a')}` (score={_fmt(top.get('score', float('nan')))})."
        )
    if aggregate_results:
        lines.append(f"- Aggregate summaries available: `{len(aggregate_results)}`.")

    if plot_artifacts:
        lines.extend(["", "## Plots", ""])
    for plot_name, plot_path in sorted(plot_artifacts.items()):
        relative = Path(plot_path).relative_to(output.parent)
        lines.append(f"- {plot_name}: ![{plot_name}]({relative.as_posix()})")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def generate_leaderboard_markdown(
    leaderboard: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write a dedicated leaderboard markdown artifact."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Leaderboard",
        "",
        "| Rank | Algorithm | Score | Steps To Threshold (Sustained) | Success Rate Std | Avg Success Rate | Avg Stable Success Rate | Avg Final Reward | Tasks |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in leaderboard:
        lines.append(
            "| {rank} | {algorithm} | {score} | {steps} | {std} | {success} | {stable_success} | {reward} | {tasks} |".format(
                rank=item["rank"],
                algorithm=item["algorithm"],
                score=_fmt(item.get("score", float("nan"))),
                steps=_fmt(item.get("steps_to_success_threshold", float("nan"))),
                std=_fmt(item.get("success_rate_std", float("nan"))),
                success=_fmt(item.get("avg_success_rate", float("nan"))),
                stable_success=_fmt(item.get("avg_success_rate_stable", float("nan"))),
                reward=_fmt(item.get("avg_final_reward", float("nan"))),
                tasks=item.get("tasks_covered", 0),
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


__all__ = ["generate_leaderboard_markdown", "generate_markdown_report"]
