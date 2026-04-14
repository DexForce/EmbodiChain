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

from pathlib import Path
from typing import Any


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _group_aggregate_results_by_task(
    aggregate_results: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in aggregate_results:
        grouped.setdefault(item["task"], []).append(item)
    for task_results in grouped.values():
        task_results.sort(
            key=lambda item: (
                -float(item.get("final_success_rate_stable_mean", float("-inf"))),
                -float(item.get("final_success_rate_mean", float("-inf"))),
                float(item.get("steps_to_success_threshold_mean", float("inf"))),
                item["algorithm"],
            )
        )
    return dict(sorted(grouped.items()))


def generate_markdown_report(
    run_results: list[dict[str, Any]],
    aggregate_results: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    plot_artifacts: dict[str, str],
    protocol: dict[str, Any] | None,
    output_path: str | Path,
) -> Path:
    """Write a markdown benchmark report to disk."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RL Benchmark Report",
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
            "## Leaderboard",
            "",
            "| Rank | Algorithm | Score | Steps To Threshold (Sustained) | Success Rate Std | Avg Success Rate | Avg Stable Success Rate | Avg Final Reward | Tasks |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
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

    lines.extend(
        [
            "",
            "## Aggregate Results",
            "",
            "| Task | Algorithm | Runs | Final Reward | Final Success Rate | Final Stable Success Rate | Training FPS | Env FPS |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in aggregate_results:
        lines.append(
            "| {task} | {algorithm} | {num_runs} | {reward} | {success} | {stable_success} | {train_fps} | {env_fps} |".format(
                task=item["task"],
                algorithm=item["algorithm"],
                num_runs=item["num_runs"],
                reward=_fmt(item.get("final_reward_mean", float("nan"))),
                success=_fmt(item.get("final_success_rate_mean", float("nan"))),
                stable_success=_fmt(
                    item.get("final_success_rate_stable_mean", float("nan"))
                ),
                train_fps=_fmt(item.get("training_fps_mean", float("nan"))),
                env_fps=_fmt(item.get("environment_fps_mean", float("nan"))),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Task Comparison",
            "",
            "Each table compares different algorithms on the same task.",
            "",
        ]
    )
    for task, task_results in _group_aggregate_results_by_task(
        aggregate_results
    ).items():
        lines.extend(
            [
                f"### {task}",
                "",
                "| Algorithm | Runs | Final Stable Success Rate | Final Success Rate | Steps To Threshold (Sustained) | Success Rate Std | Final Reward | Training FPS | Env FPS |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in task_results:
            lines.append(
                "| {algorithm} | {num_runs} | {stable_success} | {success} | {steps} | {std} | {reward} | {train_fps} | {env_fps} |".format(
                    algorithm=item["algorithm"],
                    num_runs=item["num_runs"],
                    stable_success=_fmt(
                        item.get("final_success_rate_stable_mean", float("nan"))
                    ),
                    success=_fmt(item.get("final_success_rate_mean", float("nan"))),
                    steps=_fmt(
                        item.get("steps_to_success_threshold_mean", float("nan"))
                    ),
                    std=_fmt(item.get("final_success_rate_std", float("nan"))),
                    reward=_fmt(item.get("final_reward_mean", float("nan"))),
                    train_fps=_fmt(item.get("training_fps_mean", float("nan"))),
                    env_fps=_fmt(item.get("environment_fps_mean", float("nan"))),
                )
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    for plot_name, plot_path in sorted(plot_artifacts.items()):
        relative = Path(plot_path).relative_to(output.parent)
        lines.append(f"### {plot_name}")
        lines.append("")
        lines.append(f"![{plot_name}]({relative.as_posix()})")
        lines.append("")
    lines.extend(
        [
            "## Stability Analysis",
            "",
            "| Task | Algorithm | Success Rate Mean | Stable Success Rate Mean | Success Rate Std | Steps To Threshold Mean | First Hit Mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in aggregate_results:
        lines.append(
            "| {task} | {algorithm} | {mean_value} | {stable_mean} | {std_value} | {steps} | {first_hit} |".format(
                task=item["task"],
                algorithm=item["algorithm"],
                mean_value=_fmt(item.get("final_success_rate_mean", float("nan"))),
                stable_mean=_fmt(
                    item.get("final_success_rate_stable_mean", float("nan"))
                ),
                std_value=_fmt(item.get("final_success_rate_std", float("nan"))),
                steps=_fmt(item.get("steps_to_success_threshold_mean", float("nan"))),
                first_hit=_fmt(
                    item.get("steps_to_success_threshold_first_hit_mean", float("nan"))
                ),
            )
        )
    lines.extend(
        [
            "",
            "## System Performance",
            "",
            "| Task | Algorithm | Training FPS | Env FPS | Peak GPU Memory (MB) |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for item in aggregate_results:
        lines.append(
            "| {task} | {algorithm} | {train_fps} | {env_fps} | {mem} |".format(
                task=item["task"],
                algorithm=item["algorithm"],
                train_fps=_fmt(item.get("training_fps_mean", float("nan"))),
                env_fps=_fmt(item.get("environment_fps_mean", float("nan"))),
                mem=_fmt(item.get("peak_gpu_memory_mb_mean", float("nan"))),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| Task | Algorithm | Seed | Final Reward | Final Success Rate | Final Stable Success Rate | Steps To Threshold | First Hit | Checkpoint |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for result in sorted(
        run_results, key=lambda item: (item["task"], item["algorithm"], item["seed"])
    ):
        lines.append(
            "| {task} | {algorithm} | {seed} | {reward} | {success} | {stable_success} | {steps} | {first_hit} | `{checkpoint}` |".format(
                task=result["task"],
                algorithm=result["algorithm"],
                seed=result["seed"],
                reward=_fmt(result.get("final_reward", float("nan"))),
                success=_fmt(result.get("final_success_rate", float("nan"))),
                stable_success=_fmt(
                    result.get("final_success_rate_stable", float("nan"))
                ),
                steps=result.get("steps_to_success_threshold", "n/a"),
                first_hit=result.get("steps_to_success_threshold_first_hit", "n/a"),
                checkpoint=result.get("checkpoint_path", ""),
            )
        )

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
