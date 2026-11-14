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

"""Seed-level statistics, raw tables and portable plots for NeRD policy training."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from .nerd_policy import BenchmarkCfg, _json

__all__: list[str] = []


def _mean_std(values: list[float | None]) -> tuple[float, float]:
    if any(value is None for value in values):
        return float("nan"), float("nan")
    data = np.asarray(values, dtype=float)
    return float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else 0.0


def _format(values: list[float | None], digits: int = 2) -> str:
    mean, std = _mean_std(values)
    return (
        f"{mean:.{digits}f} ± {std:.{digits}f}"
        if np.isfinite(mean)
        else "undefined (nonfinite episode)"
    )


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_complete(
    output: Path, cfg: BenchmarkCfg
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Admit only a complete matrix with paired actors, models and physical starts."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    expected_steps = cfg.num_envs * cfg.horizon * cfg.updates
    for robot in cfg.robots:
        previous_identity = None
        for seed in cfg.seeds:
            paired_identity = None
            for algorithm in cfg.algorithms:
                path = output / robot / algorithm / f"seed-{seed}" / "result.json"
                result = json.loads(path.read_text())
                identity = result["identity"]
                # Lists and tuples serialize to the same canonical recipe.
                if json.dumps(identity["recipe"], sort_keys=True) != json.dumps(
                    cfg.to_dict(), sort_keys=True
                ):
                    raise ValueError(f"Recipe mismatch: {path}")
                if (identity["robot"], identity["algorithm"], identity["seed"]) != (
                    robot,
                    algorithm,
                    seed,
                ):
                    raise ValueError(f"Run identity mismatch: {path}")
                evaluations = result["evaluations"]
                expected_curve = [
                    index * expected_steps // cfg.checkpoints
                    for index in range(cfg.checkpoints + 1)
                ]
                if [row["global_step"] for row in evaluations] != expected_curve:
                    raise ValueError(f"Incomplete evaluation curve: {path}")
                if any(
                    len(row["episodes"]) != cfg.eval_episodes for row in evaluations
                ):
                    raise ValueError(f"Incomplete evaluation corpus: {path}")
                shared = tuple(
                    identity[key]
                    for key in (
                        "model_weights_sha256",
                        "model_manifest_sha256",
                        "physical_signature",
                        "evaluation_initial_states_sha256",
                    )
                )
                shared = (
                    *shared,
                    json.dumps(identity.get("source_sha256", {}), sort_keys=True),
                    json.dumps(
                        identity.get("runtime", {}).get("source_sha256", {}),
                        sort_keys=True,
                    ),
                    json.dumps(
                        identity.get("runtime", {}).get("versions", {}), sort_keys=True
                    ),
                )
                paired = (
                    *shared,
                    identity["initial_actor_sha256"],
                    identity["initial_state_bank_sha256"],
                )
                if previous_identity is not None and previous_identity != shared:
                    raise ValueError(f"Model/evaluation corpus mismatch: {path}")
                if paired_identity is not None and paired_identity != paired:
                    raise ValueError(
                        f"Algorithms have unpaired initial conditions: {path}"
                    )
                previous_identity, paired_identity = shared, paired
                groups.setdefault((robot, algorithm), []).append(result)
    return groups


def _plots(
    output: Path, robot: str, groups: dict[tuple[str, str], list[dict[str, Any]]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for algorithm, color in (("apg", "#2070b4"), ("ppo", "#db791e")):
        if (robot, algorithm) not in groups:
            continue
        runs = groups[robot, algorithm]
        definitions = [
            (
                axes[0, 0],
                "Training episode return (NeRD, last 100)",
                "Transitions",
                "Return",
                "training",
            ),
            (
                axes[0, 1],
                "Deterministic return in analytical physics",
                "Transitions",
                "Episode return",
                "return_mean",
            ),
            (
                axes[1, 0],
                "Analytical return vs training time",
                "Cumulative training seconds",
                "Episode return",
                "time",
            ),
            (
                axes[1, 1],
                "Analytical episode length",
                "Transitions",
                "Control steps (60 Hz)",
                "length_mean",
            ),
        ]
        for axis, title, xlabel, ylabel, metric in definitions:
            if metric == "training":
                x = np.asarray([row["global_step"] for row in runs[0]["history"]])
                values = [
                    [row.get("charts/episode_reward_avg_100") for row in run["history"]]
                    for run in runs
                ]
            else:
                x = np.asarray(
                    [
                        [
                            (
                                row["training_seconds"]
                                if metric == "time"
                                else row["global_step"]
                            )
                            for row in run["evaluations"]
                        ]
                        for run in runs
                    ]
                ).mean(0)
                key = "return_mean" if metric == "time" else metric
                values = [[row[key] for row in run["evaluations"]] for run in runs]
            y = np.asarray(values, dtype=float)
            # Do not hide seed failures with nanmean or select the best checkpoint.
            mean = y.mean(0)
            std = y.std(0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
            axis.plot(x, mean, color=color, label=algorithm.upper(), linewidth=1.8)
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
            axis.grid(alpha=0.2)
            axis.legend()
    figure.suptitle(
        f"{robot.capitalize()}: APG / PPO, seed mean ± sample SD", fontsize=15
    )
    for extension in ("png", "svg"):
        figure.savefig(output / f"{robot}-curves.{extension}", dpi=160)
    plt.close(figure)


def write_report(output: Path, cfg: BenchmarkCfg) -> Path:
    """Write complete-matrix reports without ranking incompatible task rewards."""
    groups = _load_complete(output, cfg)
    budget = cfg.num_envs * cfg.horizon * cfg.updates
    lines = [
        "# Cartpole / Ant APG–PPO benchmark",
        "",
        f"Fixed budget: **{budget:,} training transitions per run**; seeds **{list(cfg.seeds)}**; {cfg.eval_episodes} held-out episodes/checkpoint. Entries are mean ± sample SD **across training seeds**, not across pooled episodes.",
        "",
        "## Protocol",
        "",
        "Both algorithms train in the same frozen NeRD export, using the registered DifferentiableEnv tasks. The actor is 64–64 tanh with tanh output; the initial actor, log standard deviation, reset bank, action limits, reward, failure rules, discount, vector count and transition budget are paired. Both instantiate the same actor-critic policy; only PPO trains the critic. APG uses pathwise TBPTT; PPO uses GAE, configured epochs and minibatches (see recipe.json). No checkpoint selection or per-seed tuning is applied.",
        "",
        "Every checkpoint, including initialization, is evaluated deterministically in the model-recorded analytical Scene domain, with its original solver, macro timestep, substeps, gravity, asset/contact/friction parameters. Physical signatures and codec layouts are checked before training and evaluation. Every fixed initial state is evaluated once; early finishers are not resampled. Evaluation seeds and RNG are independent of training. Evaluation is performed after training, so it cannot change training RNG or environment state.",
        "",
        "## Final-checkpoint analytical returns and time",
        "",
        "| Task | Algorithm | Return | Episode length | Failure rate | Training seconds | Pre-training setup + warmup seconds | Evaluation seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    runtime = next(iter(groups.values()))[0]["identity"]["runtime"]
    lines[2:2] = [
        f"Generated: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        f"Runtime: {runtime['device_name']}; Python {runtime['python']}; CUDA {runtime['cuda_runtime']}; "
        + "; ".join(f"{key} {value}" for key, value in runtime["versions"].items())
        + ".",
        "",
    ]
    summary = []
    evaluations, histories, episodes = [], [], []
    for robot in cfg.robots:
        for algorithm in cfg.algorithms:
            runs = groups[robot, algorithm]
            final = [run["evaluations"][-1] for run in runs]
            lines.append(
                f"| {robot} | {algorithm.upper()} | {_format([r['return_mean'] for r in final])} | {_format([r['length_mean'] for r in final])} | {_format([r['failure_rate'] for r in final], 3)} | {_format([r['training_seconds'] for r in runs])} | {_format([r['setup_and_warmup_seconds'] + r['training_setup_seconds'] for r in runs])} | {_format([r['evaluation_seconds'] for r in runs])} |"
            )
            row = {"robot": robot, "algorithm": algorithm, "seeds": list(cfg.seeds)}
            for key in (
                "return_mean",
                "length_mean",
                "failure_rate",
                "quality_mean",
                "nonfinite_episodes",
            ):
                mean, std = _mean_std([point[key] for point in final])
                row[key] = {"mean": mean, "sample_std": std}
            row["training_seconds"] = dict(
                zip(
                    ("mean", "sample_std"),
                    _mean_std([run["training_seconds"] for run in runs]),
                )
            )
            summary.append(row)
            for run in runs:
                common = {
                    "robot": robot,
                    "algorithm": algorithm,
                    "seed": run["identity"]["seed"],
                }
                histories.extend({**common, **point} for point in run["history"])
                for point in run["evaluations"]:
                    evaluations.append(
                        {
                            **common,
                            **{k: v for k, v in point.items() if k != "episodes"},
                        }
                    )
                    episodes.extend(
                        {**common, "global_step": point["global_step"], **episode}
                        for episode in point["episodes"]
                    )
    lines.extend(
        [
            "",
            "Training time is synchronized completed wall time for rollout collection + optimizer updates + trainer bookkeeping. It excludes resource download, imports, environment/model setup, throwaway warmup, checkpoint writes, analytical evaluation and report generation. Warmup updates are discarded, and measured runs reconstruct the policy/environment/reset bank and RNG. Torch and Warp are synchronized at each timing window boundary. Pre-training setup and evaluation are reported separately. The setup column includes pre-training domain validation, throwaway warmup and reconstruction of the training runtime; post-training analytical Scene creation is excluded from evaluation timing. Published NeRD data generation/model pretraining was not rerun; its historical time is unavailable and is **not included** in policy-training speed comparisons. Hardware is shared; each run owns one GPU, with paired algorithms sequential on that GPU and order alternating by seed. CPU threads, software versions, source hashes and model hashes are in each result identity. Torch allocator peak memory excludes native/Warp allocations.",
            "",
            "## Per-seed evidence",
            "",
            "| Task | Algorithm | Seed | Initial return | Final return | Length | Failure rate | Nonfinite episodes | Quality | Train s | Torch peak MiB |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for (robot, algorithm), runs in groups.items():
        for run in runs:
            first, last = run["evaluations"][0], run["evaluations"][-1]
            peak = run.get("peak_torch_allocated_mb")
            peak_text = f"{peak:.1f}" if peak is not None else "n/a"
            lines.append(
                f"| {robot} | {algorithm.upper()} | {run['identity']['seed']} | {first['return_mean']} | {last['return_mean']} | {last['length_mean']:.2f} | {last['failure_rate']:.3f} | {last['nonfinite_episodes']} | {last['quality_mean']:.3f} | {run['training_seconds']:.2f} | {peak_text} |"
            )
    lines.extend(
        [
            "",
            "## Final task quality",
            "",
            "| Task | Algorithm | Mean absolute pole angle (rad) / forward speed (m/s) | Nonfinite episodes | Skipped APG updates |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for (robot, algorithm), runs in groups.items():
        quality = _format([run["evaluations"][-1]["quality_mean"] for run in runs], 3)
        invalid = sum(run["evaluations"][-1]["nonfinite_episodes"] for run in runs)
        skipped = (
            str(
                sum(
                    point.get("train/skipped_update", 0)
                    for run in runs
                    for point in run["history"]
                )
            )
            if algorithm == "apg"
            else "n/a"
        )
        lines.append(
            f"| {robot} | {algorithm.upper()} | {quality} | {invalid} | {skipped} |"
        )
    lines.extend(
        [
            "",
            "Quality is mean absolute wrapped pole angle (radians, lower is better) for Cartpole, and mean forward COM velocity (+Y m/s, higher is better) for Ant, averaged over each episode's active steps before averaging episodes. The Cartpole reward is nonpositive: a short failed episode can have a less negative total return. Interpret return together with length, angle and failures. Ant's survival/heading bonus can reward standing; inspect forward speed as well. No task reward or failure penalty was modified for this benchmark. Nonfinite episodes are counted and retained; any such episode makes the checkpoint mean return undefined rather than silently excluding it.",
            "",
            "These are fixed-budget baseline runs, not evidence of convergence or a paper-scale reproduction. Three training seeds provide limited uncertainty estimates; bands show sample SD, not a confidence interval. Seeds fix RNG streams; GPU kernels can differ in low-order floating-point results across runs, and bitwise replay is not claimed.",
            "",
        ]
    )
    lines.extend(["## Observed final-checkpoint behavior", ""])
    for robot in cfg.robots:
        for algorithm in cfg.algorithms:
            runs = groups[robot, algorithm]
            points = [run["evaluations"][-1] for run in runs]
            metric_name = (
                "mean absolute pole angle (rad)"
                if robot == "cartpole"
                else "mean forward speed (m/s)"
            )
            lines.append(
                f"- {robot} {algorithm.upper()}: episode length {_format([point['length_mean'] for point in points])}, failure rate {_format([point['failure_rate'] for point in points], 3)}, {metric_name} {_format([point['quality_mean'] for point in points], 3)}."
            )
    lines.extend(["", "## Curves", ""])
    for robot in cfg.robots:
        _plots(output, robot, groups)
        lines.extend(
            [f"![{robot} training and analytical evaluation]({robot}-curves.png)", ""]
        )
    _csv(output / "training.csv", histories)
    _csv(output / "evaluation.csv", evaluations)
    _csv(output / "episodes.csv", episodes)
    _json(output / "summary.json", summary)
    _json(output / "recipe.json", cfg.to_dict())
    path = output / "report.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"Markdown report saved: {path}", flush=True)
    return path
