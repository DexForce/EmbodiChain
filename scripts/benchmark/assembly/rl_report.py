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
"""Summarize observed native peg PPO learning curves and strict physics health.

Run: python -m scripts.benchmark.assembly.rl_report --run RUN --health HEALTH.json --output outputs/peg_rl_report

This CPU-only reader writes scalar CSV, JSON, a standalone PNG, and one Markdown
report with three tables. Validation and optional final episode evaluation stay
separate. Missing metrics remain unmeasured; interrupted and superseded runs are
never upgraded to qualified learning results.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap
from typing import Any

__all__: list[str] = []

_TRAIN_TAGS = {
    "charts/SPS",
    "charts/episode_reward_avg_100",
    "charts/episode_length_avg_100",
    "train/auxiliary_loss",
}


def _number(value: Any) -> float | None:
    """Keep finite recorded numeric values without substituting zero."""
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(value)
        else None
    )


def _file(path: Path) -> dict[str, Any]:
    """Record immutable small inputs by content digest."""
    data = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _series(run: Path) -> tuple[dict[str, dict[int, dict]], list[dict]]:
    """Read actual scalar events; newest wall time wins at duplicate steps."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series: dict[str, dict[int, dict]] = {}
    files = sorted(run.glob("logs/**/events.out.tfevents.*"))
    sources = []
    for path in files:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        sources.append({"path": str(path), "bytes_at_read": path.stat().st_size})
        for tag in accumulator.Tags()["scalars"]:
            if tag not in _TRAIN_TAGS and not tag.startswith("eval/"):
                continue
            values = series.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                record = {
                    "step": event.step,
                    "wall_time": event.wall_time,
                    "value": _number(event.value),
                }
                if (
                    event.step not in values
                    or event.wall_time >= values[event.step]["wall_time"]
                ):
                    values[event.step] = record
    if not files:
        raise FileNotFoundError(f"No native TensorBoard events in {run / 'logs'}")
    return series, sources


def _validation(series: dict[str, dict[int, dict]]) -> list[dict]:
    """Join only metrics recorded at the same completed validation step."""
    return [
        {
            "global_step": step,
            **{
                tag: values[step]["value"]
                for tag, values in series.items()
                if tag.startswith("eval/") and step in values
            },
        }
        for step in sorted(series.get("eval/success_rate", {}))
    ]


def _health_statuses(
    scopes: dict[str, dict], required: tuple[str, ...]
) -> dict[str, str]:
    """Keep capacity, convergence, raw-state and clock evidence independent."""
    observed = all(
        name in scopes
        and scopes[name].get("strict") is True
        and scopes[name].get("overflow_available") is True
        and scopes[name].get("control_boundaries_observed", 0) > 0
        for name in required
    )
    loss = any(
        value.get("capacity_overflow_seen") or value.get("unclassified_overflow_seen")
        for value in scopes.values()
    )
    convergence = any(
        value.get("convergence_budget_exhausted_seen") is True
        for value in scopes.values()
    )
    states = [scopes.get(name, {}).get("native_state") or {} for name in required]
    nonfinite = any(
        state.get("nonfinite_seen") is True or state.get("all_samples_finite") is False
        for state in states
    )
    complete = all(
        state.get("available") is True
        and {"qpos", "qvel", "qacc"} <= set(state.get("fields", []))
        and state.get("observations", 0) > 0
        and state.get("complete_observations") == state.get("observations")
        and state.get("all_samples_finite") is True
        for state in states
    )
    clocks = [
        scopes.get(name, {}).get("control_clock_observations") or []
        for name in required
    ]
    clock_bad = any(
        sample.get("all_world_clock_deltas_match_request") is False
        for samples in clocks
        for sample in samples
    )
    clock_complete = all(
        samples
        and all(
            sample.get("native_clock_available") is True
            and sample.get("all_world_clock_deltas_match_request") is True
            for sample in samples
        )
        for samples in clocks
    )
    return {
        "native_capacity_status": (
            "unsafe_capacity_or_collision_loss"
            if loss
            else (
                "no_capacity_loss_observed_through_snapshot"
                if observed
                else "unverified_missing_or_incomplete_capacity"
            )
        ),
        "native_convergence_status": (
            "budget_exhaustion_observed"
            if convergence
            else (
                "no_budget_exhaustion_observed_through_snapshot"
                if observed
                and all(
                    "convergence_budget_exhausted_seen" in scopes[name]
                    for name in required
                )
                else "unverified_missing_or_incomplete_convergence"
            )
        ),
        "raw_native_state_status": (
            "unsafe_nonfinite_native_state"
            if nonfinite
            else (
                "all_observed_native_states_finite"
                if complete
                else "unverified_missing_or_incomplete_native_state"
            )
        ),
        "control_clock_status": (
            "control_clock_mismatch_observed"
            if clock_bad
            else (
                "recorded_control_clock_checks_match_request"
                if clock_complete
                else "unverified_missing_or_incomplete_clock"
            )
        ),
    }


def _health(path: Path | None, run: Path) -> dict[str, Any]:
    """Summarize verified RUN association and the health snapshot's coverage."""
    if path is None:
        return {
            **_health_statuses({}, ("train", "eval")),
            "snapshot_transitions": None,
        }
    document = json.loads(path.read_text())
    if document.get("schema") != "embodichain-peg-native-training-health/v1":
        raise ValueError("Unsupported native training health schema.")
    training = document.get("training", {})
    recorded_run = training.get("run_path")
    if recorded_run is not None and Path(recorded_run).resolve() != run:
        raise ValueError("Health JSON belongs to a different RUN.")
    scopes = document.get("health", {})
    return {
        "source": _file(path),
        "run_status": document.get("status"),
        **_health_statuses(
            scopes if recorded_run is not None else {}, ("train", "eval")
        ),
        "snapshot_transitions": training.get("global_step"),
        "snapshot_updates": training.get("num_updates"),
        "elapsed_wall_seconds": _number(document.get("elapsed_wall_seconds")),
        "scope": document.get("observation_scope"),
        "guards": scopes,
        "environments": document.get("environments", {}),
    }


def _checkpoint(run: Path) -> tuple[dict, dict]:
    """Read frozen config and native checkpoint counters using CPU loading."""
    import torch
    import yaml
    from embodichain.learning.rl.policy_evaluation.manifest import RunManifest

    manifest = RunManifest.load(run, check_files=False)
    config = yaml.safe_load(manifest.configs["train"].read_text())
    path = manifest.checkpoints.get("latest")
    checkpoint: dict[str, Any] = {"path": str(path) if path else None}
    if path is not None and path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        checkpoint.update(
            {
                key: _number(payload.get(key))
                for key in ("global_step", "num_updates", "best_eval_value")
            }
        )
    fidelity_path = run / "configs/resources/assembly/fidelity.json"
    checkpoint["fidelity"] = (
        json.loads(fidelity_path.read_text()) if fidelity_path.is_file() else None
    )
    return config, checkpoint


def _heldout(
    path: Path | None, run: Path, health_path: Path | None = None
) -> dict | None:
    """Keep the final independent episode evaluation separate from validation."""
    if path is None:
        if health_path is not None:
            raise ValueError("--heldout-health requires --heldout-eval.")
        return None
    document = json.loads(path.read_text())
    if Path(document.get("inputs", {}).get("run", "")).resolve() != run:
        raise ValueError("Final evaluation JSON must identify this RUN.")
    result = document.get("result", {})
    if not result.get("episodes", 0) > 0:
        raise ValueError("Final evaluation must contain completed episodes.")
    health = _health_statuses({}, ("evaluation",))
    checkpoint = None
    evaluation_seed = None
    if health_path is not None:
        recorded = json.loads(health_path.read_text())
        if recorded.get("schema") != "embodichain-peg-native-evaluation-health/v1":
            raise ValueError("Unsupported final evaluation health schema.")
        if (
            recorded.get("status") != "completed"
            or recorded.get("episodes_requested") != result["episodes"]
        ):
            raise ValueError(
                "Final evaluation health must complete the same episode count."
            )
        if Path(recorded.get("run", "")).resolve() != run:
            raise ValueError("Final evaluation health belongs to another RUN.")
        reference = recorded.get("evaluation_report")
        reference = reference.get("path") if isinstance(reference, dict) else reference
        if not reference or Path(reference).resolve() != path.resolve():
            raise ValueError(
                "Final evaluation health must identify the supplied evaluation JSON."
            )
        health = {
            "source": _file(health_path),
            **_health_statuses(
                {"evaluation": recorded.get("native_health", {})}, ("evaluation",)
            ),
            "guards": recorded.get("native_health", {}),
        }
        checkpoint = recorded.get("checkpoint")
        evaluation_seed = recorded.get("evaluation_reset_seed")
    return {
        "source": _file(path),
        "inputs": document["inputs"],
        "episodes": result["episodes"],
        "metrics": {
            key: _number(value) for key, value in result.get("metrics", {}).items()
        },
        "health": health,
        "checkpoint": checkpoint,
        "evaluation_reset_seed": evaluation_seed,
        "scope": "independent final episodes; held-out geometry is not implied",
    }


def _selection(path: Path | None, run: Path, assessment: dict | None) -> dict | None:
    """Bind a balanced validation decision to the actually tested weights."""
    if path is None:
        return None
    from scripts.benchmark.assembly.select_peg import _select_best

    document = json.loads(path.read_text())
    if (
        document.get("schema") != "embodichain-peg-balanced-selection/v1"
        or document.get("status") != "completed"
        or Path(document.get("run", "")).resolve() != run
        or document.get("protocol", {}).get("episode_allocation") != "balanced"
    ):
        raise ValueError("Selection must complete balanced validation of this RUN.")
    candidates = document["candidates"]
    declared = document["declared_candidates"]
    identity = lambda row: (row["num_updates"], row["sha256"])
    if len(candidates) != len(declared) or {identity(row) for row in candidates} != {
        identity(row) for row in declared
    }:
        raise ValueError("Every declared checkpoint must appear in selection evidence.")
    selected = _select_best(candidates)
    if identity(selected) != identity(document["selected_checkpoint"]):
        raise ValueError("The selected checkpoint disagrees with validation ranking.")
    if (
        hashlib.sha256(Path(selected["path"]).read_bytes()).hexdigest()
        != selected["sha256"]
    ):
        raise ValueError("Validation-selected checkpoint contents changed.")
    if assessment is not None:
        checkpoint = assessment.get("checkpoint") or {}
        if checkpoint.get("sha256") != selected["sha256"]:
            raise ValueError(
                "Final evaluation used a different validation-selected checkpoint."
            )
        if (
            assessment.get("evaluation_reset_seed") is None
            or assessment.get("evaluation_reset_seed") == document["protocol"]["seed"]
        ):
            raise ValueError("Final evaluation seed must differ from selection seed.")
    return {
        "source": _file(path),
        "protocol": document["protocol"],
        "selected_checkpoint": selected,
        "points": [
            {
                "global_step": row["global_step"],
                "num_updates": row["num_updates"],
                "success_rate": row["success_rate"],
            }
            for row in candidates
        ],
        "scope": "balanced development checkpoint selection before independent testing",
    }


def _plot(series: dict, validation: list[dict], report: dict, output: Path) -> None:
    """Render recorded validation confidence intervals and unsmoothed training data."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, layout="constrained"
    )
    color = "#2864ad" if report["disposition"] == "active" else "#9a4c36"
    points = [row for row in validation if row.get("eval/success_rate") is not None]
    if points:
        x = np.array([row["global_step"] for row in points]) / 1e6
        axes[0].plot(
            x,
            [100 * row["eval/success_rate"] for row in points],
            "o-",
            color=color,
            label="Training validation (first_completed)",
        )
        if any(row.get("eval/success_ci95_low") is not None for row in points):
            axes[0].fill_between(
                x,
                [
                    (
                        100 * row["eval/success_ci95_low"]
                        if row.get("eval/success_ci95_low") is not None
                        else np.nan
                    )
                    for row in points
                ],
                [
                    (
                        100 * row["eval/success_ci95_high"]
                        if row.get("eval/success_ci95_high") is not None
                        else np.nan
                    )
                    for row in points
                ],
                color=color,
                alpha=0.18,
                label="Recorded 95% Wilson CI",
            )
    elif report.get("checkpoint_selection") is None:
        axes[0].text(
            0.5,
            0.5,
            "No completed validation recorded",
            transform=axes[0].transAxes,
            ha="center",
        )
    selection = report.get("checkpoint_selection")
    if selection is not None:
        points = selection["points"]
        axes[0].plot(
            [row["global_step"] / 1e6 for row in points],
            [100 * row["success_rate"] for row in points],
            "s-",
            color="#2b7c52",
            label="Balanced checkpoint validation",
        )
    if validation or selection:
        axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set(ylim=(0, 100), ylabel="Success (%)")
    for axis, tag, label in (
        (
            axes[1],
            "charts/episode_reward_avg_100",
            "Mean return, last 100\ncompleted training episodes",
        ),
        (axes[2], "train/auxiliary_loss", "Auxiliary contact loss"),
    ):
        values = sorted(series.get(tag, {}).values(), key=lambda row: row["step"])
        if values:
            axis.plot(
                [row["step"] / 1e6 for row in values],
                [
                    row["value"] if row["value"] is not None else np.nan
                    for row in values
                ],
                color=color,
                linewidth=1.5,
            )
        else:
            axis.text(0.5, 0.5, "Unmeasured", transform=axis.transAxes, ha="center")
        axis.set_ylabel(label)
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Observed training transitions (millions)")
    title = f"{report['disposition'].upper()} — {Path(report['run']).name}"
    figure.suptitle(textwrap.fill(title, 85), fontsize=10)
    figure.savefig(output / "learning_curves.png", dpi=180)
    plt.close(figure)


def _markdown(report: dict, output: Path) -> None:
    """Write exactly three tables without conflating validation and final testing."""

    def formatted(value: Any) -> str:
        if value is None:
            return "unmeasured"
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    health = report["health"]
    validation = report["validation"]
    rows = (
        [("latest training validation (first_completed)", validation[-1])]
        if validation
        else []
    )
    if report.get("checkpoint_selection") is not None:
        rows.append(
            (
                "selected balanced development validation",
                report["checkpoint_selection"]["selected_checkpoint"]["metrics"],
            )
        )
    final_health_note = ""
    if report["final_episode_evaluation"] is not None:
        final_health = report["final_episode_evaluation"]["health"]
        final_health_note = (
            f"Fresh final-evaluation coverage only: capacity `{final_health['native_capacity_status']}`, "
            f"convergence `{final_health['native_convergence_status']}`, raw qpos/qvel/qacc `{final_health['raw_native_state_status']}`, "
            f"clock `{final_health['control_clock_status']}`. This does not upgrade earlier training coverage."
        )
        rows.append(
            (
                "independent final episodes",
                report["final_episode_evaluation"]["metrics"],
            )
        )
    lines = [
        "# Native peg RL learning record",
        "",
        f"Disposition: **{report['disposition']}**. {report['qualification']}",
        "",
        f"Observed completed-update transitions: {report['observed']['last_completed_update_transitions']}; recorded updates: {report['observed']['completed_updates_recorded']}.",
        f"Native capacity: `{health['native_capacity_status']}` through snapshot step {health.get('snapshot_transitions')}; current health status: {health.get('run_status', 'unavailable')}.",
        f"Convergence budget: `{health['native_convergence_status']}`. Raw native qpos/qvel/qacc: `{health['raw_native_state_status']}`. Control clock: `{health['control_clock_status']}`.",
        final_health_note,
        "",
        "## Time & Memory",
        "",
        "| implementation | cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb | measured_scope |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Native PPO | {formatted(health.get('elapsed_wall_seconds') * 1000 if health.get('elapsed_wall_seconds') is not None else None)} | unmeasured | unmeasured | unmeasured | health-wrapper wall through snapshot, includes setup/validation |",
        "",
        "## Success & Other Metrics",
        "",
        "| scope | success_rate | ci95_low | ci95_high | success_samples | avg_reward |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for scope, metrics in rows:
        values = [
            formatted(metrics.get("eval/" + key))
            for key in (
                "success_rate",
                "success_ci95_low",
                "success_ci95_high",
                "success_samples",
                "avg_reward",
            )
        ]
        lines.append("| " + " | ".join([scope, *values]) + " |")
    if not rows:
        lines.append(
            "| no completed episode evaluation | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured |"
        )
    selected = rows[-1][1].get("eval/success_rate") if rows else None
    lines += [
        "",
        "## Leaderboard",
        "",
        "| rank | algorithm | overall_success_rate | scope | qualification |",
        "| --- | --- | --- | --- | --- |",
        f"| {'1' if report['disposition'] == 'active' else 'excluded'} | Native PPO | {formatted(selected)} | {rows[-1][0] if rows else 'unmeasured'} | nominal single-run evidence only; {report['disposition']} |",
        "",
        "![Recorded learning curves](learning_curves.png)",
        "",
        "One geometry per run, uncalibrated robot/gripper dynamics and local resets; this does not reproduce the full paper distribution or establish sim-to-real capability.",
        "Confidence intervals are the recorded episode intervals, not uncertainty across training seeds. Curves are unsmoothed. A running health snapshot may lag TensorBoard events; no missing metrics are filled with zero.",
        "Finite RL losses or sanitized actor observations do not establish raw native state finiteness. Missing native-state checks remain unverified. Recorded clock checks cover only their explicitly observed initial controls, not all individual physics substeps.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def run_all_benchmarks(args: argparse.Namespace) -> Path:
    """Read the native run and persist its observed learning evidence."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    run = args.run.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    series, events = _series(run)
    config, checkpoint = _checkpoint(run)
    health = _health(args.health.expanduser().resolve() if args.health else None, run)
    validation = _validation(series)
    update_steps = sorted(series.get("train/auxiliary_loss", {}))
    health_step = health.get("snapshot_transitions")
    final_evaluation = _heldout(
        args.heldout_eval.resolve() if args.heldout_eval else None,
        run,
        args.heldout_health.resolve() if args.heldout_health else None,
    )
    selection = _selection(
        (
            args.selection_report.resolve()
            if getattr(args, "selection_report", None)
            else None
        ),
        run,
        final_evaluation,
    )
    health_summaries = [health] + (
        [final_evaluation["health"]] if final_evaluation else []
    )
    report = {
        "schema": "embodichain-assembly-rl-learning-report/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(run),
        "disposition": (
            "unsafe"
            if any(
                summary[key].startswith("unsafe")
                for summary in health_summaries
                for key in ("native_capacity_status", "raw_native_state_status")
            )
            else args.disposition
        ),
        "requested_disposition": args.disposition,
        "qualification": "Nominal single-geometry development evidence; no formal paper-reproduction qualification.",
        "observed": {
            "last_completed_update_transitions": max(update_steps, default=0),
            "completed_updates_recorded": len(update_steps),
            "health_covers_last_completed_update": health_step is not None
            and health_step >= max(update_steps, default=0),
            "checkpoint": checkpoint,
        },
        "configured_budget": {
            key: config["trainer"].get(key)
            for key in ("num_envs", "iterations", "buffer_size", "seed")
        },
        "health": health,
        "validation": validation,
        "checkpoint_selection": selection,
        "final_episode_evaluation": final_evaluation,
        "tensorboard_sources": events,
        "scalar_policy": "Newest wall time wins at duplicate tag/step; non-finite recorded values remain null, never zero. Completed updates count unique auxiliary-loss events, not the configured budget.",
        "series": {
            tag: [values[step] for step in sorted(values)]
            for tag, values in series.items()
        },
    }
    with (output / "scalars.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["tag", "global_step", "wall_time", "value"])
        for tag, values in report["series"].items():
            writer.writerows(
                (tag, row["step"], row["wall_time"], row["value"]) for row in values
            )
    (output / "learning_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    _plot(series, validation, report, output)
    _markdown(report, output)
    print(f"Markdown report saved: {output / 'report.md'}", flush=True)
    return output


def main() -> None:
    """Select the run, its optional strict health snapshot and final evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--health", type=Path)
    parser.add_argument("--heldout-eval", type=Path)
    parser.add_argument("--heldout-health", type=Path)
    parser.add_argument("--selection-report", type=Path)
    parser.add_argument(
        "--disposition", choices=("active", "superseded", "unsafe"), default="active"
    )
    parser.add_argument("--output", type=Path, required=True)
    run_all_benchmarks(parser.parse_args())


if __name__ == "__main__":
    main()
