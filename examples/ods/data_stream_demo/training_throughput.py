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

"""Measure generator, recorder and hot-learner service without changing reuse caps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

__all__ = ["analyze_run", "write_calibration", "compare_calibrations"]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _duration(rows: list[dict[str, Any]]) -> float:
    """Return interval union length, never a sum of overlapping services."""
    spans = sorted((float(r["start_seconds"]), float(r["end_seconds"])) for r in rows)
    merged: list[list[float]] = []
    for start, end in spans:
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
            raise ValueError("Service intervals must have finite causal clocks")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _rate(count: int, seconds: float) -> float | None:
    return count / seconds if count > 0 and seconds > 0 else None


def analyze_run(directory: Path, source_hashes: dict[str, str]) -> dict[str, Any]:
    """Describe measured rates and whether a later source generation was trained.

    Generator and recording rates use accepted episodes per respective busy
    interval union. The hot learner includes parent batch preparation, transfer
    and synchronized updates, excluding quota/data waits. These independent
    service times are not additive wall-clock components.
    """
    directory = Path(directory)
    summary = _read(directory / "summary.json")
    config, metrics = summary["config"], summary.get("training_metrics", {})
    event_path = directory / "training-events.jsonl"
    events = (
        [json.loads(s) for s in event_path.read_text().splitlines() if s]
        if event_path.is_file()
        else []
    )
    updates = [row for row in events if row.get("kind") == "update_completed"]
    intervals = metrics.get("intervals", [])
    generation = [row for row in intervals if row.get("name") == "generation"]
    generation_ids = {row.get("generation") for row in generation}
    accepted = [
        row
        for row in metrics.get("episodes", [])
        if row.get("generation") in generation_ids
    ]
    recording = [
        row
        for row in intervals
        if row.get("name") == "sdk_recording"
        and row.get("sdk_phase") == "sdk_episode_write_encode"
    ]
    hot = [
        row
        for row in intervals
        if row.get("name") in {"batch_service", "device_transfer", "training"}
    ]
    generation_seconds, recording_seconds, hot_seconds = (
        _duration(rows) for rows in (generation, recording, hot)
    )
    train_seconds = sum(
        float(row["time_seconds"]) - float(row["train_started_seconds"])
        for row in updates
    )
    all_generation_rate = _rate(len(accepted), generation_seconds)
    later_generation = [
        row
        for row in generation
        if isinstance(row.get("generation"), int) and row["generation"] > 1
    ]
    later_ids = {row["generation"] for row in later_generation}
    later_accepted = [row for row in accepted if row["generation"] in later_ids]
    later_seconds = _duration(later_generation)
    later_rate = _rate(len(later_accepted), later_seconds)
    gamma = later_rate if later_rate is not None else all_generation_rate
    recording_rate, hot_rate = _rate(len(recording), recording_seconds), _rate(
        len(updates), hot_seconds
    )
    effective = min(gamma, recording_rate) if gamma and recording_rate else None
    batch, cap = config.get("batch_size"), config.get("episode_budget")
    supply = (
        effective * cap / batch
        if effective
        and type(batch) is int
        and batch > 0
        and type(cap) is int
        and cap > 0
        else None
    )
    later_updates = sum(
        any(c.get("generation", 0) > 1 for c in row.get("chunks", []))
        for row in updates
    )
    trained_generations = sorted(
        {c["generation"] for row in updates for c in row.get("chunks", [])}
    )
    reasons = []
    if (
        summary.get("status") != "complete"
        or summary.get("qualification_passed") is not True
    ):
        reasons.append("run did not complete its engineering qualification")
    if len(updates) != metrics.get("completed_updates"):
        reasons.append("completed-update event count disagrees with summary")
    if not (1 in trained_generations and any(g > 1 for g in trained_generations)):
        reasons.append("training did not consume both initial and later generations")
    if not later_generation:
        reasons.append("no measured post-initial generation service")
    if later_updates < 8:
        reasons.append(
            "fewer than eight updates consumed later-generation data; steady-state evidence is short"
        )
    if any(value is None for value in (gamma, recording_rate, hot_rate)):
        reasons.append(
            "generator, SDK recording or hot learner service evidence is missing"
        )
    if len(recording) != metrics.get("published_sources") or len(
        accepted
    ) != metrics.get("published_sources"):
        reasons.append("service spans do not cover every accepted episode")
    for name in ("batch_service", "device_transfer", "training"):
        if sum(row.get("name") == name for row in hot) != len(updates):
            reasons.append(f"{name} service count differs from completed updates")
    pair = summary.get("device_supervision", {}).get("primary_pair", {})
    hardware = {
        "simulation_uuid": pair.get("simulation", {}).get("uuid"),
        "training_uuid": pair.get("training", {}).get("uuid"),
        "cpu_affinity": pair.get("cpu_affinity"),
        "auxiliary_graphics": (
            {
                key: summary.get("device_supervision", {})
                .get("aux_graphics_scope", {})
                .get(key)
                for key in ("uuid", "scope", "max_per_pid_bytes", "idle_required")
            }
            if isinstance(
                summary.get("device_supervision", {}).get("aux_graphics_scope"), dict
            )
            else None
        ),
    }
    if not all(
        hardware.get(k) for k in ("simulation_uuid", "training_uuid", "cpu_affinity")
    ):
        reasons.append("actual GPU pair or CPU affinity is missing")
    recipe = {
        key: config.get(key)
        for key in (
            "vision",
            "updates",
            "batch_size",
            "buffer_size",
            "episode_budget",
            "history_length",
            "prediction_horizon",
            "action_horizon",
            "max_episode_steps",
            "seed",
        )
    }
    recipe["prefetch_policy"] = config.get("prefetch_policy", "eager")
    recipe["handoff_mode"] = config.get("handoff_mode", "generation")
    recipe["encoder_threads"] = config.get("encoder_threads", 1)
    contract = {"hardware": hardware, "recipe": recipe, "source_sha256": source_hashes}
    implied = (
        math.ceil(hot_rate * batch / effective)
        if hot_rate and effective and type(batch) is int and batch > 0
        else None
    )
    return {
        "vision": config["vision"],
        "status": summary["status"],
        "output": str(directory),
        "producer_envs": config.get("producer_envs"),
        "updates": len(updates),
        "accepted_episode_rate": all_generation_rate,
        "generation_episodes_per_busy_second": gamma,
        "generation_rate_scope": (
            "post-initial generation service"
            if later_rate is not None
            else "initial generation only; not eligible for P selection"
        ),
        "initial_fill_service_seconds": _duration(
            [row for row in generation if row.get("generation") == 1]
        ),
        "later_generation_observed_seconds": later_seconds,
        "later_generation_service_count": len(later_ids),
        "later_generation_accepted_episodes": len(later_accepted),
        "recording_episode_service_count": len(recording),
        "learner_updates_per_second": _rate(len(updates), train_seconds),
        "recording_episodes_per_busy_second": recording_rate,
        "hot_learner_updates_per_second": hot_rate,
        "effective_data_episodes_per_second": effective,
        "supply_updates_per_second": supply,
        "hot_demand_over_supply": hot_rate / supply if hot_rate and supply else None,
        "implied_episode_budget_for_full_hot_utilization": implied,
        "implied_budget_within_declared_cap": (
            implied <= cap if implied is not None and type(cap) is int else None
        ),
        "generation_observed_seconds": generation_seconds,
        "recording_observed_seconds": recording_seconds,
        "training_observed_seconds": train_seconds,
        "hot_learner_observed_seconds": hot_seconds,
        "trained_generations": trained_generations,
        "later_generation_updates": later_updates,
        "later_generation_consumed": any(g > 1 for g in trained_generations),
        "selection_eligible": not reasons,
        "later_generation_evidence_sufficient": 1 in trained_generations
        and any(g > 1 for g in trained_generations)
        and later_updates >= 8
        and bool(later_generation),
        "long_run_stationarity_established": False,
        "selection_exclusion_reasons": reasons,
        "comparison_contract": contract,
        "comparison_contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True).encode()
        ).hexdigest(),
        "limits": "Busy-service estimates, not steady-state end-to-end guarantees. Generation selection uses post-initial services; a default 40-update run may measure only one such generation. Implied K is diagnostic only: never increase a protected reuse cap automatically.",
    }


def _method() -> str:
    return """# Three-stage calibration method

Measure the accepted generator rate γ and the SDK episode write/encode busy
rate ρ independently. Preserve the all-generation rate for compatibility, but
use post-initial generations for effective supply and P selection: the initial
fill includes simulation startup. Report the later-generation service count;
the default 40 updates may observe only one later generation.
 SDK create/finalize and queue waits remain separate
costs. Effective data service is min(γ, ρ). Pure model-update rate is retained;
the hot learner rate ν instead includes batch preparation, transfer and real
synchronized updates, excluding data/quota waits. Merge overlapping intervals
within each service; never sum independent stage times into total wall time.

At fixed batch B and protected lifetime episode budget K, the supply ceiling
is min(γ,ρ) K/B updates/s. Demand/supply is ν divided by that ceiling. The
implied ceil(ν B/min(γ,ρ)) is a diagnostic, not permission or a recommendation
to raise K. A demand/supply ratio above one indicates expected data waiting.

Consume the initial cache and at least eight later-generation updates before
using a candidate for P selection; this is a short service-capacity estimate,
not a steady-state end-to-end throughput guarantee. Explicit shorter calibrations remain visible and are
excluded from selection. With the same hardware, source and recipe, select
the smallest measured P reaching 90% of the best effective data rate. Do not
compare different camera modes, hardware placements or training recipes.
"""


def write_calibration(
    root: Path, results: list[dict[str, Any]], source_hashes: dict[str, str]
) -> dict[str, Any]:
    """Write per-run three-stage observations and their measurement definitions."""
    packet = {
        "schema": "ods-visual-training-calibration/v2",
        "observations": [
            analyze_run(Path(r["output"]), source_hashes) for r in results
        ],
        "source_sha256": source_hashes,
        "limits": "Calibration only; service spans overlap and must not be added as wall time. A short or unqualified calibration never recommends a producer count or a larger K.",
    }
    _write(root / "calibration.json", packet)
    (root / "calibration.md").write_text(_method(), encoding="utf-8")
    return packet


def compare_calibrations(roots: list[Path], output: Path) -> dict[str, Any]:
    """Select the smallest measured P near the best rate only for matching inputs."""
    observations = [
        row
        for root in roots
        for row in _read(Path(root) / "calibration.json")["observations"]
    ]
    selections = []
    for vision in sorted({row["vision"] for row in observations}):
        candidates = [row for row in observations if row["vision"] == vision]
        eligible = [
            row
            for row in candidates
            if row.get("selection_eligible") is True
            and type(row.get("producer_envs")) is int
            and row["producer_envs"] > 0
            and isinstance(row.get("comparison_contract"), dict)
            and row.get("effective_data_episodes_per_second", 0) > 0
        ]
        fingerprints = {
            json.dumps(row["comparison_contract"], sort_keys=True) for row in eligible
        }
        reasons = []
        if not eligible:
            reasons.append("no completed, later-generation, three-stage calibration")
        elif len(fingerprints) != 1:
            reasons.append(
                "hardware, source or training recipe differs; producer counts are not comparable"
            )
        chosen = None
        best = max(
            (row["effective_data_episodes_per_second"] for row in eligible),
            default=None,
        )
        if not reasons:
            qualifying = [
                row
                for row in eligible
                if row["effective_data_episodes_per_second"] >= 0.9 * best
                or math.isclose(
                    row["effective_data_episodes_per_second"], 0.9 * best, rel_tol=1e-12
                )
            ]
            chosen = min(
                qualifying,
                key=lambda row: (
                    row["producer_envs"],
                    -row["effective_data_episodes_per_second"],
                ),
            )
        selections.append(
            {
                "vision": vision,
                "comparable": not reasons,
                "reasons": reasons,
                "best_effective_episode_rate": best if not reasons else None,
                "selected_producer_envs": chosen["producer_envs"] if chosen else None,
                "selected_calibration": chosen["output"] if chosen else None,
                "threshold_fraction": 0.9,
                "candidates": candidates,
            }
        )
    packet = {
        "schema": "ods-three-stage-producer-selection/v1",
        "roots": [str(root) for root in roots],
        "selections": selections,
        "limits": "Selects among measured P only; never changes K or hardware. Repeat the chosen recipe in a complete pilot.",
    }
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "producer-selection.json", packet)
    lines = [
        _method(),
        "",
        "| Vision | Comparable | Selected P | Reason |",
        "| --- | --- | ---: | --- |",
    ]
    lines += [
        f"| {row['vision']} | {row['comparable']} | {row['selected_producer_envs'] if row['selected_producer_envs'] is not None else '—'} | {'; '.join(row['reasons']) or 'smallest measured P at ≥90% of best effective supply'} |"
        for row in selections
    ]
    (output / "producer-selection.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return packet


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare_calibrations(args.roots, args.output), allow_nan=False))


if __name__ == "__main__":
    _main()
