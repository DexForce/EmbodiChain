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

"""Compare existing PickPlace run artifacts without launching experiments.

Example:
    python examples/data_pipeline/ods_pick_place_compare.py RUN_ODS RUN_OFFLINE --output REPORT

The same paths may be supplied with ``--runs``. Missing measurements stay null;
T80 ratios are omitted for an entire pairing when any intended pair is censored,
unqualified on the reserved threshold test, or unmatched. This command never
opens test data or selects a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

__all__ = ["load_run", "build_comparison", "write_comparison", "main"]


def _number(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        return float(value)
    return None


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _curve(
    events: list[dict[str, Any]], end: float | None, observed: float | None
) -> dict[str, list[float]] | None:
    if end is None or end < 0:
        return None
    seconds, success = [0.0], [0.0]
    best = 0.0
    results = []
    for event in events:
        if event.get("kind") != "evaluation":
            continue
        elapsed, rate = _number(event.get("seconds")), _number(
            event.get("success_rate")
        )
        if elapsed is None or elapsed < 0 or (rate is not None and not 0 <= rate <= 1):
            raise ValueError("Invalid evaluation time or success fraction in timeline")
        if elapsed > end or (observed is not None and elapsed > observed):
            continue
        results.append((elapsed, rate))
    for elapsed, rate in sorted(results, key=lambda value: value[0]):
        best = max(best, rate) if rate is not None else best
        seconds.append(elapsed)
        success.append(best)
    seconds.append(end)
    success.append(best)
    return {"seconds": seconds, "best_success": success}


def load_run(path: Path | str) -> dict[str, Any]:
    """Read one run, preserving failed/incomplete runs and missing measurements."""
    path = Path(path).resolve()
    summary = _read(path / "summary.json")
    efficiency = _read(path / "efficiency.json") or summary.get("efficiency", {})
    config = _read(path / "config.json")
    timeline = _read(path / "timeline.json")
    data_manifest = _read(path / "data_manifest.json")
    environment = _read(path / "environment.json")
    if not any((summary, efficiency, config)):
        raise FileNotFoundError(f"No run artifacts in {path}")
    seed = summary.get("training_seed", summary.get("seed", config.get("seed")))
    if seed is not None and not isinstance(seed, (int, str)):
        raise ValueError(f"Invalid training seed in {path}")
    ready = efficiency.get("data_ready")
    reached = efficiency.get("efficiency/reached_80")
    if not isinstance(reached, bool):
        reached = None
    t80 = _number(efficiency.get("efficiency/t80_dev_seconds"))
    if reached and (t80 is None or t80 < 0):
        raise ValueError(f"Attainment without a valid T80 in {path}")
    observed = _number(efficiency.get("observed_seconds"))
    horizon = _number(efficiency.get("horizon_seconds"))
    gpu_ids = efficiency.get("cost/allocated_physical_gpu_ids")
    gpu_count = (
        len({str(value) for value in gpu_ids}) if isinstance(gpu_ids, list) else None
    )
    missing = [
        name
        for name in (
            "summary.json",
            "efficiency.json",
            "timeline.json",
            "data_manifest.json",
            "environment.json",
        )
        if not (path / name).exists()
    ]
    events = timeline.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"Timeline events must be a list: {path}")
    return {
        "path": str(path),
        "name": path.name,
        "mode": summary.get("mode", config.get("mode", "unknown")),
        "policy": summary.get("policy", config.get("policy", "unknown")),
        "stage": summary.get("stage", config.get("stage", "unknown")),
        "matching": summary.get("matching"),
        "training_seed": seed,
        "scenario": (
            "data-ready" if ready is True else "cold" if ready is False else "unknown"
        ),
        "smoke": summary.get("smoke", config.get("smoke")),
        "reached_80": reached,
        "t80_dev_seconds": t80 if reached else None,
        "right_censored": None if reached is None else not reached,
        "censor_seconds": _number(efficiency.get("efficiency/censor_seconds")),
        "threshold_test_success": _number(summary.get("threshold_test_success")),
        "final_test_success": _number(
            summary.get(
                "final_success_rate", efficiency.get("quality/test_success_at_budget")
            )
        ),
        "auc_success_time": _number(efficiency.get("efficiency/auc_success_time")),
        "allocated_gpu_hours": _number(efficiency.get("cost/allocated_gpu_hours")),
        "allocated_gpu_count": gpu_count,
        "data_wait_fraction": _number(efficiency.get("pipeline/data_wait_fraction")),
        "host_pss_sampled_peak_bytes": _number(
            efficiency.get(
                "cost/host_pss_sampled_peak_bytes",
                efficiency.get("cost/host_pss_peak_bytes"),
            )
        ),
        "data_disk_peak_bytes": _number(efficiency.get("cost/data_disk_peak_bytes")),
        "observed_seconds": observed,
        "horizon_seconds": horizon,
        "horizon_prespecified": efficiency.get("horizon_prespecified") is True,
        "end_reason": efficiency.get("end_reason"),
        "missing_artifacts": missing,
        "development_curve": (
            _curve(events, horizon if horizon is not None else observed, observed)
            if "timeline.json" not in missing
            else None
        ),
        "config": config,
        "data_manifest": data_manifest,
        "environment": environment,
    }


def _known_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _gpu_models(run: dict[str, Any]) -> list[tuple[str, str]] | None:
    """Compare recorded model/memory/count, ignoring physical index and UUID."""
    if run["allocated_gpu_count"] == 0:
        return []
    hardware = run["environment"].get("allocated_gpu_hardware")
    if not isinstance(hardware, str) or not hardware.strip():
        return None
    rows = list(csv.reader(hardware.splitlines(), skipinitialspace=True))
    if len(rows) != run["allocated_gpu_count"] or any(
        len(row) != 4
        or not row[2].strip()
        or not row[3].strip()
        or any(
            value.strip().lower() in {"unknown", "n/a", "[not supported]"}
            for value in row[2:]
        )
        for row in rows
    ):
        return None
    return sorted((row[2].strip(), row[3].strip()) for row in rows)


def _fingerprint(run: dict[str, Any]) -> dict[str, Any]:
    config, environment = run["config"], run["environment"]
    schema = run["data_manifest"].get("schema", {})
    schema = schema if isinstance(schema, dict) else {}
    hashes = environment.get("source_sha256")
    versions = {
        key: environment.get(key) for key in ("python", "torch", "dexsim", "commit")
    }
    affinity = environment.get("cpu_affinity")
    threads = environment.get("torch_num_threads")
    learning_rate = _number(config.get("learning_rate"))
    result = {
        "learning_rate": (
            learning_rate if learning_rate is not None and learning_rate > 0 else None
        ),
        "pose_ik": (
            config.get("pose_ik")
            if config.get("pose_ik") in {"dls", "solver"}
            else None
        ),
        "source_versions": (
            versions
            if all(
                isinstance(value, str) and value.strip() for value in versions.values()
            )
            else None
        ),
        "source_sha256": (
            hashes
            if isinstance(hashes, dict)
            and hashes
            and all(_known_hash(value) for value in hashes.values())
            else None
        ),
        "gpu_model_memory_count": _gpu_models(run),
        "cpu_affinity": (
            sorted(set(affinity))
            if isinstance(affinity, list)
            and affinity
            and all(type(value) is int and value >= 0 for value in affinity)
            else None
        ),
        "torch_num_threads": threads if type(threads) is int and threads > 0 else None,
    }
    for name in ("observation", "action"):
        component = schema.get(name)
        result[f"{name}_schema"] = (
            component
            if isinstance(component, dict)
            and isinstance(component.get("names"), list)
            and component["names"]
            else None
        )
    return result


def _comparability_notes(runs: list[dict[str, Any]]) -> list[str]:
    """Require recorded evidence rather than equating two missing values."""
    fingerprints = [_fingerprint(run) for run in runs]
    notes = []
    for field in fingerprints[0]:
        missing = [
            run["name"]
            for run, fingerprint in zip(runs, fingerprints)
            if fingerprint[field] is None
        ]
        if missing:
            notes.append(f"Missing {field} fingerprint evidence: {', '.join(missing)}")
        values = {
            json.dumps(fingerprint[field], sort_keys=True)
            for fingerprint in fingerprints
            if fingerprint[field] is not None
        }
        if len(values) > 1:
            notes.append(f"Different {field} fingerprints confound the comparison")
    return notes


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _groups(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[(run["mode"], run["policy"], run["stage"], run["scenario"])].append(run)
    result = []
    for key, members in sorted(grouped.items()):
        entry = dict(zip(("mode", "policy", "stage", "scenario"), key))
        entry.update(
            runs=len(members),
            attained=sum(r["reached_80"] is True for r in members),
            censored=sum(r["right_censored"] is True for r in members),
            unobserved=sum(r["reached_80"] is None for r in members),
        )
        entry["matching_statuses"] = sorted(
            {r["matching"] for r in members if r["matching"] is not None}
        )
        entry["causal_overlap_headline_supported"] = (
            False if key[0] == "serial" else None
        )
        observed = {}
        for metric in (
            "t80_dev_seconds",
            "threshold_test_success",
            "final_test_success",
            "auc_success_time",
            "allocated_gpu_hours",
            "data_wait_fraction",
            "host_pss_sampled_peak_bytes",
            "data_disk_peak_bytes",
        ):
            values = [r[metric] for r in members if r[metric] is not None]
            # T80 medians describe attainers only; failures are retained in n/censored.
            label = (
                "t80_dev_seconds_attainers" if metric == "t80_dev_seconds" else metric
            )
            entry[f"median_{label}"] = _median(values)
            entry[f"range_{label}"] = [min(values), max(values)] if values else None
            observed[metric] = len(values)
        entry["measured_runs"] = observed
        result.append(entry)
    return result


def _paired_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contexts: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        contexts[(run["policy"], run["stage"], run["scenario"])][run["mode"]].append(
            run
        )
    comparisons = []
    for context, modes in sorted(contexts.items()):
        if "online" not in modes:
            continue
        for baseline in sorted(set(modes) - {"online"}):
            both = modes["online"] + modes[baseline]
            failures = []
            notes = []
            confounds = _comparability_notes(both)
            by_mode: dict[str, dict[str, list[dict[str, Any]]]] = {}
            for mode in ("online", baseline):
                mapping: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for run in modes[mode]:
                    if run["training_seed"] is None:
                        failures.append(f"{mode}: missing training seed")
                    else:
                        mapping[str(run["training_seed"])].append(run)
                by_mode[mode] = mapping
            seeds = sorted(set(by_mode["online"]) | set(by_mode[baseline]))
            pairs = []
            for seed in seeds:
                online, offline = by_mode["online"].get(seed, []), by_mode[
                    baseline
                ].get(seed, [])
                if len(online) != 1 or len(offline) != 1:
                    failures.append(f"seed {seed}: requires exactly one run per arm")
                    continue
                pair = {
                    "training_seed": seed,
                    "online": online[0]["path"],
                    "baseline": offline[0]["path"],
                }
                pairs.append(pair)
                for field in ("initial_model_sha256", "normalization_sha256"):
                    values = [
                        run["data_manifest"].get(field)
                        for run in (online[0], offline[0])
                    ]
                    if not all(_known_hash(value) for value in values):
                        confounds.append(
                            f"seed {seed}: missing {field} fingerprint evidence"
                        )
                    elif values[0] != values[1]:
                        confounds.append(
                            f"seed {seed}: different {field} fingerprints confound the comparison"
                        )
                for run in (online[0], offline[0]):
                    if not run["reached_80"] or run["t80_dev_seconds"] <= 0:
                        failures.append(
                            f"{run['name']}: T80 is censored or unavailable"
                        )
                    if (
                        run["threshold_test_success"] is None
                        or run["threshold_test_success"] < 0.8
                    ):
                        failures.append(
                            f"{run['name']}: T80 checkpoint lacks >=80% reserved-test success"
                        )
                for field in ("horizon_seconds", "allocated_gpu_count"):
                    if (
                        online[0][field] is None
                        or offline[0][field] is None
                        or online[0][field] != offline[0][field]
                    ):
                        failures.append(f"seed {seed}: unequal or missing {field}")
                for field in ("batch_size", "updates_per_round", "rounds"):
                    values = [r["config"].get(field) for r in (online[0], offline[0])]
                    if None not in values and values[0] != values[1]:
                        failures.append(f"seed {seed}: unequal {field} caps")
            if not seeds:
                failures.append("No paired training seeds")
            if any(not run["horizon_prespecified"] for run in both):
                failures.append("Wall-time horizon was not prespecified for every run")
            if context[0] == "unknown" or context[2] == "unknown":
                failures.append("Policy or acquisition scenario is unknown")
            if len(seeds) < 5:
                notes.append("Fewer than five paired seeds; preliminary evidence only")
            matching = None
            if baseline == "serial":
                statuses = {run["matching"] for run in modes[baseline]}
                matching = next(iter(statuses)) if len(statuses) == 1 else None
                notes.append(
                    "Serial refresh schedules are unmatched or unverified at each update; observed time ratios are descriptive and cannot establish a causal overlap speedup"
                )
            for run in both:
                config = run["config"]
                if run["smoke"] is not False or run["stage"] != "showcase":
                    notes.append(
                        f"{run['name']}: smoke/diagnostic or unspecified qualification"
                    )
                if any(
                    config.get(field, 0) < required
                    for field, required in (
                        ("qualification_scenes", 40),
                        ("dev_scenes", 20),
                        ("final_test_scenes", 100),
                    )
                ):
                    notes.append(
                        f"{run['name']}: full expert/dev/reserved-test panel sizes not established"
                    )
                if run["final_test_success"] is None or run["final_test_success"] < 0.8:
                    notes.append(
                        f"{run['name']}: final budget-selected policy lacks >=80% test success"
                    )
                if (
                    config.get("no_viewer") is not True
                    or config.get("no_video") is not True
                ):
                    notes.append(
                        f"{run['name']}: disabled visualization/video not established"
                    )
            failures = sorted(set(failures))
            confounds = sorted(set(confounds))
            notes.extend(confounds)
            ratios = None
            if not failures:
                ratios = {
                    seed: by_mode[baseline][seed][0]["t80_dev_seconds"]
                    / by_mode["online"][seed][0]["t80_dev_seconds"]
                    for seed in seeds
                }
            comparisons.append(
                {
                    **dict(zip(("policy", "stage", "scenario"), context)),
                    "baseline_mode": baseline,
                    "reference_mode": "online",
                    "matching": matching,
                    "causal_overlap_headline_supported": (
                        False if baseline == "serial" else None
                    ),
                    "intended_pairs": len(seeds),
                    "complete_pairs": len(pairs),
                    "pairs": pairs,
                    "ratios_supported": not failures,
                    "ratios_descriptive_only": bool(notes),
                    "comparability_supported": not confounds,
                    "comparability_notes": confounds,
                    "ratio_definition": "baseline T80_dev / online T80_dev; above 1 favors online",
                    "ratios_by_seed": ratios,
                    "median_ratio": _median(list(ratios.values())) if ratios else None,
                    "range_ratio": (
                        [min(ratios.values()), max(ratios.values())] if ratios else None
                    ),
                    "headline_efficiency_supported": not failures
                    and not notes
                    and baseline == "offline-full",
                    "omission_reasons": failures,
                    "qualification_notes": sorted(set(notes)),
                }
            )
    return comparisons


def build_comparison(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize all runs; never select an attaining subset for paired ratios."""
    if not runs:
        raise ValueError("At least one run is required")
    if len({r["path"] for r in runs}) != len(runs):
        raise ValueError("Run directories must be distinct")
    comparisons = _paired_comparisons(runs)
    headline = any(value["headline_efficiency_supported"] for value in comparisons)
    warnings = []
    if not headline:
        warnings.append(
            "Preliminary diagnostic comparison: headline efficiency is unsupported. Qualification, complete pairing, target attainment and reserved-test confirmation must be established before an efficiency claim."
        )
    if any(run["missing_artifacts"] for run in runs):
        warnings.append(
            "Some run artifacts are missing; unavailable measurements remain null and missing curves/bars are omitted."
        )
    if any(run["right_censored"] for run in runs):
        warnings.append(
            "Censored runs remain in attainment counts. Reported T80 medians describe attainers only and are not an overall time-to-target estimate."
        )
    if any(run["mode"] == "serial" for run in runs):
        warnings.append(
            "Serial timing is descriptive: unmatched or unverified refresh schedules do not support a causal overlap headline (matching=unmatched_refresh_schedule when declared)."
        )
    if any(not comparison["comparability_supported"] for comparison in comparisons):
        warnings.append(
            "Missing or different experiment fingerprints confound one or more comparisons. Their time ratios are descriptive only; headline efficiency is unsupported for those comparisons."
        )
    return {
        "format": "ods-pick-place-comparison-v1",
        "headline_efficiency_supported": headline,
        "warnings": warnings,
        "runs": runs,
        "groups": _groups(runs),
        "paired_comparisons": comparisons,
        "figures": [],
    }


def _format(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:.1%}" if percent else f"{value:.3g}"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PickPlace ODS/offline comparison",
        "",
        *[f"> {warning}\n" for warning in report["warnings"]],
        "Individual runs (times in seconds; missing values shown as —).",
        "",
        "| Run | Arm / policy | Seed | Scenario | Reached | T80 / censor | Threshold test | Final test | AUC | GPU h | Wait |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in report["runs"]:
        endpoint = (
            _format(run["t80_dev_seconds"])
            if run["reached_80"]
            else (
                "censored @ " + _format(run["censor_seconds"])
                if run["right_censored"]
                else "—"
            )
        )
        reached = (
            "unknown"
            if run["reached_80"] is None
            else "yes" if run["reached_80"] else "no"
        )
        lines.append(
            f"| {run['name']} | {run['mode']} / {run['policy']} | {run['training_seed'] if run['training_seed'] is not None else '—'} | {run['scenario']} | {reached} | {endpoint} | {_format(run['threshold_test_success'], percent=True)} | {_format(run['final_test_success'], percent=True)} | {_format(run['auc_success_time'])} | {_format(run['allocated_gpu_hours'])} | {_format(run['data_wait_fraction'], percent=True)} |"
        )
    lines.extend(
        [
            "",
            "Group medians. T80 is conditional on attainment; the attained/total column includes every supplied run.",
            "",
            "| Arm / policy / scenario | Attained / total | Censored | Unobserved | T80 median (attainers) | Final test median | AUC median |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for group in report["groups"]:
        lines.append(
            f"| {group['mode']} / {group['policy']} / {group['scenario']} | {group['attained']}/{group['runs']} | {group['censored']} | {group['unobserved']} | {_format(group['median_t80_dev_seconds_attainers'])} | {_format(group['median_final_test_success'], percent=True)} | {_format(group['median_auc_success_time'])} |"
        )
    lines.extend(
        [
            "",
            "Paired T80 ratios use baseline/online; values above 1 favor online. No surviving-pair average is produced when any intended pair fails qualification.",
            "",
        ]
    )
    if not report["paired_comparisons"]:
        lines.append(
            "No online/baseline pairs are available within the same policy, stage and acquisition scenario."
        )
    for pair in report["paired_comparisons"]:
        label = (
            f"{pair['baseline_mode']} vs online / {pair['policy']} / {pair['scenario']}"
        )
        if pair["ratios_supported"]:
            qualification = (
                "descriptive only (qualification incomplete or confounded)"
                if pair["ratios_descriptive_only"]
                else "qualified"
            )
            lines.append(
                f"- {label}: {qualification}; median ratio {_format(pair['median_ratio'])}; {pair['complete_pairs']} pairs; individual ratios {json.dumps(pair['ratios_by_seed'], sort_keys=True)}."
            )
        else:
            lines.append(
                f"- {label}: ratios omitted. {'; '.join(pair['omission_reasons'])}."
            )
        if pair["qualification_notes"]:
            lines.append(f"  Qualification: {'; '.join(pair['qualification_notes'])}.")
    if report["figures"]:
        lines.extend(
            [
                "",
                "Figures:",
                "",
                *[f"![{Path(name).stem}]({name})" for name in report["figures"]],
            ]
        )
    return "\n".join(lines) + "\n"


def _plots(report: dict[str, Any], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figures = []
    for gpu_axis, filename in (
        (False, "development_success.png"),
        (True, "allocated_gpu_hours.png"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5))
        count = 0
        for run in report["runs"]:
            curve = run["development_curve"]
            if curve is None or (
                gpu_axis
                and (not run["allocated_gpu_count"] or run["observed_seconds"] is None)
            ):
                continue
            x = curve["seconds"]
            if gpu_axis:
                x = [
                    min(value, run["observed_seconds"])
                    * run["allocated_gpu_count"]
                    / 3600
                    for value in x
                ]
            label = f"{run['mode']} / {run['policy']} / seed {run['training_seed']}"
            line = axis.step(x, curve["best_success"], where="post", label=label)[0]
            if (
                gpu_axis
                and run["final_test_success"] is not None
                and run["allocated_gpu_hours"] is not None
            ):
                axis.scatter(
                    run["allocated_gpu_hours"],
                    run["final_test_success"],
                    marker="x",
                    color=line.get_color(),
                )
            count += 1
        if count:
            axis.axhline(0.8, color="0.4", linestyle="--", linewidth=1)
            axis.set(
                xlabel=(
                    "Allocated physical GPU-hours"
                    if gpu_axis
                    else "Elapsed seconds from recorded scenario start"
                ),
                ylabel="Best available development success",
                ylim=(0, 1.04),
            )
            axis.legend(fontsize="small")
            axis.grid(alpha=0.2)
            figure.suptitle(
                "Preliminary diagnostic"
                if not report["headline_efficiency_supported"]
                else "PickPlace efficiency comparison"
            )
            if gpu_axis:
                figure.text(
                    0.5,
                    0.01,
                    "Curves: development results; ×: reserved final test (verification cost reported separately)",
                    ha="center",
                    fontsize=8,
                )
            figure.tight_layout(rect=(0, 0.04, 1, 0.95))
            figure.savefig(output / filename, dpi=160)
            figures.append(filename)
        plt.close(figure)
    metrics = (
        ("host_pss_sampled_peak_bytes", "Sampled owned-process PSS peak"),
        ("data_disk_peak_bytes", "Persistent training-data peak"),
    )
    if any(run[key] is not None for run in report["runs"] for key, _ in metrics):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, (key, title) in zip(axes, metrics):
            measured = [run for run in report["runs"] if run[key] is not None]
            if measured:
                axis.bar(range(len(measured)), [run[key] / 1024**2 for run in measured])
                axis.set_xticks(
                    range(len(measured)),
                    [f"{r['mode']}\nseed {r['training_seed']}" for r in measured],
                    rotation=30,
                    ha="right",
                )
                axis.set_ylabel("MiB")
            else:
                axis.text(
                    0.5, 0.5, "No measurements", transform=axis.transAxes, ha="center"
                )
            axis.set_title(title)
        figure.tight_layout()
        figure.savefig(output / "resource_peaks.png", dpi=160)
        plt.close(figure)
        figures.append("resource_peaks.png")
    return figures


def write_comparison(run_paths: list[Path | str], output: Path | str) -> dict[str, Any]:
    """Write JSON, Markdown and available static plots for supplied run paths."""
    report = build_comparison([load_run(path) for path in run_paths])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report["figures"] = _plots(report, output)
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (output / "comparison.md").write_text(_markdown(report))
    return report


def main(argv: list[str] | None = None) -> None:
    """Read existing artifacts and produce a conservative comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_paths", nargs="*", type=Path)
    parser.add_argument("--runs", nargs="+", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = args.run_paths + args.runs
    if not paths:
        parser.error("Supply run directories positionally or with --runs")
    report = write_comparison(paths, args.output)
    print(
        f"Wrote {args.output / 'comparison.md'} and {args.output / 'comparison.json'}"
    )
    for warning in report["warnings"]:
        print(warning)


if __name__ == "__main__":
    main()
