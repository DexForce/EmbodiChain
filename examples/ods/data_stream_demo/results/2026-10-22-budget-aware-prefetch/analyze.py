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

"""Rebuild the one-seed scheduling comparison from retained measured summaries.

This reports systems costs and trained coverage, not policy quality. The
separate independent audit reconstructs the original completed-update events.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

__all__ = ["main"]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _runs(root: Path) -> list[dict[str, Any]]:
    from examples.ods.data_stream_demo.training_report import _fairness

    rows = []
    for path in sorted(root.rglob("summary.json")):
        summary = _read(path)
        if "training_metrics" not in summary or "config" not in summary:
            continue
        cfg, metrics = summary["config"], summary["training_metrics"]
        if cfg.get("phase") != "pilot":
            continue
        label = (
            "Offline-32"
            if cfg["arm"] == "offline"
            else (
                "ODS Serial"
                if cfg["arm"] == "ods_serial"
                else (
                    "ODS Eager" if cfg["prefetch_policy"] == "eager" else "ODS Budget"
                )
            )
        )
        row = {
            "vision": cfg["vision"],
            "arm": label,
            "seed": cfg["seed"],
            "producer_envs": cfg["producer_envs"],
            "status": summary["status"],
            "qualification_passed": (
                summary.get("qualification_passed") is True
                and summary.get("post_audit_status") == "complete"
                and summary.get("post_audit", {}).get("passed") is True
            ),
            "path": str(path.parent),
        }
        facts, errors = _fairness(
            {"path": str(path), "summary": summary, "config": cfg, "metrics": metrics}
        )
        row["comparison_evidence"], row["fairness_errors"] = facts, errors
        for key in (
            "total_seconds",
            "first_update_seconds",
            "training_complete_seconds",
            "post_training_drain_seconds",
            "completed_updates",
            "trained_chunks",
            "valid_target_occurrences",
            "published_sources",
            "trained_sources",
            "untrained_sources",
            "effective_sources",
            "initial_xy_occupied_cells",
            "anchor_xyz_occupied_voxels",
            "unique_anchors",
            "unique_target_frames",
            "target_duplicate_fraction",
            "max_source_trained_chunks",
            "dataset_bytes",
        ):
            row[key] = metrics.get(key)
        rows.append(row)
    if len({(r["vision"], r["arm"]) for r in rows}) != len(rows):
        raise ValueError("This one-seed analysis rejects duplicate arm/vision outcomes")
    return rows


def _comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for vision in ("single", "dual"):
        group = {r["arm"]: r for r in rows if r["vision"] == vision}
        current = group.get("ODS Budget")
        for baseline in ("Offline-32", "ODS Eager", "ODS Serial"):
            other = group.get(baseline)
            reasons = []
            eligible = (
                current is not None
                and other is not None
                and current["status"] == other["status"] == "complete"
                and current["qualification_passed"]
                and other["qualification_passed"]
            )
            if not eligible:
                reasons.append("missing complete qualified outcomes")
            else:
                reasons.extend(current["fairness_errors"] + other["fairness_errors"])
                if (
                    current["seed"] != other["seed"]
                    or current["producer_envs"] != other["producer_envs"]
                ):
                    reasons.append("paired seed or producer count differs")
                for key in (
                    "training_budget",
                    "initial_model_sha256",
                    "normalization_sha256",
                    "initial_numeric_prefix",
                    "camera_order",
                    "allocation",
                    "auxiliary_graphics",
                    "requested_evaluation_scenes",
                ):
                    if current["comparison_evidence"].get(key) != other[
                        "comparison_evidence"
                    ].get(key):
                        reasons.append(f"paired {key} differs")
            eligible = eligible and not reasons
            results.append(
                {
                    "vision": vision,
                    "baseline": baseline,
                    "complete_qualified_outcomes": eligible,
                    "exclusion_reasons": reasons,
                    "budget_time_saving_fraction": (
                        1 - current["total_seconds"] / other["total_seconds"]
                        if eligible
                        else None
                    ),
                    "scope": "one seed; inspect strict primary report and independent manifest audit before attribution",
                }
            )
    return results


def _figures(
    rows: list[dict[str, Any]], calibration: list[dict[str, Any]], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Offline-32", "ODS Budget", "ODS Eager", "ODS Serial"]
    colors = ["#748195", "#1b998b", "#dc8a36", "#416aa0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, vision in zip(axes, ("single", "dual")):
        observed = {r["arm"]: r for r in rows if r["vision"] == vision}
        for i, (label, color) in enumerate(zip(labels, colors)):
            r = observed.get(label)
            if r is None or r["total_seconds"] is None:
                continue
            valid = r["status"] == "complete" and r["qualification_passed"]
            ax.bar(
                i, r["total_seconds"], color=color if valid else "none", edgecolor=color
            )
            ax.scatter(
                i,
                r["training_complete_seconds"],
                marker="_",
                color="black",
                s=180,
                zorder=3,
            )
            ax.text(
                i,
                r["total_seconds"] + 4,
                f"{r['trained_sources']}/{r['published_sources']} trained",
                ha="center",
                fontsize=8,
            )
        ax.set_xticks(range(len(labels)), [x.replace(" ", "\n", 1) for x in labels])
        ax.set_ylabel("Complete pipeline (seconds)")
        ax.set_title(f"{vision}: same U128 / B32 / H2 / F32")
        ax.set_ylim(bottom=0)
    fig.suptitle(
        "One-seed scheduling pilot · black mark: final optimizer update · no policy-quality claim",
        fontsize=10,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"scheduling-comparison.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, vision in zip(axes, ("single", "dual")):
        group = sorted(
            (r for r in calibration if r["vision"] == vision),
            key=lambda r: r["producer_envs"],
        )
        for field, label, color in (
            (
                "generation_episodes_per_busy_second",
                "Post-initial generation",
                "#416aa0",
            ),
            ("recording_episodes_per_busy_second", "SDK write/encode", "#dc8a36"),
            (
                "effective_data_episodes_per_second",
                "Effective supply estimate",
                "#1b998b",
            ),
        ):
            eligible = [
                r
                for r in group
                if r.get("selection_eligible") and r.get(field) is not None
            ]
            ax.plot(
                [r["producer_envs"] for r in eligible],
                [r[field] for r in eligible],
                "o-",
                label=label,
                color=color,
            )
        ax.set_xticks([2, 4, 8])
        ax.set_xlabel("Producer environments P")
        ax.set_ylabel("Accepted episodes / busy second")
        ax.set_title(vision)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Short calibration · one later generation per run · service capacities, not pipeline rates",
        fontsize=10,
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"capacity-calibration.{suffix}", dpi=180)
    plt.close(fig)


def main() -> None:
    """Read retained inputs and write derived JSON, CSV and scientific figures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="Raw study root or packaged inputs directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    args.output.mkdir(parents=True, exist_ok=True)
    rows = _runs(args.root / "pilot")
    if not rows:
        raise ValueError("No pilot measurements; refusing to generate an empty report")
    calibration = [
        r
        for p in sorted(args.root.glob("calibration-p*/calibration.json"))
        for r in _read(p)["observations"]
    ]
    _write(
        args.output / "comparison.json",
        {
            "runs": rows,
            "comparisons": _comparisons(rows),
            "limits": "Measured one-seed systems pilot, not a formal aggregate or policy-performance result",
        },
    )
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        fields = [
            key
            for key in rows[0]
            if key not in {"comparison_evidence", "fairness_errors"}
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _figures(rows, calibration, args.output)


if __name__ == "__main__":
    main()
