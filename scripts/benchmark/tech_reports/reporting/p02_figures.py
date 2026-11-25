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

"""Plot continuous P-02 measurements and retain all plotted source arrays."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.analysis / "episode-metrics.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    group_keys = sorted(
        {
            (
                r["platform"],
                r["phase"],
                r["scene"],
                r["dt_s"],
                r["solver"],
                r["duration_s"],
            )
            for r in records
        }
    )
    summaries = []
    for key in group_keys:
        rows = [
            r
            for r in records
            if (
                r["platform"],
                r["phase"],
                r["scene"],
                r["dt_s"],
                r["solver"],
                r["duration_s"],
            )
            == key
        ]
        stat = {}
        for field in rows[0]:
            numbers = [
                r.get(field)
                for r in rows
                if isinstance(r.get(field), (int, float))
                and not isinstance(r.get(field), bool)
            ]
            if numbers:
                stat[field] = dict(
                    mean=float(np.mean(numbers)),
                    min=float(np.min(numbers)),
                    max=float(np.max(numbers)),
                    median=float(np.median(numbers)),
                )
        summaries.append(
            dict(
                platform=key[0],
                phase=key[1],
                scene=key[2],
                dt_s=key[3],
                solver=key[4],
                duration_s=key[5],
                n=len(rows),
                evaluation_role=rows[0]["evaluation_role"],
                run_ids=sorted({r["run_id"] for r in rows}),
                statistics=stat,
                event_count=sum(
                    r.get("geometric_event_first_s") is not None for r in rows
                ),
            )
        )
    (args.analysis / "configuration-summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    colors = {"embodichain": "#146a8b", "isaaclab": "#ce7728"}
    names = {"embodichain": "DexUni, 10 iterations", "isaaclab": "PhysX TGS, 4/1"}
    figures = []

    def save(fig: plt.Figure, name: str, rows: list[dict]) -> None:
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(args.output / f"{name}.{ext}")
        plt.close(fig)
        (args.output / f"{name}-data.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
        figures.append(name)

    sliding = ["slope_30_mu54", "slope_45_mu54", "slope_30_mu32"]
    labels = ["30 deg, mu_d=0.4", "45 deg, mu_d=0.4", "30 deg, mu_d=0.2"]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    used = []
    for platform in colors:
        selected = [
            r
            for r in records
            if r["platform"] == platform
            and r["evaluation_role"] == "primary"
            and r["scene"] in sliding
        ]
        used += selected
        for ax, field in zip(
            axs, ["acceleration_relative_error_pct", "position_max_error_1s_mm"]
        ):
            vals = [
                np.array([r[field] for r in selected if r["scene"] == s])
                for s in sliding
            ]
            means = np.array([v.mean() for v in vals])
            error = np.array(
                [
                    [m - v.min() for m, v in zip(means, vals)],
                    [v.max() - m for m, v in zip(means, vals)],
                ]
            )
            x = np.arange(3) + (-0.18 if platform == "embodichain" else 0.18)
            ax.bar(x, means, width=0.33, color=colors[platform], label=names[platform])
            ax.errorbar(x, means, yerr=error, fmt="none", color="#25343d", capsize=3)
            ax.set_xticks(range(3), labels, rotation=12)
    axs[0].set_ylabel("Fitted acceleration error (%)")
    axs[1].set_ylabel("Max position error, 0-1 s (mm)")
    axs[0].legend(fontsize=9)
    fig.suptitle("Inclined-plane baseline: mean and range across 10 initial states")
    fig.tight_layout()
    save(fig, "slope-errors", used)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    used = []
    for ax, scene in zip(axs, ["slope_15_mu54", "slope_30_mu86"]):
        for j, platform in enumerate(colors):
            rows = [
                r
                for r in records
                if r["platform"] == platform
                and r["evaluation_role"] == "primary"
                and r["scene"] == scene
            ]
            used += rows
            vals = np.array([r["drift_max_mm"] for r in rows])
            ax.scatter(
                np.full(len(vals), j) + np.linspace(-0.08, 0.08, len(vals)),
                vals,
                c=colors[platform],
                s=30,
            )
            ax.plot([j - 0.15, j + 0.15], [vals.mean()] * 2, color="black")
        ax.set_xticks([0, 1], ["DexUni 10", "TGS 4/1"])
        ax.set_ylabel("Max COM displacement, 0-10 s (mm)")
        ax.set_title(
            "15 deg, mu_s/mu_d=0.5/0.4"
            if "15" in scene
            else "30 deg, mu_s/mu_d=0.8/0.6"
        )
    fig.tight_layout()
    save(fig, "static-drift", used)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    used = []
    for platform in colors:
        selected = [
            r
            for r in records
            if r["platform"] == platform
            and r["evaluation_role"] == "primary"
            and r["scene"].startswith("stack")
        ]
        used += selected
        for ax, field in zip(
            axs,
            ["xy_drift_full_max_mm", "tilt_full_max_deg", "penetration_full_max_mm"],
        ):
            vals = [
                np.array([r[field] for r in selected if r["layers"] == n])
                for n in [1, 5, 10]
            ]
            mean = np.array([v.mean() for v in vals])
            err = np.array(
                [
                    [m - v.min() for m, v in zip(mean, vals)],
                    [v.max() - m for m, v in zip(mean, vals)],
                ]
            )
            ax.errorbar(
                [1, 5, 10],
                mean,
                yerr=err,
                marker="o",
                color=colors[platform],
                label=names[platform],
                capsize=3,
            )
            ax.set_xlabel("Dynamic cube layers")
            ax.set_xticks([1, 5, 10])
    for ax, label in zip(
        axs,
        [
            "Max horizontal drift (mm)",
            "Max tilt (deg)",
            "Max geometric penetration (mm)",
        ],
    ):
        ax.set_ylabel(label)
    axs[0].legend(fontsize=8)
    fig.suptitle("Full 0-12 s window, including settling")
    fig.tight_layout()
    save(fig, "stack-baseline", used)
    fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex="col")
    used = []
    for platform in colors:
        for col, scene in enumerate(["stack_05", "stack_10"]):
            r = next(
                r
                for r in records
                if r["platform"] == platform
                and r["evaluation_role"] == "primary"
                and r["scene"] == scene
                and r["initial"] == 0
            )
            used.append(r)
            c = np.load(args.analysis / "curves" / r["run_id"] / "episode_00.npz")
            axs[0, col].plot(
                c["t"],
                c["center_height_m"][:, -1],
                color=colors[platform],
                label=names[platform],
            )
            axs[1, col].plot(
                c["t"], c["penetration_m"].max(axis=1) * 1000, color=colors[platform]
            )
            axs[0, col].set_title(f"{int(scene.split('_')[1])}-layer stack")
    for ax in axs.ravel():
        ax.axvline(2, color="gray", linestyle="--", linewidth=1)
    for ax in axs[0]:
        ax.set_ylabel("Original top cube height (m)")
    for ax in axs[1]:
        ax.set_ylabel("Max penetration (mm)")
        ax.set_xlabel("Time since release (s)")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "stack-traces-initial0", used)
    # Retain all full-window supplementary configurations, with unmodified baselines.
    selected = [
        s
        for s in summaries
        if s["scene"] in ["slope_30_mu54", "stack_10"]
        and (
            (s["scene"] == "slope_30_mu54" and s["phase"] in ["main", "supplement"])
            or (
                s["scene"] == "stack_10"
                and s["phase"] in ["independent", "independent_supplement"]
            )
        )
    ]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, scene, field, label in zip(
        axs,
        ["slope_30_mu54", "stack_10"],
        ["acceleration_relative_error_pct", "xy_drift_full_max_mm"],
        ["Acceleration error (%)", "Full-window horizontal drift (mm)"],
    ):
        rows = [s for s in selected if s["scene"] == scene]
        labels = []
        for i, s in enumerate(rows):
            stat = s["statistics"][field]
            m = stat["mean"]
            ax.errorbar(
                i,
                m,
                yerr=[[m - stat["min"]], [stat["max"] - m]],
                fmt="o",
                capsize=3,
                color=colors[s["platform"]],
            )
            labels.append(
                s["solver"]
                .replace("DexUni / VBD ", "VBD ")
                .replace(" iterations", " it")
                .replace("PhysX ", "")
                + f"\n{s['dt_s']*1000:g} ms"
            )
        ax.set_xticks(range(len(rows)), labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(
            "30° slope, μd = 0.4" if scene.startswith("slope") else "10-layer stack"
        )
    fig.tight_layout()
    save(fig, "configuration-sensitivity", selected)
    (args.output / "plot-config.json").write_text(
        json.dumps(
            dict(
                source=__file__,
                statistics="Across initial states: mean and min-max; not confidence intervals",
                figures=figures,
                formats=["png", "svg", "pdf"],
                colors=colors,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
