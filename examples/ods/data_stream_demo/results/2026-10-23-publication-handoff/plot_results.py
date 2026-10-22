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

"""Compare visual BC diversity and complete-pipeline time at a fixed update budget.

Run ``python -m examples.ods.data_stream_demo.train --help``. The supervisor
selects physical GPU UUIDs before launching the real acquisition/learner worker.
Every arm records standard LeRobot data; offline training really decodes it.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__: list[str] = []


def main() -> None:
    """Render frozen audit values without accessing simulator or raw corpora."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(__file__).resolve().parent
    audit = json.loads((root / "independent-audit.json").read_text())
    assert audit["passed"] and len(audit["runs"]) == 10
    rows = {(r["vision"], r["label"]): r["metrics"] for r in audit["runs"]}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    labels = ["Publication", "Generation", "Serial", "Offline-32", "Eager"]
    modes = ["ods_overlap", "generation_control", "ods_serial", "offline32", "eager"]
    colors = ["#167d9a", "#828b94", "#7f64a3", "#d89031", "#bb5963"]
    for ax, vision in zip(axes[:2], ("single", "dual")):
        values = [rows[(vision, mode)]["total_seconds"] for mode in modes]
        bars = ax.barh(labels, values, color=colors)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
        ax.set_xlim(0, max(values) * 1.16)
        ax.set_xlabel("Complete pipeline time (s)")
        ax.set_title(f"{vision.title()} view: 128 actual updates")
    ax = axes[2]
    x = np.arange(4)
    for offset, mode, label, color in (
        (-0.18, "ods_overlap", "Publication", colors[0]),
        (0.18, "generation_control", "Generation", colors[1]),
    ):
        values = [
            rows[(vision, mode)]["post_initial_publication_to_first_training_seconds"][
                q
            ]
            for vision in ("single", "dual")
            for q in ("p50", "p95")
        ]
        bars = ax.bar(x + offset, values, 0.36, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, rotation=90, padding=3)
    ax.set_xticks(x, ["Single p50", "Single p95", "Dual p50", "Dual p95"], rotation=25)
    ax.set_ylim(0, 59)
    ax.set_ylabel("Publish to first completed update (s)")
    ax.set_title("Post-initial trained sources only")
    ax.legend(loc="upper left")
    fig.suptitle("One seed; identical update budget; no policy-quality claim")
    for suffix in ("png", "svg", "pdf"):
        target = root / f"publication-comparison.{suffix}"
        fig.savefig(target, dpi=180)
        if suffix == "svg":
            target.write_text(
                "".join(
                    line.rstrip() + "\n" for line in target.read_text().splitlines()
                )
            )
    plt.close(fig)


if __name__ == "__main__":
    main()
