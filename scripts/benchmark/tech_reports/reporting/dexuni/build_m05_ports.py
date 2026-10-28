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
"""Build M05 EmbodiChain physical plots and native/reference comparison media.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_m05_ports.py
Isaac Lab's unsupported gas cavity is reported separately; no timing or ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m05_common import (
    OUTPUT,
    TABLE_Z,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m05_replay import (
    sha256,
    write_json,
)

__all__ = ["main"]


def main() -> None:
    """Check full recorded episodes and produce physical comparison artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run = args.run.resolve()
    ec = run / "preview/embodichain"
    manifest = json.loads((ec / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["frames"] != 2100:
        raise ValueError("Require one completed 35-second EmbodiChain preview")
    # Reuse the native metric definitions through a temporary directory view,
    # preserving its accepted source and reference artifacts byte-for-byte.
    with tempfile.TemporaryDirectory(prefix="m05_port_media_") as scratch:
        stage = Path(scratch)
        shutil.copy2(ec / "manifest.json", stage / "manifest.json")
        (stage / "inputs").symlink_to(run / "inputs", target_is_directory=True)
        (stage / "preview").symlink_to(ec, target_is_directory=True)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("build_m05_media.py")),
                "--run",
                str(stage),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        analysis = ec / "analysis"
        analysis.mkdir(exist_ok=True)
        for path in (stage / "analysis").iterdir():
            shutil.copy2(path, analysis / path.name)
        summary_path = analysis / "scene_summary.json"
        summary = json.loads(summary_path.read_text())
        summary["scope"] = (
            "one completed EmbodiChain physical preview; "
            "no performance samples or success-rate estimate"
        )
        write_json(summary_path, summary)
    rows = {
        "DexSim reference": json.loads((run / "preview/metrics.json").read_text()),
        "EmbodiChain": json.loads((ec / "metrics.json").read_text()),
    }
    data = {}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label, samples in rows.items():
        time = np.array([r["time"] for r in samples])
        if len(time) != 2101 or abs(time[-1] - 35) > 1e-7:
            raise ValueError(f"Incomplete recorded clock for {label}")
        height = np.array([r["bag_height_m"] for r in samples])
        centre = np.array([r["bag_top_centre_z_m"] for r in samples])
        bottom = np.array([r["block_bottom_z_m"] for r in samples])
        data[label.replace(" ", "_").lower() + "_height_m"] = height
        data[label.replace(" ", "_").lower() + "_centre_z_m"] = centre
        data[label.replace(" ", "_").lower() + "_block_bottom_z_m"] = bottom
        for axis, values in zip(axes, (height, centre - TABLE_Z, bottom - TABLE_Z)):
            axis.plot(time, values * 1000, label=label, linewidth=1.5)
    data["time_s"] = time
    for axis, title in zip(
        axes,
        (
            "Pouch envelope height",
            "Upper central material nodes",
            "Cube bottom above table",
        ),
    ):
        axis.set_title(title)
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel("Height (mm)")
        axis.axvline(14.6, color="grey", linewidth=0.8, linestyle="--")
        axis.axvline(28.6, color="grey", linewidth=0.8, linestyle="--")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(run / "analysis/platform_response.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(run / "analysis/platform_plot_data.npz", **data)
    paths = [run / "preview/dexsim.mp4", ec / "video.mp4"]
    readers = [imageio.get_reader(path) for path in paths]
    writer = None
    count = 0
    try:
        for reader in readers:
            if reader.get_meta_data()["fps"] != 30:
                raise ValueError("M05 preview video cadence must be 30fps")
        writer = imageio.get_writer(
            str(run / "preview/comparison.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        )
        for first, second in zip(readers[0], readers[1], strict=True):
            if first.shape != (720, 1280, 3) or second.shape != first.shape:
                raise ValueError("M05 main views must have identical resolution")
            writer.append_data(np.concatenate((first, second), axis=1))
            count += 1
        if count != 1051:
            raise ValueError(f"M05 expected 1051 frames, got {count}")
    finally:
        if writer is not None:
            writer.close()
        for reader in readers:
            reader.close()
    write_json(
        run / "analysis/platform_media.json",
        {
            "platforms": ["accepted native DexSim reference", "EmbodiChain replay"],
            "isaaclab": "unsupported native closed isothermal gas cavity; no video",
            "comparison_sha256": sha256(run / "preview/comparison.mp4"),
            "source_video_sha256": {str(p.relative_to(run)): sha256(p) for p in paths},
            "frames": count,
            "fps": 30,
            "simulation_duration_s": 35,
            "scope": "physical preview only; no performance or quality ranking",
        },
    )
    print(f"M05 current comparison ready: {run / 'preview/comparison.mp4'}")


if __name__ == "__main__":
    main()
