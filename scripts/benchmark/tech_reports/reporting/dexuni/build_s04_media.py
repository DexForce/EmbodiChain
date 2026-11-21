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
"""Build the S04 platform comparison video from existing scene previews.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_s04_media.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scripts.benchmark.tech_reports.physics.experiments.dexuni.s04_common import OUTPUT

__all__ = ["main"]


def main() -> None:
    """Align both videos by simulation time and export descriptive state curves."""
    platforms = ("embodichain", "isaaclab")
    readers = {
        p: imageio.get_reader(str(OUTPUT / "preview" / p / "video.mp4"))
        for p in platforms
    }
    videos = OUTPUT / "videos"
    videos.mkdir(exist_ok=True)
    mapping = []
    try:
        for p, reader in readers.items():
            if reader.count_frames() != 361 or reader.get_meta_data()["fps"] != 30:
                raise ValueError(f"Unexpected preview frame cadence: {p}")
        with imageio.get_writer(
            str(videos / "s04_comparison.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for frame in range(361):
                panels = [
                    np.asarray(
                        Image.fromarray(readers[p].get_data(frame)).resize((960, 540))
                    )
                    for p in platforms
                ]
                writer.append_data(np.concatenate(panels, axis=1))
                mapping.append(
                    {
                        "output_frame": frame,
                        "simulation_time": frame / 30,
                        "sources": {
                            p: {"frame": frame, "simulation_time": frame / 30}
                            for p in platforms
                        },
                    }
                )
    finally:
        for reader in readers.values():
            reader.close()
    (videos / "s04_time_mapping.json").write_text(json.dumps(mapping, indent=2))
    rows = []
    for platform in platforms:
        data = np.load(OUTPUT / "preview" / platform / "states.npz")
        for i, time in enumerate(data["time"]):
            rows.append(
                {
                    "platform": platform,
                    "time_s": float(time),
                    "rigid_bottom_m": float(data["rigid_pose"][i, 2, 3]) - 0.045,
                    "soft_lowest_m": float(data["particles"][i, :, 2].min()),
                }
            )
    with (videos / "s04_response_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for platform in platforms:
        selected = [r for r in rows if r["platform"] == platform]
        for axis, key in zip(axes, ["rigid_bottom_m", "soft_lowest_m"]):
            axis.plot(
                [r["time_s"] for r in selected],
                [1000 * (r[key] - 0.86) for r in selected],
                label=platform,
            )
    for axis, title in zip(axes, ["Rigid block bottom", "Soft body lowest node"]):
        axis.set(
            title=title, xlabel="Simulation time (s)", ylabel="Height above table (mm)"
        )
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("S04 scene observations; no performance or quality ranking")
    fig.tight_layout()
    fig.savefig(videos / "s04_response.png", dpi=150)
    plt.close(fig)
    (videos / "s04_media_manifest.json").write_text(
        json.dumps(
            {
                "mode": "scene_preview_only",
                "frames": 361,
                "fps": 30,
                "size": [1920, 540],
                "max_source_time_error_s": 0,
                "platform_preview_approved": False,
                "performance_measurements_started": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f'Platform preview ready: {videos / "s04_comparison.mp4"}')


if __name__ == "__main__":
    main()
