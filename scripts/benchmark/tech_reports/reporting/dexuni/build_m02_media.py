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
"""Build the M02 EmbodiChain / Isaac Lab comparison video and fold curves.

Once the formal measurement exists, each panel shows the live simulation rate:
the 1 s rolling mean of the measured step time (mean of the five episodes).

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_m02_media.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m02_common import (
    OUTPUT,
    TABLE_Z,
)

__all__ = ["main"]
PLATFORMS = ("embodichain", "isaaclab")
LABELS = {
    "embodichain": "EmbodiChain / DexUni surface-fast",
    "isaaclab": "Isaac Lab / Newton VBD",
}
WINDOW = 60  # steps (1 s) in the rolling mean


def measured_step_ms(steps: int) -> dict[str, np.ndarray] | None:
    """Per-step measured time (ms), mean of all episodes, 1 s rolling mean.

    Index i is the step that ends at simulation time i * dt (i = 1..steps);
    index 0 repeats step 1. Returns None before the formal measurement.
    """
    path = OUTPUT / "measurement/results/steps.csv"
    if not path.exists():
        return None
    table: dict[str, dict[str, list[float]]] = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            table.setdefault(row["platform"], {}).setdefault(row["repeat"], []).append(
                float(row["step_s"])
            )
    result = {}
    for platform in PLATFORMS:
        runs = np.asarray(list(table[platform].values()))
        if runs.shape[1] != steps:
            raise ValueError(f"Unexpected measured step count: {platform}")
        mean = 1e3 * runs.mean(axis=0)
        kernel = np.ones(WINDOW) / WINDOW
        padded = np.pad(mean, (WINDOW // 2, WINDOW - WINDOW // 2 - 1), mode="edge")
        rolling = np.convolve(padded, kernel, mode="valid")
        result[platform] = np.r_[rolling[0], rolling]
    return result


def main() -> None:
    """Align both previews by simulation time and export the fold curves."""
    config = json.loads((OUTPUT / "inputs/m02_config.json").read_text())
    frames = config["steps"] // 2 + 1  # previews are written every second step
    videos = OUTPUT / "videos"
    videos.mkdir(exist_ok=True)
    readers = {
        p: imageio.get_reader(str(OUTPUT / "preview" / p / "video.mp4"))
        for p in PLATFORMS
    }
    mapping = []
    rates = measured_step_ms(config["steps"])
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    try:
        for p, reader in readers.items():
            if reader.count_frames() != frames or reader.get_meta_data()["fps"] != 30:
                raise ValueError(f"Unexpected preview frame cadence: {p}")
        with imageio.get_writer(
            str(videos / "m02_comparison.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for frame in range(frames):
                panels = []
                for p in PLATFORMS:
                    image = Image.fromarray(readers[p].get_data(frame)).resize(
                        (1280, 720)
                    )
                    if rates is not None:
                        ms = float(rates[p][min(2 * frame, config["steps"])])
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 670, 1280, 720), fill="black")
                        draw.text(
                            (14, 680),
                            f"{LABELS[p]}  |  {1e3 / ms:6.1f} steps/s  ({ms:5.2f} ms/step, measured)",
                            font=font,
                            fill=(255, 220, 80),
                        )
                    panels.append(np.asarray(image))
                writer.append_data(np.concatenate(panels, axis=1))
                mapping.append(
                    {
                        "output_frame": frame,
                        "simulation_time": frame / 30,
                        "sources": {
                            p: {"frame": frame, "simulation_time": frame / 30}
                            for p in PLATFORMS
                        },
                    }
                )
    finally:
        for reader in readers.values():
            reader.close()
    (videos / "m02_time_mapping.json").write_text(json.dumps(mapping, indent=2))
    rows = []
    for platform in PLATFORMS:
        for item in json.loads(
            (OUTPUT / "preview" / platform / "metrics.json").read_text()
        ):
            rows.append(
                {
                    "platform": platform,
                    "time_s": item["time"],
                    "left_xy_error_mm": 1e3 * item["left_xy_error_m"],
                    "right_xy_error_mm": 1e3 * item["right_xy_error_m"],
                    "highest_node_mm": 1e3 * (item["highest_z_m"] - TABLE_Z),
                    "lowest_node_mm": 1e3 * (item["lowest_z_m"] - TABLE_Z),
                }
            )
    with (videos / "m02_response_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = (
        "left_xy_error_mm",
        "right_xy_error_mm",
        "highest_node_mm",
        "lowest_node_mm",
    )
    titles = (
        "Left corner pair XY distance",
        "Right corner pair XY distance",
        "Highest node above table",
        "Lowest node above table",
    )
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    for platform in PLATFORMS:
        selected = [r for r in rows if r["platform"] == platform]
        for axis, key in zip(axes, keys):
            axis.plot(
                [r["time_s"] for r in selected],
                [r[key] for r in selected],
                label=platform,
            )
    for axis, title in zip(axes, titles):
        axis.set(title=title, xlabel="Simulation time (s)", ylabel="mm")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(
        "M02 scene observations (preview run); no performance or quality ranking"
    )
    fig.tight_layout()
    fig.savefig(videos / "m02_response.png", dpi=150)
    plt.close(fig)
    (videos / "m02_media_manifest.json").write_text(
        json.dumps(
            {
                "mode": "scene_preview_only",
                "frames": frames,
                "fps": 30,
                "size": [2560, 720],
                "rate_overlay": (
                    None
                    if rates is None
                    else "1 s rolling mean of measured completed outer-step time, mean of the five formal episodes per platform (measurement/results/steps.csv); not the video playback rate"
                ),
                "max_source_time_error_s": 0,
                # The user approved the side-by-side preview before measurement;
                # this rebuild adds only the measured-rate overlay to the same frames.
                "platform_preview_approved": True,
                "approved_preview_sha256": (
                    json.loads((OUTPUT / "measurement/protocol.json").read_text())[
                        "approved_preview_sha256"
                    ]
                    if (OUTPUT / "measurement/protocol.json").exists()
                    else None
                ),
                "performance_measurements_started": rates is not None,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Platform preview ready: {videos / 'm02_comparison.mp4'}")


if __name__ == "__main__":
    main()
