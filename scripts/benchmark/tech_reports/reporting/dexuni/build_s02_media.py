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
"""Build timestamp-aligned S02 comparison videos and response plots.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_s02_media.py --batch PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

__all__ = ["frame_index", "main"]


def frame_index(now: float, fps: float, count: int) -> int | None:
    """Map a simulation time to a regular source video frame.

    Args:
        now: Requested simulation time in seconds.
        fps: Source sampling frequency in Hz.
        count: Number of source frames.

    Returns:
        Frame index or None when the source has ended.
    """
    index = int(round(now * fps))
    return index if 0 <= index < count else None


def main() -> None:
    """Render both tasks from exact media attempts and save source mappings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=("s02",), default=("s02",))
    args = parser.parse_args()
    batch = args.batch.resolve()
    attempts = json.loads((batch / "attempts.json").read_text())
    output = batch / "videos"
    output.mkdir(exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for task in args.tasks:
        selected = [
            a
            for a in attempts
            if a["task"] == task and a["mode"] == "media" and "run" in a
        ]
        if len(selected) != 2:
            raise RuntimeError(f"Need both media runs for {task}")
        platforms = ("embodichain", "isaaclab")
        runs = {a["platform"]: batch / a["run"] for a in selected}
        cameras = ("oblique", "front")
        readers, counts, states, manifests, fps = {}, {}, {}, {}, {}
        for platform in platforms:
            path = runs[platform]
            states[platform] = dict(np.load(path / "states.npz"))
            manifests[platform] = json.loads((path / "manifest.json").read_text())
            config = json.loads((path / "config.json").read_text())
            for camera in cameras:
                key = (platform, camera)
                reader = imageio.get_reader(str(path / f"rgb_{camera}.mp4"))
                readers[key] = reader
                counts[key] = reader.count_frames()
                fps[key] = float(reader.get_meta_data()["fps"])
                expected = (
                    int(
                        manifests[platform]["frames"]
                        // (
                            config["trajectory"]["frame_rate"]
                            // config["measurement"]["camera_rate_hz"]
                        )
                    )
                    + 1
                )
                if counts[key] != expected:
                    raise RuntimeError(
                        f"Unexpected source cadence {key}: {counts[key]} vs {expected}"
                    )
        duration = max((counts[k] - 1) / fps[k] for k in readers)
        mapping = []
        phases = (
            "settle",
            "squeeze",
            "squeeze_settle",
            "lift",
            "hold",
            "transport",
            "lower",
            "release",
            "retreat",
            "observe",
            "done",
        )
        try:
            with imageio.get_writer(
                str(output / f"{task}_comparison.mp4"),
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=16,
            ) as writer:
                for frame in range(int(round(duration * 30)) + 1):
                    now = frame / 30
                    canvas = Image.new("RGB", (1280, 960))
                    item = {
                        "output_frame": frame,
                        "simulation_time": now,
                        "sources": {},
                    }
                    for col, platform in enumerate(platforms):
                        for row, camera in enumerate(cameras):
                            key = (platform, camera)
                            index = frame_index(now, fps[key], counts[key])
                            item["sources"][f"{platform}/{camera}"] = {
                                "frame": index,
                                "simulation_time": (
                                    None if index is None else index / fps[key]
                                ),
                            }
                            panel = (
                                Image.new("RGB", (640, 480))
                                if index is None
                                else Image.fromarray(
                                    readers[key].get_data(index)
                                ).resize((640, 480))
                            )
                            draw = ImageDraw.Draw(panel)
                            data = states[platform]
                            state_index = int(np.argmin(np.abs(data["time"] - now)))
                            phase = (
                                "ended"
                                if index is None
                                else phases[int(data["phase"][state_index])]
                            )
                            draw.rectangle((0, 0, 640, 34), fill="black")
                            draw.text(
                                (10, 8),
                                f"{task.upper()} | {platform} | {camera} | t={now:.3f}s | {phase}",
                                fill="white",
                            )
                            canvas.paste(panel, (col * 640, row * 480))
                    writer.append_data(np.asarray(canvas))
                    mapping.append(item)
        finally:
            for reader in readers.values():
                reader.close()
        (output / f"{task}_time_mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plot_rows = []
        for platform in platforms:
            data, manifest = states[platform], manifests[platform]
            config = json.loads((runs[platform] / "config.json").read_text())
            t = data["time"]
            compression = 100 * (1 - data["width"] / manifest["controller"]["w0"])
            zmin = 1000 * (
                data["particles"][:, :, 2].min(axis=1) - config["scene"]["table_z"]
            )
            center = data["particles"].mean(axis=1)
            error = 1000 * np.linalg.norm(
                center[:, :2] - np.asarray(config["target_area"]["center"]), axis=1
            )
            axes[0, 0].plot(t, compression, label=platform)
            axes[0, 1].plot(t, zmin, label=platform)
            axes[1, 0].plot(t, error, label=platform)
            timing = np.genfromtxt(
                runs[platform] / "timing.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding=None,
            )
            axes[1, 1].plot(
                timing["sim_time"][30:],
                1000 * timing["step_seconds"][30:],
                label=platform,
            )
            plot_rows += [
                (platform, float(ti), float(ci), float(zi), float(ei))
                for ti, ci, zi, ei in zip(t, compression, zmin, error)
            ]
        axes[0, 0].axhspan(30, 40, color="gray", alpha=0.15)
        axes[0, 1].axhline(80, color="gray", linestyle="--")
        axes[1, 0].axhline(20, color="gray", linestyle="--")
        for ax, ylabel, title in zip(
            axes.flat,
            (
                "Width compression (%)",
                "Lowest-node height (mm)",
                "Placement XY error (mm)",
                "Completed step (ms)",
            ),
            (
                "Grasp and recovery",
                "Lift, hold and lower",
                "Transport and placement",
                "Media-run timing; excluded from main samples",
            ),
        ):
            ax.set(xlabel="Simulation time (s)", ylabel=ylabel, title=title)
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle("S02 independent media runs; raw curves retained")
        fig.tight_layout()
        fig.savefig(output / "s02_response.png", dpi=160)
        plt.close(fig)
        import csv

        with (output / "s02_response_data.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "platform",
                    "simulation_time",
                    "compression_pct",
                    "lowest_height_mm",
                    "placement_xy_error_mm",
                )
            )
            writer.writerows(plot_rows)
        print(
            "Built S02 video, timestamp mapping, response plot and plotting CSV",
            flush=True,
        )


if __name__ == "__main__":
    main()
