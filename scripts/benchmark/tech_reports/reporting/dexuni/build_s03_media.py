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
"""Build timestamp-aligned S03 comparison videos and response plots.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_s03_media.py --batch PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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
    """Compose the two platform media runs and save simulation-time mappings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Build a scene preview without computing response or timing metrics",
    )
    parser.add_argument("--tasks", nargs="+", choices=("s03",), default=("s03",))
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
            if a["task"] == task
            and "run" in a
            and (
                (a["mode"] == "media" and a.get("success"))
                or (
                    args.video_only
                    and a["mode"] == "preview"
                    and a.get("status") == "completed"
                )
            )
        ]
        if len(selected) != 2:
            raise RuntimeError(f"Need both media runs for {task}")
        platforms = ("embodichain", "isaaclab")
        runs = {a["platform"]: batch / a["run"] for a in selected}
        first_config = json.loads((runs[platforms[0]] / "config.json").read_text())
        cameras = tuple(first_config["measurement"]["cameras"])
        panel_width = 960 if len(cameras) == 1 else 640
        width, height = first_config["measurement"]["camera_resolution"]
        panel_height = round(panel_width * height / width)
        try:
            caption_font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except OSError:
            caption_font = ImageFont.load_default()
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
            "close",
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
                macro_block_size=2,
            ) as writer:
                for frame in range(int(round(duration * 30)) + 1):
                    now = frame / 30
                    canvas = Image.new(
                        "RGB",
                        (len(platforms) * panel_width, len(cameras) * panel_height),
                    )
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
                                Image.new("RGB", (panel_width, panel_height))
                                if index is None
                                else Image.fromarray(
                                    readers[key].get_data(index)
                                ).resize((panel_width, panel_height))
                            )
                            draw = ImageDraw.Draw(panel)
                            data = states[platform]
                            state_index = int(np.argmin(np.abs(data["time"] - now)))
                            phase = (
                                "ended"
                                if index is None
                                else phases[int(data["phase"][state_index])]
                            )
                            draw.rectangle((0, 0, panel_width, 34), fill="black")
                            draw.text(
                                (10, 8),
                                f"{'EmbodiChain' if platform == 'embodichain' else 'Isaac Lab'} | {camera} | t={now:.2f}s | {phase}",
                                fill="white",
                                font=caption_font,
                            )
                            canvas.paste(panel, (col * panel_width, row * panel_height))
                    writer.append_data(np.asarray(canvas))
                    mapping.append(item)
        finally:
            for reader in readers.values():
                reader.close()
        (output / f"{task}_time_mapping.json").write_text(
            json.dumps(mapping, indent=2) + "\n"
        )
        if args.video_only:
            print(
                "Built scene preview video only; no task or performance metrics.",
                flush=True,
            )
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plot_rows = []
        for platform in platforms:
            data = states[platform]
            config = json.loads((runs[platform] / "config.json").read_text())
            t = data["time"]
            corner = data["particles"][:, data["corner_ids"]].mean(axis=1)
            ref = (data["phase"] == 0) & (
                t >= config["measurement"]["reference_window"][0]
            )
            h0 = corner[ref, 2].mean()
            lift = 1000 * (corner[:, 2] - h0)
            height = 1000 * (corner[:, 2] - config["scene"]["table_z"])
            xy = 1000 * np.linalg.norm(corner[:, :2] - corner[0, :2], axis=1)
            axes[0, 0].plot(t, lift, label=platform)
            axes[0, 1].plot(t, height, label=platform)
            axes[1, 0].plot(t, xy, label=platform)
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
                (platform, float(ti), float(li), float(hi), float(di))
                for ti, li, hi, di in zip(t, lift, height, xy)
            ]
        axes[0, 0].axhline(150, color="gray", linestyle="--")
        axes[0, 1].axhline(25, color="gray", linestyle="--")
        for ax, label, title in zip(
            axes.flat,
            (
                "Corner lift (mm)",
                "Corner above table (mm)",
                "Corner XY displacement (mm)",
                "Completed step (ms)",
            ),
            (
                "Lift and 2s hold",
                "Lower and physical release",
                "Corner XY motion",
                "Media diagnostic; excluded from performance samples",
            ),
        ):
            ax.set(xlabel="Simulation time (s)", ylabel=label, title=title)
            ax.legend()
            ax.grid(alpha=0.2)
        fig.suptitle("S03 independent media runs; no pinned nodes or attachment")
        fig.tight_layout()
        fig.savefig(output / "s03_response.png", dpi=160)
        plt.close(fig)
        import csv

        with (output / "s03_response_data.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                (
                    "platform",
                    "simulation_time",
                    "corner_lift_mm",
                    "corner_height_above_table_mm",
                    "corner_xy_displacement_mm",
                )
            )
            writer.writerows(plot_rows)
        print(
            "Built S03 comparison video, timestamp mapping, response plot and plotting CSV",
            flush=True,
        )


if __name__ == "__main__":
    main()
