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
"""Combine the actual C02 low/high-budget recordings with EmbodiChain.

Run: .venv/bin/python build_iteration_comparison.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

__all__ = ["main"]


def main() -> None:
    """Join equal-duration real recordings without padding or rerunning physics."""
    work = (
        Path(__file__).resolve().parents[5]
        / "outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing/budget_reduction"
    )
    root = work.parent
    sources = [
        root / "platforms/embodichain",
        work / "i24_s12_video",
        root / "platforms/isaaclab",
    ]
    titles = [
        "EmbodiChain | DexUni 12 x 6",
        "Isaac Lab | VBD 24 x 12",
        "Isaac Lab | VBD 96 x 24",
    ]
    outcomes = [
        "Packing and loaded lift recorded",
        "Bag collapsed during handover",
        "Packing and loaded lift recorded",
    ]
    videos = [p / "preview/video.mp4" for p in sources]
    header = Image.new("RGB", (2880, 100), (20, 20, 20))
    draw = ImageDraw.Draw(header)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    records = []
    for i, (path, video) in enumerate(zip(sources, videos, strict=True)):
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["status"] == "completed" and manifest["actual_steps"] == 3840
        with imageio.get_reader(video) as reader:
            assert reader.count_frames() == 1921 and reader.get_meta_data()["fps"] == 30
        draw.text((18 + 960 * i, 8), titles[i], font=font, fill="white")
        draw.text(
            (18 + 960 * i, 43),
            outcomes[i],
            font=small,
            fill="#ffce8a" if i == 1 else "#b7edb7",
        )
        draw.text(
            (18 + 960 * i, 72),
            "Iterations x substeps | actual motion, 0-64 s",
            font=small,
            fill="#d0d0d0",
        )
        records.append(
            dict(
                path=str(video),
                sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                frames=1921,
                simulation_seconds=64,
                effective=manifest["effective"],
            )
        )
    banner = work / "iteration_comparison_header.png"
    header.save(banner)
    output = work / "iteration_comparison.mp4"
    if output.exists():
        raise FileExistsError(output)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y"]
    for video in videos:
        cmd += ["-i", str(video)]
    cmd += ["-i", str(banner)]
    filters = [f"[{i}:v]scale=960:540,setpts=PTS-STARTPTS[p{i}]" for i in range(3)]
    filters += [
        "[p0][p1][p2]hstack=inputs=3:shortest=1,pad=2880:640:0:100[bg]",
        "[bg][3:v]overlay=0:0[v]",
    ]
    cmd += [
        "-filter_complex_threads",
        "1",
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v]",
        "-frames:v",
        "1921",
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-crf",
        "20",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with (work / "iteration_comparison_encode.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    with imageio.get_reader(output) as reader:
        assert reader.count_frames() == 1921
        assert reader.get_meta_data()["size"] == (2880, 640)
        for second in (10, 24, 58, 64):
            imageio.imwrite(
                work / f"comparison_{second:02d}s.jpg", reader.get_data(second * 30)
            )
    metadata = dict(
        source_videos=records,
        output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        frames=1921,
        fps=30,
        size=[2880, 640],
        held_image_panels=[],
        interpretation="96x24 is the tested budget for the recorded successful flow; no minimum-budget claim. EC strategy and robot trajectories differ as documented.",
    )
    (work / "iteration_comparison.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
