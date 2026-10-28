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
"""Build a complete, original-speed C02 comparison from existing recordings.

Shorter panels explicitly hold their final image after their recording ends.
This tool never runs simulation or modifies either original platform recording.
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
    """Encode and verify the longest full recording with explicit panel status."""
    work = (
        Path(__file__).resolve().parents[5]
        / "outputs/benchmarks/tech_reports/dexuni/c02_bag_frame_packing/analysis/trajectory_repair"
    )
    root = work.parents[1]
    paths = [root / "platforms" / name for name in ("embodichain", "isaaclab")]
    manifests = [json.loads((p / "manifest.json").read_text()) for p in paths]
    videos = [p / "preview/video.mp4" for p in paths]
    frames = []
    for video in videos:
        with imageio.get_reader(video) as reader:
            assert reader.get_meta_data()["fps"] == 30
            frames.append(reader.count_frames())
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    header = Image.new("RGBA", (1920, 80), (20, 20, 20, 255))
    draw = ImageDraw.Draw(header)
    badge_paths = []
    for i, (label, meta, count) in enumerate(
        zip(("EmbodiChain", "Isaac Lab"), manifests, frames, strict=True)
    ):
        completed = meta["status"] == "completed"
        seconds = meta.get("actual_time_s", meta["actual_steps"] / 60)
        trajectory = (
            "adjusted robot trajectory"
            if meta.get("controller_trajectory")
            else "reference trajectory"
        )
        draw.text(
            (18 + 960 * i, 10), f"{label} | {trajectory}", font=font, fill="white"
        )
        text = (
            f"Completed {seconds:g}s | actual recorded motion"
            if completed
            else f"Failed at {seconds:g}s | final image held afterwards"
        )
        draw.text(
            (18 + 960 * i, 46),
            text,
            font=small,
            fill="#b0e9b0" if completed else "#ffcd80",
        )
        badge = Image.new(
            "RGBA", (960, 90), (20, 45, 20, 245) if completed else (55, 10, 10, 245)
        )
        b = ImageDraw.Draw(badge)
        b.text(
            (18, 10),
            "RECORDING ENDED" if completed else "STOPPED: task check failed",
            font=font,
            fill="white",
        )
        b.text(
            (18, 50),
            f"Held image from {(count-1)/30:.4f}s; no later simulation",
            font=small,
            fill="white",
        )
        target = work / f"panel_{i}_ended.png"
        badge.save(target)
        badge_paths.append(target)
    header_path = work / "full_comparison_header.png"
    header.save(header_path)
    filters = [
        f"[{i}:v]scale=960:540,tpad=stop_mode=clone:stop=-1,setpts=N/(30*TB)[p{i}]"
        for i in range(2)
    ]
    filters += [
        "[p0][p1]hstack=inputs=2:shortest=1,pad=1920:620:0:80[bg]",
        "[bg][2:v]overlay=0:0[v0]",
    ]
    for i, count in enumerate(frames):
        filters.append(
            f"[v{i}][{i+3}:v]overlay={960*i}:80:enable=gte(t\\,{count/30:.8f})[v{i+1}]"
        )
    output = root / "platforms/comparison.mp4"
    temporary = work / "comparison_encoding.mp4"
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y"]
    for source in [*videos, header_path, *badge_paths]:
        command += ["-i", str(source)]
    command += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[v2]",
        "-frames:v",
        str(max(frames)),
        "-r",
        "30",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    with (work / "comparison_encode.log").open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    with imageio.get_reader(temporary) as reader:
        assert reader.count_frames() == max(frames)
        assert reader.get_meta_data()["size"] == (1920, 620)
    temporary.replace(output)
    meta_path = work / "comparison.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        new_frames=max(frames),
        fps=30,
        size=[1920, 620],
        ec_physics_seconds=manifests[0].get(
            "actual_time_s", manifests[0]["actual_steps"] / 60
        ),
        isaac_physics_seconds=manifests[1].get(
            "actual_time_s", manifests[1]["actual_steps"] / 60
        ),
        source_video_frames=frames,
        last_isaac_captured_time=(frames[1] - 1) / 30,
        last_captured_time_s=[(count - 1) / 30 for count in frames],
        held_image_panel_indices=[
            i for i, count in enumerate(frames) if count < max(frames)
        ],
        source_videos={
            str(v.relative_to(root)): hashlib.sha256(v.read_bytes()).hexdigest()
            for v in videos
        },
        padding=(
            "None; both panels contain the same number of actual recorded frames"
            if len(set(frames)) == 1
            else "Each shorter panel explicitly holds its last captured image after recording ends; no physical states or playback speeds modified"
        ),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        new_video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Encoded full comparison: {max(frames)} frames at30fps")


if __name__ == "__main__":
    main()
