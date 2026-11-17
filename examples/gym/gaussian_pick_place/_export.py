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

"""Export paired native FR3 camera streams without recreating a World."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from embodichain.data_pipeline.recording.media import MediaFrameReader
from _recording import CAMERAS, STEP_DT

__all__ = ["export_video"]


def export_video(episode: Path, metrics: dict[str, Any]) -> None:
    encoders = {}
    thumbnails = {}
    selected = sorted(
        {
            0,
            metrics["max_lift_frame"],
            max(0, metrics["frames"] - 40),
            metrics["frames"] - 1,
        }
    )
    try:
        for name in CAMERAS:
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "960x540",
                "-r",
                "25",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(episode / f"{name}.mp4"),
            ]
            encoders[name] = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.PIPE
            )
        with MediaFrameReader(episode / "media/manifest.json") as reader:
            for index in range(metrics["frames"]):
                _, frames = reader.read(index)
                for name in CAMERAS:
                    rgb = frames[f"{name}/rgb"]
                    encoders[name].stdin.write(rgb.tobytes())
                    if index in selected:
                        tile = Image.fromarray(rgb)
                        tile.save(episode / f"{name}_frame_{index:04d}.png")
                        thumbnails[name, index] = tile.resize((480, 270))
        for encoder in encoders.values():
            encoder.stdin.close()
            error = encoder.stderr.read().decode()
            if encoder.wait() != 0:
                raise RuntimeError(error)
    except BaseException:
        for encoder in encoders.values():
            if encoder.poll() is None:
                encoder.kill()
            encoder.wait()
        raise
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    filters = f"[0:v]pad=960:576:0:36:color=0x141820,drawtext=fontfile={font}:text='Third-person camera':fontcolor=white:fontsize=19:x=12:y=8[a];[1:v]pad=960:576:0:36:color=0x141820,drawtext=fontfile={font}:text='Wrist camera - rigid mount on fr3_hand':fontcolor=white:fontsize=19:x=12:y=8[b];[a][b]hstack=inputs=2[v]"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(episode / "cam_third.mp4"),
            "-i",
            str(episode / "cam_wrist.mp4"),
            "-filter_complex",
            filters,
            "-map",
            "[v]",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(episode / "sensor_full.mp4"),
        ],
        check=True,
    )
    mosaic = Image.new("RGB", (480 * len(selected), 600), (20, 24, 32))
    draw = ImageDraw.Draw(mosaic)
    for column, index in enumerate(selected):
        draw.text(
            (column * 480 + 10, 8), f"s{index}, {index * STEP_DT:.2f}s", fill="white"
        )
        for row, name in enumerate(CAMERAS):
            mosaic.paste(thumbnails[name, index], (column * 480, 30 + row * 300))
            draw.text((column * 480 + 10, 303 + row * 300), name, fill="white")
    mosaic.save(episode / "preview.jpg", quality=92)
