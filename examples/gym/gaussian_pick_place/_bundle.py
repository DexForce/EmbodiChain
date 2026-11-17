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

"""Audit three real task recordings and export complete sensor previews."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from embodichain.data_pipeline.recording.media import MediaFrameReader
from _prepare import SCENES, digest, write_json
from _verify import verify_run
from _recording import CAMERAS

__all__ = ["build_bundle", "main"]


def _video_integrity(episode: Path, frames: int) -> dict[str, Any]:
    results = {}
    for name in CAMERAS:
        video = cv2.VideoCapture(str(episode / f"{name}.mp4"))
        if not video.isOpened() or abs(video.get(cv2.CAP_PROP_FPS) - 25) > 1e-6:
            raise ValueError(f"Invalid sensor video: {name}")
        maximum_error = 0.0
        try:
            with MediaFrameReader(
                episode / "media/manifest.json", fields=(f"{name}/rgb",)
            ) as reader:
                for index in range(frames):
                    success, bgr = video.read()
                    if not success or bgr.shape != (540, 960, 3):
                        raise ValueError(f"Video is missing {name} s{index}")
                    _, raw = reader.read(index)
                    decoded = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    error = float(
                        np.abs(
                            decoded.astype(np.int16)
                            - raw[f"{name}/rgb"].astype(np.int16)
                        ).mean()
                    )
                    maximum_error = max(maximum_error, error)
                    if error > 8:
                        raise ValueError(f"Video differs from raw RGB: {name} s{index}")
                if video.read()[0]:
                    raise ValueError("Video has extra frames")
        finally:
            video.release()
        results[name] = {
            "frames_decoded": frames,
            "max_mean_absolute_compression_error_u8": maximum_error,
        }
    return results


def build_bundle(runs: list[Path], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    selected = {}
    for directory in runs:
        run = json.loads((directory / "run.json").read_text())
        scene = run["scene"]
        if run["robot"]["model"] != "fr3" or run["cameras"] != list(CAMERAS):
            raise ValueError("Expected the FR3 embodiment and both cameras")
        if scene not in SCENES or scene in selected or run["status"] != "complete":
            raise ValueError(
                "Expected exactly one complete run per selected background"
            )
        episode = directory / run["episode"]
        metrics = verify_run(directory)
        steps = metrics["steps"]
        execution = json.loads((episode / "execution_result.json").read_text())
        for segment, kind in zip(
            execution["segments"], ["attach", "release"], strict=True
        ):
            runtime = segment["metadata"]["runtime"]
            effect = runtime["effects"][-1]
            if effect["effect_spec"]["effect_kind"] != kind or effect["decision"][
                "success_mask"
            ] != [True]:
                raise ValueError(f"No measured {kind} evidence: {scene}")
            if effect["monitor"]["monitor_id"] != "builtin.composite_effect":
                raise ValueError("Unexpected effect monitor")
            if any(
                evidence.get("valid_mask") != [True]
                for evidence in effect["evidence"].values()
            ):
                raise ValueError("Physical effect has unavailable evidence")
            held = runtime["task_state"]["held_objects"]
            if kind == "attach" and not any(
                item["object_id"] == "cube" for item in held
            ):
                raise ValueError("Pick did not acquire the dynamic object")
            if kind == "release" and held:
                raise ValueError("Object remains held after Place")
        post = execution["segments"][1]["metadata"]["post_policies"]
        if not post or post[-1]["result"]["status"] != "settled":
            raise ValueError("Placement has no measured settled result")
        integrity = _video_integrity(episode, metrics["frames"])
        shutil.copyfile(episode / "sensor_full.mp4", output / f"{scene}.mp4")
        for camera in CAMERAS:
            shutil.copyfile(episode / f"{camera}.mp4", output / f"{scene}.{camera}.mp4")
        shutil.copyfile(
            episode / "physical_metrics.json", output / f"{scene}.metrics.json"
        )
        selected[scene] = {
            "run": os.path.relpath(directory, output),
            "episode_uuid": episode.name,
            "metrics": metrics,
            "video_integrity": integrity,
            "pick_effect_verified": True,
            "release_effect_verified": True,
            "settled": True,
            "target_center": run["physics"]["target_center"],
            "video": f"{scene}.mp4",
            "raw_rgb_manifest": os.path.relpath(
                episode / "media/manifest.json", output
            ),
        }
        print(
            f"AUDITED {scene}: {steps} actions, {metrics['frames']} RGB frames",
            flush=True,
        )
    if set(selected) != set(SCENES):
        raise ValueError("One or more selected backgrounds are missing")
    names = ["discoverse", "office", "robosimgs"]
    titles = ["DISCOVERSE lab3", "RoboSplatter office", "RoboSimGS tabletop"]
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for name in names:
        base += ["-i", str(output / f"{name}.mp4")]
    filters = []
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    for index, title in enumerate(titles):
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,pad=1920:612:0:36:color=0x141820,drawtext=fontfile={font}:text='{title} - verified physical Pick/Place':fontcolor=white:fontsize=19:x=12:y=8[v{index}]"
        )
    filters.append("[v0][v1][v2]concat=n=3:v=1:a=0[v]")
    options = [
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
    ]
    subprocess.run(
        [
            *base,
            "-filter_complex",
            ";".join(filters),
            *options,
            str(output / "complete_tasks.mp4"),
        ],
        check=True,
    )
    filters = []
    for index, title in enumerate(titles):
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,scale=1280:384,pad=1280:416:0:32:color=0x141820,drawtext=fontfile={font}:text='{title}':fontcolor=white:fontsize=19:x=12:y=8[v{index}]"
        )
    filters.append("[v0][v1][v2]vstack=inputs=3[v]")
    subprocess.run(
        [
            *base,
            "-filter_complex",
            ";".join(filters),
            *options,
            str(output / "comparison.mp4"),
        ],
        check=True,
    )
    collage = Image.new("RGB", (1280, 1188), (20, 24, 32))
    draw = ImageDraw.Draw(collage)
    for index, (name, title) in enumerate(zip(names, titles, strict=True)):
        episode = (
            (output / selected[name]["run"]).resolve()
            / "episodes"
            / selected[name]["episode_uuid"]
        )
        frame = selected[name]["metrics"]["max_lift_frame"]
        for column, camera in enumerate(CAMERAS):
            image = Image.open(episode / f"{camera}_frame_{frame:04d}.png").resize(
                (640, 360)
            )
            collage.paste(image, (640 * column, 396 * index + 36))
            draw.text(
                (640 * column + 12, 396 * index + 10),
                title + " / " + camera,
                fill="white",
            )
    collage.save(output / "preview.png")
    report = {
        "status": "complete",
        "scenes": selected,
        "episodes": 3,
        "total_physical_actions": sum(v["metrics"]["steps"] for v in selected.values()),
        "total_rgb_images": sum(v["metrics"]["rgb_images"] for v in selected.values()),
        "total_paired_observations": sum(
            v["metrics"]["frames"] for v in selected.values()
        ),
        "full_video": "complete_tasks.mp4",
        "comparison": "comparison.mp4",
        "scope": "FR3 Task Program dynamic-cube Pick/Place with third-person and wrist cameras with measured attach/release, stable placement and full RGB. No metric GS depth claim.",
        "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
    }
    write_json(output / "bundle.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_bundle([p.resolve() for p in args.runs], args.output.resolve())
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "status",
                    "episodes",
                    "total_physical_actions",
                    "total_rgb_images",
                    "full_video",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
