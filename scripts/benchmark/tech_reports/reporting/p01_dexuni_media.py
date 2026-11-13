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

"""Package native camera frames and verify their trajectories against P-01 runs."""

from __future__ import annotations

import os

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()


def main() -> None:
    out = ROOT / "reports/P-01/dexuni/native"
    out.mkdir(parents=True, exist_ok=True)
    main_runs = json.loads(
        (ROOT / "configs/P-01/embodichain-dexuni-runs.execution.json").read_text()
    )
    captures = json.loads(
        (ROOT / "configs/P-01/embodichain-dexuni-capture.execution.json").read_text()
    )
    records = []
    for capture in captures:
        assert capture["status"] == "completed"
        case = capture["entry"]["case"]
        run = ROOT / "results/P-01" / capture["run_id"]
        baseline = next(
            e
            for e in main_runs
            if e["entry"]["case"] == case and e["entry"]["dt"] == capture["entry"]["dt"]
        )
        baseline_file = ROOT / "results/P-01" / baseline["run_id"] / "episode_00.npz"
        expected = np.load(baseline_file)
        actual = np.load(run / "episode_00.npz")
        for key in ["t", "state", "body_pose", "com_position"]:
            np.testing.assert_array_equal(actual[key], expected[key])
        frames = json.loads((run / "native_camera/frames.json").read_text())
        for frame in frames["frames"]:
            np.testing.assert_array_equal(
                frame["state"], actual["state"][frame["step"]]
            )
            assert frame["time_s"] == actual["t"][frame["step"]]
        video = out / f"{case}.mp4"
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-framerate",
            "40",
            "-i",
            str(run / "native_camera/frame_%05d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ]
        subprocess.run(command, check=True)
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,width,height,r_frame_rate,duration",
                    "-of",
                    "json",
                    str(video),
                ],
                text=True,
            )
        )["streams"][0]
        assert int(probe["nb_read_frames"]) == len(frames["frames"])
        assert probe["r_frame_rate"] == "40/1"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True
        )
        shutil.copy2(
            run / "native_camera" / frames["frames"][0]["file"], out / f"{case}.png"
        )
        files = [p for p in sorted(run.rglob("*")) if p.is_file()]
        records.append(
            dict(
                case=case,
                run_id=capture["run_id"],
                baseline_run_id=baseline["run_id"],
                arrays_exactly_equal=True,
                first_physics_time_s=frames["frames"][0]["time_s"],
                last_physics_time_s=frames["frames"][-1]["time_s"],
                playback_speed=1,
                encode_command=command,
                probe=probe,
                video=str(video.relative_to(ROOT)),
                video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                raw_files={
                    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files
                },
            )
        )
    (out / "manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    print(
        json.dumps(
            [{k: v for k, v in e.items() if k != "raw_files"} for e in records],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
