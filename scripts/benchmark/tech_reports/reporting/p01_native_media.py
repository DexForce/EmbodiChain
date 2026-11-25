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

"""Verify P-01 recording runs and assemble native media for the Markdown draft."""

from __future__ import annotations

import os

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
OUT = ROOT / "reports/P-01/native"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = {"source": "Live Isaac Lab Camera / IsaacRtxRenderer", "runs": {}}
    for case, dt, shot in [("freefall", "0025", 20), ("pendulum", "005", 0)]:
        run = ROOT / f"results/P-01/20261014_capture_{case}_dt{dt}_01"
        baseline = ROOT / f"results/P-01/20261014_main_{case}_dt{dt}"
        recording = json.loads((run / "run.json").read_text())
        original = json.loads((baseline / "run.json").read_text())
        assert recording["status"] == "completed"
        a, b = np.load(run / "episode_00.npz"), np.load(baseline / "episode_00.npz")
        differences = {}
        for key in a.files:
            differences[key] = float(np.max(np.abs(a[key] - b[key])))
            assert np.array_equal(a[key], b[key]), (case, key, differences[key])
        config_keys = [
            k
            for k in original
            if k.startswith("effective_body_") or k.startswith("effective_joint_")
        ] + [
            "dt_s",
            "duration_s",
            "position_iterations",
            "velocity_iterations",
            "solver",
            "isaaclab_commit",
            "packages",
        ]
        for key in config_keys:
            assert recording[key] == original[key], (case, key)
        source = run / "native_camera"
        capture = json.loads((source / "capture.json").read_text())
        frame_records = []
        for frame in capture["frames"]:
            assert np.array_equal(frame["state"], a["state"][frame["step"]])
            assert abs(frame["time_s"] - a["t"][frame["step"]]) < 1e-12
            path = source / frame["file"]
            rgb = np.asarray(Image.open(path)).astype(float)
            mask = (
                (rgb[..., 0] < 100)
                & (rgb[..., 2] > rgb[..., 0] + 50)
                & (rgb[..., 1] > rgb[..., 0] + 30)
            )
            y, x = np.nonzero(mask)
            assert len(x) > 100, (case, frame["file"], "body not visible")
            eye, target = np.array(frame["camera_eye"]), np.array(
                frame["camera_target"]
            )
            forward = target - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            pos = (
                a["com_position"][frame["step"]]
                if case == "pendulum"
                else np.array([0, 0, frame["state"][0]])
            )
            rel = pos - eye
            projected = np.array([640, 360]) + (1280 * 40 / 36) * np.array(
                [right @ rel, -up @ rel]
            ) / (forward @ rel)
            error = float(np.linalg.norm(projected - [x.mean(), y.mean()]))
            assert error < 8, (case, frame["file"], error)
            frame_records.append(
                {
                    "file": frame["file"],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "projected_center_error_px": error,
                }
            )
        shutil.copy2(source / "native_1x.mp4", OUT / f"{case}_1x.mp4")
        screenshot = f"{case}_t{'05025' if case == 'freefall' else '0005'}.png"
        shutil.copy2(source / capture["frames"][shot]["file"], OUT / screenshot)
        # GIFs are reduced in resolution/frame rate, with no synthetic frames or overlays.
        speed_filter = "setpts=4*PTS,fps=10" if case == "freefall" else "fps=20"
        gif = f"{case}_{'quarter_speed' if case == 'freefall' else '1x'}.gif"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(OUT / f"{case}_1x.mp4"),
                "-filter_complex",
                f"{speed_filter},scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=bayer",
                "-loop",
                "0",
                str(OUT / gif),
            ],
            check=True,
        )
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=width,height,r_frame_rate,nb_frames,duration",
                    "-of",
                    "json",
                    str(OUT / f"{case}_1x.mp4"),
                ]
            )
        )
        manifest["runs"][case] = {
            "run": str(run.relative_to(ROOT)),
            "baseline": str(baseline.relative_to(ROOT)),
            "exact_array_equality": True,
            "max_abs_differences": differences,
            "matching_config_keys": config_keys,
            "capture": {k: v for k, v in capture.items() if k != "frames"},
            "screenshot": {"file": screenshot, "source": capture["frames"][shot]},
            "video_probe": probe,
            "raw_frames": frame_records,
            "max_projected_center_error_px": max(
                r["projected_center_error_px"] for r in frame_records
            ),
            "input_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    run / "episode_00.npz",
                    baseline / "episode_00.npz",
                    run / "run.json",
                    source / "capture.json",
                ]
            },
        }
    manifest["outputs_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.iterdir())
        if p.suffix in {".png", ".gif", ".mp4"}
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {
                    "exact_array_equality": v["exact_array_equality"],
                    "max_projected_center_error_px": v["max_projected_center_error_px"],
                }
                for k, v in manifest["runs"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
