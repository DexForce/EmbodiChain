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

"""Verify and package P-01 native videos; no reconstructed simulation frames."""

from __future__ import annotations

import os

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from scipy.ndimage import label as connected_components, binary_opening

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
OUT = ROOT / "reports/P-01/common-view"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cfg_path = ROOT / "assets/P-01/common-view.json"
    cfg = json.loads(cfg_path.read_text())
    selected = json.loads(
        (ROOT / "configs/P-01/common-view-v2-runs.execution.json").read_text()
    )[:2]
    selected += json.loads(
        (ROOT / "configs/P-01/common-view-v3-runs.execution.json").read_text()
    )
    assert len(selected) == 6 and all(e["status"] == "completed" for e in selected)
    quality = json.loads((ROOT / "reports/P-01/comparison/summary.json").read_text())
    frames_by_case, entries, commands, hashes = {}, [], [], {}
    for entry in selected:
        case, solver, platform = entry["case"], entry["solver"], entry["platform"]
        label = "DexUni" if solver == "DexUni" else f"{solver} 4/1"
        baseline = next(
            s["run_id"]
            for s in quality
            if (s["solver"], s["case"], s["dt_s"]) == (label, case, 0.005)
        )
        run = ROOT / "results/P-01" / entry["run_id"]
        original = ROOT / "results/P-01" / baseline / "episode_00.npz"
        source = run / "episode_00.npz"
        a, b = np.load(source), np.load(original)
        equal = {k: bool(np.array_equal(a[k], b[k])) for k in a.files if k in b.files}
        assert all(equal.values()), (run, equal)
        assert len(a["t"]) == 2001 and a["t"][-1] == 10
        capture_path = run / "native_camera/capture.json"
        capture = json.loads(capture_path.read_text())
        assert capture["common_config"] == cfg
        frames = capture["frames"]
        count = 40 if case == "freefall" else 400
        assert len(frames) == count
        times = np.array([f["time_s"] for f in frames])
        np.testing.assert_allclose(
            times, np.arange(1, count + 1) / 40, rtol=0, atol=1e-12
        )
        if case in frames_by_case:
            np.testing.assert_array_equal(times, frames_by_case[case])
        else:
            frames_by_case[case] = times
        projection_errors = []
        camera_errors = []
        pixel_checks = []
        expected_k = np.array(
            [[1400.0, 0.0, 640.0], [0.0, 1400.0, 360.0], [0.0, 0.0, 1.0]]
        )
        mislabeled = False
        for frame in frames:
            i = frame["step"]
            np.testing.assert_array_equal(frame["state"], a["state"][i])
            eye = np.array(cfg[case]["eye"], dtype=float)
            target = np.array(cfg[case]["target"], dtype=float)
            if case == "freefall":
                z = (
                    cfg["freefall_z0_m"]
                    - 0.5 * cfg["gravity_m_s2"] * frame["time_s"] ** 2
                )
                eye[2] += z
                target[2] += z
            np.testing.assert_allclose(eye, frame["camera_eye"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(
                target, frame["camera_target"], rtol=0, atol=1e-12
            )
            np.testing.assert_array_equal(expected_k, frame["intrinsics"])
            expected = np.eye(4)
            expected[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
            expected[:3, 3] = eye
            if platform == "embodichain":
                camera = np.array(frame["camera_world_opengl"])
            else:
                camera = np.eye(4)
                camera[:3, 3] = frame["camera_pos_readback"]
                # This installed Isaac Lab API returns xyzw. Two early captures
                # used a wrong key suffix; retain raw metadata and interpret the
                # stored values according to the API, documented in the manifest.
                q = frame.get("camera_quat_opengl_xyzw")
                if q is None:
                    q = frame["camera_quat_opengl_wxyz"]
                    mislabeled = True
                camera[:3, :3] = Rotation.from_quat(q).as_matrix()
            error = float(np.max(np.abs(camera - expected)))
            camera_errors.append(error)
            assert error < 1e-6
            state = a["state"][i]
            point = (
                np.array([0.0, -0.025, state[0]])
                if case == "freefall"
                else np.array(
                    [-0.5 * np.sin(state[0]), 0.0, 1 - 0.5 * np.cos(state[0])]
                )
            )
            local = camera[:3, :3].T @ (point - camera[:3, 3])
            uv = np.array(
                [
                    640 + 1400 * local[0] / (-local[2]),
                    360 - 1400 * local[1] / (-local[2]),
                ]
            )
            fpath = run / "native_camera" / frame["file"]
            image = np.asarray(Image.open(fpath).convert("RGB"), dtype=np.int16)
            assert image.shape == (720, 1280, 3)
            blue = (image[:, :, 2] - image[:, :, 0] > 40) & (
                image[:, :, 2] - image[:, :, 1] > 15
            )
            # Reject disconnected blue-tinted background pixels from indirect light.
            # The 5 px opening removes the thin display rod when it reflects blue.
            components, _ = connected_components(
                binary_opening(blue, structure=np.ones((5, 5)))
            )
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            yy, xx = np.nonzero(components == sizes.argmax())
            assert len(xx) > 100, (run, frame["file"])
            center = np.array([(xx.min() + xx.max()) / 2, (yy.min() + yy.max()) / 2])
            delta = float(np.linalg.norm(center - uv))
            projection_errors.append(delta)
            pixel_checks.append(
                dict(
                    file=frame["file"],
                    time_s=frame["time_s"],
                    predicted_uv=uv.tolist(),
                    measured_blue_bbox_center_uv=center.tolist(),
                    error_px=delta,
                )
            )
            hashes[str(fpath.relative_to(ROOT))] = hashlib.sha256(
                fpath.read_bytes()
            ).hexdigest()
        assert max(projection_errors) < 3, (run, max(projection_errors))
        key = f"{platform}_{solver.lower()}_{case}"
        (OUT / f"{key}_projection.json").write_text(
            json.dumps(pixel_checks, indent=2) + "\n"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "40",
            "-i",
            str(run / "native_camera/frame_%04d.png"),
            "-frames:v",
            str(count),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUT / f"{key}.mp4"),
        ]
        subprocess.run(cmd, check=True)
        commands.append(cmd)
        shutil.copy2(
            run / "native_camera" / frames[20 if case == "freefall" else 0]["file"],
            OUT / f"{key}.png",
        )
        for path in [source, original, capture_path, run / "run.json"]:
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        entries.append(
            dict(
                key=key,
                **entry,
                baseline_run_id=baseline,
                state_arrays_exactly_equal=equal,
                frame_count=count,
                window_s=[float(times[0]), float(times[-1])],
                video_duration_s=count / 40,
                camera_matrix_max_difference=max(camera_errors),
                blue_object_projection_max_error_px=max(projection_errors),
                quaternion_metadata_note=(
                    "Raw key camera_quat_opengl_wxyz is mislabeled; values are xyzw per installed CameraData.quat_w_opengl documentation. Raw file retained unchanged."
                    if mislabeled
                    else None
                ),
            )
        )
        print("Verified", key, "pixel error", max(projection_errors), flush=True)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    for case in ["freefall", "pendulum"]:
        clips = [e for e in entries if e["case"] == case]
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for e in clips:
            cmd += ["-i", str(OUT / f'{e["key"]}.mp4')]
        filters = []
        for i, e in enumerate(clips):
            title = (
                "EmbodiChain DexUni"
                if e["solver"] == "DexUni"
                else f'Isaac Lab {e["solver"]} 4/1'
            )
            filters.append(
                f"[{i}:v]scale=640:360,pad=640:432:0:42:color=0x172431,drawtext=fontfile={font}:text='{title}  |  dt 5 ms':fontcolor=white:fontsize=20:x=16:y=10,drawtext=fontfile={font}:text='Initial 0  |  40 fps  |  1x speed':fontcolor=white:fontsize=15:x=16:y=410[p{i}]"
            )
        filters.append("[p0][p1][p2]hstack=inputs=3[out]")
        cmd += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUT / f"{case}_comparison.mp4"),
        ]
        subprocess.run(cmd, check=True)
        commands.append(cmd)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            "0.5" if case == "freefall" else "0",
            "-i",
            str(OUT / f"{case}_comparison.mp4"),
            "-frames:v",
            "1",
            str(OUT / f"{case}_comparison.png"),
        ]
        subprocess.run(cmd, check=True)
        commands.append(cmd)
    probes = {}
    for path in sorted(OUT.glob("*.mp4")):
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
                    "stream=width,height,r_frame_rate,nb_read_frames,duration",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
            )
        )["streams"][0]
        count = 40 if "freefall" in path.name else 400
        assert probe["r_frame_rate"] == "40/1" and int(probe["nb_read_frames"]) == count
        assert abs(float(probe["duration"]) - count / 40) < 1e-6
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True
        )
        probes[path.name] = probe
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    for path in [Path(__file__), cfg_path]:
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest = dict(
        entries=entries,
        validation="All six full 10 s state trajectories match their baselines exactly; intrinsics, camera poses, sample times and native image projections checked.",
        image_projection_method="Largest connected blue-body bounding-box center after 5 px morphological opening of the detection mask (removes the thin display rod), vs native-state projection. Native RGB files are unchanged. Cube uses its visible front face; sphere uses COM. Edge filtering and antialiasing allow up to 3 px.",
        camera_convention="OpenGL: right +X, up +Y, forward -Z; quaternion xyzw.",
        video_probes=probes,
        encoding_commands=commands,
        sources_sha256=hashes,
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("All six recordings and eight MP4 files verified.")


if __name__ == "__main__":
    main()
