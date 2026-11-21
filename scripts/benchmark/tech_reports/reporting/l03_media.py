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

"""Encode L-03 native captures and verify paired cameras and physical timestamps.

Run: python -m scripts.benchmark.tech_reports.reporting.l03_media
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

__all__ = ["main"]


def main() -> None:
    """Retain source PNGs, native states, encoding commands and independent readback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/benchmarks/L-03/three-robots-v1")
    )
    args = parser.parse_args()
    base = args.root
    videos = base / "videos"
    videos.mkdir(exist_ok=True)
    records = []
    for run in sorted((base / "runs").glob("*media*")):
        cfg = json.loads((run / "config.json").read_text())
        meta = json.loads((run / "run.json").read_text())
        if meta["status"] != "completed":
            continue
        folder = run / "native_camera"
        capture = json.loads((folder / "capture.json").read_text())
        frames = sorted(folder.glob("frame_*.png"))
        assert len(frames) == 60
        for frame in frames:
            with Image.open(frame) as im:
                assert im.size == (960, 540)
        states = np.load(run / "recording-states.npz")
        assert states["qpos"].shape[:2] == (60, 4)
        assert all(np.isfinite(states[k]).all() for k in states.files)
        target = folder / "native_1x.mp4"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "20",
            "-i",
            str(folder / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ]
        if not target.exists():
            subprocess.run(command, check=True)
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "stream=width,height,nb_frames,r_frame_rate:format=duration",
                    "-of",
                    "json",
                    str(target),
                ]
            )
        )
        assert (
            float(probe["format"]["duration"]) == 3
            and int(probe["streams"][0]["nb_frames"]) == 60
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], check=True
        )
        camera = np.asarray(capture["camera"]["camera_world_opengl"])
        camera[:3, 3] -= np.load(run / "model.npz")["env_origins"].mean(0)
        reference = (
            Path("outputs/benchmarks/L-03/preflight/three-robots-v1")
            / f"{cfg['robot']}_0004_{cfg['platform']}"
        )
        trace = np.load(reference / "initialization-trace.npz")
        errors = []
        for i, t in enumerate(capture["physical_times_s"]):
            step = int(round(t / 0.005))
            key = f"step_{step:04d}_link_pose"
            if key in trace:
                errors.append(
                    float(
                        np.linalg.norm(
                            states["link_pose"][i, ..., :3] - trace[key][..., :3],
                            axis=-1,
                        ).max()
                    )
                )
        first = np.load(run / "states.npz")["first_link_pose"]
        rec = dict(
            run_id=run.name,
            run_path=str(run),
            platform=cfg["platform"],
            robot=cfg["robot"],
            num_envs=4,
            camera_centered_world=camera.tolist(),
            intrinsics=capture["camera"]["intrinsics"],
            camera_instances=capture["camera"]["camera_instances"],
            physical_times_s=capture["physical_times_s"],
            fps=20,
            speed=1,
            duration_s=3,
            matched_preflight_frames=len(errors),
            max_preflight_link_difference_m=max(errors),
            max_link_drift_from_first_m=float(
                np.linalg.norm(
                    states["link_pose"][..., :3] - first[None, ..., :3], axis=-1
                ).max()
            ),
            command=command,
            probe=probe,
            mp4_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        )
        records.append(rec)
        (folder / "encoding.json").write_text(json.dumps(rec, indent=2) + "\n")
    assert len(records) == 6
    paired = []
    for robot in ("cartpole", "franka", "go2"):
        a, b = [
            next(r for r in records if r["robot"] == robot and r["platform"] == p)
            for p in ("embodichain", "isaaclab")
        ]
        ce = float(
            np.abs(
                np.asarray(a["camera_centered_world"]) - b["camera_centered_world"]
            ).max()
        )
        ke = float(np.abs(np.asarray(a["intrinsics"]) - b["intrinsics"]).max())
        te = float(
            np.abs(np.asarray(a["physical_times_s"]) - b["physical_times_s"]).max()
        )
        # Calibration tolerances are numerical readback precision, not physics quality thresholds.
        assert (
            ce < 1e-5
            and ke < 1e-3
            and te == 0
            and a["camera_instances"] == b["camera_instances"] == 4
        )
        paired.append(
            dict(
                robot=robot,
                camera_max_matrix_error=ce,
                intrinsics_max_error=ke,
                physical_time_max_error_s=te,
            )
        )
    (videos / "validation.json").write_text(
        json.dumps(dict(records=records, paired=paired), indent=2) + "\n"
    )
    lines = [
        "# L-03 原生仿真视频",
        "",
        "三种机器人各有两平台原片。每段为 4 个真实 arena、960×540、20 fps、3 s、1×。两端相机相对阵列中心的位姿、内参和物理采样时刻一致；每端实际创建 4 台相机，仅保存第一台。主测没有相机，录像不参与加载时间统计。",
        "",
        "场景为零重力、无驱动的加载初态，静止画面表示模型就绪后的实际运行，并非运动控制演示。光照、背景和材质呈现存在差异；未修改源资产来匹配外观，也不将本素材视为渲染质量评测。",
        "",
        "前 0.5 s 的 9 个采样帧与无渲染预检逐点核对，完整 3 s 保留状态及相对首步的漂移。",
        "",
        "| 机器人 | 平台 | 原片 | 原始帧／参数 | 对预检最大 link 差（mm） |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for r in sorted(records, key=lambda r: (r["robot"], r["platform"])):
        path = f"../runs/{r['run_id']}/native_camera"
        lines.append(
            f"| {r['robot']} | {r['platform']} | [3 s 原片]({path}/native_1x.mp4) | [采样与相机]({path}/capture.json) · [编码核验]({path}/encoding.json) | {r['max_preflight_link_difference_m']*1000:.6f} |"
        )
    for r in sorted(records, key=lambda r: (r["robot"], r["platform"])):
        path = f"../runs/{r['run_id']}/native_camera"
        lines += ["", f"![{r['robot']} · {r['platform']}]({path}/frame_0029.png)", ""]
    lines += [
        "",
        "[配对相机、时间与状态核验](validation.json)。原始 PNG 全部保留在对应运行的 native_camera/。",
        "",
    ]
    (videos / "index.md").write_text("\n".join(lines))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
