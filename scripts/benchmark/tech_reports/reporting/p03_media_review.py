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
"""Validate P-03 native videos, camera calibration and recorded state trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .p03_metrics import metrics


def main() -> None:
    root = Path("outputs/benchmarks/P-03")
    selection = json.loads((root / "analysis/grasp-v1/selection.json").read_text())
    fingers = dict(np.load(root / "prepared/collision-v1/fingers.npz"))
    out = root / "videos"
    out.mkdir(exist_ok=True)
    results = []
    cameras = {}
    for shape in ["cube", "cylinder"]:
        for platform in ["embodichain", "isaaclab"]:
            tag = (
                "native_v4" if platform == "embodichain" else "native_v4_partition_fix"
            )
            candidates = sorted(
                (root / "runs").glob(f"*_{tag}_{platform}_{shape}_physics_n0001*")
            )
            run = candidates[-1]
            meta = json.loads((run / "run.json").read_text())
            assert meta["status"] == "completed"
            camera = run / "native_camera"
            cap = json.loads((camera / "capture.json").read_text())
            assert len(cap["frames"]) == 200
            times = np.array([f["time_s"] for f in cap["frames"]])
            assert np.allclose(times, np.arange(1, 201) * 0.05, atol=1e-10)
            matrices = []
            for f in cap["frames"]:
                if platform == "embodichain":
                    m = np.array(f["camera_world_opengl"])
                else:
                    m = np.eye(4)
                    m[:3, :3] = Rotation.from_quat(
                        f["camera_quat_opengl_xyzw"]
                    ).as_matrix()
                    m[:3, 3] = f["camera_pos_readback"]
                matrices.append(m)
                assert np.allclose(
                    f["intrinsics"],
                    [[750, 0, 480], [0, 750, 360], [0, 0, 1]],
                    atol=1e-5,
                )
            matrices = np.array(matrices)
            cameras[(shape, platform)] = matrices
            video = camera / "native_1x.mp4"
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
                        "stream=width,height,avg_frame_rate,nb_read_frames,duration",
                        "-of",
                        "json",
                        str(video),
                    ],
                    text=True,
                )
            )["streams"][0]
            assert (probe["width"], probe["height"], int(probe["nb_read_frames"])) == (
                960,
                720,
                200,
            )
            assert (
                probe["avg_frame_rate"] == "20/1"
                and abs(float(probe["duration"]) - 10) < 1e-6
            )
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            base = root / "runs" / selection["quality"][shape][platform]
            x = np.load(run / "episode_00_state.npy")[:, 0]
            b = np.load(base / "episode_00_state.npy")[:, 0]
            t = np.load(run / "episode_00_time.npy")
            measured, _ = metrics(x, t, shape, fingers)
            reference, _ = metrics(b, t, shape, fingers)
            assert measured["cycle_completed"] == reference["cycle_completed"]
            pixels = {}
            for k in [0, 39, 59, 119, 199]:
                rgb = np.asarray(Image.open(camera / f"frame_{k:04d}.png"))[
                    ..., :3
                ].astype(float)
                orange = (
                    (rgb[..., 0] > 1.4 * rgb[..., 1])
                    & (rgb[..., 1] > 1.25 * rgb[..., 2])
                    & (rgb[..., 0] > 80)
                )
                pixels[str(k)] = int(orange.sum())
                assert (
                    orange.sum() > 20
                ), "Target missing from camera; inspect renderer partition visibility"
            results.append(
                dict(
                    run_id=run.name,
                    shape=shape,
                    platform=platform,
                    baseline=base.name,
                    video=probe,
                    video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                    frame_hashes=[
                        dict(
                            file=f["file"],
                            sha256=hashlib.sha256(
                                (camera / f["file"]).read_bytes()
                            ).hexdigest(),
                        )
                        for f in cap["frames"]
                    ],
                    camera_temporal_max_component_drift=float(
                        abs(matrices - matrices[0]).max()
                    ),
                    object_position_max_difference_mm=float(
                        np.linalg.norm(x[:, 39:42] - b[:, 39:42], axis=1).max() * 1000
                    ),
                    joint_position_max_difference=np.abs(x[:, :9] - b[:, :9])
                    .max(0)
                    .tolist(),
                    metrics=measured,
                    orange_pixels_in_checked_frames=pixels,
                )
            )
    compare = {
        s: float(abs(cameras[(s, "embodichain")] - cameras[(s, "isaaclab")]).max())
        for s in ["cube", "cylinder"]
    }
    assert max(compare.values()) < 1e-6
    invalid = [p.name for p in (root / "runs").glob("*native_v4_isaaclab_*")] + [
        p.name for p in (root / "runs").glob("*contact_audit_isaaclab*")
    ]
    (out / "validation.json").write_text(
        json.dumps(
            dict(
                records=results,
                camera_pair_max_matrix_difference=compare,
                invalid_capture_runs=invalid,
                invalid_capture_reason="External Isaac camera lacked scene partition visibility; ground only. Raw physics is retained. Fixed through show_all_partitions_by_default=True; no physics change.",
            ),
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# P-03 同视角原生录像\n",
        "以下四段均来自正在运行的模拟器相机。共同镜头：960×720，fx=fy=750 px，主点 (480,360)；eye=(1.35,−0.6,1.35) m，target=(0.45,0.5,0.9) m。200 帧，物理时刻 0.05–10.00 s，20 fps、1× 播放；视频播放时间比帧的物理时刻小 0.05 s。\n",
        "| 物体／平台 | 原片 | t=3 s 原始截图 | 相对主测的物体位置最大差 |",
        "|---|---|---|---:|",
    ]
    for r in results:
        path = f"../runs/{r['run_id']}/native_camera"
        lines.append(
            f"| {r['shape']}／{r['platform']} | [10 s 原片]({path}/native_1x.mp4) · [逐帧时刻与编码命令]({path}/capture.json) | ![t=3 s]({path}/frame_0059.png) | {r['object_position_max_difference_mm']:.4f} mm |"
        )
    lines += [
        "\n录像重跑均保持对应任务事件，不增加独立初态数。位置差包括各自重跑的接触过程差异，未通过改写位姿来对齐画面。两平台相机读回矩阵最大分量差 "
        + f"{max(compare.values()):.3g}"
        + "，全部视频完成 200 帧解码核验；原始 PNG、状态、相机参数和哈希均保留。渲染器的材质、背景与照明不同，本项不作渲染质量比较。\n",
        "[核验数据](validation.json)。旧 Isaac 录像因场景分区不可见而只显示地面，已标为无效素材并保留；补录使用公开渲染选项显示环境分区，物理参数未改变。\n",
    ]
    (out / "index.md").write_text("\n".join(lines))
    print(
        json.dumps(
            dict(
                camera_errors=compare,
                recordings=[
                    {k: r[k] for k in ["run_id", "object_position_max_difference_mm"]}
                    for r in results
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
