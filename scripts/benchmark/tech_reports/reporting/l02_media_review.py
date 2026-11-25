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

"""Encode and verify real L-02 simulator frames; no synthesized scene imagery."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image


def main() -> None:
    root = Path("outputs/benchmarks/L-02")
    rows = json.loads((root / "analysis/runs.json").read_text())
    out = root / "videos"
    out.mkdir(exist_ok=True)
    records = []
    for row in rows:
        if row["purpose"] != "media" or row["status"] != "completed":
            continue
        folder = root / "runs" / row["run_id"]
        frames = folder / "native_camera"
        capture = json.loads((frames / "capture.json").read_text())
        images = sorted(frames.glob("frame_*.png"))
        assert len(images) == 60
        cfg = json.loads((folder / "config.json").read_text())
        states = np.load(folder / "recording_states.npy")
        initial = np.load(folder / "first_state.npy")
        assert (
            states.shape == (60, cfg["object_count"], 13) and np.isfinite(states).all()
        )
        expected_times = 0.005 + np.arange(1, 61) * 0.05
        np.testing.assert_allclose(
            capture["physical_times_s"], expected_times, rtol=0, atol=1e-12
        )
        for frame in images:
            assert Image.open(frame).size == (640, 480)
        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-framerate",
            "20",
            "-i",
            str(frames / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(frames / "native_1x.mp4"),
        ]
        if not (frames / "native_1x.mp4").exists():
            subprocess.run(command, check=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(frames / "native_1x.mp4"),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_streams",
                    "-show_format",
                    "-of",
                    "json",
                    str(frames / "native_1x.mp4"),
                ],
                text=True,
            )
        )
        stream = probe["streams"][0]
        assert (
            int(stream["nb_frames"]) == 60
            and stream["width"] == 640
            and stream["height"] == 480
            and stream["avg_frame_rate"] == "20/1"
        )
        assert abs(float(probe["format"]["duration"]) - 3) < 0.001
        audit = json.loads((folder / "model-audit.json").read_text())
        peers = [
            x
            for x in rows
            if x["purpose"] == "main"
            and x["timing_eligible"]
            and all(
                x[k] == row[k]
                for k in ["shape", "platform", "object_count", "asset_mode"]
            )
        ]
        peer = peers[0] if peers else None
        assert peer is not None, "A matching completed measurement is required"
        peer_cfg = json.loads(
            (root / "runs" / peer["run_id"] / "config.json").read_text()
        )
        for key in ["dt", "dexuni_iterations", "objects", "positions", "camera"]:
            assert cfg[key] == peer_cfg[key], key
        reference = np.load(root / "runs" / peer["run_id"] / "first_state.npy")
        delta = (
            float(
                np.linalg.norm(
                    initial[:, :3]
                    - np.load(root / "runs" / peer["run_id"] / "first_state.npy")[
                        :, :3
                    ],
                    axis=1,
                ).max()
            )
            if peer
            else None
        )
        records.append(
            dict(
                run_id=row["run_id"],
                platform=row["platform"],
                shape=row["shape"],
                object_count=row["object_count"],
                comparison_run=peer["run_id"] if peer else None,
                initial_position_difference_max_m=delta,
                initial_full_state_component_difference_max=float(
                    abs(initial - reference).max()
                ),
                recorded_full_state_component_drift_max=float(
                    abs(states - initial[None]).max()
                ),
                renderer=audit["renderer"],
                recorded_position_drift_max_m=float(
                    np.linalg.norm(
                        states[:, :, :3] - initial[None, :, :3], axis=-1
                    ).max()
                ),
                camera=audit["camera_world_opengl"],
                intrinsics=audit["intrinsics"],
                capture=capture,
                ffmpeg_command=command,
                probe=probe,
                frame_sha256={
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in images
                },
                video_sha256=hashlib.sha256(
                    (frames / "native_1x.mp4").read_bytes()
                ).hexdigest(),
            )
        )
    latest = {(r["platform"], r["shape"]): r["run_id"] for r in records}
    for record in records:
        record["selected"] = (
            record["run_id"] == latest[record["platform"], record["shape"]]
        )
    camera_diffs = {}
    for shape in ["cube", "sphere"]:
        pair = {
            r["platform"]: r for r in records if r["shape"] == shape and r["selected"]
        }
        if len(pair) == 2:
            difference = float(
                np.max(
                    abs(
                        np.array(pair["embodichain"]["camera"])
                        - np.array(pair["isaaclab"]["camera"])
                    )
                )
            )
            assert difference < 1e-6
            camera_diffs[shape] = difference
    representatives = []
    lines = [
        "# L-02 测速首帧原片",
        "",
        "各平台、各形状选择 1000 个共享物体 H 组中首帧时间位于排序中间的一次运行；偶数样本取上中位的实际样本。这些图片就是计时终点采集的原片，未作降噪或后期重绘，不代表收敛画质。展示视频另用更高采样设置。",
        "",
        "| 配置 | 首帧时间（s） | 计时原片与来源 |",
        "|---|---:|---|",
    ]
    for shape in ["cube", "sphere"]:
        for platform in ["embodichain", "isaaclab"]:
            candidates = sorted(
                [
                    r
                    for r in rows
                    if r["purpose"] == "main"
                    and r["timing_eligible"]
                    and r["shape"] == shape
                    and r["platform"] == platform
                    and r["asset_mode"] == "S"
                    and r["cache_mode"] == "H"
                    and r["object_count"] == 1000
                ],
                key=lambda r: r["T_frame_s"],
            )
            if candidates:
                r = candidates[len(candidates) // 2]
                representatives.append(r["run_id"])
                lines.append(
                    f"| {shape}／{platform} | {r['T_frame_s']:.3f} | [首帧 PNG](../runs/{r['run_id']}/first_frame.png) · [原始计时](../runs/{r['run_id']}/run.json) |"
                )
    (out / "first-frames.md").write_text("\n".join(lines) + "\n")
    (out / "validation.json").write_text(
        json.dumps(
            dict(
                records=records,
                selection_rule="Latest completed and verified native capture per platform/shape. Older captures and their source settings remain preserved.",
                camera_max_component_difference=camera_diffs,
                timing_frame_representatives=representatives,
            ),
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# L-02 原生仿真素材",
        "",
        "录像来自两端实际运行的相机，1000 个物体、零重力、相同初态、同一固定镜头。640×480，fx=fy=500 px，eye=(1.8,1.8,1.6) m，target=(0.45,0.45,0.55) m。每段 60 帧、20 fps、3 s、1×；画面展示已加载场景，**不是加载过程的屏幕录像**。加载时间取运行中的真实计时点。",
        "",
        "物理条件与主测一致，展示视频配置每像素 8 样本，DexUni 使用 OptiX 降噪及固定环境光强度 20；Isaac 的补录使用 DLSS 质量模式及原生 DL 降噪，以减少原始预览噪声。实际设置以各 run 的 config 与 model-audit 为准；录像运行不替代主测计时，测速首帧未作后期重绘。",
        "",
        "每个平台／形状选用最近一次完整核验的采集；早期预览噪声较重的版本保留在末尾。",
        "",
        "[测速首帧原片及选择规则](first-frames.md)另列；下方缩略图来自各段视频中间的实际采集帧。",
        "",
        "| 形状／平台 | 原片与元数据 | 原生帧 | 相对主测初态最大差 |",
        "|---|---|---|---:|",
    ]
    for r in records:
        if not r["selected"]:
            continue
        relative = "../runs/" + r["run_id"]
        lines.append(
            f"| {r['shape']}／{r['platform']} | [3 s 原片]({relative}/native_camera/native_1x.mp4) · [相机与时刻]({relative}/native_camera/capture.json) · [实际设置]({relative}/model-audit.json) | ![原生帧]({relative}/native_camera/frame_0029.png) | {r['initial_position_difference_max_m']*1000:.6f} mm |"
        )
    lines += [
        "",
        "所有原片均完成逐帧解码检查；原始 PNG、逐帧状态、编码命令和 SHA-256 保留。物理时刻按显式 5 ms 步计数，播放 0 s 对应第一次采集，初始化内部步不当作视频时长。完整[核验记录](validation.json)。",
    ]
    older = [r for r in records if not r["selected"]]
    if older:
        lines += ["", "## 保留的早期预览", ""]
        for r in older:
            relative = "../runs/" + r["run_id"]
            lines.append(
                f"- {r['shape']}／{r['platform']}：[原片]({relative}/native_camera/native_1x.mp4)；{r['renderer']}。"
            )
    (out / "index.md").write_text("\n".join(lines) + "\n")
    print("Verified media:", len(records))


if __name__ == "__main__":
    main()
