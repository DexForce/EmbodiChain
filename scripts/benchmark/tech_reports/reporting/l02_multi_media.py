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

"""Encode and verify native multi-arena captures, retaining every original frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

__all__ = ["main"]
ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
BASE = ROOT / "outputs/benchmarks/L-02/multi-arena-v1"


def main() -> None:
    """Encode native captures and report matched camera and state evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=BASE)
    parser.add_argument("--reference-root", type=Path)
    args = parser.parse_args()
    base = args.input_root.resolve()
    (base / "videos").mkdir(exist_ok=True)
    records = []
    for run in sorted((base / "runs").glob("*media*")):
        meta = json.loads((run / "run.json").read_text())
        if meta["status"] != "completed":
            continue
        cfg = json.loads((run / "config.json").read_text())
        folder = run / "native_camera"
        capture = json.loads((folder / "capture.json").read_text())
        frames = sorted(folder.glob("frame_*.png"))
        assert len(frames) == 60
        for frame in frames:
            with Image.open(frame) as im:
                assert im.size == (640, 480)
        states = np.load(run / "recording_states.npy")
        assert (
            states.shape == (60, cfg["num_envs"], 1, 13) and np.isfinite(states).all()
        )
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
            str(folder / "native_1x.mp4"),
        ]
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
                    str(folder / "native_1x.mp4"),
                ]
            )
        )
        assert (
            float(probe["format"]["duration"]) == 3
            and int(probe["streams"][0]["nb_frames"]) == 60
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(folder / "native_1x.mp4"),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        origins = np.load(run / "model.npz")["env_origins"]
        camera = np.asarray(capture["camera"])
        camera[:3, 3] -= origins.mean(0)
        rec = dict(
            run_id=run.name,
            run_path=str(run.relative_to(ROOT)),
            platform=cfg["platform"],
            shape=cfg["shape"],
            num_envs=cfg["num_envs"],
            rigid_enable_sleep=(
                json.loads((run / "model-audit.json").read_text()).get(
                    "rigid_enable_sleep", True
                )
                if cfg["platform"] == "embodichain"
                else None
            ),
            camera_centered_world=camera.tolist(),
            intrinsics=capture["intrinsics"],
            camera_instances=capture["camera_instances"],
            fps=20,
            duration_s=3,
            speed=1,
            state_max_drift_m=float(
                np.linalg.norm(
                    states[..., :3] - np.asarray(cfg["positions"]), axis=-1
                ).max()
            ),
            command=command,
            probe=probe,
            mp4_sha256=hashlib.sha256(
                (folder / "native_1x.mp4").read_bytes()
            ).hexdigest(),
        )
        records.append(rec)
        (folder / "encoding.json").write_text(json.dumps(rec, indent=2) + "\n")
    if args.reference_root:
        reference = args.reference_root.resolve()
        existing = {(r["platform"], r["shape"]) for r in records}
        for rec in json.loads((reference / "videos/validation.json").read_text())[
            "records"
        ]:
            if (rec["platform"], rec["shape"]) in existing:
                continue
            run = reference / "runs" / rec["run_id"]
            assert (
                hashlib.sha256(
                    (run / "native_camera/native_1x.mp4").read_bytes()
                ).hexdigest()
                == rec["mp4_sha256"]
            )
            rec["run_path"] = str(run.relative_to(ROOT))
            records.append(rec)
    # Keep earlier native previews on disk, publish the latest verified pair.
    records = list({(r["platform"], r["shape"]): r for r in records}.values())
    assert len(records) == 4, len(records)
    matched = []
    for shape in ["cube", "sphere"]:
        a, b = [
            next(x for x in records if x["shape"] == shape and x["platform"] == p)
            for p in ["embodichain", "isaaclab"]
        ]
        camera_error = float(
            np.max(
                np.abs(
                    np.asarray(a["camera_centered_world"]) - b["camera_centered_world"]
                )
            )
        )
        intrinsics_error = float(
            np.max(np.abs(np.asarray(a["intrinsics"]) - b["intrinsics"]))
        )
        captures = [
            json.loads(
                (ROOT / r["run_path"] / "native_camera/capture.json").read_text()
            )
            for r in [a, b]
        ]
        time_error = float(
            np.max(
                np.abs(
                    np.asarray(captures[0]["physical_times_s"])
                    - captures[1]["physical_times_s"]
                )
            )
        )
        assert time_error == 0 and all(
            c["fps"] == 20 and c["speed"] == 1 and c["camera_instances"] == 16
            for c in captures
        )
        assert camera_error < 1e-5 and intrinsics_error < 1e-3, (
            camera_error,
            intrinsics_error,
        )
        matched.append(
            dict(
                shape=shape,
                camera_max_matrix_error=camera_error,
                intrinsics_max_error=intrinsics_error,
                physical_time_max_error_s=time_error,
            )
        )
    (base / "videos/validation.json").write_text(
        json.dumps(dict(records=records, pairs=matched), indent=2) + "\n"
    )
    lines = [
        "# L-02 多 arena 原生视频",
        "",
        "每段均为实际仿真运行采集：16 个独立 arena，每 arena 一个物体，20 fps、3 s、1×。场景处于零重力静态初态；视频展示复制后的原生画面，不表示加载速度或物理吞吐。",
        "",
        "两端均从阵列中心的相同偏移观察，内参与方向一致；只消除两端原生网格整体平移的区别，不改动场景对象。两端的公共相机接口均按环境创建 16 台相机，本素材只导出第一台的全景。因此这些运行只用于展示，不纳入无相机主测的加载时间或显存统计。主测两端都无相机。",
        "",
        "EmbodiChain 使用 Hybrid、8 spp、OptiX；Isaac 使用 RTX、DLSS quality 和降噪。Isaac 仅在录像中关闭渲染分区，以展示全部 arena；物理碰撞隔离仍开启。早期分区预览保留在原运行目录。材质和灯光显示不同，不作相同画质效率排名。原始帧、每帧完整状态和采集时刻全部保留。",
        "",
    ]
    for r in records:
        prefix = os.path.relpath(ROOT / r["run_path"], base / "videos")
        sleep = (
            f"rigid_enable_sleep={r['rigid_enable_sleep']}"
            if r["platform"] == "embodichain"
            else "PhysX 原配置"
        )
        lines += [
            f"## {r['shape']} · {r['platform']} · {sleep}",
            "",
            f"[原片]({prefix}/native_camera/native_1x.mp4) · [原始帧]({prefix}/native_camera/) · [完整状态]({prefix}/recording_states.npy) · [采集时刻]({prefix}/native_camera/capture.json) · [编码记录]({prefix}/native_camera/encoding.json)",
            "",
            f"![原生画面]({prefix}/native_camera/frame_0029.png)",
            "",
        ]
    lines += ["[相机配对与解码核验](validation.json)", ""]
    (base / "videos/index.md").write_text("\n".join(lines))
    print(json.dumps(dict(records=len(records), pairs=matched), indent=2))


if __name__ == "__main__":
    main()
