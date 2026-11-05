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

"""Validate matched native cameras and create labeled paired previews."""

from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stack-limit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    entries = []
    refs = {}
    plan = (
        json.loads((root / "capture-plan.json").read_text()) if args.stack_limit else []
    )
    plan_by_run = {j["run_id"]: j for j in plan}
    for p in sorted((root / "runs").glob("*/run.json")):
        meta = json.loads(p.read_text())
        if meta["status"] != "completed":
            continue
        key = (
            meta["platform"],
            meta["scene"]["id"],
            meta["solver"],
            meta["dt_s"],
            meta["episode_start"],
        )
        if meta["phase"] in [
            "main",
            "supplement",
            "independent",
            "independent_supplement",
        ] or (args.stack_limit and meta["phase"] == "diagnostic"):
            refs[key] = p.parent
    for p in sorted((root / "runs").glob("*/native_camera/capture.json")):
        run = p.parent.parent
        meta = json.loads((run / "run.json").read_text())
        if meta["status"] != "completed" or meta["duration_s"] < 1:
            continue
        cap = json.loads(p.read_text())
        frames = cap["frames"]
        key = (
            meta["platform"],
            meta["scene"]["id"],
            meta["solver"],
            meta["dt_s"],
            meta["episode_start"],
        )
        if key not in refs:
            continue
        reference = refs[key]
        initial = meta["episode_start"]
        a = np.load(run / f"episode_{initial:02d}.npz")["state"]
        b = np.load(reference / f"episode_{initial:02d}.npz")["state"][: len(a)]
        if len(a) != len(b):
            raise ValueError("Reference shorter than capture")
        error_position = float(np.abs(a[..., :3] - b[..., :3]).max())
        error_velocity = float(np.abs(a[..., 7:10] - b[..., 7:10]).max())
        qdot = np.clip(
            np.abs(
                np.sum(a[..., 3:7] * b[..., 3:7], axis=-1)
                / (
                    np.linalg.norm(a[..., 3:7], axis=-1)
                    * np.linalg.norm(b[..., 3:7], axis=-1)
                )
            ),
            0,
            1,
        )
        angular_error = float(np.rad2deg(2 * np.arccos(qdot)).max())
        camera_errors = []
        intrinsic_errors = []
        centers_visible = []
        for f in frames:
            np.testing.assert_array_equal(np.array(f["state"]), a[f["step"]])
            eye = np.array(f["camera_eye"])
            target = np.array(f["camera_target"])
            front = target - eye
            front /= np.linalg.norm(front)
            right = np.cross(front, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, front)
            expected = np.eye(4)
            expected[:3, :3] = np.column_stack((right, up, -front))
            expected[:3, 3] = eye
            if "camera_world_opengl" in f:
                actual = np.array(f["camera_world_opengl"])
            else:
                actual = np.eye(4)
                actual[:3, :3] = Rotation.from_quat(
                    f["camera_quat_opengl_xyzw"]
                ).as_matrix()
                actual[:3, 3] = f["camera_pos_readback"]
            camera_errors.append(float(np.abs(actual - expected).max()))
            k = np.array(f["intrinsics"])
            expected_k = np.array([[1100, 0, 640], [0, 1100, 360], [0, 0, 1]])
            intrinsic_errors.append(float(np.abs(k - expected_k).max()))
            camera = (a[f["step"], :, :3] - actual[:3, 3]) @ actual[:3, :3]
            pixels = np.column_stack(
                (
                    640 + 1100 * camera[:, 0] / -camera[:, 2],
                    360 - 1100 * camera[:, 1] / -camera[:, 2],
                )
            )
            centers_visible.append(
                int(
                    np.sum(
                        (camera[:, 2] < 0)
                        & (pixels[:, 0] >= 0)
                        & (pixels[:, 0] < 1280)
                        & (pixels[:, 1] >= 0)
                        & (pixels[:, 1] < 720)
                    )
                )
            )
        assert max(camera_errors) < 2e-5 and max(intrinsic_errors) < 0.01, (
            run,
            max(camera_errors),
            max(intrinsic_errors),
        )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(p.parent / "native_1x.mp4"),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        video_probe = json.loads(
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
                    str(p.parent / "native_1x.mp4"),
                ],
                text=True,
            )
        )["streams"][0]
        assert int(video_probe["nb_read_frames"]) == len(frames)
        assert (video_probe["width"], video_probe["height"]) == (1280, 720)
        assert video_probe["r_frame_rate"] == f"{cap['fps']}/1"
        assert abs(float(video_probe["duration"]) - len(frames) / cap["fps"]) < 1e-6
        np.testing.assert_allclose(
            [f["time_s"] for f in frames],
            np.arange(1, len(frames) + 1) / cap["fps"],
            atol=1e-10,
        )
        if args.stack_limit:
            from scripts.benchmark.tech_reports.reporting.p02_stack_metrics import (
                measure_stack,
            )

            metric, curves = measure_stack(run / f"episode_{initial:02d}.npz", meta)
            dest = root / "analysis/capture"
            dest.mkdir(exist_ok=True)
            (dest / f"{run.name}.json").write_text(json.dumps(metric, indent=2) + "\n")
            np.savez_compressed(dest / f"{run.name}.npz", **curves)
        entries.append(
            dict(
                run_id=run.name,
                platform=meta["platform"],
                scene=meta["scene"]["id"],
                layers=meta["scene"].get("layers"),
                fps=cap["fps"],
                role=plan_by_run.get(run.name, {}).get("role"),
                solver=meta["solver"],
                dt_s=meta["dt_s"],
                initial=initial,
                frame_count=len(frames),
                first_time_s=frames[0]["time_s"],
                last_time_s=frames[-1]["time_s"],
                reference_run=reference.name,
                state_array_identical=bool(np.array_equal(a, b)),
                max_position_difference_m=error_position,
                max_velocity_difference_m_s=error_velocity,
                max_orientation_difference_deg=angular_error,
                max_camera_matrix_difference=max(camera_errors),
                max_intrinsics_difference_px=max(intrinsic_errors),
                min_visible_centers=min(centers_visible),
                video=str(p.parent / "native_1x.mp4"),
                capture_metadata=str(p),
                native_decode_verified=True,
                encoded_stream=video_probe,
            )
        )
    pairs = []
    for dex in [e for e in entries if e["platform"] == "embodichain"]:
        match = [
            e
            for e in entries
            if e["platform"] == "isaaclab"
            and (
                e["role"] == dex["role"]
                if args.stack_limit
                else e["scene"] == dex["scene"]
            )
            and e["dt_s"] == dex["dt_s"]
            and (args.stack_limit or e["initial"] == dex["initial"])
            and e["frame_count"] == dex["frame_count"]
        ]
        if not match:
            continue
        isaac = match[-1]
        name = dex["scene"] + ("_it100" if "100 iterations" in dex["solver"] else "")
        if "contact grouping" in dex["solver"]:
            name += "_grouped"
        if "contact history" in dex["solver"]:
            name += "_history"
        if dex["initial"]:
            name += f'_initial{dex["initial"]}'
        if args.stack_limit:
            name = dex["role"]
            dcap = json.loads(Path(dex["capture_metadata"]).read_text())
            icap = json.loads(Path(isaac["capture_metadata"]).read_text())
            for df, inf in zip(dcap["frames"], icap["frames"], strict=True):
                for field in ["time_s", "camera_eye", "camera_target"]:
                    np.testing.assert_allclose(df[field], inf[field], atol=1e-12)
        output = root / "videos" / f"{name}_paired_1x.mp4"
        output.parent.mkdir(exist_ok=True)
        lhs = (
            "DexUni 100 iterations"
            if "100 iterations" in dex["solver"]
            else "DexUni 10 iterations"
        )
        if "contact grouping" in dex["solver"]:
            lhs += " + grouping"
        if "contact history" in dex["solver"]:
            lhs += " + history"
        rhs = isaac["solver"]
        if args.stack_limit:
            lhs = f"DexUni 100 + groups | {dex['layers']} layers"
            rhs = f"PhysX TGS 100/1 | {isaac['layers']} layers"
        filters = f"[0:v]scale=640:360,pad=640:400:0:40:white,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{lhs} | init {dex['initial']} | {dex['dt_s'] * 1000:g} ms | 1x':x=16:y=10:fontsize=20:fontcolor=black[a];[1:v]scale=640:360,pad=640:400:0:40:white,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{rhs} | init {isaac['initial']} | {isaac['dt_s'] * 1000:g} ms | 1x':x=16:y=10:fontsize=20:fontcolor=black[b];[a][b]hstack=inputs=2[out]"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            dex["video"],
            "-i",
            isaac["video"],
            "-filter_complex",
            filters,
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
            str(output),
        ]
        subprocess.run(cmd, check=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "-"], check=True
        )
        pairs.append(
            dict(
                scene=dex["scene"],
                dexuni_run=dex["run_id"],
                isaaclab_run=isaac["run_id"],
                output=str(output),
                encoder_command=cmd,
                paired_decode_verified=True,
            )
        )
        # A native-frame still for Markdown. No geometry reconstruction is used.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(
                    min(
                        9.95 if args.stack_limit else 0.975,
                        (dex["frame_count"] - 1) / dex["fps"],
                    )
                ),
                "-i",
                str(output),
                "-frames:v",
                "1",
                str(root / "figures" / f"{name}_native.png"),
            ],
            check=True,
        )
    (root / "videos" / "validation.json").write_text(
        json.dumps(dict(captures=entries, pairs=pairs), indent=2) + "\n"
    )
    lines = [
        "# P-02 原生仿真素材",
        "",
        "主素材使用初态 0；事件核查片段在表中注明其初态编号。1280×720，40 fps，正常速度。原片和逐帧状态／相机参数全部保留。堆叠视频包括 2 s 稳定阶段；30°／45° 斜面展示共同的前 1 s，相机使用同一解析轨迹；15° 静止场景展示 10 s。",
        "",
        "| 场景／DexUni 配置 | 同步预览 | DexUni 原片 | Isaac Lab 原片 |",
        "| --- | --- | --- | --- |",
    ]
    for p in pairs:
        d = next(e for e in entries if e["run_id"] == p["dexuni_run"])
        i = next(e for e in entries if e["run_id"] == p["isaaclab_run"])
        lines.append(
            f"| {d['scene']} / 初态 {d['initial']} / {d['solver']} | [1×]({Path(p['output']).name}) | [MP4](../runs/{d['run_id']}/native_camera/native_1x.mp4) | [MP4](../runs/{i['run_id']}/native_camera/native_1x.mp4) |"
        )
    lines += [
        "",
        "[核验记录](validation.json)包括与主测轨迹差异、实际相机矩阵／内参偏差和解码检查。每个原片目录含原始 PNG、时间戳和帧对应状态；录制不增加独立初态样本数。",
        "",
        "## 录制与主测的差异",
        "",
        "下表只列完整状态数组不完全相同的录制。差值按相同物理时刻比较；位置、速度取各坐标分量的最大绝对差。录制保留自己的原始轨迹，不能将不一致的片段当作原主测轨迹的逐帧证明。",
        "",
        "| 平台／场景／初态 | 最大位置分量差 / mm | 最大速度分量差 / m·s⁻¹ | 最大姿态差 / ° |",
        "| --- | ---: | ---: | ---: |",
    ]
    for e in entries:
        if not e["state_array_identical"]:
            lines.append(
                f"| {e['solver']} / {e['scene']} / {e['initial']} | {e['max_position_difference_m']*1000:.6g} | {e['max_velocity_difference_m_s']:.6g} | {e['max_orientation_difference_deg']:.6g} |"
            )
    (root / "videos" / "index.md").write_text("\n".join(lines) + "\n")
    if args.stack_limit:
        lines = [
            "# P-02 堆叠边界原生录像",
            "",
            "1280×720、20 fps、正常速度，覆盖释放后完整 10 s。两平台共用固定相机内外参与物理时刻。边界展示允许各自层数不同，标题明确标注；它展示各自观测边界，不是同层数的质量排名。原 PNG、状态、镜头读回及编码命令均保留。",
            "",
            "**复录结果与选片目的不同：**DexUni 51 层原测量完整，复录倒塌；52 层原测量失稳，复录完整。角色名 stable_boundary／next_layer 记录测量时的选择依据，不代表录像必然成功／失败。全部片段保留；[无录像复跑与重复性核查](../reports/repeatability.md)也发现结果变化。",
            "",
            "| 内容 | 同步预览 | DexUni 层数／初态 | PhysX 层数／初态 |",
            "| --- | --- | --- | --- |",
        ]
        for pair in pairs:
            d = next(e for e in entries if e["run_id"] == pair["dexuni_run"])
            i = next(e for e in entries if e["run_id"] == pair["isaaclab_run"])
            lines.append(
                f"| {d['role']} | [1×]({Path(pair['output']).name}) | {d['layers']}／{d['initial']} | {i['layers']}／{i['initial']} |"
            )
        lines += [
            "",
            "## 原片与测量轨迹复核",
            "",
            "| 平台／条件 | 原片 | 完整状态与测量相同 | 最大位置分量差 / mm | 录制首次事件 / s | 最终连续层数 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        for e in entries:
            m = json.loads(
                (root / "analysis/capture" / f"{e['run_id']}.json").read_text()
            )
            event = (
                "无"
                if m["first_geometry_event_s"] is None
                else f"{m['first_geometry_event_s']:.3f}"
            )
            lines.append(
                f"| {e['solver']}／{e['layers']} 层／初态 {e['initial']} | [MP4](../runs/{e['run_id']}/native_camera/native_1x.mp4) | {e['state_array_identical']} | {e['max_position_difference_m']*1000:.6g} | {event} | {m['longest_ordered_chain_final']} |"
            )
        lines += [
            "",
            "[核验记录](validation.json)保存相机、时间、解码及状态差异；[录制选择与命令](../capture-plan.json)说明初态选择。录制不计入独立测量初态数；有差异的片段只证明该次录制，不能冒充原测量的逐帧证据。录制自身的结构事件及连续指标在 `../analysis/capture/`。",
            "",
        ]
        (root / "videos" / "index.md").write_text("\n".join(lines))
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    main()
