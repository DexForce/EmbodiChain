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

"""Render a 150-second subtitled explanation from verified offline replay data.

Only saved measured states are rendered. The simulator is never stepped and
unexecuted comparison targets are shown solely as labelled path diagrams.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import gymnasium
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import torch

from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.scripts.run_task import (
    _create_parser,
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import VisualObjectCfg
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg
from examples.expansion.trajectory.real2sim_carton.prepare import digest, write_json

__all__: list[str] = []

W, H, FPS = 1600, 900, 20
BG, INK, MUTED = "#10283b", "#eef4f8", "#b6cbd9"
BLUE, AMBER, GREEN = "#75aaff", "#ffb36b", "#5de0ba"
SECTIONS = (
    (
        18,
        "真实输入",
        "从公开真实记录出发",
        "DROID 原始 HDF5、提交命令与三个相机视角分别保留；视频通过采集时钟映射。",
    ),
    (
        16,
        "运动原型",
        "七个阶段，三个原子技能",
        "人工阶段标注进入同一规范化链路；Pick、MoveHeldObject、Place 保留固定 revision。",
    ),
    (
        18,
        "物体与 affordance",
        "尺寸、可抓侧面与方向",
        "多视角与夹爪开口约束得到估计盒体；抓取投影到合法侧面，并保留世界向上的抬升。",
    ),
    (
        24,
        "轨迹修正",
        "同一输入的三种对照",
        "蓝色是输入目标，橙色是修正目标，绿色是仿真实测；原样与刚性对照没有执行成功声明。",
    ),
    (
        24,
        "真实来源迁移",
        "接近、抓取、搬运与定向放置",
        "对象参考系随布局重定向；接触与持有事件保留。源视频和仿真实测通过显式时间映射同步。",
    ),
    (
        20,
        "独立合成来源",
        "可控误差解释修正过程",
        "独立原生参考增加毫米级平移、3° 姿态偏差和缓慢漂移；没有复制真实轨迹或使用干净答案。",
    ),
    (
        18,
        "一条示范的扩增",
        "13 个布局 × 6 种配方",
        "对象和目标的平移、朝向与组合布局，接入现有轨迹算子和映射感知时间扩增。",
    ),
    (
        12,
        "验收与提交",
        "实测、附件与 confirmed receipt",
        "两条来源分别统计；只有物理检查、附件校验和回执均通过才计入成功。结论限定于声明物理代理。",
    ),
)


def render(
    prepared: Path, replay: Path, output: Path, *, preview: bool = False
) -> Path:
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((replay / "manifest.json").read_text())
    report = json.loads((replay / "acceptance-report.json").read_text())
    cases = {origin: [] for origin in ("measured", "synthetic_reference")}
    for entry in manifest["cases"]:
        if entry["status"] == "passed":
            case = json.loads((replay / entry["url"]).read_text())
            if not case.get("receipt", {}).get("confirmed") or not case.get("frames"):
                raise ValueError(
                    "video requires measured states and existing confirmed receipts"
                )
            cases[entry["origin"]].append(case)
    if not all(cases.values()):
        raise ValueError(
            "video requires a verified execution from each separate origin"
        )
    real = next(
        c
        for c in cases["measured"]
        if c["case"]["layout"] == "baseline" and c["case"]["recipe"] == "nominal"
    )
    synthetic = next(
        c
        for c in cases["synthetic_reference"]
        if c["case"]["layout"] == "baseline" and c["case"]["recipe"] == "nominal"
    )
    videos = {
        serial: list(iio.imiter(replay / f"media/{serial}.mp4", plugin="FFMPEG"))
        for serial in ("20521388", "24259877", "13062452")
    }
    clock = manifest["qualification"]["frame_clock_map"]
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font_path.is_file():
        raise RuntimeError("install Noto Sans CJK for Chinese explanation subtitles")
    fonts = {
        size: ImageFont.truetype(str(font_path), size)
        for size in (18, 21, 24, 28, 36, 46)
    }
    args = _create_parser().parse_args(
        [
            "--gym-config",
            str(prepared / "task.native.yaml"),
            "--device",
            "cpu",
            "--headless",
            "--renderer",
            "hybrid",
            "--disable-sensor",
        ]
    )
    discover_task_packages()
    execute_init_hooks()
    cfg, deployment, actions = build_env_cfg_from_args(args)
    cfg.record_trajectory = False
    cfg.trajectory_auto_save = False
    environment = gymnasium.make(id=deployment["id"], cfg=cfg, **actions)
    process = None
    target_file = output / "real2sim-carton-explained.mp4"
    try:
        environment.reset(seed=0, options={"save_data": False})
        env, sim = environment.unwrapped, environment.unwrapped.sim
        carton, board = (sim.get_rigid_object(uid) for uid in ("carton", "worktop"))
        dimensions = manifest["object_size"]
        front = sim.add_visual_object(
            VisualObjectCfg(
                uid="replay_front_direction",
                shape=CubeCfg(size=(0.0005, dimensions[1], dimensions[2])),
            )
        )
        marks = []
        for i, (size, offset) in enumerate(
            (
                (
                    (dimensions[0] + 0.02, 0.001, 0.0005),
                    (0, -dimensions[1] / 2 - 0.01, 0),
                ),
                (
                    (dimensions[0] + 0.02, 0.001, 0.0005),
                    (0, dimensions[1] / 2 + 0.01, 0),
                ),
                (
                    (0.001, dimensions[1] + 0.02, 0.0005),
                    (-dimensions[0] / 2 - 0.01, 0, 0),
                ),
                (
                    (0.001, dimensions[1] + 0.02, 0.0005),
                    (dimensions[0] / 2 + 0.01, 0, 0),
                ),
                ((0.04, 0.003, 0.0005), (dimensions[0] / 2 + 0.035, 0, 0)),
            )
        ):
            visual = sim.add_visual_object(
                VisualObjectCfg(uid=f"replay_target_mark_{i}", shape=CubeCfg(size=size))
            )
            marks.append((visual, offset))
        sim.prepare()
        blue = sim.create_visual_material(
            VisualMaterialCfg(
                uid="replay_blue", base_color=[0.03, 0.28, 0.67, 1.0], roughness=0.7
            )
        )
        white = sim.create_visual_material(
            VisualMaterialCfg(
                uid="replay_paper", base_color=[0.93, 0.94, 0.90, 1.0], roughness=0.8
            )
        )
        grey = sim.create_visual_material(
            VisualMaterialCfg(
                uid="replay_worktop", base_color=[0.53, 0.59, 0.63, 1.0], roughness=0.8
            )
        )
        carton.set_visual_material(white)
        board.set_visual_material(grey)
        front.set_visual_material(blue)
        for visual, _ in marks:
            visual.set_visual_material(blue)
        camera = sim.add_sensor(
            CameraCfg(
                uid="carton_replay_camera",
                width=1024,
                height=650,
                intrinsics=(950.0, 950.0, 512.0, 325.0),
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=(1.42, -1.45, 1.12), target=(0.43, -0.035, 0.35), up=(0, 0, 1)
                ),
            )
        )
        previous = None

        def simulation(case: dict, fraction: float) -> Image.Image:
            nonlocal previous
            data = case["frames"]
            j = min(
                len(data["t"]) - 1,
                int(np.searchsorted(data["t"], fraction * data["t"][-1], side="right"))
                - 1,
            )
            key = (case["case"]["dataset"], j)
            if previous != key:
                env.robot.set_qpos(
                    torch.tensor([data["q"][j]], dtype=torch.float32), target=False
                )
                pose = np.eye(4)
                pose[:3, 3] = data["object"][j][:3]
                pose[:3, :3] = Rotation.from_quat(data["object"][j][3:]).as_matrix()
                carton.set_local_pose(torch.tensor(pose[None], dtype=torch.float32))
                board_pose = np.array(case["target_pose"])
                board.set_local_pose(
                    torch.tensor(board_pose[None], dtype=torch.float32)
                )
                face = np.eye(4)
                face[0, 3] = dimensions[0] / 2 + 0.0003
                front.set_local_pose(
                    torch.tensor((pose @ face)[None], dtype=torch.float32)
                )
                for visual, offset in marks:
                    local = np.eye(4)
                    local[:3, 3] = offset
                    local[2, 3] = manifest["support_size"][2] / 2 + 0.0008
                    visual.set_local_pose(
                        torch.tensor((board_pose @ local)[None], dtype=torch.float32)
                    )
                sim.sync_render_state()
                for _ in range(3 if previous is None else 1):
                    camera.update()
                previous = key
            return Image.fromarray(
                camera.get_data()["color"][0, :, :, :3].cpu().numpy()
            )

        def video_frame(serial: str, source_seconds: float) -> Image.Image:
            acquisition = np.interp(
                source_seconds, clock["source_seconds"], clock["acquisition_frame"]
            )
            return Image.fromarray(
                videos[serial][
                    min(len(videos[serial]) - 1, max(0, int(round(acquisition))))
                ]
            )

        def text(
            draw: ImageDraw.ImageDraw,
            xy: tuple[int, int],
            value: str,
            size: int = 24,
            color: str = INK,
        ) -> None:
            draw.text(xy, value, font=fonts[size], fill=color)

        def paths(
            image: Image.Image,
            case: dict,
            rect: tuple[int, int, int, int],
            mode: str = "corrected",
        ) -> None:
            draw = ImageDraw.Draw(image)
            x, y, width, height = rect
            draw.rounded_rectangle(
                (x, y, x + width, y + height), radius=10, fill="#19384d"
            )
            points = np.concatenate([np.array(c["input"]) for c in case["curves"]])
            lower, upper = points[:, :2].min(0) - 0.02, points[:, :2].max(0) + 0.02

            def line(values, color):
                if values is None:
                    return
                p = np.array(values)[:, :2]
                p = (p - lower) / (upper - lower)
                draw.line(
                    [
                        (
                            x + 22 + float(a) * (width - 44),
                            y + height - 25 - float(b) * (height - 55),
                        )
                        for a, b in p
                    ],
                    fill=color,
                    width=3,
                )

            for curve in case["curves"]:
                line(curve["input"], BLUE)
                if mode != "input":
                    line(curve[mode], AMBER)
            if mode == "corrected":
                line(case["frames"]["tcp"], GREEN)
            text(draw, (x + 16, y + 8), "TCP / XY · 目标路径与实测", 18)

        if not preview:
            process = subprocess.Popen(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{W}x{H}",
                    "-framerate",
                    str(FPS),
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-threads",
                    "4",
                    str(target_file),
                ],
                stdin=subprocess.PIPE,
            )
        elapsed = 0
        subtitles = []
        for section, (seconds, label, title, subtitle) in enumerate(SECTIONS):
            subtitles.append(dict(start=elapsed, stop=elapsed + seconds, text=subtitle))
            frames = [int(seconds * FPS * 0.55)] if preview else range(seconds * FPS)
            for frame_number in frames:
                u = frame_number / max(1, seconds * FPS - 1)
                image = Image.new("RGB", (W, H), BG)
                draw = ImageDraw.Draw(image)
                text(draw, (35, 22), "EMBODICHAIN   /   REAL2SIM", 21, BLUE)
                text(draw, (35, 61), title, 36)
                text(draw, (1280, 30), f"{section+1:02d} / 08 · {label}", 21, MUTED)
                case = synthetic if section in (3, 5) else real
                if section == 6:
                    choices = [
                        c
                        for c in cases["measured"]
                        if c["case"]["layout"]
                        in ("object_yaw_plus", "target_yaw_minus", "combined_a")
                        and c["case"]["recipe"] in ("task_space", "time_130")
                    ]
                    case = (choices or [real])[
                        min(len(choices or [real]) - 1, int(u * len(choices or [real])))
                    ]
                fraction = u if section in (4, 5) else (0.45 if section == 2 else u)
                if section in (0, 1):
                    source_sec = manifest["source_interval"][0] + u * (
                        manifest["source_interval"][1] - manifest["source_interval"][0]
                    )
                    image.paste(
                        video_frame("20521388", source_sec).resize((1010, 568)),
                        (30, 127),
                    )
                    image.paste(
                        video_frame("13062452", source_sec).resize((480, 270)),
                        (1080, 127),
                    )
                    image.paste(
                        video_frame("24259877", source_sec).resize((480, 270)),
                        (1080, 423),
                    )
                    draw = ImageDraw.Draw(image)
                    text(
                        draw,
                        (49, 706),
                        f"DROID · RAIL / 2023-11-20 · 源时钟 {source_sec:.3f} s",
                        21,
                        MUTED,
                    )
                    if section == 1:
                        labels = (
                            "Pick · 接近 / 闭合 / 抬升",
                            "Move · 持物搬运",
                            "Place · 接近 / 释放 / 撤离",
                        )
                        for i, value in enumerate(labels):
                            text(draw, (45 + i * 510, 750), value, 21, BLUE)
                elif section == 7:
                    for i, origin in enumerate(("measured", "synthetic_reference")):
                        stats = manifest["sources"][origin]
                        x = 60 + i * 770
                        text(
                            draw,
                            (x, 160),
                            "真实运动来源" if i == 0 else "独立合成来源",
                            36,
                            BLUE,
                        )
                        text(
                            draw,
                            (x, 228),
                            f"{stats['passed']} / {stats['expected']}  通过",
                            46,
                            GREEN,
                        )
                        text(
                            draw,
                            (x, 301),
                            f"失败 {stats['failed']} · 全部候选均保留结果",
                            24,
                            MUTED,
                        )
                        for j, c in enumerate(
                            [c for c in report["cases"] if c["origin"] == origin]
                        ):
                            col, row = j % 6, j // 6
                            draw.rounded_rectangle(
                                (
                                    x + col * 85,
                                    365 + row * 18,
                                    x + col * 85 + 77,
                                    377 + row * 18,
                                ),
                                radius=3,
                                fill=GREEN if c["status"] == "passed" else AMBER,
                            )
                    text(
                        draw,
                        (60, 648),
                        "实测中心 ≤ 5 mm   ·   有向 yaw ≤ 3°   ·   抬升 ≥ 30 mm",
                        28,
                    )
                    text(
                        draw,
                        (60, 699),
                        "T+1 状态 + 实际动作 + 来源 / 时间映射 + 完整附件 + confirmed receipt",
                        24,
                    )
                    text(
                        draw,
                        (60, 748),
                        "公开真实运动在估计盒体代理中的迁移；未验证真实动力学或实机部署。",
                        24,
                        MUTED,
                    )
                else:
                    comparison_mode = (
                        ("input", "rigid", "corrected")[min(2, int(u * 3))]
                        if section == 3
                        else "corrected"
                    )
                    if section == 3 and comparison_mode != "corrected":
                        paths(image, case, (25, 125, 1010, 642), comparison_mode)
                    else:
                        image.paste(
                            simulation(case, fraction).resize((1010, 642)), (25, 125)
                        )
                    draw = ImageDraw.Draw(image)
                    audit = case["case"]["audit"]
                    text(
                        draw,
                        (1063, 140),
                        "合成参考 → 仿真" if case is synthetic else "真实运动 → 仿真",
                        28,
                        BLUE,
                    )
                    j = min(
                        len(case["frames"]["t"]) - 1,
                        int(
                            np.searchsorted(
                                case["frames"]["t"], fraction * case["frames"]["t"][-1]
                            )
                        ),
                    )
                    text(draw, (1063, 188), f"阶段：{case['frames']['phase'][j]}", 21)
                    paths(image, case, (1055, 238, 510, 252))
                    draw = ImageDraw.Draw(image)
                    if section == 2:
                        draw.rectangle(
                            (1100, 302, 1495, 440),
                            fill="#e9ece7",
                            outline=BLUE,
                            width=3,
                        )
                        draw.line((1100, 302, 1495, 302), fill=GREEN, width=8)
                        draw.line((1100, 440, 1495, 440), fill=GREEN, width=8)
                        draw.line((1495, 302, 1495, 440), fill=BLUE, width=10)
                        text(draw, (1120, 323), "合法抓面 ±Y", 24, "#17364b")
                        text(draw, (1120, 373), "+X 正面 →", 24, "#17364b")
                        text(draw, (1063, 522), "合法抓面：对象 ±Y 侧面", 24)
                        text(draw, (1063, 564), "正面方向：对象 +X（蓝面）", 24)
                        text(
                            draw,
                            (1063, 606),
                            "长度 96.6 / 宽 47.2 / 高 36.6 mm",
                            21,
                            MUTED,
                        )
                        text(
                            draw, (1063, 648), "尺寸与 18 mm 安装件均为估计", 21, MUTED
                        )
                    elif section == 3:
                        mode = comparison_mode
                        paths(image, case, (1055, 238, 510, 252), mode)
                        draw = ImageDraw.Draw(image)
                        text(
                            draw,
                            (1063, 522),
                            {
                                "input": "01 原样目标 · 未执行",
                                "rigid": "02 刚性对齐 · 未执行",
                                "corrected": "03 完整修正 · 实测通过",
                            }[mode],
                            24,
                            AMBER,
                        )
                        text(draw, (1063, 565), "偏差：+6 / −4 / +3 mm，yaw +3°", 21)
                        text(draw, (1063, 606), "自由阶段缓慢漂移 ≤ 3 mm", 21)
                        text(
                            draw,
                            (1063, 648),
                            "所有物理画面均来自完整修正执行",
                            21,
                            MUTED,
                        )
                    else:
                        text(
                            draw,
                            (1063, 523),
                            f"中心误差 {audit['xy_error_m']*1000:.2f} mm",
                            28,
                            GREEN,
                        )
                        text(
                            draw,
                            (1063, 568),
                            f"yaw 误差 {audit['yaw_error_deg']:.2f}°",
                            28,
                            GREEN,
                        )
                        text(
                            draw,
                            (1063, 612),
                            f"实测抬升 {audit['lift_m']*1000:.1f} mm",
                            24,
                        )
                        text(
                            draw,
                            (1063, 656),
                            case["case"]["layout"] + " / " + case["case"]["recipe"],
                            21,
                            MUTED,
                        )
                        sec = case["frames"]["source_seconds"][j]
                        if case["case"]["origin"] == "measured" and sec is not None:
                            image.paste(
                                video_frame("13062452", sec).resize((240, 135)),
                                (790, 630),
                            )
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 809, W, H), fill="#0b1e2d")
                # Two fixed lines avoid cramped auto-wrapping of CJK subtitles.
                half = min(52, len(subtitle))
                cut = max(
                    subtitle.rfind("；", 0, half) + 1, subtitle.rfind("，", 0, half) + 1
                )
                cut = cut if 20 < cut < half else half
                text(draw, (35, 826), subtitle[:cut], 24)
                if subtitle[cut:]:
                    text(draw, (35, 863), subtitle[cut:], 21, MUTED)
                if preview:
                    image.save(output / f"section-{section+1:02d}.png")
                else:
                    process.stdin.write(image.tobytes())
            print(f"Section {section+1}/8 rendered", flush=True)
            elapsed += seconds
        if process is not None:
            process.stdin.close()
            if process.wait():
                raise RuntimeError("video encoder failed")
            process = None

        def stamp(seconds):
            return f"00:{int(seconds)//60:02d}:{int(seconds)%60:02d},000"

        (output / "explanation.zh.srt").write_text(
            "\n\n".join(
                f"{i+1}\n{stamp(s['start'])} --> {stamp(s['stop'])}\n{s['text']}"
                for i, s in enumerate(subtitles)
            )
            + "\n"
        )
        write_json(
            output / "video-evidence.json",
            dict(
                schema="carton-explanation-video/v1",
                duration_seconds=elapsed,
                fps=FPS,
                subtitled=True,
                audio=False,
                preview=preview,
                physics_steps_during_replay=0,
                source_manifest_sha256=digest(replay / "manifest.json"),
                video_sha256=digest(target_file) if not preview else None,
                sections=subtitles,
                comparison_targets_executed=False,
                receipt_identities_created=False,
            ),
        )
    finally:
        if process is not None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.wait()
        environment.unwrapped.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()
    return target_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "replay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print(
        render(
            args.prepared.resolve(),
            args.replay.resolve(),
            args.output.resolve(),
            preview=args.preview,
        )
    )


if __name__ == "__main__":
    main()
