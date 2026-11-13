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
"""Build M03 synchronized comparison, state charts and untimed task report.

Run: .venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_m03_comparison.py
"""

from __future__ import annotations
import argparse
import datetime
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_common import (
    OUTPUT,
    load_inputs,
    write_json,
    digest,
)

__all__ = ["main"]
PLATFORMS = ("embodichain", "isaaclab")
LABELS = {
    "embodichain": "EmbodiChain / DexUni surface-fast (base 8)",
    "isaaclab": "Isaac Lab / Newton VBD (ordinary 20)",
}


def build_video(output: Path = OUTPUT) -> dict:
    """Compose equal timestamp perspective/detail panels without distortion."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    preview = output / "preview"
    config = json.loads((output / "inputs/config.json").read_text())
    count = round(config["motion"]["duration"] * 30) + 1
    times = {
        p: json.loads((preview / p / "frame_times.json").read_text()) for p in PLATFORMS
    }
    for t in times.values():
        if len(t) != count or not np.allclose(
            t, np.arange(count) / 30, rtol=0, atol=1e-8
        ):
            raise ValueError("Video time grid differs")
    readers = {p: imageio.get_reader(str(preview / p / "video.mp4")) for p in PLATFORMS}
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    try:
        with imageio.get_writer(
            str(preview / "comparison.mp4"),
            fps=30,
            codec="libx264",
            quality=8,
            macro_block_size=2,
        ) as writer:
            for frame in range(count):
                panel = Image.new("RGB", (1440, 1080))
                for col, p in enumerate(PLATFORMS):
                    source = Image.fromarray(readers[p].get_data(frame))
                    if source.size != (1680, 720):
                        raise ValueError(f"Unexpected {p} video dimensions")
                    for row, box in enumerate(((0, 0, 960, 720), (960, 42, 1680, 720))):
                        view = ImageOps.contain(source.crop(box), (720, 540))
                        panel.paste(
                            view,
                            (
                                col * 720 + (720 - view.width) // 2,
                                row * 540 + (540 - view.height) // 2,
                            ),
                        )
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((col * 720, 0, (col + 1) * 720, 65), fill="black")
                    draw.text((col * 720 + 10, 6), LABELS[p], font=font, fill="white")
                    draw.text(
                        (col * 720 + 10, 34),
                        f"M03 | one-way, kinematic | CUDA Graph | t={times[p][frame]:.2f}s",
                        font=font,
                        fill="white",
                    )
                writer.append_data(np.asarray(panel))
                if frame == count - 1:
                    panel.save(preview / "comparison_final.jpg")
    finally:
        for reader in readers.values():
            reader.close()
    result = dict(
        frames=count,
        fps=30.0,
        width=1440,
        height=1080,
        actual_final_time_s=float(times["embodichain"][-1]),
        actual_time=times["embodichain"],
        sha256=digest(preview / "comparison.mp4"),
        human_approved=False,
    )
    write_json(preview / "comparison_frames.json", result)
    return result


def build_charts(inputs: dict, states: dict, output: Path = OUTPUT) -> None:
    """Plot all four independently observed layers for both platforms."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, layout="constrained")
    colors = inputs["config"]["cloth"]["colors"]
    timing = inputs["config"]["motion"]["timing"]
    target = np.array(inputs["config"]["support"]["target_center_xy"])
    plot_data = {}
    for row, p in enumerate(PLATFORMS):
        d = states[p]
        t = d["time"]
        center = d["particles"].mean(2)
        clearance = (
            d["particles"][..., 2].min(2) - inputs["config"]["support"]["pickup_z"]
        ) * 1000
        error = np.linalg.norm(center[..., :2] - target, axis=2) * 1000
        reference = abs(t - 2).argmin()
        motion = (center[..., 0] - center[reference, :, 0]) * 1000
        plot_data.update(
            {
                f"{p}_time": t,
                f"{p}_clearance_mm": clearance,
                f"{p}_x_motion_mm": motion,
                f"{p}_target_error_mm": error,
            }
        )
        for k, color in enumerate(colors):
            for col, values in enumerate((clearance, motion, error)):
                axes[row, col].plot(
                    t, values[:, k], color=color, label=f"Layer {k}", lw=1.4
                )
        for col, title in enumerate(
            (
                "Minimum clearance / mm",
                "X displacement since settled / mm",
                "Centroid distance to target / mm",
            )
        ):
            ax = axes[row, col]
            ax.set_title(f"{p}: {title}")
            ax.grid(alpha=0.25)
            ax.axvspan(timing["lift"], timing["hold"], color="gray", alpha=0.15)
            ax.axvspan(timing["lay"], timing["release"], color="teal", alpha=0.1)
            ax.set_xlim(0, inputs["config"]["motion"]["duration"])
        axes[row, 0].axhline(50, color="black", ls="--", lw=1)
        axes[row, 1].axhline(150, color="black", ls="--", lw=1)
        axes[row, 2].axhline(20, color="black", ls="--", lw=1)
        axes[row, 0].legend(ncol=4, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Actual simulation time / s")
    fig.suptitle(
        "M03: four independent dynamic cloths | untimed single previews | gray: hold, teal: release"
    )
    fig.savefig(output / "preview/platform_layer_curves.png", dpi=170)
    plt.close(fig)
    np.savez_compressed(output / "preview/platform_plot_data.npz", **plot_data)


def main() -> None:
    """Validate raw episodes, produce comparison media and current records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    inputs = load_inputs()
    task_output = inputs["output_root"]
    states = {}
    manifests = {}
    observations = {}
    qualities = {}
    for p in PLATFORMS:
        root = task_output / "preview" / p
        m = json.loads((root / "manifest.json").read_text())
        if m["status"] != "completed" or m["performance_measurements_started"]:
            raise ValueError(f"Invalid preview status: {p}")
        if m["input_manifest_sha256"] != digest(task_output / "inputs/manifest.json"):
            raise ValueError(f"Stale platform inputs: {p}")
        if m["video_sha256"] != digest(root / "video.mp4"):
            raise ValueError(f"Video checksum differs: {p}")
        if m["effective"]["cuda_graph_status"] != "captured":
            raise ValueError(f"Graph not captured: {p}")
        with np.load(root / "states.npz") as data:
            d = {k: data[k] for k in data.files}
        assert d["particles"].shape == (inputs["steps"] + 1, 4, 961, 3) and all(
            np.isfinite(v).all() for v in d.values()
        )
        assert np.allclose(d["time"], inputs["commands"]["time"], rtol=0, atol=1e-8)
        assert np.allclose(d["qpos"], inputs["targets"], rtol=0, atol=1e-5)
        assert (
            np.linalg.norm(d["tcp"] - inputs["commands"]["tcp_baked"], axis=2).max()
            < 0.004
        )
        states[p] = d
        manifests[p] = m
        observations[p] = json.loads((root / "observations.json").read_text())
        qualities[p] = json.loads((root / "quality.json").read_text())
    video = build_video(task_output)
    build_charts(inputs, states, task_output)
    text = "# M03 最新场景预览（待审阅）\n\n"
    text += "[对照视频](preview/comparison.mp4) · [原生视频](preview/dexsim.mp4) · [逐层曲线](preview/platform_layer_curves.png) · [绘图数据](preview/platform_plot_data.npz)\n\n"
    text += f"当前仅保留一套输入、状态和媒体。实际时长 {inputs['config']['motion']['duration']:.1f} s；四张独立 18 g 布料，无绑定或 pin。one_way / kinematic，surface-fast 基础8次、Isaac普通20次，10子步，CUDA Graph captured。材料与命名关节轨迹共用；本组为统一任务配置下的求解器级策略优化比较，基础迭代数是求解策略的一部分。尚未开展性能测量。\n\n"
    text += "| 平台 | 各层中心误差 mm（底→顶） | 中央材料线反向段 | 各层高度极差 mm | 四层抬升 | 放置线 |\n|---|---|---|---|---|---|\n"
    for p in PLATFORMS:
        o, q = observations[p], qualities[p]
        errors = ", ".join(
            f"{v['final_target_xy_error_m']*1000:.2f}" for v in o["layers"]
        )
        heights = ", ".join(f"{v*1000:.2f}" for v in q["final_layer_z_range_m"])
        text += f"| {p} | {errors} | {q['final_centerline_reversed_segments']} | {heights} | {o['all_four_clear']} | {o['all_four_candidate_placement']} |\n"
    text += "\n中央线反向段只跟踪固定材料线，不能代替完整折叠或网格交叉证明。放置线保持中心误差≤20 mm、区内节点≥95%，不根据某个平台放宽；本表是单次回放描述，不是质量排名或成功率。\n\n"
    text += "铺放优化：降低近边结束高度，远边轻触后缓慢铺回；先原地张开，再退出并收回朝前指尖，避免夹指拖回已铺布边。所有平台沿用同一场景参数。碰撞检查见 clearance.json。\n"
    (task_output / "REPORT.md").write_text(text)
    (task_output / "README.md").write_text(
        "# M03 最新一套\n\n[对照视频](preview/comparison.mp4) · [原生视频](preview/dexsim.mp4) · [完整记录](REPORT.md)\n\n仅保留当前 inputs / preview / source / logs；视频待用户审阅，未开始性能测量。\n"
    )
    write_json(
        task_output / "cross_platform_completion.json",
        dict(
            scope="latest_scene_preview_only",
            human_comparison_approved=False,
            performance_measurements_started=False,
            video=video,
            observations=observations,
            final_layer_max_z_m={
                p: states[p]["particles"][-1, ..., 2].max(1) for p in PLATFORMS
            },
            platform_manifest_sha256={
                p: digest(task_output / "preview" / p / "manifest.json")
                for p in PLATFORMS
            },
            input_manifest_sha256=digest(task_output / "inputs/manifest.json"),
        ),
    )
    print(task_output / "preview/comparison.mp4", flush=True)


if __name__ == "__main__":
    main()
