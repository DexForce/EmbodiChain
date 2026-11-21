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

"""Build publication figures and explicitly labelled replays from P-01 data."""

from __future__ import annotations

import os

import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
OUT = ROOT / "reports/P-01/publication"
FONT = ROOT / "assets/fonts/NotoSansSC-Regular.ttf"
BOLD = ROOT / "assets/fonts/NotoSansSC-Semibold.ttf"
COLORS = ["#CB6034", "#2568AC", "#008284"]
INK, MUTED, GRID, BG = "#172D40", "#596E7D", "#DCE5EB", "#F5F8FA"
OUT.mkdir(parents=True, exist_ok=True)
font_manager.fontManager.addfont(str(FONT))
plt.rcParams.update(
    {
        "font.family": "Noto Sans SC",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#AABAC5",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
    }
)
summary = json.loads((ROOT / "reports/P-01/summary.json").read_text())
traces = {}
inputs = {}
for case in ["freefall", "pendulum"]:
    for dt, tag in [(0.01, "010"), (0.005, "005"), (0.0025, "0025")]:
        p = ROOT / f"results/P-01/20261014_main_{case}_dt{tag}/episode_00.npz"
        with np.load(p) as a:
            traces[case, dt] = (a["t"].copy(), a["state"].copy())
        inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
p = ROOT / "results/P-01/20261014_diagnostic_pendulum_dt005_iter1/episode_00.npz"
with np.load(p) as a:
    diagnostic = (a["t"].copy(), a["state"].copy())
inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
p = ROOT / "reports/P-01/pendulum_reference.npz"
with np.load(p) as a:
    ref_t, ref = a["t"].copy(), a["state"][:, 0].copy()
inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()


def save(fig: plt.Figure, name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(
            OUT / f"{name}.{ext}", dpi=320, facecolor="white", bbox_inches="tight"
        )
    plt.close(fig)


def energy(state: np.ndarray) -> np.ndarray:
    e = 0.025025 * state[:, 1] ** 2 / 2 + 0.4905 * (1 - np.cos(state[:, 0]))
    return (e / e[0] - 1) * 100


# Scientific figures: actual sample times; no smoothing or phase correction.
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), layout="constrained")
for dt, color in zip([0.01, 0.005, 0.0025], COLORS):
    t, s = traces["freefall", dt]
    mask = t <= 1
    e = np.abs(s[:, 0] - (s[0, 0] - 4.905 * t**2)) * 1000
    axes[0].plot(t[mask], e[mask], color=color, lw=2, label=f"{dt*1000:g} ms")
    row = next(x for x in summary if x["case"] == "freefall" and x["dt_s"] == dt)
    v = row["position_max_0_1s_m"]
    axes[1].errorbar(
        dt * 1000,
        v["mean"] * 1000,
        yerr=np.array([[v["mean"] - v["min"]], [v["max"] - v["mean"]]]) * 1000,
        color=color,
        fmt="o",
        ms=8,
        capsize=4,
        zorder=3,
    )
    axes[1].annotate(
        f'{v["mean"]*1000:.3f}',
        (dt * 1000, v["mean"] * 1000),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        color=color,
        fontsize=10,
    )
for ax in axes:
    ax.axhline(5, color=MUTED, linestyle=(0, (4, 3)), lw=1)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_ylim(0, 15)
    ax.set_ylabel("位置绝对误差（mm）")
axes[0].set(xlabel="物理时间（s）", xlim=(0, 1), title="A  前 1 s 的误差轨迹 · 初态 0")
axes[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
axes[1].set(
    xlabel="物理步长（ms）", xlim=(1.5, 11), title="B  最大误差 · 10 初态均值与范围"
)
axes[1].set_xticks([2.5, 5, 10])
axes[1].text(10.8, 5.3, "5 mm 判据", ha="right", color=MUTED, fontsize=9)
save(fig, "freefall_accuracy")

fig, axes = plt.subplots(2, 1, figsize=(10.2, 5), layout="constrained", sharex=True)
t, s = traces["pendulum", 0.005]
axes[0].plot(t, np.rad2deg(s[:, 0]), color=COLORS[1], lw=1.8, label="Isaac Lab · 5 ms")
axes[0].plot(ref_t, np.rad2deg(ref[:, 0]), color=INK, lw=1.3, ls="--", label="独立参考")
axes[0].set(
    ylabel="摆角（°）", title="A  周期接近参考，振幅却逐渐增加 · 初态 0", ylim=(-15, 15)
)
axes[0].legend(frameon=False, ncol=2, loc="lower left", fontsize=9)
for dt, color in zip([0.01, 0.005, 0.0025], COLORS):
    t, s = traces["pendulum", dt]
    axes[1].plot(t, energy(s), color=color, lw=1.8, label=f"{dt*1000:g} ms")
axes[1].axhline(2, color=MUTED, lw=1, ls="--")
axes[1].set(
    ylabel="能量变化（%）",
    xlabel="物理时间（s）",
    title="B  能量相对初值的变化 · 初态 0",
    ylim=(-5, 120),
    xlim=(0, 10),
)
axes[1].legend(frameon=False, ncol=3, loc="upper left", fontsize=9)
axes[1].text(9.85, 6, "2% 判据", ha="right", fontsize=9, color=MUTED)
for ax in axes:
    ax.grid(axis="y", color=GRID, lw=0.7)
save(fig, "pendulum_accuracy")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2), layout="constrained")
for (t, s), color, name in [
    (traces["pendulum", 0.005], COLORS[1], "位置迭代 4 次"),
    (diagnostic, COLORS[2], "位置迭代 1 次"),
]:
    axes[0].plot(t, energy(s), color=color, lw=1.8, label=name)
    r = np.interp(t, ref_t, ref[:, 0])
    axes[1].plot(t, np.rad2deg(s[:, 0] - r), color=color, lw=1.5)
axes[0].axhline(2, color=MUTED, ls="--", lw=1)
axes[0].axhline(-2, color=MUTED, ls="--", lw=1)
axes[0].set(ylabel="能量变化（%）", ylim=(-5, 50), title="A  能量变化")
axes[0].legend(frameon=False, loc="upper left", fontsize=9)
axes[1].set(ylabel="摆角误差（°）", title="B  与独立参考的摆角差")
for ax in axes:
    ax.set(xlabel="物理时间（s）", xlim=(0, 10))
    ax.grid(axis="y", color=GRID, lw=0.7)
save(fig, "iteration_sensitivity")

# Equations use real typesetting, separately from explanatory Chinese text.
fig = plt.figure(figsize=(10.2, 1.7))
fig.text(
    0.025,
    0.77,
    r"$z_{\mathrm{ref}}(t)=z_0-\frac{1}{2}gt^2,\quad v_{\mathrm{ref}}(t)=-gt$",
    fontsize=16,
)
fig.text(
    0.025,
    0.42,
    r"$I_p\ddot{\theta}=-mgL\sin\theta,\quad I_p=m(L^2+\frac{2}{5}r^2)$",
    fontsize=16,
)
fig.text(
    0.025,
    0.06,
    r"$E(t)=\frac{1}{2}I_p\omega^2+mgL(1-\cos\theta),\quad \varepsilon_E=\max_t\frac{|E(t)-E(0)|}{E(0)}$",
    fontsize=16,
)
save(fig, "equations")

fonts = {
    n: ImageFont.truetype(str(FONT), n) for n in [16, 18, 20, 22, 24, 26, 28, 32, 36]
}
bolds = {n: ImageFont.truetype(str(BOLD), n) for n in [22, 26, 28, 32, 36, 42]}


def label(
    d: ImageDraw.ImageDraw,
    xy: tuple,
    text: str,
    size: int = 24,
    fill: str = INK,
    bold: bool = False,
) -> None:
    d.text(xy, text, font=(bolds if bold else fonts)[size], fill=fill)


def ball(
    d: ImageDraw.ImageDraw,
    x: float,
    y: float,
    r: float,
    color: str,
    outline: bool = False,
) -> None:
    d.ellipse(
        (x - r, y - r, x + r, y + r),
        fill=BG if outline else color,
        outline=color,
        width=3,
    )
    if not outline:
        d.ellipse(
            (x - r * 0.55, y - r * 0.6, x - r * 0.05, y - r * 0.1), fill="#8DBEDF"
        )


# Model illustration: metric lengths on the pendulum, enlarged falling object icon.
im = Image.new("RGB", (1440, 720), BG)
d = ImageDraw.Draw(im)
label(d, (45, 30), "A  自由落体", 32, bold=True)
label(d, (760, 30), "B  无驱动单摆", 32, bold=True)
d.line((712, 35, 712, 670), fill=GRID, width=2)
label(d, (45, 86), "检查重力作用下的位置与速度", 22, MUTED)
label(d, (760, 86), "检查关节运动的周期与能量保持", 22, MUTED)
for i, t in enumerate([0, 0.25, 0.5, 0.75, 1]):
    y = 155 + 450 * t * t
    x = 185
    d.rectangle(
        (x - 16, y - 16, x + 16, y + 16),
        fill=COLORS[1] if i == 4 else "#BED6E7",
        outline=COLORS[1],
        width=2,
    )
    label(d, (242, y - 17), f"{t:.2f} s", 22)
d.line((120, 155, 120, 605), fill=MUTED, width=2)
for y in [155, 605]:
    d.line((110, y, 130, y), fill=MUTED, width=2)
label(d, (370, 195), "质量 0.1 kg", 24)
label(d, (370, 238), "边长 50 mm", 24)
label(d, (370, 310), "g = 9.81 m/s²", 24)
d.line((460, 380, 460, 478), fill=COLORS[1], width=4)
d.polygon([(448, 466), (472, 466), (460, 488)], fill=COLORS[1])
label(d, (370, 517), "无地面、无碰撞", 22, MUTED)
label(d, (45, 650), "等时间间隔位置示意；方块图标放大", 20, MUTED)
ox, oy, length = 1068, 164, 410
angle = np.deg2rad(10)
bx, by = ox - length * np.sin(angle), oy + length * np.cos(angle)
d.rectangle((1015, 143, 1121, 160), fill=INK)
d.line((ox, oy, ox, oy + length + 22), fill="#9CAFBF", width=2)
d.line((ox, oy, bx, by), fill=COLORS[1], width=6)
d.arc((ox - 94, oy - 94, ox + 94, oy + 94), 90, 100, fill=COLORS[0], width=3)
ball(d, bx, by, 21, COLORS[1])
ball(d, ox, oy, 6, INK)
label(d, (1125, 155), "固定转轴", 22)
label(d, (800, 295), "L = 0.5 m", 24)
label(d, (1110, 285), "初角 9°–11°", 22)
label(d, (1100, 504), "质量 0.1 kg", 22)
label(d, (1100, 543), "半径 25 mm", 22)
label(d, (760, 650), "连线仅示意质心距离；无有质量连接杆", 20, MUTED)
im.save(OUT / "experiment_setup.png")


def replay_frame(case: str, time: float) -> Image.Image:
    im = Image.new("RGB", (1280, 720), BG)
    d = ImageDraw.Draw(im)
    is_pend = case == "pendulum"
    dt = 0.005 if is_pend else 0.0025
    t, s = traces[case, dt]
    label(
        d,
        (38, 25),
        "单摆的振幅与能量" if is_pend else "自由落体的位置误差",
        36,
        bold=True,
    )
    label(
        d,
        (39, 82),
        f"实测轨迹重建  |  初态 0  |  {dt*1000:g} ms  |  "
        + ("1× 播放" if is_pend else "0.25× 慢放"),
        22,
        MUTED,
    )
    label(d, (1000, 32), f"t = {time:05.2f} s", 28, bold=True)
    d.line((36, 129, 1244, 129), fill=GRID, width=2)
    gx, gy, gw, gh = 650, 205, 555, 350
    if is_pend:
        theta = float(np.interp(time, t, s[:, 0]))
        theta_ref = float(np.interp(time, ref_t, ref[:, 0]))
        ox, oy, length = 305, 180, 365
        d.rectangle((255, 162, 355, 178), fill=INK)
        d.line((ox, oy, ox, oy + length + 22), fill=GRID, width=2)
        for q, color, outline in [(theta_ref, MUTED, True), (theta, COLORS[1], False)]:
            x, y = ox - length * np.sin(q), oy + length * np.cos(q)
            d.line((ox, oy, x, y), fill=color, width=3 if outline else 5)
            ball(d, x, y, length * 0.05, color, outline)
        ball(d, ox, oy, 6, INK)
        label(d, (57, 584), f"实测摆角  {np.rad2deg(theta):+.2f}°", 26, COLORS[1])
        label(d, (57, 625), f"参考摆角  {np.rad2deg(theta_ref):+.2f}°", 22, MUTED)
        ymax, xmax = 50, 10
        ys = energy(s)
        lines = [(t, ys, COLORS[1])]
        label(d, (gx, 156), "能量相对初值的变化（%）", 24, bold=False)
        label(
            d,
            (gx, 607),
            f"当前能量变化  {np.interp(time,t,ys):+.2f}%",
            28,
            COLORS[1],
            True,
        )
    else:
        z0 = s[0, 0]
        drop = z0 - float(np.interp(time, t, s[:, 0]))
        drop_ref = 4.905 * time * time
        for x, distance, color, name in [
            (240, drop, COLORS[2], "实测"),
            (410, drop_ref, MUTED, "解析参考"),
        ]:
            label(d, (x - 35, 155), name, 22, color)
            d.line((x, 210, x, 550), fill=GRID, width=2)
            y = 210 + 340 * distance / 5
            d.rectangle((x - 13, y - 13, x + 13, y + 13), fill=color)
        for i in range(6):
            y = 210 + 340 * i / 5
            d.line((125, y, 145, y), fill=MUTED, width=2)
            label(d, (67, y - 13), f"{i} m", 18, MUTED)
        label(d, (58, 587), "方块图标放大；下落距离按比例", 20, MUTED)
        label(d, (58, 628), "无地面，完整主测持续 10 s", 20, MUTED)
        ymax, xmax = 15, 1
        lines = []
        for dd, color in zip([0.01, 0.005, 0.0025], COLORS):
            tt, ss = traces["freefall", dd]
            mask = tt <= 1
            ee = np.abs(ss[:, 0] - (ss[0, 0] - 4.905 * tt * tt)) * 1000
            lines.append((tt[mask], ee[mask], color))
        label(d, (gx, 156), "位置绝对误差（mm）", 24)
        for idx, (name, color) in enumerate(zip(["10 ms", "5 ms", "2.5 ms"], COLORS)):
            label(d, (gx + idx * 180, 607), name, 24, color)
    # Shared axes and partial curves end at the displayed physical time.
    for tick in np.linspace(0, ymax, 6):
        y = gy + gh * (1 - tick / ymax)
        d.line((gx, y, gx + gw, y), fill=GRID, width=1)
        label(d, (gx - 43, y - 13), f"{tick:g}", 18, MUTED)
    for tick in np.linspace(0, xmax, 6):
        x = gx + gw * tick / xmax
        label(d, (x - 8, gy + gh + 9), f"{tick:g}", 18, MUTED)
    label(d, (1050, gy + gh + 40), "物理时间（s）", 18, MUTED)
    for tt, yy, color in lines:
        sel = tt <= time + 1e-10
        points = [
            (gx + gw * float(tx) / xmax, gy + gh * (1 - float(v) / ymax))
            for tx, v in zip(tt[sel], yy[sel])
        ]
        if len(points) > 1:
            d.line(points, fill=color, width=3)
        if points:
            x, y = points[-1]
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    label(
        d,
        (38, 681),
        "依据原始状态逐帧绘制，帧间线性插值；非仿真器原生录像。",
        18,
        MUTED,
    )
    return im


for case, duration, rate in [("freefall", 1.0, 0.25), ("pendulum", 10.0, 1.0)]:
    fps = 30
    frames = round(duration / rate * fps)
    dest = OUT / f"{case}_replay.mp4"
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1280x720",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    for i in range(frames):
        # Last encoded frame is one frame before the window end; no endpoint hold.
        frame = replay_frame(case, i / fps * rate)
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    replay_frame(case, 0.75 if case == "freefall" else 9.2).save(
        OUT / f"{case}_replay_poster.png"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(dest),
            "-filter_complex",
            "fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
            "-loop",
            "0",
            str(OUT / f"{case}_replay.gif"),
        ],
        check=True,
    )

# Four precisely timestamped samples provide a printable experimental sequence.
im = Image.new("RGB", (1440, 410), BG)
d = ImageDraw.Draw(im)
for i, time in enumerate([0.0, 2.84, 5.69, 8.54]):
    x0 = 180 + i * 360
    theta = float(
        np.interp(
            time, traces["pendulum", 0.005][0], traces["pendulum", 0.005][1][:, 0]
        )
    )
    qref = float(np.interp(time, ref_t, ref[:, 0]))
    label(d, (x0 - 83, 15), f"t = {time:.2f} s", 24, bold=False)
    ox, oy, length = x0, 87, 235
    d.rectangle((ox - 27, oy - 9, ox + 27, oy - 2), fill=INK)
    for q, c, o in [(qref, MUTED, True), (theta, COLORS[1], False)]:
        bx, by = ox - length * np.sin(q), oy + length * np.cos(q)
        d.line((ox, oy, bx, by), fill=c, width=3)
        ball(d, bx, by, length * 0.05, c, o)
    label(d, (x0 - 78, 355), f"摆角 {np.rad2deg(theta):.2f}°", 22, COLORS[1])
im.save(OUT / "pendulum_sequence.png")

manifest = {
    "source_inputs": inputs,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "visualization": "Scientific diagrams and data-driven trajectory replays; not native simulator footage.",
    "pendulum": {
        "run": "20261014_main_pendulum_dt005",
        "initial_id": 0,
        "physical_window_s": [0, 10],
        "playback_speed": 1,
        "fps": 30,
        "interpolation": "linear",
    },
    "freefall": {
        "run": "20261014_main_freefall_dt0025",
        "initial_id": 0,
        "physical_window_s": [0, 1],
        "playback_speed": 0.25,
        "fps": 30,
        "interpolation": "linear",
        "object_icon": "enlarged",
    },
}
(OUT / "media_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
)
print(f"Publication figures and replays saved to {OUT}")
