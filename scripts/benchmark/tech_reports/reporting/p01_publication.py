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

"""Build the public P-01 chapter as a paginated PDF and offline media article."""

from __future__ import annotations

import os

import base64
import hashlib
import html
import json
import re
from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

ROOT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[4])
).resolve()
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "P-01/publication"
FONT_DIR = ROOT / "assets/fonts"
pdfmetrics.registerFont(TTFont("Noto", str(FONT_DIR / "NotoSansSC-Regular.ttf")))
pdfmetrics.registerFont(TTFont("NotoBold", str(FONT_DIR / "NotoSansSC-Semibold.ttf")))
pdfmetrics.registerFontFamily(
    "Noto", normal="Noto", bold="NotoBold", italic="Noto", boldItalic="NotoBold"
)
summary = json.loads((REPORTS / "P-01/summary.json").read_text())
rows = {(r["case"], r["dt_s"]): r for r in summary}
INK, MUTED, BLUE, TEAL, LINE = "#172D40", "#596E7D", "#2568AC", "#008284", "#DCE5EB"
WIDTH, HEIGHT, MARGIN = 595.276, 841.89, 48
CONTENT = WIDTH - 2 * MARGIN
PDF = REPORTS / "P-01-isaaclab.pdf"
C = canvas.Canvas(str(PDF), pagesize=(WIDTH, HEIGHT), pageCompression=1)
C.setTitle("基础动力学准确性 自由落体与无驱动单摆")
C.setAuthor("机器人仿真引擎技术报告")
C.setSubject("Isaac Lab GPU PhysX TGS 的自由落体与单摆实验")
article = []
markdown = []
layout = []


def paragraph(
    text: str,
    top: float,
    size: float = 10.3,
    leading: float = 17,
    color: str = INK,
    width: float = CONTENT,
    x: float = MARGIN,
    bold: bool = False,
) -> float:
    style = ParagraphStyle(
        "body",
        fontName="NotoBold" if bold else "Noto",
        fontSize=size,
        leading=leading,
        textColor=colors.HexColor(color),
        wordWrap="CJK",
        alignment=TA_LEFT,
        splitLongWords=True,
    )
    p = Paragraph(text, style)
    _, height = p.wrap(width, 1000)
    assert top + height <= 783, (text[:50], top, height)
    p.drawOn(C, x, HEIGHT - top - height)
    layout.append(
        {
            "page": C.getPageNumber(),
            "top": top,
            "bottom": top + height,
            "text": re.sub("<[^>]+>", "", text),
        }
    )
    return top + height


def prose(text: str, top: float, **kwargs) -> float:
    article.append(f"<p>{text}</p>")
    markdown.append(
        re.sub(r"<b>(.*?)</b>", r"**\1**", text).replace("<br/>", "\n") + "\n"
    )
    return paragraph(text, top, **kwargs)


def section(number: int, title: str, subtitle: str) -> float:
    if number > 1:
        C.showPage()
    C.bookmarkPage(f"s{number}")
    C.addOutlineEntry(title, f"s{number}")
    C.setFillColor(colors.HexColor(MUTED))
    C.setFont("Noto", 8)
    C.drawString(MARGIN, HEIGHT - 33, "机器人仿真引擎技术报告   /   P-01")
    C.drawRightString(WIDTH - MARGIN, HEIGHT - 33, "ISAAC LAB   ·   2026.10")
    C.setStrokeColor(colors.HexColor(LINE))
    C.line(MARGIN, 45, WIDTH - MARGIN, 45)
    C.setFont("Noto", 8)
    C.drawString(MARGIN, 29, "基础动力学准确性   ·   自由落体与无驱动单摆")
    C.drawRightString(WIDTH - MARGIN, 29, f"{number:02d} / 05")
    article.append(
        f'<section id="s{number}"><div class="eyebrow">{number:02d} / P-01</div><h2>{"研究概览" if number == 1 else title}</h2><p class="deck">{subtitle}</p>'
    )
    markdown.append(f"## {title}\n\n{subtitle}\n")
    paragraph(title, 68, 23, 32, "#000000", bold=True)
    return paragraph(subtitle, 109, 10.2, 17, MUTED)


def figure(name: str, caption: str, top: float, height: float | None = None) -> float:
    p = ASSETS / f"{name}.png"
    with Image.open(p) as im:
        ratio = im.height / im.width
    h = height or CONTENT * ratio
    w = h / ratio
    C.drawImage(str(p), MARGIN + (CONTENT - w) / 2, HEIGHT - top - h, width=w, height=h)
    end = paragraph(caption, top + h + 7, 8.4, 13, MUTED)
    data = base64.b64encode(p.read_bytes()).decode()
    article.append(
        f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'
    )
    markdown.append(f"![{caption}](P-01/publication/{name}.png)\n\n{caption}\n")
    return end


def table(
    headers: list[str],
    data: list[list[str]],
    widths: list[float],
    top: float,
    caption: str,
) -> float:
    paragraph(caption, top, 9.3, 15, bold=True)
    cells = []
    for r, values in enumerate([headers] + data):
        cells.append(
            [
                Paragraph(
                    v.replace("\n", "<br/>"),
                    ParagraphStyle(
                        "cell",
                        fontName="NotoBold" if r == 0 else "Noto",
                        fontSize=8.3,
                        leading=12.5,
                        wordWrap="CJK",
                        textColor=colors.HexColor(INK),
                    ),
                )
                for v in values
            ]
        )
    t = Table(cells, colWidths=widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF3F7")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F8FAFB")],
                ),
                ("LINEABOVE", (0, 0), (-1, 0), 0.6, colors.HexColor("#93A8B7")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#93A8B7")),
                ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#93A8B7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    _, h = t.wrap(CONTENT, 1000)
    assert top + 22 + h < 783, caption
    t.drawOn(C, MARGIN, HEIGHT - top - 22 - h)
    article.append(
        '<div class="table-wrap"><table><caption>'
        + caption
        + "</caption><thead><tr>"
        + "".join("<th>" + h.replace("\n", "<br>") + "</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + v.replace("\n", "<br>") + "</td>" for v in rr)
            + "</tr>"
            for rr in data
        )
        + "</tbody></table></div>"
    )
    markdown.append(
        caption
        + "\n\n| "
        + " | ".join(h.replace("\n", " ") for h in headers)
        + " |\n| "
        + " | ".join(["---"] * len(headers))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(v.replace("\n", " ") for v in rr) + " |" for rr in data
        )
        + "\n"
    )
    return top + 22 + h


def video(name: str, caption: str) -> None:
    movie = base64.b64encode((ASSETS / f"{name}_replay.mp4").read_bytes()).decode()
    poster = base64.b64encode(
        (ASSETS / f"{name}_replay_poster.png").read_bytes()
    ).decode()
    article.append(
        f'<figure class="motion"><video controls playsinline preload="metadata" poster="data:image/png;base64,{poster}" aria-label="{html.escape(caption)}"><source src="data:video/mp4;base64,{movie}" type="video/mp4"></video><figcaption>{caption}</figcaption></figure>'
    )
    markdown.append(f"![{caption}](P-01/publication/{name}_replay.gif)\n\n{caption}\n")


# Page 1: give an unfamiliar reader the problem, model and actual conclusion.
section(1, "基础动力学准确性", "自由落体与无驱动单摆   |   Isaac Lab 实验研究")
prose(
    "仿真中的物体既要沿正确轨迹运动，也要保持合理的速度和能量。本章通过自由落体和无驱动单摆，评估 Isaac Lab 在无接触条件下的基础动力学误差，并检查减小时间步长能否改善结果。",
    145,
)
figure(
    "experiment_setup",
    "图 1  两类实验的物理模型。左侧为等时间间隔的位置示意；右侧以固定转轴连接球体。示意图不代表仿真器渲染质量。",
    222,
)
paragraph("主要发现", 524, 16, 23, bold=True)
prose(
    "<b>自由落体的位置误差随步长减小而下降。</b>步长为 2.5 ms 时，前 1 s 的最大位置误差均值为 3.030 mm，10 个初态均满足本研究的判据。",
    561,
)
prose(
    "<b>周期准确并不代表能量保持准确。</b>单摆三档步长的平均周期误差均低于 0.10%，但最大相对能量漂移均值为 20.06% 至 107.54%，均超过 2% 判据。",
    622,
)
prose(
    "<b>求解器设置会显著影响结果。</b>5 ms 下的单初态诊断中，位置迭代由 4 次降为 1 次，最大能量漂移由 44.12% 降至 1.12%；该结果尚未覆盖其他初态。",
    683,
)
article.append("</section>")

# Page 2: sufficient method detail for scientific interpretation, not run logs.
section(
    2, "实验设计与评价方法", "两种模型、三档步长、每档 10 个初态，共 60 条主测轨迹。"
)
prose(
    "每条轨迹持续 10 s，逐物理步记录状态。三个步长分别为 10、5 和 2.5 ms；每次仅运行一个实例。10 个初态用于观察初始条件差异，不能视为同一工况的 10 次独立重复测量。",
    146,
)
table(
    ["项目", "配置"],
    [
        [
            "软件",
            "Isaac Lab 17.0.2（包元数据）；Isaac Sim 6.1.0.0；isaaclab-physx 6.0.0",
        ],
        ["计算设备", "NVIDIA GeForce RTX 5090 D v2；驱动 580.159.04"],
        ["求解设置", "GPU PhysX TGS；位置迭代 4 次，速度迭代 1 次 [1]"],
        ["共同条件", "重力 9.81 m/s²；无碰撞、无驱动、无阻尼；关闭休眠与稳定化"],
        ["自由落体", "立方体边长 50 mm、质量 0.1 kg；初高 0.95–1.05 m，初速为零"],
        [
            "单摆",
            "球体质量 0.1 kg、半径 25 mm；质心距轴 0.5 m；初角 9°–11°，初始角速度为零 [2]",
        ],
    ],
    [86, CONTENT - 86],
    212,
    "表 1  主测条件",
)
paragraph("独立参考与指标", 445, 14, 21, bold=True)
prose(
    "落体采用解析解；单摆包含球体自身惯量，使用独立数值积分求解。角度、角速度、位置和速度均按相同物理时刻比较，不做平滑或相位补偿。",
    476,
)
figure(
    "equations",
    "其中 θ 为摆角、ω 为角速度，L 为转轴到质心的距离，r 为球体半径。能量以最低点为势能零点。",
    526,
    76,
)
criteria_end = table(
    ["实验", "本研究判据", "评价窗口"],
    [
        ["自由落体", "最大位置误差 ≤5 mm；最大速度误差 ≤0.01 m/s", "0–1 s"],
        ["单摆", "周期相对误差 ≤1%；最大相对能量漂移 ≤2%", "0–10 s"],
    ],
    [72, CONTENT - 144, 72],
    637,
    "表 2  预先设定的评价标准（均需同时满足）",
)
prose(
    "周期取同方向过零点的平均间隔，过零时刻线性插值。参考从实际读回初态出发；10 个初态由固定种子 20261014 生成，三档步长共享同一初态数组。",
    criteria_end + 10,
    size=8.6,
    leading=13,
    color=MUTED,
)
article.append("</section>")

# Page 3: figure, compact numerical table, then interpretation and limits.
section(
    3,
    "自由落体的位置精度随步长改善",
    "最小步长满足前 1 s 的判据，完整 10 s 的误差仍需单独观察。",
)
figure(
    "freefall_accuracy",
    "图 2  自由落体误差。A 为初态 0；B 为 10 初态的均值与最小至最大范围，范围较小，接近标记尺寸。虚线为 5 mm 判据。",
    147,
    192,
)
fall = []
for dt in [0.01, 0.005, 0.0025]:
    r = rows["freefall", dt]
    fall.append(
        [
            f"{dt*1000:g}",
            f"{r['position_max_0_1s_m']['mean']*1000:.3f}",
            f"{r['velocity_max_0_1s_m_s']['mean']*1000:.4f}",
            f"{r['position_max_error']['mean']*1000:.3f}",
            f"{r['velocity_max_error']['mean']*1000:.3f}",
            f"{r['candidate_pass_count']}/10",
        ]
    )
table(
    [
        "步长\n(ms)",
        "0–1 s 最大\n位置误差 (mm)",
        "0–1 s 最大\n速度误差 (mm/s)",
        "0–10 s 最大\n位置误差 (mm)",
        "0–10 s 最大\n速度误差 (mm/s)",
        "通过\n初态数",
    ],
    fall,
    [40, 94, 102, 94, 105, CONTENT - 435],
    392,
    "表 3  自由落体误差（各初态最大误差的均值）",
)
prose(
    "前 1 s 内，步长每减半一次，最大位置误差也近似减半：12.268、6.117 和 3.030 mm。这表明本次区间内的位置误差呈一阶下降趋势；5 ms 和 10 ms 的失败均来自位置误差，速度指标全部通过。",
    553,
)
prose(
    "完整 10 s 的位置误差同样下降，但速度误差并非单调改善。2.5 ms 的最大速度误差均值为 12.206 mm/s，高于另外两档。本次未分离内部误差来源，不能把这一现象直接归因于某个积分环节。",
    620,
)
prose(
    "无地面条件下，10 s 的理论下落距离为 490.5 m。因此，完整轨迹用于检查误差积累，前 1 s 用于评价较短运动窗口的绝对误差；前 1 s 达标不等于全程达标。",
    687,
)
video(
    "freefall",
    "动画 1  自由落体实测轨迹重建。左侧为 2.5 ms、初态 0 的运动；右侧比较三档步长的位置误差。展示 0–1 s，以 0.25 倍速播放；方块图标放大，位移按比例。",
)
article.append("</section>")

# Page 4: make the important failure legible rather than burying it in a table.
section(
    4,
    "单摆周期接近参考但能量持续增长",
    "三档步长均通过周期判据，均未通过能量漂移判据。",
)
figure(
    "pendulum_accuracy",
    "图 3  单摆实测与独立参考。曲线均为初态 0；A 展示 5 ms 的摆角，B 展示三档步长的能量变化。表 4 汇总全部 10 初态。",
    144,
    230,
)
pend = []
for dt in [0.01, 0.005, 0.0025]:
    r = rows["pendulum", dt]
    pend.append(
        [
            f"{dt*1000:g}",
            f"{r['position_rmse']['mean']*180/3.141592653589793:.3f}",
            f"{r['period_relative_error']['mean']*100:.4f}",
            f"{r['energy_max_relative_drift']['mean']*100:.2f}",
            f"{r['candidate_pass_count']}/10",
        ]
    )
table(
    [
        "步长\n(ms)",
        "摆角 RMSE\n(°)",
        "周期相对误差\n(%)",
        "最大相对能量漂移\n(%)",
        "通过\n初态数",
    ],
    pend,
    [47, 96, 119, 151, CONTENT - 413],
    420,
    "表 4  单摆误差（10 初态指标均值）",
)
prose(
    "RMSE 表示整段轨迹的均方根误差。步长减小时，摆角 RMSE 和能量漂移都下降，但 2.5 ms 下的最大能量漂移均值仍为 20.06%，明显超过 2% 判据。三档对应的绝对能量漂移均值分别为 8.47、3.47 和 1.58 mJ。",
    586,
)
prose(
    "振幅增长在 10 个初态中均出现。以 5 ms、初态 0 为例，初始摆角约 9.994°，最后 2 s 的最大绝对摆角约 11.994°。这说明仅凭摆动周期或视觉上的往复运动，无法判断能量保持是否准确。",
    659,
)
video(
    "pendulum",
    "动画 2  单摆实测轨迹重建。蓝色为 5 ms、初态 0；灰色轮廓为独立参考。右侧同步显示能量变化。展示 0–10 s，按真实物理时间播放；连线仅示意质心距离。",
)
article.append("</section>")

# Page 5: controlled diagnostic and a conclusion bounded by the evidence.
section(5, "迭代设置的影响与结论", "补充对照揭示配置敏感性，尚未定位内部误差机制。")
prose(
    "位置迭代是每个物理步内的求解设置。保持 5 ms 步长、初态 0、模型参数和速度迭代 1 次不变，仅将位置迭代从 4 次改为 1 次；该诊断未替换主测结果。",
    144,
)
figure(
    "iteration_sensitivity",
    "图 4  单初态诊断。减少位置迭代后，能量变化和摆角误差均明显缩小；这一结果仅适用于当前无接触模型及该初态。",
    205,
    144,
)
table(
    ["位置迭代", "周期误差 (%)", "最大能量漂移 (%)", "摆角 RMSE (°)"],
    [
        ["4 次（主测）", "0.03954", "44.124", "0.808"],
        ["1 次（诊断）", "0.00197", "1.117", "0.081"],
    ],
    [105, 115, 160, CONTENT - 380],
    391,
    "表 5  5 ms、初态 0 的同条件比较",
)
prose(
    "<b>结论。</b>在 GPU PhysX TGS 的 4／1 迭代配置下，自由落体在 2.5 ms 时达到前 1 s 的精度要求；单摆在三个步长下均未满足能量漂移要求，因此该主配置不能视为本项全部通过。1 次位置迭代值得进一步验证，但不能据单次结果推荐给接触或抓取任务。",
    514,
)
prose(
    "<b>可信度与边界。</b>已核对质量、质心、惯量及零驱动参数。RK4 参考步长由 0.1 ms 减为 0.05 ms 时，最大角度差为 8.44×10<super>−15</super> rad；初态 0 与 DOP853 的最大角度差为 6.41×10<super>−13</super> rad。本章只评估无接触动力学，不涉及渲染质量、运行吞吐或跨引擎优劣。评价门槛为本研究预设标准，不是行业认证。",
    602,
    size=9.4,
    leading=15.5,
)
paragraph("方法与版本说明", 707, 9, 14, bold=True)
paragraph(
    'Isaac Lab 源码提交：ae37b028ea415c91ea2bc32609efcd759ed2b974。<br/>[1] NVIDIA PhysX 文档：<link href="https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html#island-management" color="#2568AC">Simulation / Island Management</link>。<br/>[2] NVIDIA PhysX 文档：<link href="https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html" color="#2568AC">Articulations</link>。文档用于方法背景，不代表本机 SDK 版本。',
    727,
    7.5,
    11.5,
    MUTED,
)
article.append(
    '<div class="references"><h3>方法与版本说明</h3><p>实测日期：2026-10-14。Isaac Lab 源码提交：<code>ae37b028ea415c91ea2bc32609efcd759ed2b974</code>。</p><p>[1] NVIDIA PhysX 文档：<a href="https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html#island-management">Simulation / Island Management</a>。<br>[2] NVIDIA PhysX 文档：<a href="https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html">Articulations</a>。文档用于方法背景，不代表本机 SDK 版本。</p></div></section>'
)
markdown.append(
    "方法与版本说明：实测日期 2026-10-14；Isaac Lab 源码提交 `ae37b028ea415c91ea2bc32609efcd759ed2b974`。\n"
)
C.save()

css = """
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf2f5;color:#172d40;font-family:'Noto Sans SC','Noto Sans CJK SC','Microsoft YaHei',sans-serif;line-height:1.9}main{max-width:1040px;margin:40px auto;background:#fff;padding:65px 80px;box-shadow:0 12px 50px #1731470b}header{margin-bottom:44px}header .series{font-size:13px;letter-spacing:.16em;color:#596e7d}h1{font-size:38px;font-weight:600;letter-spacing:.02em;margin:12px 0;color:#000}header .subtitle{font-size:20px;color:#596e7d}nav{display:flex;gap:24px;flex-wrap:wrap;margin-top:30px;padding:17px 0;border-top:1px solid #dce5eb;border-bottom:1px solid #dce5eb;font-size:14px}a{color:#2568ac;text-decoration:none}a:hover{text-decoration:underline}section{padding-top:34px;margin-bottom:48px;scroll-margin-top:20px}.eyebrow{font-size:12px;letter-spacing:.12em;color:#008284}h2{font-size:27px;line-height:1.4;margin:8px 0 12px;color:#000;font-weight:600}h3{font-size:17px;font-weight:600}.deck{color:#596e7d;font-size:17px;margin-top:0}p{font-size:16px;margin:22px 0}b{font-weight:600}figure{margin:32px 0}figure img,video{width:100%;height:auto;display:block;background:#f5f8fa}figcaption{font-size:13px;line-height:1.7;color:#596e7d;margin-top:12px}.table-wrap{overflow-x:auto;margin:30px 0}table{width:100%;border-collapse:collapse;font-size:14px;line-height:1.6}caption{text-align:left;font-size:14px;font-weight:600;margin:0 0 13px}th,td{padding:13px 12px;text-align:left;border-bottom:1px solid #dce5eb;vertical-align:middle}th{background:#edf3f7;font-weight:600;border-top:1px solid #93a8b7}tbody tr:nth-child(even){background:#f8fafb}.references{border-top:1px solid #dce5eb;padding-top:16px}.references p{font-size:13px;word-break:break-word}footer{padding-top:24px;border-top:1px solid #dce5eb;font-size:12px;color:#596e7d}code{overflow-wrap:anywhere}sup{font-size:.7em} @media(max-width:760px){main{margin:0;padding:30px 22px}h1{font-size:29px}h2{font-size:23px}p{font-size:15px}nav{gap:14px}table{min-width:600px}figure{margin:24px 0}}@media print{body{background:#fff}main{box-shadow:none;margin:0;padding:0;max-width:none}nav,.motion{display:none}section{break-before:page}section:first-of-type{break-before:auto}}
"""
# A single HTML file contains all media and needs no network access to play it.
body = "\n".join(article).replace("<super>", "<sup>").replace("</super>", "</sup>")
output = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>基础动力学准确性 自由落体与无驱动单摆</title><style>{css}</style></head><body><main><header><div class="series">机器人仿真引擎技术报告 · P-01</div><h1>基础动力学准确性</h1><div class="subtitle">自由落体与无驱动单摆 · Isaac Lab</div><nav aria-label="章节导航"><a href="#s1">研究概览</a><a href="#s2">实验方法</a><a href="#s3">自由落体</a><a href="#s4">单摆</a><a href="#s5">诊断与结论</a></nav></header>{body}<footer>2026-10-14 · 本文动画为已保存实验状态的轨迹重建，非仿真器原生录像。视频显示帧率与物理步长分别定义。</footer></main></body></html>"""
(REPORTS / "P-01-isaaclab.html").write_text(output)
md = (
    "# 基础动力学准确性\n\n自由落体与无驱动单摆 · Isaac Lab · 2026-10-14\n\n[排版版 PDF](P-01-isaaclab.pdf) · [含动画阅读版](P-01-isaaclab.html)\n\n"
    + "\n".join(markdown)
)
(REPORTS / "P-01/internal/publication-source-generated.md").write_text(
    md.replace("<super>", "<sup>")
    .replace("</super>", "</sup>")
    .replace("](P-01/publication/", "](../publication/")
    .replace("](P-01-isaaclab.", "](../../P-01-isaaclab.")
)
(ROOT / "tmp/P-01-publication/layout.json").write_text(
    json.dumps(layout, ensure_ascii=False, indent=2)
)
manifest = {
    "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "summary_sha256": hashlib.sha256(
        (REPORTS / "P-01/summary.json").read_bytes()
    ).hexdigest(),
    "outputs": {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [PDF, REPORTS / "P-01-isaaclab.html"]
    },
}
(ASSETS / "publication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Created {PDF} and standalone animated HTML")
