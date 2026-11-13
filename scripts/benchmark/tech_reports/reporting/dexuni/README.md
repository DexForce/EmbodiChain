# DexUni 离线报告与媒体

当前报告由 `physics/experiments/dexuni/run_s01.py`、`run_s02.py`、`run_s03.py` 读取明确 run 路径的原始状态生成，支持 --rebuild-report。
`build_s01_media.py` 合成 S01-C/T 对比视频、仿真时间映射和响应曲线；两侧视频结束时间不同则显示 ended。
新报告保留 Time & Memory、Success & Other Metrics、Leaderboard 三表，整体平台比较不做可靠性排名。

`s01c/`、`s01t/` 的 analyze.py 中 task_metrics 继续提供几何判据复算；旧 main 的目录筛选、全帧计时和视频合成不用于本轮。
`s02/`、`s03/` 的 analyze.py 提供各任务指标；build_s02_media.py/build_s03_media.py 与
plot_s02_repetitions.py/plot_s03_repetitions.py 合成独立媒体和全五次重复图，同时导出绘图 CSV。
parity、compare_models 保留历史诊断用途；旧工作区已删除，旧 S02 试跑证据见清理账本。

命令见 [SCRIPT_CONTRACT](../../physics/experiments/dexuni/SCRIPT_CONTRACT.md)，当前范围见 [PLAN](../../physics/experiments/dexuni/PLAN.md)。
所有生成产物在 outputs/benchmarks/tech_reports/dexuni/；原始数据不写回源码/资产目录。

## 已发布结果与原始数据

[13 项任务对比](RESULTS.md)提供各任务报告入口，并明确正式测量、摸测及失败条件。
[Hugging Face 报告](https://huggingface.co/datasets/DexForceAI/embodichain_reports/tree/cafe3e14f650f93123d47fe338e49c003443f3a4/reports/dexuni-physics-2026-10-27)
提供可浏览的原始 JSON/CSV、输入资产和逐文件哈希。视频和完整结果大包仅在本地保留。
按[恢复命令](../../physics/experiments/dexuni/SETUP.md)准备输入；完整数组复算仍需本地保留的测量结果。

## 单文件 HTML 总览

从本地保留的结果生成可离线浏览的交互报告：

```bash
.venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_html_report.py
```

产物为 `outputs/benchmarks/tech_reports/dexuni/report.html`，可直接用浏览器打开。
页面包含 13 个任务、正式测量对比图、分类/搜索、逐项效果与配置、C02 两个计时窗口、
可选 SDK 冷启动复验，以及 CSV/JSON 导出和打印样式。数据与录像截帧内嵌，无 CDN 或网络脚本。

生成器只读 `comparison.json` 及其冻结来源，先校验每项来源 SHA256，再生成页面；
不重跑仿真、不覆盖原始结果。默认使用已有 `imageio_ffmpeg` 从本地视频提取截图，
`--no-posters` 可跳过截图且无需该依赖。`--root` 指向已有完整结果根目录，
`--output` 可指定 HTML 输出位置。构建需要本地汇总及原始 JSON/CSV/报告，
仅下载运行输入不包含全部报告构建数据。

单独分享 HTML 可保留图表、截图和数据导出。视频不内嵌、不上传，播放使用相对于
HTML 的本地路径；移动 HTML 后若未保留目录关系，仍可浏览截图，但需从原目录打开视频。
在线原始报告和表格链接固定到资产发布版本。M05、C01 不计算虚构速度比，
C02 摸测独立展示，M06 的替换样本和各项失败检查保留说明。


## 新 W1 资产的视频与图片

EC 的 S04、M01–M05、C01、C02 脚本支持显式指定预览资产及独立输出：

```bash
python scripts/benchmark/tech_reports/physics/platforms/dexuni/embodichain/m04.py \
  --robot-urdf "$HOME/下载/assets/w1-030/robot.urdf" \
  --out outputs/benchmarks/tech_reports/dexuni/presentation/w1-refresh/m04
```

沿用任务原环境与 SDK 启动方式。此选项只支持非计时录制，拒绝与 `--measure`
或 `--pilot-performance` 组合，并拒绝覆盖已有 manifest。`robot_asset_override.json`
记录实际使用的 URDF、网格哈希，原 manifest 中的冻结输入身份仍保留为来源。
新资产的链接/碰撞模型可能变化，因此本轮只更新视频、图片，测量数据使用历史批次。

将这一批素材接入 HTML：

```bash
.venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_html_report.py \
  --preview-root outputs/benchmarks/tech_reports/dexuni/presentation/w1-refresh
```

预览目录用小写任务编号（如 `m04/`、`c02/`）；缺少素材时仍可构建历史结果。
页面以摘要、实验条件、正式结果表、效果限制、C02 配置取舍为主，素材更新单列。
新资产图像不会覆盖历史均值、标准差、效果指标或原始视频。


### 正文动态视频

第 3 节以左右并排播放器展示全部 13 项任务。EC 的 8 个 W1 场景只使用
`--preview-root` 指定的新资产录像；5 个 Franka 任务从当前保留批次读取各平台独立录像。
旧 W1 组合视频不再进入报告播放器。原视频文件仍保留，测量数据不改写。

摘要、结果表及播放器突出 M05 / C01 在本次 Isaac Lab / Newton 后端的完整场景缺口；
不使用 C01 单面包视频替代完整对照。C02 单独标明保形限制，可切换 96×24 与 24×12 失败配置。
支持一起播放、暂停、从头重播、倍速与单侧进度/全屏控制。按各录像自身时间推进，
不宣称动作阶段或视频时长严格对齐；结束一侧停在末帧。进入视野时静音预览，离开时暂停，
遵循减少动态效果偏好，也可关闭自动预览。打印保留两侧对应截图。
MP4 使用本地相对路径，没有上传视频；分享时需一并保留对应目录。


### 对外展示与便携发布包

首屏以实际 W1 仿真画面和可追溯的 M04/M03 数据呈现性能，突出含气腔场景能力。
关键数字从内嵌测量数据计算；M03 的效果差异、M06 的替换样本及后端支持范围仍明确列出。
页面只保留资产相对文件名与 SHA256，不包含运行机器的用户目录。

```bash
.venv/bin/python scripts/benchmark/tech_reports/reporting/dexuni/build_html_report.py \
  --preview-root outputs/benchmarks/tech_reports/dexuni/presentation/w1-refresh-20261112 \
  --output outputs/benchmarks/tech_reports/dexuni/publication/embodichain-physics-report/index.html \
  --bundle-media
```

`--bundle-media` 将当前选择的 25 段平台录像复制到 HTML 旁的 `media/`，并写入
`media/manifest.json` 校验清单。整个目录可以搬移或托管；此命令不上传任何内容。
仅复制 HTML 不包含 MP4。原始测量、输入资产及录像均不改写。


### 性能出版图与逐任务点评

性能图采用双平台点线图，默认按耗时降幅排序；可切换任务顺序和 Isaac=100 的相对视图。
绝对视图显示进程间标准差；相对视图只比较均值，不绘制归一化误差线。
构建需要 `matplotlib` 和中文字体（推荐 Noto Sans CJK）。生成的 4 组 SVG / 200 DPI PNG
位于 HTML 旁的 `figures/`，页面提供下载入口，便携发布包包含这些文件。
SVG 同时内嵌进 HTML，独立分享 HTML 仍能显示图表。

第 3.1 节展开全部 13 项任务：每项包含两侧录像（缺少等价场景时明确说明）、
原测量统计、已测结果、定性效果点评及 EC 优势解读。点评由 `task_commentary.json`
管理，基于保留录像与任务结果；不把画面观感当作物理精度或新测量。
M03 的放置判据差异、M06 的替换样本、C02 的不同迭代预算均就地说明。
基础任务画面效果接近时，只陈述已有性能结果，不推断更高的仿真精度。
逐任务录像按需播放，离开视野暂停；切换任务时仅保留当前任务的双侧播放。
