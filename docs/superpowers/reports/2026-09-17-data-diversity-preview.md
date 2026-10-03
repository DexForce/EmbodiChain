# 数据多样性工作台首版验收

2026-09-17。基于 main `3224ac1`，分支 `codex/data-diversity-p0-p1`，独立 worktree `.worktrees/data-diversity-p0-p1`。本报告验证用户要求的可体验首版，**不宣称完整 P0/P1 标准全部完成**。

## 如何体验

本机工作台：http://127.0.0.1:7865 。GenSim 的 Gradio 顶部增加 `Data analysis ↗` 入口，指向单独启动的此工作台。

```bash
cd /home/dex/workspace/sources/EmbodiChain/.worktrees/data-diversity-p0-p1
/tmp/embodichain-analysis-env/bin/python -m embodichain analyze-data serve \
  --catalog /home/dex/.cache/embodichain/data_analysis/repeated-pick-place-preview/catalog.sqlite \
  --port 7865
```

当前环境通过临时 venv 复用 `open` 环境的既有依赖，仅覆盖安装仓库约束内的 Gradio 6.17.3；没有修改共享环境。通常可安装本项目的 `analysis` extra，再用相同 CLI。分析入口不加载 GenSim 工作流、Blender 或 DexSim。

先查看分布，再选择 `glossy_orange` 并应用筛选，可看到 6 条记录。进入“轨迹与场景回放”，选择 episode 并点击“打开 / 更新回放”。Viser 右侧的 Frame / Play recording 控制轨迹、场景和 RGB 时间轴，可选择第二条 episode 叠加 TCP 轨迹。导出按钮固定导出当前已应用的屏幕数据集，不使用未应用的筛选草稿。

重新采集：

```bash
python -m embodichain analyze-data collect-preview --output /path/to/new-run --count 12 --seed 17
python -m embodichain analyze-data summary --catalog /path/to/new-run/catalog.sqlite
python -m embodichain analyze-data import --catalog /path/to/catalog.sqlite --source /path/to/lerobot --format lerobot
python -m embodichain analyze-data import --catalog /path/to/catalog.sqlite --source /path/to/episodes.jsonl --format jsonl
```

## 真实任务证据

| 项目 | 实测结果 |
|---|---|
| 既有部署 | `repeated_pick_place/task.franka.yaml`，FrankaPanda，文件指定 default physics |
| 运行 ID | `pick-place-b2d6275f60` |
| 采集 | 12 次有界尝试，12 条完整记录，36 个任务片段，2,892 个含初始状态的样本 |
| 物体 | 45 mm / 50 mm 两种实际立方体尺寸；不是两个不同语义类别 |
| 初始姿态 | 12 个实际测量的位置；旋转固定，未声称旋转泛化 |
| 材质 | matte_blue / glossy_orange，配置已应用到真实渲染对象 |
| 光照 | 3 / 7 两个配置强度；保留 configured 来源，不冒充测量值 |
| Affordance | 一类任务标注 antipodal_grasp |
| 时长 | 每条 9.6 s，记录间隔 0.04 s |
| TCP 路径长度 | 3.154–3.388 m；几何族未聚类，明确 unknown |
| 最大抬升高度 | 0.158–0.283 m，相对初始物体高度 |
| 最终 XY 落点误差 | 0.0108–0.0235 m |
| 实测检查 | 12/12 满足抬升 ≥0.05 m、最终 XY 误差 ≤0.06 m、最终高度距初始 ≤0.03 m |
| 程序语义 | projected 执行策略；单独保存 program_reported_success，不作为物理证明 |
| 产物体积 | 约 443 MiB，包含 scene JSON、轨迹 NPZ、RGB NPZ、账本、日志 |

判定器只验证抬升和最终落点，未验证每次中间放置、持续接触稳定性或抓持力。第 000 条在成功提交后遇到 Gym wrapper 关闭签名错误；产物与物理证据完整。随后改为 `env.unwrapped.close(exit_process=False)`，后续 11 条日志无该异常。保留原始日志，不隐藏开发过程中的异常。

数据和日志：`/home/dex/.cache/embodichain/data_analysis/repeated-pick-place-preview/`。`episodes.jsonl` 可以重建目录：首次导入 12 inserted；再次导入 12 unchanged。早期调试失败另保留在 `/tmp/embodichain-diversity-preview/`，未混入正式 12 条验收批次。

## 功能验收边界

| 标准 | 本版状态与证据 / 剩余项 |
|---|---|
| P0-01 | 部分：版本化来源/单位/参考系/未知值、非法 JSON 校验已实现；逐维语义类型、适用条件注册表未完成 |
| P0-02 | 部分：desired、configured、初始 measured、committed 已区分；通用运行时随机化生效事件未接入 |
| P0-03 | 部分：run/candidate/attempt/episode 和 task segment 可查；多源变换血缘只有字段契约，未接真实拼接扩增 |
| P0-04 | 通过首版范围：完整状态账本、原因、12 候选固定夹具及失败注入；采集器异常暂统一 rollout_failed，未细分全部规划异常 |
| P0-05 | 部分：原子目录写入、文件检查、partial_commit 和恢复转换已测试；自动媒体/sidecar 对账尚未实现 |
| P0-06 | 部分：SQLite 事务、幂等、冲突拒绝、JSONL 重建已验证；所有提取器中断恢复尚未实现 |
| P0-07 | 部分：历史 LeRobot 新旧元数据夹具、明确 unknown、只读导入已验证；历史媒体转换/真实真机数据未验收 |
| P0-08 | 部分：配置路径、seed、实际位置、时间、场景几何可追溯；依赖内容哈希、完整动作契约与版本冻结不足 |
| P0-09 | 部分：采用独立采集适配器，未修改原生成器；相关 demo/录制/可视化回归通过，未覆盖全部 GPU 路径 |
| P1-01 | 部分：状态总览、观察分布、未知计数、列表已实现；独立源族数和完整六维汇总未完成 |
| P1-02 | 部分：资产/材质/光照/状态筛选，后端连续范围查询；界面多选、六维范围筛选和切片恢复未完成 |
| P1-03 | 部分：位置散点、材质×光照热图、共享筛选；其余二维图及单元格下钻未完成 |
| P1-04 | 部分：无目标不报覆盖率；4 格/3 格=75% 单测通过；UI 目标网格配置未完成 |
| P1-05 | 部分：实际 TCP 路径和时长/长度已记录；阶段对齐、几何族去重、计划偏差和速度图未完成 |
| P1-06 | 部分：资产和 Affordance 标注可追溯；交互区域几何未接入 |
| P1-07 | 部分：真实 3D、RGB、TCP、Frame/Play、双轨迹已验证；关节曲线、动作面板、阶段联动未完成 |
| P1-08 | 部分：不可变快照、ID/数量和材质分布差异已验证；覆盖/近重复和维度定义迁移未完成 |
| P1-09 | 部分：屏幕固定切片的 ID/完整记录引用已导出；目标缺口驱动的可执行扩增配方未实现 |
| P1-10 | 部分：幂等元数据追加/导入；特征版本缓存和重算机制未实现 |

浏览器实际验证：总览 12 条；glossy_orange 筛选 6 条；下载文件中的六个 ID 与屏幕一致；保存快照 #2；与采集快照 #1 比较为 12→12、无增减；Viser 场景与 RGB 显示、两条 TCP 轨迹叠加。回放时间精度另由不均匀时间戳和控制时钟测试验证。

## 工程验证

- 277 个相关测试通过：分析核心/录制/回放/筛选导出、Viser 协议与后端、CLI、现有 demo 与轨迹记录、扩增覆盖、API 文档检查器、项目上下文。
- 黑盒 CLI JSONL 导入/再次导入分别返回 12 inserted / 12 unchanged。
- 子进程阻断 DexSim 导入时，离线回放导入通过；不以带仿真初始化的 mock 代替离线边界。
- Black 26.3.1 格式化；`git diff --check`；API 覆盖 2117/2117；context check 和自然语言路由通过。
- Sphinx dummy 构建退出成功，但有 714 条现有模块警告/文档格式问题；日志未发现新增 data_analysis 页警告，不宣称全仓文档无警告。
- 独立复查指出的写盘中断状态、非均匀回放时间、导出与显示不一致三项已修复，并经复查确认。

## 元数据性能

环境：Linux 6.8 x86_64，Python 3.11.14，32 逻辑 CPU，本地盘。固定 10,000 episode / 100,000 segment 合成元数据，每条具备六类维度键，未测量项显式 unknown；不含真实 native 事件大载荷、视频或逐帧特征。

- SQLite 约 82.5 MB；首次逐条事务建库 51.95 s。
- 首次查询 0.244 s；预热后 30 次同一筛选/分布查询，p95 **0.379 s**。
- 建库和查询进程峰值 RSS **464.5 MiB**；单独重复查询进程峰值约 260.0 MiB。
- 未清空 OS 文件缓存；这些是服务层数据，不代表页面延迟，也未覆盖所有查询组合和增量特征提取。

原始数值见 `2026-09-17-data-diversity-performance.json`。完整性能标准中的分页、按需媒体解码、增量特征抽取仍未实现；当前 NPZ 回放一次加载一条 episode（比较时两条），仅适合本版小规模数据。
