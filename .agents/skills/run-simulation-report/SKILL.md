---
name: run-simulation-report
description: Continue the L-02–L-05, R-01, P-01–P-07 and D-01 simulation report experiments, preserve raw evidence, rebuild reports, and prepare their publication. Use for this report suite rather than unrelated benchmarks.
---

# 仿真技术报告实验

先读 [当前快照](../../../scripts/benchmark/tech_reports/reporting/releases/simulation-report-2026-10-28/README.md)、对应系列的 `experiments/*spec.md` 和各次 `run.json`，再按本次指定范围执行。
源码根为 `scripts/benchmark/tech_reports/`；历史数据工作区通过 `SIM_REPORT_ROOT` 指定。
快照与运行证据优先于旧 Spec 中的候选门槛，不因接手自动重跑已有矩阵。

## 共同要求

- 用 Isaac Lab 对照评测 EmbodiChain；通过公共接口实现，记录 DexSim 依赖。不得因报告需要修改求解器源码。
- 保留全部原始数组、配置、版本、命令、退出状态、失败记录和源码快照。新测量使用新 run_id，不覆盖旧证据。
- 区分连续误差、实际任务完成事件、运行异常及有应用依据的容差。不默认用人为 pass/fail、9/10 或单个综合分数决定优劣；较差结果必须保留。
- 明确区分确认 bug、实测差距、待定位现象和评测限制。重要差距集中列在结论及状态页，不以其他优势淡化。
- 仿真分平台独立进程；计时确认实际运行时、同步边界、全部复制体状态和独占 GPU。共享资源观测不当作公平性能比较。
- 状态分本轮完成、部分完成、待开展；实验完成不等于问题修复或正式发布就绪。

## 加载与资产

- 所有 L 系列用原生 `num_envs` 复制完整且隔离的 arena；每环境复杂度与环境总数分开记录。N=1 只是参考点，单环境多对象不能替代批量并行加载。
- L-02 当前主配置为 `rigid_enable_sleep=False`，必须显式传 `--sleep off`。历史入口仍保留 sleep on 默认值；不要挪用旧结果。该配置不自动影响其他实验。
- L-03 当前正式范围是 Cartpole、Franka、Go2。G1 旧矩阵因碰撞遗漏排除；新版修复预检不自动补入旧性能统计。纯关节模型的实际 SolverMuJoCo 分支不等于 VBD。
- 两平台使用同一份冻结 USD 和依赖，核对数量、质量／惯量、关节、mimic、碰撞几何、局部变换、根状态及重叠环境隔离；遍历 USD 时包含 instance proxies。
- 后续优先使用来源明确、体积适中的原生 USD。避免默认从本地 URDF 转换通用资产；旧 G1／抽屉转换产物只留作历史证据。优先从 NVIDIA SimReady Central 寻找小型候选，实际作者与许可按资产本身记录。
- L-04 柜体候选及固定修订见快照 README。候选存在不等于可运行；先 N=1／4 预检，再启动矩阵。主测不含相机；视频单独录制。
- Isaac 对照仍用原 Isaac Lab／Isaac Sim 6.1 环境；历史 5.1 离线转换不构成切换测量环境的授权。

## 分析、报告与素材

两端有结果时维护三章初稿：简短对比与结论、EmbodiChain 详细结果、Isaac Lab 详细结果。文字简洁，正文不提人名；列实际参数、完整范围、限制及可复算的原始出处。
另生成 `/benchmark` 约定的三表摘要：Time & Memory、Success & Other Metrics、Leaderboard。未测量写缺失原因；无二元任务目标时成功率不适用，不为填表重新创造门槛。

实验配图／录像来自真实平台相机或窗口；两端统一镜头、物理时刻、时长和播放倍率。保存原始帧、相机参数、状态和编码命令，补录核对主测轨迹。分析图不能替代原生画面。
素材按现象选代表组，不按参数矩阵逐点保存视频。加载实验通常保留一个能看清多 arena 的规模，两平台配对；物理实验保留典型行为、边界和不同异常，不能只留下有利结果。效果雷同的数量／条件变体、无效录制、旧视角和完全相同副本可清理，连同其逐帧图片；保留片段仍保存原片、原始帧、相机参数、状态和编码命令。全部数值结果、配置、日志、失败记录及源码快照保留。清理时记录路径、原因及原 SHA-256，并同步素材索引、网页和上传清单。大型文本场景可无损压缩，解压哈希核验后删除未压缩副本；临时重建／验证副本用完清理。

修改报告后更新 `reporting/site/catalog.json` 并构建检查 HTML；未开展项仅标题，P-03 在正式性能完成前保持部分完成。
重建在数据副本中进行，不改写原始运行或冻结清单。图表保留绘图数据和 SVG／PDF。

## 提交与发布

遵守技术报告根 README 和 `assets/README.md`：实验定义／编排在系列 experiments，平台执行在 platforms，离线分析在 reporting，少量手写输入在 assets，运行产物在 outputs/benchmarks 或数据集。
不要把尚未实现的共享 core 或下载 CLI 写成可用命令。
当前 draft 仍有机器路径及冻结依赖耦合，只有离线验证明确通过的入口可称已迁移验证；新机器执行前先参数化及最小试跑。

数据集为 `DexForceAI/embodichain_reports`。当前用户选择先建 PR、上传稍后处理；不得声称已上传。
`reporting.prepare_release` 仅生成本地清单与输入候选，不提供网络发布功能。逐文件许可必须核实；输入 index 不包含结果／视频，也不能覆盖远端已有其他报告条目。
真实发布将输入、元数据和合并 index 放入同一提交，验证后记录完整 SHA。新版本结果另存，不能把维护分支提交冒充测量版本。
