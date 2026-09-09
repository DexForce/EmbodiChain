# Agent Lab

独立于 Task Engine 的开放式代码求解实验。原子技能和 Task Program 都是可选工具，
不是动作白名单。默认采用 B 模式：保留 GenSim 接触动力学，开放控制策略和机器人调参。
原始资产不修改；资产修复属于需要另行确认的 C 模式。

[API 参考](../../../docs/source/api_reference/gen_sim_agent_lab.rst) 记录本实验模块的接口。

## 开始运行

在仓库根目录执行：

```bash
source "/home/dex/miniconda3/etc/profile.d/conda.sh"
conda activate embodichain040
python -B -m embodichain.gen_sim.agent_lab inventory
python -B -m embodichain.gen_sim.agent_lab pipeline --task task1187 --minutes 45
```

`pipeline` 一次完成创建新工作区、启动全新 Codex 和最终交付，不需要手填 `--run-dir`。
默认完整任务目标 `solve`、非交互运行、`gpt-6-astra / xhigh` 和 45 分钟预算。
替换 `--task` 选择其他任务；加 `--window` 改为独立交互窗口，加 `--objective cold_start`
只验证接入与控制。`--batch` 可显式指定默认自动模式，不能与 pipeline 的 `--window` 同时使用。
还支持 `--task-file`、`--assets`、`--output-root`、`--robot-component`、`--model` 和
`--reasoning-effort`。它复用新 `launch` 流程，不调用旧的 `solve/resume` 搜索循环。

### 模型与思考强度

```bash
python -B -m embodichain.gen_sim.agent_lab pipeline \
  --task task1187 --minutes 45 --model gpt-6-astra --reasoning-effort xhigh
```

模型 ID 与思考强度作为 CLI 配置传入，不由任务提示词选择。`pipeline`、`launch` 和新建
`solve` 未指定时使用 `gpt-6-astra / xhigh`，不会修改个人全局配置。
旧入口 `resume` 未指定的项分别继承最近一次启动记录；只传 `--model` 会保留原思考强度，
只传 `--reasoning-effort` 会保留原模型。没有历史配置时才使用项目默认值。
配置继承不改变会话语义：`resume` 仍是旧 solve 的续跑入口，不是 pipeline/launch 的会话恢复接口。

框架拒绝空参数，但不硬编码所有可用模型及模型/强度组合。不支持的配置由 Codex 后端报错，
错误退出保留日志并返回非零 CLI 状态，不静默换模型或降低强度。
做模型对比应为每组配置创建独立 run，固定任务、资产、种子和预算。

`usage.json.configurations` 和最终报告按每次启动列出请求配置、Codex 会话元数据中
观察到的配置、会话耗时、token 与完整性。观察记录缺失时显示未取得，不用请求值冒充。
同一启动中观察到多种配置时保留配置集合，不推算每个模型的花费；会话元数据也不证明
服务端内部实际路由。中途换配置的全程用量不能统一归到最后一个模型。

需要分步准备或检查工作区时，仍可使用 `prepare` 后接 `launch --run-dir`。
`launch` 默认是交互式 Codex；`--window` 在 Linux GNOME Terminal 中打开新窗口，
省略则使用当前终端。首次目录信任由用户在 Codex 窗口确认，不修改全局信任设置。
加 `--batch` 使用同一套工作区入口执行自动化测试，不需要图形终端或目录信任交互。
启动器独立负责 GPU 请求服务及时间预算，不依赖另一个聊天任务提供服务。
沙箱通过文件心跳确认宿主，停止请求也通过文件传递，不依赖跨 PID 命名空间查询。
交互式窗口可由用户直接接手；时间预算到期仍会停止本轮进程并保存已有产物。

`prepare --objective cold_start` 只要求自主检查环境、场景并验证一次真实控制，
不要求完成完整题目。默认 `--objective solve` 才要求解完整任务。
新工作区不复制旧代码、旧笔记或旧会话。`launch` 总是新建对话，显式关闭个人
记忆注入及生成，发送固定入口消息并要求读取宿主用量快照，不注入具体解法。项目及系统通用指令仍会
生效；这不是隔离所有文件访问的盲测环境。首次冷启动应始终重新 `prepare`，
不要把在已有笔记目录中新开对话称为无历史输入实验。

### 自包含入口

- `workspace/START.md`、`AGENTS.md`：不含具体任务解法的固定导航说明。
- `workspace/task.json`：题目、资产和验收。`task100` 只是一个输入适配器。
- `workspace/environment.json`：当前解释器、项目导航与本轮目标。
- `workspace/physics.json`：物理约定与明确批准的修复范围。
- `workspace/lab.json`：可替换的场景/机器人装配预设；`--robot-component` 可另选组件。
- `workspace/lab.py`：独立运行工具，自动选择解释器与本轮宿主，不需手填队列环境变量。
- `launch.json`：宿主身份、预算、启动模式与实际命令所在目录。
- `heartbeat.json`、`stop.json`：本次启动的存活状态与停止请求。
- `bootstrap_hashes.json`：创建时入口文件指纹，便于核对后续是否被改动。

其他资产可以直接用 `prepare --task-file "/absolute/task.json"`，不要求 task100 编号。
最小输入字段为 `task_id`、`instruction`、`acceptance`、`source_dir`；可以另提供
`scene_config`、`image`、`level`。路径使用绝对路径。框架不把资产存在当作场景合格，
也不自动重建或修复不适用的场景。

在工作区中：

```bash
python3 lab.py --help
python3 lab.py run --script probe.py --config lab.json --timeout 300
python3 lab.py run --script standalone.py --timeout 300
python3 lab.py inspect --attempt "/absolute/attempt"
python3 lab.py status
python3 lab.py usage
python3 lab.py stop
```

带 `--config` 时调用 `solve(lab)`，不带则运行独立 Python 程序，产物目录由
`GENSIM_LAB_OUTPUT` 提供。独立程序也必须保留视频/状态，但自动接入 Lab 录制并非
强制；框架不会把缺视频的正常退出算作控制验证成功。
先输出 `scene_review.json` 再决定是否控制；结束由 Codex 写 `handoff.json` 交接声明。
宿主不供应目标位姿或修复建议，也不把这份声明当作验收结论。

### 原有自动求解入口

`solve`、`resume` 保留供旧实验重跑，它们带自动续轮与候选重跑，**不是冷启动验证入口**。
使用已经登录的本地 Codex CLI，不复制认证文件、不切换模型供应商。启动和续跑都显式
使用解析后的模型与强度，不依赖个人默认配置；新实验默认 `gpt-6-astra / xhigh`。
`resume` 继承上次配置，显式传入 `--model` 或 `--reasoning-effort` 则单项覆盖，
并在每个 `codex/turn_*/agent_settings.json` 中记录。
每个任务创建独立 Git 工作目录供 Codex 写代码，默认
`workspace-write` 沙箱；GPU 仿真请求由沙箱外的宿主监督器执行，并通过文件队列回传。
原仓库供检索与导入，不应直接修改。候选是受信任的任意 Python，宿主执行器不是恶意代码沙箱。

任务和中间产物位于 `outputs/task100/<task_id>/<run_id>/`：

- `run.json`、`source.diff`：任务、代码版本、解释器和源工作区增量。
- `workspace/`：候选 Python 程序、配置、实验记录。
- `codex/`、`session.json`：每次 Codex 调用的 JSONL 事件、错误输出、会话 ID。
- `requests/`：Codex 到宿主 GPU 运行器的请求和回复。
- `attempts/`：每次仿真独立保存代码、配置、MP4、图片、实际状态和错误。
- `attempts/*/experiment.json`：冻结本次实验的 B/C 模式及明确批准的资产改动范围。
- `summary.json`：候选重跑结果。`task_success: null` 表示仍需验收，不是假定成功。

## 最终交付

每轮结束后统一读取 `final/result.json`、`final/report.md` 和可用时的 `final/video.mp4`。
`launch`（交互/批处理）、`solve/resume`、宿主单独 `run` 都会自动收尾；
Codex 通过队列发出的中间尝试不会提前发布整轮交付。
正常结束、失败、超时和停止都生成同一格式报告。断电/强杀无法保证执行 finally，
恢复后可在宿主运行以下命令，只整理既有证据，不启动 Codex 或仿真：

```bash
python -B -m embodichain.gen_sim.agent_lab finalize --run-dir "/absolute/run"
python -B -m embodichain.gen_sim.agent_lab finalize \
  --run-dir "/absolute/run" --attempt "attempts/selected_attempt_id"
```

`result.json` 的 `schema_version` 固定为 `agent-lab-delivery/v1`：

| 字段 | 含义 |
| --- | --- |
| `run` | 任务、运行身份、工作区版本、本轮声明的 B/C 模式与授权 |
| `execution` | 宿主运行状态、模型和思考强度；与物理任务是否成功分开 |
| `resources` | 宿主计量的累计耗时和 token 用量、来源与统计完整性 |
| `assessment` | 验证范围、Codex 声明、既有评审记录；整理器不冒充任务验收器 |
| `selection` | 选定 attempt、选择来源和理由 |
| `video` | 可播放录像的来源、性质、帧数、帧率、时长、格式及 SHA-256；缺失时为 null |
| `video_missing_reason` | 没有发布录像的原因，不生成空白占位视频 |
| `attempts` | 所有尝试的执行状态、错误、指标、原始证据路径及已记录的物理模式 |
| `reproduction` | 冻结代码/配置的复现命令、文件指纹、解释器和源资产指纹 |
| `issues` | 损坏文件、无效选择、缺失证据等整理问题 |
| `publication` | 不可变交付版本及输入指纹 |

`video.path` 相对 final 目录；`video.source`、`attempts[].path` 和 `publication.directory`
相对运行目录。宿主状态为 completed / failed / timed_out / stopped / interrupted / unknown，
其中 completed 只表示宿主正常结束，不表示完整任务完成。

视频选择顺序：命令行显式指定、`handoff.json.selected_attempt`、此前命令行选择、
旧入口的 replay 记录，最后才回退到最近可解码的诊断录像。最后一种选择不表示最好或成功。
`selected_attempt` 使用本轮相对路径 `attempts/<ID>`，也接受本轮绝对路径；
外部尝试和跨目录视频软链接不能作为本轮交付。显式选择无效时直接报告问题，不偷偷换录像。

最终录像来自单次尝试，不拼接不同尝试。采用 MP4/H.264、YUV420P；兼容录像保持原字节，
其他可解码格式转换后再次检查帧数，奇数尺寸仅补齐到偶数。完整解码不等于动作或任务成功。
新尝试在执行前保存 `inputs.json`，整理时重新检查冻结代码与配置；不匹配时标注问题，
不提供误导性的复现命令。旧尝试缺少此记录时标为 `not_recorded`，不伪造运行前校验。
`complete_attempt` 仅表示 worker 正常完成且帧数未发现缺失，不表示完整题目通过；
`partial_attempt` 明确用于异常/中断/记录不完整的尝试。

`report.md` 从同一份结果 JSON 生成。`assessment.task_success` 始终为 null，
`verification_status` 为 `not_run`；既有 `acceptance.json` 原样作为评审记录保留，
不会因为整理成功或 Codex 自报成功而提升为独立通过。

`final` 是指向 `.deliveries/<版本>/` 的固定目录软链接。报告和录像全部完成后才原子切换；
上个版本与原始尝试都保留。相同输入重复整理复用原版本；无视频的新交付不会带上旧视频。
多个文件的读取需要锁定同一版本时，先解析 `final` 的实际目录或使用 `publication.directory`。
不要手工修改 `final` 内容，也不要把 `final` 替换成真实目录。运行中的宿主或 worker 会阻止发布。

### 累计耗时与 Token

提示词明确要求、宿主负责计量、报告统一展示。`START.md` 和旧求解入口均要求 Codex
读取 `usage.json`，引用统计范围、`as_of` 与完整性，不按回答长度或账户额度猜数。
运行中宿主约每 5 秒原子更新快照；`python3 lab.py usage` 只读这些数字，不暴露对话内容。
`handoff.json` 可以提交 `usage_file` 和中文 `usage_notes`，但不能覆盖最终宿主计数。

`usage.json` 使用 `agent-lab-usage/v1`，最终原样进入 `result.json.resources`，
报告开头固定展示“累计耗时与用量”。统计仅覆盖当前 run，不混入本框架开发对话或其他任务。

| 字段 | 统计口径 |
| --- | --- |
| `timing.total_wall_seconds` | 所有宿主、Codex 和 worker 运行区间的并集，不重复计算嵌套等待，不计启动间停机间隔 |
| `timing.codex_wall_seconds` | Codex 进程的累计墙钟时间，包含工具等待，不是纯模型思考时间 |
| `timing.simulation_wall_seconds` | 仿真 worker 墙钟耗时累计，不是物理仿真时间或视频时长；不与总耗时相加 |
| `tokens.total_tokens` | 输入加输出；缓存输入和推理输出是子项，不重复累加 |
| `tokens.turns` | 按关联线程/回合去重的用量、来源、最新快照时间与缺失说明 |
| `as_of` | 数据截至的 Unix 时间，而非报告每次打开的时间 |

新运行使用 `usage_sessions/*.json` 保存每次启动的宿主计时，包含调试、重试、最终重跑
及视频验证/复制开销，截止到报告序列化前的计量快照。最终写文件和原子发布的极小尾部不计入。
人工等待如果发生在进程存活期间仍计入；不会猜测并扣除“思考”或“人工等待”时间。
旧运行通过进程记录重建，标为 `reconstructed`，可能遗漏未记录的初始化/收尾间隙。
缺少结束记录的时长为未知或已观测下界，不使用文件修改时间冒充运行时间。

Token 优先使用 `turn.completed.usage`；中断或交互模式可从关联会话的
`token_count` 数字元数据恢复。宿主只按线程 ID、工作区和调用时间窗口选取对应记录，
不向 Codex 注入历史对话。累计快照只取最后有效计数；续跑时区分计数重置与延续，
延续时计算差值。计数语义异常则降低完整性，不将多个快照重复求和。

- `reported`：已有完整用量事件。
- `partial`：仅部分调用或中断前快照可用，显示“至少”；未知部分不补零。
- `unavailable`：没有可靠用量数据，显示未知。

没有有依据的估算来源就不生成估计值。计数不是价格、账单或账户额度；未关联到本轮的
外部模型调用不计入。重复 `finalize` 不启动新会话，也不把恢复/查看历史报告耗时加回旧实验。

## 单独调试

```bash
python -B -m embodichain.gen_sim.agent_lab prepare --task task1103
python -B -m embodichain.gen_sim.agent_lab resume \
  --run-dir "/absolute/run/directory" --minutes 45
python -B -m embodichain.gen_sim.agent_lab run \
  --run-dir "/absolute/run/directory" \
  --script "/absolute/solution.py" \
  --config "/absolute/lab.json" --timeout 300
python -B -m embodichain.gen_sim.agent_lab inspect --attempt "/absolute/attempt/directory"
```

便利入口脚本实现普通 `solve(lab)` 函数。`lab.sim`、`lab.robot`、`lab.objects`、
`lab.articulations` 是真实运行时对象，不经过新的声明式执行器。
`lab.scene.planner_objects` 提供源对象元数据与路径，`lab.scene.table_top_z` 是桌面高度。
直接使用项目 API、NumPy、Torch、SciPy、抓取生成器或任意规划算法。

可选便利函数：

- `lab.step(seconds)`：推进仿真，默认一个 0.04 秒控制周期。
- `lab.move_joints(qpos, part="left_arm", seconds=2)`：平滑发送关节目标。
- `lab.move_tcp(pose, part="left_arm", seconds=2)`：单目标 IK 后发送关节目标。
  这只是最小便利方法，不宣称保证笛卡尔直线、无碰撞或无 IK 分支跳变。
- `lab.capture("grasp")`、`lab.event("lifted", height=...)`：关键帧和阶段证据。
- `lab.snapshot()`：实际机器人状态、物体位姿、关节物体 qpos。

所有 `sim.update()` 调用被本次运行的记录器观察，即使绕开便利函数也持续录制。
视频按仿真时间流式写入分片 MP4，不把整个 episode 缓存在内存后才编码。
正常异常和终止信号会执行清理；原生崩溃/强杀时可能只有可恢复的视频前缀。
`latest.png` 持续更新，`initial.png`、`final.png` 和事件图片用于直接观察。
`loaded.png` 与视频首帧记录物理初始化前的加载画面，不作为执行成功证据。
`metrics.json` 中的 IK 计数只覆盖便利函数，不能解释为所有自定义求解器的计数。
`inspect` 完整解码视频并汇总实际关节/物体运动；它不把这些通用指标当作任务验收。
每次执行冻结脚本及同目录辅助模块，保存运行器源代码快照与源资产指纹。
时间预算包含搜索和最终重跑，预留至多五分钟重跑时间。`resume` 延续明确的会话 ID，
不会使用可能属于另一个任务的 `--last`。

## 物理与验收

允许重新抓取、改变路径、调整速度/阻尼/刚度、在执行前调整机器人基座。
遇到宿主 GPU 原生兼容性故障，可显式比较 CPU PhysX 后端，仍保留真实接触动力学，
并在实验记录中区分后端；不能用关闭物理或在沙箱内强行运行 CUDA 代替正确宿主执行。
不允许瞬移物体、逐帧覆盖机器人实际状态、附加吸附/焊接、关闭接触或任意修改质量摩擦。
源场景归一化、物理默认值和双 Franka 装配复用 GenSim 的现有代码。
`lab.json` 的 `robot_overrides`、`sim` 是普通配置覆盖，不是新的参数白名单。
实际合成配置保存于每次运行的 `physical_setup.json`，便于审计。

仅由 Python 正常返回不能证明任务成功。验收需要观察完整连续视频，检查实际 qpos、
对象运动、任务阶段及稳定终态；单独区分资产缺陷、方法失败和数值问题。
求解器返回的数据保存在 `solver_result`，明确不是独立验收结论。
候选通过 `candidate.json` 交付，包含 `script`、`config`、`rationale`，监督器在新进程
重跑。第一阶段是开放资料的单场景攻关，不宣称盲测、跨种子泛化或完整物理资格认证。

首次混合资产测试还验证了加载顺序的重要性：运行器沿用 Gym 的机器人、背景、
关节物体、动态刚体顺序。源资产能够导出不代表本机 GPU 后端一定能稳定加载；
原生加载失败、运动规划失败与任务验收失败在报告中应分别解释。

获准进行 C 模式实验时，先制作运行副本并保存改动审计，将 `run.json` 的 `mode`
设为 `C`，在 `approved_asset_changes` 记录明确授权范围，再让 `lab.json` 指向副本。
续跑提示会携带这份授权；其他资产修复仍需确认。不得把修复后结果混入原始资产 B 模式结果。

## L3/L4 挑战

资产清单保留任务文档中的 `level`。优先验证复合阶段、交接与视觉关系，而不是重复
简单抓放。L3 按完整任务顺序记录中间结果，不能用单手悬停代替交接；L4 先从原图和
实际场景推导目标并保存 `grounding.json`，不能把任务文档的答案提示当作推理成果。
任务文档可能更新，新实验总是重新读取，不复用旧实验里的任务描述。

```bash
python -B -m embodichain.gen_sim.agent_lab solve --task task1178 --minutes 45
python -B -m embodichain.gen_sim.agent_lab solve --task task1194 --minutes 45
```

这两个示例分别是双臂载物托盘搬运后交接杯子，以及根据中央黑线补全对称图案。
仍需检查所选资产是否足够表达任务；部分阶段成功不等于整题成功。
