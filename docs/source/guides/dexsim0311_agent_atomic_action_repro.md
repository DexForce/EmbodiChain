# DexSim 0.3.11 Agent 原子动作迁移复现指南

这份指南用于在本地 DexSim 0.3.11 运行时复现当前 Agent 原子动作迁移验证结果。
当前分支已经为 DexSim 0.4.0 兼容迁移做了准备，但本文记录的验证基线是在
DexSim 0.3.11 上得到的。请先用这份指南确认公共原子动作接口、Agent adapter
路径、graph recovery runtime、视频录制和输出 artifacts 是否正常，再判断哪些部分需要继续针对
DexSim 0.4.0 修改。

## 复现结论

这套复现流程在本地已经实际跑通过，前提条件如下：

- 本地 conda 环境：`embodichain`
- 实际 DexSim 版本：`0.3.11`
- 工作目录：`/home/dex/桌面/EmbodiChain/dexsim040/agentchoard`
- 运行方式：全部使用 `conda run --no-capture-output -n embodichain ...`
- `pour_water_recovery_compare.py` 使用离线 cached artifacts，不实时调用 LLM
- 默认错误注入是 bottle 水平平移，不主动把 bottle 弄倒

本地已验证结果：

- `tests/sim/agent`、`tests/sim/atomic_actions`、
  `tests/sim/utility/test_solver_utils.py`：`103 passed`
- `grasp_cup.py`：生成正常视角视频，Agent skill pick-place 路径可跑
- `pour_water_recovery_compare.py --case all --continue_on_case_failure`：
  10 个 case 全部完成，`summary.tsv` 中 `expectation_matched=True`

本地验证输出示例：

```text
outputs/grasp_cup_dexsim040_migration_candidates_20260515_1502
outputs/pour_water_recovery_compare_20260515_1532_translation_error_full_matrix
```

这些输出目录只是验证记录示例，不要求复现者必须使用相同目录名。

## 覆盖范围

这次复现覆盖以下改动：

- 公共原子动作函数式 API：
  `move`, `pick_up`, `place`, `gripper_open` 和 `gripper_close`
- Agent 原子技能 adapter 路径：
  public move、place、gripper、grasp 的接入和 fallback 逻辑
- `embodichain.lab.sim.atom_actions` 旧 import path 兼容 shim
- 支持 monitor 触发 recovery branch 的 graph recovery runtime
- task-state validation，用于避免只看脚本退出码而忽略视频/物理状态
- DexSim renderer 兼容层：
  在 DexSim 0.3.11 下保持 raster 渲染，同时为 DexSim 0.4.0 的 `hybrid`
  配置做准备
- 两个 gym 验证 demo：
  `grasp_cup.py` 和 `pour_water_recovery_compare.py`

当前复现不覆盖：

- 真正 DexSim 0.4.0 环境下的完整视频和物理结果
- 无缓存 artifacts 时的实时 LLM graph 生成质量
- strict semantic grasp 在所有物体和所有场景下的稳定性

## 环境检查

所有命令都在仓库根目录下运行：

```bash
cd "/home/dex/桌面/EmbodiChain/dexsim040/agentchoard"
```

确认当前分支：

```bash
git branch --show-current
git status --short --branch
```

预期分支：

```text
ljd/agentchoard_dexsim040_migration
```

确认 DexSim 版本：

```bash
conda run --no-capture-output -n embodichain python - <<'PY'
import dexsim
print(dexsim.__version__)
PY
```

预期输出：

```text
0.3.11
```

重要说明：

- 迁移分支中的项目元数据当前指向 `dexsim_engine==0.4.0`。如果目标是严格复现当前
  0.3.11 验证结果，不要直接从头重建环境。
- 在 0.3.11 下保持默认 raster 渲染路径。除非明确测试 renderer 改动，否则不要传入
  `--enable_rt`。
- 不要直接运行裸 `python`。裸 `python` 可能不在 `embodichain` 环境里，常见错误是
  `ModuleNotFoundError: No module named 'gymnasium'`。

## 必需资源

### Python / Conda 环境

必须能在 `embodichain` 环境中 import 这些依赖：

```bash
conda run --no-capture-output -n embodichain python - <<'PY'
import dexsim
import gymnasium
import torch
import cv2
print("dexsim", dexsim.__version__)
print("torch", torch.__version__)
print("ok")
PY
```

### DexSim / GPU / 渲染

本地验证使用 GPU + Vulkan + Raster renderer。运行 demo 时日志里应看到类似信息：

```text
Renderer: Raster
Using CUDA device: NVIDIA ...
Vulkan device driver: ...
```

如果视频黑屏、视角异常或窗口无法创建，优先排查 renderer/camera/GPU，而不是先改
Agent 原子动作逻辑。

### Agent artifacts 缓存

`scripts/tutorials/gym/pour_water_recovery_compare.py` 默认按离线方式运行，会从下面目录复制预生成的 graph artifacts：

```text
outputs/agent_repro_compare
```

这样评审或测试人员在做确定性 demo 验证时，不需要实时调用 LLM。

运行 pour-water matrix 前，先检查这些文件是否存在：

```bash
test -f "outputs/agent_repro_compare/single_normal/artifacts/agent_task_graph.json"
test -f "outputs/agent_repro_compare/single_recovery/artifacts/agent_recovery_spec.json"
test -f "outputs/agent_repro_compare/dual_normal/artifacts/agent_task_graph.json"
test -f "outputs/agent_repro_compare/dual_recovery/artifacts/agent_recovery_spec.json"
```

如果这些文件不存在，`pour_water_recovery_compare.py` 可能会尝试进入 graph 生成流程，
但默认离线 LLM stub 会阻止实时 LLM 调用，最终报错。此时应先从验证机器复制
`outputs/agent_repro_compare`，或者配置真实 LLM 后显式重新生成 artifacts。

## 静态检查和单元测试

运行：

```bash
git diff --check

conda run --no-capture-output -n embodichain pytest -q \
  tests/sim/agent \
  tests/sim/atomic_actions \
  tests/sim/utility/test_solver_utils.py
```

已验证的 0.3.11 运行中的预期结果：

```text
103 passed
```

第三方 warnings 可以接受，但以下情况需要处理：

- test collection 失败：通常是环境或 import path 问题
- `solver_utils` 失败：优先检查 CobotMagic chain 的 CPU/GPU device 处理
- atomic actions 测试失败：优先检查 `MoveActionCfg`、`PickUpActionCfg`、
  `PlaceActionCfg`、`GripperActionCfg` 的接口是否被改动
- agent 测试失败：优先检查 graph spec、monitor/recovery binding、task-state validation

## Demo 1：Agent Skill Pick-Place

这个 demo 对应一个更小的 Agent 原子技能闭环验证。它复用
`scripts/tutorials/sim/atomic_actions.py` 的 mug 场景风格，但执行路径不是直接调用
`AtomicActionEngine.execute_static()`，而是通过 Agent 原子技能完成动作序列。

主要验证内容：

- `grasp`：通过 Agent skill 抓取 mug。
- `move_by_relative_offset` / `move_to_absolute_position`：抓取后移动到中间和放置位置。
- `place_on_table`：调用 public place 路径完成释放和撤离。
- `back_to_initial_pose`：动作结束后回到初始位姿。
- public atomic action adapter：验证 Agent skill 能调到 public move/place/gripper/grasp
  路径，并在需要时保留 fallback。

运行：

```bash
TS="$(date +%Y%m%d_%H%M)"
OUT="outputs/grasp_cup_repro_${TS}"

conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/grasp_cup.py" \
  --output_root "${OUT}" \
  2>&1 | tee "${OUT}.runner.log"
```

预期输出：

```text
${OUT}/summary.tsv
${OUT}/outputs/videos/episode_0_cam1.mp4
${OUT}.runner.log
```

预期行为：

- 相机视角应清楚显示桌面、机械臂和 mug。
- Agent skill 序列应完成 pick、lift、move、place 和 return 步骤。
- `summary.tsv` 应标记运行成功。
- 视频长度应足够观察完整动作，不应只有很短一段。
- 如果启用了 semantic grasp，mug 应在 lift 阶段跟随夹爪上升；如果只看到夹爪移动而
  mug 留在桌面上，应判定 grasp 失败。
- place 后 mug 不应被明显拖拽、推飞或从桌面边缘掉落。

建议检查的输出文件：

```text
${OUT}/summary.tsv
${OUT}/outputs/videos/episode_0_cam1.mp4
${OUT}.runner.log
```

`summary.tsv` 用于快速判断脚本结果，视频用于确认真实视觉效果。两者不一致时，以视频和
物理状态为准。例如脚本成功但视频中 mug 没有被抓起，应继续排查 grasp validation。

如果 `grasp_cup.py` 失败，优先按这个顺序判断：

1. 视频是否正常：黑屏/裁切/奇怪视角通常是 renderer/camera 问题。
2. mug 是否可见且在桌面上：如果场景初始化异常，先排查 assets/cache。
3. 是否是 grasp 失败：semantic grasp 对 cache、approach direction、candidate ranking
   比较敏感。
4. 是否是 place 失败：重点看 public place 的 upright pose、释放高度和 gripper open。
5. 是否是回初始位姿失败：重点看 `back_to_initial_pose` 的 public move 规划。

## Demo 2：Pour-Water Recovery Matrix

这个 demo 是主要的 recovery 验证入口。它一次性比较单臂/双臂、正常/错误、启用/禁用
recovery 的组合，目的是证明：

- clean case 在没有错误时可以正常完成。
- no-recovery error case 在 monitor 触发后应停止，作为对照组。
- with-recovery error case 在相同错误注入下应恢复并完成任务。
- blind no-recovery case 用于观察“没有 monitor recovery 时继续执行或停止”的失败状态。
- public move/place/gripper 路径和 Agent recovery runtime 可以在同一任务中协同工作。

运行：

```bash
TS="$(date +%Y%m%d_%H%M)"
OUT="outputs/pour_water_recovery_compare_repro_${TS}"

conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/pour_water_recovery_compare.py" \
  --case all \
  --continue_on_case_failure \
  --output_root "${OUT}" \
  2>&1 | tee "${OUT}.runner.log"
```

预期输出：

```text
${OUT}/logs/summary.tsv
${OUT}/report.md
${OUT}/<case>/artifacts/case_result.json
${OUT}/<case>/outputs/videos/episode_0_cam1.mp4
${OUT}.runner.log
```

### 10 个 case 的含义

| case | 机械臂 | recovery graph | runtime recovery | 错误注入 | blind 注入 | 期望脚本结果 | 作用 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `single_clean_no_recovery` | 单臂 | 否 | 否 | 否 | 否 | 成功 | 验证单臂无恢复基础任务是否能跑通 |
| `single_error_no_recovery` | 单臂 | 是 | 否 | 是 | 否 | 失败 | 对照组：monitor 应触发，但不允许执行 recovery |
| `single_error_blind_no_recovery` | 单臂 | 否 | 否 | 是 | 是 | 失败 | 对照组：直接移动指定对象，不依赖 recovery graph |
| `single_clean_with_recovery` | 单臂 | 是 | 是 | 否 | 否 | 成功 | 验证启用 recovery 不应破坏正常任务 |
| `single_error_with_recovery` | 单臂 | 是 | 是 | 是 | 否 | 成功 | 验证单臂 monitor 触发后能执行 recovery |
| `dual_clean_no_recovery` | 双臂 | 否 | 否 | 否 | 否 | 成功 | 验证双臂无恢复基础任务是否能跑通 |
| `dual_error_no_recovery` | 双臂 | 是 | 否 | 是 | 否 | 失败 | 对照组：双臂 monitor 触发但不允许 recovery |
| `dual_error_blind_no_recovery` | 双臂 | 否 | 否 | 是 | 是 | 失败 | 对照组：双臂 blind 错误注入下不启用 recovery |
| `dual_clean_with_recovery` | 双臂 | 是 | 是 | 否 | 否 | 成功 | 验证双臂启用 recovery 不应破坏正常任务 |
| `dual_error_with_recovery` | 双臂 | 是 | 是 | 是 | 否 | 成功 | 验证双臂 monitor 触发后能执行 recovery |

这里的“期望脚本结果”指 `program_success` 的预期值，不等同于所有 case 都应该完成倒水。
例如 `single_error_no_recovery` 和 `dual_error_no_recovery` 预期就是失败，因为它们是
no-recovery 对照组。只要 `summary.tsv` 中 `expectation_matched=True`，就说明该 case
符合预期。

### 输出目录结构

完整 matrix 会把每个 case 隔离到单独子目录中：

```text
${OUT}/
  logs/
    summary.tsv
    single_clean_no_recovery.log
    ...
    dual_error_with_recovery.log
  report.md
  single_clean_no_recovery/
    artifacts/
      agent_task_graph.json
      agent_compiled_graph.json
      case_result.json
    outputs/videos/episode_0_cam1.mp4
  single_error_with_recovery/
    artifacts/
      agent_task_graph.json
      agent_recovery_spec.json
      agent_compiled_graph.json
      case_result.json
    outputs/videos/episode_0_cam1.mp4
  ...
```

重点文件说明：

- `logs/summary.tsv`：最先看的总表，包含 exit code、程序判定、语义判定和预期是否匹配。
- `report.md`：脚本自动生成的概要报告。
- `<case>/artifacts/case_result.json`：任务状态验证结果，包含 object pose、height drop、
  semantic success 等细节。
- `<case>/artifacts/agent_compiled_graph.json`：最终执行的 graph runtime artifact。
- `<case>/outputs/videos/episode_0_cam1.mp4`：最终人工验收必须看的视频。

已验证的 0.3.11 运行中，`summary.tsv` 的预期行为：

- 所有 case 都应为 `expectation_matched=True`。
- clean case 应为 `program_success=True` 且 `semantic_success=True`。
- 不启用 recovery 的 error case 应在 monitor 触发后停止，因此预期
  `program_success=False`。
- 启用 recovery 的 error case 应为 `program_success=True` 且
  `semantic_success=True`。
- `single_error_no_recovery`、`single_error_blind_no_recovery`、
  `dual_error_no_recovery`、`dual_error_blind_no_recovery` 的失败是预期行为，不应当作为
  demo 崩溃处理。
- 如果某个 case 的 `expectation_matched=False`，再进入该 case 的 log、video 和
  `case_result.json` 进一步分析。

### 默认错误注入语义

默认确定性错误是 bottle 水平平移：

```text
--error_injection_type misplaced_object
--error_injection_offset 0.12,0.0,0.0
```

它不应主动将 bottle 弄倒。只有在需要做 fallen-object 压力测试时，才使用：

```bash
--error_injection_type fallen_object
```

这点很重要：当前 matrix 的默认目标是验证“物体被平移后 recovery 能否找回任务状态”，
不是验证“bottle 倒下后机械臂能否扶正”。如果测试时发现默认 case 里 bottle 直接倒下，
应先检查实际命令、runner log 和脚本版本，而不是直接归因到 recovery graph 失败。

### 推荐的分步复现方式

如果一次性跑 `--case all` 失败，建议按下面顺序缩小问题：

```bash
# 1. 单臂正常路径
conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/pour_water_recovery_compare.py" \
  --case single_clean_no_recovery

# 2. 单臂错误但不允许 recovery，预期失败
conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/pour_water_recovery_compare.py" \
  --case single_error_no_recovery

# 3. 单臂错误并允许 recovery，预期成功
conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/pour_water_recovery_compare.py" \
  --case single_error_with_recovery

# 4. 双臂正常路径
conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/pour_water_recovery_compare.py" \
  --case dual_clean_no_recovery

# 5. 双臂错误并允许 recovery，预期成功
conda run --no-capture-output -n embodichain python \
  "scripts/tutorials/gym/pour_water_recovery_compare.py" \
  --case dual_error_with_recovery
```

如果上述单 case 都符合预期，再运行完整 matrix。这样更容易区分是单臂/双臂差异、错误注入差异，
还是完整 matrix 子进程调度导致的问题。

## 关键帧检查

可以使用下面的辅助脚本抽取代表性视频帧：

```bash
conda run --no-capture-output -n embodichain python - <<'PY'
import cv2
from pathlib import Path

root = Path("outputs/pour_water_recovery_compare_repro_YYYYMMDD_HHMM")
out_dir = root / "keyframes_check"
out_dir.mkdir(parents=True, exist_ok=True)

for case in [
    "single_error_no_recovery",
    "single_error_with_recovery",
    "dual_error_blind_no_recovery",
    "dual_error_with_recovery",
]:
    video = root / case / "outputs/videos/episode_0_cam1.mp4"
    cap = cv2.VideoCapture(str(video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    for label, idx in [
        ("mid", frames // 2),
        ("late", int(frames * 0.8)),
        ("final", max(frames - 1, 0)),
    ]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out_dir / f"{case}_{label}_{idx:04d}.jpg"), frame)
    cap.release()
PY
```

请将 `pour_water_recovery_compare_repro_YYYYMMDD_HHMM` 替换为实际输出目录名。

视觉验收标准：

- bottle 和 cup 应保持在相机画面中可见。
- 默认错误注入应表现为 bottle 平移，而不是将 bottle 弄倒。
- recovery case 应能恢复并完成 pour-water 序列。
- no-recovery error case 应记录足够长的失败状态视频，便于检查。
- 如果脚本返回成功但视频里物体明显飞出、倒下或被夹偏拖动，应按失败处理。

## 常见失败和排查建议

### 1. 缺少 `gymnasium`

现象：

```text
ModuleNotFoundError: No module named 'gymnasium'
```

可能原因：

- 使用了裸 `python`，没有进入 `embodichain` conda 环境。

处理：

```bash
conda run --no-capture-output -n embodichain python ...
```

### 2. 缺少 cached artifacts

现象：

- `pour_water_recovery_compare.py` 报 offline LLM stub 错误。
- 日志中出现类似“expected cached agent artifacts and should not call the LLM”。
- case 的 `artifacts/agent_task_graph.json` 或 `agent_recovery_spec.json` 缺失。

可能原因：

- 从 GitHub 新 clone 后没有 `outputs/agent_repro_compare`。
- `outputs/` 通常不适合作为源码提交内容，因此 artifacts 需要单独提供。

处理：

- 从验证机器复制 `outputs/agent_repro_compare`。
- 或者配置真实 LLM，使用 `--regenerate` 重新生成 artifacts。
- 如果目的是验证 runtime/action 迁移，优先使用缓存 artifacts，避免把 LLM 输出不稳定性混入复现。

### 3. 视频黑屏、视角异常或画面被裁切

可能原因：

- DexSim renderer/camera 兼容问题。
- 在 DexSim 0.3.11 下误用了 0.4.0 的 `hybrid` 或 RT 路径。
- GPU/Vulkan 初始化异常。
- `--enable_rt` 改变了渲染路径。

处理：

- 不传 `--enable_rt`，保持默认 raster。
- 检查日志中 renderer 是否为 `Raster`。
- 优先确认 `grasp_cup.py` 的单 demo 视频正常，再跑完整 matrix。

### 4. 单测通过但 demo 失败

可能原因：

- 单测主要覆盖接口、配置传递和 graph spec，不等价于真实物理和相机验证。
- 真实 demo 还依赖 assets、DexSim 物理、GPU、camera recorder、cached artifacts。

处理：

- 先看 `${OUT}.runner.log`。
- 再看 `summary.tsv` 和 `case_result.json`。
- 最后抽帧看视频，不能只看退出码。

### 5. `grasp_cup.py` 抓取失败

可能原因：

- semantic grasp cache 或候选方向不适配当前 mug。
- 夹爪接触几何和 candidate ranking 没有选中稳定抓取。
- place 时释放高度或 upright pose 不合适。
- DexSim 版本变化导致接触物理结果漂移。

处理：

- 先确认默认参数下是否失败，不要一开始开启 strict semantic grasp。
- 检查 `summary.tsv` 和 public grasp attempt 日志。
- 对比视频中失败发生在 grasp、lift、move 还是 place 阶段。

### 6. `pour_water_recovery_compare.py` clean case 失败

可能原因：

- cached artifacts 与当前代码或场景配置不匹配。
- public place / public move fallback 行为变化。
- CobotMagic solver device 处理异常。
- 初始物体位置或 assets 不一致。

处理：

- 先单独跑 `single_clean_no_recovery`。
- 再跑 `dual_clean_no_recovery`。
- clean case 未通过前，不建议继续分析 recovery case。

### 7. error no-recovery case 没有失败

可能原因：

- 错误注入 offset 太小，没有触发 monitor。
- error injection edge 或 step 配置不对。
- cached graph 中 monitor binding 与当前 case 不匹配。

处理：

- 检查 runner log 是否出现 `Injected forced recovery error`。
- 检查是否出现 `Monitor function monitor_object_moved triggered`。
- 必要时显式设置：

```bash
--error_injection_offset 0.12,0.0,0.0
--error_injection_type misplaced_object
```

### 8. error with-recovery case 失败

可能原因：

- monitor 已触发，但 recovery branch 没有正确执行。
- recovery artifacts 不是当前 case 对应的版本。
- public action 规划失败，例如 grasp、place 或 back_to_initial_pose 失败。
- task-state validation 判定视频/物理状态不合格。

处理：

- 检查 `${OUT}/<case>/artifacts/agent_compiled_graph.json`。
- 检查 `${OUT}/<case>/artifacts/case_result.json`。
- 看视频确认失败发生在 regrasp、move、place 还是 final validation。

### 9. 默认错误注入把 bottle 弄倒

正常情况下不应该发生。当前默认应是：

```text
--error_injection_type misplaced_object
--error_injection_offset 0.12,0.0,0.0
```

如果 bottle 倒下，可能原因：

- 命令显式传入了 `--error_injection_type fallen_object`。
- 使用了旧版本脚本，默认仍是 fallen-object。
- 物体被机械臂碰撞后倒下，而不是 error injection 直接弄倒。

处理：

- 先检查 runner log 中的 `error_type=`。
- 如果 log 显示 `misplaced_object` 但视频中后续倒下，说明是动作/碰撞导致，需要继续分析具体帧。

### 10. DexSim 0.4.0 下结果不同

这是预期风险，不应直接视为 0.3.11 baseline 复现失败。

可能差异包括：

- renderer / camera / recorder 行为变化
- contact physics 细节变化
- articulation 或 object pose 同步时序变化
- `WorldConfig` 或 `Renderer` enum 差异

处理：

- 先用 0.3.11 跑通本文 baseline。
- 再在 0.4.0 环境下运行同样命令。
- 对比 `summary.tsv`、`case_result.json`、runner log 和抽帧图片。
- 如果只有视频视角异常，优先修 camera/renderer。
- 如果视频正常但物理失败，优先分析 grasp/place/recovery 的动作轨迹。

## 需要提交给评审或测试人员的材料

建议同时提供：

- 当前分支名和 commit hash
- `docs/source/guides/dexsim0311_agent_atomic_action_repro.md`
- `outputs/agent_repro_compare` artifact bundle
- `grasp_cup.py` 的 runner log、`summary.tsv` 和视频
- `pour_water_recovery_compare.py --case all` 的 runner log、`summary.tsv`、`report.md`
  和关键视频
- 如果 0.4.0 下失败，提供 0.3.11 与 0.4.0 的同 case 对比视频和日志
