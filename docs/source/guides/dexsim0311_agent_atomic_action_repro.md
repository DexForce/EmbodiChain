# DexSim 0.3.11 Agent 原子动作迁移复现指南

这份指南用于在本地 DexSim 0.3.11 运行时复现当前 Agent 原子动作迁移验证结果。
当前分支已经为 DexSim 0.4.0 兼容迁移做了准备，但下面记录的验证结果是在
DexSim 0.3.11 上得到的。请先使用这份指南验证公共原子动作接口、Agent adapter
路径、graph recovery runtime、视频录制和输出 artifacts，再判断哪些部分还需要针对
DexSim 0.4.0 继续修改。

## 范围

已验证分支：

```bash
git switch ljd/agentchoard_dexsim040_migration
```

这次复现覆盖的主要改动：

- 公共原子动作函数式 API：
  `move`, `pick_up`, `place`, `gripper_open` 和 `gripper_close`。
- Agent 原子技能 adapter 路径，用于通过 public move、place、gripper 和 grasp 执行。
- 支持 monitor 触发 recovery branch 的 graph recovery runtime。
- DexSim renderer 兼容层：在 DexSim 0.3.11 下保持 raster 渲染，同时为
  DexSim 0.4.0 的 `hybrid` 配置做准备。
- 两个 gym 验证 demo：
  `grasp_cup.py` 和 `pour_water_recovery_compare.py`。

## 环境

所有命令都在仓库根目录下运行：

```bash
cd "/home/dex/desktop/EmbodiChain/dexsim040/agentchoard"
```

如果本地路径包含中文字符，请使用实际路径：

```bash
cd "/home/dex/桌面/EmbodiChain/dexsim040/agentchoard"
```

使用已有 conda 环境：

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
- 在 0.3.11 下保持默认 raster 渲染路径。除非明确要测试 renderer 改动，否则不要传入
  `--enable_rt`。
- 使用 `conda run --no-capture-output -n embodichain ...`；直接运行裸 `python`
  可能因为缺少 `gymnasium` 等包而失败。

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

如果测试通过，第三方包产生的 warnings 可以接受。

## 必需的 Agent Artifacts 缓存

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

如果缓存缺失，请从验证机器复制 `outputs/agent_repro_compare` artifact bundle，
或者在已配置 LLM 的环境中重新生成 artifacts。默认 compare runner 会安装离线 LLM stub，
正常复现时预期使用 cached artifacts。

## Demo 1：Agent Skill Pick-Place

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

如果视频是黑屏、画面裁切异常，或者视角不符合预期，请优先调试 DexSim
renderer/camera 路径，再调试 Agent action 逻辑。

## Demo 2：Pour-Water Recovery Matrix

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

预期 case：

```text
single_clean_no_recovery
single_error_no_recovery
single_error_blind_no_recovery
single_clean_with_recovery
single_error_with_recovery
dual_clean_no_recovery
dual_error_no_recovery
dual_error_blind_no_recovery
dual_clean_with_recovery
dual_error_with_recovery
```

已验证的 0.3.11 运行中，`summary.tsv` 的预期行为：

- 所有 case 都应为 `expectation_matched=True`。
- clean case 应为 `program_success=True` 且 `semantic_success=True`。
- 不启用 recovery 的 error case 应在 monitor 触发后停止，因此预期
  `program_success=False`。
- 启用 recovery 的 error case 应为 `program_success=True` 且
  `semantic_success=True`。

默认确定性错误是 bottle 水平平移：

```text
--error_injection_type misplaced_object
--error_injection_offset 0.12,0.0,0.0
```

它不应主动将 bottle 弄倒。只有在需要做 fallen-object 压力测试时，才使用
`--error_injection_type fallen_object`。

## 可选关键帧检查

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

## 常见失败

`ModuleNotFoundError: No module named 'gymnasium'`
: 使用 `conda run --no-capture-output -n embodichain ...`，不要使用裸
  `python`。

缺少 cached artifacts
: 从验证机器复制 `outputs/agent_repro_compare`，或者配置 LLM 环境并显式重新生成
  artifacts。

视频视角异常或黑屏
: 优先检查 DexSim renderer/camera 兼容性。在 DexSim 0.3.11 下保持默认 raster
  renderer。不要在这份复现流程中使用 DexSim 0.4.0 的 hybrid renderer 路径。

DexSim 0.4.0 下物理结果不同
: 将其视为 DexSim 0.4.0 迁移问题，而不是 0.3.11 baseline 复现失败。安装
  DexSim 0.4.0 后重新运行相同命令，并对比视频、`summary.tsv` 和
  `case_result.json`。

## PR Commit Message 格式

标题使用下面任一前缀：

```text
NEW:
ENH:
FIX:
OTH:
```

commit 正文应使用下面格式：

```text
WHY: <为什么需要这次修改>

URL: <需求链接、bug 链接或 N/A>

TEST: <本次影响范围和已执行的测试；内容应具体，不能只写很短的占位说明>
```

对这次迁移，一个合适的 PR 标题是：

```text
ENH: Add dexsim040 agent atomic action migration
```

PR 描述中需要明确说明：该分支为 DexSim 0.4.0 兼容迁移做了准备，但这份指南中的复现
baseline 使用的是 DexSim 0.3.11。
