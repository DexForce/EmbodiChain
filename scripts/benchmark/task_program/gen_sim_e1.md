# E1 有界闭环实验

这是当前工作区的固定 cohort 实验入口，不是新的执行器，也不承诺任意 E 组合。
任务编译、运动、物理与验收全部调用生产实现；原始规范、资产和旧收据不改写。

## 固定范围

- 开发：q3 的 `block_001`；留出：事前选定的 `block_002`。
- 对照：原 q3 四块完整规范与 bundle，不重新编译或 reseal。
- 单块样例通过生产多物体展开后选择一个精确 UID，清空前驱表示独立新 episode。
  它不是原四块任务成功，也不计原20成绩。
- 单块编译在语义 planning 前加载同一 `FrozenPhysicalInput`。
  场景、机器人、执行策略与资产 digest 必须保持一致。
- 保留 `support_contract_revision=1`、`pick_contact_revision=2`，不改安全门。
- seed=0、单环境，沿用已有串行资源监督和1200秒单次 wall guard。
- 第0轮为三个基线；随后最多两轮受控假设，每轮同样三个 case，共最多9次入场。
  同一轮同一 case 只创建一次，不自动重试、删除旧结果或换简单样例。

## 运行

从仓库根目录运行。默认只运行一个 case，不自动推进整个 campaign：

```bash
python -m scripts.benchmark.task_program.gen_sim_e1 \
  --plan "outputs/gen_sim/e_small_steps_20261015/plan.json" --case dev --round 0
```

另外两个预选 case 是 `holdout` 和 `control`。后续轮次只能验证本批已证明的
同一个机制，不能为了耗尽预算而强行启动。`--prepare-only` 只作静态准备，
不获得 native 资格；该 attempt 仍保留，不允许原地覆盖重跑。

这是依赖当前已归档 q3 bundle 和现有 guarded worker 的本地实验入口，
不是可脱离这些输入独立运行的公共 benchmark 套件。

## 判读

输出在 plan 同级的 `attempts/r<round>-<case>/`。源码 zip、输入 manifest、
资源收据、原生执行报告、TaskSpec、轨迹与视频分别保留。
`status.json` 仅供进度页展示；资格依据是原始报告和附带哈希的独立审计。

命令退出、视频生成、`program_success=true` 或 post-policy 接受，都不单独
代表任务合格。必须区分软件验证、调用规划/运行、严格支撑证据和完整 TaskSpec。
睡眠保留 patch 的原始 sample 时间不能改成读取时间，缺证不能改判通过。

进度页沿用 `http://127.0.0.1:8767/#eComposition`；旧批次单独保留。

```bash
python -m pytest -q -c /dev/null --noconftest \
  "tests/benchmark/task_program/test_gen_sim_e1.py"
```

## 已封装输入

后续阶段可通过 `gensim.prepared-e-plan/v1` 复用同一串行 runner，不重复实现执行器。
每个 case 明确指定物理reference、source config、固定digest和已封装bundle。
不同bundle必须携带同一物理输入manifest；只能运行预先列出的case/round。
阶段性 TaskSpec 即使为 satisfied，也只按该实验范围判读，不自动认证整个E家族。
入场被资源门拒绝的记录保留，计入保守入场预算，不覆盖或自动重试。
