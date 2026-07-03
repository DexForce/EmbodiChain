# Demo 脚本标准类 / 共享工具设计文档

**Date:** 2026-07-04  
**Author:** Claude Code (brainstorming session)  
**Status:** Approved

---

## 1. 背景与问题

`scripts/tutorials/` 与 `examples/` 中包含大量 demo 脚本，它们在完成不同任务的同时，重复了相当多的样板代码：

- `np.set_printoptions` / `torch.set_printoptions`
- `SimulationManager` 配置、灯光、窗口打开
- 命令行参数解析（`--headless`、`--device`、`--renderer`、`--auto_play`、`--record_*`）
- 视频录制（`start_window_record` / `stop_window_record` / `wait_window_record_saves`）
- 轨迹回放循环
- `KeyboardInterrupt` + `sim.destroy()` 清理

`scripts/tutorials/atomic_action/tutorial_utils.py` 已经封装了一部分功能，但：

1. 位置偏（仅 atomic_action 使用）；
2. 缺少 headless / 离屏 camera 录制；
3. 没有可复用的生命周期基类。

本设计旨在抽象出一个**可复用的 demo 工具层 + 可选生命周期基类**，降低新 demo 的书写成本，统一录制、清理等行为。

---

## 2. 设计目标

1. **减少样板代码**：新 demo 只需要关注核心逻辑（setup + run）。
2. **统一参数与录制**：所有 demo 共享一致的 CLI 参数和视频录制行为。
3. **向后兼容**：不破坏现有 `tutorial_utils.py` 的导入。
4. **渐进式迁移**：先改造 `atomic_action` 作为试点，再推广到 `examples/sim/demo/`。
5. **不强迫短脚本**：gizmo / sensor / solver 等生命周期短的脚本可继续扁平编写。

---

## 3. 非目标

- 不替代 `embodichain.lab.gym.utils.gym_utils.add_env_launcher_args_to_parser()`，而是在其之上追加 demo 专用参数。
- 不封装所有可能的机器人配置；只提供常见 preset 工厂，复杂配置仍由脚本自行构造。
- 不引入重量级插件系统或配置化 DSL。

---

## 4. 架构

新增两个共享模块：

```
embodichain/lab/sim/
├── utility/
│   └── demo_utils.py      # 可独立调用的工具函数 / 小类
└── demo_base.py           # 可选的轻量 DemoBase 生命周期基类
```

### 4.1 `demo_utils.py` — 工具函数层（核心）

所有 demo 脚本均可按需调用，保持扁平自由。

主要 API：

| 函数 / 类 | 说明 |
|---|---|
| `add_demo_args(parser)` | 在 `add_env_launcher_args_to_parser()` 基础上追加 `--auto_play`、`--record_steps`、`--record_fps`、`--record_save_path`、`--no_vis_eef_axis`。 |
| `create_default_sim(args, *, width, height, physics_dt, arena_space, add_default_light)` | 返回配置好的 `SimulationManager`，默认加一盏主灯。 |
| `shutdown_sim(sim)` | 安全调用 `sim.destroy()`。 |
| `DemoRecording(sim, args, prefix)` | 上下文管理器，统一处理 window record / headless offscreen camera 录制、文件名生成、停止与等待保存。 |
| `replay_trajectory(sim, robot, traj, *, post_steps, step_size, sleep, arm_name)` | 统一轨迹回放循环。 |
| `maybe_open_window(sim, args)` | `if not args.headless: sim.open_window()`。 |
| `maybe_wait_for_user(args, prompt)` | `if not args.auto_play: input(prompt)`。 |
| `maybe_pause_for_inspection(args)` | `if not args.auto_play: input("Press Enter to finish...")`。 |
| `maybe_init_gpu_physics(sim)` | `if sim.is_use_gpu_physics: sim.init_gpu_physics()`。 |
| `setup_print_options()` | 设置 `np` / `torch` 打印格式。 |
| `format_tensor(tensor)` | 标准化 tensor 字符串输出。 |
| `create_robot_from_preset(sim, preset, **kwargs)` | 常见机器人 preset 工厂（如 `"ur5_gripper"`、`"ur10"`、`"dexforce_w1"`），复杂配置仍由脚本自行构造。 |

### 4.2 `demo_base.py` — 可选生命周期基类

适合有完整 setup/run/cleanup 的 demo。

```python
class DemoBase(ABC):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sim: SimulationManager | None = None

    @abstractmethod
    def setup(self) -> None:
        """Create sim, robot, camera, etc."""

    @abstractmethod
    def run(self) -> None:
        """Demo main logic."""

    def cleanup(self) -> None:
        if self.sim is not None:
            shutdown_sim(self.sim)

    def main(self) -> None:
        self.setup()
        try:
            self.run()
        finally:
            self.cleanup()
```

子类只需实现 `setup()` 与 `run()`，生命周期由基类保证。

---

## 5. 数据流

### 5.1 使用 `DemoBase` 的脚本（如 `atomic_action/pickup.py`）

```
main()
  └── DemoBase.main()
        ├── add_demo_args(parser) + parse_args()
        ├── setup()
        │     ├── create_default_sim(args) → SimulationManager + light
        │     ├── create_robot_from_preset(sim, "ur5_gripper")
        │     └── create_default_motion_generator(robot)
        ├── run()
        │     ├── maybe_wait_for_user(args, "Press Enter to plan...")
        │     ├── with DemoRecording(sim, args, prefix="pickup"):
        │     │     └── replay_trajectory(sim, robot, traj)
        │     └── maybe_pause_for_inspection(args)
        └── cleanup()
              └── shutdown_sim(sim)
```

### 5.2 使用纯工具函数的短脚本（如 `motion_generator.py`）

```python
args = add_demo_args(parser).parse_args()
sim = create_default_sim(args)
try:
    # ... demo-specific logic ...
    with DemoRecording(sim, args, prefix="motion_generator"):
        # ...
finally:
    shutdown_sim(sim)
```

---

## 6. 错误处理

1. **录制失败不中断 demo**：`DemoRecording` 内部捕获 `start_window_record` 失败，打印 warning 并跳过录制，主流程继续。
2. **清理必执行**：`DemoBase.main()` 使用 `try/finally`；纯函数场景在文档中给出 `try/finally` 示例。
3. **参数缺失降级**：`--record_save_path` 未指定时，默认保存到 `./recordings/<prefix>_<timestamp>.mp4`。
4. **GPU 物理初始化显式化**：`create_default_sim()` 不自动调用 `init_gpu_physics()`，保留 `maybe_init_gpu_physics(sim)` 让用户在 setup 后按需调用，避免隐藏副作用。

---

## 7. 迁移策略

1. **保留 `tutorial_utils.py`**：`scripts/tutorials/atomic_action/tutorial_utils.py` 保留 UR5+gripper 相关的专用配置函数（如 `create_ur5_gripper_robot_cfg`），并把通用工具函数（如 `start_auto_play_recording`、`stop_auto_play_recording`、`draw_axis_marker`）改为从新的 `demo_utils.py` re-export，现有导入继续工作。
2. **试点**：优先改造 `scripts/tutorials/atomic_action/` 6 个脚本，验证 API 设计。
3. **推广**：第二步改造 `examples/sim/demo/` 4 个脚本（`grasp_cup_to_caffe`、`pick_up_cloth`、`press_softbody`、`scoop_ice`）。
4. **不强制迁移**：gizmo / sensor / solver 等短脚本可继续使用原有写法或仅引用部分工具函数。

---

## 8. 测试策略

1. **单元测试**：`tests/lab/sim/test_demo_utils.py`
   - `add_demo_args` 添加了预期参数；
   - `format_tensor` 输出格式正确；
   - `DemoRecording` 在 `auto_play=False` 时不启动录制；
   - `shutdown_sim` 正确调用 `sim.destroy()`。

2. **集成测试**：改造 `atomic_action/pickup.py` 后，以 `--headless --auto_play` 运行，确认不 crash 且生成视频文件。

3. **向后兼容测试**：确认旧脚本继续导入 `tutorial_utils.py` 无异常。

---

## 9. 后续工作

1. 实现 `embodichain/lab/sim/utility/demo_utils.py`。
2. 实现 `embodichain/lab/sim/demo_base.py`。
3. 更新 `scripts/tutorials/atomic_action/tutorial_utils.py` 为 re-export 模式。
4. 改造 `atomic_action` 脚本并跑通集成测试。
5. 视情况推广到 `examples/sim/demo/`。
