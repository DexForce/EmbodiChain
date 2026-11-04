# 单球 payload 修复与资源生命周期回归

本改动基于 #935 合并提交 `45d986e5f597c8c68bad092b0d0ed2be3511ab2c`，位于分支 `codex/observed-query-lifecycle`。只修改 EmbodiChain 的单球附件适配并补充原生回归与生命周期说明，不修改安装的 cuRobo，不增加并发服务、实例池或隐式全局 GC。

## 单球修复

cuRobo 0.8.0 的 `AttachmentManager.update()` 对 `transform_points()` 的结果执行 `.squeeze(0)`。只有一个球时，结果从 `(1,3)` 降为 `(3,)`，随后与 `(1,1)` 半径拼接失败。

在 `CuroboPlanner._attach_payload()` 中，单球路径先使用该 backend model 的观测关节 FK 和同一次观测的物体位姿计算 object-to-tool 变换，将球心转换成 link-local `(1,3)`，再调用 AttachmentManager 的无变换写入路径。保留一个球槽和原半径，不重复球体、不填充虚假球、不修改第三方库。多球路径保持原样。模型获取、别名去重、部分构造失败清理继续复用既有实现。

GPU 回归对全部独立 IK/优化/metrics 球体缓冲区直接核对结果：期望中心使用矩阵 `inverse(T_base_tool) @ T_base_world @ T_world_object` 独立计算，覆盖非零球心、旋转和位移。测试还交错执行单球/27 球查询，检查历史 positions/dt、机器人 qpos、正常 close、重复 close 及调用方抛异常后的回收。

同一个新回归在原始生产代码上失败（原生维度错误），在修复后通过；不是仅断言“返回结果存在”。失败注入和 attachment alias 清理继续使用已有 `test_curobo_query_snapshot.py`，未重复新增 mock 场景。

## 延迟回收原因

通过 `gc.DEBUG_SAVEALL` 捕获不可达对象，对这些对象的 `gc.get_referents()` 建图并求强连通分量，发现实际 native 对象回调环，详见 `native-cycles.json`：

- `LBFGSOpt → _core: GradientOptCore → step_direction_fn/on_resize/on_shift 等 bound methods → LBFGSOpt`。GradientOptCore 与 LineSearchContext 之间也存在通过 bound-method 回调形成的环。
- `PRMGraphPlanner → NodeSamplingStrategy/GraphConstructor/LinearConnector/PathPruner → check_samples_feasibility/_find_path_for_index_pairs 等 bound methods → PRMGraphPlanner`。

核对安装代码：`curobo/_src/optim/gradient/lbfgs.py` 构造 core 时传入这些 bound methods；`optim/components/gradient_opt_core.py` 保存回调；`graph_planner/graph_planner_prm.py` 初始化辅助对象时保存回调。`motion/motion_planner.py:destroy()` 的职责是释放 CUDA Graph，并不切断这些 Python 引用环。环关联的 optimizer/model 状态继续持有 CUDA tensor，直到 Python cyclic GC 执行。

诊断中的 DEBUG_SAVEALL 会有意保留 garbage，且建图局部变量本身也能持有引用，所以 `native-cycles.json` 的 after-GC 显存不是正常回收的测量。先前独立无 DEBUG_SAVEALL 探针中，保留历史轨迹时，删除最后局部 native 引用并 GC 后，allocated 从 529.70 MiB 降至 8.26 MiB。不能把这个结论扩张为所有长期负载均无泄漏。

## 明确的回收边界

`close()` 负责附件 detach、原生 planner destroy、清空 owned backend cache；不承诺立即消除第三方 Python 引用环，也不清空 PyTorch reserved allocator cache。

需要确定显存准入边界的 host，应完成查询并关闭 scope、释放 native 局部引用，在 owning GPU 的同步边界执行 cyclic GC 并再次同步，然后测量 live allocated。不要在每次查询中隐式执行全进程 GC，也不要直接篡改第三方 callback 字段来断环。批次长度及 GC 的延迟成本仍应由实际 host 的预算决定。

新的 GPU 生命周期测试显式禁用自动 GC，使边界不依赖 Python 阈值。在每个 host 批次边界执行 collect，使用 weakref 验证所有独立原生球体缓冲区消失，同时持有旧轨迹用于验证数据所有权。共六次查询，包含调用方异常退出；排除首次初始化后，剩余 live allocated 波动必须小于 8 MiB。该阈值远小于原先每查询约 50 MiB 的积累，并容纳少量合法历史轨迹/运行库分配。测试不要求 reserved 显存归零。

这是对已定位生命周期的明确验证；没有声称 close 本身已经变成即时释放，也没有增加长驻 worker。

## 验证环境与命令

RTX PRO 5000 72GB Blackwell，物理 GPU 3；Python 3.11、Torch 2.7.0+cu128、cuRobo 0.8.0、Warp 1.17.0、DexSim 0.5.1rc1。原生 GPU 验证不等于物理抓持或执行成功认证。

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 /root/miniconda3/envs/py311/bin/python -m pytest \
  tests/sim/motion/planners/test_curobo_integration.py::test_observed_payload_buffers_and_query_lifetimes \
  tests/sim/motion/planners/test_curobo_integration.py::test_curobo_v2_plans_a_joint_space_move \
  --run-gpu -q --junitxml=gpu.xml -o junit_family=xunit1

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 /root/miniconda3/envs/py311/bin/python -m pytest \
  tests/sim/motion/planners/test_curobo_query_snapshot.py \
  tests/sim/motion/planners/test_curobo_planner.py -q
```

GPU 验证 **2 passed**，已有 cuRobo 回归 **127 passed / 1 skipped / 2 deselected**，上下文测试 **42 passed**。六次回收后 live allocated 从 8,632,320 到 8,647,680 字节，仅增加 15,360 字节；所有原生缓冲区 weakref 失效，保留的历史轨迹仍有效。详细结果及字节序列见同目录 `checks.txt` 和 `gpu.xml`。API 未新增，read-only API gate 通过；motion collision-world 上下文补齐 native 回调环和 host GC 边界说明。`origin/main` 在本机落后于 #935，因此 affected 输出还包含基础 PR 的改动；本修复相对 HEAD 仅涉及 Motion 实现、原生回归及其上下文。
