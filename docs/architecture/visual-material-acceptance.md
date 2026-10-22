# VisualMaterial 设计与正确性验收

## 适用版本与运行环境

本方案针对 `dexsim_engine==0.5.1rc1`，使用该版本要求的
`warp-lang==1.17.0`、`newton[sim]==1.6.0`。验收必须读取实际导入环境的版本，
不能用修改包元数据、旧版替身或源码隔离探针替代原生集成结果。

本次复现环境为独立虚拟环境 `/tmp/embodichain-material-rc1`，底层 Python 3.10，
GPU 为 NVIDIA GeForce RTX 5090。本机 wheel 来自
`~/Downloads/build_vcpkg-linux-cuda-py310/build_Release/lib/python_package/pip_package/`。
其他机器使用同版本、匹配 Python ABI 的正式 wheel，并把下方 `python` 指向对应环境。

```bash
python -c 'from importlib.metadata import version; import dexsim; assert version("dexsim-engine") == "0.5.1rc1"; print(dexsim.__file__, version("warp-lang"), version("newton"))'
```

## 设计契约

- `VisualMaterialCfg` 描述材质；`material.py:_material_desc_from_cfg` 是运行时和
  Spawn 共用的属性校验、发光强度和材质类型转换边界。
- 未命名材质的 `uid=None`。形状声明使用对象范围及标准化外观指纹生成名字，允许同 UID
  对象更换外观，并复用相同定义；独立运行时创建使用唯一名字。两条路径共用定义注册表，
  显式相同 UID 只允许等价定义复用，不同定义在原生写入前报错。等价别名保留已有运行时编辑。
- 一个模板内每个 UID 对应一个稳定的 `VisualMaterialInst`。重复创建、查询不修改实例，
  不追加重复登记，也不重复写默认属性。需要重设时显式调用 `set_default_properties`；
  确定性刚体和群组 reset 事件负责这个操作。
- 实例持有确定的原生句柄。包装导入资产、创建复用实例后，所有 setter 操作同一句柄。
  材质身份使用原生 material ID，不能仅依赖模板名与实例名。
- metallic、roughness、IOR 和颜色读取真实原生状态。`0.5.1rc1` 的颜色 getter 只暴露
  display-sRGB 的 RGB，因此 `instance.base_color` 返回 RGB；配置和
  `set_base_color` 继续接收 RGBA。不能根据 getter 声称验证了 alpha。
- 发光 RGB 与非负强度相乘；零强度可恢复。两条创建路径使用同一转换，
  `BRDF/BTDF/BSDF` 都传递给原生材质。
- 非共享随机化仅影响所选环境；共享随机化要求选择全部环境，并且只采样、修改一次。
  空选择是无操作；重复、越界或非整数环境索引必须在材质写入前失败。
- 纯色分支保留有界纹理色板，同时应用配置的 metallic、roughness 和 IOR 范围。
  相同规范化路径共享纹理缓存；不同路径即使目录同名也不能混用。
- Spawn 在拓扑重建前保存已绑定对象的当前逐段材质，在绑定阶段重放。
  保存的是当前状态，reset 基线仍由对象持有。重放失败时保留快照，成功后释放；
  移除对象时丢弃它的待重放记录，不能把旧对象快照写给同 UID 的新对象。

## 自动化验收矩阵

| 层级 | 用例与独立判据 | 覆盖入口 | 通过标准 |
|---|---|---|---|
| CPU：实例身份 | 同 UID 创建/查询 1,025 次；先修改属性再重复查询 | 模板、实例 | 包装器身份稳定；属性保留；登记容量只与唯一 UID 数有关；缓存命中无原生属性写入 |
| CPU：原生句柄 | 资产包装器对颜色、PBR 和各纹理通道写入 | `from_existing`、工作实例 | 原始句柄接收写入；不再按名字寻找替代实例 |
| CPU：身份比较 | 两个句柄名字相同、material ID 不同 | 逐段赋值 | 必须发生重新绑定；相同 ID 的等价句柄跳过重复绑定 |
| CPU：创建一致性 | 三种材质类型 × 强度 0、1、10；非默认颜色、roughness、metallic、IOR | 运行时与 Spawn | 两条路径提交相同属性；alpha 写入与启用标志保留 |
| CPU：非法输入 | 错误颜色维度、NaN、越界因子、非正 IOR、负发光强度、未知类型 | 两条创建路径 | 原生实例创建前失败 |
| CPU：命名与冲突 | 两个未命名红/蓝材质；显式同 UID 的等价与冲突定义 | Manager、Spawn | 独立材质不串色；等价复用；冲突失败且原注册不变 |
| CPU：跨入口别名 | 两个创建顺序；已有原生属性修改后创建等价运行时别名 | 共用定义注册表、原生默认实例 | 两个方向都拒绝冲突；等价别名不重置属性；Native 默认实例不出现未初始化灰色 |
| CPU：群组 reset | 群组实例修改属性后重放确定性事件 | 群组 setter、固定材质事件 | 所选环境恢复配置；未选环境保留修改 |
| CPU：环境选择 | 空集、非连续选择、共享局部选择、共享全量乱序选择 | 复用与回退路径 | 非目标环境无写入；共享局部选择拒绝；共享全量只修改一次 |
| CPU：随机化参数 | 固定 PBR 范围；仅重置一个关节体环境 | 纯色、纹理库、回退 | 固定范围精确生效；仅为所选环境/链接采样 |
| CPU：资源上界 | 多次调用与两个同名纹理目录 | 纹理缓存、色板 | 预热后无纹理创建；不同来源各自加载；同一来源复用 |
| CPU：重建与失败恢复 | 陈旧原生描述模拟重建；逐段重放中途失败 | 刚体、关节体、体/面可变形对象 | 当前材质逐段恢复；失败后重试成功；成功后快照释放 |
| CPU：失败后替换 | 重放失败后移除，再添加相同 UID、不同段数的新对象 | Spawn 移除与重试 | 旧快照被丢弃；新对象的材质与默认值不被污染 |
| 原生：Hybrid | 两环境，Default 与 Newton；独立红/蓝材质；临时赋值→恢复→增删其他对象 | Manager、Spawn、刚体、纯渲染对象 | 原生颜色/PBR 与配置一致；当前替换和 reset 结果在重建后保留；另一环境不变 |
| 原生：持续复用 | 每个后端执行 1,025 次局部随机化，读取原生模板实例数与最终属性 | 原生工作实例、PBR、选择隔离 | 预热后原生实例数不增长；所选范围生效；未选环境保留原材质 |
| 原生：替换与别名 | 同 UID 改色替换、相同定义重复替换、移除后重加、两个入口的等价别名 | Default 与 Newton | 外观正确改变；相同定义不新增登记；别名不重置已编辑原生属性 |

浮点比较采用 `pytest.approx`；原生颜色往返绝对容差为 `1e-5`。
随机化逻辑用固定范围验证参数生效，避免以随机采样“恰好不同”作为测试判据。
纹理色板的颜色受 uint8 量化及有限色板分布影响，不应要求每次产生任意连续颜色。

## 执行命令

CPU 与原生测试分开执行，保持原生资源串行使用。

```bash
python -m pytest -q -m 'not requires_sim and not gpu' \
  tests/sim/test_material.py tests/sim/test_material_texture_obj.py \
  tests/sim/test_material_cache_config.py \
  tests/sim/objects/test_asset_material_initialization.py \
  tests/sim/objects/test_rigid_object_reuse_material.py \
  tests/sim/objects/test_articulation_reuse_material.py \
  tests/gym/envs/managers/test_randomize_visual_material.py \
  tests/sim/objects/test_visual_object.py \
  tests/sim/spawn tests/sim/test_cfg.py tests/gym/utils/test_gym_utils.py

python -m pytest -q --run-gpu \
  tests/sim/test_material_integration.py \
  tests/sim/objects/test_visual_object_integration.py::test_render_only_object_native_lifecycle

python -m black --check .
python docs/scripts/check_api_docs.py
python .agents/skills/project-dev-context/scripts/context.py check
```

## 本次执行记录（2026-10-23）

使用上面的独立环境和真实 rc1 wheel；项目以 editable 方式安装，包含官方机器人与任务扩展。
CPU 结果中的 4 个 deselected 用例由资源标签排除，不计入通过数。

| 检查 | 实际结果 |
|---|---|
| 扩展 CPU 回归（含 Spawn、配置与 Gym 配置加载） | 685 passed，4 deselected，12.37 s |
| 原生 Hybrid / Default + Newton | 4 passed，17.83 s；每个后端包含 1,025 次持续局部随机化 |
| 项目上下文测试 | 42 passed |
| Black 26.3.1 全仓检查 | 1,506 个 Python 文件无需更改 |
| 公共 API 文档覆盖 | 2,805 / 2,805，零遗漏 |
| 上下文链接与路由、diff 空白检查 | 通过 |

API、引擎生命周期和 HPC 三个独立复核发现的群组 reset、跨入口身份、同 UID 替换、
失败后快照和等价别名状态问题均已修正并补测。此记录不包含下面的光学图像和长期显存资格项目。

## 兼容性与后续发布验收

旧配置如果显式保存了 `uid: default_mat`，它现在是一个显式共享名字；需要独立外观时
删除该字段或指定不同 UID。`get_instance` 和同 UID 的 `create_instance` 返回同一包装器，
不再隐式重置属性；旧调用者若依赖重复创建覆盖属性，应改为显式应用配置。
读取颜色的调用者应接受原生 RGB；RGBA 写入保持支持。
`clean_materials()` 清理运行时包装器和未使用的原生缓存，已登记的显式 UID 定义在 World
生命周期内保持身份，避免现有场景或原生句柄被同名不同属性的定义悄悄替换；不同定义使用新 UID。

这组测试验收的是材质状态、资源上界和生命周期，不能等同于完整光学图像验收。
发布时还应在独立 FastRT 进程增加以下资格验证，并保存带环境信息的图片：

1. 同一灯光、相机、曝光与几何下，BRDF/BTDF/BSDF 的反射和透射表现；透明物体后放置对照色块。
2. 发光强度 0、1、10 的暗室对照；比较数值与图像，避免仅凭亮度饱和判断强度。
3. 带多个材质段的 URDF/网格、关节体与可变形对象，在局部 reset 和拓扑重建后保持各段材质。
4. 文件、Tensor 与原生 Texture 三种输入的颜色空间、法线贴图方向和重复绑定行为。
5. 记录 10,000 次 reset 的原生实例数、纹理数和 VRAM。区分预热缓存与持续增长，
   不把 Torch 分配器统计当成全部 Vulkan/OptiX 显存。

每次验收记录：提交/工作树、实际 wheel 路径与版本、Python/Warp/Newton、GPU/驱动、
后端与渲染器、命令、通过/失败/跳过项及原因。任何未执行的图像或后端矩阵保持待验证，
不能由 CPU mock 测试替代。
