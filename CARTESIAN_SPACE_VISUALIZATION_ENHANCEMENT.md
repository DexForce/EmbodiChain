# WorkspaceAnalyzer Cartesian Space 可视化增强

## 修改概述

本次修改增强了 `WorkspaceAnalyzer` 在 Cartesian space 模式下的可视化功能，现在可以同时显示可达和不可达的点位，并用不同的颜色和大小进行区分。

## 主要修改

### 1. `compute_reachability` 方法修改

**修改前：** 只返回可达的点
```python
def compute_reachability(...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 只返回: reachable_points, success_rates, best_configs
```

**修改后：** 返回所有点及其可达性信息
```python
def compute_reachability(...) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 返回: all_points, reachable_points, success_rates, reachability_mask, best_configs
```

### 2. Cartesian space 分析结果存储

**修改前：** 只存储可达的点
```python
self.workspace_points = reachable_points  # 只存储可达点
```

**修改后：** 存储所有点及可达性信息
```python
self.workspace_points = all_points          # 存储所有采样点
self.reachable_points = reachable_points    # 存储只可达点
self.reachability_mask = reachability_mask  # 存储可达性掩码
```

### 3. 可视化颜色和大小生成

**新增方法：** `_generate_point_colors_and_sizes`
- 可达点：🟢 绿色，1.5倍大小
- 不可达点：🔴 红色，0.7倍大小

**向后兼容：** 保持原有 `_generate_point_colors` 方法

### 4. 可视化方法增强

**修改前：** 只支持颜色
```python
vis_obj = visualizer.visualize(points_np, colors=colors)
```

**修改后：** 支持颜色和大小
```python
try:
    vis_obj = visualizer.visualize(points_np, colors=colors, sizes=sizes)
except TypeError:
    vis_obj = visualizer.visualize(points_np, colors=colors)  # 降级支持
```

## 使用方法

### 基本用法

```python
from embodichain.lab.sim.utility.workspace_analyzer import (
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
    AnalysisMode,
)

# 配置 Cartesian space 模式
config = WorkspaceAnalyzerConfig(
    mode=AnalysisMode.CARTESIAN_SPACE,
    ik_success_threshold=0.8,  # IK成功阈值
    ik_samples_per_point=3,    # 每个点的IK尝试次数
)

# 创建分析器
analyzer = WorkspaceAnalyzer(robot=robot, config=config)

# 分析并可视化
results = analyzer.analyze(num_samples=2000, visualize=True)
```

### 可视化配置

```python
from embodichain.lab.sim.utility.workspace_analyzer.configs import VisualizationConfig

viz_config = VisualizationConfig(
    enabled=True,
    vis_type="point_cloud",
    point_size=8.0,     # 基础点大小
    alpha=0.8,          # 透明度
    color_by_distance=False,  # 按可达性着色而不是距离
    show_unreachable_points=True,  # 是否显示不可达点位
)

config = WorkspaceAnalyzerConfig(
    mode=AnalysisMode.CARTESIAN_SPACE,
    visualization=viz_config,
)
```

## 结果解释

### 分析结果包含的新字段

```python
results = {
    "all_points": all_points,              # 所有采样的Cartesian点
    "reachable_points": reachable_points,  # 只包含可达点
    "success_rates": success_rates,        # 每个点的IK成功率
    "reachability_mask": reachability_mask, # 可达性布尔掩码
    "num_samples": 2000,                   # 总采样数
    "num_reachable": 1200,                 # 可达点数量
}
```

### 可视化效果

- **🟢 绿色大点**：可达的Cartesian位置（success_rate >= ik_success_threshold）
- **🔴 红色小点**：不可达的Cartesian位置（success_rate < ik_success_threshold）
- **点大小**：
  - 可达点：`point_size * 1.2`
  - 不可达点：`point_size * 0.7`

### 显示控制选项

- **`show_unreachable_points=True`**（默认）：显示所有点
  - 绿色大点 + 红色小点
  - 清晰显示工作空间边界和"洞"
- **`show_unreachable_points=False`**：只显示可达点
  - 仅绿色大点
  - 突出显示有效工作区域

## 配置参数

### 关键参数说明

- **`ik_success_threshold`**: IK成功阈值（默认0.9）
  - 值越高，标准越严格，更少的点被视为可达
  - 推荐范围：0.5-0.9

- **`ik_samples_per_point`**: 每个Cartesian点尝试的IK种子数（默认1）
  - 值越高，IK求解越可靠，但计算时间更长
  - 推荐范围：1-5

- **`min_bounds/max_bounds`**: Cartesian采样空间边界
  - 限制采样范围可提高效率
  - 格式：`[x_min, y_min, z_min]` / `[x_max, y_max, z_max]`

## 兼容性

- ✅ 保持与现有代码的完全向后兼容
- ✅ Joint space 模式不受影响
- ✅ 现有的可视化接口继续工作
- ✅ 支持所有现有的后端（sim_manager, open3d, matplotlib）

## 示例文件

新增示例文件：`examples/cartesian_space_reachability_visualization.py`

包含以下示例：
1. 基本的Cartesian space可达性可视化
2. Joint space vs Cartesian space对比
3. 不同IK阈值的影响分析

## 性能说明

- Cartesian space 模式比 Joint space 模式慢，因为需要进行IK计算
- 增加 `ik_samples_per_point` 会线性增加计算时间
- 建议在调试时使用较少样本（~500），在最终分析时使用更多样本（2000+）