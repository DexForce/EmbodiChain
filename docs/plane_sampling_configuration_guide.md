# 平面采样工作空间分析配置指南

本文档详细介绍了EmbodiChain中平面采样工作空间分析的各种配置选项和最佳实践。

## 1. 基础配置结构

### 1.1 完整配置示例

```python
import torch
import numpy as np
from embodichain.lab.sim.utility.workspace_analyzer.workspace_analyzer import (
    WorkspaceAnalyzerConfig, AnalysisMode
)
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    SamplingConfig, SamplingStrategy,
    VisualizationConfig, VisualizationType,
    CacheConfig, DimensionConstraint, MetricConfig
)

# 平面采样完整配置
plane_config = WorkspaceAnalyzerConfig(
    # 核心设置
    mode=AnalysisMode.PLANE_SAMPLING,
    
    # 平面参数
    plane_normal=torch.tensor([0.0, 0.0, 1.0]),     # 平面法向量
    plane_point=torch.tensor([0.3, 0.0, 0.8]),      # 平面上的一点
    plane_bounds=torch.tensor([[-0.5, 0.5], [-0.5, 0.5]]),  # 2D边界
    
    # IK参数
    ik_success_threshold=0.8,    # IK成功阈值
    ik_samples_per_point=3,      # 每点IK尝试次数
    
    # 子配置
    sampling=sampling_config,
    visualization=visualization_config,
    cache=cache_config,
    constraint=constraint_config,
    metric=metric_config
)
```

## 2. 采样配置 (SamplingConfig)

### 2.1 采样策略详解

```python
# 均匀采样 - 适合全面覆盖
uniform_config = SamplingConfig(
    strategy=SamplingStrategy.UNIFORM,
    num_samples=2000,
    grid_resolution=20,    # 网格分辨率
    batch_size=500,
    seed=42
)

# 随机采样 - 快速估算
random_config = SamplingConfig(
    strategy=SamplingStrategy.RANDOM,
    num_samples=5000,
    batch_size=1000,
    seed=42
)

# Halton序列 - 准随机，良好覆盖
halton_config = SamplingConfig(
    strategy=SamplingStrategy.HALTON,
    num_samples=3000,
    batch_size=600
)

# Sobol序列 - 低差异序列
sobol_config = SamplingConfig(
    strategy=SamplingStrategy.SOBOL,
    num_samples=4096,      # 建议使用2的幂次
    batch_size=512
)

# 拉丁超立方采样 - 实验设计
lhs_config = SamplingConfig(
    strategy=SamplingStrategy.LATIN_HYPERCUBE,
    num_samples=1000,
    batch_size=200
)

# 高斯采样 - 概率性采样
gaussian_config = SamplingConfig(
    strategy=SamplingStrategy.GAUSSIAN,
    num_samples=2000,
    gaussian_mean=0.0,     # 均值（相对于边界中心）
    gaussian_std=0.3,      # 标准差（相对于边界范围）
    batch_size=400
)
```

### 2.2 采样策略选择指南

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| UNIFORM | 均匀覆盖，确定性 | 计算量大，维度诅咒 | 详细分析，小规模空间 |
| RANDOM | 快速，简单 | 覆盖不均匀 | 快速估算，大规模采样 |
| HALTON | 准随机，良好覆盖 | 高维性能下降 | 中等规模，平衡性能 |
| SOBOL | 低差异，高维友好 | 需要2^k个样本 | 高维空间，精确分析 |
| LATIN_HYPERCUBE | 每维度均匀 | 实现复杂 | 实验设计，参数研究 |
| GAUSSIAN | 概率重要性 | 可能遗漏边缘区域 | 特定区域重点分析 |

## 3. 平面定义参数

### 3.1 平面法向量设置

```python
# 水平面 (XY平面)
plane_normal = torch.tensor([0.0, 0.0, 1.0])

# 垂直面 (YZ平面)
plane_normal = torch.tensor([1.0, 0.0, 0.0])

# 垂直面 (XZ平面)
plane_normal = torch.tensor([0.0, 1.0, 0.0])

# 45度倾斜面
plane_normal = torch.tensor([0.0, 0.707, 0.707])

# 自定义方向（必须归一化）
custom_normal = torch.tensor([1.0, 2.0, 1.0])
plane_normal = custom_normal / torch.norm(custom_normal)
```

### 3.2 平面位置和边界

```python
# 平面上的基准点
plane_point = torch.tensor([0.3, 0.0, 0.8])  # [x, y, z]

# 2D平面坐标边界 [[u_min, u_max], [v_min, v_max]]
plane_bounds = torch.tensor([
    [-0.5, 0.5],  # u方向范围
    [-0.3, 0.7]   # v方向范围
])

# 大范围分析
large_bounds = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

# 小范围精细分析
fine_bounds = torch.tensor([[-0.2, 0.2], [-0.2, 0.2]])
```

## 4. 可视化配置 (VisualizationConfig)

```python
# 基础可视化
basic_viz = VisualizationConfig(
    enabled=True,
    vis_type=VisualizationType.POINT_CLOUD,
    point_size=3.0,
    show_unreachable_points=True,
    alpha=0.7
)

# 高质量可视化
quality_viz = VisualizationConfig(
    enabled=True,
    vis_type=VisualizationType.SPHERE,  # 球体渲染
    point_size=2.0,
    show_unreachable_points=False,
    alpha=0.9,
    color_by_distance=True,    # 按距离着色
    voxel_size=0.01           # 体素大小
)

# 性能优化可视化
fast_viz = VisualizationConfig(
    enabled=True,
    vis_type=VisualizationType.VOXEL,   # 体素化
    voxel_size=0.05,                   # 较大体素
    is_voxel_down=True,                # 下采样
    show_unreachable_points=False
)
```

## 5. 约束配置 (DimensionConstraint)

```python
# 基础约束
basic_constraint = DimensionConstraint(
    joint_limits_checking=True,
    joint_limits_scale=0.95,           # 关节限制缩放
    collision_checking=False,
    min_bounds=np.array([-1.5, -1.5, -0.5]),
    max_bounds=np.array([1.5, 1.5, 1.5])
)

# 严格约束
strict_constraint = DimensionConstraint(
    joint_limits_checking=True,
    joint_limits_scale=0.9,            # 更保守的关节限制
    collision_checking=True,           # 启用碰撞检测
    min_bounds=np.array([-1.0, -1.0, 0.0]),
    max_bounds=np.array([1.0, 1.0, 1.2]),
    # 排除区域
    exclude_zones=[
        (np.array([0.2, -0.3, 0.0]), np.array([0.5, 0.3, 0.8]))
    ]
)

# 宽松约束
loose_constraint = DimensionConstraint(
    joint_limits_checking=True,
    joint_limits_scale=0.98,           # 接近完整关节范围
    collision_checking=False,
    min_bounds=np.array([-2.0, -2.0, -1.0]),
    max_bounds=np.array([2.0, 2.0, 2.0])
)
```

## 6. 缓存配置 (CacheConfig)

```python
# 开发环境缓存
dev_cache = CacheConfig(
    enabled=True,
    cache_dir="./dev_cache",
    compression=False,          # 快速访问
    max_cache_size_mb=500,
    cache_format="pkl"          # Python pickle格式
)

# 生产环境缓存
prod_cache = CacheConfig(
    enabled=True,
    cache_dir="/app/cache",
    compression=True,           # 节省空间
    max_cache_size_mb=5000,
    cache_format="h5",          # HDF5格式，大数据友好
    use_hash=True              # 使用哈希验证
)

# 禁用缓存
no_cache = CacheConfig(enabled=False)
```

## 7. 度量配置 (MetricConfig)

```python
# 基础度量
basic_metrics = MetricConfig(
    compute_volume=True,
    compute_manipulability=False,   # 计算量大
    save_results=True
)

# 全面度量
comprehensive_metrics = MetricConfig(
    compute_volume=True,
    compute_manipulability=True,
    compute_density=True,
    save_results=True,
    # 可达性配置
    reachability=ReachabilityConfig(
        voxel_size=0.02,
        compute_coverage=True
    ),
    # 可操作性配置
    manipulability=ManipulabilityConfig(
        jacobian_threshold=0.001,
        compute_isotropy=True
    )
)
```

## 8. IK参数调优

```python
# 快速IK配置
fast_ik_config = WorkspaceAnalyzerConfig(
    ik_success_threshold=0.7,      # 较低阈值
    ik_samples_per_point=1,        # 每点一次尝试
    # ... 其他配置
)

# 高精度IK配置
precise_ik_config = WorkspaceAnalyzerConfig(
    ik_success_threshold=0.95,     # 高阈值
    ik_samples_per_point=5,        # 多次尝试
    # ... 其他配置
)

# 平衡配置
balanced_ik_config = WorkspaceAnalyzerConfig(
    ik_success_threshold=0.8,      # 中等阈值
    ik_samples_per_point=3,        # 适中尝试次数
    # ... 其他配置
)
```

## 9. 常见平面采样场景

### 9.1 桌面操作分析

```python
# 桌面高度的水平面分析
desktop_config = WorkspaceAnalyzerConfig(
    mode=AnalysisMode.PLANE_SAMPLING,
    plane_normal=torch.tensor([0.0, 0.0, 1.0]),    # 水平
    plane_point=torch.tensor([0.0, 0.0, 0.75]),     # 桌面高度
    plane_bounds=torch.tensor([[-0.8, 0.8], [-0.6, 0.6]]),  # 桌面范围
    sampling=SamplingConfig(
        strategy=SamplingStrategy.UNIFORM,
        num_samples=2000
    )
)
```

### 9.2 墙面操作分析

```python
# 垂直墙面分析
wall_config = WorkspaceAnalyzerConfig(
    mode=AnalysisMode.PLANE_SAMPLING,
    plane_normal=torch.tensor([1.0, 0.0, 0.0]),    # 垂直于X轴
    plane_point=torch.tensor([0.6, 0.0, 0.5]),     # 墙面位置
    plane_bounds=torch.tensor([[-0.4, 0.4], [0.2, 1.2]]),  # 墙面范围
    sampling=SamplingConfig(
        strategy=SamplingStrategy.HALTON,
        num_samples=1500
    )
)
```

### 9.3 倾斜表面分析

```python
# 30度倾斜表面
tilt_config = WorkspaceAnalyzerConfig(
    mode=AnalysisMode.PLANE_SAMPLING,
    plane_normal=torch.tensor([0.0, 0.5, 0.866]),  # 30度倾斜
    plane_point=torch.tensor([0.0, 0.3, 0.8]),
    plane_bounds=torch.tensor([[-0.5, 0.5], [-0.3, 0.3]]),
    sampling=SamplingConfig(
        strategy=SamplingStrategy.SOBOL,
        num_samples=2048
    )
)
```

## 10. 性能优化建议

### 10.1 采样数量指导

```python
# 快速预览 (< 5秒)
preview_samples = 500

# 标准分析 (5-30秒)
standard_samples = 2000

# 详细分析 (30秒-5分钟)
detailed_samples = 10000

# 高精度分析 (> 5分钟)
precision_samples = 50000
```

### 10.2 批处理大小优化

```python
# 内存限制的设置
memory_constrained = SamplingConfig(
    num_samples=10000,
    batch_size=500     # 小批次
)

# 性能优化设置
performance_optimized = SamplingConfig(
    num_samples=10000,
    batch_size=2000    # 大批次
)
```

## 11. 故障排除

### 11.1 常见问题

1. **IK成功率过低**
   - 降低 `ik_success_threshold`
   - 增加 `ik_samples_per_point`
   - 检查平面位置是否在机器人工作范围内

2. **采样速度慢**
   - 减少 `num_samples`
   - 增加 `batch_size`
   - 选择更快的采样策略（RANDOM vs UNIFORM）

3. **内存不足**
   - 减小 `batch_size`
   - 禁用不必要的度量计算
   - 使用体素化可视化

### 11.2 调试配置

```python
debug_config = WorkspaceAnalyzerConfig(
    mode=AnalysisMode.PLANE_SAMPLING,
    sampling=SamplingConfig(
        strategy=SamplingStrategy.UNIFORM,
        num_samples=100,        # 小样本
        batch_size=25
    ),
    visualization=VisualizationConfig(
        enabled=True,
        point_size=5.0,         # 大点便于观察
        show_unreachable_points=True
    ),
    cache=CacheConfig(enabled=False),  # 禁用缓存
    ik_success_threshold=0.5,          # 宽松阈值
    ik_samples_per_point=1
)
```

这个配置指南提供了全面的平面采样设置选项，您可以根据具体需求选择合适的配置组合。