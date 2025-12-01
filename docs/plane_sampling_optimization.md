# 平面采样优化总结

## 优化概述

我们对 EmbodiChain 工作区分析器的平面采样功能进行了全面优化，提升了采样效率、精度和适用性。

## 主要改进

### 1. 架构重设计
- **分离关注点**: 将 `PlaneSampler` 设计为专门的高级采样器，而不是基础采样器的子类
- **组合而非继承**: `PlaneSampler` 内部使用基础采样器，提供更灵活的架构
- **明确职责**: 基础采样器负责坐标生成，平面采样器负责策略和优化

### 2. 智能边界计算
```python
# 新增配置选项
smart_bounds_calculation: bool = True
```
- **机器人几何感知**: 基于机器人关节限制估算工作空间边界
- **平面投影优化**: 将3D工作空间边界智能投影到2D平面坐标系
- **自适应边距**: 根据工作空间大小自动调整采样边距

### 3. 多种平面采样策略
```python
class PlaneSamplingStrategy(Enum):
    GRID = "grid"                    # 规则网格采样
    RANDOM = "random"                # 随机采样  
    STRATIFIED = "stratified"        # 分层采样（网格+随机抖动）
    IMPORTANCE_WEIGHTED = "importance_weighted"  # 重要性加权采样
    POLAR = "polar"                  # 极坐标采样
    ADAPTIVE_DENSITY = "adaptive_density"       # 自适应密度采样
```

### 4. 自适应采样密度
- **机器人位置感知**: 在机器人附近区域增加采样密度
- **工作空间几何优化**: 根据工作空间形状调整采样策略
- **混合策略**: 组合多种采样方法以获得最佳覆盖

### 5. 多层平面采样
```python
# 新增配置选项
plane_sampling_layers: int = 1        # 平行平面层数
plane_layer_spacing: Optional[float] = None  # 层间距
```
- **3D工作空间切片**: 支持在多个平行平面上同时采样
- **自动间距计算**: 根据工作空间范围自动计算最佳层间距
- **分布式采样**: 在各层之间合理分配采样点数

### 6. 优化的坐标系生成
```python
def _generate_orthogonal_basis(self, plane_normal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```
- **数值稳定性**: 使用改进的正交基向量生成方法
- **自动选择**: 基于平面法向量自动选择最稳定的坐标轴
- **可选方向优化**: 支持根据工作空间几何优化平面坐标系方向

## 使用示例

### 基础平面采样
```python
config = WorkspaceAnalyzerConfig(
    mode=AnalysisMode.CARTESIAN_SPACE,
    enable_plane_sampling=True,
    plane_normal=torch.tensor([0., 0., 1.]),  # XY平面
    plane_point=torch.tensor([0., 0., 0.8]),  # 高度0.8m
)
```

### 高级平面采样
```python
config = WorkspaceAnalyzerConfig(
    # 基础配置...
    adaptive_plane_sampling=True,
    smart_bounds_calculation=True,
    plane_sampling_layers=3,
    plane_layer_spacing=0.1,
    plane_sampling_strategy="adaptive_density",
)
```

### 直接使用平面采样器
```python
from embodichain.lab.sim.utility.workspace_analyzer.samplers.plane_sampler import (
    PlaneSampler, PlaneSamplingStrategy
)

sampler = PlaneSampler(
    strategy=PlaneSamplingStrategy.IMPORTANCE_WEIGHTED,
    base_sampler_strategy="sobol",
    importance_center=robot_position_2d,
    importance_decay=1.5
)

samples = sampler.sample(plane_bounds, num_samples)
```

## 性能提升

1. **采样质量**: 更均匀的样本分布，更好的工作空间覆盖
2. **计算效率**: 智能边界计算减少无效采样点
3. **机器人适配**: 基于机器人位置的自适应采样提高相关区域密度
4. **可扩展性**: 模块化设计便于添加新的采样策略

## 兼容性

- **向后兼容**: 保持原有API接口不变
- **配置驱动**: 通过配置选项控制新功能的启用
- **渐进式升级**: 可以逐步启用各项优化功能

## 配置选项总结

| 选项 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_plane_sampling` | bool | False | 启用平面采样 |
| `adaptive_plane_sampling` | bool | True | 启用自适应采样 |
| `smart_bounds_calculation` | bool | True | 智能边界计算 |
| `plane_orientation_optimization` | bool | True | 平面方向优化 |
| `plane_sampling_layers` | int | 1 | 平行平面层数 |
| `plane_layer_spacing` | float | None | 层间距（自动计算） |
| `plane_sampling_strategy` | str | None | 平面采样策略 |
| `plane_normal` | Tensor | [0,0,1] | 平面法向量 |
| `plane_point` | Tensor | [0,0,0] | 平面上的点 |
| `plane_bounds` | Tensor | None | 2D边界（自动计算） |

这些优化使得平面采样功能更加智能、高效和实用，特别适合桌面操作、墙面安装等常见机器人应用场景。