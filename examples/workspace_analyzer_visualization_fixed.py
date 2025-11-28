#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：修复后的工作空间分析器可视化配置

展示如何正确配置不同类型的可视化，避免参数传递错误。
"""

from embodichain.lab.sim.utility.workspace_analyzer import (
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
    AnalysisMode,
)
from embodichain.lab.sim.utility.workspace_analyzer.configs import VisualizationConfig
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import VisualizationType


def example_point_cloud_visualization(robot, sim_manager=None):
    """使用点云可视化的示例 - 修复了参数传递问题"""

    viz_config = VisualizationConfig(
        enabled=True,
        vis_type=VisualizationType.POINT_CLOUD,
        point_size=6.0,  # 点云特有参数
        alpha=0.8,
    )

    config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.JOINT_SPACE,
        visualization=viz_config,
    )

    analyzer = WorkspaceAnalyzer(
        robot=robot,
        config=config,
        sim_manager=sim_manager,
    )

    # 分析并可视化
    results = analyzer.analyze(num_samples=1000, visualize=True)
    return results


def example_voxel_visualization(robot, sim_manager=None):
    """使用体素可视化的示例 - 修复了参数传递问题"""

    viz_config = VisualizationConfig(
        enabled=True,
        vis_type=VisualizationType.VOXEL,
        voxel_size=0.03,  # 体素特有参数
        alpha=0.6,
    )

    config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.JOINT_SPACE,
        visualization=viz_config,
    )

    analyzer = WorkspaceAnalyzer(
        robot=robot,
        config=config,
        sim_manager=sim_manager,
    )

    # 分析并可视化
    results = analyzer.analyze(num_samples=1000, visualize=True)
    return results


def example_sphere_visualization(robot, sim_manager=None):
    """使用球体可视化的示例 - 修复了参数传递问题"""

    viz_config = VisualizationConfig(
        enabled=True,
        vis_type=VisualizationType.SPHERE,
        sphere_radius=0.008,  # 球体特有参数
        sphere_resolution=12,  # 球体特有参数
        alpha=0.4,
    )

    config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.JOINT_SPACE,
        visualization=viz_config,
    )

    analyzer = WorkspaceAnalyzer(
        robot=robot,
        config=config,
        sim_manager=sim_manager,
    )

    # 分析并可视化
    results = analyzer.analyze(num_samples=1000, visualize=True)
    return results


def example_complete_configuration(robot, sim_manager=None):
    """完整配置示例，展示所有新增的配置选项"""

    viz_config = VisualizationConfig(
        enabled=True,
        vis_type=VisualizationType.POINT_CLOUD,  # 或者 "point_cloud"
        # 点云可视化参数
        point_size=5.0,
        # 体素可视化参数
        voxel_size=0.04,
        # 球体可视化参数
        sphere_radius=0.006,
        sphere_resolution=15,
        # 通用可视化参数
        alpha=0.7,
        # 其他处理参数
        nb_neighbors=20,
        std_ratio=2.0,
        is_voxel_down=True,
        color_by_distance=True,
    )

    config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.JOINT_SPACE,
        visualization=viz_config,
    )

    analyzer = WorkspaceAnalyzer(
        robot=robot,
        config=config,
        sim_manager=sim_manager,
    )

    return analyzer


def example_runtime_safe_switching(robot, sim_manager=None):
    """安全的运行时可视化类型切换示例"""

    # 使用默认配置
    analyzer = WorkspaceAnalyzer(robot=robot, sim_manager=sim_manager)

    # 先进行分析
    results = analyzer.analyze(num_samples=1000, visualize=False)
    print(f"分析完成，找到 {len(analyzer.workspace_points)} 个工作空间点")

    # 现在可以安全地尝试不同的可视化类型
    visualization_types = [
        VisualizationType.POINT_CLOUD,
        VisualizationType.VOXEL,
        VisualizationType.SPHERE,
    ]

    for viz_type in visualization_types:
        try:
            print(f"尝试 {viz_type.value} 可视化...")
            analyzer.visualize(vis_type=viz_type, show=True)
            print(f"✓ {viz_type.value} 可视化成功")
        except Exception as e:
            print(f"✗ {viz_type.value} 可视化失败: {e}")

    return results


if __name__ == "__main__":
    print("工作空间分析器可视化配置示例")
    print("=" * 50)

    print("可用的可视化类型:")
    for viz_type in VisualizationType:
        print(f"  - {viz_type.name}: '{viz_type.value}'")

    print("\n配置参数说明:")
    print("点云可视化 (POINT_CLOUD):")
    print("  - point_size: 点的大小")

    print("\n体素可视化 (VOXEL):")
    print("  - voxel_size: 体素大小")

    print("\n球体可视化 (SPHERE):")
    print("  - sphere_radius: 球体半径")
    print("  - sphere_resolution: 球体网格分辨率")

    print("\n通用参数:")
    print("  - alpha: 透明度")
    print("  - sim_manager, control_part_name: 仿真相关")
    print("\n注意: 坐标轴显示功能已移除，如需要请单独实现")

    # 注意：需要提供实际的 robot 和 sim_manager 实例来运行示例
    # example_complete_configuration(robot, sim_manager)
    # example_runtime_safe_switching(robot, sim_manager)
