#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例：Cartesian space 模式下可视化所有点（可达和不可达）

展示如何使用修改后的 WorkspaceAnalyzer 在 Cartesian space 模式下
可视化所有采样点，其中：
- 可达的点：绿色，较大的size
- 不可达的点：红色，较小的size
"""

import numpy as np
import torch
from embodichain.lab.sim.utility.workspace_analyzer import (
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
    AnalysisMode,
)
from embodichain.lab.sim.utility.workspace_analyzer.configs import (
    VisualizationConfig,
    SamplingConfig,
    DimensionConstraint,
)
from embodichain.lab.sim.utility.workspace_analyzer.visualizers import VisualizationType


def example_cartesian_space_reachability_visualization(
    robot, sim_manager=None, show_unreachable=True
):
    """
    Cartesian space 模式下可视化所有点的示例

    Args:
        robot: 机器人实例
        sim_manager: 仿真管理器
        show_unreachable: 是否显示不可达的点位

    Returns:
        分析结果字典
    """

    # 配置可视化 - 使用较大的点来突出差异
    viz_config = VisualizationConfig(
        enabled=True,
        vis_type=VisualizationType.POINT_CLOUD,
        point_size=8.0,  # 基础点大小（不可达点会是这个的0.7倍，可达点会是这个的1.2倍）
        alpha=0.8,
        color_by_distance=False,  # 不按距离着色，而是按可达性着色
        show_unreachable_points=show_unreachable,  # 控制是否显示不可达点
    )

    # 配置采样参数
    sampling_config = SamplingConfig(
        num_samples=2000,  # 增加采样数量以更好地看到效果
        batch_size=128,
        seed=42,
    )

    # 配置维度约束（可选：限制采样空间）
    constraint_config = DimensionConstraint(
        min_bounds=[-0.5, -0.5, 0.2],  # 工作空间下界 [x, y, z]
        max_bounds=[0.8, 0.5, 1.0],  # 工作空间上界 [x, y, z]
    )

    # 创建完整的配置
    config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.CARTESIAN_SPACE,  # 使用 Cartesian space 模式
        sampling=sampling_config,
        constraint=constraint_config,
        visualization=viz_config,
        ik_success_threshold=0.8,  # IK成功阈值，低于此值的点被视为不可达
        ik_samples_per_point=3,  # 每个Cartesian点尝试的IK种子数量
    )

    # 创建工作空间分析器
    analyzer = WorkspaceAnalyzer(
        robot=robot,
        config=config,
        sim_manager=sim_manager,
        control_part_name="left_arm",  # 指定要分析的控制部分
    )

    print("开始 Cartesian space 工作空间分析...")
    print(f"将采样 {sampling_config.num_samples} 个Cartesian点")
    print(f"IK成功阈值: {config.ik_success_threshold}")
    print(f"每个点尝试 {config.ik_samples_per_point} 个IK种子")

    # 执行分析并可视化
    results = analyzer.analyze(num_samples=None, force_recompute=True, visualize=True)

    # 打印结果统计
    print("\n=== 分析结果 ===")
    print(f"采样的Cartesian点总数: {results['num_samples']}")
    print(f"可达的点数量: {results['num_reachable']}")
    print(f"不可达的点数量: {results['num_samples'] - results['num_reachable']}")
    print(f"可达性比例: {results['num_reachable'] / results['num_samples'] * 100:.1f}%")
    print(f"分析耗时: {results['analysis_time']:.2f}秒")

    # 可视化说明
    print("\n=== 可视化说明 ===")
    if show_unreachable:
        print("🟢 绿色大点：可达的Cartesian位置")
        print("🔴 红色小点：不可达的Cartesian位置")
        print("点的大小差异体现了可达性")
    else:
        print("🟢 绿色大点：仅显示可达的Cartesian位置")
        print("不可达的点位已被隐藏")

    return results


def example_compare_joint_vs_cartesian_visualization(robot, sim_manager=None):
    """
    对比 Joint space 和 Cartesian space 两种模式的可视化效果

    Args:
        robot: 机器人实例
        sim_manager: 仿真管理器

    Returns:
        包含两种分析结果的字典
    """

    # 通用配置
    viz_config = VisualizationConfig(
        enabled=True,
        vis_type=VisualizationType.POINT_CLOUD,
        point_size=6.0,
        alpha=0.7,
    )

    sampling_config = SamplingConfig(
        num_samples=1500,
        seed=42,
    )

    results = {}

    # 1. Joint space 分析
    print("=== Joint Space 分析 ===")
    joint_config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.JOINT_SPACE,
        sampling=sampling_config,
        visualization=viz_config,
    )

    joint_analyzer = WorkspaceAnalyzer(
        robot=robot,
        config=joint_config,
        sim_manager=sim_manager,
        control_part_name="left_arm",
    )

    joint_results = joint_analyzer.analyze(visualize=False)
    print(f"Joint space: {joint_results['num_valid']} 个有效点")

    # 2. Cartesian space 分析
    print("\n=== Cartesian Space 分析 ===")
    cartesian_config = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.CARTESIAN_SPACE,
        sampling=sampling_config,
        visualization=viz_config,
        ik_success_threshold=0.7,
        ik_samples_per_point=2,
    )

    cartesian_analyzer = WorkspaceAnalyzer(
        robot=robot,
        config=cartesian_config,
        sim_manager=sim_manager,
        control_part_name="left_arm",
    )

    cartesian_results = cartesian_analyzer.analyze(visualize=False)
    print(
        f"Cartesian space: {cartesian_results['num_reachable']}/{cartesian_results['num_samples']} 个可达点"
    )

    # 3. 分别可视化
    print("\n可视化 Joint space 结果...")
    joint_analyzer.visualize(show=True, save_path="joint_space_workspace.png")

    print("可视化 Cartesian space 结果...")
    cartesian_analyzer.visualize(show=True, save_path="cartesian_space_workspace.png")

    results["joint_space"] = joint_results
    results["cartesian_space"] = cartesian_results

    return results


def example_show_hide_unreachable_points(robot, sim_manager=None):
    """
    演示显示/隐藏不可达点位功能的示例

    Args:
        robot: 机器人实例
        sim_manager: 仿真管理器

    Returns:
        包含两种配置结果的字典
    """

    print("=== 演示显示/隐藏不可达点位功能 ===")

    sampling_config = SamplingConfig(
        num_samples=1500,
        seed=42,
    )

    results = {}

    # 1. 显示所有点（可达和不可达）
    print("\n--- 配置1：显示所有点（可达+不可达） ---")
    viz_config_show_all = VisualizationConfig(
        enabled=True,
        vis_type=VisualizationType.POINT_CLOUD,
        point_size=6.0,
        alpha=0.7,
        show_unreachable_points=True,  # 显示不可达点
    )

    config_show_all = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.CARTESIAN_SPACE,
        sampling=sampling_config,
        visualization=viz_config_show_all,
        ik_success_threshold=0.8,
        ik_samples_per_point=2,
    )

    analyzer_show_all = WorkspaceAnalyzer(
        robot=robot,
        config=config_show_all,
        sim_manager=sim_manager,
        control_part_name="left_arm",
    )

    result_show_all = analyzer_show_all.analyze(visualize=False)
    print(f"总采样点: {result_show_all['num_samples']}")
    print(f"可达点: {result_show_all['num_reachable']}")
    print(f"不可达点: {result_show_all['num_samples'] - result_show_all['num_reachable']}")

    # 2. 只显示可达点
    print("\n--- 配置2：只显示可达点 ---")
    viz_config_hide_unreachable = VisualizationConfig(
        enabled=True,
        vis_type=VisualizationType.POINT_CLOUD,
        point_size=6.0,
        alpha=0.7,
        show_unreachable_points=False,  # 隐藏不可达点
    )

    config_hide_unreachable = WorkspaceAnalyzerConfig(
        mode=AnalysisMode.CARTESIAN_SPACE,
        sampling=sampling_config,
        visualization=viz_config_hide_unreachable,
        ik_success_threshold=0.8,
        ik_samples_per_point=2,
    )

    analyzer_hide_unreachable = WorkspaceAnalyzer(
        robot=robot,
        config=config_hide_unreachable,
        sim_manager=sim_manager,
        control_part_name="left_arm",
    )

    result_hide_unreachable = analyzer_hide_unreachable.analyze(visualize=False)
    print(f"总采样点: {result_hide_unreachable['num_samples']}")
    print(f"可达点: {result_hide_unreachable['num_reachable']}")
    print(f"显示的点: 仅可达点（{result_hide_unreachable['num_reachable']}个）")

    # 3. 分别可视化
    print("\n可视化对比...")
    print("1. 显示所有点（绿色可达点 + 红色不可达点）")
    analyzer_show_all.visualize(show=True, save_path="cartesian_show_all_points.png")

    print("2. 只显示可达点（仅绿色点）")
    analyzer_hide_unreachable.visualize(
        show=True, save_path="cartesian_reachable_only.png"
    )

    results["show_all"] = result_show_all
    results["reachable_only"] = result_hide_unreachable

    return results


def example_detailed_reachability_analysis(robot, sim_manager=None):
    """
    详细的可达性分析示例，展示不同IK参数的影响

    Args:
        robot: 机器人实例
        sim_manager: 仿真管理器

    Returns:
        分析结果列表
    """

    # 测试不同的IK成功阈值
    thresholds = [0.5, 0.7, 0.9]
    results = []

    for threshold in thresholds:
        print(f"\n=== 测试IK成功阈值: {threshold} ===")

        viz_config = VisualizationConfig(
            enabled=True,
            vis_type=VisualizationType.POINT_CLOUD,
            point_size=5.0,
            alpha=0.8,
        )

        config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.CARTESIAN_SPACE,
            sampling=SamplingConfig(num_samples=1000, seed=42),
            visualization=viz_config,
            ik_success_threshold=threshold,
            ik_samples_per_point=3,
        )

        analyzer = WorkspaceAnalyzer(
            robot=robot,
            config=config,
            sim_manager=sim_manager,
            control_part_name="left_arm",
        )

        result = analyzer.analyze(visualize=False)

        print(
            f"阈值 {threshold}: {result['num_reachable']}/{result['num_samples']} 个可达点 "
            f"({result['num_reachable'] / result['num_samples'] * 100:.1f}%)"
        )

        # 保存可视化结果
        analyzer.visualize(
            show=False, save_path=f"cartesian_workspace_threshold_{threshold:.1f}.png"
        )

        results.append(result)

    return results


if __name__ == "__main__":
    # 注意：这只是示例代码，需要实际的robot和sim_manager实例
    print("这是 Cartesian space 可视化增强功能的示例代码")
    print("请在实际的仿真环境中运行，传入有效的 robot 和 sim_manager 实例")
    print("\n主要功能:")
    print("1. 在 Cartesian space 模式下可视化所有采样点")
    print("2. 可达点用绿色大点显示")
    print("3. 不可达点用红色小点显示")
    print("4. 支持不同的 IK 成功阈值和参数配置")
