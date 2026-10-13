# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Session-local English and Chinese copy for the analysis workbench."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DEFAULT_LOCALE",
    "LANGUAGE_CHOICES",
    "availability_choices",
    "dimension_choices",
    "dimension_label",
    "joint_choices",
    "normalize_locale",
    "status_choices",
    "text",
]

DEFAULT_LOCALE = "en"
LANGUAGE_CHOICES = [("English", "en"), ("中文", "zh")]

_DIMENSIONS = {
    "asset": {"en": "Object asset", "zh": "物体资产"},
    "pose_x": {"en": "Initial X", "zh": "初始 X"},
    "pose_y": {"en": "Initial Y", "zh": "初始 Y"},
    "pose_z": {"en": "Initial Z", "zh": "初始 Z"},
    "yaw": {"en": "Initial yaw", "zh": "初始偏航角"},
    "affordance": {"en": "Affordance", "zh": "Affordance"},
    "approach": {"en": "Approach / direction annotation", "zh": "接近方式 / 方向标注"},
    "trajectory_family": {"en": "Trajectory geometry family", "zh": "轨迹几何族"},
    "duration_s": {"en": "Trajectory duration", "zh": "轨迹时长"},
    "material": {"en": "Material", "zh": "材质"},
    "light": {"en": "Light intensity", "zh": "光照强度"},
}

_TEXT = {
    "en": {
        "app_title": "EmbodiChain · Data Diversity Workbench",
        "workbench_title": "Data Diversity Analysis Workbench",
        "header": "# Data Diversity Analysis Workbench\n**EmbodiChain** · Simulation augmentation preview · Gradio + Viser",
        "intro": "Acceptance task: **Franka repeated pick-and-place**. From augmentation conditions → data distributions → measured episode evidence → 3D replay.",
        "language": "Language",
        "filters": "Diversity filters (empty multi-select means all; numeric bounds are inclusive)",
        "ranges": "Pose, lighting, and duration ranges",
        "minimum": "minimum",
        "maximum": "maximum",
        "any_dimension": "Any dimension",
        "availability_dimension": "Data completeness: dimension",
        "availability_state": "Data completeness: state",
        "collection_status": "Collection status",
        "apply_filters": "Apply filters / Refresh",
        "all": "All",
        "known": "Known",
        "explicit_unknown": "Explicitly unknown",
        "missing": "Missing field",
        "invalid": "Incompatible type or unit",
        "all_statuses": "All statuses",
        "slice_count": "Current slice: {count} episodes",
        "slice_drilled": " (joint cell selected; apply filters to return to the conditional slice)",
        "tab_distribution": "Distribution & coverage",
        "definitions": "Dimension definitions, sources, and completeness",
        "dimension": "Dimension",
        "dimension_family": "Family",
        "unit_type": "Unit / type",
        "definition_version": "Definition version",
        "completeness_rate": "Completeness %",
        "source_counts": "Source counts",
        "dimension_distribution": "Dimension distribution",
        "dataset_overview": "Dataset overview (asset / material / position)",
        "slice_distribution": "Current slice distribution",
        "distribution_note": "Positions are initial measured values; assets, materials, and lighting come from applied configuration. Joint plots show observed counts. **No coverage rate is claimed without a target grid.**",
        "joint_statistics": "Joint statistics",
        "joint_asset_pose": "Asset × initial X bin",
        "joint_affordance_approach": "Affordance × approach / direction annotation",
        "joint_material_light": "Material × light",
        "drill_note": "Select a joint-cell row below to drill into its episodes; charts, table, and export update together. Approach annotations are preserved without inferring unrecorded geometry.",
        "joint_cells": "Joint cells (select a row to drill down)",
        "dimension_a": "Dimension A",
        "dimension_b": "Dimension B",
        "episode_count": "Episode count",
        "current_slice": "Current data slice",
        "status": "Status",
        "asset": "Asset",
        "material": "Material",
        "light": "Light",
        "initial_x_m": "Initial X / m",
        "initial_y_m": "Initial Y / m",
        "lift_m": "Lift / m",
        "placement_error_m": "Placement error / m",
        "segment_count": "Segments",
        "export_slice": "Export current slice references",
        "slice_json": "Slice JSON",
        "tab_replay": "Trajectory & scene replay",
        "comparison_episode": "Comparison episode (trajectory overlay)",
        "no_comparison": "None",
        "open_replay": "Open / Update replay",
        "replay_placeholder": "Select an episode to open Viser. Scene, trajectory, and camera share one timeline.",
        "phase_section": "Phase-aligned TCP / joint / speed curves",
        "phase": "Task phase (matched by name and occurrence)",
        "analyze": "Analyze trajectory / Compare curves",
        "trajectory_note": "The horizontal axis uses normalized phase time; speed still uses recorded time. Geometric distance compares paths by arc length, so retiming does not create a new path.",
        "trajectory_chart": "Measured trajectory curves",
        "trajectory_metrics": "Geometry and timing metrics",
        "record_evidence": "Augmentation parameters, provenance, and measured metrics",
        "tab_versions": "Dataset versions",
        "version_note": "Snapshots freeze record contents at creation time; later collection does not alter old versions. Comparisons include committed data.",
        "snapshot_name": "Snapshot name",
        "save_snapshot": "Save catalog snapshot",
        "baseline_snapshot": "Baseline snapshot ID",
        "comparison_snapshot": "Comparison snapshot ID",
        "compare_versions": "Compare versions",
        "version_diff": "Version difference",
        "tab_scope": "Scope & definitions",
        "scope": """### Connected in this preview
- SQLite collection ledger, idempotent writes, status and missing-value validation.
- Recording and inspection of assets, poses, materials, lighting, Affordance, and trajectory length/duration.
- Six-family multi-select/range/availability filters, completeness and source statistics, three joint views, and cell drilldown.
- Task-phase alignment, TCP/joint/speed curves, and separate geometry/timing metrics.
- Slice-reference export and immutable snapshots.
- Offline Viser scene and trajectory replay without rerunning simulation.

### Current limitations
- The acceptance collector is currently bound to the Franka repeated pick-and-place task; historical LeRobot metadata can be imported.
- Affordance comes from task annotation; policy-reported success is stored separately from physical object-motion checks.
- Geometry-family clustering, augmentation-recipe feedback, and model-performance attribution are not implemented; trajectory family remains unknown.
- The physical check validates lift and final placement, not per-phase contact stability.
""",
        "total_attempts": "All attempts",
        "committed": "Committed",
        "filtered": "Current filter",
        "physical_passed": "Physical checks passed",
        "unknown_dimensions": "Has unknown dimensions",
        "ledger": "Collection ledger",
        "category": "Category",
        "unknown": "Unknown",
        "incompatible": "Incompatible",
        "episodes": "Episodes",
        "asset_distribution": "Asset distribution",
        "material_distribution": "Material distribution",
        "initial_position": "Initial object position · recorded",
        "world_x": "World X (m)",
        "world_y": "World Y (m)",
        "trajectory_duration": "Trajectory duration",
        "recorded_time": "Recorded time (s)",
        "unknown_asset": "Unknown / incompatible asset",
        "invalid_cell": "Invalid joint cell.",
        "episode_not_selected": "No episode selected.",
        "record_missing": "Record not found.",
        "physical_check": "Physical object-motion check",
        "passed": "Passed",
        "not_passed": "Not passed / not measured",
        "program_success": "Program-reported success",
        "no_physical_evidence": "Historical record has no physical-check evidence",
        "open_episode_first": "Select an episode first.",
        "no_replay": "This record has no replayable trajectory.",
        "open_viser": "Open Viser replay separately",
        "snapshot_saved": "Saved immutable snapshot #{identifier}: {name}",
    },
    "zh": {
        "app_title": "EmbodiChain · 数据多样性工作台",
        "workbench_title": "数据多样性分析工作台",
        "header": "# 数据多样性分析工作台\n**EmbodiChain** · 仿真扩增预览 · Gradio + Viser",
        "intro": "验收任务：**Franka 重复抓放**。从扩增条件 → 数据分布 → episode 实测证据 → 3D 回放。",
        "language": "语言",
        "filters": "多样性筛选（多选为空表示全部；数值边界包含端点）",
        "ranges": "姿态、光照与时长范围",
        "minimum": "最小值",
        "maximum": "最大值",
        "any_dimension": "不限维度",
        "availability_dimension": "数据完整性：维度",
        "availability_state": "数据完整性：状态",
        "collection_status": "采集状态",
        "apply_filters": "应用筛选 / 刷新",
        "all": "不限",
        "known": "已知",
        "explicit_unknown": "明确未知",
        "missing": "字段缺失",
        "invalid": "类型或单位不兼容",
        "all_statuses": "全部状态",
        "slice_count": "当前切片：{count} 条",
        "slice_drilled": "（已下钻联合单元格；点击应用筛选可回到条件筛选结果）",
        "tab_distribution": "分布与覆盖",
        "definitions": "维度定义、来源与完整率",
        "dimension": "维度",
        "dimension_family": "维度族",
        "unit_type": "单位 / 类型",
        "definition_version": "定义版本",
        "completeness_rate": "完整率 %",
        "source_counts": "来源计数",
        "dimension_distribution": "查看维度分布",
        "dataset_overview": "数据集概览（资产 / 材质 / 位置）",
        "slice_distribution": "当前切片分布",
        "distribution_note": "位置使用初始实测值；资产、材质和光照来自已应用配置。联合图显示观察到的计数，**未设置目标网格时不宣称覆盖率**。",
        "joint_statistics": "联合统计",
        "joint_asset_pose": "资产 × 初始 X 分箱",
        "joint_affordance_approach": "Affordance × 接近方式 / 方向标注",
        "joint_material_light": "材质 × 光照",
        "drill_note": "点击下面的联合单元格行，下钻到对应 episode；图表、列表、导出同步更新。接近方式保留原标注，未采集几何方向时不推断。",
        "joint_cells": "联合单元格（点击下钻）",
        "dimension_a": "维度 A",
        "dimension_b": "维度 B",
        "episode_count": "Episode 数量",
        "current_slice": "当前数据切片",
        "status": "状态",
        "asset": "资产",
        "material": "材质",
        "light": "光照",
        "initial_x_m": "初始 X / m",
        "initial_y_m": "初始 Y / m",
        "lift_m": "抬升 / m",
        "placement_error_m": "落点误差 / m",
        "segment_count": "片段数",
        "export_slice": "导出当前切片引用",
        "slice_json": "切片 JSON",
        "tab_replay": "轨迹与场景回放",
        "comparison_episode": "对比 Episode（轨迹叠加）",
        "no_comparison": "无",
        "open_replay": "打开 / 更新回放",
        "replay_placeholder": "选择 episode 后打开 Viser。场景、轨迹和相机共用一条时间轴。",
        "phase_section": "阶段对齐与 TCP / 关节 / 速度曲线",
        "phase": "任务片段（按名称及出现次数匹配）",
        "analyze": "分析轨迹 / 对比曲线",
        "trajectory_note": "横轴使用所选片段的归一化时间；速度仍按原始记录时间计算。几何差异按路径弧长比较，不把变速当作新路径。",
        "trajectory_chart": "实测轨迹曲线",
        "trajectory_metrics": "几何与时间指标",
        "record_evidence": "扩增参数、数据血缘与实测指标",
        "tab_versions": "数据版本",
        "version_note": "快照固定当时的记录内容；后续采集不会改变旧版本。比较范围为已提交数据。",
        "snapshot_name": "快照名称",
        "save_snapshot": "保存当前目录快照",
        "baseline_snapshot": "基线快照 ID",
        "comparison_snapshot": "对比快照 ID",
        "compare_versions": "比较版本",
        "version_diff": "版本差异",
        "tab_scope": "范围与口径",
        "scope": """### 本版已接通
- SQLite 采集账本、幂等写入、状态和缺失值校验。
- 资产 / 姿态 / 材质 / 光照 / Affordance / 轨迹长度与时长的记录与查看。
- 六维多选/范围/未知值筛选，完整率与来源统计，三类联合统计及单元格下钻。
- 任务片段对齐、TCP/关节/速度曲线，路径与时间指标分离。
- 切片引用导出、不可变快照。
- 离线 Viser 场景和轨迹回放，不重新运行仿真。

### 当前限制
- 验收采集器目前绑定 Franka 重复抓放任务；历史 LeRobot 可通过元数据导入器接入。
- Affordance 来自任务标注；执行策略报告的成功与物体运动检查分开记录。
- 几何族自动聚类、扩增配方闭环、模型性能归因尚未完成；轨迹族保持未知。
- 物理检查验证抬升和最终落点，不等价于逐阶段接触稳定性验证。
""",
        "total_attempts": "全部尝试",
        "committed": "已提交",
        "filtered": "当前筛选",
        "physical_passed": "实测检查通过",
        "unknown_dimensions": "包含未知维度",
        "ledger": "采集账本",
        "category": "分类",
        "unknown": "未知",
        "incompatible": "不兼容",
        "episodes": "Episodes",
        "asset_distribution": "资产分布",
        "material_distribution": "材质分布",
        "initial_position": "初始物体位置 · 实测",
        "world_x": "世界 X（m）",
        "world_y": "世界 Y（m）",
        "trajectory_duration": "轨迹时长",
        "recorded_time": "记录时间（s）",
        "unknown_asset": "未知 / 不兼容资产",
        "invalid_cell": "无效的联合单元格。",
        "episode_not_selected": "未选择 episode。",
        "record_missing": "记录不存在。",
        "physical_check": "实际物体运动检查",
        "passed": "通过",
        "not_passed": "未通过 / 未测量",
        "program_success": "程序报告成功",
        "no_physical_evidence": "历史数据无物理检查证据",
        "open_episode_first": "请先选择 episode。",
        "no_replay": "该记录没有可回放的轨迹。",
        "open_viser": "独立打开 Viser 回放",
        "snapshot_saved": "已保存不可变快照 #{identifier}：{name}",
    },
}


def normalize_locale(locale: str | None) -> str:
    """Return a supported locale, falling back to English."""
    return locale if locale in _TEXT else DEFAULT_LOCALE


def text(key: str, locale: str | None = None, **values: Any) -> str:
    """Return formatted UI copy for a locale."""
    selected = normalize_locale(locale)
    return _TEXT[selected][key].format(**values)


def dimension_label(key: str, locale: str | None = None) -> str:
    """Return the localized label for one stable dimension key."""
    return _DIMENSIONS[key][normalize_locale(locale)]


def dimension_choices(locale: str | None = None) -> list[tuple[str, str]]:
    """Return localized display labels with stable dimension values."""
    return [(dimension_label(key, locale), key) for key in _DIMENSIONS]


def availability_choices(locale: str | None = None) -> list[tuple[str, str]]:
    """Return localized completeness choices with stable values."""
    return [
        (text("all", locale), "all"),
        (text("known", locale), "known"),
        (text("explicit_unknown", locale), "unknown"),
        (text("missing", locale), "missing"),
        (text("invalid", locale), "invalid"),
    ]


def status_choices(locale: str | None = None) -> list[tuple[str, str]]:
    """Return localized collection status labels with stable values."""
    return [
        ("committed", "committed"),
        (text("all_statuses", locale), "all"),
        ("rejected", "rejected"),
        ("rollout_failed", "rollout_failed"),
        ("planning_failed", "planning_failed"),
        ("partial_commit", "partial_commit"),
    ]


def joint_choices(locale: str | None = None) -> list[tuple[str, str]]:
    """Return localized joint-analysis labels with stable values."""
    return [
        (text("joint_asset_pose", locale), "asset_pose"),
        (text("joint_affordance_approach", locale), "affordance_approach"),
        (text("joint_material_light", locale), "material_light"),
    ]
