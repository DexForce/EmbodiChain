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

"""Build an interactive, CPU-only laboratory for trajectory augmentation.

The generated HTML compares every implemented pure-trajectory method on the
same planar three-link reference.  All candidates are produced by the public
motion-expansion APIs; the browser owns only playback and drawing.  No image,
affordance proposal, contact-point sampling, simulator, or task environment is
involved.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import webbrowser

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import torch

from embodichain.compute.kinematics import DifferentiableIKCfg
from embodichain.lab.sim.motion.expansion import (
    ManipulabilityBands,
    MinimumJerkBridgePlanner,
    TaskSpaceDeformationConstraint,
    TrajectoryPhase,
    TrajectorySegmentBank,
    TrajectoryTemplate,
    deform_task_space_trajectory,
    describe_manipulability,
    extract_trajectory_segments,
    joint_residual,
    manipulability_guided_residual,
    nullspace_residual,
    perturb_approach_direction,
    project_task_space_trajectory,
    retime,
    stitch_trajectory_segments,
    task_space_from_joint_template,
    via_points,
)

__all__ = ["main"]

_DT = 0.05
_LINK_LENGTHS = torch.tensor((0.34, 0.28, 0.22), dtype=torch.float64)
_JOINT_LIMITS = torch.tensor(
    ((-math.pi, math.pi),) * 3,
    dtype=torch.float64,
)
_FREE_PHASES = ("approach", "transport", "retreat")
_OPERATORS = ("joint_residual", "via_points", "nullspace_residual", "retime")
_SEEDS = tuple(range(9))
_STRENGTHS = (0.02, 0.04, 0.07)
_TASK_SPACE_STRENGTHS = (0.025, 0.05, 0.08)
_APPROACH_ANGLES = (0.12, 0.24, 0.36)
_STITCH_POSTURE_OFFSETS = (0.12, 0.20, 0.30)
_TEMPLATE_PATH = Path(__file__).with_name("trajectory_augmentation_lab_template.html")


def _smooth_segment(
    start: torch.Tensor,
    goal: torch.Tensor,
    sample_count: int,
) -> torch.Tensor:
    progress = torch.linspace(0, 1, sample_count, dtype=torch.float64)
    smooth = progress.square() * (3 - 2 * progress)
    return torch.lerp(start, goal, smooth[:, None])


def _template_from_waypoints(
    source_id: str,
    waypoints: tuple[torch.Tensor, ...],
) -> TrajectoryTemplate:
    lengths = (21, 11, 31, 11, 31)
    parts = tuple(
        _smooth_segment(start, goal, length)
        for start, goal, length in zip(
            waypoints[:-1], waypoints[1:], lengths, strict=True
        )
    )
    positions = torch.cat((parts[0], *(part[1:] for part in parts[1:])))
    ranges = []
    start_index = 0
    for length in lengths:
        stop_index = start_index + length
        ranges.append((start_index, stop_index))
        start_index = stop_index - 1
    phases = tuple(
        TrajectoryPhase(
            phase_id,
            start,
            stop,
            kind,
            _OPERATORS if kind == "free" else (),
        )
        for phase_id, (start, stop), kind in zip(
            ("approach", "grasp", "transport", "release", "retreat"),
            ranges,
            ("free", "contact", "free", "contact", "free"),
            strict=True,
        )
    )
    dt = torch.full((positions.shape[0],), _DT, dtype=torch.float64)
    dt[0] = 0
    return TrajectoryTemplate(
        source_id=source_id,
        source_revision="one",
        template_id=f"{source_id}_pick_place",
        joint_names=("shoulder", "elbow", "wrist"),
        positions=positions,
        dt=dt,
        phases=phases,
        allowed_operators=_OPERATORS,
        validator_id="visualization_only",
        controlled_joint_indices=(0, 1, 2),
    )


def _reference_template() -> TrajectoryTemplate:
    values = (
        (-0.90, 1.40, 0.20),
        (-0.45, 1.20, -0.25),
        (-0.45, 1.20, -0.25),
        (0.25, 1.05, -0.55),
        (0.25, 1.05, -0.55),
        (0.65, 0.75, -0.30),
    )
    return _template_from_waypoints(
        "reference",
        tuple(torch.tensor(value, dtype=torch.float64) for value in values),
    )


def _planar_kinematics(
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    angles = positions.cumsum(dim=-1)
    cosine = angles.cos()
    sine = angles.sin()
    lengths = _LINK_LENGTHS.to(device=positions.device, dtype=positions.dtype)
    poses = torch.eye(4, dtype=positions.dtype, device=positions.device).repeat(
        positions.shape[0], 1, 1
    )
    poses[:, 0, 0] = cosine[:, -1]
    poses[:, 0, 1] = -sine[:, -1]
    poses[:, 1, 0] = sine[:, -1]
    poses[:, 1, 1] = cosine[:, -1]
    poses[:, 0, 3] = (cosine * lengths).sum(dim=-1)
    poses[:, 1, 3] = (sine * lengths).sum(dim=-1)
    jacobians = torch.zeros(
        positions.shape[0], 6, 3, dtype=positions.dtype, device=positions.device
    )
    for joint_index in range(3):
        jacobians[:, 0, joint_index] = -(
            sine[:, joint_index:] * lengths[joint_index:]
        ).sum(dim=-1)
        jacobians[:, 1, joint_index] = (
            cosine[:, joint_index:] * lengths[joint_index:]
        ).sum(dim=-1)
        jacobians[:, 5, joint_index] = 1
    return poses, jacobians


def _planar_inverse(x: float, y: float, heading: float) -> torch.Tensor:
    first, second, tool = (float(value) for value in _LINK_LENGTHS)
    wrist_x = x - tool * math.cos(heading)
    wrist_y = y - tool * math.sin(heading)
    cosine = (
        wrist_x * wrist_x + wrist_y * wrist_y - first * first - second * second
    ) / (2 * first * second)
    if abs(cosine) > 1 + 1e-8:
        raise RuntimeError("The analytic visualizer requested an unreachable pose.")
    elbow = math.acos(max(-1.0, min(1.0, cosine)))
    shoulder = math.atan2(wrist_y, wrist_x) - math.atan2(
        second * math.sin(elbow), first + second * math.cos(elbow)
    )
    return torch.tensor(
        (shoulder, elbow, heading - shoulder - elbow),
        dtype=torch.float64,
    )


def _tool_points(template: TrajectoryTemplate) -> torch.Tensor:
    return _planar_kinematics(template.positions)[0][:, :2, 3]


def _path_length(points: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum())


def _phase_name(phase_id: str) -> str:
    return phase_id.split(":", 1)[-1]


def _record(
    template: TrajectoryTemplate,
    reference: TrajectoryTemplate,
    *,
    details: dict[str, object] | None = None,
    phase_sources: dict[str, str] | None = None,
    bridge_phase_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    tool_points = _tool_points(template)
    reference_tool = _tool_points(reference)
    jacobians = _planar_kinematics(template.positions)[1][:, :2]
    profile = describe_manipulability(jacobians, phases=template.phases)
    phase_sources = phase_sources or {}
    phases = [
        {
            "id": phase.phase_id,
            "label": _phase_name(phase.phase_id),
            "kind": "bridge" if phase.phase_id in bridge_phase_ids else phase.kind,
            "start": phase.start_index,
            "stop": phase.stop_index,
            "source": phase_sources.get(phase.phase_id, template.source_id),
        }
        for phase in template.phases
    ]
    metrics = {
        "duration_s": round(float(template.dt.sum()), 4),
        "sample_count": template.positions.shape[0],
        "tool_path_ratio": round(
            _path_length(tool_points) / _path_length(reference_tool), 4
        ),
        "minimum_manipulability": round(profile.bottleneck, 6),
    }
    metrics.update(details or {})
    return {
        "q": template.positions.round(decimals=6).tolist(),
        "time": template.dt.cumsum(0).round(decimals=6).tolist(),
        "tool": tool_points.round(decimals=6).tolist(),
        "phases": phases,
        "metrics": metrics,
    }


def _joint_residual_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    return [
        [
            _record(
                joint_residual(
                    reference,
                    joint_limits=_JOINT_LIMITS,
                    normalized_scale=strength,
                    generator=torch.Generator().manual_seed(seed),
                ),
                reference,
                details={"joint_range_fraction": strength},
            )
            for seed in _SEEDS
        ]
        for strength in _STRENGTHS
    ]


def _via_point_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    return [
        [
            _record(
                via_points(
                    reference,
                    joint_limits=_JOINT_LIMITS,
                    via_count=3,
                    normalized_scale=strength,
                    generator=torch.Generator().manual_seed(seed),
                ),
                reference,
                details={"via_points_per_free_phase": 3},
            )
            for seed in _SEEDS
        ]
        for strength in _STRENGTHS
    ]


def _nullspace_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    task_jacobians = _planar_kinematics(reference.positions)[1][:, :2]
    return [
        [
            _record(
                nullspace_residual(
                    reference,
                    task_jacobians=task_jacobians,
                    joint_limits=_JOINT_LIMITS,
                    normalized_scale=strength,
                    generator=torch.Generator().manual_seed(seed),
                ),
                reference,
                details={"constrained_task_rows": "tool x, tool y"},
            )
            for seed in _SEEDS
        ]
        for strength in _STRENGTHS
    ]


def _manipulability_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    reference_jacobians = _planar_kinematics(reference.positions)[1][:, :2]
    reference_score = describe_manipulability(
        reference_jacobians, phases=reference.phases
    ).bottleneck
    bands = ManipulabilityBands((0.85, 1.15), reference=reference_score)
    result = []
    for strength in _STRENGTHS:
        row = []
        for seed in _SEEDS:
            guided = manipulability_guided_residual(
                reference,
                joint_limits=_JOINT_LIMITS,
                normalized_scale=strength,
                generator=torch.Generator().manual_seed(seed),
                jacobian_fn=lambda positions: _planar_kinematics(positions)[1][:, :2],
                bands=bands,
                target_band=seed % bands.count,
                proposals=12,
            )
            row.append(
                _record(
                    guided.template,
                    reference,
                    details={
                        "target_band": seed % bands.count,
                        "reached_band": guided.band,
                        "matched": guided.matched,
                        "proposals_scored": guided.proposals_scored,
                    },
                )
            )
        result.append(row)
    return result


def _task_space_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    task_space = task_space_from_joint_template(
        reference,
        frame_id="planar_tool",
        kinematics=_planar_kinematics,
        editable_phase_ids=_FREE_PHASES,
    )
    ik_cfg = DifferentiableIKCfg(
        max_iterations=40,
        damping=1e-3,
        step_size=0.8,
        pos_eps=2e-5,
        rot_eps=2e-5,
    )
    result = []
    for strength in _TASK_SPACE_STRENGTHS:
        constraints = {
            phase_id: TaskSpaceDeformationConstraint(
                phase_id,
                translation_axes=(True, True, False),
                rotation_axes=(False, False, False),
                max_translation_m=strength,
            )
            for phase_id in _FREE_PHASES
        }
        row = []
        for seed in _SEEDS:
            target = deform_task_space_trajectory(
                task_space,
                constraints=constraints,
                generator=torch.Generator().manual_seed(seed),
            )
            projection = project_task_space_trajectory(
                target,
                reference,
                kinematics=_planar_kinematics,
                joint_limits=_JOINT_LIMITS,
                ik_cfg=ik_cfg,
            )
            if not projection.accepted:
                raise RuntimeError(
                    f"Analytic task-space variant seed {seed} failed trajectory IK."
                )
            row.append(
                _record(
                    projection.template,
                    reference,
                    details={
                        "translation_bound_m": strength,
                        "maximum_ik_error_m": round(
                            float(projection.position_error.norm(dim=-1).max()), 8
                        ),
                    },
                )
            )
        result.append(row)
    return result


def _timing_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    duration_scales = (1.15, 1.40, 1.80)
    profiles = ("uniform", "ease_in", "ease_out")
    return [
        [
            _record(
                retime(
                    reference,
                    duration_scale=duration_scale,
                    control_dt=_DT,
                    profile=profiles[seed % len(profiles)],
                ),
                reference,
                details={
                    "duration_scale": duration_scale,
                    "timing_profile": profiles[seed % len(profiles)],
                },
            )
            for seed in _SEEDS
        ]
        for duration_scale in duration_scales
    ]


def _approach_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    contact_index = reference.phases[0].stop_index - 1
    contact_pose = _planar_kinematics(
        reference.positions[contact_index : contact_index + 1]
    )[0][0]
    result = []
    for maximum_angle in _APPROACH_ANGLES:
        row = []
        for seed in _SEEDS:
            signed_fraction = (seed - (_SEEDS[-1] / 2)) / (_SEEDS[-1] / 2)
            polar_angle = maximum_angle * signed_fraction
            azimuth = math.pi / 2 if seed % 2 == 0 else -math.pi / 2
            standoff = perturb_approach_direction(
                contact_pose,
                approach_axis=torch.tensor((1.0, 0.0, 0.0), dtype=torch.float64),
                standoff_distance=0.18,
                polar_angles=torch.tensor((polar_angle,), dtype=torch.float64),
                azimuth_angles=torch.tensor((azimuth,), dtype=torch.float64),
                align_tool=True,
            )[0]
            heading = math.atan2(float(standoff[1, 0]), float(standoff[0, 0]))
            standoff_qpos = _planar_inverse(
                float(standoff[0, 3]), float(standoff[1, 3]), heading
            )
            positions = reference.positions.clone()
            phase = reference.phases[0]
            positions[phase.start_index : phase.stop_index] = _smooth_segment(
                standoff_qpos,
                positions[contact_index],
                phase.stop_index - phase.start_index,
            )
            candidate = replace(reference, positions=positions)
            row.append(
                _record(
                    candidate,
                    reference,
                    details={
                        "cone_angle_deg": round(math.degrees(polar_angle), 3),
                        "fixed_contact_error_m": 0.0,
                        "requires_replanning": True,
                    },
                )
            )
        result.append(row)
    return result


def _posture_shifted_template(
    reference: TrajectoryTemplate,
    source_id: str,
    heading_offset: float,
) -> TrajectoryTemplate:
    indices = [phase.start_index for phase in reference.phases]
    indices.append(reference.phases[-1].stop_index - 1)
    poses = _planar_kinematics(reference.positions[indices])[0]
    waypoints = []
    for pose in poses:
        heading = math.atan2(float(pose[1, 0]), float(pose[0, 0]))
        waypoints.append(
            _planar_inverse(
                float(pose[0, 3]),
                float(pose[1, 3]),
                heading + heading_offset,
            )
        )
    return _template_from_waypoints(source_id, tuple(waypoints))


def _stitching_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    result = []
    for offset in _STITCH_POSTURE_OFFSETS:
        sources = (
            reference,
            _posture_shifted_template(reference, "demo_B", offset),
            _posture_shifted_template(reference, "demo_C", -offset),
        )
        segments = {
            source.source_id: {
                segment.phase_id: segment
                for segment in extract_trajectory_segments(source)
            }
            for source in sources
        }
        bank = TrajectorySegmentBank(
            tuple(segments["demo_B"].values()) + tuple(segments["demo_C"].values())
        )
        bridge_planner = MinimumJerkBridgePlanner(
            control_dt=_DT,
            velocity_limits=torch.tensor((1.8, 1.8, 1.8), dtype=torch.float64),
            acceleration_limits=torch.tensor((6.0, 6.0, 6.0), dtype=torch.float64),
            minimum_sample_count=5,
        )
        row = []
        for seed in _SEEDS:
            transport_ordinal = seed % 2
            retreat_ordinal = (seed + 1) % 2
            reference_segments = segments["reference"]
            selected = (
                reference_segments["approach"],
                reference_segments["grasp"],
                bank.select(
                    reference_segments["transport"],
                    strategy="fifo",
                    ordinal=transport_ordinal,
                ),
                bank.select(
                    reference_segments["release"],
                    strategy="fifo",
                    ordinal=transport_ordinal,
                ),
                bank.select(
                    reference_segments["retreat"],
                    strategy="fifo",
                    ordinal=retreat_ordinal,
                ),
            )
            stitched = stitch_trajectory_segments(
                selected,
                source_id="stitched",
                source_revision=f"offset:{offset}",
                template_id=f"seed:{seed}",
                bridge_planner=bridge_planner,
            )
            source_by_phase = {
                placement.output_phase_id: placement.source_id
                for placement in stitched.placements
            }
            row.append(
                _record(
                    stitched.template,
                    reference,
                    details={
                        "bank_size": len(bank),
                        "bridge_count": len(stitched.bridge_phase_ids),
                        "selected_sources": [segment.source_id for segment in selected],
                    },
                    phase_sources=source_by_phase,
                    bridge_phase_ids=stitched.bridge_phase_ids,
                )
            )
        result.append(row)
    return result


def _method(
    *,
    label: str,
    badge: str,
    equation: str,
    principle: str,
    implementation: str,
    guarantee: str,
    strength_labels: tuple[str, str, str],
    variants: list[list[dict[str, object]]],
) -> dict[str, object]:
    return {
        "label": label,
        "badge": badge,
        "equation": equation,
        "principle": principle,
        "implementation": implementation,
        "guarantee": guarantee,
        "strength_labels": strength_labels,
        "variants": variants,
    }


def _payload() -> dict[str, object]:
    reference = _reference_template()
    reference_record = _record(reference, reference)
    methods = {
        "joint_residual": _method(
            label="关节残差",
            badge="qpos",
            equation="参考关节轨迹 → 端点包络残差 → 正运动学 FK → 末端路径",
            principle="在已授权自由阶段内叠加零值、零导数的平滑残差。不同随机种子改变关节方向和幅度。",
            implementation="joint_residual(template, joint_limits, normalized_scale, generator)",
            guarantee="阶段端点与 contact/hold 样本保持不变；碰撞、动力学和任务成功仍由宿主验证。",
            strength_labels=("2% 关节范围", "4% 关节范围", "7% 关节范围"),
            variants=_joint_residual_variants(reference),
        ),
        "via_points": _method(
            label="途经点",
            badge="qpos",
            equation="参考轨迹 → 每自由段 3 个内部结点 → Clamp Hermite → 末端路径",
            principle="采样内部结点并用端点导数为零的三次 Hermite 曲线连接，可改变路径形状而不仅是整体鼓包。",
            implementation="via_points(template, via_count=3, normalized_scale, generator)",
            guarantee="所有标注路点保持精确；只修改显式授权的受控关节。",
            strength_labels=("2% 结点偏移", "4% 结点偏移", "7% 结点偏移"),
            variants=_via_point_variants(reference),
        ),
        "nullspace": _method(
            label="零空间",
            badge="Jacobian",
            equation="关节残差 → J 的零空间投影 → 同一 XY 任务 → 不同机械臂姿态",
            principle="把残差投影到平面末端 XY Jacobian 的零空间，改变冗余姿态，同时一阶保持任务行。",
            implementation="nullspace_residual(template, task_jacobians, ...)",
            guarantee="端点严格不变，内部任务约束仅一阶保持，必须用 FK 和物理执行再次验证。",
            strength_labels=("2% 投影前幅度", "4% 投影前幅度", "7% 投影前幅度"),
            variants=_nullspace_variants(reference),
        ),
        "manipulability": _method(
            label="可操作度引导",
            badge="guided",
            equation="多次关节残差候选 → Yoshikawa 评分 → 目标区间 → 保留最近候选",
            principle="不均匀盲采样，而是按参考轨迹归一化的可操作度区间选择候选，从而覆盖不同姿态条件。",
            implementation="manipulability_guided_residual(..., bands, target_band, proposals=12)",
            guarantee="可操作度只用于提议和覆盖，不构成可行性或安全证明。",
            strength_labels=("2% 候选幅度", "4% 候选幅度", "7% 候选幅度"),
            variants=_manipulability_variants(reference),
        ),
        "task_space": _method(
            label="任务空间",
            badge="trajectory IK",
            equation="参考 TCP 轨迹 → 有界 XY 形变 → 批量 DLS 轨迹 IK → qpos",
            principle="直接改变末端路径的内部样本，再投影回完整关节轨迹；锁定样本沿用参考 qpos。",
            implementation="deform_task_space_trajectory() + project_task_space_trajectory()",
            guarantee="只编辑授权自由阶段，阶段端点精确；结果携带逐样本 IK 成功和位姿残差。",
            strength_labels=("2.5 cm", "5 cm", "8 cm"),
            variants=_task_space_variants(reference),
        ),
        "timing": _method(
            label="时间重参数化",
            badge="retime",
            equation="同一几何路径 → 时长缩放 + ease profile → 控制时钟重采样",
            principle="保持空间路径不变，改变自由阶段的速度分布和总时长；种子在 uniform/ease-in/ease-out 间切换。",
            implementation="retime(template, duration_scale, control_dt, profile)",
            guarantee="contact/hold 时长和阶段端点不变；路径视图重合，关节曲线视图显示时间差异。",
            strength_labels=("1.15× 时长", "1.40× 时长", "1.80× 时长"),
            variants=_timing_variants(reference),
        ),
        "approach": _method(
            label="接近方向",
            badge="fixed contact",
            equation="固定接触位姿 → 锥面采样 standoff → 解析 IK → 重新生成接近段",
            principle="只改变接触前的接近方向和预接触姿态，接触位姿本身不变；这不是接触点或 affordance 采样。",
            implementation="perturb_approach_direction() + showcase-owned planar replanning",
            guarantee="候选 standoff 仍需可达性、净空和路径规划验证；页面中的解析 IK 只服务于这个平面示例。",
            strength_labels=("±6.9° 锥角", "±13.8° 锥角", "±20.6° 锥角"),
            variants=_approach_variants(reference),
        ),
        "stitching": _method(
            label="分段拼接",
            badge="multi-demo",
            equation="同任务示教库 → 按语义阶段选段 → 不连续边界显式桥接 → 新轨迹",
            principle="从多个同任务示教中按 approach/grasp/transport/release/retreat 重组，保留来源谱系。",
            implementation="TrajectorySegmentBank + stitch_trajectory_segments(..., bridge_planner)",
            guarantee="兼容性按阶段语义和关节布局检查；最小 jerk 桥只保证采样连续，不保证碰撞安全。",
            strength_labels=(
                "±0.12 rad 姿态库",
                "±0.20 rad 姿态库",
                "±0.30 rad 姿态库",
            ),
            variants=_stitching_variants(reference),
        ),
    }
    return {
        "title": "轨迹扩增实验室",
        "subtitle": "同一条参考轨迹 · 平面三关节机械臂 · 生产 API 原理演示",
        "link_lengths": _LINK_LENGTHS.tolist(),
        "seed_count": len(_SEEDS),
        "reference": reference_record,
        "methods": methods,
        "scope": {
            "included": "关节、任务空间、时序、接近方向与多示教分段重组",
            "excluded": "图像、affordance、接触点/抓取位姿采样和场景随机化",
        },
    }


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an interactive pure-trajectory augmentation lab."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPOSITORY_ROOT
        / "output"
        / "trajectory_augmentation_lab"
        / "index.html",
        help="Self-contained HTML output path.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default browser.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Generate the interactive pure-trajectory augmentation laboratory.

    Args:
        argv: Optional command-line arguments for programmatic invocation.
    """
    args = _create_parser().parse_args(argv)
    if not _TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Missing HTML template: {_TEMPLATE_PATH}")
    payload = json.dumps(
        _payload(), ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    marker = "__TRAJECTORY_AUGMENTATION_DATA__"
    if html.count(marker) != 1:
        raise RuntimeError("The HTML template must contain exactly one data marker.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html.replace(marker, payload), encoding="utf-8")
    print(f"trajectory augmentation lab: {args.output}")
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
