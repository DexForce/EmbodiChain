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

The generated HTML compares every implemented pure-trajectory method on one
seven-DoF Franka Panda reference. The browser renders the robot with the same
serial-chain geometry as the official EmbodiChain preset. All candidates are
produced by the public motion-expansion APIs; the browser owns only playback
and drawing. No image, affordance proposal, contact-point sampling, simulator,
or task environment is involved.
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

from embodichain.compute.kinematics import DifferentiableIKCfg, solve_dls_ik
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
_JOINT_LIMITS = torch.tensor(
    (
        (-2.7437, 2.7437),
        (-1.7837, 1.7837),
        (-2.9007, 2.9007),
        (-3.0421, -0.1518),
        (-2.8065, 2.8065),
        (0.5445, 4.5169),
        (-3.0159, 3.0159),
    ),
    dtype=torch.float64,
)
_FRANKA_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
# Serial chain from the official PandaWithHand URDF used by FrankaPandaCfg.
# Entries are (name, type, origin xyz, origin rpy, axis).
_FRANKA_KINEMATIC_JOINTS = (
    ("fr3_base_joint", "fixed", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ("fr3_joint1", "revolute", (0.0, 0.0, 0.333), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (
        "fr3_joint2",
        "revolute",
        (0.0, 0.0, 0.0),
        (-math.pi / 2, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    (
        "fr3_joint3",
        "revolute",
        (0.0, -0.316, 0.0),
        (math.pi / 2, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    (
        "fr3_joint4",
        "revolute",
        (0.0825, 0.0, 0.0),
        (math.pi / 2, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    (
        "fr3_joint5",
        "revolute",
        (-0.0825, 0.384, 0.0),
        (-math.pi / 2, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    (
        "fr3_joint6",
        "revolute",
        (0.0, 0.0, 0.0),
        (math.pi / 2, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    (
        "fr3_joint7",
        "revolute",
        (0.088, 0.0, 0.0),
        (math.pi / 2, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    ("fr3_joint8", "fixed", (0.0, 0.0, 0.107), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (
        "fr3_hand_joint",
        "fixed",
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -math.pi / 4),
        (0.0, 0.0, 1.0),
    ),
    (
        "fr3_hand_tcp_joint",
        "fixed",
        (0.0, 0.0, 0.1034),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
)
_FREE_PHASES = ("approach", "transport", "retreat")
_OPERATORS = ("joint_residual", "via_points", "nullspace_residual", "retime")
_SEEDS = tuple(range(9))
_STRENGTHS = (0.01, 0.025, 0.04)
_TASK_SPACE_STRENGTHS = (0.015, 0.03, 0.05)
_APPROACH_ANGLES = (0.12, 0.24, 0.36)
_STITCH_POSTURE_OFFSETS = (0.08, 0.14, 0.22)
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
        joint_names=_FRANKA_JOINT_NAMES,
        positions=positions,
        dt=dt,
        phases=phases,
        allowed_operators=_OPERATORS,
        validator_id="visualization_only",
        controlled_joint_indices=tuple(range(7)),
    )


def _reference_template() -> TrajectoryTemplate:
    values = (
        (0.00, -0.569, 0.00, -2.55, 0.00, 2.75, 0.741),
        (0.32, -0.42, 0.18, -2.30, 0.12, 2.38, 0.82),
        (0.32, -0.42, 0.18, -2.30, 0.12, 2.38, 0.82),
        (-0.52, -0.48, -0.12, -2.18, 0.28, 2.22, 0.36),
        (-0.52, -0.48, -0.12, -2.18, 0.28, 2.22, 0.36),
        (-0.18, -0.62, 0.08, -2.42, 0.08, 2.58, 0.58),
    )
    return _template_from_waypoints(
        "reference",
        tuple(torch.tensor(value, dtype=torch.float64) for value in values),
    )


def _rotation_matrix(
    rpy: tuple[float, float, float], reference: torch.Tensor
) -> torch.Tensor:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x = reference.new_tensor(((1, 0, 0), (0, cr, -sr), (0, sr, cr)))
    y = reference.new_tensor(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)))
    z = reference.new_tensor(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)))
    return z @ y @ x


def _origin_transform(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
    reference: torch.Tensor,
) -> torch.Tensor:
    transform = torch.eye(4, dtype=reference.dtype, device=reference.device).repeat(
        reference.shape[0], 1, 1
    )
    transform[:, :3, :3] = _rotation_matrix(rpy, reference)
    transform[:, :3, 3] = reference.new_tensor(xyz)
    return transform


def _axis_rotation(
    axis: tuple[float, float, float], angles: torch.Tensor
) -> torch.Tensor:
    direction = angles.new_tensor(axis)
    direction = direction / torch.linalg.vector_norm(direction)
    x, y, z = direction.unbind()
    zero = angles.new_zeros(())
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    identity = torch.eye(3, dtype=angles.dtype, device=angles.device)
    outer = direction[:, None] * direction[None, :]
    cosine = angles.cos()[:, None, None]
    sine = angles.sin()[:, None, None]
    rotation = cosine * identity + (1 - cosine) * outer + sine * skew
    transform = torch.eye(4, dtype=angles.dtype, device=angles.device).repeat(
        angles.shape[0], 1, 1
    )
    transform[:, :3, :3] = rotation
    return transform


def _franka_forward(
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the official Franka Panda serial-chain geometry."""
    if positions.ndim != 2 or positions.shape[1] != len(_FRANKA_JOINT_NAMES):
        raise ValueError("Franka positions must have shape (N, 7)")
    transform = torch.eye(4, dtype=positions.dtype, device=positions.device).repeat(
        positions.shape[0], 1, 1
    )
    active_index = 0
    joint_origins = []
    joint_axes = []
    link_points = [positions.new_zeros((positions.shape[0], 3))]
    for _, joint_type, xyz, rpy, axis in _FRANKA_KINEMATIC_JOINTS:
        transform = transform @ _origin_transform(xyz, rpy, positions)
        if joint_type == "revolute":
            joint_origins.append(transform[:, :3, 3].clone())
            local_axis = positions.new_tensor(axis).expand(positions.shape[0], -1)
            joint_axes.append(
                (transform[:, :3, :3] @ local_axis.unsqueeze(-1)).squeeze(-1)
            )
            transform = transform @ _axis_rotation(axis, positions[:, active_index])
            active_index += 1
        link_points.append(transform[:, :3, 3].clone())
    tcp = transform[:, :3, 3]
    jacobians = positions.new_zeros((positions.shape[0], 6, positions.shape[1]))
    for index, (origin, axis) in enumerate(zip(joint_origins, joint_axes, strict=True)):
        jacobians[:, :3, index] = torch.cross(axis, tcp - origin, dim=-1)
        jacobians[:, 3:, index] = axis
    return transform, jacobians, torch.stack(link_points, dim=1)


def _franka_kinematics(
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    poses, jacobians, _ = _franka_forward(positions)
    return poses, jacobians


def _tool_points(template: TrajectoryTemplate) -> torch.Tensor:
    return _franka_kinematics(template.positions)[0][:, :3, 3]


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
    jacobians = _franka_kinematics(template.positions)[1][:, :3]
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
    task_jacobians = _franka_kinematics(reference.positions)[1][:, :3]
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
                details={"constrained_task_rows": "EEF x, y, z"},
            )
            for seed in _SEEDS
        ]
        for strength in _STRENGTHS
    ]


def _manipulability_variants(
    reference: TrajectoryTemplate,
) -> list[list[dict[str, object]]]:
    reference_jacobians = _franka_kinematics(reference.positions)[1][:, :3]
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
                jacobian_fn=lambda positions: _franka_kinematics(positions)[1][:, :3],
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
        frame_id="fr3_hand_tcp",
        kinematics=_franka_kinematics,
        editable_phase_ids=_FREE_PHASES,
    )
    ik_cfg = DifferentiableIKCfg(
        max_iterations=80,
        damping=2e-3,
        step_size=0.7,
        pos_eps=5e-5,
        rot_eps=5e-5,
    )
    result = []
    for strength in _TASK_SPACE_STRENGTHS:
        constraints = {
            phase_id: TaskSpaceDeformationConstraint(
                phase_id,
                translation_axes=(True, True, True),
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
                kinematics=_franka_kinematics,
                joint_limits=_JOINT_LIMITS,
                ik_cfg=ik_cfg,
            )
            if not projection.accepted:
                raise RuntimeError(
                    f"Franka task-space variant seed {seed} failed trajectory IK."
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
    contact_pose = _franka_kinematics(
        reference.positions[contact_index : contact_index + 1]
    )[0][0]
    ik_cfg = DifferentiableIKCfg(
        max_iterations=120,
        damping=2e-3,
        step_size=0.65,
        pos_eps=5e-5,
        rot_eps=5e-5,
    )
    result = []
    for maximum_angle in _APPROACH_ANGLES:
        row = []
        for seed in _SEEDS:
            signed_fraction = (seed - (_SEEDS[-1] / 2)) / (_SEEDS[-1] / 2)
            polar_angle = maximum_angle * signed_fraction
            azimuth = math.pi / 2 if seed % 2 == 0 else -math.pi / 2
            standoff = perturb_approach_direction(
                contact_pose,
                approach_axis=torch.tensor((0.0, 0.0, 1.0), dtype=torch.float64),
                standoff_distance=0.12,
                polar_angles=torch.tensor((polar_angle,), dtype=torch.float64),
                azimuth_angles=torch.tensor((azimuth,), dtype=torch.float64),
                align_tool=False,
            )[0]
            solved = solve_dls_ik(
                standoff.unsqueeze(0),
                reference.positions[reference.phases[0].start_index],
                _franka_kinematics,
                _JOINT_LIMITS[:, 0],
                _JOINT_LIMITS[:, 1],
                ik_cfg,
            )
            if not bool(solved.success[0]):
                raise RuntimeError(
                    f"Franka approach variant seed {seed} failed standoff IK."
                )
            positions = reference.positions.clone()
            phase = reference.phases[0]
            positions[phase.start_index : phase.stop_index] = _smooth_segment(
                solved.qpos[0],
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
    posture_offset: float,
) -> TrajectoryTemplate:
    indices = [phase.start_index for phase in reference.phases]
    indices.append(reference.phases[-1].stop_index - 1)
    pattern = reference.positions.new_tensor((0.35, -0.15, 0.8, 0.0, -0.6, 0.0, 0.5))
    phase_signs = reference.positions.new_tensor((1.0, 1.0, -1.0, -1.0, 0.7, 0.7))
    lower, upper = _JOINT_LIMITS[:, 0], _JOINT_LIMITS[:, 1]
    waypoints = []
    for index, sign in zip(indices, phase_signs, strict=True):
        value = reference.positions[index] + sign * posture_offset * pattern
        waypoints.append(value.clamp(min=lower + 1e-3, max=upper - 1e-3))
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
            velocity_limits=torch.tensor(
                (2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5), dtype=torch.float64
            ),
            acceleration_limits=torch.full((7,), 6.0, dtype=torch.float64),
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
            label="Joint Residual",
            badge="qpos",
            equation="Reference joint positions → smooth residual → trajectory variant",
            principle="Adds a smooth joint-space offset inside authorized free phases. Each seed changes the offset direction and amplitude.",
            implementation="joint_residual(template, joint_limits, normalized_scale, generator)",
            guarantee="Preserved: phase endpoints and contact/hold samples. The host still validates collisions, dynamics, and task success.",
            strength_labels=(
                "1% of joint range",
                "2.5% of joint range",
                "4% of joint range",
            ),
            variants=_joint_residual_variants(reference),
        ),
        "via_points": _method(
            label="Via Points",
            badge="qpos",
            equation="Reference path → interior via points → reshaped trajectory",
            principle="Pulls each free phase through sampled interior joint targets. This creates local bends instead of one broad offset.",
            implementation="via_points(template, via_count=3, normalized_scale, generator)",
            guarantee="Preserved: every annotated waypoint. Only explicitly authorized controlled joints change.",
            strength_labels=(
                "1% waypoint offset",
                "2.5% waypoint offset",
                "4% waypoint offset",
            ),
            variants=_via_point_variants(reference),
        ),
        "nullspace": _method(
            label="Null Space",
            badge="Jacobian",
            equation="Joint residual → Jacobian null space → posture variant",
            principle="Changes redundant joint posture while preserving the selected EEF task rows to first order.",
            implementation="nullspace_residual(template, task_jacobians, ...)",
            guarantee="Preserved: phase endpoints. Recheck the first-order EEF constraint with FK and physical execution.",
            strength_labels=(
                "1% before projection",
                "2.5% before projection",
                "4% before projection",
            ),
            variants=_nullspace_variants(reference),
        ),
        "manipulability": _method(
            label="Manipulability",
            badge="guided",
            equation="Generate candidates → score manipulability → select by band",
            principle="Selects joint-residual candidates across reference-normalized manipulability bands, covering different posture conditions.",
            implementation="manipulability_guided_residual(..., bands, target_band, proposals=12)",
            guarantee="Manipulability guides sampling coverage only; it is not a feasibility or safety certificate.",
            strength_labels=(
                "1% proposal scale",
                "2.5% proposal scale",
                "4% proposal scale",
            ),
            variants=_manipulability_variants(reference),
        ),
        "task_space": _method(
            label="Task Space",
            badge="trajectory IK",
            equation="Deform EEF path → trajectory IK → joint positions",
            principle="Moves interior EEF samples, then projects the edited path back to a full joint trajectory. Locked samples keep their reference qpos.",
            implementation="deform_task_space_trajectory() + project_task_space_trajectory()",
            guarantee="Only authorized free phases change. Endpoints stay exact, and every sample reports IK success and pose residual.",
            strength_labels=("1.5 cm", "3 cm", "5 cm"),
            variants=_task_space_variants(reference),
        ),
        "timing": _method(
            label="Retiming",
            badge="retime",
            equation="Same geometry → new duration and timing profile",
            principle="Keeps the spatial path and changes free-phase speed distribution and total duration. Seeds select uniform, ease-in, or ease-out timing.",
            implementation="retime(template, duration_scale, control_dt, profile)",
            guarantee="Preserved: geometry, contact/hold duration, and phase endpoints. Open Joint Positions to see the timing change.",
            strength_labels=("1.15× duration", "1.40× duration", "1.80× duration"),
            variants=_timing_variants(reference),
        ),
        "approach": _method(
            label="Approach",
            badge="fixed contact",
            equation="Fixed contact pose → sampled standoff → replanned approach",
            principle="Changes the pre-contact direction and standoff while keeping the contact pose fixed. This is not affordance sampling.",
            implementation="perturb_approach_direction() + solve_dls_ik()",
            guarantee="Preserved: contact pose. The sampled standoff still needs reachability, clearance, and planning checks.",
            strength_labels=("±6.9° cone", "±13.8° cone", "±20.6° cone"),
            variants=_approach_variants(reference),
        ),
        "stitching": _method(
            label="Stitching",
            badge="multi-demo",
            equation="Compatible phase segments → explicit bridges → new trajectory",
            principle="Recombines semantic phases from demonstrations of the same task and keeps source lineage for every segment.",
            implementation="TrajectorySegmentBank + stitch_trajectory_segments(..., bridge_planner)",
            guarantee="Compatibility covers phase semantics and joint layout. Minimum-jerk bridges are continuous but not collision-certified.",
            strength_labels=(
                "±0.08 rad source range",
                "±0.14 rad source range",
                "±0.22 rad source range",
            ),
            variants=_stitching_variants(reference),
        ),
    }
    return {
        "title": "Trajectory Augmentation Lab",
        "subtitle": "Compare trajectory variants for one task on the 7-DoF Franka Panda.",
        "robot": {
            "name": "Franka Panda",
            "dof": len(_FRANKA_JOINT_NAMES),
            "joint_names": list(_FRANKA_JOINT_NAMES),
            "kinematic_joints": [
                {
                    "name": name,
                    "type": joint_type,
                    "xyz": list(xyz),
                    "rpy": list(rpy),
                    "axis": list(axis),
                }
                for name, joint_type, xyz, rpy, axis in _FRANKA_KINEMATIC_JOINTS
            ],
        },
        "seed_count": len(_SEEDS),
        "reference": reference_record,
        "methods": methods,
        "scope": {
            "included": "joint motion, EEF path, timing, approach, and demonstration segments",
            "excluded": "task, contact poses, affordances, observations, and scene",
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
