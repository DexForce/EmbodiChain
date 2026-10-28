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

"""Bounded Cartesian sampling with explicit reachability selection provenance."""

from __future__ import annotations

from dataclasses import replace

import math
from typing import TYPE_CHECKING

import torch

from embodichain.compute.kinematics import pose_error, damped_least_squares_step
from embodichain.utils.math import axis_angle_to_rotation_matrix

from ..config import stable_hash
from ..models import BenchmarkCase
from .free_space import _start_qpos_for_bin

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from ..config import FreeSpaceTrackCfg, SuiteCfg, TrackCfg

__all__: list[str] = []

_TRANSLATION_RANGES = {"near": (0.02, 0.06), "mid": (0.08, 0.18), "far": (0.20, 0.35)}
_ROTATION_RANGES = {"near": (0.08, 0.20), "mid": (0.25, 0.50), "far": (0.60, 1.00)}


class _RowKinematics:
    """Keep arena transforms aligned when screening only unaccepted rows."""

    def __init__(self, robot: Robot, rows: torch.Tensor) -> None:
        self.robot = robot
        self.rows = rows
        self.device = robot.device

    def get_qpos_limits(self, name: str) -> torch.Tensor:
        limits = self.robot.get_qpos_limits(name=name)
        return limits if limits.shape[0] == 1 else limits[self.rows]

    def compute_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ) -> torch.Tensor:
        return self.robot.compute_fk(
            qpos=qpos, name=name, to_matrix=to_matrix, env_ids=self.rows
        )

    def compute_batch_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool
    ) -> torch.Tensor:
        return self.robot.compute_batch_fk(
            qpos=qpos, name=name, to_matrix=to_matrix, env_ids=self.rows
        )


def _generator(*identity: object) -> torch.Generator:
    seed = int(stable_hash(identity)[:15], 16)
    return torch.Generator(device="cpu").manual_seed(seed)


def _fk_jacobian(
    robot: Robot, qpos: torch.Tensor, part: str, position_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FK and a scaled geometric Jacobian with one batched FK call.

    qpos has shape (B, K, D). Forward differences avoid relying on a
    solver-specific Jacobian API. Storage is O(B K D squared), with K bounded
    by the configured candidate pool and D the controlled joint count.
    """
    batch, count, dof = qpos.shape
    epsilon = 1.0e-3
    probes = qpos[:, :, None].expand(-1, -1, dof + 1, -1).clone()
    probes[:, :, 1:] += epsilon * torch.eye(dof, device=qpos.device, dtype=qpos.dtype)
    poses = robot.compute_batch_fk(
        qpos=probes.reshape(batch, count * (dof + 1), dof), name=part, to_matrix=True
    ).reshape(batch, count, dof + 1, 4, 4)
    translation = (poses[:, :, 1:, :3, 3] - poses[:, :, :1, :3, 3]) / epsilon
    relative = poses[:, :, 1:, :3, :3] @ poses[:, :, :1, :3, :3].transpose(-1, -2)
    angular = torch.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        dim=-1,
    ) / (2.0 * epsilon)
    jacobian = torch.cat((translation / position_scale, angular), dim=-1).transpose(
        -1, -2
    )
    return poses[:, :, 0], jacobian


def _conditioning(
    robot: Robot, qpos: torch.Tensor, part: str, position_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    _, jacobian = _fk_jacobian(robot, qpos, part, position_scale)
    singular_values = torch.linalg.svdvals(jacobian)
    minimum = singular_values[..., -1]
    ratio = minimum / singular_values[..., 0].clamp_min(1.0e-12)
    return minimum, ratio


def _sample_starts(
    robot: Robot,
    part: str,
    settings: FreeSpaceTrackCfg,
    robot_id: str,
    seed: int,
    start_bin: str,
    batch_size: int,
    batch_mode: str,
) -> tuple[torch.Tensor, dict[str, object]]:
    limits = robot.get_qpos_limits(name=part)[0].detach().cpu()
    lower, upper = limits.unbind(-1)
    span = upper - lower
    population = 1 if batch_mode == "homogeneous" else batch_size
    count = settings.singularity_candidates if start_bin == "near_singularity" else 1
    rows = []
    for row in range(population):
        generator = _generator("start-v1", seed, start_bin, row)
        base = _start_qpos_for_bin(start_bin, limits, generator, robot_id)
        noise = torch.rand(count, len(base), generator=generator) - 0.5
        if start_bin == "near_limit":
            # Vary which limits are approached and the actual distance to them.
            sides = torch.rand(count, len(base), generator=generator) > 0.5
            margins = 0.01 + 0.03 * torch.rand(count, len(base), generator=generator)
            candidates = torch.where(
                sides, upper - margins * span, lower + margins * span
            )
        else:
            fraction = 0.04 if start_bin == "near_singularity" else 0.02
            candidates = base + noise * span * fraction
        candidates = torch.maximum(
            torch.minimum(candidates, upper - span * 0.005), lower + span * 0.005
        )
        rows.append(candidates)
    candidates = torch.stack(rows)
    if population == 1:
        candidates = candidates.expand(batch_size, -1, -1).clone()
    candidates = candidates.to(robot.device)
    minimum, ratio = _conditioning(
        robot, candidates, part, settings.jacobian_position_scale_m
    )
    selected = ratio.argmin(dim=1)
    indices = torch.arange(batch_size, device=candidates.device)
    starts = candidates[indices, selected]
    minimum = minimum[indices, selected]
    ratio = ratio[indices, selected]
    margin = torch.minimum(starts.cpu() - lower, upper - starts.cpu()) / span
    return starts, {
        "requested_start_state_bin": start_bin,
        "joint_margin_fraction": margin.amin(dim=-1).tolist(),
        "jacobian_sigma_min": minimum.detach().cpu().tolist(),
        "jacobian_condition_ratio": ratio.detach().cpu().tolist(),
        "jacobian_position_scale_m": settings.jacobian_position_scale_m,
        "conditioning_class": [
            "near_singular" if value < settings.near_singular_threshold else "regular"
            for value in ratio.detach().cpu().tolist()
        ],
        "joint_limit_class": [
            "near_limit" if value <= settings.near_limit_threshold else "interior"
            for value in margin.amin(dim=-1).tolist()
        ],
        "near_singular_threshold": settings.near_singular_threshold,
        "near_limit_threshold": settings.near_limit_threshold,
        "conditioning_source": "finite_difference_geometric_jacobian_v1",
    }


def _target_path(
    start_pose: torch.Tensor,
    count: int,
    shape: str,
    component: str,
    distance_bin: str,
    generator: torch.Generator,
    batch_mode: str,
    translation_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    batch = start_pose.shape[0]
    population = 1 if batch_mode == "homogeneous" else batch
    draws = torch.randn(population, 6, generator=generator)
    direction = torch.nn.functional.normalize(draws[:, :3], dim=-1)
    axis = torch.nn.functional.normalize(draws[:, 3:], dim=-1)
    side = torch.cross(direction, axis, dim=-1)
    # Nearly parallel vectors are rare, but must not collapse a shaped path.
    fallback = torch.eye(3)[direction.abs().argmin(dim=-1)]
    side = torch.where(
        (side.norm(dim=-1) < 1.0e-5)[:, None],
        torch.cross(direction, fallback, dim=-1),
        side,
    )
    side = torch.nn.functional.normalize(side, dim=-1)
    draws = torch.rand(population, 2, generator=generator)
    low, high = translation_range or _TRANSLATION_RANGES[distance_bin]
    distance = low + (high - low) * draws[:, 0]
    low, high = _ROTATION_RANGES[distance_bin]
    angle = low + (high - low) * draws[:, 1]
    if shape == "l_turn":
        # Include the actual corner even for a three-waypoint request.
        alpha = torch.cat((torch.tensor([0.5]), torch.linspace(0.5, 1.0, count)[1:]))
    else:
        alpha = torch.arange(1, count + 1, dtype=torch.float32) / count
    offsets = alpha[None, :, None] * distance[:, None, None] * direction[:, None]
    if shape != "direct":
        bend = (
            (2.0 * torch.minimum(alpha, 1.0 - alpha))
            if shape == "l_turn"
            else torch.sin(2.0 * math.pi * alpha)
        )
        offsets += 0.3 * distance[:, None, None] * bend[None, :, None] * side[:, None]
    if component == "rotation":
        offsets.zero_()
    rotvec = alpha[None, :, None] * angle[:, None, None] * axis[:, None]
    if component == "translation":
        rotvec.zero_()
    rotations = axis_angle_to_rotation_matrix(rotvec.reshape(-1, 3)).reshape(
        population, count, 3, 3
    )
    targets = start_pose[:, None].expand(-1, count, -1, -1).clone()
    targets[:, :, :3, 3] += offsets.to(start_pose)
    targets[:, :, :3, :3] = start_pose[:, None, :3, :3] @ rotations.to(start_pose)
    return targets


def _reference_ik(
    robot: Robot,
    part: str,
    seed: torch.Tensor,
    target: torch.Tensor,
    iterations: int,
    position_tolerance: float,
    rotation_tolerance: float,
    restarts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve references using FK alone, with bounded numerical restarts.

    The robot's configured IK and measured adapters are never invoked. Per
    iteration storage is O(B * restarts * D squared), independent of path length.
    This remains a solver-conditioned population, recorded in every case.
    """
    lower, upper = robot.get_qpos_limits(name=part).to(seed).unbind(-1)
    middle = ((lower + upper) * 0.5).expand_as(seed)
    random = (
        lower[:, None]
        + torch.rand(
            seed.shape[0],
            max(0, restarts - 2),
            seed.shape[1],
            device=seed.device,
            dtype=seed.dtype,
        )
        * (upper - lower)[:, None]
    )
    qpos = torch.cat((seed[:, None], middle[:, None], random), dim=1)[:, :restarts]
    solved = torch.zeros(qpos.shape[:2], dtype=torch.bool, device=seed.device)
    for iteration in range(iterations + 1):
        current, jacobian = _fk_jacobian(robot, qpos, part, 0.3)
        position, rotation = pose_error(current, target[:, None])
        solved = (
            (position.norm(dim=-1) <= position_tolerance)
            & (rotation.norm(dim=-1) <= rotation_tolerance)
            & torch.isfinite(qpos).all(-1)
        )
        if iteration == iterations or (
            iteration % 4 == 0 and bool(solved.any(-1).all())
        ):
            break
        error = torch.cat((position / 0.3, rotation), dim=-1)
        step = damped_least_squares_step(error, jacobian, 0.02).clamp(-0.25, 0.25)
        qpos = torch.where(
            solved[..., None], qpos, (qpos + step).clamp(lower[:, None], upper[:, None])
        )
    score = (
        position.norm(dim=-1) / position_tolerance
        + rotation.norm(dim=-1) / rotation_tolerance
    )
    # Prefer a passing restart even if a failing one has a smaller summed error.
    score = torch.where(solved, score, torch.full_like(score, torch.inf))
    best = score.argmin(-1)
    rows = torch.arange(seed.shape[0], device=seed.device)
    return solved[rows, best], qpos[rows, best]


def _screen_path(
    robot: Robot,
    part: str,
    starts: torch.Tensor,
    targets: torch.Tensor,
    batch_mode: str,
    position_tolerance: float,
    rotation_tolerance: float,
    joint_tolerance: float,
    solver: str = "finite_difference_dls",
    iterations: int = 80,
    restarts: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed = starts.clone()
    alive = torch.ones(starts.shape[0], dtype=torch.bool, device=starts.device)
    references = []
    limits = robot.get_qpos_limits(name=part).to(starts)
    for index in range(targets.shape[1]):
        if solver == "robot_ik":
            success, solved = robot.compute_ik(
                pose=targets[:, index], joint_seed=seed, name=part
            )
        else:
            success, solved = _reference_ik(
                robot,
                part,
                seed,
                targets[:, index],
                iterations,
                position_tolerance,
                rotation_tolerance,
                restarts,
            )
        success = (
            torch.as_tensor(success, device=starts.device, dtype=torch.bool)
            .reshape(-1)
            .expand(starts.shape[0])
        )
        solved = torch.as_tensor(
            solved, device=starts.device, dtype=starts.dtype
        ).reshape_as(starts)
        if batch_mode == "homogeneous":
            success = success[:1].expand_as(success)
            solved = solved[:1].expand_as(solved).clone()
        measured = robot.compute_fk(qpos=solved, name=part, to_matrix=True)
        translation, rotation = pose_error(
            measured.to(torch.float64), targets[:, index].to(torch.float64)
        )
        pos_error, rotation_error = translation.norm(dim=-1), rotation.norm(dim=-1)
        valid = (
            success
            & torch.isfinite(solved).all(-1)
            & (pos_error <= position_tolerance)
            & (rotation_error <= rotation_tolerance)
        )
        valid &= (
            (solved >= limits[..., 0] - joint_tolerance)
            & (solved <= limits[..., 1] + joint_tolerance)
        ).all(-1)
        alive &= valid
        seed = torch.where(valid[:, None], solved, seed)
        references.append(seed.clone())
    return alive, torch.stack(references, dim=1)


def _sample_request(
    robot: Robot,
    part: str,
    starts: torch.Tensor,
    settings: FreeSpaceTrackCfg,
    seed: int,
    count: int,
    shape: str,
    component: str,
    distance_bin: str,
    batch_mode: str,
    suite: SuiteCfg,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    start_pose = robot.compute_fk(qpos=starts, name=part, to_matrix=True)
    targets = start_pose[:, None].expand(-1, count, -1, -1).clone()
    references = starts[:, None].expand(-1, count, -1).clone()
    accepted = torch.zeros(starts.shape[0], device=starts.device, dtype=torch.bool)
    attempts = torch.zeros_like(accepted, dtype=torch.int64)
    rng = _generator("target-v1", seed, shape, component, distance_bin)
    translation_range = _translation_range(settings, distance_bin)
    # Isolate stochastic IK from algorithm construction and other populations.
    devices = [starts.device] if starts.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(
            int(
                stable_hash(("screen-v1", seed, shape, component, distance_bin))[:15],
                16,
            )
        )
        for attempt in range(settings.max_generation_attempts):
            proposed = _target_path(
                start_pose,
                count,
                shape,
                component,
                distance_bin,
                rng,
                batch_mode,
                translation_range,
            )
            pending = (~accepted).nonzero().flatten()
            compact = settings.reachability_solver == "finite_difference_dls" and len(
                pending
            ) < len(starts)
            screening_robot = _RowKinematics(robot, pending) if compact else robot
            valid, qpos = _screen_path(
                screening_robot,
                part,
                starts[pending] if compact else starts,
                proposed[pending] if compact else proposed,
                batch_mode,
                min(0.001, suite.protocol.position_threshold_m * 0.25),
                min(0.01, suite.protocol.rotation_threshold_rad * 0.25),
                min(1.0e-5, suite.protocol.joint_limit_tolerance_rad),
                settings.reachability_solver,
                settings.screening_iterations,
                settings.screening_restarts,
            )
            if compact:
                newly_accepted = pending[valid]
                targets[newly_accepted] = proposed[newly_accepted]
                references[newly_accepted] = qpos[valid]
                attempts[newly_accepted] = attempt + 1
                accepted[newly_accepted] = True
            else:
                newly_accepted = valid & ~accepted
                targets[newly_accepted] = proposed[newly_accepted]
                references[newly_accepted] = qpos[newly_accepted]
                attempts[newly_accepted] = attempt + 1
                accepted |= valid
            if bool(accepted.all()):
                break
    if not bool(accepted.all()):
        missing = (~accepted).nonzero().flatten().cpu().tolist()
        raise ValueError(
            f"invalid_case: no independently reachable {component}/{distance_bin}/{shape} request for rows {missing} after {settings.max_generation_attempts} attempts (seed={seed})."
        )
    return targets, references, attempts.cpu().tolist()


def _translation_range(
    settings: FreeSpaceTrackCfg, distance_bin: str
) -> tuple[float, float]:
    """Resolve declared distances to metres without guessing robot dimensions."""
    default = _TRANSLATION_RANGES[distance_bin]
    if settings.distance_unit == "reach_fraction":
        # The default fractional bins match historical metres at a 0.8 m reach.
        default = tuple(value / 0.8 for value in default)
    bounds = settings.translation_ranges.get(distance_bin, default)
    scale = (
        settings.robot_reach_m if settings.distance_unit == "reach_fraction" else 1.0
    )
    return float(bounds[0] * scale), float(bounds[1] * scale)


def generate_cartesian_cases(
    suite: SuiteCfg,
    track: TrackCfg,
    settings: FreeSpaceTrackCfg,
    robot: Robot,
    part: str,
    batch_size: int,
) -> list[BenchmarkCase]:
    """Freeze screened workloads; generation never invokes a measured adapter."""
    cases = []
    for batch_mode in settings.batch_modes:
        for start_bin in settings.start_state_bins:
            for seed in settings.seeds:
                starts, start_measurements = _sample_starts(
                    robot,
                    part,
                    settings,
                    suite.robot.id,
                    seed,
                    start_bin,
                    batch_size,
                    batch_mode,
                )
                start_pose = robot.compute_fk(qpos=starts, name=part, to_matrix=True)
                for distance_bin in settings.distance_bins:
                    for component in settings.motion_components:
                        meaningful_pairs = [
                            (count, shape)
                            for count in settings.waypoint_counts
                            for shape in settings.path_shapes
                            if shape in {"direct", "l_turn", "s_curve"}
                            and (
                                (count >= 3 and component != "rotation")
                                or shape == "direct"
                            )
                        ]
                        if not meaningful_pairs:
                            raise ValueError(
                                f"No meaningful Cartesian path for {component}."
                            )
                        offset = int(
                            stable_hash(
                                (start_bin, distance_bin, component, batch_mode)
                            )[:8],
                            16,
                        )
                        selected_pair = meaningful_pairs[
                            (offset + settings.seeds.index(seed))
                            % len(meaningful_pairs)
                        ]
                        for count in settings.waypoint_counts:
                            for shape in settings.path_shapes:
                                # A single target has no turn, and pure rotation
                                # has no Cartesian translation path shape.
                                if shape not in {"direct", "l_turn", "s_curve"}:
                                    continue
                                if (
                                    count < 3 or component == "rotation"
                                ) and shape != "direct":
                                    continue
                                if (
                                    settings.population_design == "balanced"
                                    and (count, shape) != selected_pair
                                ):
                                    continue
                                targets, references, attempts = _sample_request(
                                    robot,
                                    part,
                                    starts,
                                    settings,
                                    seed,
                                    count,
                                    shape,
                                    component,
                                    distance_bin,
                                    batch_mode,
                                    suite,
                                )
                                displacement = (
                                    targets[:, -1, :3, 3] - start_pose[:, :3, 3]
                                ).norm(dim=-1)
                                _, rotation_residual = pose_error(
                                    start_pose.to(torch.float64),
                                    targets[:, -1].to(torch.float64),
                                )
                                rotation = rotation_residual.norm(dim=-1)
                                # Reuse one independently screened population for
                                # both goal modalities, without leaking IK into EEF adapters.
                                for target_type in (
                                    ["eef"]
                                    if settings.constraint_patterns
                                    else settings.target_types
                                ):
                                    identity = {
                                        "suite": suite.suite_version,
                                        "track": track.id,
                                        "seed": seed,
                                        "batch": batch_size,
                                        "mode": batch_mode,
                                        "start": start_bin,
                                        "distance": distance_bin,
                                        "component": component,
                                        "count": count,
                                        "shape": shape,
                                        "target": target_type,
                                    }
                                    fingerprints = [
                                        stable_hash(
                                            {
                                                "start": torch.round(
                                                    starts[row].cpu() * 1.0e5
                                                ).tolist(),
                                                "targets": torch.round(
                                                    (
                                                        references[row]
                                                        if target_type == "joint"
                                                        else targets[row]
                                                    ).cpu()
                                                    * 1.0e5
                                                ).tolist(),
                                                "target_type": target_type,
                                            }
                                        )
                                        for row in range(batch_size)
                                    ]
                                    parameters = {
                                        "target_type": target_type,
                                        "motion_validity": (
                                            "ordered_joint_waypoints"
                                            if target_type == "joint"
                                            else "ordered_cartesian_waypoints"
                                        ),
                                        "joint_threshold_rad": settings.joint_threshold_rad,
                                        "sampling_method": "cartesian_v2",
                                        "population_design": settings.population_design,
                                        "batch_mode": batch_mode,
                                        "distance_bin": distance_bin,
                                        "motion_component": component,
                                        "path_frame": "arena_cartesian",
                                        "path_shape_semantics": "cartesian_waypoint_layout",
                                        "distance_unit": settings.distance_unit,
                                        "robot_reach_m": settings.robot_reach_m,
                                        "translation_range_m": list(
                                            _translation_range(settings, distance_bin)
                                        ),
                                        "tcp_translation_m": displacement.detach()
                                        .cpu()
                                        .tolist(),
                                        "tcp_rotation_rad": rotation.detach()
                                        .cpu()
                                        .tolist(),
                                        "generation_attempts": attempts,
                                        "input_fingerprints": fingerprints,
                                        "unique_input_rows": len(set(fingerprints)),
                                        "reachability_evidence": f"{settings.reachability_solver}_and_fk_v2",
                                        "reachability_solver": settings.reachability_solver,
                                        "screening_iterations": settings.screening_iterations,
                                        "screening_restarts": settings.screening_restarts,
                                        "selection_condition": "reachable_by_declared_screening_solver",
                                        **start_measurements,
                                    }
                                    cases.append(
                                        BenchmarkCase(
                                            suite_version=suite.suite_version,
                                            track=track.id,
                                            scenario_id=(
                                                "joint_"
                                                if target_type == "joint"
                                                else ""
                                            )
                                            + (
                                                "waypoint_path"
                                                if count > 1
                                                else "reach"
                                            ),
                                            case_id=f"free_space_{stable_hash(identity)[:16]}",
                                            seed=seed,
                                            batch_size=batch_size,
                                            num_waypoints=count,
                                            path_shape=shape,
                                            start_state_bin=start_bin,
                                            start_qpos=starts.clone(),
                                            target_waypoints=targets.clone(),
                                            reference_qpos=references.clone(),
                                            robot_id=suite.robot.id,
                                            primary_success="geometry_valid",
                                            case_parameters=parameters,
                                        )
                                    )
    if not cases:
        raise ValueError(
            "Cartesian free-space configuration generated no meaningful workloads."
        )
    return (
        _project_constraints(cases, settings) if settings.constraint_patterns else cases
    )


def _project_constraints(
    bases: list[BenchmarkCase], settings: FreeSpaceTrackCfg
) -> list[BenchmarkCase]:
    """Project masks without drawing new starts, poses or IK witnesses."""
    cases = []
    for base in bases:
        eligible = [
            p for p in settings.constraint_patterns if len(p) <= base.num_waypoints
        ]
        if settings.constraint_pattern_design == "balanced":
            selected = [p for p in eligible if len(p) == 1]
            offset = int(
                stable_hash(
                    {
                        "start": base.start_state_bin,
                        "shape": base.path_shape,
                        "k": base.num_waypoints,
                        "distance": base.case_parameters["distance_bin"],
                        "component": base.case_parameters["motion_component"],
                    }
                )[:8],
                16,
            ) + settings.seeds.index(base.seed)
            for length in sorted({len(p) for p in eligible if len(p) > 1}):
                patterns = [p for p in eligible if len(p) == length]
                selected.append(patterns[offset % len(patterns)])
            eligible = selected
        for pattern in eligible:
            modalities = [pattern[i % len(pattern)] for i in range(base.num_waypoints)]
            homogeneous = len(set(modalities)) == 1
            modality = modalities[0] if homogeneous else "mixed"
            # Preserve the classical planner input contract for the common subset.
            # Their EEF goals are poses; reference_qpos remains screening evidence.
            validity = {
                "joint": "ordered_joint_waypoints",
                "pose": "ordered_cartesian_waypoints",
            }.get(modality, "ordered_constraint_waypoints")
            fingerprints = []
            for row in range(base.batch_size):
                payloads = []
                for index, kind in enumerate(modalities):
                    pose = base.target_waypoints[row, index]
                    value = (
                        base.reference_qpos[row, index]
                        if kind == "joint"
                        else (
                            pose[:3, 3]
                            if kind == "position"
                            else pose[:3, :3] if kind == "orientation" else pose
                        )
                    )
                    payloads.append(torch.round(value.detach().cpu() * 1.0e5).tolist())
                fingerprints.append(
                    stable_hash(
                        {
                            "start": torch.round(
                                base.start_qpos[row].detach().cpu() * 1.0e5
                            ).tolist(),
                            "modalities": modalities,
                            "payloads": payloads,
                        }
                    )
                )
            parameters = {
                **base.case_parameters,
                "input_fingerprints": fingerprints,
                "unique_input_rows": len(set(fingerprints)),
                "base_case_id": base.case_id,
                "constraint_pattern": ",".join(pattern),
                "constraint_modalities": modalities,
                "constraint_family": (
                    modality if homogeneous else f"mixed_{len(pattern)}"
                ),
                "target_type": modality,
                "motion_validity": validity,
                "joint_error_norm": "linf",
                "sample_grid": "native_with_linear_subdivision",
            }
            cases.append(
                replace(
                    base,
                    case_id=f"free_space_{stable_hash((base.case_id, modalities))[:16]}",
                    scenario_id="constraint_sequence",
                    case_parameters=parameters,
                )
            )
    if not cases:
        raise ValueError("No constraint pattern fits the declared waypoint counts.")
    return cases
