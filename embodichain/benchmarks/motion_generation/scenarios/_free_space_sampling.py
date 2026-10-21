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

"""Bounded, planner-independent sampling of reachable Cartesian requests."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from embodichain.compute.kinematics import pose_error
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


def _generator(*identity: object) -> torch.Generator:
    seed = int(stable_hash(identity)[:15], 16)
    return torch.Generator(device="cpu").manual_seed(seed)


def _conditioning(
    robot: Robot, qpos: torch.Tensor, part: str, position_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure a scaled geometric Jacobian with one batched FK call.

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
            "near_singular" if value < 0.02 else "regular"
            for value in ratio.detach().cpu().tolist()
        ],
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
    low, high = _TRANSLATION_RANGES[distance_bin]
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


def _screen_path(
    robot: Robot,
    part: str,
    starts: torch.Tensor,
    targets: torch.Tensor,
    batch_mode: str,
    position_tolerance: float,
    rotation_tolerance: float,
    joint_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed = starts.clone()
    alive = torch.ones(starts.shape[0], dtype=torch.bool, device=starts.device)
    references = []
    limits = robot.get_qpos_limits(name=part).to(starts)
    for index in range(targets.shape[1]):
        success, solved = robot.compute_ik(
            pose=targets[:, index], joint_seed=seed, name=part
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
                start_pose, count, shape, component, distance_bin, rng, batch_mode
            )
            valid, qpos = _screen_path(
                robot,
                part,
                starts,
                proposed,
                batch_mode,
                min(0.001, suite.protocol.position_threshold_m * 0.25),
                min(0.01, suite.protocol.rotation_threshold_rad * 0.25),
                min(1.0e-5, suite.protocol.joint_limit_tolerance_rad),
            )
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
                                for target_type in settings.target_types:
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
                                        "sampling_method": "cartesian_v1",
                                        "population_design": settings.population_design,
                                        "batch_mode": batch_mode,
                                        "distance_bin": distance_bin,
                                        "motion_component": component,
                                        "path_frame": "arena_cartesian",
                                        "tcp_translation_m": displacement.detach()
                                        .cpu()
                                        .tolist(),
                                        "tcp_rotation_rad": rotation.detach()
                                        .cpu()
                                        .tolist(),
                                        "generation_attempts": attempts,
                                        "input_fingerprints": fingerprints,
                                        "unique_input_rows": len(set(fingerprints)),
                                        "reachability_evidence": "independent_sequential_ik_and_fk_v1",
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
    return cases
