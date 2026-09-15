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

"""Bounded environment-row planning for explicitly unloaded free trajectories."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    MotionSnapshot,
    TrajectoryPhase,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.motion.motion_generator import MotionGenerator

__all__ = ["EEFPath", "EnvRowMotionPlanner"]


@dataclass(frozen=True)
class EEFPath:
    """Explicit TCP samples in the source row's local arena frame.

    The first pose is the initial TCP anchor. ``dt`` contains arrival intervals,
    starting at zero. Optional solved joint samples preserve a chosen IK branch
    exactly and are checked against the poses through FK, without another IK.
    Paths have no attachment or contact-permission representation.
    """

    identity: CandidateIdentity
    source_row_index: int
    poses: torch.Tensor
    dt: torch.Tensor
    phases: tuple[TrajectoryPhase, ...]
    solved_joint_targets: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CandidateIdentity):
            raise ValueError("identity must be CandidateIdentity")
        if type(self.source_row_index) is not int or self.source_row_index < 0:
            raise ValueError("source_row_index must be a nonnegative integer")
        poses = self.poses
        if (
            not isinstance(poses, torch.Tensor)
            or not poses.is_floating_point()
            or poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or not poses.shape[0]
            or not bool(torch.isfinite(poses).all())
        ):
            raise ValueError("poses must be finite floating shape (N >= 1, 4, 4)")
        rotation = poses[:, :3, :3].double()
        identity = torch.eye(3, device=poses.device, dtype=torch.float64)
        if (
            not torch.allclose(
                rotation.transpose(1, 2) @ rotation,
                identity.expand_as(rotation),
                atol=1e-5,
                rtol=0,
            )
            or not torch.allclose(
                torch.linalg.det(rotation),
                torch.ones(len(poses), device=poses.device, dtype=torch.float64),
                atol=1e-5,
                rtol=0,
            )
            or not torch.allclose(
                poses[:, 3],
                poses.new_tensor([0, 0, 0, 1]).expand(len(poses), 4),
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError("poses must contain proper SE(3) transforms")
        reference = TrajectoryTemplate(
            self.identity.source_id,
            self.identity.source_revision,
            self.identity.template_id,
            ("validation_joint",),
            poses.new_zeros((len(poses), 1)),
            self.dt,
            self.phases,
        )
        solved = self.solved_joint_targets
        if solved is not None:
            if (
                not isinstance(solved, torch.Tensor)
                or not solved.is_floating_point()
                or solved.ndim != 2
                or solved.shape[0] != len(poses)
                or not solved.shape[1]
                or not bool(torch.isfinite(solved).all())
            ):
                raise ValueError(
                    "solved_joint_targets must be finite floating shape (N, D)"
                )
            solved = solved.detach().clone()
        object.__setattr__(self, "poses", poses.detach().clone())
        object.__setattr__(self, "dt", reference.dt)
        object.__setattr__(self, "phases", reference.phases)
        object.__setattr__(self, "solved_joint_targets", solved)


def _check(status: str, detail: str, **metrics: float) -> ValidationResult:
    return ValidationResult(
        (ValidationCheck("path_collision", status, detail, metrics),)
    )


class EnvRowMotionPlanner:
    """Convert EEF paths and check qpos without changing simulator batch size.

    Only fully annotated free motion with no held object is supported. Every
    uncontrolled joint must remain at the snapshot and robot configuration's
    initial value, matching the current planner's locked-joint model. Callers
    keep the live root poses and collision model synchronized with these same
    snapshots. Scene entities use canonical IDs in the backend collision world.

    Collision validation densifies joint segments, including phase boundaries,
    then checks those samples through ``MotionGenerator``. This is a bounded
    sampled approximation, not continuous collision detection. It neither steps
    physics nor verifies task success or controller velocity/acceleration limits.

    Args:
        motion_generator: Existing generator attached to the real batch robot.
        control_part: Ordered control part represented by the backend model.
        held_object_ids: Explicit held-object declaration; nonempty is unsupported.
        max_joint_step: Maximum absolute joint increment between checked samples.
        max_validation_samples: Maximum samples per candidate, including anchors.
        fk_position_tolerance: Maximum TCP translation residual in meters.
        fk_rotation_tolerance: Maximum TCP rotation residual in radians.
    """

    def __init__(
        self,
        motion_generator: MotionGenerator,
        *,
        control_part: str,
        held_object_ids: Sequence[str],
        max_joint_step: float = 0.02,
        max_validation_samples: int = 4096,
        fk_position_tolerance: float = 1e-3,
        fk_rotation_tolerance: float = 1e-2,
    ) -> None:
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        if not isinstance(control_part, str) or not control_part:
            raise ValueError("control_part must be a nonempty name")
        self.control_part = control_part
        self.joint_names = tuple(self.robot.joint_names)
        self.joint_ids = tuple(self.robot.get_joint_ids(control_part))
        if not self.joint_ids or len(set(self.joint_ids)) != len(self.joint_ids):
            raise ValueError("control_part must contain unique joints")
        self.fixed_ids = tuple(
            i for i in range(len(self.joint_names)) if i not in self.joint_ids
        )
        if isinstance(held_object_ids, str) or any(
            not isinstance(x, str) or not x for x in held_object_ids
        ):
            raise ValueError("held_object_ids must be an explicit sequence of names")
        self.held_object_ids = tuple(held_object_ids)
        for name, value in (
            ("max_joint_step", max_joint_step),
            ("fk_position_tolerance", fk_position_tolerance),
            ("fk_rotation_tolerance", fk_rotation_tolerance),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")
            setattr(self, name, float(value))
        if type(max_validation_samples) is not int or max_validation_samples < 1:
            raise ValueError("max_validation_samples must be a positive integer")
        self.max_validation_samples = max_validation_samples

    def _snapshots(
        self, snapshots: Sequence[MotionSnapshot]
    ) -> tuple[MotionSnapshot, ...]:
        values = tuple(snapshots)
        if len(values) != self.robot.num_instances or not values:
            raise ValueError("snapshots must cover the complete real robot batch")
        if any(
            not isinstance(value, MotionSnapshot)
            or value.joint_names != self.joint_names
            for value in values
        ):
            raise ValueError("snapshots must use the robot's full ordered joint names")
        current_roots = self.robot.get_local_pose(to_matrix=True)
        for row, snapshot in enumerate(values):
            if not torch.allclose(
                current_roots[row].to(snapshot.root_pose),
                snapshot.root_pose,
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("snapshot root pose differs from its live source row")
        return values

    @staticmethod
    def _rounds(indices: Sequence[int], candidate_ids: Sequence[int]):
        queues: dict[int, list[int]] = {}
        for candidate in candidate_ids:
            queues.setdefault(indices[candidate], []).append(candidate)
        while any(queues.values()):
            yield {row: queue.pop(0) for row, queue in queues.items() if queue}

    def _unsupported(
        self, phases: tuple[TrajectoryPhase, ...], length: int
    ) -> str | None:
        if self.held_object_ids:
            return "held-object sweep geometry is unavailable"
        end = 0
        for phase in phases:
            if phase.kind != "free" or phase.start_index != end:
                return (
                    "only fully annotated free motion has supported collision semantics"
                )
            end = phase.stop_index
        return (
            None if end == length else "every sample must have an explicit free phase"
        )

    def _obstacles(self, snapshots, obstacle_poses):
        names = self.motion_generator.dynamic_collision_entity_ids
        if obstacle_poses is None and names:
            if any(
                name not in snapshot.entity_poses
                for name in names
                for snapshot in snapshots
            ):
                raise ValueError("snapshots lack required dynamic collision poses")
            obstacle_poses = {
                name: torch.stack(
                    [snapshot.entity_poses[name] for snapshot in snapshots]
                )
                for name in names
            }
        if obstacle_poses is None:
            return None
        if set(obstacle_poses) != set(names):
            raise ValueError(
                "obstacle poses must match the backend's dynamic entity IDs"
            )
        result = {}
        for name, pose in obstacle_poses.items():
            if (
                not isinstance(pose, torch.Tensor)
                or pose.shape != (len(snapshots), 4, 4)
                or not bool(torch.isfinite(pose).all())
            ):
                raise ValueError("obstacle poses must cover all real source rows")
            result[name] = pose.detach().clone()
        return result

    def validate_qpos(
        self,
        batch: CandidateTrajectoryBatch,
        snapshots: Sequence[MotionSnapshot],
        *,
        obstacle_poses: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[ValidationResult, ...]:
        """Check original candidates without replacing their positions or timing.

        Return one result per candidate in input order. Unsupported geometry or
        backend capability yields ``unavailable``; collisions and invalid paths
        yield ``failed``. No success is inferred from padding or planning alone.

        Measured qpos may be checked after execution while the live base frames
        remain fixed. The first sample must match its snapshot within ``1e-6``.
        Obstacle poses describe one constant world for the entire path; callers
        must independently establish that the world stayed fixed throughout a
        measured rollout. Neither initial nor final obstacle poses reconstruct
        a moving obstacle's history.
        """
        snapshots = self._snapshots(snapshots)
        if batch.joint_names != self.joint_names:
            raise ValueError(
                "candidate joint names must match the complete robot order"
            )
        count = len(batch.identities)
        if not count:
            return ()
        if batch.source_row_indices is None:
            raise ValueError(
                "source_row_indices are required for environment-row planning"
            )
        indices = batch.source_row_indices.tolist()
        if min(indices) < 0 or max(indices) >= len(snapshots):
            raise ValueError("source_row_indices exceed the real robot batch")
        obstacles = self._obstacles(snapshots, obstacle_poses)
        device = self.robot.device
        starts = torch.stack([snapshot.joint_positions for snapshot in snapshots]).to(
            device
        )
        initial_values = self.robot.cfg.init_qpos
        initial = torch.as_tensor(
            [] if initial_values is None else initial_values,
            device=device,
            dtype=starts.dtype,
        )
        limits = self.robot.get_qpos_limits().to(device)
        results = [
            _check("not_run", "candidate has not been checked") for _ in range(count)
        ]
        sample_counts: dict[int, int] = {}
        for candidate, source in enumerate(indices):
            identity = batch.identities[candidate]
            case = snapshots[source].scene_case
            if (identity.scene_case_id, identity.initial_state_id) != (
                case.scene_case_id,
                case.initial_state_id,
            ):
                raise ValueError(
                    "candidate identity does not match its source snapshot"
                )
            length = int(batch.valid_length[candidate])
            q = batch.positions[candidate, :length].to(device)
            dt = batch.dt[candidate, :length]
            unsupported = self._unsupported(batch.phases[candidate], length)
            if unsupported:
                results[candidate] = _check("unavailable", unsupported)
                continue
            if (
                not bool(torch.isfinite(q).all())
                or not bool(torch.isfinite(dt).all())
                or dt[0] != 0
                or bool((dt[1:] <= 0).any())
            ):
                results[candidate] = _check(
                    "failed", "non-finite positions or invalid arrival intervals"
                )
                continue
            if not torch.allclose(q[0], starts[source].to(q), atol=1e-6, rtol=0):
                results[candidate] = _check(
                    "failed", "trajectory does not start at the source initial state"
                )
                continue
            if self.fixed_ids and (
                initial.shape != starts.shape[1:]
                or not torch.allclose(
                    q[:, self.fixed_ids],
                    starts[source, self.fixed_ids].expand(length, -1).to(q),
                    atol=1e-6,
                    rtol=0,
                )
                or not torch.allclose(
                    starts[source, self.fixed_ids],
                    initial[list(self.fixed_ids)],
                    atol=1e-6,
                    rtol=0,
                )
            ):
                results[candidate] = _check(
                    "unavailable",
                    "changing or noninitial locked joints require a full-state collision model",
                )
                continue
            if bool(((q < limits[source, :, 0]) | (q > limits[source, :, 1])).any()):
                results[candidate] = _check("failed", "joint limits exceeded")
                continue
            if not self.motion_generator.supports_joint_trajectory_validation:
                results[candidate] = _check(
                    "unavailable", "backend cannot validate exact joint samples"
                )
                continue
            increments = torch.ceil(
                (q[1:, self.joint_ids].double() - q[:-1, self.joint_ids].double())
                .abs()
                .amax(dim=1)
                / self.max_joint_step
            ).clamp_min(1)
            if float(increments.sum()) + 1 > self.max_validation_samples:
                results[candidate] = _check(
                    "unavailable", "collision sampling exceeds max_validation_samples"
                )
                continue
            sample_counts[candidate] = int(increments.sum()) + 1
        for round_rows in self._rounds(indices, list(sample_counts)):
            horizon = max(sample_counts[index] for index in round_rows.values())
            inputs = starts[:, None, self.joint_ids].expand(-1, horizon, -1).clone()
            for row, candidate in round_rows.items():
                length = int(batch.valid_length[candidate])
                q = batch.positions[candidate, :length, self.joint_ids].to(device)
                increments = (
                    torch.ceil(
                        (q[1:].double() - q[:-1].double()).abs().amax(dim=1)
                        / self.max_joint_step
                    )
                    .clamp_min(1)
                    .long()
                )
                cursor = 1
                inputs[row, 0] = q[0]
                for index, intervals in enumerate(increments.tolist()):
                    alpha = (
                        torch.arange(
                            1, intervals + 1, device=device, dtype=torch.float64
                        )
                        / intervals
                    )
                    samples = q[index].double() + alpha[:, None] * (
                        q[index + 1].double() - q[index].double()
                    )
                    inputs[row, cursor : cursor + intervals] = samples.to(inputs)
                    cursor += intervals
                inputs[row, cursor:] = q[-1]
            try:
                valid = self.motion_generator.validate_joint_trajectory(
                    inputs, control_part=self.control_part, obstacle_poses=obstacles
                )
            except (ValueError, RuntimeError, NotImplementedError) as error:
                for candidate in round_rows.values():
                    results[candidate] = _check(
                        "unavailable", f"backend validation failed: {error}"
                    )
                continue
            for row, candidate in round_rows.items():
                mask = valid[row, : sample_counts[candidate]]
                results[candidate] = _check(
                    "passed" if bool(mask.all()) else "failed",
                    "sampled joint bounds, self and environment collision checks; no continuous guarantee",
                    samples=float(len(mask)),
                    max_joint_step=self.max_joint_step,
                )
        return tuple(results)

    def plan_eef(
        self,
        paths: Sequence[EEFPath],
        snapshots: Sequence[MotionSnapshot],
        *,
        obstacle_poses: Mapping[str, torch.Tensor] | None = None,
    ) -> tuple[CandidateTrajectoryBatch, tuple[ValidationResult, ...]]:
        """Solve exact EEF samples, preserve supplied branches, and check qpos.

        Returned rows retain input order, phases, intervals, and source indices.
        Failed rows hold their full initial state and must be rejected using the
        paired validation result. IK always uses the real environment batch.
        """
        snapshots = self._snapshots(snapshots)
        paths = tuple(paths)
        if any(not isinstance(path, EEFPath) for path in paths):
            raise TypeError("paths must contain EEFPath values")
        if any(path.source_row_index >= len(snapshots) for path in paths):
            raise ValueError("EEF source row exceeds the real robot batch")
        device = self.robot.device
        starts = torch.stack([snapshot.joint_positions for snapshot in snapshots]).to(
            device
        )
        indices = [path.source_row_index for path in paths]
        horizon = max((len(path.poses) for path in paths), default=0)
        if horizon > self.max_validation_samples:
            raise ValueError("EEF input horizon exceeds max_validation_samples")
        for path in paths:
            if path.solved_joint_targets is not None:
                starts = starts.to(
                    dtype=torch.promote_types(
                        starts.dtype, path.solved_joint_targets.dtype
                    )
                )
        positions = starts.new_empty((len(paths), horizon, len(self.joint_names)))
        timing_dtype = paths[0].dt.dtype if paths else starts.dtype
        for path in paths[1:]:
            timing_dtype = torch.promote_types(timing_dtype, path.dt.dtype)
        dt = torch.zeros((len(paths), horizon), device=device, dtype=timing_dtype)
        failures: dict[int, ValidationResult] = {}
        for index, path in enumerate(paths):
            positions[index] = starts[path.source_row_index]
            dt[index, : len(path.dt)] = path.dt.to(dt)
            unsupported = self._unsupported(path.phases, len(path.poses))
            if unsupported:
                failures[index] = _check("unavailable", unsupported)
            if (
                path.solved_joint_targets is not None
                and path.solved_joint_targets.shape[1] != len(self.joint_ids)
            ):
                raise ValueError(
                    "solved_joint_targets must match the ordered control part"
                )
        for round_rows in self._rounds(
            indices, [i for i in range(len(paths)) if i not in failures]
        ):
            seeds = starts[:, self.joint_ids].clone()
            active = dict(round_rows)
            for sample in range(
                max(len(paths[index].poses) for index in active.values())
            ):
                expected = self.robot.compute_fk(
                    qpos=seeds, name=self.control_part, to_matrix=True
                )
                targets = expected.clone()
                unsolved = []
                supplied = {}
                for row, candidate in active.items():
                    path = paths[candidate]
                    if sample >= len(path.poses):
                        continue
                    targets[row] = path.poses[sample].to(targets)
                    if sample and path.solved_joint_targets is None:
                        unsolved.append(row)
                    elif path.solved_joint_targets is not None:
                        supplied[row] = path.solved_joint_targets[sample].to(seeds)
                solved = seeds.clone()
                success = torch.ones(len(snapshots), dtype=torch.bool, device=device)
                if unsolved:
                    ik_targets = expected.clone()
                    ik_targets[unsolved] = targets[unsolved]
                    try:
                        mask, ik = self.robot.compute_ik(
                            pose=ik_targets, name=self.control_part, joint_seed=seeds
                        )
                        mask = torch.as_tensor(
                            mask, device=device, dtype=torch.bool
                        ).reshape(-1)
                        ik = torch.as_tensor(ik, device=device, dtype=seeds.dtype)
                        if ik.shape == (len(snapshots), 1, len(self.joint_ids)):
                            ik = ik[:, 0]
                        if mask.shape != success.shape or ik.shape != seeds.shape:
                            raise ValueError(
                                "IK result has an invalid real-batch shape"
                            )
                        mask &= torch.isfinite(ik).all(dim=1)
                        success[unsolved] = mask[unsolved]
                        solved[unsolved] = torch.where(
                            mask[unsolved, None], ik[unsolved], seeds[unsolved]
                        )
                    except (ValueError, RuntimeError, TypeError) as error:
                        for row in unsolved:
                            failures[active[row]] = _check(
                                "failed", f"IK failed: {error}"
                            )
                            success[row] = False
                for row, value in supplied.items():
                    solved[row] = value
                actual = self.robot.compute_fk(
                    qpos=solved, name=self.control_part, to_matrix=True
                )
                translation = torch.linalg.vector_norm(
                    actual[:, :3, 3] - targets[:, :3, 3], dim=1
                )
                relative = actual[:, :3, :3].transpose(1, 2) @ targets[:, :3, :3]
                cosine = ((relative.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(
                    -1, 1
                )
                angle = torch.acos(cosine)
                success &= (
                    torch.isfinite(actual).all(dim=2).all(dim=1)
                    & (translation <= self.fk_position_tolerance)
                    & (angle <= self.fk_rotation_tolerance)
                )
                for row, candidate in tuple(active.items()):
                    if sample >= len(paths[candidate].poses):
                        continue
                    if not success[row]:
                        failures.setdefault(
                            candidate,
                            _check(
                                "failed",
                                "IK failed or TCP FK residual exceeded tolerance",
                            ),
                        )
                        positions[candidate] = starts[row]
                        del active[row]
                    else:
                        positions[candidate, sample, self.joint_ids] = solved[row]
                        seeds[row] = solved[row]
                if not active:
                    break
        for index, path in enumerate(paths):
            positions[index, len(path.poses) :] = positions[index, len(path.poses) - 1]
        batch = CandidateTrajectoryBatch(
            positions,
            dt,
            torch.tensor(
                [len(path.poses) for path in paths], dtype=torch.int64, device=device
            ),
            tuple(path.identity for path in paths),
            self.joint_names,
            tuple(path.phases for path in paths),
            source_row_indices=torch.tensor(indices, dtype=torch.int64, device=device),
        )
        results = list(
            self.validate_qpos(batch, snapshots, obstacle_poses=obstacle_poses)
        )
        for index, failure in failures.items():
            results[index] = failure
        return batch, tuple(results)
