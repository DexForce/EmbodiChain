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

"""Cartesian approach sampling as a task-owned motion-planning policy."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import json
import math
from typing import Any, Iterator
import xml.etree.ElementTree as ET

import torch
from embodichain.utils import logger

from embodichain.lab.sim.motion.motion_generator import (
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners.utils import (
    MoveType,
    PlanResult,
    PlanState,
    interpolate_xpos_batched,
)

__all__: list[str] = []

MOTION_VALIDATION_REVISION = 4
VELOCITY_RETIME_SAMPLES = (260, 320)


@dataclass
class _PlaceIKRecovery:
    used: bool = False


_PLACE_IK_RECOVERY: ContextVar[_PlaceIKRecovery | None] = ContextVar(
    "gen_sim_place_ik_recovery", default=None
)


@contextmanager
def place_ik_recovery() -> Iterator[_PlaceIKRecovery]:
    """Scope the bounded fallback to one GenSim Place planning call."""
    state = _PlaceIKRecovery()
    token = _PLACE_IK_RECOVERY.set(state)
    try:
        yield state
    finally:
        _PLACE_IK_RECOVERY.reset(token)


def _local_place_ik(
    generator: MotionGenerator, targets: list[PlanState], options: MotionGenOptions
) -> PlanResult | None:
    """Recover a single-environment path without changing targets or timing."""
    robot = generator.robot
    seed = options.start_qpos.clone()
    bounds = robot.get_qpos_limits(name=options.control_part).to(seed)
    lower, upper = bounds[..., 0], bounds[..., 1]
    limits = _joint_velocity_limits(robot, options.control_part).to(seed)
    step_dt = options.interpolation_dt
    allowed = limits * step_dt + 1e-5
    positions = [seed]
    repairs = 0

    def solve(
        target: PlanState, trial: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = MotionGenerator.generate(
            generator,
            [target],
            options=replace(
                options,
                start_qpos=trial,
                sample_count=2,
                preserve_cartesian_samples=True,
            ),
        )
        return result.success, result.positions[:, -1]

    def feasible(ok: torch.Tensor, qpos: torch.Tensor) -> bool:
        return bool(
            ok.all()
            and torch.isfinite(qpos).all()
            and (qpos >= lower).all()
            and (qpos <= upper).all()
            and ((qpos - seed).abs() <= allowed).all()
        )

    # Failed searches must not perturb the later grasp-candidate random stream.
    devices = [seed.device.index] if seed.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        for target in targets:
            ok, qpos = solve(target, seed)
            qpos = qpos.reshape_as(seed)
            best = qpos if feasible(ok, qpos) else None
            margin = (
                float(torch.minimum(best - lower, upper - best).min())
                if best is not None
                else -1.0
            )
            if margin < 0.02:
                # Near a joint stop, global random seeds can miss a nearby
                # redundant branch. Bias only seeds, never emitted commands.
                for joint in range(seed.shape[1]):
                    for offset in (-0.15, 0.15, -0.05, 0.05):
                        trial = seed.clone()
                        trial[:, joint] += offset
                        trial = trial.clamp(lower, upper)
                        ok, candidate = solve(target, trial)
                        candidate = candidate.reshape_as(seed)
                        if feasible(ok, candidate):
                            candidate_margin = float(
                                torch.minimum(
                                    candidate - lower, upper - candidate
                                ).min()
                            )
                            if candidate_margin > margin:
                                best, margin = candidate, candidate_margin
                repairs += 1
            if best is None:
                return None
            seed = best
            positions.append(seed)
    path = torch.stack(positions, dim=1)
    dt = path.new_full(path.shape[:2], step_dt)
    dt[:, 0] = 0.0
    success = _velocity_validity(path, dt, limits)
    if not success.all():
        return None
    logger.log_info(
        f"GenSim Place local IK recovery accepted: {repairs} local searches, "
        f"{path.shape[1]} unchanged-time samples."
    )
    return PlanResult(success=success, positions=path, dt=dt)


def _velocity_retry_samples(sample_count: int | None) -> tuple[int, ...]:
    """Return denser fallback budgets without changing any joint limit."""
    if sample_count is None:
        return ()
    return tuple(value for value in VELOCITY_RETIME_SAMPLES if value > sample_count)


def _joint_velocity_limits(robot: Any, control_part: str | None) -> torch.Tensor:
    """Intersect runtime limits with the assembled model's named URDF limits."""
    declared = {}
    for joint in ET.parse(robot.cfg.fpath).getroot().findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "velocity" in limit.attrib:
            declared[joint.attrib["name"]] = float(limit.attrib["velocity"])
    names = [robot.joint_names[int(i)] for i in robot.get_joint_ids(name=control_part)]
    if any(
        name not in declared or not math.isfinite(declared[name]) or declared[name] <= 0
        for name in names
    ):
        raise ValueError(
            "Every controlled joint requires a finite positive URDF velocity limit."
        )
    runtime = robot.get_qvel_limits(name=control_part)
    if runtime.ndim != 2 or runtime.shape[1] != len(names):
        raise ValueError(
            "Runtime velocity limits must match the named control-part joints."
        )
    return torch.minimum(
        runtime, runtime.new_tensor([declared[name] for name in names])[None]
    )


class CheckedMotionGenerator(MotionGenerator):
    """Preserve planner output while rejecting velocity-infeasible rows."""

    def generate(
        self, target_states: list[PlanState], options: MotionGenOptions | None = None
    ) -> PlanResult:
        result = super().generate(target_states, options=options)
        if (
            result.positions is not None
            and options is not None
            and options.control_part is not None
        ):
            limits = _joint_velocity_limits(self.robot, options.control_part).to(
                result.positions
            )
            valid_velocity = _velocity_validity(result.positions, result.dt, limits)
            success = torch.as_tensor(
                result.success, device=valid_velocity.device, dtype=torch.bool
            )
            if (success & ~valid_velocity).any():
                logger.log_warning(
                    "GenSim motion exceeds declared joint velocity limits; rejecting affected rows."
                )
                logger.log_warning(
                    "GenSim velocity diagnostics: "
                    + json.dumps(
                        _velocity_diagnostics(result.positions, result.dt, limits),
                        allow_nan=False,
                    )
                )
                result = replace(result, success=success & valid_velocity)
        return result


def _velocity_diagnostics(
    positions: torch.Tensor, dt: torch.Tensor, limits: torch.Tensor
) -> list[dict[str, Any]]:
    """Report the largest finite-interval speed ratio per environment."""
    records = []
    for env_id in range(positions.shape[0]):
        delta = (positions[env_id, 1:] - positions[env_id, :-1]).abs()
        allowed = dt[env_id, 1:, None] * limits[env_id, None]
        ratio = torch.where(
            allowed > 0, delta / allowed, torch.where(delta == 0, 0.0, float("inf"))
        )
        if not ratio.numel():
            continue
        index = int(torch.nan_to_num(ratio, nan=float("inf")).argmax())
        frame, joint = divmod(index, positions.shape[2])
        values = {
            "previous_qpos": float(positions[env_id, frame, joint]),
            "next_qpos": float(positions[env_id, frame + 1, joint]),
            "dt": float(dt[env_id, frame + 1]),
            "velocity_limit": float(limits[env_id, joint]),
            "speed_ratio": float(ratio[frame, joint]),
        }
        records.append(
            {
                "env_id": env_id,
                "frame": frame + 1,
                "control_joint_index": joint,
                **{
                    key: value if math.isfinite(value) else None
                    for key, value in values.items()
                },
            }
        )
    return records


def _velocity_validity(
    positions: torch.Tensor, dt: torch.Tensor, limits: torch.Tensor
) -> torch.Tensor:
    """Reject timed joint paths that cannot respect the declared velocity limits."""
    if (
        positions.ndim != 3
        or dt.shape != positions.shape[:2]
        or limits.shape != (positions.shape[0], positions.shape[2])
    ):
        raise ValueError(
            "Motion velocity checks require matching batched positions, dt and limits."
        )
    if not torch.isfinite(limits).all() or (limits < 0).any():
        raise ValueError("Joint velocity limits must be finite and non-negative.")
    delta = (positions[:, 1:] - positions[:, :-1]).abs()
    allowed = limits[:, None] * dt[:, 1:, None]
    return (
        torch.isfinite(positions).all(-1).all(-1)
        & torch.isfinite(dt).all(-1)
        & (dt >= 0).all(-1)
        & (delta <= allowed + 1e-5).all(-1).all(-1)
    )


def _cartesian_samples(
    start: torch.Tensor,
    targets: list[PlanState],
    count: int,
) -> list[PlanState]:
    """Include every supplied waypoint without changing the output time budget."""
    if count - 1 < len(targets):
        raise ValueError("Cartesian approach sampling needs one sample per target.")
    result: list[PlanState] = []
    previous = start
    remaining = count - 1
    for index, target in enumerate(targets):
        assert target.xpos is not None
        pose = target.xpos.to(device=start.device, dtype=start.dtype)
        if pose.ndim == 2:
            pose = pose.unsqueeze(0).expand(start.shape[0], -1, -1)
        if pose.shape != start.shape:
            raise ValueError("Cartesian approach targets must match the start batch.")
        intervals = remaining // (len(targets) - index)
        interpolated = interpolate_xpos_batched(previous, pose, intervals + 1)
        result.extend(
            PlanState(move_type=MoveType.EEF_MOVE, xpos=interpolated[:, point])
            for point in range(1, intervals + 1)
        )
        previous = pose
        remaining -= intervals
    return result


class ApproachMotionGenerator(CheckedMotionGenerator):
    """Sample EEF paths in Cartesian space, including single-target transports."""

    def generate(
        self,
        target_states: list[PlanState],
        options: MotionGenOptions | None = None,
    ) -> PlanResult:
        input_target_count = len(target_states)
        original_targets = target_states
        original_options = options
        input_poses = [
            state.xpos.detach().cpu().tolist()
            for state in target_states
            if state.xpos is not None
        ]

        def build_samples(
            sample_count: int,
        ) -> tuple[list[PlanState], MotionGenOptions]:
            sampled = _cartesian_samples(start, original_targets, sample_count)
            return sampled, replace(
                original_options,
                sample_count=sample_count,
                preserve_cartesian_samples=True,
                is_linear=True,
            )

        if (
            options is not None
            and options.strategy == "ik_interp"
            and not options.preserve_cartesian_samples
            and len(target_states) >= 1
            and all(state.move_type is MoveType.EEF_MOVE for state in target_states)
        ):
            if (
                options.start_qpos is None
                or options.control_part is None
                or options.sample_count is None
            ):
                raise ValueError(
                    "Approach planning requires a bound start and sample count."
                )
            start = self.robot.compute_fk(
                qpos=options.start_qpos,
                name=options.control_part,
                to_matrix=True,
            )
            target_states, options = build_samples(options.sample_count)
        result = super().generate(target_states, options=options)
        if (
            isinstance(result.success, torch.Tensor)
            and not result.success.any()
            and original_options is not None
            and original_options.strategy == "ik_interp"
            and original_options.sample_count is not None
            and original_options.sample_count <= VELOCITY_RETIME_SAMPLES[0]
            and original_options.control_part is not None
            and not original_options.preserve_cartesian_samples
            and all(state.move_type is MoveType.EEF_MOVE for state in original_targets)
        ):
            # A coarse IK interpolation can violate a real URDF/runtime limit
            # even when the geometric path is valid. Retry with denser samples
            # before reporting failure; limits remain unchanged.
            for sample_count in _velocity_retry_samples(original_options.sample_count):
                retry_targets, retry_options = build_samples(sample_count)
                retry = super().generate(retry_targets, options=retry_options)
                if isinstance(retry.success, torch.Tensor) and retry.success.any():
                    target_states, options, result = (
                        retry_targets,
                        retry_options,
                        retry,
                    )
                    logger.log_info(
                        f"GenSim adaptive velocity retime accepted sample_count={sample_count}."
                    )
                    break
            recovery = _PLACE_IK_RECOVERY.get()
            if (
                recovery is not None
                and not result.success.any()
                and options.start_qpos.shape[0] == 1
                and options.interpolation_dt is not None
                and options.interpolation_dt > 0
            ):
                retry = _local_place_ik(self, target_states, options)
                if retry is not None:
                    result = retry
                    recovery.used = True
        logger.log_info(
            "GenSim motion plan: "
            + json.dumps(
                {
                    "control_part": None if options is None else options.control_part,
                    "input_target_count": input_target_count,
                    "planned_target_count": len(target_states),
                    "input_poses": input_poses,
                    "success": (
                        result.success.detach().cpu().tolist()
                        if isinstance(result.success, torch.Tensor)
                        else result.success
                    ),
                }
            )
        )
        return result
