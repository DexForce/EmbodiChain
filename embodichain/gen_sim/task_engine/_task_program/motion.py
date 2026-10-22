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
import time
from typing import Any, Callable, Iterator
import xml.etree.ElementTree as ET

import torch
from embodichain.utils import logger

from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.compute.trajectory import interpolate_with_distance
from embodichain.lab.sim.motion.planners.utils import (
    MoveType,
    PlanResult,
    PlanState,
    interpolate_xpos_batched,
    normalize_success_mask,
)

__all__: list[str] = []

MOTION_VALIDATION_REVISION = 14
SAMPLED_MOTION_CONTRACT = "gen_sim.sampled_velocity/v1"
SHARED_MOTION_CONTRACT = "lab.motion_limits/v1"


def validate_motion_contract(value: object) -> str:
    if type(value) is not str or value not in {
        SAMPLED_MOTION_CONTRACT,
        SHARED_MOTION_CONTRACT,
    }:
        raise ValueError("Unsupported GenSim motion_contract; regenerate the bundle.")
    return value


VELOCITY_RETIME_SAMPLES = (260, 320)

_CPU_BATCH_THREADS: ContextVar[int | None] = ContextVar(
    "gen_sim_cpu_batch_threads", default=None
)


@dataclass
class _PlaceIKRecovery:
    used: bool = False
    defer_retries: bool = False


_PLACE_IK_RECOVERY: ContextVar[_PlaceIKRecovery | None] = ContextVar(
    "gen_sim_place_ik_recovery", default=None
)

_CARTESIAN_APPROACH: ContextVar[bool] = ContextVar(
    "gen_sim_cartesian_approach", default=False
)
_WAYPOINT_KINEMATICS: ContextVar[list[torch.Tensor] | None] = ContextVar(
    "gen_sim_place_waypoint_kinematics", default=None
)


@contextmanager
def cartesian_approach_scope(enabled: bool) -> Iterator[None]:
    """Select one invocation's planner behavior and restore it even on failure."""
    if type(enabled) is not bool:
        raise TypeError("Cartesian planning mode must be a bool.")
    token = _CARTESIAN_APPROACH.set(enabled)
    try:
        yield
    finally:
        _CARTESIAN_APPROACH.reset(token)


@contextmanager
def place_ik_recovery(*, defer_retries: bool = False) -> Iterator[_PlaceIKRecovery]:
    """Scope the bounded fallback to one GenSim Place planning call."""
    state = _PlaceIKRecovery(defer_retries=defer_retries)
    token = _PLACE_IK_RECOVERY.set(state)
    try:
        yield state
    finally:
        _PLACE_IK_RECOVERY.reset(token)


def resolve_continuous_ik_path(
    generator: MotionGenerator,
    targets: list[PlanState],
    options: MotionGenOptions,
    *,
    candidate_predicate: Callable[[torch.Tensor], bool] | None = None,
    operation: str = "Place",
) -> PlanResult | None:
    """Resolve one Cartesian path on a continuous joint branch.

    The resolver keeps the previous waypoint as the next IK seed, retries near
    joint limits with bounded local seed perturbations, and accepts only paths
    that satisfy the declared per-step velocity limits. It never changes the
    requested Cartesian targets or timing.
    """
    robot = generator.robot
    seed = options.start_qpos.clone()
    bounds = robot.get_qpos_limits(name=options.control_part).to(seed)
    lower, upper = bounds[..., 0], bounds[..., 1]
    limits = _joint_velocity_limits(robot, options.control_part).to(seed)
    step_dt = options.interpolation_dt

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

    # Failed searches must not perturb the later grasp-candidate random stream.
    devices = [seed.device.index] if seed.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        points = resolve_seeded_ik_path(
            targets,
            seed,
            solve=solve,
            lower=lower,
            upper=upper,
            limits=limits,
            dt=step_dt,
            candidate_predicate=candidate_predicate,
        )
    if points is None:
        return None
    path = points.unsqueeze(0)
    dt = path.new_full(path.shape[:2], step_dt)
    dt[:, 0] = 0.0
    success = _velocity_validity(path, dt, limits)
    if not success.all():
        return None
    logger.log_info(
        f"GenSim {operation} continuous IK recovery accepted: "
        f"{path.shape[1]} unchanged-time samples."
    )
    return PlanResult(success=success, positions=path, dt=dt)


def resolve_seeded_ik_path(
    poses: torch.Tensor | list[PlanState],
    start: torch.Tensor,
    *,
    solve: Callable[[Any, torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    solve_many: (
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
    pose_valid: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    lower: torch.Tensor,
    upper: torch.Tensor,
    limits: torch.Tensor,
    dt: float,
    candidate_predicate: Callable[[torch.Tensor], bool] | None = None,
) -> torch.Tensor | None:
    """Resolve a Cartesian waypoint path with one shared continuous branch.

    ``solve`` and ``solve_many`` are solver-specific adapters; the branch,
    joint-step, and joint-limit policy stays GenSim-owned and is shared by
    contact Pick paths and articulation interaction recovery.
    """
    if start.ndim != 2 or start.shape[0] != 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError(
            "Continuous IK requires one environment and a positive timestep."
        )
    if not torch.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError("Continuous IK requires finite positive velocity limits.")
    seed = start.clone()
    points = [seed]
    allowed = limits * dt
    from ._search_cost import record_search

    def feasible(ok: torch.Tensor, q: torch.Tensor) -> bool:
        return bool(
            ok.all()
            and torch.isfinite(q).all()
            and (q >= lower).all()
            and (q <= upper).all()
            and ((q - seed).abs() <= allowed + 1e-5).all()
            and (candidate_predicate is None or candidate_predicate(q))
        )

    for pose in poses:
        target = pose[None] if isinstance(pose, torch.Tensor) else pose
        record_search("ik_solve")
        ok, q = solve(target, seed)
        q = q.reshape_as(seed)
        valid = feasible(ok, q)
        margin = float(torch.minimum(q - lower, upper - q).min()) if valid else -1.0
        if margin < 0.02:
            if solve_many is None:
                best = q if valid else None
                for joint in range(seed.shape[-1]):
                    for offset in (-0.15, 0.15, -0.05, 0.05):
                        trial = seed.clone()
                        trial[:, joint] += offset
                        record_search("ik_solve")
                        good, candidate = solve(target, trial.clamp(lower, upper))
                        candidate = candidate.reshape_as(seed)
                        if feasible(good, candidate):
                            score = float(
                                torch.minimum(
                                    candidate - lower, upper - candidate
                                ).min()
                            )
                            if score > margin:
                                best, margin = candidate, score
                if best is None:
                    return None
                q = best
                seed = q
                points.append(seed)
                continue
            trials = [seed[0]]
            for offset in (-0.05, 0.05, -0.15, 0.15):
                for joint in range(seed.shape[-1]):
                    trial = seed[0].clone()
                    trial[joint] += offset
                    trials.append(trial.clamp(lower, upper).reshape(-1))
            record_search("ik_batch_solve")
            record_search("ik_seed", count=len(trials))
            good, candidates = solve_many(target, torch.stack(trials))
            if valid:
                candidates = torch.cat((q, candidates))
                good = torch.cat(
                    (torch.ones(1, dtype=torch.bool, device=good.device), good)
                )
            good &= torch.isfinite(candidates).all(-1)
            good &= ((candidates >= lower) & (candidates <= upper)).all(-1)
            good &= ((candidates - seed).abs() <= allowed + 1e-5).all(-1)
            if not good.any():
                return None
            margins = torch.minimum(candidates - lower, upper - candidates).amin(-1)
            index = torch.argmax(torch.where(good, margins, -torch.inf))
            q = candidates[index : index + 1]
        seed = q
        points.append(seed)
    result = torch.cat(points)
    if pose_valid is not None and not pose_valid(result[1:], poses).all():
        return None
    return result


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
    """Validate ordinary paths and honor the current Place's Cartesian policy."""

    def __init__(
        self, cfg: MotionGenCfg, *, motion_contract: str = SHARED_MOTION_CONTRACT
    ) -> None:
        self._motion_contract = validate_motion_contract(motion_contract)
        super().__init__(cfg)

    def _sampled_contract(self) -> bool:
        return (
            getattr(self, "_motion_contract", SHARED_MOTION_CONTRACT)
            == SAMPLED_MOTION_CONTRACT
        )

    def _generate_ik_interpolation(
        self, target_states: list[PlanState], options: MotionGenOptions
    ) -> PlanResult:
        if self._sampled_contract():
            if (
                options.velocity_limit is not None
                or options.acceleration_limit not in (None, 0.5)
            ):
                raise ValueError(
                    "Sampled GenSim cannot discard explicit motion limits; select lab.motion_limits/v1."
                )
            options = replace(options, acceleration_limit=None)
        return super()._generate_ik_interpolation(target_states, options)

    def _interpolate_ik_keyframes(
        self, keyframes: torch.Tensor, options: MotionGenOptions
    ) -> torch.Tensor:
        if self._sampled_contract():
            return interpolate_with_distance(
                trajectory=keyframes,
                interp_num=options.sample_count,
                device=keyframes.device,
            )
        return super()._interpolate_ik_keyframes(keyframes, options)

    def _interpolate_cartesian_with_limits(
        self,
        target_poses: torch.Tensor,
        solved_keyframes: torch.Tensor,
        options: MotionGenOptions,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._sampled_contract():
            # The outer checked facade rejects over-speed rows on this exact
            # grid. It does not silently change the invocation's time budget.
            valid = torch.ones(
                solved_keyframes.shape[0],
                dtype=torch.bool,
                device=solved_keyframes.device,
            )
            return valid, solved_keyframes
        return super()._interpolate_cartesian_with_limits(
            target_poses, solved_keyframes, options
        )

    def generate(
        self, target_states: list[PlanState], options: MotionGenOptions | None = None
    ) -> PlanResult:
        from .articulation_recovery import motion_scope
        from .held_alignment_recovery import (
            motion_scope as held_alignment_scope,
            screen_alignment_result,
        )

        held = held_alignment_scope(self)
        if held is not None and held.replay_result is not None:
            return held.replay(target_states, options)

        scope = motion_scope(self)
        if scope is not None and scope.tracks is not None:
            return scope.replay(target_states, options)
        if _CARTESIAN_APPROACH.get() or _PLACE_IK_RECOVERY.get() is not None:
            result = _generate_cartesian_approach(self, target_states, options)
        else:
            result = self._generate_checked(target_states, options)
        if scope is not None:
            scope.record(target_states, options)
        return screen_alignment_result(self, result, options, targets=target_states)

    def _solve_cartesian_waypoints(
        self,
        poses: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
        *,
        known_start: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .place_branch import solve_waypoints

        result = solve_waypoints(self, poses, start_qpos, control_part)
        if result is not None:
            return result
        if self._sampled_contract() and known_start is None:
            return self._solve_seeded_cartesian_waypoints(
                poses, start_qpos, control_part
            )
        if known_start is not None:
            return super()._solve_cartesian_waypoints(
                poses, start_qpos, control_part, known_start=known_start
            )
        return super()._solve_cartesian_waypoints(poses, start_qpos, control_part)

    def _solve_seeded_cartesian_waypoints(
        self,
        poses: torch.Tensor,
        start_qpos: torch.Tensor,
        control_part: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve a fixed-time path with the preceding valid posture as seed.

        This bounded-memory alternative preserves the random stream and branch
        continuity of sequential point queries. It does not perform retiming.
        """
        seed = start_qpos.to(self._runtime_device())
        batch, dof = seed.shape
        if (
            poses.ndim != 4
            or poses.shape[0] != batch
            or poses.shape[2:] != (4, 4)
            or poses.shape[1] == 0
        ):
            raise ValueError("Cartesian waypoints must have shape (B, N, 4, 4), N > 0.")
        success = torch.ones(batch, dtype=torch.bool, device=seed.device)
        points = []
        for index in range(poses.shape[1]):
            valid, qpos = self.robot.compute_ik(
                pose=poses[:, index].to(seed.device),
                name=control_part,
                joint_seed=seed,
            )
            valid = normalize_success_mask(
                valid,
                num_envs=batch,
                device=seed.device,
                name=f"IK success for target state {index}",
            )
            qpos = torch.as_tensor(qpos, dtype=seed.dtype, device=seed.device)
            if qpos.shape != (batch, dof):
                raise ValueError(f"IK qpos must have shape ({batch}, {dof}).")
            valid = valid & torch.isfinite(qpos).all(dim=1)
            seed = torch.where(valid[:, None], qpos, seed)
            success &= valid
            points.append(seed)
        positions = torch.stack(points, dim=1)
        return success, torch.where(
            success[:, None, None], positions, start_qpos.to(seed)[:, None]
        )

    def _generate_checked(
        self, target_states: list[PlanState], options: MotionGenOptions | None = None
    ) -> PlanResult:
        """Run the shared backend without re-entering task-scoped dispatch."""
        result = super().generate(target_states, options=options)
        evidence = _WAYPOINT_KINEMATICS.get()
        if evidence is not None and all(
            state.move_type is MoveType.EEF_MOVE for state in target_states
        ):
            evidence.append(torch.as_tensor(result.success).detach().clone())
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


def _generate_cartesian_approach(
    generator: CheckedMotionGenerator,
    target_states: list[PlanState],
    options: MotionGenOptions | None = None,
) -> PlanResult:
    """Share one policy between Place-scoped and legacy Cartesian requests."""
    started = time.perf_counter()
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
        start = generator.robot.compute_fk(
            qpos=options.start_qpos,
            name=options.control_part,
            to_matrix=True,
        )
        target_states, options = build_samples(options.sample_count)
    result = generator._generate_checked(target_states, options=options)
    recovery = _PLACE_IK_RECOVERY.get()
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
        and (recovery is None or not recovery.defer_retries)
    ):
        # A coarse IK interpolation can violate a real URDF/runtime limit
        # even when the geometric path is valid. Retry with denser samples
        # before reporting failure; limits remain unchanged.
        for sample_count in _velocity_retry_samples(original_options.sample_count):
            retry_targets, retry_options = build_samples(sample_count)
            retry = generator._generate_checked(retry_targets, options=retry_options)
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
        if (
            recovery is not None
            and not result.success.any()
            and options.start_qpos.shape[0] == 1
            and options.interpolation_dt is not None
            and options.interpolation_dt > 0
        ):
            retry = resolve_continuous_ik_path(generator, target_states, options)
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
                "elapsed_seconds": round(time.perf_counter() - started, 3),
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


class ApproachMotionGenerator(CheckedMotionGenerator):
    """Sample EEF paths in Cartesian space, including single-target transports."""

    def generate(
        self,
        target_states: list[PlanState],
        options: MotionGenOptions | None = None,
    ) -> PlanResult:
        return _generate_cartesian_approach(self, target_states, options)
