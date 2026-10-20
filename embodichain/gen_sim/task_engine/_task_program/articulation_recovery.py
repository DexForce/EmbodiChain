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

"""Failure-only GenSim E6 planning scopes; shared skills still assemble plans."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import random
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.atomic_actions.goals import resolve_pose_goal
from embodichain.lab.sim.atomic_actions.plans import ActionPlan, PlanningFailure
from embodichain.lab.sim.atomic_actions.primitives.slide import Slide
from embodichain.lab.sim.motion.motion_generator import MotionGenOptions
from embodichain.lab.sim.motion.planners.utils import (
    PlanResult,
    PlanState,
    MoveType,
    interpolate_xpos_batched,
)
from embodichain.lab.sim.motion.solvers import PytorchSolver
from embodichain.lab.sim.motion.solvers.qpos_seed_sampler import QposSeedSampler
from embodichain.lab.sim.sim_manager import SimulationManager
from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

from .articulation_collision import check_free_motion, check_robot_motion
from .motion import (
    _cartesian_samples,
    _joint_velocity_limits,
    _velocity_validity,
    resolve_seeded_ik_path,
)

__all__: list[str] = []
ARTICULATION_RECOVERY_REVISION = 5
MAX_SLIDE_FRAMES = 390
MAX_ENDPOINTS = 4


class _ReplayMismatch(ValueError):
    """The freshly grounded shared skill no longer matches the recovery targets."""


@dataclass
class MotionCall:
    targets: list[PlanState]
    options: MotionGenOptions


@dataclass
class PlanningScope:
    generator: Any
    kind: str
    withdrawal_samples: int | None = None
    calls: list[MotionCall] = field(default_factory=list)
    tracks: tuple[torch.Tensor, ...] | None = None
    cursor: int = 0
    search: Any = None
    compatible: bool = True

    def record(
        self, targets: list[PlanState], options: MotionGenOptions | None
    ) -> None:
        """Snapshot supported calls without changing unsupported normal plans."""
        if (
            options is None
            or options.start_qpos is None
            or not targets
            or any(s.move_type is not MoveType.EEF_MOVE for s in targets)
        ):
            self.compatible = False
            return
        self.calls.append(
            MotionCall(
                [replace(s, xpos=s.xpos.clone()) for s in targets],
                replace(options, start_qpos=options.start_qpos.clone()),
            )
        )

    def replay(
        self, targets: list[PlanState], options: MotionGenOptions | None
    ) -> PlanResult:
        """Supply a verified subpath only when fresh grounding still matches."""
        if (
            options is None
            or not targets
            or self.tracks is None
            or self.cursor >= len(self.tracks)
        ):
            raise _ReplayMismatch("Unexpected motion call during E6 recovery.")
        index = self.cursor
        track = self.tracks[index]
        original = self.calls[index]
        if (
            options.control_part != original.options.control_part
            or options.sample_count != len(track)
            or options.interpolation_dt != original.options.interpolation_dt
            or not torch.allclose(options.start_qpos, track[:1], atol=1e-5, rtol=0)
            or not torch.allclose(
                targets[-1].xpos, original.targets[-1].xpos, atol=1e-5, rtol=0
            )
        ):
            raise _ReplayMismatch("E6 target, start or timing changed during recovery.")
        if self.kind == "slide" and index == 0:
            wanted = targets[-1].xpos
            checked = track[-1:]
        else:
            samples = (
                targets
                if options.preserve_cartesian_samples
                else _cartesian_samples(
                    self.generator.robot.compute_fk(
                        qpos=options.start_qpos,
                        name=options.control_part,
                        to_matrix=True,
                    ),
                    targets,
                    len(track),
                )
            )
            wanted = torch.cat([s.xpos for s in samples])
            checked = track[1:]
        if not self.search.pose_valid(checked, self.search.to_root(wanted)).all():
            raise _ReplayMismatch("Rebuilt E6 samples do not match the grounded path.")
        self.cursor += 1
        dt = track.new_full((1, len(track)), options.interpolation_dt)
        dt[:, 0] = 0
        return PlanResult(
            success=torch.ones(1, dtype=torch.bool, device=track.device),
            positions=track[None],
            dt=dt,
        )


_SCOPE: ContextVar[PlanningScope | None] = ContextVar(
    "gen_sim_articulation_recovery", default=None
)


def motion_scope(generator: Any) -> PlanningScope | None:
    scope = _SCOPE.get()
    return scope if scope is not None and scope.generator is generator else None


@contextmanager
def planning_scope(
    generator: Any, kind: str, *, withdrawal_samples: int | None = None
) -> Iterator[PlanningScope]:
    scope = PlanningScope(generator, kind, withdrawal_samples=withdrawal_samples)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def slide_budget(original: int) -> int | None:
    """Do not shorten larger caller budgets or exceed the agreed fallback cap."""
    if original > MAX_SLIDE_FRAMES:
        return None
    return min(MAX_SLIDE_FRAMES, original * 3 // 2)


class _Search:
    """Bounded single-environment CPU PytorchSolver continuation, without stepping."""

    def __init__(self, robot: Any, part: str, dt: float) -> None:
        self.robot, self.part, self.dt = robot, part, dt
        self.solver = robot.get_solver(part)
        self.pos_eps, self.rot_eps = self.solver._pos_eps, self.solver._rot_eps
        self.lower, self.upper = (
            self.solver.lower_qpos_limits,
            self.solver.upper_qpos_limits,
        )
        self.limits = _joint_velocity_limits(robot, part)[0]
        self.inverse = torch.linalg.inv(
            robot.get_link_pose(self.solver.root_link_name, to_matrix=True)
        )
        self.chain = create_pk_serial_chain(
            urdf_path=self.solver.urdf_path,
            root_link_name=self.solver.root_link_name,
            end_link_name=self.solver.end_link_name,
            device=torch.device("cpu"),
        ).to(dtype=torch.float64)

    def to_root(self, poses: torch.Tensor) -> torch.Tensor:
        return self.inverse @ poses

    def pose_valid(self, q: torch.Tensor, poses: torch.Tensor) -> torch.Tensor:
        fk = self.chain.forward_kinematics_tensor(q.double())[-1] @ torch.as_tensor(
            self.solver.tcp_xpos, dtype=torch.float64
        )
        position = (fk[:, :3, 3] - poses[:, :3, 3].double()).norm(dim=-1)
        rotation = fk[:, :3, :3] @ poses[:, :3, :3].double().transpose(-1, -2)
        skew = torch.stack(
            (
                rotation[:, 2, 1] - rotation[:, 1, 2],
                rotation[:, 0, 2] - rotation[:, 2, 0],
                rotation[:, 1, 0] - rotation[:, 0, 1],
            ),
            dim=-1,
        )
        angle = torch.atan2(
            skew.norm(dim=-1) / 2, (rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        )
        return (position <= self.pos_eps) & (angle <= self.rot_eps)

    def solve_seeds(
        self, pose: torch.Tensor, seeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Preserve per-seed validity; public return_all_solutions exposes only a row mask.
        flange = pose @ torch.linalg.inv(
            torch.as_tensor(self.solver.tcp_xpos, dtype=pose.dtype)
        )
        ok, q = self.solver._compute_inverse_kinematics(
            flange.expand(len(seeds), -1, -1), seeds
        )
        valid, q = self.solver._qpos_map_to_limits(q.reshape(-1, seeds.shape[-1]))
        return ok.reshape(-1) & valid & self.pose_valid(q, pose), q

    def path(self, poses: torch.Tensor, start: torch.Tensor) -> torch.Tensor | None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            return resolve_seeded_ik_path(
                poses,
                start,
                solve=lambda pose, seed: self.solver.get_ik(pose, qpos_seed=seed),
                solve_many=self.solve_seeds,
                pose_valid=self.pose_valid,
                lower=self.lower,
                upper=self.upper,
                limits=self.limits,
                dt=self.dt,
            )

    def endpoints(
        self, pose: torch.Tensor, start: torch.Tensor
    ) -> list[tuple[int, torch.Tensor]]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            seeds = QposSeedSampler(32, start.shape[-1], start.device).sample(
                start, self.lower, self.upper, 1
            )
            valid, q = self.solve_seeds(pose, seeds)
        ids = valid.nonzero().flatten().tolist()
        chosen = [min(ids, key=lambda i: float((q[i] - start).norm()))] if ids else []
        while len(chosen) < min(MAX_ENDPOINTS, len(ids)):
            remaining = [i for i in ids if i not in chosen]
            index = max(
                remaining, key=lambda i: float((q[chosen] - q[i]).norm(dim=-1).min())
            )
            if float((q[chosen] - q[index]).norm(dim=-1).min()) < 0.02:
                break
            chosen.append(index)
        return [(i, q[i : i + 1]) for i in chosen]


def _tracks(
    scope: PlanningScope, request: Any, search: _Search
) -> Iterator[tuple[Any, tuple[torch.Tensor, ...], dict[str, Any]]]:
    calls = scope.calls
    start = calls[0].options.start_qpos
    if scope.kind == "withdraw":
        options = calls[0].options
        poses = _cartesian_samples(
            search.robot.compute_fk(qpos=start, name=search.part, to_matrix=True),
            calls[0].targets,
            options.sample_count,
        )
        path = search.path(search.to_root(torch.cat([p.xpos for p in poses])), start)
        if path is not None:
            yield request, (path,), {"method": "local_ik", "frames": len(path)}
        return
    if len(calls) == 5:
        yield from _push_tracks(scope, request, search)
        return
    budget = slide_budget(request.motion_policy.sample_count)
    if budget is None:
        return
    hand = request.skill_options.hand_interp_steps
    lengths = Slide._motion_segment_lengths(budget, hand, direction="pull")
    approach, grasp, pull, release = [search.to_root(c.targets[-1].xpos) for c in calls]

    def line(a: torch.Tensor, b: torch.Tensor, count: int) -> torch.Tensor:
        return interpolate_xpos_batched(a, b, count)[0, 1:]

    for index, endpoint in search.endpoints(pull, start):
        released = search.path(line(pull, release, hand), endpoint)
        if released is None:
            continue
        pulled = search.path(line(pull, grasp, lengths[2]), endpoint)
        if pulled is None:
            continue
        reached = search.path(line(grasp, approach, lengths[1]), pulled[-1:])
        if reached is None:
            continue
        approached = start + torch.linspace(0, 1, lengths[0])[:, None] * (
            reached[-1:] - start
        )
        selected = replace(
            request, motion_policy=replace(request.motion_policy, sample_count=budget)
        )
        yield selected, (approached, reached.flip(0), pulled.flip(0), released), {
            "method": "terminal_first",
            "frames": budget,
            "endpoint_seed_index": index,
        }


def _push_tracks(
    scope: PlanningScope, request: Any, search: _Search
) -> Iterator[tuple[Any, tuple[torch.Tensor, ...], dict[str, Any]]]:
    """Connect a free-space approach to one continuous, unchanged-budget push.

    Grasp-first seeds keep the reach and push on one branch. All five original
    targets and sample counts are retained; the shared Slide rebuild owns hand
    commands, effects and execution. No scene motion is predicted here.
    """
    calls = scope.calls
    start = calls[0].options.start_qpos
    approach, grasp, pushed, release, returned = [
        search.to_root(call.targets[-1].xpos) for call in calls
    ]
    counts = [call.options.sample_count for call in calls]

    def line(a: torch.Tensor, b: torch.Tensor, count: int) -> torch.Tensor:
        return interpolate_xpos_batched(a, b, count)[0, 1:]

    for index, endpoint in search.endpoints(grasp, start):
        reached_back = search.path(line(grasp, approach, counts[1]), endpoint)
        if reached_back is None:
            continue
        pushed_path = search.path(line(grasp, pushed, counts[2]), endpoint)
        if pushed_path is None:
            continue
        released = search.path(line(pushed, release, counts[3]), pushed_path[-1:])
        if released is None:
            continue
        return_path = search.path(line(release, returned, counts[4]), released[-1:])
        if return_path is None:
            continue
        approached = start + torch.linspace(0, 1, counts[0])[:, None] * (
            reached_back[-1:] - start
        )
        yield request, (
            approached,
            reached_back.flip(0),
            pushed_path,
            released,
            return_path,
        ), {
            "method": "push_grasp_first",
            "frames": request.motion_policy.sample_count,
            "endpoint_seed_index": index,
        }


def _preview_withdrawal(
    scope: PlanningScope,
    request: Any,
    context: Any,
    search: _Search,
    trajectory: Any,
) -> dict[str, Any]:
    """Reject terminal branches without an unchanged-budget cleanup continuation.

    The articulation and its contents have not moved yet, so scene clearance is
    deliberately deferred to the fresh actual withdrawal plan. No preview path
    is cached or dispatched.
    """
    from .articulation_slide import _withdrawal_targets
    from .held_alignment_recovery import _RandomState

    saved = _RandomState.capture(set())
    try:
        terminal = trajectory.positions[:, -1]
        ids = list(search.robot.get_joint_ids(name=search.part))
        arm = terminal[:, ids]
        current = search.robot.compute_fk(qpos=arm, name=search.part, to_matrix=True)
        link = resolve_pose_goal(request.goal.target_pose, context, name="E6 link")
        axis = request.goal.semantics.affordance.translation_axis.to(current)
        outward = -(link[:, :3, :3] @ axis)
        targets = [
            PlanState(move_type=MoveType.EEF_MOVE, xpos=pose)
            for pose in _withdrawal_targets(current, outward).unbind(dim=1)
        ]
        samples = _cartesian_samples(current, targets, scope.withdrawal_samples)
        path = search.path(search.to_root(torch.cat([p.xpos for p in samples])), arm)
        evidence = {
            "scope": "kinematics_and_robot_meshes_not_future_scene",
            "samples": scope.withdrawal_samples,
            "valid": False,
        }
        if path is None:
            return {**evidence, "reason": "no_continuous_ik"}
        full = terminal.expand(len(path), -1).clone()
        full[:, ids] = path
        dt = full.new_full((1, len(path)), search.dt)
        dt[:, 0] = 0
        if not _velocity_validity(
            full[None], dt, _joint_velocity_limits(search.robot, None).to(full)
        ).all():
            return {**evidence, "reason": "final_command_velocity"}
        collision = check_robot_motion(search.robot, full, search.part)
        return {**evidence, "valid": collision["valid"], "robot_collision": collision}
    finally:
        saved.restore()


def _screen_withdrawal(scope: PlanningScope, plan: ActionPlan) -> ActionPlan:
    """Require fresh actual-scene clearance; lookahead is not a safety certificate."""
    try:
        trajectory = plan.joint_trajectory
        if len(scope.calls) != 1 or trajectory is None:
            raise ValueError("Withdrawal requires one complete motion trajectory.")
        robot = scope.generator.robot
        limits = _joint_velocity_limits(robot, None).to(trajectory.positions)
        if not _velocity_validity(trajectory.positions, trajectory.dt, limits).all():
            raise ValueError("Withdrawal exceeds final command velocity limits.")
        simulation = SimulationManager.get_instance(
            scope.generator.planner.cfg.sim_instance_id
        )
        evidence = check_free_motion(
            robot,
            simulation,
            trajectory.positions[0],
            scope.calls[0].options.control_part,
        )
    except (ImportError, ValueError) as error:
        evidence = {"valid": False, "reason": f"{type(error).__name__}: {error}"}
    return replace(
        plan,
        plan_success=(
            plan.plan_success
            if evidence["valid"]
            else torch.zeros_like(plan.plan_success)
        ),
        diagnostics=replace(
            plan.diagnostics,
            failure=(
                plan.diagnostics.failure
                if evidence["valid"]
                else PlanningFailure(
                    "gen_sim_withdrawal_clearance_failed", retryable=False
                )
            ),
            metadata={
                **plan.diagnostics.metadata,
                "gen_sim_withdrawal_clearance": evidence,
            },
        ),
    )


def recover(
    scope: PlanningScope,
    request: Any,
    context: Any,
    original: ActionPlan,
    plan: Callable[[Any], ActionPlan],
) -> ActionPlan:
    """Replay only a fully screened recovery through the unchanged shared skill."""
    if (
        scope.kind == "withdraw"
        and scope.withdrawal_samples is not None
        and original.plan_success.all()
    ):
        original = _screen_withdrawal(scope, original)
    if original.plan_success.any() or not scope.compatible or not scope.calls:
        return original
    push = scope.kind == "slide" and any(s.name == "push" for s in original.segments)
    expected = (5 if push else 4) if scope.kind == "slide" else 1
    if len(scope.calls) != expected or context.batch_size != 1:
        return original
    if (
        scope.kind == "withdraw"
        and scope.calls[0].options.sample_count > MAX_SLIDE_FRAMES
    ):
        return original
    if scope.kind == "slide" and (
        request.skill_options.release_retreat_distance <= 0
        or not any(s.name == ("push" if push else "pull") for s in original.segments)
        or slide_budget(request.motion_policy.sample_count) is None
    ):
        return original
    robot = scope.generator.robot
    part = scope.calls[0].options.control_part
    solver = robot.get_solver(part)
    if (
        not isinstance(solver, PytorchSolver)
        or solver.device.type != "cpu"
        or solver._seed_sampler is not None
        or robot.get_qpos().shape[0] != 1
        or any(c.options.control_part != part for c in scope.calls)
    ):
        return original
    details: dict[str, Any] = {"kind": scope.kind, "status": "exhausted"}
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            search = _Search(robot, part, context.require_control_dt())
            scope.search = search
            for selected, tracks, metadata in _tracks(scope, request, search):
                scope.tracks, scope.cursor = tracks, 0
                try:
                    recovered = plan(selected)
                except _ReplayMismatch as error:
                    details["reason"] = str(error)
                    continue
                trajectory = recovered.joint_trajectory
                if (
                    not recovered.plan_success.all()
                    or scope.cursor != len(tracks)
                    or trajectory is None
                ):
                    continue
                if trajectory.positions.shape[1] > min(
                    MAX_SLIDE_FRAMES, selected.motion_policy.sample_count
                ):
                    details["reason"] = "final_command_frame_budget"
                    continue
                limits = _joint_velocity_limits(robot, None).to(trajectory.positions)
                if not _velocity_validity(
                    trajectory.positions, trajectory.dt, limits
                ).all():
                    details["reason"] = "final_command_velocity"
                    continue
                if push:
                    robot_collision = check_robot_motion(
                        robot, trajectory.positions[0], part
                    )
                    if not robot_collision["valid"]:
                        details["robot_collision"] = robot_collision
                        continue
                count = (
                    len(tracks[0])
                    if scope.kind == "slide"
                    else trajectory.positions.shape[1]
                )
                simulation = SimulationManager.get_instance(
                    scope.generator.planner.cfg.sim_instance_id
                )
                collision = check_free_motion(
                    robot, simulation, trajectory.positions[0, :count], part
                )
                if not collision["valid"]:
                    details["collision"] = collision
                    continue
                if scope.kind == "slide" and scope.withdrawal_samples is not None:
                    preview = _preview_withdrawal(
                        scope, request, context, search, trajectory
                    )
                    details.setdefault("withdrawal_candidates", []).append(
                        {
                            "endpoint_seed_index": metadata["endpoint_seed_index"],
                            **preview,
                        }
                    )
                    if not preview["valid"]:
                        continue
                    metadata = {
                        **metadata,
                        "withdrawal_preview": preview,
                        "withdrawal_candidates": details["withdrawal_candidates"],
                        "withdrawal_selection_changed": len(
                            details["withdrawal_candidates"]
                        )
                        > 1,
                    }
                return replace(
                    recovered,
                    diagnostics=replace(
                        recovered.diagnostics,
                        metadata={
                            **recovered.diagnostics.metadata,
                            "gen_sim_articulation_recovery": {
                                **metadata,
                                "kind": scope.kind,
                                "status": "accepted",
                                "collision": collision,
                            },
                        },
                    ),
                )
    except (ImportError, ValueError) as error:
        details["reason"] = f"{type(error).__name__}: {error}"
    finally:
        scope.tracks = None
        scope.search = None
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return replace(
        original,
        diagnostics=replace(
            original.diagnostics,
            metadata={
                **original.diagnostics.metadata,
                "gen_sim_articulation_recovery": details,
            },
        ),
    )
