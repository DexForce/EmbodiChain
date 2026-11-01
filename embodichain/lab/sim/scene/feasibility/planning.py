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

"""Bounded full-joint branch planning for single and synchronized dual arms."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import time
from typing import Protocol, TYPE_CHECKING

import numpy as np
import torch

from .budget import FeasibilityBudget, FeasibilityBudgetExhausted
from .cfg import SceneRobotFeasibilityCfg
from .collision import FCLCollisionChecker, UnsupportedFeasibilityError
from .contracts import FeasibilityCheck, TaskPhase, TaskOperationMode

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = [
    "TaskPlanningBackend",
    "TaskPlan",
    "plan_task_mode",
    "RobotFeasibilityBackend",
]


class TaskPlanningBackend(Protocol):
    """Full-state solver port; each target set represents simultaneous resources."""

    initial_qpos: np.ndarray
    joint_limits: np.ndarray
    joint_names: tuple[str, ...]

    def solutions(
        self,
        phase: TaskPhase,
        start: np.ndarray,
        entities: Mapping[str, np.ndarray],
        budget: FeasibilityBudget,
    ) -> Sequence[np.ndarray]:
        """Return bounded complete-joint configurations for all phase targets."""
        ...

    def connect(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        phase: TaskPhase,
        entities: Mapping[str, np.ndarray],
        budget: FeasibilityBudget,
    ) -> np.ndarray | None:
        """Return a collision-checked full-joint path or no solution in budget."""
        ...

    def resulting_poses(
        self, qpos: np.ndarray, phase: TaskPhase, entities: Mapping[str, np.ndarray]
    ) -> Mapping[str, np.ndarray]:
        """Update projected planning state, never measured task evidence."""
        ...


@dataclass(frozen=True)
class TaskPlan:
    """One complete branch with stage-indexed paths; no physical-success claim."""

    checks: tuple[FeasibilityCheck, ...]
    paths: tuple[np.ndarray, ...] = ()
    phase_ids: tuple[str, ...] = ()
    clearance_m: float = 0.0

    def __post_init__(self) -> None:
        values = []
        for path in self.paths:
            value = np.array(path, dtype=float, copy=True)
            if value.ndim != 2 or len(value) < 2 or not np.isfinite(value).all():
                raise ValueError(
                    "Task paths must be finite full-joint arrays with endpoints"
                )
            value.setflags(write=False)
            values.append(value)
        object.__setattr__(self, "paths", tuple(values))
        object.__setattr__(self, "phase_ids", tuple(self.phase_ids))
        object.__setattr__(self, "checks", tuple(self.checks))
        if len(self.paths) != len(self.phase_ids) or len(set(self.phase_ids)) != len(
            self.phase_ids
        ):
            raise ValueError("Task paths require unique matching phase IDs")

    @property
    def accepted(self) -> bool:
        """Return whether all stages have a complete connected solution."""
        return (
            bool(self.paths)
            and {"kinematics", "motion"} <= {c.stage for c in self.checks}
            and all(c.status == "passed" for c in self.checks)
        )


def plan_task_mode(
    mode: TaskOperationMode,
    backend: TaskPlanningBackend,
    entity_poses: Mapping[str, object],
    budget: FeasibilityBudget,
) -> TaskPlan:
    """Connect compatible IK branches across an explicitly ordered task graph.

    Args:
        mode: Task-preserving ordered phases; synchronized targets stay together.
        backend: Full-joint state, geometry and kinematic implementation.
        entity_poses: Actual initial arena poses.
        budget: Shared call/deadline and beam limits.
    Returns:
        A complete plan, or the first unresolved stage with its true failure scope.
    """
    initial = np.array(backend.initial_qpos, dtype=float, copy=True)
    limits = np.asarray(backend.joint_limits)
    if (
        initial.shape != (len(backend.joint_names),)
        or limits.shape != (len(initial), 2)
        or not np.isfinite(initial).all()
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] > limits[:, 1])
    ):
        raise ValueError("backend requires finite full-joint state/limits")
    if np.any(initial < limits[:, 0] - 1e-4) or np.any(initial > limits[:, 1] + 1e-4):
        return TaskPlan(
            (
                FeasibilityCheck(
                    "kinematics",
                    "failed",
                    "Measured initial joints exceed limits beyond the 1e-4 allowance",
                ),
            )
        )
    beam = [
        (
            initial,
            {k: np.array(v, dtype=float, copy=True) for k, v in entity_poses.items()},
            (),
            0.0,
        )
    ]
    for phase in mode.phases:
        next_beam, solved = [], False
        for start, entities, paths, cost in beam:
            budget.check()
            for goal in backend.solutions(phase, start, entities, budget):
                goal = np.asarray(goal, dtype=float)
                if goal.shape != initial.shape or not np.isfinite(goal).all():
                    raise ValueError("IK backend returned malformed full-joint state")
                if np.any(goal < limits[:, 0]) or np.any(goal > limits[:, 1]):
                    continue
                solved = True
                path = backend.connect(start, goal, phase, entities, budget)
                if path is None:
                    continue
                path = np.asarray(path, dtype=float)
                if (
                    path.ndim != 2
                    or path.shape[1:] != initial.shape
                    or len(path) < 2
                    or not np.isfinite(path).all()
                    or not np.allclose(path[0], start, atol=1e-7, rtol=0)
                    or not np.allclose(path[-1], goal, atol=1e-7, rtol=0)
                    or np.any(path[0] < limits[:, 0] - 1e-4)
                    or np.any(path[0] > limits[:, 1] + 1e-4)
                    or np.any(path[1:] < limits[:, 0])
                    or np.any(path[1:] > limits[:, 1])
                ):
                    raise ValueError(
                        "planner must return a finite endpoint-preserving full-joint path"
                    )
                updated = backend.resulting_poses(goal, phase, entities)
                length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
                next_beam.append(
                    (goal.copy(), dict(updated), (*paths, path.copy()), cost + length)
                )
        if not next_beam:
            stage = "motion" if solved else "kinematics"
            collision = getattr(backend, "last_collision", None)
            detail = "No complete connection found within the declared branch/planning budget"
            if collision is not None and collision.status != "passed":
                detail += ": " + collision.reason
            return TaskPlan(
                (FeasibilityCheck(stage, "no_solution", detail, phase.phase_id),)
            )
        # Retain distinct joint branches rather than duplicate solutions from seeds.
        beam = []
        for row in sorted(next_beam, key=lambda item: item[3]):
            if not any(
                np.allclose(row[0], existing[0], atol=1e-5, rtol=0) for existing in beam
            ):
                beam.append(row)
            if len(beam) >= budget.cfg.beam_width:
                break
    best = min(beam, key=lambda row: row[3])
    return TaskPlan(
        (
            FeasibilityCheck("kinematics", "passed"),
            FeasibilityCheck(
                "motion",
                "passed",
                "Complete synchronized discrete collision-checked path",
            ),
        ),
        best[2],
        tuple(p.phase_id for p in mode.phases),
    )


class RobotFeasibilityBackend:
    """Live B=1 Robot adapter with multi-seed IK and full-joint RRT fallback.

    Args:
        robot: Prepared robot at the candidate's actual installation.
        collision: Complete physical collision geometry including both arms.
        cfg: Bounded branch and discretization settings.
        entity_poses: Settled arena poses for static and movable scene entities.
        seed: Candidate-local planner randomness.

    The fallback plans in full joint coordinates, with the same collision checks
    for straight connections and sampled detours. It does not replace runtime
    motion policies; actual Task Program execution replans from fresh state.
    """

    def __init__(
        self,
        robot: Robot,
        collision: FCLCollisionChecker,
        cfg: SceneRobotFeasibilityCfg,
        entity_poses: Mapping[str, object],
        *,
        seed: int = 0,
    ) -> None:
        if robot.get_qpos().shape[0] != 1 or robot.pk_chain is None:
            raise UnsupportedFeasibilityError(
                "Robot planning requires B=1 and full-link kinematics"
            )
        self.robot, self.collision, self.cfg = robot, collision, cfg
        self.joint_names = tuple(robot.joint_names)
        self.initial_qpos = robot.get_qpos()[0].detach().cpu().double().numpy().copy()
        self.joint_limits = (
            robot.get_qpos_limits()[0].detach().cpu().double().numpy().copy()
        )
        self._root = (
            robot.get_local_pose(to_matrix=True)[0].detach().cpu().double().numpy()
        )
        self._entities = {k: np.asarray(v) for k, v in entity_poses.items()}
        self._rng = np.random.default_rng(seed)
        self.last_collision = None
        self._parts = {}
        self.initial_qpos = self._project_mimic(self.initial_qpos)
        self._chain_indices = [
            self.joint_names.index(n)
            for n in robot.pk_chain.get_joint_parameter_names()
        ]

    def _project_mimic(self, qpos: np.ndarray) -> np.ndarray:
        result = qpos.copy()
        for child, parent, multiplier, offset in zip(
            self.robot.mimic_ids,
            self.robot.mimic_parents,
            self.robot.mimic_multipliers,
            self.robot.mimic_offsets,
            strict=True,
        ):
            result[child] = result[parent] * multiplier + offset
        return result

    def _indices(self, part: str) -> list[int]:
        if part not in self._parts:
            if robot_solver := self.robot.get_solver(name=part):
                indices = self.robot.get_joint_ids(name=part)
                if len(indices) != robot_solver.dof:
                    raise UnsupportedFeasibilityError(
                        "Solver/control-part joint width mismatch"
                    )
                self._parts[part] = list(indices)
            else:
                raise UnsupportedFeasibilityError(f"No IK solver for {part}")
        return self._parts[part]

    def solutions(
        self,
        phase: TaskPhase,
        start: np.ndarray,
        entities: Mapping[str, np.ndarray],
        budget: FeasibilityBudget,
    ) -> Sequence[np.ndarray]:
        """Solve all simultaneous TCPs, retaining distinct full-state branches."""
        if len(phase.targets) > 1:
            # Coupled roots (e.g. torso + arm) need per-branch transforms. Try
            # bounded target orderings rather than solving all roots at live q.
            from itertools import islice, permutations

            joint_ids = {
                self.joint_names.index(name) for name, _ in phase.joint_targets
            }
            occupied = set()
            for target in phase.targets:
                indices = set(self._indices(target.control_part))
                if indices & (occupied | joint_ids):
                    raise UnsupportedFeasibilityError(
                        "Simultaneous targets require disjoint explicitly commanded joints"
                    )
                occupied |= indices
            output = []
            for ordering in islice(permutations(phase.targets), self.cfg.beam_width):
                rows = [start.copy()]
                for index, target in enumerate(ordering):
                    partial = replace(
                        phase,
                        targets=(target,),
                        joint_targets=phase.joint_targets if index == 0 else (),
                    )
                    rows = [
                        solution
                        for row in rows
                        for solution in self.solutions(partial, row, entities, budget)
                    ][: self.cfg.beam_width]
                    if not rows:
                        break
                output.extend(
                    row for row in rows if self._targets_valid(row, phase, entities)
                )
                if len(output) >= self.cfg.beam_width:
                    break
            return output[: self.cfg.beam_width]
        if np.any(start < self.joint_limits[:, 0] - 1e-4) or np.any(
            start > self.joint_limits[:, 1] + 1e-4
        ):
            return ()
        commanded = np.clip(start, self.joint_limits[:, 0], self.joint_limits[:, 1])
        commanded_indices = set()
        for name, value in phase.joint_targets:
            if name not in self.joint_names:
                raise ValueError(f"Unknown joint target {name}")
            index = self.joint_names.index(name)
            if index in self.robot.mimic_ids:
                raise ValueError("Explicit mimic targets must command the leader joint")
            commanded[index] = value
            commanded_indices.add(index)
        commanded = self._project_mimic(commanded)
        rows = [commanded]
        occupied = set()
        for target in phase.targets:
            indices = self._indices(target.control_part)
            if occupied.intersection(indices):
                raise UnsupportedFeasibilityError(
                    "Simultaneous overlapping control parts require a coupled IK solver"
                )
            if commanded_indices.intersection(indices):
                raise ValueError(
                    "TCP targets and explicit joint targets cannot command the same joints"
                )
            occupied.update(indices)
            pose = (
                np.array(target.pose)
                if target.frame_uid is None
                else np.asarray(entities[target.frame_uid]) @ target.pose
            )
            solver = self.robot.get_solver(name=target.control_part)
            live_root = (
                self.robot.get_link_pose(
                    link_name=solver.root_link_name, env_ids=[0], to_matrix=True
                )[0]
                .detach()
                .cpu()
                .double()
                .numpy()
            )
            hypothetical = self.frame_poses(commanded, TaskPhase("ik_root"), entities)[
                f"{self.robot.uid}/{solver.root_link_name}"
            ]
            live_target = live_root @ np.linalg.inv(hypothetical) @ pose
            seeds = [commanded[indices]]
            workspace_cfg = self.robot.cfg.workspace_cfg or {}
            if target.control_part in workspace_cfg:
                budget.check()
                samples = self.robot.sample_reachable_pose(
                    name=target.control_part,
                    env_ids=[0],
                    num_samples=self.cfg.ik_seeds,
                    max_attempts=self.cfg.ik_seeds,
                    generator=torch.Generator(device=self.robot.device).manual_seed(
                        int(self._rng.integers(2**31))
                    ),
                )
                selected = samples.valid[0].nonzero(as_tuple=False).flatten()
                if samples.score is not None:
                    scores = samples.score[0, selected]
                    if not bool(torch.isfinite(scores).all()):
                        raise ValueError(
                            "Valid workspace samples contain nonfinite scores"
                        )
                    selected = selected[
                        torch.argsort(scores, descending=True, stable=True)
                    ]
                cached = samples.qpos[0, selected]
                if not bool(torch.isfinite(cached).all()):
                    raise ValueError("Valid workspace samples contain nonfinite joints")
                seeds.extend(cached.detach().cpu().numpy())
            limits = self.joint_limits[indices]
            while len(seeds) < self.cfg.ik_seeds:
                seeds.append(self._rng.uniform(limits[:, 0], limits[:, 1]))
            solutions = []
            for seed in seeds[: self.cfg.ik_seeds]:
                budget.charge("ik")
                result = self.robot.compute_ik(
                    pose=torch.tensor(
                        live_target[None], dtype=torch.float32, device=self.robot.device
                    ),
                    joint_seed=torch.tensor(
                        np.asarray(seed)[None],
                        dtype=torch.float32,
                        device=self.robot.device,
                    ),
                    name=target.control_part,
                    env_ids=[0],
                    return_all_solutions=True,
                )
                budget.check()
                if result is None:
                    raise UnsupportedFeasibilityError("IK solver unavailable")
                mask, q = result
                q = q.detach().cpu().double().numpy()
                valid = mask.detach().cpu().numpy()
                if (
                    q.ndim not in (2, 3)
                    or q.shape[0] != 1
                    or q.shape[-1] != len(indices)
                    or valid.dtype != np.bool_
                ):
                    raise ValueError("IK result shape/type mismatch")
                candidates = q.reshape(-1, len(indices))
                if valid.size == 1:
                    valid = np.full(len(candidates), bool(valid.reshape(-1)[0]))
                elif valid.shape == q.shape[:-1]:
                    valid = valid.reshape(-1)
                else:
                    raise ValueError(
                        "IK validity mask does not match returned branches"
                    )
                valid &= (
                    np.isfinite(candidates).all(axis=1)
                    & (candidates >= limits[:, 0]).all(axis=1)
                    & (candidates <= limits[:, 1]).all(axis=1)
                )
                indices_to_check = np.flatnonzero(valid)
                indices_to_check = indices_to_check[
                    np.argsort(
                        np.linalg.norm(candidates[indices_to_check] - seed, axis=1),
                        kind="stable",
                    )
                ]
                for index in indices_to_check[
                    : max(32, self.cfg.beam_width * self.cfg.ik_seeds)
                ]:
                    budget.check()
                    q = candidates[index]
                    if any(np.allclose(q, old, atol=1e-5, rtol=0) for old in solutions):
                        continue
                    fk = (
                        self.robot.compute_fk(
                            qpos=torch.tensor(
                                q[None], dtype=torch.float32, device=self.robot.device
                            ),
                            name=target.control_part,
                            env_ids=[0],
                            to_matrix=True,
                        )[0]
                        .detach()
                        .cpu()
                        .double()
                        .numpy()
                    )
                    fk = hypothetical @ np.linalg.inv(live_root) @ fk
                    if (
                        np.linalg.norm(fk[:3, 3] - pose[:3, 3]) <= 1e-3
                        and math_acos_trace(fk[:3, :3].T @ pose[:3, :3]) <= 1e-2
                    ):
                        solutions.append(q.copy())
                    if len(solutions) >= self.cfg.beam_width:
                        break
                if len(solutions) >= self.cfg.beam_width:
                    break
            combined = []
            for row in rows:
                for q in sorted(
                    solutions, key=lambda q: float(np.linalg.norm(q - row[indices]))
                ):
                    value = row.copy()
                    value[indices] = q
                    combined.append(value)
                    if len(combined) >= self.cfg.beam_width:
                        break
                if len(combined) >= self.cfg.beam_width:
                    break
            rows = combined
        return [
            self._project_mimic(row)
            for row in rows
            if self._targets_valid(self._project_mimic(row), phase, entities)
        ]

    def _targets_valid(
        self, row: np.ndarray, phase: TaskPhase, entities: Mapping[str, np.ndarray]
    ) -> bool:
        frames = self.frame_poses(row, TaskPhase("final_ik_check"), entities)
        for target in phase.targets:
            indices = self._indices(target.control_part)
            solver = self.robot.get_solver(name=target.control_part)
            root = frames[f"{self.robot.uid}/{solver.root_link_name}"]
            local = (
                solver.get_fk(
                    qpos=torch.tensor(
                        row[indices][None],
                        dtype=torch.float32,
                        device=self.robot.device,
                    )
                )[0]
                .detach()
                .cpu()
                .double()
                .numpy()
            )
            actual = root @ local
            desired = (
                np.array(target.pose)
                if target.frame_uid is None
                else entities[target.frame_uid] @ target.pose
            )
            if (
                np.linalg.norm(actual[:3, 3] - desired[:3, 3]) > 1e-3
                or math_acos_trace(actual[:3, :3].T @ desired[:3, :3]) > 1e-2
            ):
                return False
        return True

    def frame_poses(
        self,
        qpos: np.ndarray,
        phase: TaskPhase,
        entities: Mapping[str, np.ndarray],
        *,
        measured: bool = False,
    ) -> dict[str, np.ndarray]:
        """Compute full-link and payload transforms without writing simulator state."""
        if not measured:
            qpos = self._project_mimic(qpos)
        tensor = torch.tensor(
            qpos[self._chain_indices][None],
            dtype=torch.float32,
            device=self.robot.device,
        )
        chain = self.robot.pk_chain.forward_kinematics(tensor)
        output = dict(entities)
        for name, value in chain.items():
            output[f"{self.robot.uid}/{name}"] = (
                np.asarray(entities.get(self.robot.uid, self._root))
                @ value.get_matrix()[0].detach().cpu().double().numpy()
            )
        for held in phase.held_objects:
            indices = self._indices(held.control_part)
            solver = self.robot.get_solver(name=held.control_part)
            local = (
                solver.get_fk(
                    qpos=torch.tensor(
                        qpos[indices][None],
                        dtype=torch.float32,
                        device=self.robot.device,
                    )
                )[0]
                .detach()
                .cpu()
                .double()
                .numpy()
            )
            tcp = output[f"{self.robot.uid}/{solver.root_link_name}"] @ local
            output[held.entity_uid] = tcp @ held.tcp_from_object
        return output

    def check_state(
        self,
        qpos: np.ndarray,
        phase: TaskPhase,
        entities: Mapping[str, np.ndarray],
        *,
        measured: bool = False,
    ) -> bool:
        """Check synchronized arms, payload and environment at one configuration."""
        tolerance = 1e-4 if measured else 0.0
        if (
            not np.isfinite(qpos).all()
            or np.any(qpos < self.joint_limits[:, 0] - tolerance)
            or np.any(qpos > self.joint_limits[:, 1] + tolerance)
        ):
            return False
        if not np.allclose(qpos, self._project_mimic(qpos), atol=1e-6, rtol=0):
            return False
        check, _ = self.collision.check(
            self.frame_poses(qpos, phase, entities),
            allowed_contacts=phase.allowed_contacts,
            moving_objects=[h.entity_uid for h in phase.held_objects],
        )
        self.last_collision = check
        return check.status == "passed"

    def connect(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        phase: TaskPhase,
        entities: Mapping[str, np.ndarray],
        budget: FeasibilityBudget,
    ) -> np.ndarray | None:
        """Try a straight edge, then bounded joint-space sampling around obstacles."""
        budget.charge("planning")
        deadline = budget.clock() + min(self.cfg.planning_timeout_s, budget.remaining_s)
        checked_samples = 0
        measured_start = start.copy()
        if not self.check_state(measured_start, phase, entities, measured=True):
            return None
        start = self._project_mimic(
            np.clip(start, self.joint_limits[:, 0], self.joint_limits[:, 1])
        )

        def with_observed_start(path):
            return (
                np.concatenate((measured_start[None], path))
                if not np.array_equal(measured_start, start)
                else path
            )

        if not self.check_state(start, phase, entities) or not self.check_state(
            goal, phase, entities
        ):
            return None

        def edge(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
            nonlocal checked_samples
            budget.check()
            if budget.clock() >= deadline:
                return None
            count = max(
                2,
                int(np.ceil(np.max(np.abs(b - a)) / self.cfg.joint_resolution_rad)) + 1,
            )
            if checked_samples + count > self.cfg.max_motion_samples:
                return None
            path = np.linspace(a, b, count)
            for q in path:
                budget.check()
                if budget.clock() >= deadline:
                    return None
                checked_samples += 1
                if not self.check_state(q, phase, entities):
                    return None
            return path

        direct = edge(start, goal)
        if direct is not None:
            return with_observed_start(direct)
        nodes, parents, edges = [start.copy()], [-1], [None]
        # Only joints needed for this phase may explore; resource assignments and
        # held inactive parts remain fixed rather than moving unrelated joints.
        active = sorted(
            {i for t in phase.targets for i in self._indices(t.control_part)}
            - set(self.robot.mimic_ids)
        )
        for _ in range(self.cfg.max_rrt_nodes):
            budget.check()
            if (
                budget.clock() >= deadline
                or checked_samples >= self.cfg.max_motion_samples
            ):
                break
            sample = start.copy()
            if self._rng.random() < 0.2:
                sample = goal.copy()
            else:
                sample[active] = self._rng.uniform(
                    self.joint_limits[active, 0], self.joint_limits[active, 1]
                )
            sample = self._project_mimic(sample)
            nearest = min(
                range(len(nodes)),
                key=lambda i: float(np.linalg.norm(nodes[i] - sample)),
            )
            delta = sample - nodes[nearest]
            length = float(np.linalg.norm(delta))
            point = nodes[nearest] + delta * min(1.0, 0.25 / max(length, 1e-12))
            segment = edge(nodes[nearest], point)
            if segment is None:
                continue
            nodes.append(point)
            parents.append(nearest)
            edges.append(segment)
            finish = edge(point, goal)
            if finish is not None:
                parts = [finish]
                index = len(nodes) - 1
                while parents[index] >= 0:
                    parts.append(edges[index])
                    index = parents[index]
                return with_observed_start(
                    np.concatenate(
                        [p if i == 0 else p[1:] for i, p in enumerate(reversed(parts))]
                    )
                )
        return None

    def resulting_poses(
        self, qpos: np.ndarray, phase: TaskPhase, entities: Mapping[str, np.ndarray]
    ) -> Mapping[str, np.ndarray]:
        """Propagate held-object and authored post-stage planning projections."""
        output = dict(entities)
        frames = self.frame_poses(qpos, phase, entities)
        for held in phase.held_objects:
            output[held.entity_uid] = frames[held.entity_uid]
        for change in phase.resulting_poses:
            output[change.entity_uid] = np.asarray(change.arena_pose)
        return output


def math_acos_trace(rotation: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1.0, 1.0)))
