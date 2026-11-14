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

"""One World, rolling physical rows and receipt-qualified workload quotas."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
import math
import os
from threading import get_ident
import time
from typing import Any, Protocol, TYPE_CHECKING

from embodichain.utils import configclass
from .contracts import CommitReceipt

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from embodichain.lab.gym.envs.types import ControllerAction
    from .program import ProgramBatchResult

__all__ = [
    "ExpansionWorkloadCfg",
    "WorkloadExecutionCfg",
    "WorkloadAssignment",
    "WorkloadRowResult",
    "WorkloadSource",
    "WorkloadProgramExecutor",
]


@configclass
class ExpansionWorkloadCfg:
    """Bound one logical source independently of its physical slot.

    Args:
        workload_id: Unique source/recipe group identifier.
        target_episodes: Confirmed qualified receipts required for this group.
        max_attempts: Physical candidate budget, including rejected preparations.
        max_wall_time_s: Admission budget starting with the first candidate.
            Already active trajectories retain their ordinary safe-stop path.
    """

    workload_id: str = "default"
    target_episodes: int = 1
    max_attempts: int = 8
    max_wall_time_s: float = 600.0

    def __post_init__(self) -> None:
        if (
            type(self.workload_id) is not str
            or not self.workload_id
            or self.workload_id.strip() != self.workload_id
        ):
            raise ValueError("workload_id must be a nonempty stripped string")
        for name in ("target_episodes", "max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_attempts < self.target_episodes:
            raise ValueError("max_attempts must cover target_episodes")
        if (
            type(self.max_wall_time_s) not in (int, float)
            or not math.isfinite(self.max_wall_time_s)
            or self.max_wall_time_s <= 0
        ):
            raise ValueError("max_wall_time_s must be finite and positive")


@configclass
class WorkloadExecutionCfg:
    """Declare physical capacity without allocating processes or Worlds.

    Args:
        workload_count: Maximum simultaneous independent source groups, W.
        batch_size: Candidate rows reserved for each active group, B.
        max_steps: Maximum ordinary controller commands per candidate.
        settling_steps: Minimum physical substeps before a row becomes active.
        seed: Base seed; each candidate owns a stream keyed by local ordinal.
        batch_planning: Cooperatively merge compatible native motion/solver calls.
    """

    workload_count: int = 1
    batch_size: int = 1
    max_steps: int = 2000
    settling_steps: int = 5
    seed: int = 0
    batch_planning: bool = True

    def __post_init__(self) -> None:
        for name in ("workload_count", "batch_size", "max_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("settling_steps", "seed"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if type(self.batch_planning) is not bool:
            raise TypeError("batch_planning must be a bool")

    @property
    def num_envs(self) -> int:
        """Return the required single-World capacity, W multiplied by B."""
        return self.workload_count * self.batch_size


@dataclass(frozen=True)
class WorkloadAssignment:
    """Logical candidate identity bound to a physical row for one attempt.

    Args:
        workload_id: Source group, independent of lane/row allocation.
        ordinal: Zero-based candidate index local to this workload.
        env_id: Physical environment row in the sole World.
        seed: Candidate-owned random stream seed.
    """

    workload_id: str
    ordinal: int
    env_id: int
    seed: int


@dataclass(frozen=True)
class WorkloadRowResult:
    """Canonical bridge outcome and exact World control-boundary span.

    Args:
        assignment: Candidate whose bridge and measured buffer were consumed.
        program: Original Task Program result for this one physical row.
        world_control_start: First actually consumed command boundary, or None.
        world_control_end: Last consumed command boundary, or None.
    """

    assignment: WorkloadAssignment
    program: ProgramBatchResult
    world_control_start: int | None
    world_control_end: int | None


class WorkloadSource(Protocol):
    """Task-owned scene/plan/evidence ports; the executor alone steps the World.

    Sources retain native ExpansionSession admission, required physical checks,
    evidence assembly and sink receipts. ``complete`` may return a receipt only
    after all required checks pass and native persistence confirms the write.
    Callbacks must not advance physics or create another World. A completed
    group's frozen evidence can be exported independently of the live World.
    """

    def prepare(self, assignments: tuple[WorkloadAssignment, ...]) -> None:
        """Apply selected-row changes inside the trusted reset callback."""
        ...

    def activate(
        self, assignment: WorkloadAssignment, observation: object
    ) -> TaskProgramDemoBridge | None:
        """Validate measured preparation, capture T+1 initial data and bind a row."""
        ...

    def started(self, assignment: WorkloadAssignment, world_control_step: int) -> None:
        """Mark rollout started in the native session after its first actual step."""
        ...

    def observe(
        self,
        assignments: tuple[WorkloadAssignment, ...],
        action: ControllerAction,
        observation: object,
    ) -> Sequence[int]:
        """Record every active boundary; return rows requiring safe rejection."""
        ...

    def complete(self, result: WorkloadRowResult) -> CommitReceipt | None:
        """Assemble native evidence and return its confirmed receipt or rejection."""
        ...


class WorkloadProgramExecutor:
    """Refill independent rows while preserving native bridge acknowledgement.

    Args:
        environment: One initialized environment with exactly W times B rows.
        workloads: Ordered source groups with independent quotas and budgets.
        source: Host-owned scene, Task Program, validation and persistence ports.
        cfg: Capacity and control/settling budgets; creates no World itself.
        checkpoint: Optional cooperative cancellation/window callback, called at
            control boundaries on the owner thread.
        on_workload_done: Optional frozen-group notification. It receives an
            owned count mapping and must not advance or mutate the live World.
    """

    def __init__(
        self,
        environment: Any,
        workloads: Sequence[ExpansionWorkloadCfg],
        source: WorkloadSource,
        cfg: WorkloadExecutionCfg,
        *,
        checkpoint: Callable[[], None] | None = None,
        on_workload_done: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        import torch

        if not isinstance(cfg, WorkloadExecutionCfg):
            raise TypeError("cfg must be WorkloadExecutionCfg")
        cfg.__post_init__()
        self.env = getattr(environment, "unwrapped", environment)
        if self.env.num_envs != cfg.num_envs:
            raise ValueError(
                "the single World must contain workload_count * batch_size environments"
            )
        events = getattr(self.env, "event_manager", None)
        if events is not None and events.active_functors.get("interval"):
            raise ValueError("rolling expansion requires fixed episode parameters")
        if not workloads or any(
            not isinstance(w, ExpansionWorkloadCfg) for w in workloads
        ):
            raise ValueError("workloads must contain ExpansionWorkloadCfg values")
        for workload in workloads:
            workload.__post_init__()
        self.workloads = {
            w.workload_id: ExpansionWorkloadCfg(**w.to_dict()) for w in workloads
        }
        if len(self.workloads) != len(workloads) or cfg.workload_count > len(workloads):
            raise ValueError("workload IDs must be unique and cover every active lane")
        self.cfg, self.source = WorkloadExecutionCfg(**cfg.to_dict()), source
        self.checkpoint, self.on_workload_done = checkpoint, on_workload_done
        self.owner = get_ident()
        from embodichain.lab.sim import SimulationManager

        if len(SimulationManager._instances) != 1:
            raise ValueError(
                "rolling workload collection requires one live World in its process"
            )
        state_device = self.env.robot.get_qpos().device
        if state_device.type != self.env.sim.device.type:
            raise RuntimeError("measured robot state and World devices disagree")
        self._world_info = {
            "process_id": os.getpid(),
            "world_instance_id": self.env.sim.instance_id,
            "world_object_id": id(self.env.sim.get_world()),
            "world_count": len(SimulationManager._instances),
            "physics_backend": self.env.sim.physics_backend,
            "physics_device": str(self.env.sim.device),
            "state_device": str(state_device),
            "gpu_physics": bool(self.env.sim.is_use_gpu_physics),
            "solvers": {
                name: {"class": type(solver).__name__, "device": str(solver.device)}
                for name, solver in self.env.robot._solvers.items()
            },
        }
        self._used = False
        self.lanes: list[str | None] = [None] * cfg.workload_count
        self.counts = {
            name: {"attempts": 0, "rollouts": 0, "committed": 0}
            for name in self.workloads
        }
        self.started_at: dict[str, float] = {}
        self.warming: dict[int, dict[str, Any]] = {}
        self.live: dict[int, dict[str, Any]] = {}
        self.completed: set[str] = set()
        self.receipts: set[tuple[str, str, str]] = set()
        self.control_steps = self.active_control_steps = self.refills = 0
        self.hold = self.env.robot.get_qpos(target=True).clone()
        self.observation = None
        self.planning_stats: dict[str, int] = {}
        self.torch = torch

    @property
    def world_info(self) -> Mapping[str, object]:
        """Return owned actual process, World, state and solver device identities."""
        from copy import deepcopy

        return deepcopy(self._world_info)

    def _eligible(self, name: str) -> bool:
        workload, counts = self.workloads[name], self.counts[name]
        return (
            counts["committed"] < workload.target_episodes
            and counts["attempts"] < workload.max_attempts
            and (
                name not in self.started_at
                or time.monotonic() - self.started_at[name] < workload.max_wall_time_s
            )
        )

    def _inflight(self, name: str) -> int:
        return sum(
            state["assignment"].workload_id == name
            for state in (*self.live.values(), *self.warming.values())
        )

    def _group_done(self, name: str) -> None:
        if name not in self.completed:
            self.completed.add(name)
            if self.on_workload_done is not None:
                self.on_workload_done(name, dict(self.counts[name]))

    def _fill(self) -> None:
        added = []
        occupied = self.live.keys() | self.warming.keys()
        for lane, name in enumerate(self.lanes):
            if (
                name is not None
                and not self._inflight(name)
                and not self._eligible(name)
            ):
                self._group_done(name)
                name = None
            if name is None:
                name = next(
                    (
                        key
                        for key in self.workloads
                        if key not in self.lanes and self._eligible(key)
                    ),
                    None,
                )
            self.lanes[lane] = name
            if name is None:
                continue
            for row in range(
                lane * self.cfg.batch_size, (lane + 1) * self.cfg.batch_size
            ):
                if row in occupied or not self._eligible(name):
                    continue
                counts, spec = self.counts[name], self.workloads[name]
                if counts["committed"] + self._inflight(name) >= spec.target_episodes:
                    break
                assignment = WorkloadAssignment(
                    name, counts["attempts"], row, self.cfg.seed + counts["attempts"]
                )
                counts["attempts"] += 1
                self.started_at.setdefault(name, time.monotonic())
                self.warming[row] = {"assignment": assignment, "settled": 0}
                added.append(assignment)
        if not added:
            return
        self.refills += bool(self.live)
        assignments = tuple(added)
        result = self.env.reset(
            seed=self.cfg.seed,
            options={
                "save_data": False,
                "reset_ids": [a.env_id for a in added],
                "episode_prepare": lambda: self.source.prepare(assignments),
            },
        )
        self.observation = result[0]
        rows = [a.env_id for a in added]
        self.hold[rows] = self.env.robot.get_qpos(target=True)[rows]

    def _activate(self) -> None:
        from ._workload_cursor import _WorkloadCursor
        from .program import ProgramBatchResult

        for row in tuple(self.warming):
            state = self.warming[row]
            if state["settled"] < self.cfg.settling_steps:
                continue
            assignment = state["assignment"]
            self.env._elapsed_steps[row] = 0
            self.env._reset_physical_objective((row,))
            bridge = self.source.activate(assignment, self.observation)
            del self.warming[row]
            if bridge is None:
                outcome = ProgramBatchResult(
                    (row,), (0,), (False,), ("preparation_rejected",), ((),)
                )
                if (
                    self.source.complete(
                        WorkloadRowResult(assignment, outcome, None, None)
                    )
                    is not None
                ):
                    raise ValueError(
                        "unstarted or rejected preparation cannot earn a receipt"
                    )
                continue
            if (
                tuple(bridge.env_ids.tolist()) != (row,)
                or bridge.clock.step_index != 0
                or not math.isclose(
                    bridge.clock.step_dt, self.env.step_dt, rel_tol=0, abs_tol=1e-12
                )
            ):
                raise ValueError(
                    "bridge must freshly bind exactly the assigned physical row and clock"
                )
            self.live[row] = {
                **state,
                "cursor": _WorkloadCursor(bridge, self.env.robot, assignment.seed),
                "start": None,
            }

    def _finish(self, row: int) -> None:
        from .program import ProgramBatchResult

        state = self.live[row]
        assignment, cursor = state["assignment"], state["cursor"]
        end = self.control_steps if cursor.length else None
        if cursor.length and end - state["start"] + 1 != cursor.length:
            raise RuntimeError("an active trajectory missed a World control boundary")
        result = ProgramBatchResult(
            (row,),
            (cursor.length,),
            (cursor.accepted,),
            (cursor.reason,),
            (tuple(cursor.results),),
        )
        receipt = self.source.complete(
            WorkloadRowResult(assignment, result, state["start"], end)
        )
        if receipt is not None:
            if (
                not isinstance(receipt, CommitReceipt)
                or not receipt.confirmed
                or not cursor.accepted
                or not cursor.length
            ):
                raise ValueError(
                    "only confirmed qualified native receipts earn workload quota"
                )
            identity = receipt.episode_id, receipt.candidate_id, receipt.commit_id
            if any(
                identity[i] == known[i] for known in self.receipts for i in range(3)
            ):
                raise ValueError(
                    "a repeated episode/candidate/commit cannot earn quota twice"
                )
            self.receipts.add(identity)
            self.counts[assignment.workload_id]["committed"] += 1
        cursor.close()
        del self.live[row]

    def run(self) -> Mapping[str, object]:
        """Collect until every group meets quota or exhausts its original budget.

        Returns:
            Owned counts, completeness and actual World control/planning stats.
            Failed quotas remain explicit; partial results never imply success.
        """
        if get_ident() != self.owner or self._used:
            raise RuntimeError(
                "a workload executor is single-use on its World owner thread"
            )
        self._used = True
        from tensordict import TensorDict
        from embodichain.lab.gym.envs.types import ControllerAction
        from ._planning_batches import batched_planning

        env, torch = self.env, self.torch
        previous_auto = getattr(env, "_demo_no_auto_reset", False)
        had_mask = hasattr(env, "_demo_active_mask")
        previous_mask = getattr(env, "_demo_active_mask", None)
        env._demo_no_auto_reset = True
        primary = None
        planning = (
            batched_planning(env.robot, self.cfg.seed)
            if self.cfg.batch_planning
            else nullcontext()
        )
        try:
            with planning as planner:
                while True:
                    if self.checkpoint is not None:
                        self.checkpoint()
                    self._fill()
                    self._activate()
                    cursors = {row: state["cursor"] for row, state in self.live.items()}
                    commands = (
                        planner.gather(cursors)
                        if planner is not None
                        else {
                            row: command
                            for row, cursor in cursors.items()
                            if (command := cursor.next()) is not None
                        }
                    )
                    for row in tuple(self.live):
                        if row not in commands:
                            self._finish(row)
                    if not commands and not self.warming:
                        if any(self._eligible(name) for name in self.workloads):
                            continue
                        break
                    qpos, qvel = self.hold.clone(), torch.zeros_like(self.hold)
                    velocity = (
                        env.expert_action_spec.joint_command_mode == "position_velocity"
                    )
                    for row in tuple(commands):
                        cursor = self.live[row]["cursor"]
                        if (
                            cursor.length >= self.cfg.max_steps
                            and cursor.emergency is None
                        ):
                            cursor.abort("step_budget_exhausted")
                            replacement = cursor.next()
                            if replacement is None:
                                del commands[row]
                                self._finish(row)
                                continue
                            commands[row] = replacement
                        value = commands[row].value
                        fields = (
                            {"qpos": value}
                            if isinstance(value, torch.Tensor)
                            else value
                        )
                        if "qpos" not in fields or set(fields) - {"qpos", "qvel"}:
                            raise ValueError(
                                "workload rows require explicit position controller commands"
                            )
                        if velocity and "qvel" not in fields:
                            raise ValueError(
                                "position-velocity evidence requires explicit velocity commands"
                            )
                        for name, target in (("qpos", qpos), ("qvel", qvel)):
                            if name not in fields:
                                continue
                            data = fields[name]
                            if data.ndim != 2 or data.shape[0] != 1:
                                raise ValueError(
                                    "row commands must retain one physical batch row"
                                )
                            columns = (
                                list(env.active_joint_ids)
                                if data.shape[1] == len(env.active_joint_ids)
                                else list(range(env.robot.dof))
                            )
                            if data.shape[1] != len(columns):
                                raise ValueError("row command joint layout differs")
                            target[row, columns] = data[0]
                            velocity |= name == "qvel"
                    if not commands and not self.warming:
                        continue
                    payload = {"qpos": qpos}
                    if velocity:
                        payload["qvel"] = qvel
                    action = ControllerAction(
                        TensorDict(payload, batch_size=[env.num_envs]),
                        metadata={
                            "row_actions": {
                                str(row): dict(command.metadata)
                                for row, command in commands.items()
                            }
                        },
                    )
                    mask = torch.zeros(
                        env.num_envs, dtype=torch.bool, device=env.device
                    )
                    mask[list(commands)] = True
                    env._demo_active_mask = mask
                    before = env.sim.physics_step_index
                    self.observation, _, terminated, truncated, _ = env.step(action)
                    advanced = env.sim.physics_step_index - before
                    if advanced <= 0:
                        raise RuntimeError(
                            "the owning World did not advance physical time"
                        )
                    self.control_steps += 1
                    self.active_control_steps += len(commands)
                    self.hold = qpos
                    for state in self.warming.values():
                        state["settled"] += advanced
                    for row in commands:
                        state, cursor = self.live[row], self.live[row]["cursor"]
                        if cursor.length == 0:
                            state["start"] = self.control_steps
                            self.counts[state["assignment"].workload_id][
                                "rollouts"
                            ] += 1
                            self.source.started(state["assignment"], self.control_steps)
                        cursor.length += 1
                        cursor.pending_consumed = True
                    assignments = tuple(
                        self.live[row]["assignment"] for row in commands
                    )
                    rejected = (
                        tuple(
                            self.source.observe(assignments, action, self.observation)
                        )
                        if commands
                        else ()
                    )
                    if (
                        len(set(rejected)) != len(rejected)
                        or set(rejected) - commands.keys()
                    ):
                        raise ValueError(
                            "source rejection must select unique active physical rows"
                        )
                    for row in commands:
                        if row in rejected:
                            self.live[row]["cursor"].abort("measured_validation_failed")
                        elif bool(terminated[row]) or bool(truncated[row]):
                            self.live[row]["cursor"].abort("environment_terminated")
                if planner is not None:
                    self.planning_stats = dict(planner.stats)
            for name in self.workloads:
                self._group_done(name)
            return {
                "complete": all(
                    self.counts[name]["committed"] == spec.target_episodes
                    for name, spec in self.workloads.items()
                ),
                "num_envs": self.cfg.num_envs,
                "workload_count": self.cfg.workload_count,
                "batch_size": self.cfg.batch_size,
                "counts": {name: dict(values) for name, values in self.counts.items()},
                "control_steps": self.control_steps,
                "active_control_steps": self.active_control_steps,
                "refills": self.refills,
                "planning": dict(self.planning_stats),
                "world": self.world_info,
            }
        except BaseException as error:
            primary = error
            raise
        finally:
            cleanup = None
            for state in self.live.values():
                try:
                    state["cursor"].close()
                except BaseException as error:
                    cleanup = error
            env._demo_no_auto_reset = previous_auto
            if had_mask:
                env._demo_active_mask = previous_mask
            elif hasattr(env, "_demo_active_mask"):
                delattr(env, "_demo_active_mask")
            if primary is None and cleanup is not None:
                raise cleanup
