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

"""One World step loop for independent Task Program arena runtimes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from copy import deepcopy
from typing import Any
import torch
from tensordict import TensorDict
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge

__all__ = [
    "ProgramBatchResult",
    "TaskProgramBatchExecutor",
    "collect_program_batch",
    "collect_vector_program",
]


@dataclass(frozen=True)
class ProgramBatchResult:
    """Measured execution and canonical bridge acceptance per assigned row.

    Args:
        env_ids: Ordered physical rows represented in this result.
        lengths: Actual submitted commands, excluding padding and completed rows.
        accepted: Canonical program completion per row; the host adds physical checks.
        reasons: Per-row terminal outcomes.
        segments: Actual segment ranges and runtime provenance.
        expansion_records: Optional per-call motion recipe evidence.
    """

    env_ids: tuple[int, ...]
    lengths: tuple[int, ...]
    accepted: tuple[bool, ...]
    reasons: tuple[str, ...]
    segments: tuple[tuple[Mapping[str, object], ...], ...]
    expansion_records: tuple[tuple[Mapping[str, object], ...], ...] = ()


class _Cursor:
    def __init__(self, bridge: TaskProgramDemoBridge) -> None:
        self.bridge = bridge
        self.segments = iter(bridge)
        self.segment = None
        self.actions = None
        self.finished = False
        self.accepted = False
        self.reason = "running"
        self.length = 0
        self.results = []
        self.segment_start = 0
        self.emergency = None
        self.pending_consumed = False

    def next(self) -> ControllerAction | None:
        if self.finished:
            return None
        if self.emergency is not None:
            try:
                action = next(self.emergency)
                self.pending_consumed = False
                return action
            except StopIteration:
                self.finished = True
                self.emergency = None
                self.actions.close()
                self.actions = None
                return None
        while not self.finished:
            if self.actions is None:
                try:
                    self.segment = next(self.segments)
                except StopIteration:
                    self.finished = True
                    self.accepted = self.bridge.program_completed and bool(
                        self.bridge.completion_mask.all()
                    )
                    self.reason = "completed" if self.accepted else "program_rejected"
                    return None
                self.segment_start = self.length
                self.actions = iter(self.segment.actions)
            try:
                action = next(self.actions)
            except StopIteration:
                # The canonical validator finalizes segment effects and row eligibility.
                valid = self.segment.validator()
                metadata = deepcopy(dict(self.segment.metadata))
                metadata.update(
                    segment_id=metadata["program_segment_index"],
                    name=self.segment.name,
                    start_step=self.segment_start,
                    end_step=self.length,
                    success=bool(torch.as_tensor(valid).all()),
                    instruction=self.segment.instruction,
                    instruction_source=self.segment.instruction_source,
                )
                self.results.append(metadata)
                self.actions = None
                if not bool(torch.as_tensor(valid).all()):
                    self.finished = True
                    self.reason = "segment_rejected"
                    return None
                continue
            if not isinstance(action, ControllerAction):
                raise TypeError("Task Program must yield ControllerAction")
            self.pending_consumed = False
            return action
        return None

    def abort(self, reason: str) -> None:
        if self.finished or self.emergency is not None:
            return
        self.reason = reason
        self.emergency = iter(
            self.segment.abort_actions(
                reason, last_action_consumed=self.pending_consumed
            )
        )

    def close(self) -> None:
        if self.actions is not None and callable(getattr(self.actions, "close", None)):
            self.actions.close()
        self.segments.close()


def collect_program_batch(
    environment: Any,
    bridges: Mapping[int, TaskProgramDemoBridge],
    *,
    max_steps: int,
    observe_step: Callable[[tuple[int, ...], ControllerAction, object], None],
    on_started: Callable[[int], None] | None = None,
    reject_row: Callable[[int], bool] | None = None,
) -> ProgramBatchResult:
    """Consume independent row bridges through a single batched Gym step.

    Args:
        environment: Initialized, prepared Gym environment owned by the caller.
        bridges: One fresh single-row bridge per selected physical environment.
        max_steps: Positive shared control-step budget for the cohort.
        observe_step: Called after each successful step with active physical row
            IDs, the submitted merged command and the actual returned observation.
            The caller captures the initial observation before this function.
        on_started: Optional notification after a row's first consumed command.
    Returns:
        Per-row completion and valid lengths. Finished rows produce no more data.
    Raises:
        ValueError: For mismatched row bindings, clocks, or configuration.
        RuntimeError: For native/runtime failures; callers must stop the world.

    This function owns neither reset nor persistence. It keeps the canonical
    segment validators and command acknowledgement lifecycle. Observers must
    bound their buffers; source-specific physical validators still own acceptance.
    """
    env = getattr(environment, "unwrapped", environment)
    ids = tuple(bridges)
    if not ids or any(type(i) is not int or not 0 <= i < env.num_envs for i in ids):
        raise ValueError("bridges must address valid physical rows")
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    for row, bridge in bridges.items():
        if (
            not isinstance(bridge, TaskProgramDemoBridge)
            or bridge.clock.step_index != 0
            or bridge.clock.step_dt != env.step_dt
        ):
            raise ValueError("bridges must be fresh and use the environment clock")
        if tuple(bridge.env_ids.tolist()) != (row,):
            raise ValueError(
                "bridge runtime must bind exactly its assigned physical row"
            )
    cursors = {row: _Cursor(bridge) for row, bridge in bridges.items()}
    before_auto = getattr(env, "_demo_no_auto_reset", False)
    before_mask = getattr(env, "_demo_active_mask", None)
    hold = env.robot.get_qpos(target=True).clone()
    active_joints = list(env.active_joint_ids)
    env._demo_no_auto_reset = True
    try:
        for tick in range(max_steps + 2):
            commands = {
                row: action
                for row, cursor in cursors.items()
                if (action := cursor.next()) is not None
            }
            if not commands:
                break
            if tick >= max_steps:
                for row in tuple(commands):
                    cursor = cursors[row]
                    if cursor.emergency is None:
                        cursor.abort("step_budget_exhausted")
                        command = cursor.next()
                        if command is None:
                            del commands[row]
                        else:
                            commands[row] = command
                if not commands:
                    break
            qpos = hold.clone()
            qvel = torch.zeros_like(qpos)
            has_velocity = False
            for row, action in commands.items():
                values = (
                    {"qpos": action.value}
                    if isinstance(action.value, torch.Tensor)
                    else action.value
                )
                if "qpos" not in values or any(
                    key not in {"qpos", "qvel"} for key in values.keys()
                ):
                    raise ValueError(
                        "batch collector requires explicit position or position-velocity commands"
                    )
                for name, target in (("qpos", qpos), ("qvel", qvel)):
                    if name not in values:
                        continue
                    value = values[name]
                    if value.ndim != 2 or value.shape[0] != 1:
                        raise ValueError("row command must retain one batch row")
                    if value.shape[1] == target.shape[1]:
                        target[row] = value[0]
                    elif value.shape[1] == len(active_joints):
                        target[row, active_joints] = value[0]
                    else:
                        raise ValueError("row command joint layout differs")
                    has_velocity |= name == "qvel"
            payload = {"qpos": qpos}
            if has_velocity:
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
            mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            mask[list(commands)] = True
            env._demo_active_mask = mask
            observation, _, terminated, truncated, _ = env.step(action)
            hold = qpos
            for row in commands:
                cursor = cursors[row]
                if cursor.length == 0 and on_started is not None:
                    on_started(row)
                cursor.length += 1
                cursor.pending_consumed = True
            observe_step(tuple(commands), action, observation)
            for row in commands:
                if reject_row is not None and reject_row(row):
                    cursors[row].abort("measured_validation_failed")
                elif bool(truncated[row]) or bool(terminated[row]):
                    cursors[row].abort("environment_terminated")
        return ProgramBatchResult(
            ids,
            tuple(cursors[row].length for row in ids),
            tuple(cursors[row].accepted for row in ids),
            tuple(cursors[row].reason for row in ids),
            tuple(tuple(cursors[row].results) for row in ids),
        )
    finally:
        try:
            for cursor in cursors.values():
                cursor.close()
        finally:
            env._demo_no_auto_reset = before_auto
            if before_mask is None:
                delattr(env, "_demo_active_mask")
            else:
                env._demo_active_mask = before_mask


def collect_vector_program(
    environment: Any,
    program: Any,
    env_ids: tuple[int, ...],
    *,
    max_steps: int,
    observe_step: Callable,
    on_started: Callable[[int], None] | None = None,
    expansion_profile: Any = None,
    recipe_indices: tuple[int, ...] | None = None,
) -> ProgramBatchResult:
    """Run the canonical vector Task Program with explicit tail-row masking.

    Args:
        environment: Prepared Gym host owning the entire World.
        program: Provider-free compiled Task Program.
        env_ids: Physical rows assigned real candidates.
        max_steps: Shared control-step budget.
        observe_step: Actual post-step observation consumer.
        on_started: First-command notification per assigned row.
        expansion_profile: Optional canonical combined motion profile.
        recipe_indices: Logical recipe indices for the full physical capacity.
    Returns:
        Canonical row outcomes, segment annotations and expansion provenance.
    """
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    env = getattr(environment, "unwrapped", environment)
    bridge = env.task_program_adapter.create_bridge(
        program,
        expansion_profile=expansion_profile,
        recipe_indices=recipe_indices,
        isolate_policy_sessions=True,
    )
    started = set()
    steps = 0

    def observe(mask, action, observation):
        nonlocal steps
        ids = tuple(i for i, active in enumerate(mask) if active)
        for row in ids:
            if row not in started:
                if on_started is not None:
                    on_started(row)
                started.add(row)
        steps += 1
        merged = ControllerAction(
            action.value,
            metadata={"row_actions": {str(row): dict(action.metadata) for row in ids}},
        )
        observe_step(ids, merged, observation)

    with env.task_program_bridge_scope(bridge):
        result = execute_demo_episode(
            env,
            segments=bridge.iter_segments(),
            task_program=program,
            active_env_ids=env_ids,
            step_observer=observe,
            should_stop=lambda: steps >= max_steps,
        )
        completion = (
            bridge.completion_mask
            if bridge.program_completed
            else torch.zeros(env.num_envs, dtype=torch.bool)
        )
        records = bridge.expansion_records
        return ProgramBatchResult(
            env_ids,
            tuple(result.lengths[row] for row in env_ids),
            tuple(bool(result.success[row] and completion[row]) for row in env_ids),
            tuple(result.terminal_reasons[row] for row in env_ids),
            tuple(
                tuple(
                    segment.to_metadata(env_id=row)
                    for segment in result.segments
                    if segment.active[row]
                )
                for row in env_ids
            ),
            tuple(
                tuple(
                    record.to_metadata() for record in records if record.env_id == row
                )
                for row in env_ids
            ),
        )


class TaskProgramBatchExecutor:
    """One execution port for Task Programs with native or prototype motion bindings.

    Args:
        environment: Prepared World owner.
        program: Common compiled task structure.
        row_bridges: Optional independently bound task runtimes for sources whose
            planning protocol is row-scoped. They still share one World step loop.
        expansion_profile: Optional native motion/affordance recipe profile.
        recipe_indices: Per-slot logical recipe selection, including inactive padding.
    """

    def __init__(
        self,
        environment: Any,
        program: Any,
        *,
        row_bridges: Mapping[int, TaskProgramDemoBridge] | None = None,
        expansion_profile: Any = None,
        recipe_indices: tuple[int, ...] | None = None,
    ) -> None:
        self.environment, self.program = environment, program
        self.row_bridges = row_bridges
        self.expansion_profile, self.recipe_indices = expansion_profile, recipe_indices
        if row_bridges is not None and any(
            bridge.program_id != program.program_id for bridge in row_bridges.values()
        ):
            raise ValueError("one batch must execute the same Task Program structure")

    def run(
        self,
        env_ids: tuple[int, ...],
        *,
        max_steps: int,
        observe_step: Callable[[tuple[int, ...], ControllerAction, object], None],
        on_started: Callable[[int], None] | None = None,
        reject_row: Callable[[int], bool] | None = None,
    ) -> ProgramBatchResult:
        """Execute the requested rows with the canonical acknowledgement protocol."""
        if self.row_bridges is not None:
            if set(self.row_bridges) != set(env_ids):
                raise ValueError("row bindings differ from assigned environments")
            return collect_program_batch(
                self.environment,
                self.row_bridges,
                max_steps=max_steps,
                observe_step=observe_step,
                on_started=on_started,
                reject_row=reject_row,
            )
        return collect_vector_program(
            self.environment,
            self.program,
            env_ids,
            max_steps=max_steps,
            observe_step=observe_step,
            on_started=on_started,
            expansion_profile=self.expansion_profile,
            recipe_indices=self.recipe_indices,
        )
