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

"""Explicit process-wide CPU thread ownership for isolated planning workers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING
import torch
from .solvers.pytorch_solver import PytorchSolver

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["CpuPlanningThreadBudget", "cpu_planning_threads", "candidate_ik_threads"]


@dataclass(frozen=True, slots=True)
class CpuPlanningThreadBudget:
    """Captured effective thread counts owned by one explicit worker scope.

    Attributes:
        initial_threads: Torch intra-op count observed before entering the scope.
        serial_threads: Intra-op count for ordinary worker operations.
        batch_threads: Initial count capped by the declared batch-thread limit.
    """

    initial_threads: int
    serial_threads: int
    batch_threads: int


_BUDGET: ContextVar[CpuPlanningThreadBudget | None] = ContextVar(
    "embodichain_cpu_planning_threads", default=None
)


@contextmanager
def cpu_planning_threads(
    *,
    device: torch.device | str,
    num_envs: int,
    serial_threads: int = 1,
    batch_thread_cap: int = 8,
) -> Iterator[CpuPlanningThreadBudget | None]:
    """Capture initial capacity, use serial work, then restore the original count.

    Args:
        device: Explicit worker tensor device. Non-CPU workers remain unchanged.
        num_envs: Worker environment count. Counts other than one remain unchanged.
        serial_threads: Positive ordinary-operation count; source behavior uses 1.
        batch_thread_cap: Positive cap on captured initial capacity; source uses 8.

    Yields:
        Immutable effective counts, or None for a non-B1/non-CPU worker. Capturing
        the budget must precede any caller-side serial-thread override.

    Torch intra-op settings are process-wide. Use this explicitly in an isolated
    worker owned by the caller, with no concurrent Python planning scope in that
    process. No action, simulator or RNG state is created.
    The default shared runtime does not enter this scope.
    """
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("CPU planning threads require a positive environment count.")
    for name, value in (
        ("serial_threads", serial_threads),
        ("batch_thread_cap", batch_thread_cap),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if num_envs != 1 or torch.device(device).type != "cpu":
        yield None
        return
    previous = torch.get_num_threads()
    budget = CpuPlanningThreadBudget(
        previous, serial_threads, min(previous, batch_thread_cap)
    )
    token = _BUDGET.set(budget)
    try:
        torch.set_num_threads(serial_threads)
        yield budget
    finally:
        try:
            torch.set_num_threads(previous)
        finally:
            _BUDGET.reset(token)


@contextmanager
def candidate_ik_threads(
    robot: Robot,
    control_part: str,
    poses: torch.Tensor,
) -> Iterator[None]:
    """Temporarily raise large B1 CPU PytorchSolver candidate batches only.

    Args:
        robot: Exact IK owner; untouched when no worker budget is active.
        control_part: Explicit controlled solver group.
        poses: Candidate poses ``(B,P,V,4,4)``. The fixed source threshold is
            P*V at least 100; unsupported devices, batches and solver types retain
            their current threading without changing candidate input/order.

    Yields:
        Control for the original solver call and shape-preserving adaptation.
        Existing higher thread counts are never lowered. Every temporary change
        is restored on completion or failure, without changing random streams.
    """
    budget = _BUDGET.get()
    if budget is None:
        yield
        return
    if poses.ndim != 5 or poses.shape[-2:] != (4, 4):
        raise ValueError("Candidate IK thread scope requires (B,P,V,4,4) poses.")
    if (
        poses.device.type != "cpu"
        or poses.shape[0] != 1
        or poses.shape[1] * poses.shape[2] < 100
        or not isinstance(robot.get_solver(control_part), PytorchSolver)
    ):
        yield
        return
    previous = torch.get_num_threads()
    if previous >= budget.batch_threads:
        yield
        return
    try:
        torch.set_num_threads(budget.batch_threads)
        yield
    finally:
        torch.set_num_threads(previous)
