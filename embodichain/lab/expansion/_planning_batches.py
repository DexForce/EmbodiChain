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


"""Cooperatively batch native kinematics on the sole World-owning thread."""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from dataclasses import fields, is_dataclass, replace
from contextvars import ContextVar, copy_context

__all__: list[str] = []

_ACTIVE: ContextVar[PlanningBatches | None] = ContextVar(
    "expansion_planning", default=None
)


def _try_motion(generator, targets, options, defer_recovery):
    active = _ACTIVE.get()
    if active is None or active.motion_passthrough:
        return False, None
    from greenlet import getcurrent

    if getcurrent() not in active.tasks:
        return False, None
    return True, active.motions.generate(
        generator, targets, options, defer_recovery=defer_recovery
    )


class PlanningBatches:
    """Suspend scalar planners only at pure solver calls, then fuse their rows.

    Native Robot budget decorators, plan transforms and validation run inside
    each original cursor. No World call runs on a worker thread. Results are
    cloned into independent rows before any scalar planner resumes.
    """

    def __init__(self, robot, seed):
        import numpy as np
        import random
        import torch
        from greenlet import getcurrent

        self.torch, self.np, self.random = torch, np, random
        self.owner = getcurrent()
        self.tasks = {}
        self.saved = []
        self.solo = False
        self.direct_depth = 0
        self.motion_passthrough = False
        self.devices = (
            [
                (
                    robot.device.index
                    if robot.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
            if robot.device.type == "cuda"
            else []
        )
        self.stats = {
            "solver_calls": 0,
            "logical_calls": 0,
            "merged_calls": 0,
            "max_group": 0,
            "motion_calls": 0,
            "logical_motion_calls": 0,
            "merged_motion_calls": 0,
            "max_motion_group": 0,
        }
        previous = self._rng()
        torch.manual_seed(seed)
        np.random.seed(seed % 2**32)
        random.seed(seed)
        self.solve_rng = self._rng()
        self._restore_rng(previous)
        for solver in dict.fromkeys(robot._solvers.values()):
            for name in ("get_fk", "get_ik", "get_jacobian"):
                self._bind(solver, name)
        from ._motion_batches import MotionBatches

        self.motions = MotionBatches(self)

    def _rng(self):
        return (
            self.torch.get_rng_state(),
            tuple(self.torch.cuda.get_rng_state(d) for d in self.devices),
            self.np.random.get_state(),
            self.random.getstate(),
            self.torch.is_grad_enabled(),
            self.torch.is_inference_mode_enabled(),
        )

    def _restore_rng(self, state):
        self.torch.set_rng_state(state[0])
        for device, value in zip(self.devices, state[1]):
            self.torch.cuda.set_rng_state(value, device)
        self.np.random.set_state(state[2])
        self.random.setstate(state[3])

    def _bind(self, solver, name):
        from greenlet import getcurrent

        original = getattr(solver, name, None)
        if original is None:
            return
        spec = signature(original)
        missing = object()
        before = solver.__dict__.get(name, missing)

        @wraps(original)
        def query(*args, **kwargs):
            task = getcurrent()
            if task not in self.tasks:
                counted = self.solo and self.direct_depth == 0
                if counted:
                    self.stats["solver_calls"] += 1
                    self.stats["logical_calls"] += 1
                    self.stats["max_group"] = max(self.stats["max_group"], 1)
                self.direct_depth += 1
                try:
                    return original(*args, **kwargs)
                finally:
                    self.direct_depth -= 1
            values = spec.bind(*args, **kwargs)
            values.apply_defaults()
            tensors, constants, count = {}, {}, None
            for key, value in values.arguments.items():
                if isinstance(value, self.torch.Tensor):
                    if value.ndim < 2 or (
                        count is not None and value.shape[0] != count
                    ):
                        return original(*args, **kwargs)
                    count = int(value.shape[0])
                    tensors[key] = value
                elif value is None or type(value) in (bool, int, float, str):
                    constants[key] = value
                elif type(value) is dict and not value:
                    constants[key] = value
                else:
                    return original(*args, **kwargs)
            if count is None:
                return original(*args, **kwargs)
            mode = (
                self.torch.is_grad_enabled(),
                self.torch.is_inference_mode_enabled(),
            )
            key = (
                id(solver),
                name,
                mode,
                tuple(
                    (key, tuple(value.shape[1:]), value.dtype, value.device)
                    for key, value in tensors.items()
                ),
                tuple((key, repr(value)) for key, value in constants.items()),
            )
            request = (key, original, spec, values, tensors, count, mode)
            self.tasks[task] = self._rng()
            self._restore_rng(self.owner_rng)
            return self.owner.switch(request)

        setattr(solver, name, query)
        self.saved.append((solver, name, before, missing))

    def _invoke(self, request, tensors):
        _, original, spec, values, _, _, mode = request
        if request[0][0] == "motion":
            with (
                self.torch.inference_mode(mode[1]),
                self.torch.set_grad_enabled(mode[0]),
            ):
                return original(tensors)
        arguments = dict(values.arguments)
        arguments.update(tensors)
        # BoundArguments retains variadic **kwargs and keyword-only arguments.
        bound = spec.bind_partial()
        bound.arguments.update(arguments)
        with self.torch.inference_mode(mode[1]), self.torch.set_grad_enabled(mode[0]):
            return original(*bound.args, **bound.kwargs)

    def _part(self, result, start, stop, total=None):
        if isinstance(result, self.torch.Tensor):
            if result.ndim and (total is None or result.shape[0] == total):
                return result[start:stop].clone()
            return result.clone()
        if isinstance(result, tuple):
            return tuple(self._part(value, start, stop, total) for value in result)
        if isinstance(result, dict):
            return {
                key: self._part(value, start, stop, total)
                for key, value in result.items()
            }
        if is_dataclass(result):
            return replace(
                result,
                **{
                    f.name: self._part(getattr(result, f.name), start, stop, total)
                    for f in fields(result)
                },
            )
        if result is None or type(result) in (bool, int, float, str):
            return result
        raise TypeError("Batched native solver returned an unsupported result")

    def gather(self, cursors):
        """Advance every native cursor once, merging compatible pending solves."""
        from greenlet import greenlet

        if len(cursors) == 1:
            row, cursor = next(iter(cursors.items()))
            self.solo = True
            try:
                action = cursor.next()
                return {} if action is None else {row: action}
            finally:
                self.solo = False
        self.owner_rng = self._rng()
        ready, waiting = {}, {}

        def advance(task, row, value=None, first=False):
            state = self.owner_rng if first else self.tasks[task]
            self._restore_rng(state)
            try:
                with (
                    self.torch.inference_mode(state[5]),
                    self.torch.set_grad_enabled(state[4]),
                ):
                    result = task.switch() if first else task.switch(value)
            finally:
                self._restore_rng(self.owner_rng)
            if task.dead:
                if result is not None:
                    ready[row] = result
            else:
                waiting[task] = (row, result)

        try:
            for row, cursor in cursors.items():
                task = greenlet(cursor.next, parent=self.owner)
                task.gr_context = copy_context()
                self.tasks[task] = self.owner_rng
                advance(task, row, first=True)
            while waiting:
                groups = defaultdict(list)
                for task, (row, request) in waiting.items():
                    groups[request[0]].append((task, row, request))
                waiting = {}
                for group in groups.values():
                    first = group[0][2]
                    tensors = {
                        key: self.torch.cat(
                            [entry[2][4][key] for entry in group], dim=0
                        )
                        for key in first[4]
                    }
                    self._restore_rng(self.solve_rng)
                    try:
                        result = self._invoke(first, tensors)
                    finally:
                        self.solve_rng = self._rng()
                        self._restore_rng(self.owner_rng)
                    if first[0][0] == "motion":
                        self.stats["motion_calls"] += 1
                        self.stats["logical_motion_calls"] += len(group)
                        self.stats["merged_motion_calls"] += len(group) > 1
                        self.stats["max_motion_group"] = max(
                            self.stats["max_motion_group"], len(group)
                        )
                    else:
                        self.stats["solver_calls"] += 1
                        self.stats["logical_calls"] += len(group)
                        self.stats["merged_calls"] += len(group) > 1
                        self.stats["max_group"] = max(
                            self.stats["max_group"], len(group)
                        )
                    start, parts = 0, []
                    total = sum(entry[2][5] for entry in group)
                    for task, row, request in group:
                        stop = start + request[5]
                        parts.append(self._part(result, start, stop, total))
                        start = stop
                    for (task, row, _), part in zip(group, parts):
                        advance(task, row, part)
            return ready
        finally:
            for task in tuple(self.tasks):
                if not task.dead:
                    self._restore_rng(self.tasks[task])
                    try:
                        state = self.tasks[task]
                        with (
                            self.torch.inference_mode(state[5]),
                            self.torch.set_grad_enabled(state[4]),
                        ):
                            task.throw()
                    finally:
                        self._restore_rng(self.owner_rng)
            self.tasks.clear()
            self._restore_rng(self.owner_rng)

    def close(self):
        self.motions.close()
        for solver, name, value, missing in reversed(self.saved):
            if value is missing:
                delattr(solver, name)
            else:
                setattr(solver, name, value)
        self.saved.clear()


@contextmanager
def batched_planning(robot, seed):
    batches = PlanningBatches(robot, seed)
    token = _ACTIVE.set(batches)
    try:
        yield batches
    finally:
        try:
            batches.close()
        finally:
            _ACTIVE.reset(token)
