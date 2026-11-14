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


"""Fuse compatible fixed-grid native motion requests before scalar transforms."""

from __future__ import annotations

from copy import copy
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from functools import wraps

__all__: list[str] = []


class MotionBatches:
    """Keep selected physical rows and native fixed-grid rejection semantics."""

    def __init__(self, broker):
        from greenlet import getcurrent
        from embodichain.lab.sim.motion.motion_generator import MotionGenerator

        self.broker = broker
        self.original = original = MotionGenerator.generate

        def direct(generator, targets, options, defer_recovery):
            previous = broker.motion_passthrough
            broker.motion_passthrough = True
            try:
                return original(
                    generator, targets, options, defer_recovery=defer_recovery
                )
            finally:
                broker.motion_passthrough = previous

        @wraps(original)
        def generate(generator, target_states, options=None, *, defer_recovery=False):
            from embodichain.lab.sim.motion._budget import _ACTIVE_BUDGET

            task = getcurrent()
            if (
                task not in broker.tasks
                or options is None
                or options.strategy != "ik_interp"
                or options.interpolation_mode != "fixed_grid"
                or options.segment_sample_count_retries
                or options.segment_continuous_ik_recovery
                or options.cartesian_path_sampler is not None
                or generator.collision_world_info is not None
                or generator._env_ids is None
                or _ACTIVE_BUDGET.get() is not None
            ):
                return direct(generator, target_states, options, defer_recovery)
            tensors = {}
            try:
                structure = self._describe(
                    (target_states, options), tensors, len(generator._env_ids)
                )
            except TypeError:
                return direct(generator, target_states, options, defer_recovery)
            tensors["rows"] = broker.torch.tensor(
                generator._env_ids, dtype=broker.torch.long
            )
            mode = (
                broker.torch.is_grad_enabled(),
                broker.torch.is_inference_mode_enabled(),
            )
            key = (
                "motion",
                id(generator.robot),
                type(generator.planner),
                repr(generator.planner.cfg),
                structure,
                defer_recovery,
                mode,
            )

            def invoke(merged):
                targets, settings = self._restore((target_states, options), merged)
                batched = copy(generator)
                batched.planner = copy(generator.planner)
                batched._env_ids = tuple(merged["rows"].tolist())
                batched.planner._planning_env_ids = batched._env_ids
                return original(
                    batched, targets, settings, defer_recovery=defer_recovery
                )

            request = (key, invoke, None, None, tensors, len(generator._env_ids), mode)
            broker.tasks[task] = broker._rng()
            broker._restore_rng(broker.owner_rng)
            return broker.owner.switch(request)

        self.generate = generate

    def _describe(self, value, tensors, count, path=""):
        torch = self.broker.torch
        if isinstance(value, torch.Tensor):
            if value.ndim < 1 or value.shape[0] != count:
                raise TypeError("Motion tensors must retain the selected batch")
            tensors[path] = value
            return ("tensor", tuple(value.shape[1:]), value.dtype, value.device)
        if is_dataclass(value):
            return (
                type(value),
                tuple(
                    (
                        f.name,
                        self._describe(
                            getattr(value, f.name), tensors, count, path + "/" + f.name
                        ),
                    )
                    for f in fields(value)
                ),
            )
        if isinstance(value, (tuple, list)):
            return (
                type(value),
                tuple(
                    self._describe(v, tensors, count, path + "/" + str(i))
                    for i, v in enumerate(value)
                ),
            )
        if (
            value is None
            or type(value) in (bool, int, float, str)
            or isinstance(value, Enum)
        ):
            return (type(value), value)
        raise TypeError("Unsupported mutable motion option")

    def _restore(self, value, tensors, path=""):
        if isinstance(value, self.broker.torch.Tensor):
            return tensors[path]
        if is_dataclass(value):
            return replace(
                value,
                **{
                    f.name: self._restore(
                        getattr(value, f.name), tensors, path + "/" + f.name
                    )
                    for f in fields(value)
                },
            )
        if isinstance(value, (tuple, list)):
            return type(value)(
                self._restore(v, tensors, path + "/" + str(i))
                for i, v in enumerate(value)
            )
        return value

    def close(self):
        return None
