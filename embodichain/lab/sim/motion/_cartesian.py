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

"""Explicit Cartesian segment sampling and bounded, unchanged-geometry recovery."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
import torch

from .ik import sample_cartesian_targets, resolve_continuous_ik_path
from .planners.utils import MotionRecoveryTrace, MoveType, PlanResult, PlanState

if TYPE_CHECKING:
    from .motion_generator import MotionGenerator, MotionGenOptions

__all__: list[str] = []


def _generate_cartesian_motion(
    generator: MotionGenerator,
    targets: list[PlanState],
    options: MotionGenOptions,
    *,
    defer_recovery: bool = False,
) -> PlanResult:
    """Sample declared keyframes, then execute the explicit segment recovery budget.

    Args:
        generator: Stateful robot/solver owner; no action or task objects enter.
        targets: Ordered end-effector keyframes in the same world frame.
        options: Explicit B1 fixed-grid sequential IK options with controlled
            start, count and interval. The two segment fields are consumed here.
        defer_recovery: A caller-owned recovery phase may first inspect the
            nominal failure before spending this request's motion alternatives.

    Returns:
        Standard plan result and its independent local attempt receipt. Dense
        retries advance the caller's random stream in source order. On exhaustion, continuous IK queries the
        original sampled path and time grid, not the last rejected alternative.
        Its own Torch fork remains in the shared continuous resolver.
    """
    if (
        type(defer_recovery) is not bool
        or options.strategy != "ik_interp"
        or options.interpolation_mode != "fixed_grid"
        or options.ik_seed_mode != "sequential"
        or options.start_qpos is None
        or options.start_qpos.ndim != 2
        or options.control_part is None
        or options.sample_count is None
        or options.interpolation_dt is None
        or not targets
        or any(t.move_type is not MoveType.EEF_MOVE or t.xpos is None for t in targets)
        or not options.is_linear
        or options.preserve_cartesian_samples
        or options.cartesian_path_sampler is not None
    ):
        raise ValueError(
            "Cartesian segment generation requires explicit unsampled fixed-grid sequential IK keyframes."
        )
    enabled = bool(
        options.segment_sample_count_retries or options.segment_continuous_ik_recovery
    )
    if enabled and options.start_qpos.shape[0] != 1:
        raise ValueError("Cartesian segment recovery currently supports B1 only.")
    start = generator.robot.compute_fk(
        qpos=options.start_qpos,
        name=options.control_part,
        to_matrix=True,
        **(
            {"env_ids": generator._env_ids}
            if getattr(generator, "_env_ids", None) is not None
            else {}
        ),
    )
    base = replace(
        options,
        segment_sample_count_retries=(),
        segment_continuous_ik_recovery=False,
        preserve_cartesian_samples=True,
        is_linear=True,
    )

    def build(count: int) -> tuple[list[PlanState], MotionGenOptions]:
        return sample_cartesian_targets(start, targets, count), replace(
            base, sample_count=count
        )

    sampled, original = build(options.sample_count)
    first = generator._generate_once(sampled, options=original)
    counts = [options.sample_count]
    retries = options.segment_sample_count_retries

    def finish(result: PlanResult, *, used_continuity: bool = False) -> PlanResult:
        result.recovery = MotionRecoveryTrace(tuple(counts), used_continuity)
        return result

    # Preserve the source eligibility boundary even under outer action retries.
    if (
        first.success.any()
        or defer_recovery
        or (retries and options.sample_count > retries[0])
    ):
        return finish(first)
    for count in retries:
        if count <= options.sample_count:
            continue
        dense, dense_options = build(count)
        counts.append(count)
        candidate = generator._generate_once(dense, options=dense_options)
        if candidate.success.any():
            return finish(candidate)
    if options.segment_continuous_ik_recovery:
        recovered = resolve_continuous_ik_path(
            generator, sampled, original, operation="Cartesian segment"
        )
        if recovered is not None:
            recovered = generator._normalize_plan_result(recovered, sampled, original)
            return finish(recovered, used_continuity=True)
    return finish(first)
