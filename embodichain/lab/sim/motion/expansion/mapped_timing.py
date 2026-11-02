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

"""Control-grid retiming with explicit hard-event and derivative mapping."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import torch

from embodichain.data_pipeline.real2sim import TrajectoryTimeMap

from .contracts import TrajectoryTemplate
from .operators import allowed_phases

__all__ = ["MappedRetimeResult", "retime_with_time_map"]


@dataclass(frozen=True)
class MappedRetimeResult:
    """A retimed trajectory and source brackets for its new control grid."""

    template: TrajectoryTemplate
    time_map: TrajectoryTimeMap

    def __post_init__(self) -> None:
        if not isinstance(self.template, TrajectoryTemplate) or not isinstance(
            self.time_map, TrajectoryTimeMap
        ):
            raise TypeError("mapped result requires a trajectory template and time map")
        if self.template.positions.shape[0] != len(self.time_map.left):
            raise ValueError("mapped result must contain one source bracket per sample")


def _allocate_intervals(lengths: list[int], target: int) -> list[int]:
    """Apportion a phase budget without removing any protected-gap endpoint."""
    if target < len(lengths):
        raise ValueError("requested phase duration cannot preserve protected stencils")
    if not lengths:
        if target:
            raise ValueError("requested phase duration has no editable intervals")
        return []
    total = sum(lengths)
    quotas = [target * length / total for length in lengths]
    remaining = target - len(lengths)
    weights = [max(quota - 1.0, 0.0) for quota in quotas]
    weight_sum = sum(weights)
    if remaining == 0:
        return [1] * len(lengths)
    if weight_sum <= 0:
        raise ValueError("requested phase duration cannot be allocated")
    raw = [remaining * weight / weight_sum for weight in weights]
    extras = [math.floor(value) for value in raw]
    missing = remaining - sum(extras)
    order = sorted(range(len(lengths)), key=lambda i: (-(raw[i] - extras[i]), i))
    for index in order[:missing]:
        extras[index] += 1
    return [1 + value for value in extras]


def retime_with_time_map(
    template: TrajectoryTemplate,
    *,
    duration_scale: float,
    control_dt: float,
    protected_indices: tuple[int, ...] = (),
    max_samples: int = 1_000_000,
) -> MappedRetimeResult:
    """Retime authorized phase interiors while preserving protected stencils.

    Source control intervals must already equal ``control_dt``. Each selected
    phase's total duration is scaled and rounded up to a control interval;
    protected intervals retain their duration and editable gaps absorb the
    difference. A one-sample
    neighborhood of every protected event remains on its original local time
    grid. Only the gaps between those neighborhoods are retimed. Every output
    sample records its source bracket; protected samples have exact mappings.
    This is a sampled geometric transform, not a collision or dynamics check.

    Args:
        template: Full-joint source with explicitly authorized retime phases.
        duration_scale: Positive scale applied to eligible phase interiors.
        control_dt: Authoritative control period in seconds.
        protected_indices: Additional source samples that must remain exact.
        max_samples: Upper bound on the output sample count.

    Returns:
        Retimed template and its source-to-output mapping.

    Raises:
        ValueError: Timing, permissions, protected samples or budget are invalid.
    """
    if not isinstance(template, TrajectoryTemplate):
        raise TypeError("template must be a TrajectoryTemplate")
    for value, name in ((duration_scale, "duration_scale"), (control_dt, "control_dt")):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("max_samples must be at least two")
    count = template.positions.shape[0]
    if count < 2 or not torch.allclose(
        template.dt[1:], torch.full_like(template.dt[1:], control_dt), atol=1e-6, rtol=0
    ):
        raise ValueError("source trajectory must use the host control grid")
    if any(
        type(index) is not int or not 0 <= index < count for index in protected_indices
    ):
        raise ValueError("protected indices must address source samples")
    selected = allowed_phases(template, "retime")
    editable = set()
    for phase in selected:
        editable.update(range(phase.start_index + 1, phase.stop_index - 1))
    hard = set(range(count)) - editable
    hard.update(protected_indices)
    for phase in template.phases:
        hard.update((phase.start_index, phase.stop_index - 1))
    # Preserve both slopes used at an event, including the immediately
    # neighboring sample on each side of a contact boundary.
    protected = {
        neighbor
        for index in hard
        for neighbor in (index - 1, index, index + 1)
        if 0 <= neighbor < count
    }
    adjustable = [False] * count
    for phase in selected:
        for interval in range(phase.start_index + 1, phase.stop_index):
            if interval - 1 not in protected and interval not in protected:
                adjustable[interval] = True
    if not any(adjustable):
        raise ValueError(
            "no retimeable interior remains after protecting event stencils"
        )
    run_lengths: dict[int, int] = {}
    for phase in selected:
        runs = []
        index = phase.start_index + 1
        while index < phase.stop_index:
            if not adjustable[index]:
                index += 1
                continue
            first = index
            while index + 1 < phase.stop_index and adjustable[index + 1]:
                index += 1
            runs.append((first, index - first + 1))
            index += 1
        phase_intervals = phase.stop_index - phase.start_index - 1
        editable_intervals = sum(length for _, length in runs)
        protected_intervals = phase_intervals - editable_intervals
        scaled_intervals = phase_intervals * duration_scale
        if not math.isfinite(scaled_intervals) or scaled_intervals > max_samples:
            raise ValueError("retimed trajectory exceeds the sample budget")
        target_intervals = math.ceil(scaled_intervals - 1e-9)
        allocation = _allocate_intervals(
            [length for _, length in runs], target_intervals - protected_intervals
        )
        run_lengths.update(
            {first: length for (first, _), length in zip(runs, allocation)}
        )
    left = [0]
    right = [0]
    weights = [0.0]
    exact: dict[int, int] = {0: 0}
    uncontrolled = [
        index
        for index in range(template.positions.shape[1])
        if index not in template.controlled_joint_indices
    ]
    index = 1
    while index < count:
        if not adjustable[index]:
            left.append(index)
            right.append(index)
            weights.append(0.0)
            exact[index] = len(left) - 1
            index += 1
            continue
        first = index
        while index + 1 < count and adjustable[index + 1]:
            index += 1
        last = index
        if uncontrolled:
            original = template.positions[first - 1 : last + 1, uncontrolled]
            if not torch.equal(original, original[:1].expand_as(original)):
                raise ValueError(
                    "retiming may not alter an uncontrolled joint schedule"
                )
        intervals = last - first + 1
        new_intervals = run_lengths[first]
        if len(left) + new_intervals > max_samples:
            raise ValueError("retimed trajectory exceeds the sample budget")
        for step in range(1, new_intervals + 1):
            numerator = step * intervals
            offset, remainder = divmod(numerator, new_intervals)
            low = first - 1 + offset
            if remainder == 0:
                left.append(low)
                right.append(low)
                weights.append(0.0)
                exact[low] = len(left) - 1
            else:
                left.append(low)
                right.append(low + 1)
                weights.append(remainder / new_intervals)
        index += 1
    if len(left) > max_samples:
        raise ValueError("retimed trajectory exceeds the sample budget")
    if any(index not in exact for index in protected):
        raise RuntimeError("protected source sample was not mapped exactly")
    device = template.positions.device
    left_tensor = torch.tensor(left, dtype=torch.long, device=device)
    right_tensor = torch.tensor(right, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weights, dtype=template.positions.dtype, device=device)
    positions = torch.lerp(
        template.positions[left_tensor],
        template.positions[right_tensor],
        weight_tensor[:, None],
    )
    phases = tuple(
        replace(
            phase,
            start_index=exact[phase.start_index],
            stop_index=exact[phase.stop_index - 1] + 1,
        )
        for phase in template.phases
    )
    dt = torch.full((len(left),), control_dt, dtype=template.dt.dtype, device=device)
    dt[0] = 0
    result = replace(template, positions=positions, dt=dt, phases=phases)
    events = np.array(sorted(protected), dtype=np.int64)
    time_map = TrajectoryTimeMap(
        np.array(left, dtype=np.int64),
        np.array(right, dtype=np.int64),
        np.array(weights, dtype=np.float64),
        events,
        np.array([exact[item] for item in events], dtype=np.int64),
        0.0,
    )
    return MappedRetimeResult(result, time_map)
