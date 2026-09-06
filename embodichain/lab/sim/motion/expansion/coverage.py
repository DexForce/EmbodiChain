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

"""Phase-aligned joint geometry and measured timing coverage for one scene case."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import TrajectoryPhase

__all__ = ["TrajectoryDescriptor", "describe_trajectory", "CoverageIndex"]


@dataclass(frozen=True)
class TrajectoryDescriptor:
    """Measured joint geometry and elapsed times at uniform phase progress.

    Args:
        phase_ids: Ordered phase identities, including contact/free mode.
        geometry: Normalized joint samples, with shape ``(P, S, D)``.
        timing: Elapsed seconds within each phase, with shape ``(P, S)``.
    """

    phase_ids: tuple[tuple[str, str], ...]
    geometry: torch.Tensor
    timing: torch.Tensor

    def __post_init__(self) -> None:
        phase_ids = []
        for pair in self.phase_ids:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(not isinstance(value, str) or not value for value in pair)
            ):
                raise ValueError("phase_ids must contain pairs of nonempty labels.")
            phase_ids.append(tuple(pair))
        object.__setattr__(self, "phase_ids", tuple(phase_ids))
        geometry = self.geometry.detach().to(device="cpu", dtype=torch.float64).clone()
        timing = self.timing.detach().to(device="cpu", dtype=torch.float64).clone()
        if geometry.ndim != 3 or timing.shape != geometry.shape[:2]:
            raise ValueError("Descriptor geometry/timing shapes must be (P,S,D)/(P,S).")
        if len(self.phase_ids) != geometry.shape[0] or min(geometry.shape) < 1:
            raise ValueError("Descriptor phases must match nonempty geometry.")
        if not torch.isfinite(geometry).all() or not torch.isfinite(timing).all():
            raise ValueError("Descriptor values must be finite.")
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "timing", timing)


def describe_trajectory(
    positions: torch.Tensor,
    timestamps: torch.Tensor,
    joint_limits: torch.Tensor,
    *,
    phases: tuple[TrajectoryPhase, ...] = (),
    samples_per_phase: int = 32,
) -> TrajectoryDescriptor:
    """Describe an actual rollout without rewarding additional time samples.

    Geometry is sampled by normalized joint arc length within each phase. Timing
    includes the phase duration, including stationary waits. Unlabelled gaps are
    included as separate phases so no measured movement is silently discarded.

    Args:
        positions: Actual joint positions with shape ``(T, D)``.
        timestamps: Strictly increasing observation timestamps, shape ``(T,)``.
        joint_limits: Finite lower/upper joint limits, shape ``(D, 2)``.
        phases: Optional half-open ranges in the actual observation sequence.
        samples_per_phase: Number of progress samples, at least two.

    Returns:
        Owned CPU descriptor separating geometry from timing.

    Raises:
        ValueError: If samples, limits, times or phase ranges are invalid.
    """
    q = positions.detach().to(device="cpu", dtype=torch.float64)
    t = timestamps.detach().to(device="cpu", dtype=torch.float64)
    limits = joint_limits.detach().to(device="cpu", dtype=torch.float64)
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] < 1:
        raise ValueError("positions must contain at least two observations.")
    if t.shape != (q.shape[0],) or limits.shape != (q.shape[1], 2):
        raise ValueError("timestamps or joint_limits shape does not match positions.")
    if not all(torch.isfinite(x).all() for x in (q, t, limits)):
        raise ValueError("Trajectory and joint limits must be finite.")
    if not (t[1:] > t[:-1]).all() or not (limits[:, 1] > limits[:, 0]).all():
        raise ValueError("Times and joint-limit intervals must increase strictly.")
    if type(samples_per_phase) is not int or samples_per_phase < 2:
        raise ValueError("samples_per_phase must be at least two.")
    q = (q - limits[:, 0]) / (limits[:, 1] - limits[:, 0])
    ranges: list[tuple[str, str, int, int]] = []
    cursor = 0
    for phase in phases:
        if phase.start_index < cursor or phase.stop_index > q.shape[0]:
            raise ValueError("Actual phase ranges must be ordered and in bounds.")
        if phase.start_index > cursor:
            ranges.append(
                (f"gap_{len(ranges)}", "unlabelled", cursor, phase.start_index)
            )
        ranges.append((phase.phase_id, phase.kind, phase.start_index, phase.stop_index))
        cursor = phase.stop_index
    if cursor < q.shape[0]:
        ranges.append((f"gap_{len(ranges)}", "unlabelled", cursor, q.shape[0]))

    geometry, timing = [], []
    for _, _, start, stop in ranges:
        # Include the entering edge of a phase; otherwise disjoint sample ranges
        # would leave their connecting motion out of the descriptor.
        first = max(0, start - 1)
        values, times = q[first:stop], t[first:stop] - t[first]
        distances = torch.linalg.vector_norm(values[1:] - values[:-1], dim=-1)
        arc = torch.cat((torch.zeros(1, dtype=q.dtype), distances.cumsum(0)))
        if float(arc[-1]) <= 1e-12:
            geometry.append(values[:1].expand(samples_per_phase, -1).clone())
            timing.append(
                torch.linspace(0, float(times[-1]), samples_per_phase, dtype=q.dtype)
            )
            continue
        # Remove repeated arc coordinates for interpolation. The final sample's
        # time is kept separately to retain terminal waits.
        keep = torch.cat((torch.ones(1, dtype=torch.bool), distances > 1e-12))
        unique_arc, unique_q, unique_t = arc[keep], values[keep], times[keep]
        grid = torch.linspace(0, float(arc[-1]), samples_per_phase, dtype=q.dtype)
        upper = torch.searchsorted(unique_arc, grid).clamp(1, len(unique_arc) - 1)
        lower = upper - 1
        weight = (grid - unique_arc[lower]) / (unique_arc[upper] - unique_arc[lower])
        geometry.append(torch.lerp(unique_q[lower], unique_q[upper], weight[:, None]))
        elapsed = torch.lerp(unique_t[lower], unique_t[upper], weight)
        elapsed[-1] = times[-1]
        timing.append(elapsed)
    return TrajectoryDescriptor(
        tuple((name, kind) for name, kind, _, _ in ranges),
        torch.stack(geometry),
        torch.stack(timing),
    )


@dataclass
class _Reservation:
    family_id: str
    descriptor: TrajectoryDescriptor
    committed: bool = False


class CoverageIndex:
    """Reserve near-distinct measured trajectories and count confirmed geometry.

    One instance belongs to one scene case. Pending writes occupy capacity but
    do not contribute to formal coverage. A timing variant cannot increase the
    geometry count even if small physical tracking differences are measured.

    Args:
        geometry_tolerance: Maximum normalized joint difference for near copies.
        target_per_geometry: Maximum distinct timing variants per geometry.
        timing_tolerance_s: Maximum timing difference for duplicate episodes.
    """

    def __init__(
        self,
        *,
        geometry_tolerance: float = 0.01,
        target_per_geometry: int = 4,
        timing_tolerance_s: float = 1e-6,
    ) -> None:
        import math

        if any(
            not math.isfinite(x) or x < 0
            for x in (geometry_tolerance, timing_tolerance_s)
        ):
            raise ValueError("Coverage tolerances must be finite and non-negative.")
        if (
            isinstance(target_per_geometry, bool)
            or not isinstance(target_per_geometry, int)
            or target_per_geometry < 1
        ):
            raise ValueError("target_per_geometry must be a positive integer.")
        self._geometry_tolerance = geometry_tolerance
        self._timing_tolerance = timing_tolerance_s
        self._target = target_per_geometry
        self._entries: dict[str, _Reservation] = {}
        self._aliases: dict[str, str] = {}

    @property
    def geometry_count(self) -> int:
        """Number of distinct geometries with a confirmed persistence receipt.

        Returns:
            The number of geometry groups containing at least one confirmed commit.
        """
        return len(
            {entry.family_id for entry in self._entries.values() if entry.committed}
        )

    def reserve(
        self,
        commit_id: str,
        geometry_family_id: str,
        descriptor: TrajectoryDescriptor,
    ) -> bool:
        """Reserve a distinct measured rollout before submitting its payload.

        Args:
            commit_id: Stable, unique persistence identity.
            geometry_family_id: Family shared by proposed timing variants.
            descriptor: Descriptor calculated from the actual execution.

        Returns:
            False for a near duplicate or a geometry whose quota is full.
        """
        if not commit_id or not geometry_family_id:
            raise ValueError("Commit and geometry-family IDs must be nonempty.")
        if commit_id in self._entries:
            raise ValueError("A commit ID already has a coverage reservation.")
        family = self._aliases.get(geometry_family_id, geometry_family_id)
        if geometry_family_id not in self._aliases:
            for entry in self._entries.values():
                old = entry.descriptor
                compatible = (
                    old.phase_ids == descriptor.phase_ids
                    and old.geometry.shape == descriptor.geometry.shape
                )
                if compatible and torch.allclose(
                    old.geometry,
                    descriptor.geometry,
                    atol=self._geometry_tolerance,
                    rtol=0,
                ):
                    family = entry.family_id
                    break
        matching = [
            entry for entry in self._entries.values() if entry.family_id == family
        ]
        for entry in matching:
            old = entry.descriptor
            compatible = (
                old.phase_ids == descriptor.phase_ids
                and old.geometry.shape == descriptor.geometry.shape
            )
            if compatible and torch.allclose(
                old.timing, descriptor.timing, atol=self._timing_tolerance, rtol=0
            ):
                return False
        if len(matching) >= self._target:
            return False
        owned = TrajectoryDescriptor(
            descriptor.phase_ids, descriptor.geometry, descriptor.timing
        )
        self._entries[commit_id] = _Reservation(family, owned)
        self._aliases[geometry_family_id] = family
        return True

    def confirm(self, commit_id: str) -> None:
        """Confirm a reserved commit; repeated confirmations are idempotent.

        Args:
            commit_id: Previously reserved persistence identity.
        """
        self._entries[commit_id].committed = True

    def release(self, commit_id: str) -> None:
        """Release an uncommitted reservation after write failure.

        Args:
            commit_id: Previously reserved persistence identity.

        Raises:
            ValueError: If persistence has already been confirmed.
        """
        entry = self._entries[commit_id]
        if entry.committed:
            raise ValueError("Committed coverage cannot be released.")
        del self._entries[commit_id]
