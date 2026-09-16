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

"""Manipulability profiles, coverage bands, and guided residual sampling.

Local motion capability is scored with the shared helpers in
:mod:`embodichain.compute.kinematics`; this module only aligns those scores
with trajectory phases, partitions them into reference-normalized bands, and
uses them to steer joint-residual proposals. Jacobians are supplied by the
caller, so nothing here touches a robot, a solver, or a simulation backend.

Manipulability describes posture conditioning, not feasibility. A band-targeted
candidate still requires the same independent path, dynamic, and task checks as
any other proposal.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import math

import torch

from embodichain.compute.kinematics import (
    select_jacobian_rows,
    yoshikawa_manipulability,
)

from .contracts import TrajectoryPhase, TrajectoryTemplate
from .operators import _allowed_phases, joint_residual

__all__ = [
    "ManipulabilityProfile",
    "describe_manipulability",
    "ManipulabilityBands",
    "GuidedResidual",
    "manipulability_guided_residual",
]


@dataclass(frozen=True)
class ManipulabilityProfile:
    """Yoshikawa manipulability sampled along one trajectory.

    Args:
        scores: Per-sample manipulability with shape ``(T,)``.
        phase_ids: Ordered labels of the phases that were scored.
        phase_bottlenecks: Minimum score inside each phase, with shape ``(P,)``.
    """

    scores: torch.Tensor
    phase_ids: tuple[str, ...] = ()
    phase_bottlenecks: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.float64)
    )

    def __post_init__(self) -> None:
        phase_ids = tuple(self.phase_ids)
        if isinstance(self.phase_ids, str) or any(
            not isinstance(value, str) or not value for value in phase_ids
        ):
            raise ValueError("phase_ids must contain nonempty labels.")
        scores = self.scores.detach().to(device="cpu", dtype=torch.float64).clone()
        bottlenecks = (
            self.phase_bottlenecks.detach()
            .to(device="cpu", dtype=torch.float64)
            .clone()
        )
        if scores.ndim != 1 or scores.numel() < 1:
            raise ValueError("Manipulability scores must be a nonempty (T,) tensor.")
        if bottlenecks.shape != (len(phase_ids),):
            raise ValueError("phase_bottlenecks must hold one value per phase.")
        if not torch.isfinite(scores).all() or not torch.isfinite(bottlenecks).all():
            raise ValueError("Manipulability values must be finite.")
        if bool((scores < 0).any()) or bool((bottlenecks < 0).any()):
            raise ValueError("Manipulability values must be non-negative.")
        object.__setattr__(self, "phase_ids", phase_ids)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "phase_bottlenecks", bottlenecks)

    @property
    def bottleneck(self) -> float:
        """Return the smallest sampled manipulability along the trajectory.

        Returns:
            The minimum score, which marks the trajectory's tightest posture.
        """
        return float(self.scores.min())

    @property
    def mean(self) -> float:
        """Return the average sampled manipulability along the trajectory.

        Returns:
            The arithmetic mean of the per-sample scores.
        """
        return float(self.scores.mean())


def describe_manipulability(
    jacobians: torch.Tensor,
    *,
    phases: tuple[TrajectoryPhase, ...] = (),
    rows: str | Sequence[int] = "all",
) -> ManipulabilityProfile:
    """Score a sampled trajectory's Jacobians and align them with its phases.

    Args:
        jacobians: Per-sample Jacobians with shape ``(T, R, DOF)``, in the same
            sample order as the trajectory positions that produced them.
        phases: Optional half-open :class:`~.contracts.TrajectoryPhase` ranges
            whose bottleneck scores are reported separately.
        rows: Jacobian row selection forwarded to
            :func:`~embodichain.compute.kinematics.select_jacobian_rows`.

    Returns:
        An owned CPU profile holding per-sample scores and phase bottlenecks.

    Raises:
        ValueError: If the Jacobians are malformed or a phase is out of bounds.
    """
    if not isinstance(jacobians, torch.Tensor) or jacobians.ndim != 3:
        raise ValueError("jacobians must be a rank-3 (T, R, DOF) tensor.")
    if jacobians.shape[0] < 1 or min(jacobians.shape[1:]) < 1:
        raise ValueError("jacobians must contain at least one nonempty sample.")
    if not bool(torch.isfinite(jacobians).all()):
        raise ValueError("jacobians must contain finite values.")
    # Score in float64: the Yoshikawa determinant of a float32 Jacobian can
    # underflow to zero and misreport a well-conditioned posture as singular.
    selected = select_jacobian_rows(
        jacobians.detach().to(device="cpu", dtype=torch.float64), rows
    )
    scores = yoshikawa_manipulability(selected)
    labels, bottlenecks = [], []
    for phase in phases:
        if phase.stop_index > scores.shape[0]:
            raise ValueError("Phase ranges must lie within the scored samples.")
        labels.append(phase.phase_id)
        bottlenecks.append(scores[phase.start_index : phase.stop_index].min())
    return ManipulabilityProfile(
        scores,
        tuple(labels),
        (
            torch.stack(bottlenecks)
            if bottlenecks
            else torch.zeros(0, dtype=torch.float64)
        ),
    )


class ManipulabilityBands:
    """Ordered manipulability bands normalized by one reference trajectory.

    Ratios are taken against a reference value, usually the reference
    trajectory's bottleneck manipulability, so band membership stays comparable
    across robots and tasks. Band ``0`` holds the least manipulable postures and
    the final band the most manipulable ones; both outer bands are open-ended.

    Args:
        edges: Strictly increasing positive ratio boundaries.
        reference: Positive reference manipulability that normalizes scores.
    """

    def __init__(self, edges: Sequence[float], *, reference: float) -> None:
        values = tuple(float(edge) for edge in edges)
        if not values or any(not math.isfinite(edge) or edge <= 0 for edge in values):
            raise ValueError("Band edges must be finite positive ratios.")
        if any(low >= high for low, high in zip(values, values[1:])):
            raise ValueError("Band edges must increase strictly.")
        if type(reference) not in (float, int) or not math.isfinite(reference):
            raise ValueError("The band reference must be a finite number.")
        if reference <= 0:
            raise ValueError("The band reference manipulability must be positive.")
        self._edges = values
        self._reference = float(reference)

    @property
    def count(self) -> int:
        """Return the number of bands, one more than the number of edges.

        Returns:
            The count of ordered bands, including both open-ended outer bands.
        """
        return len(self._edges) + 1

    def ratio(self, value: float) -> float:
        """Normalize an absolute manipulability by the configured reference.

        Args:
            value: Finite non-negative manipulability score.

        Returns:
            The score divided by the reference manipulability.
        """
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError("Manipulability must be finite and non-negative.")
        return float(value) / self._reference

    def band_of(self, value: float) -> int:
        """Return the band index owning an absolute manipulability score.

        Args:
            value: Finite non-negative manipulability score.

        Returns:
            The index in ``[0, count)`` of the band containing the score's ratio.
        """
        ratio = self.ratio(value)
        return sum(ratio >= edge for edge in self._edges)


@dataclass(frozen=True)
class GuidedResidual:
    """One selected residual proposal and the evidence used to select it.

    Args:
        template: The chosen candidate, still requiring independent validation.
        profile: Manipulability measured along the chosen candidate.
        band: Band index containing the chosen candidate's bottleneck.
        matched: Whether the chosen candidate reached the requested band.
        proposals_scored: Number of proposals actually sampled and scored.
    """

    template: TrajectoryTemplate
    profile: ManipulabilityProfile
    band: int
    matched: bool
    proposals_scored: int


def manipulability_guided_residual(
    template: TrajectoryTemplate,
    *,
    joint_limits: torch.Tensor,
    normalized_scale: float,
    generator: torch.Generator,
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
    bands: ManipulabilityBands,
    target_band: int,
    proposals: int = 4,
    rows: str | Sequence[int] = "all",
) -> GuidedResidual:
    """Sample joint residuals and keep the one nearest a requested band.

    Each proposal is an independent :func:`~.operators.joint_residual` draw from
    the supplied local generator, scored by the manipulability of the postures
    it actually visits. Steering toward a lower band deliberately retains
    tighter postures that remain task-valid, which uniform sampling around a
    comfortable reference rarely produces. Proposals that leave the joint limits
    are skipped; if every proposal fails, the last rejection is raised.

    Selection never asserts feasibility. The winning candidate still requires
    path, dynamic, and task validation exactly like an unguided residual.

    Args:
        template: Annotated full-joint reference permitting ``joint_residual``.
        joint_limits: Finite lower/upper limits with shape ``(D, 2)``.
        normalized_scale: Maximum offset as a fraction of each joint range.
        generator: Explicit local generator; global RNG is never consumed.
        jacobian_fn: Maps candidate positions ``(T, D)`` to Jacobians ``(T, R, DOF)``.
        bands: Reference-normalized bands that classify a candidate.
        target_band: Requested band index within ``bands``.
        proposals: Number of residual draws to score, at least one.
        rows: Jacobian row selection used when scoring candidates.

    Returns:
        The selected candidate with its measured profile and band.

    Raises:
        ValueError: If permissions, arguments, or every sampled residual fail.
    """
    _allowed_phases(template, "joint_residual")
    if not isinstance(bands, ManipulabilityBands):
        raise ValueError("bands must be a ManipulabilityBands instance.")
    if type(target_band) is not int or not 0 <= target_band < bands.count:
        raise ValueError("target_band must be an index within the configured bands.")
    if type(proposals) is not int or proposals < 1:
        raise ValueError("proposals must be an integer of at least one.")
    if not callable(jacobian_fn):
        raise ValueError("jacobian_fn must be callable.")
    best: GuidedResidual | None = None
    scored, rejection = 0, None
    for _ in range(proposals):
        try:
            candidate = joint_residual(
                template,
                joint_limits=joint_limits,
                normalized_scale=normalized_scale,
                generator=generator,
            )
        except ValueError as error:
            rejection = error
            continue
        jacobians = jacobian_fn(candidate.positions)
        if (
            not isinstance(jacobians, torch.Tensor)
            or jacobians.ndim != 3
            or jacobians.shape[0] != candidate.positions.shape[0]
        ):
            raise ValueError(
                "jacobian_fn must return one (R, DOF) Jacobian per candidate sample."
            )
        scored += 1
        profile = describe_manipulability(jacobians, phases=candidate.phases, rows=rows)
        band = bands.band_of(profile.bottleneck)
        # Keep the first proposal reaching the requested band; otherwise keep
        # the nearest band, breaking ties by proposal order for determinism.
        if best is None or abs(band - target_band) < abs(best.band - target_band):
            best = GuidedResidual(candidate, profile, band, band == target_band, scored)
        if best.matched:
            break
    if best is None:
        raise rejection
    return GuidedResidual(best.template, best.profile, best.band, best.matched, scored)
