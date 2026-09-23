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

"""Distinct trajectory variants for one reference whose waypoints are already fixed.

This module covers a narrower case than affordance expansion. The reference
trajectory's annotated waypoints are already settled, and only the motion
between them varies. Every operator applied here preserves each annotated
phase endpoint exactly, so contacts, grasps and placements are untouched.

Three qpos-level factors are combined here:

* **spatial** — :func:`~.operators.joint_residual` or :func:`~.operators.via_points`
  reshape the joint path between waypoints.
* **ik** — :func:`~.operators.nullspace_residual` moves through the redundancy
  of a declared task, changing arm posture while holding the constrained rows.
* **timing** — :func:`~.operators.retime` rescales and reshapes free-phase time.

Approach-direction variation is **not** applied here. It changes a Cartesian
standoff pose and must therefore be consumed before IK and path planning, by
whichever component builds the waypoints. :func:`sample_approach_cone` samples
the angles for that upstream caller; the resulting poses are the caller's to
replan and validate.

Motion-limit validation and geometric deduplication filter the proposals, but
neither establishes collision freedom nor task success. Execution, reset and
persistence remain host responsibilities.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch

from embodichain.compute.kinematics import select_jacobian_rows

from .cfg import SPATIAL_METHODS, TrajectoryAugmentationCfg
from .contracts import (
    CandidateIdentity,
    CandidateTrajectoryBatch,
    SceneCase,
    TrajectoryTemplate,
    _digest,
)
from .coverage import CoverageIndex, describe_trajectory
from .operators import (
    ProposalRejected,
    joint_residual,
    nullspace_residual,
    retime,
    validate_motion_limits,
    via_points,
)

__all__ = [
    "NOMINAL_OPERATOR",
    "default_variant_factors",
    "TrajectoryVariant",
    "TrajectoryVariantSet",
    "apply_trajectory_variant",
    "expand_trajectory_variants",
    "plan_trajectory_variants",
    "sample_approach_cone",
]

NOMINAL_OPERATOR = "none"
"""Spatial-operator name of the preserved unmodified reference branch."""

_ATTEMPT_MULTIPLIER = 4
"""Default proposal budget per requested variant when the caller gives none."""


# Magnitudes measured on the PickUp variant tutorial: a larger joint offset
# starts disturbing the grasp, and a deduplication tolerance at the same
# magnitude as the offset discards variants that really do differ.
_DEFAULT_JOINT_OFFSET_SCALE = 0.01
_DEFAULT_VIA_COUNT = 3
_DEFAULT_DEDUP_TOLERANCE = 0.004
_DEFAULT_TIMING_VARIANTS_PER_GEOMETRY = 4
_DEFAULT_DURATION_SCALES = (1.0, 1.25)


def default_variant_factors(
    *, seed: int = 0, redundancy: bool = True
) -> TrajectoryAugmentationCfg:
    """Enable every implemented factor at magnitudes known to stay executable.

    This is what the expansion helpers use when a caller asks for a number of
    variants without describing how to produce them. Callers that do care can
    build a :class:`~.cfg.TrajectoryAugmentationCfg` themselves; an explicitly
    constructed configuration is never overridden.

    The offsets are fractions of each joint's **declared** range, so their
    absolute size depends on how generous the robot's limits are. They are a
    starting point, not a guarantee: the host still has to execute and validate
    what comes back.

    Args:
        seed: Base seed for the sampling streams.
        redundancy: Whether the caller can supply task Jacobians. When false the
            null-space factor is left off instead of proposing variants that
            would be rejected for missing inputs.

    Returns:
        A configuration enabling the spatial, timing and optional ik factors.
    """
    return TrajectoryAugmentationCfg.from_mapping(
        {
            "seed": seed,
            "factors": {
                "spatial": {
                    "enabled": True,
                    "method": list(SPATIAL_METHODS),
                    "joint_offset_scale": _DEFAULT_JOINT_OFFSET_SCALE,
                    "via_count": _DEFAULT_VIA_COUNT,
                },
                "ik": {"enabled": bool(redundancy)},
                "timing": {
                    "enabled": True,
                    "duration_scales": list(_DEFAULT_DURATION_SCALES),
                    "profiles": ["uniform", "ease_in", "ease_out"],
                },
            },
            "coverage": {
                "target_per_cell": _DEFAULT_TIMING_VARIANTS_PER_GEOMETRY,
                "joint_dedup_normalized_tol": _DEFAULT_DEDUP_TOLERANCE,
            },
        }
    )


@dataclass(frozen=True)
class TrajectoryVariant:
    """One deterministic combination of augmentation factors.

    Args:
        ordinal: Zero-based enumeration index; ordinal zero is the reference.
        spatial_operator: Applied joint-path operator, or ``"none"``.
        duration_scale: Free-phase duration factor passed to retiming.
        timing_profile: Within-phase time warp passed to retiming.
    """

    ordinal: int
    spatial_operator: str
    duration_scale: float
    timing_profile: str

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        if not math.isfinite(self.duration_scale) or self.duration_scale <= 0:
            raise ValueError("duration_scale must be finite and positive")

    @property
    def is_nominal(self) -> bool:
        """Whether this variant leaves the reference trajectory unchanged.

        Returns:
            ``True`` when no spatial operator and no retiming apply.
        """
        return (
            self.spatial_operator == NOMINAL_OPERATOR
            and self.duration_scale == 1.0
            and self.timing_profile == "uniform"
        )

    @property
    def factors(self) -> dict[str, float | int | str]:
        """Return this variant's provenance for candidate-row metadata.

        Returns:
            Finite JSON-compatible factor values describing the variant.
        """
        return {
            "variant_ordinal": self.ordinal,
            "spatial_operator": self.spatial_operator,
            "duration_scale": float(self.duration_scale),
            "timing_profile": self.timing_profile,
        }


@dataclass(frozen=True)
class TrajectoryVariantSet:
    """Accepted candidate rows together with the accounting that produced them.

    Args:
        candidates: Accepted rows, the first of which is the reference branch.
        variants: One accepted variant per candidate row, in row order.
        attempted: Number of proposals evaluated, including rejections.
        rejected: Count of rejected proposals by reason.
        cfg: Settings that produced these variants, including a policy resolved
            on the caller's behalf. ``None`` only when a caller builds this
            value itself.
    """

    candidates: CandidateTrajectoryBatch
    variants: tuple[TrajectoryVariant, ...]
    attempted: int
    rejected: Mapping[str, int]
    cfg: TrajectoryAugmentationCfg | None = None

    def __post_init__(self) -> None:
        if len(self.variants) != len(self.candidates.identities):
            raise ValueError("variants must contain one entry per candidate row")
        object.__setattr__(self, "rejected", dict(self.rejected))


def _reject_cartesian_factors(cfg: TrajectoryAugmentationCfg) -> None:
    """Refuse a configuration whose factors cannot produce a qpos variant.

    The approach factor moves a Cartesian standoff pose, which has to be
    replanned through IK before it is a trajectory. Accepting it here would
    return an unchanged trajectory to a caller who asked for approach
    variation.

    Args:
        cfg: Resolved augmentation settings.

    Raises:
        ValueError: If the approach factor is enabled.
    """
    if cfg.factors.approach.enabled:
        raise ValueError(
            "the approach factor produces Cartesian standoff poses and cannot "
            "be applied to a qpos reference; consume it with "
            "perturb_approach_direction before planning, then disable it here"
        )


def plan_trajectory_variants(
    count: int, cfg: TrajectoryAugmentationCfg | None = None
) -> tuple[TrajectoryVariant, ...]:
    """Enumerate deterministic factor combinations without touching a trajectory.

    Ordinal zero is always the unmodified reference. Later ordinals cycle
    through the enabled spatial operators, then the duration scales, then the
    within-phase time warps, so a small request still spreads across every
    enabled dimension instead of exhausting one. Stochastic operators repeat
    with a different local seed, which is why the enumeration is unbounded
    while the number of distinct results is not.

    Args:
        count: Number of variants to enumerate, at least one.
        cfg: Augmentation settings whose enabled factors select the
            combinations. Defaults to :func:`default_variant_factors` with
            redundancy off, because this function cannot know whether the
            caller has task Jacobians. Pass
            ``default_variant_factors(redundancy=True)``, or use
            :func:`expand_trajectory_variants`, to include posture variants.

    Returns:
        Deterministically ordered variants beginning with the reference branch.

    Raises:
        ValueError: If ``count`` is not a positive integer, or if the
            configuration enables a factor that cannot produce a qpos variant.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    # Planning alone cannot see whether the caller can supply Jacobians, so an
    # unconfigured enumeration offers only factors that need no extra input.
    # Otherwise the documented plan-then-apply workflow would hand
    # apply_trajectory_variant a posture variant it cannot execute.
    cfg = default_variant_factors(redundancy=False) if cfg is None else cfg
    factors = cfg.factors
    _reject_cartesian_factors(cfg)
    operators: list[str] = []
    if factors.spatial.enabled:
        operators.extend(factors.spatial.method)
    if factors.ik.enabled:
        operators.append(factors.ik.method)
    if not operators:
        operators.append(NOMINAL_OPERATOR)
    # Advance the duration scale before the within-phase warp: rescaling
    # changes the whole free motion, so it separates variants faster.
    timings: list[tuple[float, str]] = [
        (scale, profile)
        for profile in factors.timing.profiles
        for scale in factors.timing.duration_scales
    ]
    variants = [TrajectoryVariant(0, NOMINAL_OPERATOR, 1.0, "uniform")]
    for ordinal in range(1, count):
        step = ordinal - 1
        scale, profile = timings[(step // len(operators)) % len(timings)]
        variants.append(
            TrajectoryVariant(ordinal, operators[step % len(operators)], scale, profile)
        )
    return tuple(variants)


def sample_approach_cone(
    cfg: TrajectoryAugmentationCfg,
    *,
    count: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tilt angles for :func:`~.operators.perturb_approach_direction`.

    The first pair is the nominal direction with zero tilt, preserving the
    reference branch. Remaining pairs are uniform over the spherical cap area,
    so directions do not bunch near the cone axis. Sampling is a geometric
    proposal only: reachability, clearance and grasp validity belong to the
    caller that replans the resulting poses.

    Args:
        cfg: Augmentation settings whose approach factor must be enabled.
        count: Number of direction candidates, at least one.
        generator: Explicit local generator; global RNG is never consumed.

    Returns:
        Polar and azimuth angle tensors in radians, each with shape ``(count,)``.

    Raises:
        ValueError: If the approach factor is disabled or ``count`` is invalid.
    """
    approach = cfg.factors.approach
    if not approach.enabled:
        raise ValueError("approach variation is disabled in this configuration")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    polar = torch.zeros(count, dtype=torch.float64)
    azimuth = torch.zeros(count, dtype=torch.float64)
    if count == 1:
        return polar, azimuth
    values = torch.rand(
        (count - 1, 2),
        generator=generator,
        device=generator.device,
        dtype=torch.float64,
    )
    # Inverting the cap area keeps the sampled directions uniform over the
    # surface rather than over the polar angle.
    cosine = 1 - values[:, 0] * (1 - math.cos(approach.cone_half_angle_rad))
    polar[1:] = torch.arccos(cosine.clamp(-1, 1))
    azimuth[1:] = (2 * math.pi) * values[:, 1] - math.pi
    return polar, azimuth


def _variant_generator(
    cfg: TrajectoryAugmentationCfg,
    template: TrajectoryTemplate,
    variant: TrajectoryVariant,
) -> tuple[str, torch.Generator]:
    """Derive a variant's stable digest and its local CPU generator."""
    digest = _digest(cfg.seed, template.template_id, variant.ordinal)
    return digest, torch.Generator(device="cpu").manual_seed(
        int(digest[:16], 16) % 2**63
    )


def apply_trajectory_variant(
    template: TrajectoryTemplate,
    variant: TrajectoryVariant,
    *,
    cfg: TrajectoryAugmentationCfg | None = None,
    joint_limits: torch.Tensor,
    control_dt: float,
    task_jacobians: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> TrajectoryTemplate:
    """Apply one joint-path operator and the variant's retiming, in that order.

    At most one joint-path operator runs per variant. Null-space projection is
    defined against the reference Jacobians, so stacking it on an already
    displaced joint path would silently invalidate its task-row claim.

    The default generator is seeded from the configuration seed, the template
    identifier and the variant ordinal, so the same variant applied to two differently
    identified references draws independently while remaining reproducible.

    Args:
        template: Annotated reference whose phase endpoints are already fixed.
        variant: Factor combination to apply; a nominal variant returns the reference.
        cfg: Augmentation settings supplying operator magnitudes and the seed.
            Defaults to :func:`default_variant_factors`, with redundancy
            enabled only when ``task_jacobians`` is supplied.
        joint_limits: Finite lower/upper limits with shape ``(D, 2)``.
        control_dt: Authoritative host control period used by retiming.
        task_jacobians: Reference spatial Jacobians ``(N, R, C)`` for the
            declared controlled joints, required only by the redundancy
            operator. ``cfg.factors.ik.task_rows`` selects the constrained rows
            from them, so supply every row the task could constrain.
        generator: Optional explicit local generator replacing the derived one.

    Returns:
        A new template requiring path and physical validation.

    Raises:
        ValueError: If the variant names an unknown operator, if required inputs
            are missing, or if an operator rejects its own proposal.
    """
    # Mirror the planner: without Jacobians a posture variant cannot run, so an
    # unconfigured application resolves the same conservative policy.
    if cfg is None:
        cfg = default_variant_factors(redundancy=task_jacobians is not None)
    _reject_cartesian_factors(cfg)
    if generator is None:
        _, generator = _variant_generator(cfg, template, variant)
    result = template
    spatial = cfg.factors.spatial
    if variant.spatial_operator == "joint_residual":
        result = joint_residual(
            result,
            joint_limits=joint_limits,
            normalized_scale=spatial.joint_offset_scale,
            generator=generator,
        )
    elif variant.spatial_operator == "via_points":
        result = via_points(
            result,
            joint_limits=joint_limits,
            via_count=spatial.via_count,
            normalized_scale=spatial.joint_offset_scale,
            generator=generator,
        )
    elif variant.spatial_operator == "nullspace_residual":
        if task_jacobians is None:
            raise ValueError(
                "nullspace_residual requires task_jacobians for the reference samples"
            )
        result = nullspace_residual(
            result,
            task_jacobians=select_jacobian_rows(
                task_jacobians, list(cfg.factors.ik.task_rows)
            ),
            joint_limits=joint_limits,
            normalized_scale=cfg.factors.ik.normalized_scale,
            generator=generator,
        )
    elif variant.spatial_operator != NOMINAL_OPERATOR:
        raise ValueError(f"unknown spatial operator: {variant.spatial_operator!r}")
    if variant.duration_scale != 1.0 or variant.timing_profile != "uniform":
        result = retime(
            result,
            duration_scale=variant.duration_scale,
            control_dt=control_dt,
            profile=variant.timing_profile,
        )
    return result


_MAX_REJECTION_REASON = 80


def _rejection_reason(error: ProposalRejected) -> str:
    """Summarize an operator rejection into a stable, bounded counter key."""
    message = " ".join(str(error).split())
    if len(message) > _MAX_REJECTION_REASON:
        message = message[:_MAX_REJECTION_REASON].rstrip() + "..."
    return f"operator: {message}" if message else "operator_rejected"


def _pad_rows(
    accepted: list[TrajectoryTemplate],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack variable-length rows using the padding rule of the batch contract."""
    horizon = max(row.positions.shape[0] for row in accepted)
    positions = torch.stack(
        [
            torch.cat(
                (
                    row.positions,
                    row.positions[-1:].expand(
                        horizon - row.positions.shape[0], row.positions.shape[1]
                    ),
                )
            )
            for row in accepted
        ]
    )
    intervals = torch.stack(
        [
            torch.cat(
                (row.dt, row.dt.new_zeros(horizon - row.dt.shape[0])),
            )
            for row in accepted
        ]
    )
    lengths = torch.tensor(
        [row.positions.shape[0] for row in accepted], dtype=torch.int64
    )
    return positions, intervals, lengths


def expand_trajectory_variants(
    template: TrajectoryTemplate,
    *,
    case: SceneCase,
    count: int,
    cfg: TrajectoryAugmentationCfg | None = None,
    joint_limits: torch.Tensor,
    control_dt: float,
    velocity_limits: torch.Tensor | None = None,
    acceleration_limits: torch.Tensor | None = None,
    task_jacobians: torch.Tensor | None = None,
    max_attempts: int | None = None,
) -> TrajectoryVariantSet:
    """Generate up to ``count`` distinct execution variants for one fixed reference.

    Row zero is the unmodified reference, so a caller always retains the
    trajectory it already trusts. Later rows apply one joint-path operator
    and one timing choice each. A proposal is discarded when an operator
    rejects it, when sampled motion limits fail, or when its measured geometry
    and timing duplicate an already accepted row. Every rejection is counted
    and reported rather than silently dropped.

    Accepted rows share the candidate padding rule: padded samples repeat the
    final valid position and carry zero arrival intervals, so a shorter row
    holds its final command instead of being executed past its own end. Use
    :attr:`~.contracts.CandidateTrajectoryBatch.valid_mask` before recording.

    Deduplication compares measured joint geometry and elapsed phase time. It
    does not certify collision freedom, task success or dynamic feasibility;
    the host must still execute and validate each accepted row.

    Args:
        template: Annotated reference whose phase endpoints are already fixed.
        case: Scene and initial-state identity shared by every candidate.
        count: Maximum number of accepted variants, at least one.
        cfg: Augmentation settings selecting factors, seed and coverage limits.
            Omit it to enable every implemented factor the supplied inputs
            support, through :func:`default_variant_factors`.
        joint_limits: Finite lower/upper limits with shape ``(D, 2)``.
        control_dt: Authoritative host control period used by retiming.
        velocity_limits: Optional positive per-joint speed bounds, shape ``(D,)``.
        acceleration_limits: Optional positive per-joint acceleration bounds.
        task_jacobians: Reference spatial Jacobians ``(N, R, C)`` required only
            when the ik factor is enabled. ``cfg.factors.ik.task_rows`` selects
            the constrained rows, so supply every row the task could constrain
            rather than pre-reducing them.
        max_attempts: Optional proposal budget; defaults to four per requested variant.

    Returns:
        The accepted candidate rows, their variants, and rejection accounting.

    Raises:
        ValueError: If arguments are malformed, if limits are supplied
            inconsistently, or if an explicitly enabled factor is missing the
            inputs it needs.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if cfg is None:
        # Without Jacobians the redundancy factor could only propose variants
        # that fail, so leave it out rather than spend attempts on them.
        cfg = default_variant_factors(redundancy=task_jacobians is not None)
    elif cfg.factors.ik.enabled and task_jacobians is None:
        raise ValueError(
            "the enabled ik factor requires task_jacobians for the reference "
            "samples; supply them or disable the factor"
        )
    if not isinstance(case, SceneCase):
        raise ValueError("case must be a SceneCase")
    if (velocity_limits is None) != (acceleration_limits is None):
        raise ValueError(
            "velocity_limits and acceleration_limits must be supplied together"
        )
    attempts = count * _ATTEMPT_MULTIPLIER if max_attempts is None else max_attempts
    if type(attempts) is not int or attempts < count:
        raise ValueError("max_attempts must be an integer of at least count")
    coverage = CoverageIndex(
        geometry_tolerance=cfg.coverage.joint_dedup_normalized_tol,
        target_per_geometry=cfg.coverage.target_per_cell,
    )
    accepted: list[TrajectoryTemplate] = []
    accepted_variants: list[TrajectoryVariant] = []
    families: list[str] = []
    rejected: dict[str, int] = {}
    evaluated = 0

    for variant in plan_trajectory_variants(attempts, cfg):
        if len(accepted) >= count:
            break
        evaluated += 1
        digest, generator = _variant_generator(cfg, template, variant)
        try:
            candidate = apply_trajectory_variant(
                template,
                variant,
                cfg=cfg,
                joint_limits=joint_limits,
                control_dt=control_dt,
                task_jacobians=task_jacobians,
                generator=generator,
            )
        except ProposalRejected as error:
            # Only a rejected draw is bookkeeping. A malformed argument or an
            # impossible configuration keeps propagating, so it cannot hide in
            # a rejection count behind an already successful nominal variant.
            reason = _rejection_reason(error)
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        if velocity_limits is not None and acceleration_limits is not None:
            limits_result = validate_motion_limits(
                candidate,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
            )
            if not limits_result.accepted:
                rejected["motion_limits"] = rejected.get("motion_limits", 0) + 1
                continue
        descriptor = describe_trajectory(
            candidate.positions,
            candidate.dt.cumsum(0),
            joint_limits,
            phases=candidate.phases,
            samples_per_phase=cfg.coverage.geometry_samples_per_phase,
        )
        commit_id = f"variant_{variant.ordinal}"
        if not coverage.reserve(commit_id, "geometry_" + digest, descriptor):
            rejected["duplicate"] = rejected.get("duplicate", 0) + 1
            continue
        coverage.confirm(commit_id)
        accepted.append(candidate)
        accepted_variants.append(variant)
        families.append(coverage.family_of(commit_id))

    if not accepted:
        raise ValueError(
            "No trajectory variant survived operator, motion-limit and duplicate "
            f"rejection after {evaluated} proposals: {rejected}."
        )
    positions, intervals, lengths = _pad_rows(accepted)
    identities = tuple(
        CandidateIdentity(
            case.scene_case_id,
            case.initial_state_id,
            "candidate_" + _digest(cfg.seed, template.template_id, variant.ordinal),
            family,
            template.source_id,
            template.source_revision,
            template.template_id,
        )
        for variant, family in zip(accepted_variants, families)
    )
    return TrajectoryVariantSet(
        candidates=CandidateTrajectoryBatch(
            positions=positions,
            dt=intervals,
            valid_length=lengths,
            identities=identities,
            joint_names=template.joint_names,
            phases=tuple(row.phases for row in accepted),
            factors=tuple(variant.factors for variant in accepted_variants),
        ),
        variants=tuple(accepted_variants),
        attempted=evaluated,
        rejected=rejected,
        cfg=cfg,
    )
