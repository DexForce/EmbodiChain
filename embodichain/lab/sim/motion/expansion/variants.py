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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
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
from .manipulability import (
    ManipulabilityBands,
    _select_guided_candidate,
    describe_manipulability,
)
from .operators import (
    ProposalRejected,
    allowed_phases,
    joint_residual,
    nullspace_residual,
    retime,
    validate_motion_limits,
    via_points,
)
from .task_space import (
    TASK_SPACE_RESIDUAL_OPERATOR,
    KinematicsFn,
    TaskSpaceDeformationConstraint,
    TaskSpaceTrajectory,
    deform_task_space_trajectory,
    project_task_space_trajectory,
    task_space_from_joint_template,
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
        task_space: Whether to deform and project the tool path before retiming.
        manipulability_band: Requested guidance band, or ``None`` for unguided draws.
    """

    ordinal: int
    spatial_operator: str
    duration_scale: float
    timing_profile: str
    task_space: bool = False
    manipulability_band: int | None = None

    def __post_init__(self) -> None:
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        if not math.isfinite(self.duration_scale) or self.duration_scale <= 0:
            raise ValueError("duration_scale must be finite and positive")
        if type(self.task_space) is not bool:
            raise ValueError("task_space must be a boolean")
        if self.manipulability_band is not None and (
            type(self.manipulability_band) is not int or self.manipulability_band < 0
        ):
            raise ValueError("manipulability_band must be a nonnegative integer")

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
            and not self.task_space
            and self.manipulability_band is None
        )

    @property
    def factors(self) -> dict[str, float | int | str]:
        """Return this variant's provenance for candidate-row metadata.

        Returns:
            Finite JSON-compatible factor values describing the variant.
        """
        result: dict[str, float | int | str] = {
            "variant_ordinal": self.ordinal,
            "spatial_operator": self.spatial_operator,
            "duration_scale": float(self.duration_scale),
            "timing_profile": self.timing_profile,
        }
        if self.task_space:
            result["task_space_operator"] = TASK_SPACE_RESIDUAL_OPERATOR
        if self.manipulability_band is not None:
            result["manipulability_target_band"] = self.manipulability_band
        return result


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
    guided = (
        factors.manipulability.enabled and factors.manipulability.guided_proposals > 1
    )
    if not operators and guided and not factors.task_space.enabled:
        operators.append("joint_residual")
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
            TrajectoryVariant(
                ordinal,
                operators[step % len(operators)],
                scale,
                profile,
                task_space=factors.task_space.enabled,
                manipulability_band=(
                    step % (len(factors.manipulability.band_edges) + 1)
                    if guided
                    else None
                ),
            )
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


@dataclass
class _ProposalBudget:
    """Bound every numerical draw, including rejected guided proposals."""

    limit: int
    attempted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str, count: int = 1) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + count


def _kinematic_jacobians(
    positions: torch.Tensor,
    template: TrajectoryTemplate,
    kinematics: KinematicsFn,
) -> torch.Tensor:
    """Evaluate controlled columns without assuming contiguous joint ordering."""
    poses, jacobians = kinematics(positions[:, list(template.controlled_joint_indices)])
    if not isinstance(poses, torch.Tensor) or poses.shape != (len(positions), 4, 4):
        raise ValueError("kinematics poses must have shape (T, 4, 4)")
    if not isinstance(jacobians, torch.Tensor) or jacobians.shape != (
        len(positions),
        6,
        len(template.controlled_joint_indices),
    ):
        raise ValueError("kinematics Jacobians must have shape (T, 6, C)")
    return jacobians


def _prepare_generation(
    template: TrajectoryTemplate,
    cfg: TrajectoryAugmentationCfg,
    kinematics: KinematicsFn | None,
    task_jacobians: torch.Tensor | None,
    manipulability_reference: float | None,
) -> tuple[TaskSpaceTrajectory | None, ManipulabilityBands | None, torch.Tensor | None]:
    """Resolve capabilities and cache reference work once per expansion call."""
    cfg.validate_semantics()
    _reject_cartesian_factors(cfg)
    if kinematics is not None and not callable(kinematics):
        raise TypeError("kinematics must be callable or None")
    guided = (
        cfg.factors.manipulability.enabled
        and cfg.factors.manipulability.guided_proposals > 1
    )
    if (cfg.factors.task_space.enabled or guided) and kinematics is None:
        raise ValueError(
            "enabled task_space or manipulability guidance requires kinematics"
        )
    if cfg.factors.ik.enabled and task_jacobians is None:
        if kinematics is None:
            raise ValueError(
                "the enabled ik factor requires task_jacobians or kinematics"
            )
        task_jacobians = _kinematic_jacobians(template.positions, template, kinematics)
    task_reference = None
    if cfg.factors.task_space.enabled:
        allowed_phases(template, TASK_SPACE_RESIDUAL_OPERATOR)
        assert kinematics is not None
        task_reference = task_space_from_joint_template(
            template,
            frame_id=cfg.factors.task_space.frame_id,
            kinematics=kinematics,
        )
    bands = None
    if guided:
        assert kinematics is not None
        if manipulability_reference is None:
            jacobians = _kinematic_jacobians(template.positions, template, kinematics)
            manipulability_reference = describe_manipulability(
                jacobians,
                rows=cfg.factors.manipulability.jacobian_rows,
            ).bottleneck
        bands = ManipulabilityBands(
            cfg.factors.manipulability.band_edges,
            reference=manipulability_reference,
        )
    return task_reference, bands, task_jacobians


def _project_target(
    target: TaskSpaceTrajectory,
    reference: TrajectoryTemplate,
    cfg: TrajectoryAugmentationCfg,
    joint_limits: torch.Tensor,
    kinematics: KinematicsFn,
) -> TrajectoryTemplate:
    """Reject a numerical IK miss without swallowing malformed caller inputs."""
    projected = project_task_space_trajectory(
        target,
        reference,
        kinematics=kinematics,
        joint_limits=joint_limits,
        ik_cfg=cfg.factors.task_space.ik,
    )
    if not projected.accepted:
        raise ProposalRejected("trajectory_ik")
    return projected.template


def _apply_variant_once(
    template: TrajectoryTemplate,
    variant: TrajectoryVariant,
    *,
    cfg: TrajectoryAugmentationCfg,
    joint_limits: torch.Tensor,
    control_dt: float,
    task_jacobians: torch.Tensor | None,
    generator: torch.Generator,
    kinematics: KinematicsFn | None,
    task_reference: TaskSpaceTrajectory | None,
) -> TrajectoryTemplate:
    """Generate one path, re-project combined joint edits, then retime it."""
    result = template
    target = None
    if variant.task_space:
        if task_reference is None or kinematics is None:
            raise ValueError(
                "task-space variant requires an enabled task_space factor and kinematics"
            )
        task_cfg = cfg.factors.task_space
        constraints = {
            phase.phase_id: TaskSpaceDeformationConstraint(
                phase.phase_id,
                task_cfg.translation_axes,
                task_cfg.rotation_axes,
                task_cfg.max_translation_m,
                task_cfg.max_rotation_rad,
            )
            for phase in task_reference.phases
            if TASK_SPACE_RESIDUAL_OPERATOR in phase.allowed_operators
        }
        target = deform_task_space_trajectory(
            task_reference, constraints=constraints, generator=generator
        )
        result = _project_target(target, result, cfg, joint_limits, kinematics)
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
        if target is not None:
            assert kinematics is not None
            task_jacobians = _kinematic_jacobians(
                result.positions, template, kinematics
            )
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
    if target is not None and variant.spatial_operator != NOMINAL_OPERATOR:
        assert kinematics is not None
        # A joint-only phase may move independently of the Cartesian phases.
        # Preserve that legal edit rather than imposing an undeclared TCP lock.
        joint_only = [
            phase
            for phase in allowed_phases(template, variant.spatial_operator)
            if TASK_SPACE_RESIDUAL_OPERATOR not in phase.allowed_operators
        ]
        if joint_only:
            poses, _ = kinematics(
                result.positions[:, list(template.controlled_joint_indices)]
            )
            updated = target.poses.clone()
            for phase in joint_only:
                updated[phase.start_index : phase.stop_index] = poses[
                    phase.start_index : phase.stop_index
                ]
            target = replace(target, poses=updated)
        result = _project_target(target, result, cfg, joint_limits, kinematics)
    if variant.duration_scale != 1.0 or variant.timing_profile != "uniform":
        result = retime(
            result,
            duration_scale=variant.duration_scale,
            control_dt=control_dt,
            profile=variant.timing_profile,
        )
    return result


def _generate_variant(
    template: TrajectoryTemplate,
    variant: TrajectoryVariant,
    *,
    cfg: TrajectoryAugmentationCfg,
    joint_limits: torch.Tensor,
    control_dt: float,
    task_jacobians: torch.Tensor | None,
    generator: torch.Generator,
    kinematics: KinematicsFn | None,
    task_reference: TaskSpaceTrajectory | None,
    bands: ManipulabilityBands | None,
    budget: _ProposalBudget,
    velocity_limits: torch.Tensor | None = None,
    acceleration_limits: torch.Tensor | None = None,
) -> tuple[TrajectoryTemplate, dict[str, float | int | str]]:
    """Share a bounded stream of validated draws with the guidance selector."""
    guided = variant.manipulability_band is not None
    if guided and (bands is None or kinematics is None):
        raise ValueError(
            "guided variant requires enabled manipulability guidance and kinematics"
        )
    if (
        variant.manipulability_band is not None
        and bands is not None
        and variant.manipulability_band >= bands.count
    ):
        raise ValueError("manipulability_band must index the configured bands")
    draws = min(
        cfg.factors.manipulability.guided_proposals if guided else 1,
        budget.limit - budget.attempted,
    )

    def candidates() -> Iterable[TrajectoryTemplate]:
        last = None
        produced = False
        for _ in range(draws):
            budget.attempted += 1
            try:
                candidate = _apply_variant_once(
                    template,
                    variant,
                    cfg=cfg,
                    joint_limits=joint_limits,
                    control_dt=control_dt,
                    task_jacobians=task_jacobians,
                    generator=generator,
                    kinematics=kinematics,
                    task_reference=task_reference,
                )
                if velocity_limits is not None and acceleration_limits is not None:
                    validation = validate_motion_limits(
                        candidate,
                        velocity_limits=velocity_limits,
                        acceleration_limits=acceleration_limits,
                    )
                    if not validation.accepted:
                        raise ProposalRejected("motion_limits")
            except ProposalRejected as error:
                last = error
                reason = str(error)
                budget.reject(
                    reason
                    if reason in {"motion_limits", "trajectory_ik"}
                    else _rejection_reason(error)
                )
                continue
            produced = True
            yield candidate
        if not produced:
            raise last if last is not None else ProposalRejected("proposal_budget")

    proposals = candidates()
    evidence = variant.factors
    if variant.task_space:
        evidence["task_space_frame_id"] = cfg.factors.task_space.frame_id
    if not guided:
        return next(proposals), evidence
    assert bands is not None and kinematics is not None
    assert variant.manipulability_band is not None
    chosen = _select_guided_candidate(
        proposals,
        jacobian_fn=lambda q: _kinematic_jacobians(q, template, kinematics),
        bands=bands,
        target_band=variant.manipulability_band,
        rows=cfg.factors.manipulability.jacobian_rows,
    )
    if chosen.proposals_scored > 1:
        budget.reject("guidance_unselected", chosen.proposals_scored - 1)
    evidence.update(
        manipulability=chosen.profile.bottleneck,
        manipulability_band=chosen.band,
        manipulability_matched=int(chosen.matched),
        manipulability_proposals_scored=chosen.proposals_scored,
    )
    return chosen.template, evidence


def apply_trajectory_variant(
    template: TrajectoryTemplate,
    variant: TrajectoryVariant,
    *,
    cfg: TrajectoryAugmentationCfg | None = None,
    joint_limits: torch.Tensor,
    control_dt: float,
    task_jacobians: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    kinematics: KinematicsFn | None = None,
    manipulability_reference: float | None = None,
) -> TrajectoryTemplate:
    """Apply a joint/task-space proposal, optional guidance, then retiming.

    At most one joint-path operator runs per proposal. A combined task-space
    proposal projects before joint edits and re-projects afterward. Null-space
    edits use fresh Jacobians when the task-space projection moved the path.
    Guidance samples up to ``guided_proposals`` draws and selects the first
    matching band, or the nearest reachable band with stable tie-breaking.

    The default generator is seeded from the configuration seed, the template
    identifier and the variant ordinal, so the same variant applied to two differently
    identified references draws independently while remaining reproducible.

    Args:
        template: Annotated reference whose phase endpoints are already fixed.
        variant: Factor combination to apply; a nominal variant returns the reference.
        cfg: Augmentation settings supplying operator magnitudes and the seed.
            Defaults to :func:`default_variant_factors`, with redundancy
            enabled when ``task_jacobians`` or ``kinematics`` is supplied.
        joint_limits: Finite lower/upper limits with shape ``(D, 2)``.
        control_dt: Authoritative host control period used by retiming.
        task_jacobians: Reference spatial Jacobians ``(N, R, C)`` for the
            declared controlled joints, required only by the redundancy
            operator. ``cfg.factors.ik.task_rows`` selects the constrained rows
            from them, so supply every row the task could constrain.
        generator: Optional explicit local generator replacing the derived one.
        kinematics: Batched root-frame FK/Jacobian callable over controlled
            columns; required by task-space deformation and guidance.
        manipulability_reference: Positive reference score for guidance bands.
            Omission measures the nominal trajectory; hosted callers use the
            initial state's registered reference.

    Returns:
        A new template requiring path and physical validation.

    Raises:
        ValueError: If the variant names an unknown operator, if required inputs
            are missing, or if an operator rejects its own proposal.
    """
    # Mirror the planner: without Jacobians a posture variant cannot run, so an
    # unconfigured application resolves the same conservative policy.
    if cfg is None:
        cfg = default_variant_factors(
            redundancy=task_jacobians is not None or kinematics is not None
        )
    if generator is None:
        _, generator = _variant_generator(cfg, template, variant)
    task_reference, bands, task_jacobians = _prepare_generation(
        template,
        cfg,
        kinematics,
        task_jacobians,
        manipulability_reference,
    )
    result, _ = _generate_variant(
        template,
        variant,
        cfg=cfg,
        joint_limits=joint_limits,
        control_dt=control_dt,
        task_jacobians=task_jacobians,
        generator=generator,
        kinematics=kinematics,
        task_reference=task_reference,
        bands=bands,
        budget=_ProposalBudget(cfg.factors.manipulability.guided_proposals),
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
    generator: torch.Generator | None = None,
    max_attempts: int | None = None,
    kinematics: KinematicsFn | None = None,
    manipulability_reference: float | None = None,
) -> TrajectoryVariantSet:
    """Generate up to ``count`` distinct execution variants for one fixed reference.

    Row zero is the unmodified reference, so a caller always retains the
    trajectory it already trusts. Later rows apply the configured joint,
    task-space and guidance factors before the timing choice. A proposal is discarded when an operator
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
        generator: Optional private CPU generator shared by the sequential
            proposal draws. Omission preserves standalone digest-based streams.
        max_attempts: Total numerical-draw budget, including guidance's inner
            proposals. Defaults to four per requested variant.
        kinematics: Batched root-frame FK/Jacobian callable over controlled
            columns, required by task-space deformation and guided selection.
        manipulability_reference: Positive band normalizer. Omission measures
            the nominal reference; coordinators use the registered initial state.

    Returns:
        The accepted candidate rows, their variants, and rejection accounting.

    Raises:
        ValueError: If arguments are malformed, if limits are supplied
            inconsistently, or if an explicitly enabled factor is missing the
            inputs it needs.
    """
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if generator is not None and not isinstance(generator, torch.Generator):
        raise TypeError("generator must be a torch.Generator or None")
    if cfg is None:
        # Live kinematics can supply reference Jacobians on demand; absent both
        # inputs, a redundancy proposal cannot run.
        cfg = default_variant_factors(
            redundancy=task_jacobians is not None or kinematics is not None
        )
    task_reference, bands, task_jacobians = _prepare_generation(
        template,
        cfg,
        kinematics,
        task_jacobians,
        manipulability_reference,
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
    accepted_factors: list[dict[str, float | int | str]] = []
    families: list[str] = []
    budget = _ProposalBudget(attempts)

    for variant in plan_trajectory_variants(attempts, cfg):
        if len(accepted) >= count or budget.attempted >= budget.limit:
            break
        digest, derived_generator = _variant_generator(cfg, template, variant)
        proposal_generator = derived_generator if generator is None else generator
        try:
            candidate, evidence = _generate_variant(
                template,
                variant,
                cfg=cfg,
                joint_limits=joint_limits,
                control_dt=control_dt,
                task_jacobians=task_jacobians,
                generator=proposal_generator,
                kinematics=kinematics,
                task_reference=task_reference,
                bands=bands,
                budget=budget,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
            )
        except ProposalRejected:
            # Only a rejected draw is bookkeeping. A malformed argument or an
            # impossible configuration keeps propagating, so it cannot hide in
            # a rejection count behind an already successful nominal variant.
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
            budget.reject("duplicate")
            continue
        coverage.confirm(commit_id)
        accepted.append(candidate)
        accepted_variants.append(variant)
        accepted_factors.append(evidence)
        families.append(coverage.family_of(commit_id))

    if not accepted:
        raise ValueError(
            "No trajectory variant survived operator, motion-limit and duplicate "
            f"rejection after {budget.attempted} proposals: {budget.rejected}."
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
            factors=tuple(accepted_factors),
        ),
        variants=tuple(accepted_variants),
        attempted=budget.attempted,
        rejected=budget.rejected,
        cfg=cfg,
    )
