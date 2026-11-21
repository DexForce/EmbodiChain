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

"""Adaptive joint-domain search with frozen, independent region validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol
import math
import time
from types import MappingProxyType

import numpy as np
import torch
from scipy.stats import beta, qmc

from embodichain.compute.geometry.planar import compose_planar_offsets
from ..expansion import ScenePoseChange, SceneVariant
from ._values import count, digest, finite, owned_json, plain
from .budget import FeasibilityBudget, FeasibilityBudgetExhausted
from .cfg import SceneRobotFeasibilityCfg
from .contracts import (
    CandidateEvaluation,
    FeasibilityCheck,
    JointSceneCandidate,
    RobotDeploymentCandidate,
    TaskFeasibilityRequest,
)

__all__ = [
    "FeasibilityEvaluator",
    "PreparedFeasibilitySearch",
    "FeasibilityObservation",
    "FeasibleRegion",
    "FeasibilityAtlas",
    "prepare_feasibility",
    "search_feasibility",
    "sample_feasibility",
    "validate_feasibility",
    "binomial_interval",
]


class FeasibilityEvaluator(Protocol):
    """Trusted host port: checks retain their stage and actual evidence scope."""

    def prepare(
        self, request: TaskFeasibilityRequest, budget: FeasibilityBudget
    ) -> None:
        """Prepare immutable geometry/policy inputs inside the shared deadline."""
        ...

    def evaluate(
        self, candidate: JointSceneCandidate, budget: FeasibilityBudget
    ) -> CandidateEvaluation:
        """Evaluate one exact scene; charge nested IK/planning/rollout calls."""
        ...


@dataclass(frozen=True)
class PreparedFeasibilitySearch:
    """Owned fixed request/configuration and typed joint-domain materialization."""

    request: TaskFeasibilityRequest
    cfg: SceneRobotFeasibilityCfg

    @property
    def dimensions(self) -> int:
        """Return XY/yaw coordinates, including fixed coordinates."""
        return 3 * len(self.request.variables)

    def candidate(
        self,
        parameters: Sequence[float],
        *,
        mode_id: str,
        ordinal: int,
        seed: int | None = None,
    ) -> JointSceneCandidate:
        """Materialize a full correlated proposal from normalized coordinates.

        Args:
            parameters: One value in [0,1] per declared coordinate.
            mode_id: Explicit permitted operation mode.
            ordinal: Logical proposal ordinal, independent of simulation slots.
            seed: Stream identity; defaults to the search seed.
        Returns:
            Owned scene and robot poses preserving group member transforms.
        """
        unit = finite(parameters, "normalized parameters")
        if len(unit) != self.dimensions or any(not 0 <= v <= 1 for v in unit):
            raise ValueError("parameters must match the normalized joint domain")
        unit = tuple(
            0.0 if a == b else unit[i * 3 + j]
            for i, v in enumerate(self.request.variables)
            for j, (a, b) in enumerate(zip(v.lower, v.upper))
        )
        count(ordinal, "ordinal", 0)
        if mode_id not in {m.mode_id for m in self.request.modes}:
            raise ValueError("unknown operation mode")
        seed = self.cfg.seed if seed is None else seed
        count(seed, "seed", 0)
        scene, robots = [], {}
        if self.request.variables:
            lo = torch.tensor(
                [v.lower for v in self.request.variables],
                dtype=torch.float64,
                device=self.cfg.device,
            )
            hi = torch.tensor(
                [v.upper for v in self.request.variables],
                dtype=torch.float64,
                device=self.cfg.device,
            )
            offsets = lo + torch.tensor(unit, dtype=lo.dtype, device=lo.device).reshape(
                -1, 3
            ) * (hi - lo)
            roots = (
                compose_planar_offsets(
                    torch.tensor(
                        [v.reference_pose for v in self.request.variables],
                        dtype=lo.dtype,
                        device=lo.device,
                    ),
                    offsets,
                )
                .cpu()
                .numpy()
            )
            for variable, root in zip(self.request.variables, roots, strict=True):
                for uid, local in variable.members.items():
                    pose = (root @ np.asarray(local)).tolist()
                    if variable.kind == "robot":
                        robots[uid] = pose
                    else:
                        scene.append(ScenePoseChange(uid, pose))
        variant = SceneVariant(
            self.request.parent_scene_id,
            seed,
            ordinal,
            tuple(scene),
            (("feasibility_request", self.request.request_id),),
        )
        return JointSceneCandidate(
            self.request.request_id,
            variant,
            RobotDeploymentCandidate(robots),
            mode_id,
            unit,
        )


@dataclass(frozen=True)
class FeasibilityObservation:
    """One search or held-out evaluation and its exact proposal."""

    candidate: JointSceneCandidate
    evaluation: CandidateEvaluation
    region_id: str | None = None
    completed: bool = True

    def __post_init__(self) -> None:
        if type(self.completed) is not bool:
            raise TypeError("observation completion must be boolean")
        if self.candidate.candidate_id != self.evaluation.candidate_id:
            raise ValueError("evaluation does not identify its exact candidate")


@dataclass(frozen=True)
class FeasibleRegion:
    """A conditional joint box; unsampled points are proposals, not certificates.

    ``mode_id`` and all request constraints condition this box. Sampling never
    independently recombines bounds from different regions. Validation trials
    count raw box proposals, including semantic/geometric rejections.
    """

    region_id: str
    mode_id: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    reference_candidate_ids: tuple[str, ...]
    validation_planned: int = 0
    validation_completed: int = 0
    validation_successes: int = 0
    confidence_interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        lo, hi = finite(self.lower, "region lower"), finite(self.upper, "region upper")
        if len(lo) != len(hi) or any(not 0 <= a <= b <= 1 for a, b in zip(lo, hi)):
            raise ValueError("region bounds must lie in the normalized joint domain")
        object.__setattr__(self, "lower", lo)
        object.__setattr__(self, "upper", hi)
        object.__setattr__(
            self, "reference_candidate_ids", tuple(self.reference_candidate_ids)
        )
        for key in (
            "validation_planned",
            "validation_completed",
            "validation_successes",
        ):
            count(getattr(self, key), key, 0)
        if (
            not self.validation_successes
            <= self.validation_completed
            <= self.validation_planned
        ):
            raise ValueError("invalid validation accounting")
        if self.confidence_interval is not None:
            if not self.validation_complete:
                raise ValueError("incomplete validation cannot publish an interval")
            expected = binomial_interval(
                self.validation_successes, self.validation_completed
            )
            if expected is None or not np.allclose(
                expected, self.confidence_interval, atol=1e-12, rtol=0
            ):
                raise ValueError("confidence interval disagrees with completed trials")
            object.__setattr__(
                self, "confidence_interval", tuple(self.confidence_interval)
            )

    @property
    def validation_complete(self) -> bool:
        """Return whether every trial in the frozen validation allocation finished."""
        return (
            self.validation_planned > 0
            and self.validation_completed == self.validation_planned
        )


@dataclass(frozen=True)
class FeasibilityAtlas:
    """Frozen request, regions, observations and accounting suitable for replay.

    Incomplete validation is descriptive only: its interval is deliberately
    omitted because interruption may depend on outcomes or execution time.
    """

    request: TaskFeasibilityRequest
    cfg: SceneRobotFeasibilityCfg
    regions: tuple[FeasibleRegion, ...]
    observations: tuple[FeasibilityObservation, ...]
    stop_reason: str
    metrics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "cfg", SceneRobotFeasibilityCfg.from_mapping(plain(self.cfg))
        )
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "metrics", owned_json(self.metrics))
        seen = set()
        region_ids = {r.region_id for r in self.regions}
        if len(region_ids) != len(self.regions):
            raise ValueError("region IDs must be unique")
        for obs in self.observations:
            if (
                obs.candidate.request_id != self.request.request_id
                or obs.candidate.candidate_id in seen
            ):
                raise ValueError("observations need unique candidates of this request")
            seen.add(obs.candidate.candidate_id)
            if obs.region_id is not None and obs.region_id not in region_ids:
                raise ValueError("validation refers to an unknown region")
        success_ids = {
            o.candidate.candidate_id
            for o in self.observations
            if o.region_id is None and o.evaluation.accepted
        }
        for region in self.regions:
            if len(region.lower) != 3 * len(
                self.request.variables
            ) or region.mode_id not in {m.mode_id for m in self.request.modes}:
                raise ValueError("region domain does not match its request")
            if (
                not region.reference_candidate_ids
                or not set(region.reference_candidate_ids) <= success_ids
            ):
                raise ValueError("regions require measured search references")
            references = [
                o
                for o in self.observations
                if o.candidate.candidate_id in region.reference_candidate_ids
            ]
            trials = [
                o
                for o in self.observations
                if o.region_id == region.region_id and o.completed
            ]
            for row in (
                *references,
                *(o for o in self.observations if o.region_id == region.region_id),
            ):
                if row.candidate.mode_id != region.mode_id or not all(
                    a <= v <= b
                    for a, v, b in zip(
                        region.lower,
                        row.candidate.parameters,
                        region.upper,
                        strict=True,
                    )
                ):
                    raise ValueError(
                        "region evidence lies outside its mode or joint bounds"
                    )
            if (
                len(trials) != region.validation_completed
                or sum(o.evaluation.accepted for o in trials)
                != region.validation_successes
            ):
                raise ValueError(
                    "region statistics disagree with held-out observations"
                )

    @property
    def atlas_id(self) -> str:
        """Return a content identity for this exact evidence snapshot."""
        return digest(self)


def binomial_interval(
    successes: int, trials: int, *, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Compute a two-sided Clopper-Pearson interval for fixed-count iid trials.

    Args:
        successes: Accepted full executions.
        trials: Completed fixed-allocation trials.
        confidence: Coverage level, strictly between zero and one.
    Returns:
        Exact binomial interval, or ``None`` for no trials. This is a marginal
        interval per region, not a simultaneous or adaptively stopped guarantee.
    """
    count(successes, "successes", 0)
    count(trials, "trials", 0)
    if successes > trials or not 0 < confidence < 1:
        raise ValueError("invalid binomial observations/confidence")
    if not trials:
        return None
    alpha = (1 - confidence) / 2
    return (
        (
            0.0
            if not successes
            else float(beta.ppf(alpha, successes, trials - successes + 1))
        ),
        (
            1.0
            if successes == trials
            else float(beta.ppf(1 - alpha, successes + 1, trials - successes))
        ),
    )


def prepare_feasibility(
    request: TaskFeasibilityRequest, cfg: SceneRobotFeasibilityCfg | None = None
) -> PreparedFeasibilitySearch:
    """Snapshot declared permissions and search settings without mutating a scene.

    Args:
        request: Explicit task and robot/scene domain.
        cfg: Optional limits; defaults to the minute-scale search profile.
    Returns:
        A reusable prepared materializer. Hosts prepare geometry inside search.
    """
    if not isinstance(request, TaskFeasibilityRequest):
        raise TypeError("request must be TaskFeasibilityRequest")
    settings = SceneRobotFeasibilityCfg.from_mapping(
        plain(cfg or SceneRobotFeasibilityCfg())
    )
    return PreparedFeasibilitySearch(request, settings)


def _quality(observation: FeasibilityObservation) -> tuple:
    result = observation.evaluation
    passed = sum(c.status == "passed" for c in result.checks)
    return (result.accepted, passed, result.clearance_m, -result.execution_cost_s)


def _regions(
    prepared: PreparedFeasibilitySearch, observations: list[FeasibilityObservation]
) -> tuple[FeasibleRegion, ...]:
    """Partition only occupied cells, separating observed failures and modes."""
    cfg = prepared.cfg
    dimension = prepared.dimensions
    active = np.array(
        [a < b for v in prepared.request.variables for a, b in zip(v.lower, v.upper)]
    )
    regions = []
    for mode in prepared.request.modes:
        rows = [
            o
            for o in observations
            if o.completed and o.candidate.mode_id == mode.mode_id
        ]
        queue = [(rows, np.zeros(dimension), active.astype(float))]
        while queue:
            group, lo, hi = queue.pop()
            passing = [o for o in group if o.evaluation.accepted]
            if not passing:
                continue
            width = np.where(active, hi - lo, 0.0)
            if not dimension or float(width.max(initial=0.0)) <= cfg.min_region_width:
                # Mixed terminal cells retain references but remain unqualified
                # until the independently frozen distribution is evaluated.
                stop = True
            else:
                stop = (
                    len(passing) == len(group)
                    and float(width.max()) <= 2 * cfg.local_radius
                )
            if stop:
                bounds = (tuple(lo), tuple(hi))
                regions.append(
                    FeasibleRegion(
                        digest((prepared.request.request_id, mode.mode_id, bounds)),
                        mode.mode_id,
                        *bounds,
                        tuple(o.candidate.candidate_id for o in passing),
                    )
                )
                continue
            axis = int(width.argmax())
            midpoint = (lo[axis] + hi[axis]) / 2
            left, right = [], []
            for row in group:
                (left if row.candidate.parameters[axis] < midpoint else right).append(
                    row
                )
            hi_left, lo_right = hi.copy(), lo.copy()
            hi_left[axis], lo_right[axis] = midpoint, midpoint
            queue.extend(((right, lo_right, hi), (left, lo, hi_left)))
    # Farthest-first selection keeps disconnected operating modes/layouts.
    by_id = {o.candidate.candidate_id: o for o in observations}
    qualities = {
        r.region_id: max(_quality(by_id[c]) for c in r.reference_candidate_ids)
        for r in regions
    }
    chosen = []
    while regions and len(chosen) < cfg.max_regions:

        def score(region: FeasibleRegion) -> tuple:
            center = (np.asarray(region.lower) + region.upper) / 2
            spread = min(
                (
                    float(np.linalg.norm(center - (np.asarray(r.lower) + r.upper) / 2))
                    + float(r.mode_id != region.mode_id)
                    for r in chosen
                ),
                default=0.0,
            )
            quality = qualities[region.region_id]
            return (spread, quality, region.region_id)

        best = max(regions, key=score)
        chosen.append(best)
        regions.remove(best)
    return tuple(chosen)


def _metrics(
    rows: Sequence[FeasibilityObservation], budget: FeasibilityBudget
) -> dict[str, object]:
    from collections import Counter

    def cell(row: FeasibilityObservation) -> tuple:
        return (
            row.candidate.mode_id,
            *(
                min(budget.cfg.archive_bins - 1, int(v * budget.cfg.archive_bins))
                for v in row.candidate.parameters
            ),
        )

    accepted = [r for r in rows if r.completed and r.evaluation.accepted]
    committed = [r for r in accepted if r.evaluation.committed]
    return {
        "calls": dict(budget.counts),
        "elapsed_s": budget.clock() - budget.started,
        "measured_successes": sum(r.evaluation.accepted for r in rows),
        "committed": sum(r.evaluation.committed for r in rows),
        "program_completions": sum(r.evaluation.program_completed for r in rows),
        "scene_coverage": {
            "observed_joint_cells": len({cell(r) for r in rows if r.completed}),
            "measured_joint_cells": len({cell(r) for r in accepted}),
            "committed_joint_cells": len({cell(r) for r in committed}),
            "exact_committed_layouts": len(
                {
                    digest(
                        (r.candidate.scene.pose_changes, r.candidate.deployment.poses)
                    )
                    for r in committed
                }
            ),
            "measured_modes": sorted({r.candidate.mode_id for r in accepted}),
        },
        "failure_stages": dict(
            Counter(
                next(
                    (c.stage for c in r.evaluation.checks if c.status != "passed"),
                    "missing_required_evidence",
                )
                for r in rows
                if not r.evaluation.accepted
            )
        ),
    }


def _evaluate(
    prepared: PreparedFeasibilitySearch,
    evaluator: FeasibilityEvaluator,
    candidate: JointSceneCandidate,
    budget: FeasibilityBudget,
    region_id: str | None = None,
) -> FeasibilityObservation:
    budget.charge("candidates")
    try:
        result = evaluator.evaluate(candidate, budget)
    except FeasibilityBudgetExhausted as error:
        result = CandidateEvaluation(
            candidate.candidate_id,
            (FeasibilityCheck("budget", "budget_exhausted", str(error)),),
        )
    if not isinstance(result, CandidateEvaluation):
        raise TypeError("evaluator must return CandidateEvaluation")
    completed = not any(c.status == "budget_exhausted" for c in result.checks)
    return FeasibilityObservation(candidate, result, region_id, completed)


def search_feasibility(
    prepared: PreparedFeasibilitySearch,
    evaluator: FeasibilityEvaluator,
    *,
    budget: FeasibilityBudget | None = None,
    validate_regions: bool = True,
    on_observation: Callable[[FeasibilityObservation], None] | None = None,
) -> FeasibilityAtlas:
    """Explore globally and refine promising cells, then validate frozen regions.

    Args:
        prepared: Fixed request and search configuration.
        evaluator: Trusted bounded geometry/planning/physical host.
        budget: Optional caller-owned clock/cancellation budget.
        validate_regions: Freeze and return regions without held-out trials when
            false; the caller can later invoke validate_feasibility once.
        on_observation: Optional diagnostic/checkpoint consumer for completed or
            explicitly interrupted observations. Errors propagate to the caller.
    Returns:
        Actual observed regions and evidence; no global optimality certificate.
    """
    cfg = prepared.cfg
    if type(validate_regions) is not bool:
        raise TypeError("validate_regions must be boolean")
    budget = budget or FeasibilityBudget(cfg)
    if plain(budget.cfg) != plain(cfg):
        raise ValueError("budget settings must match the prepared search")
    observations, archive = [], {}
    reason = "candidate_budget_exhausted"
    stream = qmc.Halton(
        max(1, prepared.dimensions), scramble=True, seed=np.random.default_rng(cfg.seed)
    )
    rng = np.random.default_rng(cfg.seed)
    try:
        evaluator.prepare(prepared.request, budget)
        global_units = np.empty((0, max(1, prepared.dimensions)))
        for ordinal in range(cfg.max_candidates):
            budget.check()
            if ordinal % cfg.batch_size == 0:
                global_units = stream.random(
                    min(cfg.batch_size, cfg.max_candidates - ordinal)
                )
            unit = global_units[ordinal % cfg.batch_size, : prepared.dimensions].copy()
            mode_index = ordinal % len(prepared.request.modes)
            mode_id = prepared.request.modes[mode_index].mode_id
            pool = [
                o
                for values in archive.values()
                for o in values
                if o.candidate.mode_id == mode_id
            ]
            is_local = (
                pool
                and (ordinal // len(prepared.request.modes))
                % max(1, round(1 / cfg.global_fraction))
                != 0
            )
            if not is_local:
                proposer = getattr(evaluator, "propose", None)
                if callable(proposer):
                    unit = np.asarray(
                        proposer(unit, mode_id, ordinal, budget), dtype=float
                    )
            if is_local:
                reference = pool[int(rng.integers(len(pool)))]
                failed = next(
                    (c for c in reference.evaluation.checks if c.status != "passed"),
                    None,
                )
                radius = cfg.local_radius * (
                    0.5 if failed and failed.stage == "execution" else 1.0
                )
                local = np.clip(
                    np.asarray(reference.candidate.parameters)
                    + (unit * 2 - 1) * radius,
                    0.0,
                    1.0,
                )
                if failed and failed.affected_variables:
                    relevant = {v.name for v in prepared.request.variables}
                    if not set(failed.affected_variables) <= relevant:
                        raise ValueError("failure refers to an unknown variable")
                    for i, variable in enumerate(prepared.request.variables):
                        if variable.name not in failed.affected_variables:
                            local[3 * i : 3 * i + 3] = reference.candidate.parameters[
                                3 * i : 3 * i + 3
                            ]
                unit = local
            candidate = prepared.candidate(unit, mode_id=mode_id, ordinal=ordinal)
            row = _evaluate(prepared, evaluator, candidate, budget)
            observations.append(row)
            if on_observation is not None:
                on_observation(row)
            if not row.completed:
                raise FeasibilityBudgetExhausted(
                    next(
                        (
                            c.reason
                            for c in row.evaluation.checks
                            if c.status == "budget_exhausted" and c.reason
                        ),
                        "evaluator_budget_exhausted",
                    )
                )
            key = (
                mode_id,
                *(
                    min(cfg.archive_bins - 1, int(v * cfg.archive_bins))
                    for v in candidate.parameters
                ),
            )
            cells = sorted((*archive.get(key, ()), row), key=_quality, reverse=True)
            best_level = _quality(cells[0])[:2]
            level = [c for c in cells if _quality(c)[:2] == best_level]
            frontier = [
                c
                for c in level
                if not any(
                    other is not c
                    and other.evaluation.clearance_m >= c.evaluation.clearance_m
                    and other.evaluation.execution_cost_s
                    <= c.evaluation.execution_cost_s
                    and (
                        other.evaluation.clearance_m > c.evaluation.clearance_m
                        or other.evaluation.execution_cost_s
                        < c.evaluation.execution_cost_s
                    )
                    for other in level
                )
            ]
            archive[key] = frontier[: cfg.archive_per_bin]
    except FeasibilityBudgetExhausted as error:
        reason = str(error)
    regions = _regions(prepared, observations)
    metrics = _metrics(observations, budget)
    proposal_metrics = getattr(evaluator, "proposal_metrics", None)
    if proposal_metrics is not None:
        metrics["proposal_source"] = proposal_metrics
    atlas = FeasibilityAtlas(
        prepared.request,
        cfg,
        regions,
        tuple(observations),
        reason,
        metrics,
    )
    if reason == "cancelled" or not validate_regions:
        return atlas
    return validate_feasibility(
        atlas, evaluator, budget=budget, prepare=False, on_observation=on_observation
    )


def validate_feasibility(
    atlas: FeasibilityAtlas,
    evaluator: FeasibilityEvaluator,
    *,
    budget: FeasibilityBudget | None = None,
    prepare: bool = True,
    on_observation: Callable[[FeasibilityObservation], None] | None = None,
) -> FeasibilityAtlas:
    """Validate previously frozen regions with independent fixed-count trials.

    Args:
        atlas: Frozen regions; an already validated atlas cannot be revalidated.
        evaluator: Same request/policy/validator host as the search.
        budget: Shared search budget, or a fresh validation-only budget.
        prepare: Prepare the host for standalone validation.
        on_observation: Optional checkpoint consumer, including interrupted trials.
    Returns:
        A new atlas containing held-out evidence and intervals for completed
        allocations only. Interrupted trials never count as completed successes.
        Costs include the saved search and this validation call, even when the
        caller provides a fresh budget.
    """
    if any(r.validation_planned for r in atlas.regions):
        raise ValueError("validation requires a frozen, not-yet-validated atlas")
    prepared = prepare_feasibility(atlas.request, atlas.cfg)
    budget = budget or FeasibilityBudget(atlas.cfg)
    if plain(budget.cfg) != plain(atlas.cfg):
        raise ValueError("budget settings must match the frozen validation profile")
    initial_calls = dict(budget.counts)
    initial_elapsed = budget.clock() - budget.started
    budget.validation = True
    rows = list(atlas.observations)
    regions = [
        replace(r, validation_planned=atlas.cfg.validation_samples)
        for r in atlas.regions
    ]
    reason = atlas.stop_reason
    try:
        if prepare:
            evaluator.prepare(atlas.request, budget)
        # Independent IID streams, allocated before observing outcomes; search's
        # adaptive/low-discrepancy samples do not enter binomial counts.
        for index, region in enumerate(regions):
            seed = int(digest((atlas.cfg.seed, "held-out", region.region_id))[:15], 16)
            rng = np.random.default_rng(seed)
            trials, successes = 0, 0
            try:
                for trial in range(atlas.cfg.validation_samples):
                    unit = rng.uniform(region.lower, region.upper)
                    candidate = prepared.candidate(
                        unit, mode_id=region.mode_id, ordinal=trial, seed=seed
                    )
                    row = _evaluate(
                        prepared, evaluator, candidate, budget, region.region_id
                    )
                    rows.append(row)
                    if on_observation is not None:
                        on_observation(row)
                    if not row.completed:
                        raise FeasibilityBudgetExhausted(
                            next(
                                (
                                    c.reason
                                    for c in row.evaluation.checks
                                    if c.status == "budget_exhausted" and c.reason
                                ),
                                "evaluator_budget_exhausted",
                            )
                        )
                    trials += 1
                    successes += int(row.evaluation.accepted)
            finally:
                regions[index] = replace(
                    region,
                    validation_completed=trials,
                    validation_successes=successes,
                    confidence_interval=(
                        binomial_interval(successes, trials)
                        if trials == atlas.cfg.validation_samples
                        else None
                    ),
                )
    except FeasibilityBudgetExhausted as error:
        reason = str(error)
    metrics = dict(atlas.metrics)
    metrics.update(_metrics(rows, budget))
    metrics["calls"] = {
        kind: atlas.metrics.get("calls", {}).get(kind, 0) + count - initial_calls[kind]
        for kind, count in metrics["calls"].items()
    }
    metrics["elapsed_s"] = atlas.metrics.get("elapsed_s", 0.0) + max(
        0.0, metrics["elapsed_s"] - initial_elapsed
    )
    metrics["validation_distribution"] = "raw_joint_region_proposals"
    return FeasibilityAtlas(
        atlas.request, atlas.cfg, tuple(regions), tuple(rows), reason, metrics
    )


def sample_feasibility(
    atlas: FeasibilityAtlas,
    *,
    count: int,
    seed: int,
    precheck: Callable[[JointSceneCandidate], Sequence[FeasibilityCheck]],
    max_attempts: int = 1000,
    device: str = "cpu",
) -> tuple[JointSceneCandidate, ...]:
    """Sample correlated region proposals and recheck geometry/semantics.

    Args:
        atlas: Region constraints and fixed task identities.
        count: Requested output count.
        seed: Independent generation seed.
        precheck: Trusted current semantic/geometric checks; both stages required.
        max_attempts: Hard proposal/rejection limit.
        device: Numeric proposal materialization device; defaults to CPU for
            portable consumption of an atlas created on another CUDA device.
    Returns:
        At most count proposals. These require new full execution acceptance;
        region success evidence is never inherited by a generated episode.
    """
    from ._values import count as check_count

    check_count(count, "count")
    check_count(seed, "seed", 0)
    check_count(max_attempts, "max_attempts")
    if not atlas.regions:
        return ()
    settings = SceneRobotFeasibilityCfg.from_mapping(
        {**plain(atlas.cfg), "device": device}
    )
    prepared = prepare_feasibility(atlas.request, settings)
    rng, output = np.random.default_rng(seed), []
    for ordinal in range(max_attempts):
        region = atlas.regions[ordinal % len(atlas.regions)]
        candidate = prepared.candidate(
            rng.uniform(region.lower, region.upper),
            mode_id=region.mode_id,
            ordinal=ordinal,
            seed=seed,
        )
        checks = tuple(precheck(candidate))
        if any(not isinstance(c, FeasibilityCheck) for c in checks):
            raise TypeError("precheck must return FeasibilityCheck values")
        if {"geometry", "semantics"} <= {c.stage for c in checks} and all(
            c.status == "passed" for c in checks
        ):
            output.append(candidate)
            if len(output) == count:
                break
    return tuple(output)
