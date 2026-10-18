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

"""Bounded, deterministic matching of prepared assets to task roles.

Relation checks use explicitly annotated axis-aligned regions in normalized
local meter/+Z frames. They certify geometric fit for upright placement, not
collision-free insertion, grasp feasibility, stability, or physical success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any

from embodichain.data.asset_catalog import PreparedAsset
from .asset_requirements import AssetRequirements, AssetRoleRequirement

__all__ = [
    "AssetRejection",
    "AssetResolutionError",
    "ResolvedAssetSet",
    "resolve_asset_set",
]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class AssetRejection:
    """One candidate or relationship rejection with its affected role."""

    role: str
    asset_id: str | None
    code: str
    detail: str
    target_role: str | None = None


class AssetResolutionError(ValueError):
    """No assignment was found within the configured candidate/search bounds."""

    def __init__(
        self,
        message: str,
        *,
        rejections: Sequence[AssetRejection],
        search_nodes: int = 0,
        budget_exhausted: bool = False,
    ) -> None:
        super().__init__(message)
        self.rejections = tuple(rejections)
        self.search_nodes = search_nodes
        self.budget_exhausted = budget_exhausted


@dataclass(frozen=True)
class ResolvedAssetSet:
    """Immutable role assignment with content and requirement identity.

    Absolute local paths are retained for loading but excluded from ``set_id``;
    copying the same prepared content into another cache keeps its identity.
    """

    assets: Mapping[str, PreparedAsset]
    seed: int
    requirements_sha256: str
    set_id: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError("seed must be an integer.")
        if not self.assets or any(
            not isinstance(role, str)
            or not role
            or not isinstance(asset, PreparedAsset)
            for role, asset in self.assets.items()
        ):
            raise ValueError("assets must map nonempty roles to PreparedAsset values.")
        if len(self.requirements_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.requirements_sha256
        ):
            raise ValueError("requirements_sha256 must be a SHA-256 hex digest.")
        assets = dict(sorted(self.assets.items()))
        object.__setattr__(self, "assets", MappingProxyType(assets))
        object.__setattr__(
            self,
            "set_id",
            _digest(
                {
                    "schema_version": 1,
                    "seed": self.seed,
                    "requirements_sha256": self.requirements_sha256,
                    "assets": {
                        role: _asset_identity(asset) for role, asset in assets.items()
                    },
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return an independent JSON-compatible manifest for batch provenance."""
        return {
            "schema_version": 1,
            "set_id": self.set_id,
            "seed": self.seed,
            "requirements_sha256": self.requirements_sha256,
            "assets": {role: asset.to_dict() for role, asset in self.assets.items()},
        }


def _asset_identity(asset: PreparedAsset) -> dict[str, Any]:
    return {
        "content_sha256": asset.content_sha256,
        "descriptor": asset.descriptor.to_dict(),
    }


def _category(value: str, aliases: Mapping[str, str]) -> str:
    while value in aliases:
        value = aliases[value]
    return value


def _candidate_rejection(
    role: str,
    requirement: AssetRoleRequirement,
    asset: PreparedAsset,
    aliases: Mapping[str, str],
) -> AssetRejection | None:
    descriptor = asset.descriptor

    def reject(code: str, detail: str) -> AssetRejection:
        return AssetRejection(role, descriptor.asset_id, code, detail)

    if not descriptor.normalized:
        return reject(
            "not_normalized", "Asset must use normalized local meter/+Z coordinates."
        )
    if descriptor.bounds_m is None:
        return reject("missing_bounds", "Asset bounding geometry is unknown.")
    query = requirement.query
    if query.category is not None and (
        descriptor.category is None
        or _category(descriptor.category, aliases) != _category(query.category, aliases)
    ):
        return reject(
            "category_mismatch",
            f"Expected category {query.category!r}; got {descriptor.category!r}.",
        )
    for key, expected in sorted(query.attributes.items()):
        actual = descriptor.attributes.get(key)
        if (
            actual is None
            or expected is None
            or type(actual) is not type(expected)
            or actual != expected
        ):
            return reject(
                "attribute_mismatch",
                f"Required attribute {key!r}={expected!r}; got {actual!r}.",
            )
    missing = sorted(set(query.capabilities) - set(descriptor.capabilities))
    if missing:
        return reject(
            "missing_capability", f"Missing declared capabilities: {missing}."
        )
    if (
        requirement.max_extent_m is not None
        and max(descriptor.bounds_m.extents) > requirement.max_extent_m
    ):
        return reject(
            "extent_exceeded",
            f"Asset exceeds maximum extent {requirement.max_extent_m} m.",
        )
    return None


def _relation_rejection(
    requirements: AssetRequirements,
    assigned: Mapping[str, PreparedAsset],
    added_role: str,
) -> AssetRejection | None:
    for relation in requirements.relations:
        if added_role not in (relation.object_role, relation.target_role):
            continue
        if relation.object_role not in assigned or relation.target_role not in assigned:
            continue
        source = assigned[relation.object_role].descriptor
        target = assigned[relation.target_role].descriptor
        region = (
            target.container_region
            if relation.kind == "fits_inside"
            else target.support_region
        )
        if region is None:
            return AssetRejection(
                relation.object_role,
                source.asset_id,
                "missing_relation_geometry",
                f"Target {target.asset_id!r} has no annotated region for {relation.kind}.",
                relation.target_role,
            )
        # Prefilter guarantees source bounds. No inferred cavity from outer AABB.
        assert source.bounds_m is not None
        axes = 3 if relation.kind == "fits_inside" else 2
        if any(
            source.bounds_m.extents[axis] + 2 * relation.clearance_m
            > region.extents[axis]
            for axis in range(axes)
        ):
            return AssetRejection(
                relation.object_role,
                source.asset_id,
                "relation_does_not_fit",
                f"{source.asset_id!r} cannot satisfy {relation.kind} with {target.asset_id!r} "
                f"at {relation.clearance_m} m clearance per side.",
                relation.target_role,
            )
    return None


def resolve_asset_set(
    requirements: AssetRequirements,
    candidates_by_role: Mapping[str, Sequence[PreparedAsset]],
    *,
    seed: int = 0,
    max_candidates_per_role: int = 100,
    max_search_nodes: int = 1000,
) -> ResolvedAssetSet:
    """Select the first valid seeded assignment using bounded iterative search.

    Candidate identities are sorted before seeded ranking, so provider iteration
    order cannot affect selection. At most ``max_candidates_per_role`` unique
    candidates per role are evaluated. This is a search budget, not a guarantee
    that a satisfying assignment will be found. Pair constraints are checked as
    soon as both roles are assigned; the Cartesian product is never allocated.

    Args:
        requirements: Validated task-role constraints and explicit aliases.
        candidates_by_role: Prepared candidates for exactly the required roles.
        seed: Integer defining deterministic candidate ranking.
        max_candidates_per_role: Candidate evaluation limit for each role.
        max_search_nodes: Maximum attempted partial-assignment extensions.

    Raises:
        ValueError: If input role sets or budgets are invalid.
        AssetResolutionError: If no assignment passes within the budgets.
    """
    if not isinstance(requirements, AssetRequirements):
        raise TypeError("requirements must be AssetRequirements.")
    for name, value in (
        ("seed", seed),
        ("max_candidates_per_role", max_candidates_per_role),
        ("max_search_nodes", max_search_nodes),
    ):
        if type(value) is not int or (name != "seed" and value <= 0):
            raise ValueError(
                f"{name} must be {'an integer' if name == 'seed' else 'a positive integer'}."
            )
    if set(candidates_by_role) != set(requirements.roles):
        raise ValueError("Candidate roles must exactly match requirement roles.")

    rejected: list[AssetRejection] = []
    detail_count = 0
    truncated = False

    def record(rejection: AssetRejection) -> None:
        nonlocal detail_count
        # Bound candidate details separately from per-role and budget summaries.
        # Summaries have no asset ID and occur at most twice per role plus once
        # for a stopped search, so they cannot grow with rejected combinations.
        if rejection.asset_id is not None:
            if detail_count >= 100:
                return
            detail_count += 1
        rejected.append(rejection)

    pools: dict[str, list[PreparedAsset]] = {}
    for role, requirement in sorted(requirements.roles.items()):
        unique: dict[str, PreparedAsset] = {}
        for asset in candidates_by_role[role]:
            if not isinstance(asset, PreparedAsset):
                raise TypeError(
                    f"Candidates for {role!r} must be PreparedAsset values."
                )
            identity = _digest(_asset_identity(asset))
            existing = unique.get(identity)
            # Source paths/evidence can differ for identical asset content.
            # Break those ties too so the emitted provenance is reproducible.
            if existing is None or _digest(asset.to_dict()) < _digest(
                existing.to_dict()
            ):
                unique[identity] = asset
        ranked = sorted(
            unique, key=lambda identity: (_digest([seed, role, identity]), identity)
        )
        if len(ranked) > max_candidates_per_role:
            truncated = True
            record(
                AssetRejection(
                    role,
                    None,
                    "candidate_budget",
                    f"Evaluating {max_candidates_per_role} of {len(ranked)} unique candidates.",
                )
            )
        pools[role] = []
        for identity in ranked[:max_candidates_per_role]:
            asset = unique[identity]
            rejection = _candidate_rejection(
                role, requirement, asset, requirements.category_aliases
            )
            if rejection is None:
                pools[role].append(asset)
            else:
                record(rejection)
        if not pools[role]:
            record(
                AssetRejection(
                    role,
                    None,
                    "no_candidates",
                    "No eligible candidate remains for this role.",
                )
            )
    if any(not pool for pool in pools.values()):
        raise AssetResolutionError(
            "No eligible candidates for one or more roles.",
            rejections=rejected,
            budget_exhausted=truncated,
        )

    # Choose the smallest domain first. Sorted role names break equal-size ties.
    roles = sorted(pools, key=lambda role: (len(pools[role]), role))
    assigned: dict[str, PreparedAsset] = {}
    offsets = [0] * len(roles)
    depth = 0
    visited = 0
    while depth >= 0:
        if depth == len(roles):
            return ResolvedAssetSet(assigned, seed, _digest(requirements.to_dict()))
        role = roles[depth]
        if offsets[depth] >= len(pools[role]):
            offsets[depth] = 0
            assigned.pop(role, None)
            depth -= 1
            continue
        if visited >= max_search_nodes:
            record(
                AssetRejection(
                    role,
                    None,
                    "search_budget",
                    f"Stopped after {visited} partial assignments.",
                )
            )
            raise AssetResolutionError(
                "Asset assignment search budget exhausted.",
                rejections=rejected,
                search_nodes=visited,
                budget_exhausted=True,
            )
        assigned[role] = pools[role][offsets[depth]]
        offsets[depth] += 1
        visited += 1
        rejection = _relation_rejection(requirements, assigned, role)
        if rejection is None:
            depth += 1
        else:
            record(rejection)
            assigned.pop(role, None)
    raise AssetResolutionError(
        "No compatible asset combination found.",
        rejections=rejected,
        search_nodes=visited,
        budget_exhausted=truncated,
    )
