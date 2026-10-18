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

"""Immutable provider-neutral contracts for normalized rigid asset catalogs.

Geometry uses meters in the asset's right-handed, Z-up local frame. A normalized
flag is an explicit preparation assertion, never inferred from a model filename.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Bounds3D",
    "AssetDescriptor",
    "AssetCandidate",
    "PreparedAsset",
    "AssetQuery",
    "PreparationContext",
    "AssetProvider",
]


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _number(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {_identifier(k, "mapping key"): _freeze(v) for k, v in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Metadata must contain only finite JSON values.")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    if isinstance(value, _Record):
        return value.to_dict()
    return value


def _mapping(value: Mapping[str, Any], *, scalars: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Expected a mapping.")
    if scalars and any(isinstance(v, (Mapping, tuple, list)) for v in value.values()):
        raise ValueError("Attributes must contain only JSON scalar values.")
    return _freeze(value)


def _strings(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence, not a string.")
    result = tuple(_identifier(v, name) for v in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate entries.")
    return result


class _Record:
    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible snapshot of this record."""
        return {item.name: _thaw(getattr(self, item.name)) for item in fields(self)}

    @classmethod
    def _decode(cls, data: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(data, Mapping):
            raise TypeError(f"{cls.__name__} requires a mapping.")
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        return dict(data)


@dataclass(frozen=True, slots=True)
class Bounds3D(_Record):
    """Axis-aligned local bounds; planar bounds are allowed for support regions.

    Args:
        lower: Three minimum coordinates, in meters.
        upper: Three maximum coordinates, in meters.
    """

    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("lower", "upper"):
            values = tuple(_number(v, name) for v in getattr(self, name))
            if len(values) != 3:
                raise ValueError(f"{name} must contain three coordinates.")
            object.__setattr__(self, name, values)
        if any(lo > hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError(
                "Bounds lower coordinates cannot exceed upper coordinates."
            )

        if any(not math.isfinite(hi - lo) for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Bounds extents must be finite.")

    @property
    def extents(self) -> tuple[float, float, float]:
        """Return the axis-aligned X, Y and Z lengths in meters."""
        return tuple(hi - lo for lo, hi in zip(self.lower, self.upper))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Bounds3D:
        """Decode a closed bounds mapping."""
        return cls(**cls._decode(data))


@dataclass(frozen=True, slots=True)
class AssetQuery(_Record):
    """Source-neutral semantic constraints used to request candidates.

    Args:
        category: Exact semantic category, or no category restriction.
        attributes: Required JSON scalar properties.
        capabilities: Explicit required capabilities, independent of category.
    """

    category: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.category is not None:
            _identifier(self.category, "category")
        object.__setattr__(self, "attributes", _mapping(self.attributes, scalars=True))
        object.__setattr__(
            self, "capabilities", _strings(self.capabilities, "capabilities")
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetQuery:
        """Decode a closed semantic query mapping."""
        return cls(**cls._decode(data))


@dataclass(frozen=True, slots=True)
class AssetDescriptor(_Record):
    """Identity, semantics and local geometry of one rigid asset revision.

    Args:
        asset_id: Stable identity within the provider.
        category: Known semantic category; ``None`` means unknown.
        attributes: Known scalar semantic properties.
        capabilities: Explicit declared capabilities.
        bounds_m: Full object geometry bounds in normalized local coordinates.
        support_region: Usable horizontal support rectangle; Z is constant.
        container_region: Usable interior box, not the exterior bounding box.
        provenance: JSON evidence/source records, optionally keyed by field.
        normalized: Whether geometry is rigid, in meters and right-handed Z-up.
    """

    asset_id: str
    category: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    bounds_m: Bounds3D | None = None
    support_region: Bounds3D | None = None
    container_region: Bounds3D | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    normalized: bool = False

    def __post_init__(self) -> None:
        _identifier(self.asset_id, "asset_id")
        query = AssetQuery(self.category, self.attributes, self.capabilities)
        object.__setattr__(self, "attributes", query.attributes)
        object.__setattr__(self, "capabilities", query.capabilities)
        object.__setattr__(self, "provenance", _mapping(self.provenance))
        if not isinstance(self.normalized, bool):
            raise TypeError("normalized must be a boolean.")
        for name in ("bounds_m", "support_region", "container_region"):
            bounds = getattr(self, name)
            if bounds is None:
                continue
            if not isinstance(bounds, Bounds3D):
                raise TypeError(f"{name} must be Bounds3D.")
            axes = bounds.extents[:2] if name == "support_region" else bounds.extents
            if any(v <= 0 for v in axes):
                raise ValueError(f"{name} must have positive extents.")
            if name == "support_region" and bounds.extents[2] != 0:
                raise ValueError("support_region must be a horizontal rectangle.")
        if self.normalized and self.bounds_m is None:
            raise ValueError("Normalized geometry requires bounds_m.")
        if self.bounds_m is not None:
            for name in ("support_region", "container_region"):
                region = getattr(self, name)
                if region is not None and any(
                    lo < outer_lo - 1e-9 or hi > outer_hi + 1e-9
                    for lo, hi, outer_lo, outer_hi in zip(
                        region.lower,
                        region.upper,
                        self.bounds_m.lower,
                        self.bounds_m.upper,
                    )
                ):
                    raise ValueError(f"{name} must lie within bounds_m.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetDescriptor:
        """Decode metadata, rejecting unknown keys and invalid nested bounds."""
        values = cls._decode(data)
        for name in ("bounds_m", "support_region", "container_region"):
            if values.get(name) is not None:
                values[name] = Bounds3D.from_dict(values[name])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class AssetCandidate(_Record):
    """Candidate identity and provider locator, with optional provisional metadata.

    Args:
        candidate_id: Stable candidate identifier within its source.
        source_uri: Provider-owned locator or generation recipe identifier.
        descriptor: Optional preliminary description, revalidated on preparation.
    """

    candidate_id: str
    source_uri: str
    descriptor: AssetDescriptor | None = None

    def __post_init__(self) -> None:
        _identifier(self.candidate_id, "candidate_id")
        _identifier(self.source_uri, "source_uri")
        if self.descriptor is not None and not isinstance(
            self.descriptor, AssetDescriptor
        ):
            raise TypeError("descriptor must be AssetDescriptor.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AssetCandidate:
        """Decode one candidate snapshot."""
        values = cls._decode(data)
        if values.get("descriptor") is not None:
            values["descriptor"] = AssetDescriptor.from_dict(values["descriptor"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PreparedAsset(_Record):
    """Prepared local files and content identity, ready for task-level validation.

    Args:
        descriptor: Normalized asset description.
        visual_path: Absolute local visual model path.
        collision_path: Absolute local collision model path.
        content_sha256: SHA-256 over descriptor and all model dependencies.
        provenance: Preparation/source evidence. ``package_root`` names an
            absolute prepared directory; ``files`` maps all package-relative
            filenames to SHA-256 digests. Neither is a task acceptance result.

    Providers return the common package snapshot consumed by
    :func:`embodichain.data.asset_catalog.verify_prepared_asset`: an
    ``asset.catalog.json`` plus all local geometry dependencies. Content identity
    hashes the canonical JSON descriptor and sorted package file-digest mapping.
    File existence and dependency hashing belong to the preparing provider.
    The record validates format and snapshots metadata without doing I/O.
    """

    descriptor: AssetDescriptor
    visual_path: str
    collision_path: str
    content_sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, AssetDescriptor):
            raise TypeError("descriptor must be AssetDescriptor.")
        if not self.descriptor.normalized:
            raise ValueError("Prepared assets require normalized rigid geometry.")
        for name in ("visual_path", "collision_path"):
            path = _identifier(getattr(self, name), name)
            if not Path(path).is_absolute():
                raise ValueError(f"{name} must be an absolute local path.")
        if (
            not isinstance(self.content_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.content_sha256) is None
        ):
            raise ValueError("content_sha256 must be a lowercase SHA-256 hex digest.")
        object.__setattr__(self, "provenance", _mapping(self.provenance))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreparedAsset:
        """Decode a prepared asset snapshot without loading its files."""
        values = cls._decode(data)
        if "descriptor" in values:
            values["descriptor"] = AssetDescriptor.from_dict(values["descriptor"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class PreparationContext(_Record):
    """Local destination supplied to asset preparation.

    Args:
        cache_dir: Absolute path to the content-addressed preparation cache.
    """

    cache_dir: str

    def __post_init__(self) -> None:
        if not Path(_identifier(self.cache_dir, "cache_dir")).is_absolute():
            raise ValueError("cache_dir must be absolute.")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PreparationContext:
        """Decode a preparation context."""
        return cls(**cls._decode(data))


@runtime_checkable
class AssetProvider(Protocol):
    """Acquire candidates and prepare local assets without task dependencies."""

    def candidates(
        self, request: AssetQuery, *, limit: int
    ) -> Sequence[AssetCandidate]:
        """Return at most ``limit`` provisional candidates for the query.

        Args:
            request: Semantic selection constraints.
            limit: Positive maximum candidate count.

        Returns:
            Candidate snapshots requiring preparation and authoritative matching.
        """
        ...

    def materialize(
        self, candidate: AssetCandidate, *, context: PreparationContext
    ) -> PreparedAsset:
        """Prepare one candidate and independently validate its resulting metadata.

        Args:
            candidate: Candidate owned by this provider.
            context: Local preparation destination.

        Returns:
            Normalized local assets with dependency-complete content identity.
        """
        ...
