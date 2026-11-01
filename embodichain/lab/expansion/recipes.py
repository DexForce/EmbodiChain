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

"""Source-pinned random streams and explicit operator capability contracts."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Protocol, Literal
from types import MappingProxyType
import hashlib
import random
import math

__all__ = [
    "ExpansionContext",
    "ExpertSource",
    "PinnedExpertSource",
    "OperatorCapability",
    "ExpansionOperator",
    "UniformRangeOperator",
]


@dataclass(frozen=True)
class ExpansionContext:
    """Read-only logical sampling identity, independent of physical row and clock."""

    source_kind: str
    source_revision: str
    seed: int
    ordinal: int

    def __post_init__(self) -> None:
        if not self.source_kind or not self.source_revision:
            raise ValueError("source identity must be nonempty")
        if any(type(v) is not int or v < 0 for v in (self.seed, self.ordinal)):
            raise ValueError("seed and logical ordinal must be nonnegative integers")

    def random(self, operator: str, version: str, field: str) -> random.Random:
        """Create a stable independent stream for one versioned operator field."""
        import json

        key = json.dumps(
            (
                self.source_kind,
                self.source_revision,
                self.seed,
                self.ordinal,
                operator,
                version,
                field,
            ),
            separators=(",", ":"),
        )
        return random.Random(int(hashlib.sha256(key.encode()).hexdigest(), 16))


class ExpertSource(Protocol):
    """Read-only source identity; execution stays with the host's task runtime."""

    kind: str
    revision: str

    def context(self, seed: int, ordinal: int) -> ExpansionContext:
        """Bind a logical candidate without materializing a static trajectory."""
        ...


@dataclass(frozen=True)
class PinnedExpertSource:
    """Configured Task Program or Real2Sim content identity used by operators."""

    kind: str
    revision: str

    def context(self, seed: int, ordinal: int) -> ExpansionContext:
        """Return a logical candidate context with no mutable simulation access."""
        return ExpansionContext(self.kind, self.revision, seed, ordinal)


@dataclass(frozen=True)
class OperatorCapability:
    """Declared backend, source, phase and resource scope for a sampled change."""

    domain: str
    phase: Literal["prepare", "plan", "render"]
    scope: Literal["environment", "world", "shared_resource"]
    backends: tuple[str, ...] = ("default", "newton")
    sources: tuple[str, ...] = ("task_program", "real2sim")


class ExpansionOperator(Protocol):
    """Pure sampling port; operators never mutate or step a World."""

    name: str
    version: str
    capability: OperatorCapability

    def sample(self, context: ExpansionContext) -> Mapping[str, object]:
        """Return an owned immutable change for a logical candidate."""
        ...


@dataclass(frozen=True)
class UniformRangeOperator:
    """Sample per-entity scalar fields with independent stable random streams.

    Args:
        name: Host-recognized physical parameter, such as mass_scale or friction.
        ranges: Entity IDs mapped to finite ordered scalar bounds.
        version: Sampling implementation version included in every random stream.
    """

    name: str
    ranges: Mapping[str, tuple[float, float]]
    version: str = "1"
    capability: OperatorCapability = OperatorCapability(
        "physics", "prepare", "environment"
    )

    def __post_init__(self) -> None:
        owned = {uid: tuple(bounds) for uid, bounds in self.ranges.items()}
        if (
            not self.name
            or not self.version
            or any(
                not uid
                or len(bounds) != 2
                or not all(math.isfinite(v) for v in bounds)
                or bounds[0] > bounds[1]
                for uid, bounds in owned.items()
            )
        ):
            raise ValueError("operator fields require named finite ordered ranges")
        object.__setattr__(self, "ranges", MappingProxyType(owned))

    def sample(self, context: ExpansionContext) -> Mapping[str, float]:
        """Sample without depending on batch size, slot or prior operator calls."""
        if context.source_kind not in self.capability.sources:
            raise ValueError("operator does not support the expert source")
        return MappingProxyType(
            {
                uid: context.random(self.name, self.version, uid).uniform(*bounds)
                for uid, bounds in self.ranges.items()
            }
        )
