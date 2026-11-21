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
import json

__all__ = [
    "BatchCompatibility",
    "RecipeOperation",
    "ExpansionRecipe",
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

    def __post_init__(self) -> None:
        if (
            self.domain not in {"physics", "motion", "scene", "visual"}
            or self.phase not in {"prepare", "plan", "render"}
            or self.scope not in {"environment", "world", "shared_resource"}
        ):
            raise ValueError("invalid operator domain, phase or scope")
        for name in ("backends", "sources"):
            values = tuple(getattr(self, name))
            if not values or len(set(values)) != len(values):
                raise ValueError("capability sets must be nonempty and unique")
            object.__setattr__(self, name, values)


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


@dataclass(frozen=True)
class BatchCompatibility:
    """Physical resources that must agree within one World cohort.

    Capacity and arena indices are deliberately excluded. Resource revisions
    identify topology, program structure, schema and shared appearance by content.
    """

    robot: str
    joint_names: tuple[str, ...]
    topology_revision: str
    program_revision: str
    backend: str
    control_dt: float
    observation_revision: str
    world_profile: str
    shared_resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for key in (
            "robot",
            "topology_revision",
            "program_revision",
            "observation_revision",
            "world_profile",
        ):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"compatibility {key} must be nonempty")
        if (
            self.backend not in {"default", "newton"}
            or not math.isfinite(self.control_dt)
            or self.control_dt <= 0
        ):
            raise ValueError("invalid backend or control clock")
        names = tuple(self.joint_names)
        if (
            not names
            or len(names) != len(set(names))
            or any(not isinstance(v, str) or not v for v in names)
        ):
            raise ValueError("compatibility requires unique joint names")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "shared_resources", tuple(self.shared_resources))

    @property
    def key(self) -> str:
        """Return the canonical cohort identity, independent of capacity."""
        from dataclasses import asdict

        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class RecipeOperation:
    """One immutable sampled operation and its execution permissions."""

    name: str
    version: str
    capability: OperatorCapability
    payload_json: str

    def __post_init__(self) -> None:
        if (
            not self.name
            or not self.version
            or not isinstance(self.capability, OperatorCapability)
        ):
            raise ValueError("operation requires a versioned capability")
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise ValueError("operation payload must be a mapping")
        object.__setattr__(
            self,
            "payload_json",
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )


@dataclass(frozen=True)
class ExpansionRecipe:
    """Source-pinned logical recipe shared by physical execution attempts.

    Args:
        context: Slot-independent source, seed and ordinal identity.
        compatibility: Exact World/resource requirements.
        operations: Sampled immutable domain changes in preparation/plan/render order.
    """

    context: ExpansionContext
    compatibility: BatchCompatibility
    operations: tuple[RecipeOperation, ...]

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        order = {"prepare": 0, "plan": 1, "render": 2}
        previous = -1
        for operation in operations:
            if not isinstance(operation, RecipeOperation):
                raise TypeError("recipe requires RecipeOperation values")
            cap = operation.capability
            if (
                self.context.source_kind not in cap.sources
                or self.compatibility.backend not in cap.backends
            ):
                raise ValueError(
                    f"{operation.name} does not support this source/backend"
                )
            current = order[cap.phase]
            if current < previous:
                raise ValueError("recipe operations violate phase ordering")
            previous = current
        if len({op.name for op in operations}) != len(operations):
            raise ValueError("a recipe must have one owner per operation")
        object.__setattr__(self, "operations", operations)

    def to_dict(self) -> dict[str, object]:
        """Return an owned JSON-serializable recipe without physical slot data."""
        from dataclasses import asdict

        return asdict(self)

    @property
    def recipe_id(self) -> str:
        """Identify the recipe across attempts, batch sizes and writer retries."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExpansionRecipe:
        """Decode a frozen recipe, validating permissions before any native write."""
        data = dict(value)
        data["context"] = ExpansionContext(**data["context"])
        data["compatibility"] = BatchCompatibility(**data["compatibility"])
        data["operations"] = tuple(
            RecipeOperation(
                **{**op, "capability": OperatorCapability(**op["capability"])}
            )
            for op in data["operations"]
        )
        return cls(**data)

    def validate_host(
        self,
        supported: Mapping[str, OperatorCapability],
        compatibility: BatchCompatibility,
    ) -> None:
        """Reject unsupported phases/scopes/resources before host preparation."""
        if compatibility != self.compatibility:
            raise ValueError("recipe requires a different World cohort")
        for operation in self.operations:
            expected = supported.get(operation.name)
            if expected is None or expected != operation.capability:
                raise ValueError(
                    f"host cannot apply {operation.name} with its declared scope"
                )
