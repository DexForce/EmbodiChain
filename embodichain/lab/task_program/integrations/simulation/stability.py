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

"""Typed geometric stability conditions owned by named settle presets."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import MISSING, fields
import math
from typing import TYPE_CHECKING

from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.sim.sensors.contact_sensor import ContactSensorCfg

__all__ = [
    "SupportSensorDeclarationCfg",
    "SimulationStabilityConditionCfg",
    "SimulationJointStabilityConditionCfg",
]


@configclass
class SupportSensorDeclarationCfg:
    """Explicit standard contact sensor used by one support pair.

    The condition's scene bindings supply the two native rigid UIDs. This
    declaration owns the stable sensor name and finite buffer capacity.
    """

    uid: str = MISSING
    max_contacts_per_env: int = 512

    def snapshot(self) -> SupportSensorDeclarationCfg:
        """Own and validate the explicit sensor declaration."""
        self.validate()
        if type(self.uid) is not str or not self.uid or self.uid != self.uid.strip():
            raise ValueError("Support sensor uid must be a nonempty exact identifier.")
        if type(self.max_contacts_per_env) is not int or self.max_contacts_per_env < 1:
            raise ValueError("Support sensor capacity must be a positive integer.")
        return deepcopy(self)

    def contact_sensor_cfg(
        self, object_uid: str, reference_uid: str
    ) -> ContactSensorCfg:
        """Resolve an explicit pair into the existing standard sensor schema.

        Args:
            object_uid: Exact native payload UID from the scene binding.
            reference_uid: Exact native support UID from the scene binding.

        Returns:
            An owned ContactSensorCfg that requires both actors in each contact.
        """
        from embodichain.lab.sim.sensors.contact_sensor import ContactSensorCfg

        owned = self.snapshot()
        return ContactSensorCfg(
            uid=owned.uid,
            rigid_uid_list=[object_uid, reference_uid],
            max_contacts_per_env=owned.max_contacts_per_env,
        )


@configclass
class SimulationStabilityConditionCfg:
    """Geometric/held/support condition in the standard settle-preset table.

    Fields retain the original duration, anchor-reset and held-failure semantics.
    Configurations are mutable while assembled; registration owns a validated
    snapshot and fingerprints the complete declaration.
    """

    entity: str = MISSING
    kind: str = MISSING
    duration: float = 3.0
    timeout: float = 12.0
    reference: str | None = None
    local_axis: tuple[float, float, float] | None = None
    reference_axis: tuple[float, float, float] | None = None
    displacement: tuple[float, float, float] | None = None
    target_position: tuple[float, float, float] | None = None
    reference_half_extents: tuple[float, float] | None = None
    object_bottom: float = 0.0
    reference_top: float = 0.0
    position_tolerance: float = 0.04
    minimum_alignment: float = 0.8660254037844386
    axis_direction_free: bool = False
    support_tolerance: float = 0.01
    translation_drift: float = 0.02
    rotation_drift: float = math.pi / 18.0
    motion_parts: tuple[str, ...] = ()
    world_axis: tuple[float, float, float] | None = None
    support_contract_revision: int | None = None
    com_tolerance: float | None = None
    support_sensor: SupportSensorDeclarationCfg | None = None

    def __post_init__(self) -> None:
        if self.entity is MISSING or self.kind is MISSING:
            return
        if type(self.axis_direction_free) is not bool or (
            self.axis_direction_free and self.local_axis is None
        ):
            raise ValueError(
                "Axis-direction freedom requires a boolean and a declared local axis."
            )
        if self.com_tolerance is not None and (
            type(self.com_tolerance) not in (int, float)
            or not math.isfinite(self.com_tolerance)
            or self.com_tolerance <= 0
        ):
            raise ValueError("COM tolerance must be finite and positive.")
        if self.support_contract_revision is not None and (
            type(self.support_contract_revision) is not int
            or self.support_contract_revision != 1
            or self.kind not in {"stack", "supported_placement", "supported_region"}
        ):
            raise ValueError("Unsupported native support contract declaration.")
        if self.kind not in {
            "upright",
            "stack",
            "hold",
            "placement",
            "supported_placement",
            "supported_region",
        }:
            raise ValueError(f"Unsupported task stability kind {self.kind!r}.")
        for name in ("entity", "reference"):
            value = getattr(self, name)
            if (name == "entity" or value is not None) and (
                type(value) is not str or not value or value != value.strip()
            ):
                raise ValueError(f"{name} must be an exact non-empty identifier.")
        for name in (
            "duration",
            "timeout",
            "position_tolerance",
            "support_tolerance",
            "translation_drift",
            "rotation_drift",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive.")
        if self.timeout < self.duration:
            raise ValueError("Stability timeout cannot precede its duration.")
        for name in ("object_bottom", "reference_top", "minimum_alignment"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite.")
        if not 0 <= self.minimum_alignment <= 1:
            raise ValueError("minimum_alignment must be in [0, 1].")
        for name in (
            "local_axis",
            "world_axis",
            "reference_axis",
            "displacement",
            "target_position",
            "reference_half_extents",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            size = 2 if name == "reference_half_extents" else 3
            if (
                not isinstance(value, (tuple, list))
                or len(value) != size
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(x)
                    for x in value
                )
            ):
                raise ValueError(f"{name} must contain {size} finite numbers.")
            vector = tuple(float(x) for x in value)
            if name.endswith("axis"):
                norm = math.sqrt(sum(x * x for x in vector))
                if norm <= 1e-8:
                    raise ValueError(f"{name} cannot be zero.")
                if not math.isfinite(norm):
                    raise ValueError(f"{name} cannot have a nonfinite norm.")
                if not math.isclose(norm, 1.0, rel_tol=0, abs_tol=1e-12):
                    vector = tuple(x / norm for x in vector)
            if name == "reference_half_extents" and min(vector) <= 0:
                raise ValueError("Support half extents must be positive.")
            object.__setattr__(self, name, vector)
        if not isinstance(self.motion_parts, (tuple, list)) or any(
            type(x) is not str or not x or x != x.strip() for x in self.motion_parts
        ):
            raise ValueError("motion_parts must contain control-part identifiers.")
        object.__setattr__(self, "motion_parts", tuple(self.motion_parts))
        if self.kind == "hold" and (
            not self.motion_parts
            or (self.target_position is None and self.local_axis is None)
        ):
            raise ValueError(
                "Held stability requires motion parts and a target position or axis."
            )
        if self.kind == "placement" and (
            self.target_position is None
            and (self.reference is None or self.displacement is None)
        ):
            raise ValueError("Placement stability requires an explicit target.")
        if self.kind == "supported_placement" and (
            self.reference is None or self.displacement is None
        ):
            raise ValueError(
                "Supported placement requires an explicit relative target."
            )
        if self.kind == "supported_region" and (
            self.reference is None
            or self.support_contract_revision is None
            and (self.displacement is not None or self.target_position is not None)
        ):
            raise ValueError(
                "Supported region requires an exact reference and no point target."
            )
        if self.target_position is not None and self.displacement is not None:
            raise ValueError(
                "Task stability cannot combine absolute and relative targets."
            )
        if self.kind == "stack" and (
            self.reference is None
            or self.reference_half_extents is None
            or self.local_axis is None
            or self.reference_axis is None
        ):
            raise ValueError("Stack stability requires support geometry and both axes.")
        if self.kind == "upright" and self.local_axis is None:
            raise ValueError("Upright stability requires its object-local axis.")
        if self.world_axis is not None and (
            self.kind != "hold" or self.local_axis is None
        ):
            raise ValueError("A world-axis hold requires its object-local axis.")
        if (self.reference is None) != (
            self.displacement is None
        ) and self.kind not in {"stack", "supported_region"}:
            raise ValueError("Relative targets require a reference and displacement.")

    def snapshot(self) -> SimulationStabilityConditionCfg:
        """Own, normalize and revalidate a complete named condition."""
        self.validate()
        owned = SimulationStabilityConditionCfg(
            **{
                field.name: deepcopy(getattr(self, field.name))
                for field in fields(self)
            }
        )
        if owned.kind in {"stack", "supported_placement", "supported_region"}:
            if type(owned.support_sensor) is not SupportSensorDeclarationCfg:
                raise ValueError(
                    "Supported stability requires an explicit support_sensor declaration."
                )
            owned.support_sensor = owned.support_sensor.snapshot()
        elif owned.support_sensor is not None:
            raise ValueError("Only support conditions may declare a support sensor.")
        return owned

    def to_condition_dict(self) -> dict[str, object]:
        """Return owned measurement parameters without sensor resource selection.

        Returns:
            Plain condition data for independent report matching. The complete
            registration seal still includes ``support_sensor``; consumers do
            not need a live runtime or this configuration class to inspect the
            returned values.
        """
        owned = self.snapshot()
        return {
            field.name: deepcopy(getattr(owned, field.name))
            for field in fields(owned)
            if field.name != "support_sensor"
        }


@configclass
class SimulationJointStabilityConditionCfg:
    """Explicit joint-position window in the existing named settle-preset table.

    The default window retains source articulation acceptance: one second
    within target tolerance and at most 0.001 runtime joint units of full-window
    drift (one millimetre for the migrated prismatic control), bounded by five
    seconds. These are declared policy values, not intrinsic
    properties of the asset or its joint. No root-pose substitution is used.
    """

    entity: str = MISSING
    """Canonical articulation identity from the scene binding."""
    joint: str = MISSING
    """Exact native joint name, with no name-based inference."""
    target: float = MISSING
    """Declared target coordinate, in the joint's native runtime units."""
    tolerance: float = MISSING
    """Inclusive absolute target tolerance in the same units."""
    duration: float = 1.0
    timeout: float = 5.0
    position_drift: float = 0.001

    def snapshot(self) -> SimulationJointStabilityConditionCfg:
        """Return a detached, normalized complete condition declaration."""
        self.validate()
        owned = deepcopy(self)
        for name in ("entity", "joint"):
            value = getattr(owned, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"Joint stability {name} must be an exact identifier.")
        for name in ("target", "tolerance", "duration", "timeout", "position_drift"):
            value = getattr(owned, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Joint stability {name} must be finite.")
            if name != "target" and value <= 0:
                raise ValueError(f"Joint stability {name} must be positive.")
            setattr(owned, name, float(value))
        if owned.timeout < owned.duration:
            raise ValueError("Joint stability timeout cannot precede its duration.")
        return owned

    def to_condition_dict(self) -> dict[str, object]:
        """Return detached policy values for independent normative matching."""
        owned = self.snapshot()
        return {
            field.name: deepcopy(getattr(owned, field.name)) for field in fields(owned)
        }
