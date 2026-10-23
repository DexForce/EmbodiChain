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

"""Named stability services; the shared bridge owns all execution decisions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, fields
import math
from typing import Any

import torch

from embodichain.lab.task_program.compiler.program import CompiledPostPolicy
from embodichain.utils.math import pose_inv

__all__: list[str] = []


def _contact_support_heights(
    supported: torch.Tensor,
    observation: Mapping[str, Any],
    like: torch.Tensor,
) -> torch.Tensor:
    """Refine heights only from NativeSupportObservation's qualified patch.

    Its contact points already require positive upward load and COM support.
    Keep the highest qualified point so penetration is not hidden by averaging.
    Unavailable rows cannot replace the existing conservative mesh envelope.
    """
    heights = torch.full_like(like, torch.nan)
    rows = observation.get("rows")
    if (
        supported.dtype != torch.bool
        or supported.shape != like.shape
        or observation.get("scope") != "native_contact_support"
        or not isinstance(rows, list)
        or len(rows) != like.shape[0]
    ):
        return heights
    for index, row in enumerate(rows):
        if not supported[index] or not isinstance(row, Mapping):
            continue
        values = tuple(
            row.get(name)
            for name in ("com_patch_distance_m", "com_tolerance_m", "upward_impulse")
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            continue
        distance, tolerance, impulse = values
        if tolerance <= 0 or not 0 <= distance <= tolerance or impulse <= 0:
            continue
        if row.get("contacts_world") is None or row.get("center_of_mass_world") is None:
            continue
        try:
            points = torch.as_tensor(
                row["contacts_world"], device=like.device, dtype=like.dtype
            )
            center = torch.as_tensor(
                row["center_of_mass_world"], device=like.device, dtype=like.dtype
            )
        except (TypeError, ValueError):
            continue
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not points.shape[0]
            or center.shape != (3,)
            or not torch.isfinite(points).all()
            or not torch.isfinite(center).all()
        ):
            continue
        heights[index] = points[:, 2].amax()
    return heights


@dataclass(frozen=True, slots=True)
class StabilityConstraint:
    """Immutable, fingerprinted definition of one task-owned stability preset."""

    entity: str
    kind: str
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

    def __post_init__(self) -> None:
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

    @classmethod
    def decode(cls, value: object) -> StabilityConstraint:
        """Reject undeclared payload fields before constructing any live service."""
        if type(value) is not dict or set(value) - {item.name for item in fields(cls)}:
            raise ValueError("Task stability constraint has unknown fields.")
        return cls(**value)


def _constraint_axis_alignment(
    cfg: StabilityConstraint, pose: torch.Tensor
) -> torch.Tensor:
    world = pose.new_tensor(cfg.world_axis or (0.0, 0.0, 1.0))
    alignment = ((pose[:, :3, :3] @ pose.new_tensor(cfg.local_axis)) * world).sum(-1)
    return alignment.abs() if cfg.axis_direction_free else alignment


def _constraint_pose_flags(
    cfg: StabilityConstraint,
    pose: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Recheck the unchanged static targets independently of a past stable mask."""
    valid = torch.isfinite(pose).all(dim=-1).all(dim=-1) & torch.isfinite(
        reference
    ).all(dim=-1).all(dim=-1)
    if cfg.local_axis is not None:
        valid &= _constraint_axis_alignment(cfg, pose) >= cfg.minimum_alignment
    target = None
    if cfg.target_position is not None:
        target = pose.new_tensor(cfg.target_position)
    elif cfg.displacement is not None:
        target = reference[:, :3, 3] + pose.new_tensor(cfg.displacement)
    if target is not None:
        valid &= (
            torch.linalg.vector_norm(pose[:, :3, 3] - target, dim=-1)
            <= cfg.position_tolerance
        )
    if cfg.kind == "stack":
        valid &= (reference[:, :3, :3] @ pose.new_tensor(cfg.reference_axis))[
            :, 2
        ] >= cfg.minimum_alignment
        gap = pose[:, 2, 3] + cfg.object_bottom - reference[:, 2, 3] - cfg.reference_top
        valid &= gap.abs() <= cfg.support_tolerance
        valid &= (
            (pose[:, :2, 3] - reference[:, :2, 3]).abs()
            <= pose.new_tensor(cfg.reference_half_extents)
        ).all(-1)
    return valid


class TaskStabilityPort:
    """Observe named stable conditions and return masks to the unmodified bridge.

    Only measurements and per-policy results are stored here. Task state,
    row eligibility, recovery, cancellation and completion belong to the core.
    """

    def __init__(
        self,
        delegate: Any,
        simulation: Any,
        robot: Any,
        scene_binding: Any,
        constraints: Mapping[str, StabilityConstraint],
        *,
        step_dt: float,
        support_observers: Mapping[tuple[str, str], Any] | None = None,
    ) -> None:
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("step_dt must be finite and positive.")
        self._delegate = delegate
        self._robot = robot
        self._dt = float(step_dt)
        self._constraints = dict(constraints)
        self._simulation = simulation
        self._rigid_bindings = {
            binding.entity_id: binding for binding in scene_binding.rigid_objects
        }
        self._support_observers = dict(support_observers or {})
        self._objects = {
            binding.entity_id: simulation.get_rigid_object(binding.simulation_uid)
            for binding in scene_binding.rigid_objects
        }
        self._references = {
            **self._objects,
            **{
                binding.entity_id: simulation.get_articulation(binding.simulation_uid)
                for binding in scene_binding.articulations
            },
        }
        articulation_bindings = {b.entity_id: b for b in scene_binding.articulations}
        self._link_references = {
            b.entity_id: (
                simulation.get_articulation(
                    articulation_bindings[b.articulation_id].simulation_uid
                ),
                b.native_link_name,
            )
            for b in getattr(scene_binding, "links", ())
            if any(c.reference == b.entity_id for c in self._constraints.values())
        }
        for cfg in self._constraints.values():
            if cfg.entity not in self._objects:
                raise ValueError(
                    "Task stability manipulated entities must be rigid objects."
                )
            for entity in (cfg.entity, cfg.reference):
                if entity is not None and (
                    entity not in self._references
                    and entity not in self._link_references
                    or entity in self._references
                    and self._references[entity] is None
                    or entity in self._link_references
                    and self._link_references[entity][0] is None
                ):
                    raise ValueError(f"Task stability entity {entity!r} is not bound.")
            if cfg.reference not in self._objects and cfg.kind in {
                "stack",
                "supported_placement",
                "supported_region",
            }:
                raise ValueError(
                    "Articulated support acceptance requires qualified link geometry."
                )
        self._support_vertices: dict[str, torch.Tensor] = {}
        for cfg in self._constraints.values():
            if cfg.kind not in {"supported_placement", "supported_region"}:
                continue
            for entity in (cfg.entity, cfg.reference):
                if entity in self._support_vertices:
                    continue
                vertices = self._objects[entity].get_vertices(scale=True)
                if (
                    not isinstance(vertices, torch.Tensor)
                    or vertices.ndim != 3
                    or vertices.shape[-1] != 3
                    or vertices.shape[1] == 0
                    or vertices.shape[0] != self._pose(entity).shape[0]
                    or not torch.isfinite(vertices).all()
                ):
                    raise ValueError(
                        "Support geometry requires finite per-environment vertices."
                    )
                self._support_vertices[entity] = vertices.detach().clone()
        from .placement_evidence import _prepare_placement_geometry

        self._region_geometry = {}
        for cfg in self._constraints.values():
            key = (cfg.entity, cfg.reference)
            if cfg.kind != "supported_region" or key in self._region_geometry:
                continue
            geometry = _prepare_placement_geometry(
                simulation,
                self._rigid_bindings[cfg.entity],
                self._rigid_bindings[cfg.reference],
                self._pose(cfg.entity).shape[0],
            )
            self._region_geometry[key] = geometry
        self._results: dict[int, torch.Tensor] = {}
        from .support_evidence import NativeSupportObservation

        for cfg in self._constraints.values():
            if cfg.kind not in {"stack", "supported_placement", "supported_region"}:
                continue
            key = (cfg.entity, cfg.reference)
            if key not in self._support_observers:
                self._support_observers[key] = NativeSupportObservation(
                    simulation,
                    self._objects[cfg.entity],
                    self._objects[cfg.reference],
                    entity_id=cfg.entity,
                    reference_id=cfg.reference,
                )
        self._metadata: dict[int, dict[str, Any]] = {}
        self._clock: Any = None

    def bind_clock(self, clock: Any) -> None:
        """Bind strong support readers to the already-owned execution clock."""
        if any(
            cfg.support_contract_revision == 1 for cfg in self._constraints.values()
        ) and (
            clock is None
            or not math.isclose(clock.step_dt, self._dt, rel_tol=0, abs_tol=1e-9)
        ):
            raise ValueError("Strong support clock differs from the control cadence.")
        self._clock = clock
        for cfg in self._constraints.values():
            if cfg.support_contract_revision == 1:
                self._support_observers[(cfg.entity, cfg.reference)].bind_clock(clock)

    def _support(
        self, cfg: StabilityConstraint, pose: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        observer = self._support_observers[(cfg.entity, cfg.reference)]
        if cfg.com_tolerance is None:
            if cfg.support_contract_revision == 1:
                return observer.observe(pose, reference, support_contract_revision=1)
            return observer.observe(pose, reference)
        previous = observer.tolerance
        try:
            if cfg.com_tolerance is not None:
                observer.tolerance = cfg.com_tolerance
            if cfg.support_contract_revision == 1:
                return observer.observe(pose, reference, support_contract_revision=1)
            return observer.observe(pose, reference)
        finally:
            observer.tolerance = previous

    def validate_policy(self, policy: Any, *, segment: Any) -> None:
        """Validate declared identity and membership without reading live state."""
        if type(policy) is not CompiledPostPolicy:
            raise TypeError("Expected an exact CompiledPostPolicy.")
        if policy.cfg.preset not in self._constraints:
            self._delegate.validate_policy(policy, segment=segment)
            return
        if not any(item is policy for item in segment.post_policies):
            raise ValueError("Task stability policy does not belong to this segment.")
        cfg = self._constraints[policy.cfg.preset]
        if policy.cfg.kind != "wait_stable" or policy.entity.entity_id != cfg.entity:
            raise ValueError("Task stability declaration and compiled policy disagree.")

    def _pose(self, entity: str) -> torch.Tensor:
        if entity in self._link_references:
            articulation, link = self._link_references[entity]
            pose = articulation.get_link_pose(link, to_matrix=True)
        else:
            pose = self._references[entity].get_local_pose(to_matrix=True)
        if (
            not isinstance(pose, torch.Tensor)
            or pose.ndim != 3
            or pose.shape[-2:] != (4, 4)
        ):
            raise ValueError("Object poses must have shape (num_envs, 4, 4).")
        return pose.clone()

    @staticmethod
    def _drift(
        pose: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        translation = torch.linalg.vector_norm(
            pose[:, :3, 3] - reference[:, :3, 3], dim=-1
        )
        rotation = reference[:, :3, :3].transpose(-1, -2) @ pose[:, :3, :3]
        cosine = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
        return translation, torch.acos(cosine)

    def _attachments(
        self, pose: torch.Tensor, parts: tuple[str, ...]
    ) -> tuple[torch.Tensor, ...]:
        qpos = self._robot.get_qpos()
        return tuple(
            pose_inv(pose)
            @ self._robot.compute_fk(
                qpos=qpos[:, self._robot.get_joint_ids(name=part)],
                name=part,
                to_matrix=True,
            )
            for part in parts
        )

    def actions(
        self,
        policy: Any,
        *,
        segment: Any,
        active_mask: torch.Tensor,
    ) -> Iterator[torch.Tensor]:
        """Yield target-qpos holds; only Gym may consume and apply them."""
        self.validate_policy(policy, segment=segment)
        if policy.cfg.preset not in self._constraints:
            yield from self._delegate.actions(
                policy, segment=segment, active_mask=active_mask
            )
            return
        cfg = self._constraints[policy.cfg.preset]
        initial = self._pose(cfg.entity)
        if cfg.support_contract_revision == 1 and self._clock is None:
            raise ValueError("Strong support requires the canonical execution clock.")
        clock_start = (
            None if self._clock is None else (self._clock.step_index, self._clock.now())
        )
        if active_mask.dtype != torch.bool or active_mask.shape != initial.shape[:1]:
            raise ValueError(
                "Task stability active rows must match the observed batch."
            )
        active = active_mask.clone()
        needed = math.ceil(cfg.duration / self._dt)
        maximum = math.ceil(cfg.timeout / self._dt)
        consecutive = torch.zeros_like(active, dtype=torch.long)
        anchor = initial.clone()
        reference_anchor = (
            self._pose(cfg.reference)
            if cfg.kind in {"stack", "supported_placement", "supported_region"}
            else None
        )
        initial_attachments = self._attachments(initial, cfg.motion_parts)
        failed = torch.zeros_like(active)
        for elapsed in range(maximum + 1):
            if cfg.support_contract_revision == 1 and (
                self._clock.step_index != clock_start[0] + elapsed
                or not math.isclose(
                    self._clock.now(),
                    clock_start[1] + elapsed * self._dt,
                    rel_tol=0,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError(
                    "Strong stability requires acknowledged environment steps."
                )
            pose = self._pose(cfg.entity)
            valid = torch.isfinite(pose).all(dim=-1).all(dim=-1)
            measurements: dict[str, Any] = {}
            if cfg.local_axis is not None:
                alignment = _constraint_axis_alignment(cfg, pose)
                valid &= alignment >= cfg.minimum_alignment
                measurements["alignment"] = alignment.tolist()
            reference = self._pose(cfg.reference) if cfg.reference is not None else None
            target = None
            if cfg.target_position is not None:
                target = pose.new_tensor(cfg.target_position)
            elif reference is not None and cfg.displacement is not None:
                target = reference[:, :3, 3] + pose.new_tensor(cfg.displacement)
            if target is not None:
                error = torch.linalg.vector_norm(pose[:, :3, 3] - target, dim=-1)
                valid &= error <= cfg.position_tolerance
                measurements["position_error"] = error.tolist()
                if cfg.reference in self._link_references:
                    measurements.update(
                        entity=cfg.entity,
                        reference=cfg.reference,
                        object_pose=pose.tolist(),
                        reference_pose=reference.tolist(),
                    )
            if cfg.kind == "stack":
                assert (
                    reference is not None
                    and reference_anchor is not None
                    and cfg.reference_axis is not None
                )
                alignment = (
                    reference[:, :3, :3] @ pose.new_tensor(cfg.reference_axis)
                )[:, 2]
                gap = (
                    pose[:, 2, 3]
                    + cfg.object_bottom
                    - reference[:, 2, 3]
                    - cfg.reference_top
                )
                delta = pose[:, :2, 3] - reference[:, :2, 3]
                valid &= (alignment >= cfg.minimum_alignment) & (
                    gap.abs() <= cfg.support_tolerance
                )
                valid &= (
                    delta.abs() <= pose.new_tensor(cfg.reference_half_extents)
                ).all(-1)
                reference_translation, reference_rotation = self._drift(
                    reference, reference_anchor
                )
                valid &= (reference_translation <= cfg.translation_drift) & (
                    reference_rotation <= cfg.rotation_drift
                )
                measurements.update(
                    support_gap=gap.tolist(),
                    reference_alignment=alignment.tolist(),
                    reference_translation_drift=reference_translation.tolist(),
                    reference_rotation_drift=reference_rotation.tolist(),
                )
            if cfg.kind in {"supported_placement", "supported_region"}:
                assert reference is not None and reference_anchor is not None
                vertices = self._support_vertices[cfg.entity].to(pose)
                support_vertices = self._support_vertices[cfg.reference].to(reference)
                bottom = (vertices @ pose[:, 2, :3, None]).squeeze(-1).amin(-1) + pose[
                    :, 2, 3
                ]
                support_world = (
                    support_vertices @ reference[:, :3, :3].transpose(-1, -2)
                    + reference[:, None, :3, 3]
                )
                top = support_world[:, :, 2].amax(-1)
                supported, support = self._support(cfg, pose, reference)
                contact_heights = _contact_support_heights(supported, support, top)
                top = torch.where(torch.isfinite(contact_heights), contact_heights, top)
                gap = bottom - top
                # Preserve the support-envelope check using observed geometry.
                # Projected mesh bounds alone are not contact evidence.
                lower = support_world[:, :, :2].amin(1) + 0.002
                upper = support_world[:, :, :2].amax(1) - 0.002
                inside = ((pose[:, :2, 3] >= lower) & (pose[:, :2, 3] <= upper)).all(-1)
                if cfg.kind == "supported_region":
                    from .placement_evidence import measure_planar_placement

                    planar, landing = measure_planar_placement(
                        self._simulation,
                        self._rigid_bindings[cfg.entity],
                        self._rigid_bindings[cfg.reference],
                        pose,
                        reference,
                        _geometry=self._region_geometry[(cfg.entity, cfg.reference)],
                    )
                    full = torch.tensor(
                        landing["full_footprint_inside"],
                        device=pose.device,
                        dtype=torch.bool,
                    )
                    valid &= torch.isfinite(contact_heights)
                    valid &= (
                        torch.tensor(planar, device=pose.device, dtype=torch.bool)
                        & full
                    )
                    measurements.update(
                        landing_region=landing,
                        full_footprint_inside=full.tolist(),
                        support_tolerance_m=cfg.support_tolerance,
                        origin_inside_support_aabb=inside.tolist(),
                    )
                else:
                    valid &= inside
                    measurements["support_footprint"] = inside.tolist()
                valid &= (gap.abs() <= cfg.support_tolerance) & supported
                reference_translation, reference_rotation = self._drift(
                    reference, reference_anchor
                )
                valid &= (reference_translation <= cfg.translation_drift) & (
                    reference_rotation <= cfg.rotation_drift
                )
                measurements.update(
                    support_gap=gap.tolist(),
                    reference_translation_drift=reference_translation.tolist(),
                    reference_rotation_drift=reference_rotation.tolist(),
                    native_support=support,
                )
            if cfg.kind == "stack":
                supported, support = self._support(cfg, pose, reference)
                valid &= supported
                measurements["native_support"] = support
            translation, rotation = self._drift(pose, anchor)
            valid &= (translation <= cfg.translation_drift) & (
                rotation <= cfg.rotation_drift
            )
            for current, original in zip(
                self._attachments(pose, cfg.motion_parts),
                initial_attachments,
                strict=True,
            ):
                slip, turn = self._drift(current, original)
                valid &= (slip <= cfg.translation_drift) & (turn <= cfg.rotation_drift)
                measurements.setdefault("attachment_translation", []).append(
                    slip.tolist()
                )
            if cfg.kind == "hold":
                failed |= active & ~valid
            else:
                anchor = torch.where(valid[:, None, None], anchor, pose)
                if reference_anchor is not None:
                    # Both objects must share the same uninterrupted stable window.
                    reference_anchor = torch.where(
                        valid[:, None, None], reference_anchor, reference
                    )
            if elapsed:
                consecutive = torch.where(valid & active & ~failed, consecutive + 1, 0)
            accepted = active & (consecutive >= needed) & ~failed
            self._results[id(policy)] = accepted.clone()
            self._metadata[id(policy)] = {
                "kind": cfg.kind,
                "preset": policy.cfg.preset,
                "elapsed_seconds": elapsed * self._dt,
                "required_seconds": cfg.duration,
                "stable_seconds": (consecutive * self._dt).tolist(),
                "accepted_mask": accepted.tolist(),
                "failed_mask": failed.tolist(),
                "measurements": measurements,
            }
            if cfg.support_contract_revision == 1:
                self._metadata[id(policy)].update(
                    support_contract_revision=1,
                    entity=cfg.entity,
                    reference=cfg.reference,
                    clock_started_at=clock_start[1],
                    clock_ended_at=self._clock.now(),
                    start_step=clock_start[0],
                    end_step=self._clock.step_index,
                    stable_seconds=[
                        int(value) * self._dt for value in consecutive.tolist()
                    ],
                )
            if not (active & ~accepted & ~failed).any() or elapsed == maximum:
                return
            targets = self._robot.get_qpos(target=True)
            observed = self._robot.get_qpos()
            if targets.shape != observed.shape or not torch.isfinite(targets).all():
                raise ValueError(
                    "Target-qpos holds require finite full robot commands."
                )
            yield torch.where(
                active[:, None] & ~failed[:, None], targets, observed
            ).clone()

    def post_policy_result(self, policy: Any, *, segment: Any) -> torch.Tensor:
        """Publish measurements; the core decides execution eligibility."""
        if policy.cfg.preset not in self._constraints:
            return self._delegate.post_policy_result(policy, segment=segment)
        self.validate_policy(policy, segment=segment)
        return self._results[id(policy)].clone()

    def post_policy_metadata(self, policy: Any, *, segment: Any) -> Mapping[str, Any]:
        """Expose independently owned physical measurements."""
        if policy.cfg.preset not in self._constraints:
            return self._delegate.post_policy_metadata(policy, segment=segment)
        self.validate_policy(policy, segment=segment)
        return deepcopy(self._metadata[id(policy)])
