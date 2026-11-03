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

"""Runtime execution of explicitly declared geometric stability conditions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from copy import copy, deepcopy
from dataclasses import dataclass
import math
from typing import Any
import torch
from embodichain.compute.geometry.rigid import (
    rigid_pose_displacement,
    pose_axis_alignment,
)
from embodichain.lab.sim.scene.planar_observation import NativePlanarPlacement
from embodichain.lab.sim.evidence import PhysicalEvidenceBatch
from embodichain.lab.task_program.compiler.program import CompiledPostPolicy
from embodichain.utils.math import pose_inv
from .stability import SimulationStabilityConditionCfg
from embodichain.lab.sim.sensors.support_observation import (
    SupportControlClock,
    SupportObservationRegistry,
)

__all__: list[str] = []


@dataclass
class _Lifetime:
    closed: bool = False


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


class _ConditionStabilityRuntime:
    """Observe named stable conditions and return masks to the unmodified bridge.

    Only measurements and per-policy results are stored here. Task state,
    row eligibility, recovery, cancellation and completion belong to the core.
    """

    def __init__(
        self,
        simulation: Any,
        robot: Any,
        scene_binding: Any,
        presets: Mapping[str, SimulationStabilityConditionCfg],
        *,
        step_dt: float,
        registry: SupportObservationRegistry | None,
        sensor_suite: MutableMapping[str, Any] | None,
    ) -> None:
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("step_dt must be finite and positive.")
        self._robot = robot
        self._dt = float(step_dt)
        self._presets = {name: cfg.snapshot() for name, cfg in presets.items()}
        self._simulation = simulation
        self._rigid_bindings = {
            binding.entity_id: binding for binding in scene_binding.rigid_objects
        }
        self._support_observers = {}
        self._registry = registry
        self._sensor_suite = sensor_suite
        self._lifetime = _Lifetime()
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
            if any(c.reference == b.entity_id for c in self._presets.values())
        }
        for cfg in self._presets.values():
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
        for cfg in self._presets.values():
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
        self._region_geometry = {}
        for cfg in self._presets.values():
            key = (cfg.entity, cfg.reference)
            if cfg.kind != "supported_region" or key in self._region_geometry:
                continue
            geometry = NativePlanarPlacement(
                simulation,
                self._rigid_bindings[cfg.entity].simulation_uid,
                self._rigid_bindings[cfg.reference].simulation_uid,
                batch_size=self._pose(cfg.entity).shape[0],
            )
            self._region_geometry[key] = geometry
        self._results: dict[int, torch.Tensor] = {}

        for cfg in self._presets.values():
            if cfg.kind not in {"stack", "supported_placement", "supported_region"}:
                continue
            key = (cfg.entity, cfg.reference)
            if sensor_suite is None or registry is None:
                raise ValueError(
                    "Supported stability requires a registry and the host sensor suite."
                )
            assert cfg.support_sensor is not None
            obj, ref = self._objects[cfg.entity], self._objects[cfg.reference]
            uid = cfg.support_sensor.uid
            if uid in sensor_suite and sensor_suite[uid] is not simulation.get_sensor(
                uid
            ):
                raise ValueError(
                    "Host support sensor UID belongs to a different instance."
                )
            observer = registry.acquire(
                obj,
                ref,
                entity_id=cfg.entity,
                reference_id=cfg.reference,
                observer_name=f"native_support:{cfg.entity}:{cfg.reference}",
                sensor_cfg=cfg.support_sensor.contact_sensor_cfg(obj.uid, ref.uid),
            )
            uid = observer.sensor.cfg.uid
            if uid in sensor_suite and sensor_suite[uid] is not observer.sensor:
                raise ValueError(
                    "Host support sensor UID belongs to a different instance."
                )
            sensor_suite[uid] = observer.sensor
            self._support_observers[key] = observer
        self._metadata: dict[int, dict[str, Any]] = {}
        self._clock: SupportControlClock | None = None

    def bind_clock(self, clock: SupportControlClock) -> None:
        """Bind strong support readers to the already-owned execution clock."""
        self._require_open()
        if any(
            cfg.support_contract_revision == 1 for cfg in self._presets.values()
        ) and (
            clock is None
            or not math.isclose(clock.step_dt, self._dt, rel_tol=0, abs_tol=1e-9)
        ):
            raise ValueError("Strong support clock differs from the control cadence.")
        self._clock = clock
        for cfg in self._presets.values():
            if cfg.support_contract_revision == 1:
                self._support_observers[(cfg.entity, cfg.reference)].bind_clock(clock)

    def _support(
        self,
        cfg: SimulationStabilityConditionCfg,
        pose: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[PhysicalEvidenceBatch, dict[str, Any]]:
        observer = self._support_observers[(cfg.entity, cfg.reference)]
        result = observer.observe(
            pose,
            reference,
            support_contract_revision=cfg.support_contract_revision,
            com_tolerance=cfg.com_tolerance,
        )
        # Unavailable rows must remain distinguishable from a measured false.
        # The caller continues holds so an initial/reset window can qualify.
        return result.support, dict(result.metadata)

    def validate_policy(self, policy: Any, *, segment: Any) -> None:
        """Validate declared identity and membership without reading live state."""
        if type(policy) is not CompiledPostPolicy:
            raise TypeError("Expected an exact CompiledPostPolicy.")
        if not any(item is policy for item in segment.post_policies):
            raise ValueError("Task stability policy does not belong to this segment.")
        cfg = self._presets[policy.cfg.preset]
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
        return rigid_pose_displacement(pose, reference)

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
        self._require_open()
        self.validate_policy(policy, segment=segment)
        cfg = self._presets[policy.cfg.preset]
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
        unavailable_observations = torch.zeros_like(active, dtype=torch.long)
        last_acquisition_errors: list[str | None] = [None] * active.numel()
        for elapsed in range(maximum + 1):
            self._require_open()
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
            support_batch: PhysicalEvidenceBatch | None = None
            if cfg.local_axis is not None:
                alignment = pose_axis_alignment(
                    pose,
                    cfg.local_axis,
                    world_axis=cfg.world_axis,
                    direction_free=cfg.axis_direction_free,
                )
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
                if reference is not None and (
                    cfg.kind == "placement" or cfg.reference in self._link_references
                ):
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
                support_batch, support = self._support(cfg, pose, reference)
                supported = support_batch.values & support_batch.valid
                contact_heights = _contact_support_heights(supported, support, top)
                top = torch.where(torch.isfinite(contact_heights), contact_heights, top)
                gap = bottom - top
                # Preserve the support-envelope check using observed geometry.
                # Projected mesh bounds alone are not contact evidence.
                lower = support_world[:, :, :2].amin(1) + 0.002
                upper = support_world[:, :, :2].amax(1) - 0.002
                inside = ((pose[:, :2, 3] >= lower) & (pose[:, :2, 3] <= upper)).all(-1)
                if cfg.kind == "supported_region":
                    landing = self._region_geometry[
                        (cfg.entity, cfg.reference)
                    ].measure(pose.detach().cpu(), reference.detach().cpu())
                    # Preserve the source planar-region qualifier separately
                    # from the measured contact-height gap below.
                    planar = [
                        inside and -0.01 <= gap <= 0.05
                        for inside, gap in zip(
                            landing["full_footprint_inside"],
                            landing["support_gap_m"],
                            strict=True,
                        )
                    ]
                    landing.update(
                        support_tolerance_m=0.05, penetration_tolerance_m=0.01
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
                support_batch, support = self._support(cfg, pose, reference)
                supported = support_batch.values & support_batch.valid
                valid &= supported
                measurements["native_support"] = support
            if support_batch is not None:
                unavailable_observations += active & ~support_batch.valid
                last_acquisition_errors = [
                    error if error is not None else previous
                    for error, previous in zip(
                        support_batch.acquisition_errors,
                        last_acquisition_errors,
                        strict=True,
                    )
                ]
                measurements["support_acquisition"] = {
                    "evidence_id": support_batch.evidence_id,
                    "values": support_batch.values.tolist(),
                    "valid": support_batch.valid.tolist(),
                    "acquisition_errors": list(support_batch.acquisition_errors),
                    "timestamp": support_batch.timestamp,
                    "env_ids": support_batch.env_ids.tolist(),
                    "observation_revision": support_batch.observation_revision,
                }
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
            if support_batch is not None:
                self._metadata[id(policy)].update(
                    support_unavailable_observations=unavailable_observations.tolist(),
                    support_last_acquisition_errors=list(last_acquisition_errors),
                    support_result_available=(
                        accepted | (unavailable_observations == 0)
                    ).tolist(),
                )
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
        self.validate_policy(policy, segment=segment)
        return self._results[id(policy)].clone()

    def post_policy_metadata(self, policy: Any, *, segment: Any) -> Mapping[str, Any]:
        """Expose independently owned physical measurements."""
        self.validate_policy(policy, segment=segment)
        return deepcopy(self._metadata[id(policy)])

    def _require_open(self) -> None:
        if self._lifetime.closed:
            raise RuntimeError("Stability runtime resources are closed.")

    def close(self) -> None:
        self._lifetime.closed = True

    def new_session(self) -> _ConditionStabilityRuntime:
        result = copy(self)
        result._results = {}
        result._metadata = {}
        return result
