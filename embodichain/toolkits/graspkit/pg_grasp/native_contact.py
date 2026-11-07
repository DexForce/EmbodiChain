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

"""Closing/opening qualification over captured convex geometry.

The host supplies isolated collision groups. This module imports no simulator,
Robot, Task Engine, or planner backend. Geometric qualification is independent
of measured acquisition/release evidence and experiment revision selection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
import time
from types import TracebackType
from typing import Protocol

import numpy as np
import torch
import trimesh

from embodichain.utils import configclass

__all__ = [
    "NativeQueryScope",
    "NativeContactPart",
    "NativeClosingQuery",
    "NativeOpeningQuery",
    "NativeRigidRestScene",
    "NativeRigidObstacle",
    "ClosingContactPolicy",
    "ConvexMeshQuery",
]


class _DistanceInterval(Protocol):
    lower: float

    def below(self, threshold: float) -> bool: ...


class ConvexMeshQuery(Protocol):
    """Owned convex geometry service supplied by the host collision backend."""

    def add_object(
        self, name: str, mesh: trimesh.Trimesh, transform: np.ndarray | None = None
    ) -> None:
        """Bind one ordered convex shape.

        Args:
            name: Unique shape identity within this query group.
            mesh: Convex solid in its local frame.
            transform: Optional local-to-query-frame pose.
        """
        ...

    def set_transform(self, name: str, transform: np.ndarray) -> None:
        """Update one query shape without changing the simulation.

        Args:
            name: Previously bound shape identity.
            transform: New local-to-query-frame rigid pose.
        """
        ...

    def contact_query(self, other: ConvexMeshQuery, limit: float) -> _DistanceInterval:
        """Return native surface predicates and bounded clearance facts.

        Args:
            other: Compatible group from the same collision backend.
            limit: Nonnegative query bound in metres.

        Returns:
            Clearance interval whose ``below`` method rejects ambiguous thresholds.
        """
        ...

    def close(self) -> None:
        """Release owned query resources."""
        ...


@configclass
class ClosingContactPolicy:
    """Explicit geometric qualification rules, independent of experiment revisions.

    Host integrations select these rules. Defaults preserve the rest-order query;
    surface-based qualification remains owned by ``HandContactChecker``.
    """

    contact_halo: str = "ignore"
    """How nonpad contact halos are used: ``ignore``, ``reject`` or ``report``."""
    reject_open_pad_contact: bool = False
    """Reject candidates whose pads are already at rest contact when fully open."""
    depth_alternatives: bool = False
    """Offer the five existing pad-bounded contact-depth offsets."""
    opening_check: bool = False
    """Apply an explicitly attached release-opening query."""
    defer_source_proposals: bool = False
    """Record externally deferred source proposal qualifications."""
    horizontal_pad_line: bool = False
    """Require the sampled pad-center line within two degrees of horizontal."""

    def __post_init__(self) -> None:
        if self.contact_halo not in ("ignore", "reject", "report"):
            raise ValueError("Unknown contact-halo policy.")
        for name in (
            "reject_open_pad_contact",
            "depth_alternatives",
            "opening_check",
            "defer_source_proposals",
            "horizontal_pad_line",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean.")


@dataclass(frozen=True, slots=True)
class NativeQueryScope:
    """Explicit identity for one fresh preparation and its B1 observation.

    Args:
        query_id: Fresh preparation identity from a source such as UUID, independent
            of planning RNG. Replans need a new identity even for the same invocation.
        observation_timestamp: Authoritative robot observation time in seconds.
        scene_version: Version of the same observed scene.
        env_id: Supported native environment row; must be zero.
    """

    query_id: str
    observation_timestamp: float
    scene_version: int
    env_id: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.query_id) is not str
            or not self.query_id
            or self.query_id != self.query_id.strip()
        ):
            raise ValueError("A native query requires an explicit fresh query_id.")
        if (
            type(self.observation_timestamp) not in (int, float)
            or not math.isfinite(self.observation_timestamp)
            or self.observation_timestamp < 0
        ):
            raise ValueError(
                "Native query observation time must be finite and nonnegative."
            )
        if type(self.scene_version) is not int or self.scene_version < 0:
            raise ValueError(
                "Native query scene_version must be a nonnegative integer."
            )
        if type(self.env_id) is not int or self.env_id != 0:
            raise ValueError("Native hand queries support only B1/env0.")


class _QueryLifetime:
    def _start_scope(self, scope: NativeQueryScope) -> None:
        if not isinstance(scope, NativeQueryScope):
            raise TypeError("scope must be NativeQueryScope.")
        self._scope = scope
        self._closed = False

    @property
    def scope(self) -> NativeQueryScope:
        """Return this query's immutable preparation/observation scope."""
        return self._scope

    @property
    def closed(self) -> bool:
        """Whether this owned query lifetime has ended."""
        return self._closed

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError(
                "Native query is closed; capture a fresh query for new work."
            )


def _neutral_identity(value: Mapping[str, object]) -> dict[str, object]:
    active: set[int] = set()

    def check(item: object) -> None:
        container = isinstance(item, (Mapping, list, tuple))
        if container:
            if id(item) in active:
                raise ValueError("Native identity metadata must not contain cycles.")
            active.add(id(item))
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise ValueError("Native identity metadata keys must be strings.")
            for child in item.values():
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise TypeError("Native query identity accepts only neutral JSON values.")
        if container:
            active.remove(id(item))

    if not isinstance(value, Mapping):
        raise TypeError("Native identity metadata must be a mapping.")
    check(value)
    return json.loads(json.dumps(dict(value), allow_nan=False))


def _rigid_pose(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (4, 4)
        or not np.isfinite(result).all()
        or not np.allclose(result[3], (0, 0, 0, 1))
        or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=2e-5)
        or not np.isclose(np.linalg.det(result[:3, :3]), 1.0, atol=2e-5)
    ):
        raise ValueError("Native hand queries require a finite rigid pose.")
    return result.copy()


def _mesh_box(meshes: tuple[trimesh.Trimesh, ...]) -> np.ndarray:
    vertices = np.concatenate([np.asarray(mesh.vertices) for mesh in meshes])
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Native closing bounds require finite nonempty vertices")
    bounds = np.stack((vertices.min(0), vertices.max(0)))
    return np.asarray(list(product(*bounds.T)), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class NativeContactPart:
    """One captured collision link and its nominal TCP-relative closing FK grid.

    Args:
        name: Exact declared physical link name.
        meshes: Body-local collision meshes, borrowed read-only. Native builders
            copy cooked shapes once; manually supplied meshes do not certify provenance.
        tcp_poses: Link-to-TCP transforms, shape (G, 4, 4), copied on construction.
        rest_offset: Original finite nonnegative native rest offset in metres.
        pad: Whether this exact link is one of the two declared contact pads.
        contact_offset: Original native contact offset, required when contact halos are used.
    """

    name: str
    meshes: tuple[trimesh.Trimesh, ...]
    tcp_poses: np.ndarray
    rest_offset: float
    pad: bool
    contact_offset: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.name) is not str
            or not self.name
            or self.name != self.name.strip()
        ):
            raise ValueError("Contact parts require explicit link names.")
        if type(self.pad) is not bool:
            raise TypeError("Contact pad membership must be explicit boolean data.")
        if (
            isinstance(self.rest_offset, (bool, np.bool_))
            or not math.isfinite(self.rest_offset)
            or self.rest_offset < 0
        ):
            raise ValueError("Contact parts require finite nonnegative rest offsets.")
        if self.contact_offset is not None and (
            isinstance(self.contact_offset, (bool, np.bool_))
            or not math.isfinite(self.contact_offset)
            or self.contact_offset < self.rest_offset
        ):
            raise ValueError(
                "Contact offsets must be finite and at least the rest offset."
            )
        meshes = tuple(self.meshes)
        if not meshes or any(not isinstance(mesh, trimesh.Trimesh) for mesh in meshes):
            raise TypeError("Contact parts require collision triangle meshes.")
        poses = np.array(self.tcp_poses, dtype=np.float64, copy=True)
        if (
            poses.ndim != 3
            or poses.shape[1:] != (4, 4)
            or not len(poses)
            or not np.isfinite(poses).all()
        ):
            raise ValueError("Contact part transforms must be finite (G, 4, 4) arrays.")
        if (
            not np.allclose(poses[:, 3], (0, 0, 0, 1))
            or not np.allclose(
                poses[:, :3, :3].transpose(0, 2, 1) @ poses[:, :3, :3],
                np.eye(3),
                atol=2e-5,
            )
            or not np.allclose(np.linalg.det(poses[:, :3, :3]), 1.0, atol=2e-5)
        ):
            raise ValueError("Contact part transforms must be proper rigid poses.")
        poses.setflags(write=False)
        object.__setattr__(self, "tcp_poses", poses)
        object.__setattr__(self, "meshes", meshes)


class NativeClosingQuery(_QueryLifetime):
    """Screen native nonpad envelopes before nominal bilateral pad contact.

    Only the pre-contact nominal sweep is screened. Continuing the command FK
    after pads touch would simulate penetration that real contact can prevent.
    Revision 4 records contact halo entry but rejects only the native rest order.
    Native acquisition/retention evidence remains independently mandatory.
    """

    def __init__(
        self,
        parts: Sequence[NativeContactPart],
        target_meshes: tuple[trimesh.Trimesh, ...],
        target_rest_offset: float,
        *,
        scope: NativeQueryScope,
        policy: ClosingContactPolicy,
        collision_factory: Callable[[], ConvexMeshQuery],
        collision_metadata: Mapping[str, object] | None = None,
        report_metadata: Mapping[str, object] | None = None,
        target_contact_offset: float | None = None,
        gravity_in_object: np.ndarray | None = None,
    ) -> None:
        """Prepare collision groups for one explicit captured closing stroke.

        Args:
            parts: Captured enabled collision links in source order, with two pads.
                The collection is frozen as a tuple during construction.
            target_meshes: Object-local captured collision meshes, borrowed read-only.
            target_rest_offset: Original target rest offset in metres.
            scope: Fresh preparation/observation identity.
            policy: Owned geometric rules selected by the Atomic Action or caller.
            collision_factory: Factory for a fresh, isolated convex query group.
            collision_metadata: Backend provenance for reports.
            report_metadata: Neutral source identifiers, never executable policy.
            target_contact_offset: Original target contact offset when contact halos are used.
            gravity_in_object: Nonzero object-frame gravity for horizontal pad-line qualification.

        Native provenance belongs to the capture builder. Supplying mesh data
        directly does not certify that it came from a native collision body.
        """
        self._start_scope(scope)
        parts = tuple(parts)
        if not isinstance(policy, ClosingContactPolicy):
            raise TypeError("policy must be ClosingContactPolicy.")
        policy = deepcopy(policy)
        policy.__post_init__()
        self._policy = policy
        self._collision_factory = collision_factory
        self._collision_metadata = _neutral_identity(collision_metadata or {})
        self._report_metadata = _neutral_identity(report_metadata or {})
        if not parts or sum(part.pad for part in parts) != 2 or not target_meshes:
            raise ValueError("Closing screen requires a complete hand and target")
        if len({part.name for part in parts}) != len(parts):
            raise ValueError("Native closing part names must be distinct.")
        counts = {len(part.tcp_poses) for part in parts}
        if len(counts) != 1 or min(counts) < 2:
            raise ValueError("Closing sample grids differ")
        offsets = [target_rest_offset, *(part.rest_offset for part in parts)]
        if any(
            isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or value < 0
            for value in offsets
        ):
            raise ValueError("Native rest offsets must be finite and nonnegative")
        if policy.contact_halo != "ignore":
            contacts = [target_contact_offset, *(part.contact_offset for part in parts)]
            if any(
                contact is None or not math.isfinite(contact) or contact < rest
                for contact, rest in zip(contacts, offsets, strict=True)
            ):
                raise ValueError("Native contact offsets must be available and >= rest")
        for part in parts:
            if (
                part.tcp_poses.shape[1:] != (4, 4)
                or not part.meshes
                or not np.isfinite(part.tcp_poses).all()
                or not np.allclose(part.tcp_poses[:, 3], (0, 0, 0, 1))
            ):
                raise ValueError("Closing link transforms are unavailable")
        self.parts = parts
        self.target_rest_offset = target_rest_offset
        self.target_contact_offset = target_contact_offset
        self.deferred_source_proposal_calls: int = 0
        self.rigid_scene_screen: NativeRigidRestScene | None = None
        self.object_pose_at_capture: torch.Tensor | None = None
        self.steps = min(counts)
        target_box = _mesh_box(target_meshes)
        self._target_bounds = np.stack((target_box.min(0), target_box.max(0)))
        boxes = []
        for part in parts:
            box = _mesh_box(part.meshes)
            boxes.append(
                box[None] @ part.tcp_poses[:, :3, :3].transpose(0, 2, 1)
                + part.tcp_poses[:, None, :3, 3]
            )
        self._part_tcp_boxes = np.stack(boxes, axis=1)
        if policy.horizontal_pad_line:
            gravity = np.asarray(gravity_in_object, dtype=np.float64)
            if (
                gravity.shape != (3,)
                or not np.isfinite(gravity).all()
                or np.linalg.norm(gravity) <= 0
            ):
                raise ValueError(
                    "Pad-line quality requires finite nonzero object-frame gravity"
                )
            self._quality_gravity = gravity / np.linalg.norm(gravity)
            pads = self._part_tcp_boxes[:, [part.pad for part in parts]]
            centers = (pads.min(2) + pads.max(2)) / 2
            axes = centers[:, 1] - centers[:, 0]
            lengths = np.linalg.norm(axes, axis=-1)
            if not np.isfinite(lengths).all() or (lengths <= 1e-8).any():
                raise ValueError("Native pad centers must define a closing line")
            self._quality_axes = axes / lengths[:, None]
            self._quality_gravity.setflags(write=False)
            self._quality_axes.setflags(write=False)
        elif gravity_in_object is not None:
            raise ValueError("Pad-line gravity requires horizontal_pad_line=True")
        self._distance_thresholds = np.array(
            [
                max(
                    part.rest_offset + target_rest_offset,
                    (
                        part.contact_offset + target_contact_offset
                        if policy.contact_halo != "ignore" and not part.pad
                        else 0.0
                    ),
                )
                for part in parts
            ]
        )
        self.target = self._collision_factory()
        for index, mesh in enumerate(target_meshes):
            self.target.add_object(str(index), mesh)
        self.managers = []
        for part in parts:
            manager = self._collision_factory()
            for index, mesh in enumerate(part.meshes):
                manager.add_object(str(index), mesh)
            self.managers.append(manager)
        self.cache: dict[bytes, dict[str, object]] = {}
        self.query_seconds = 0.0
        self.identity: dict[str, object] = {}
        self.build_seconds = 0.0
        self.opening_screen: NativeOpeningQuery | None = None

    def _separated_parts(self, pose: np.ndarray) -> np.ndarray:
        """Cull only parts whose enclosing boxes exceed every applicable offset."""
        boxes = self._part_tcp_boxes @ pose[:3, :3].T + pose[:3, 3]
        if not np.isfinite(boxes).all():
            raise ValueError("Native closing bounds are nonfinite")
        lower, upper = boxes.min(2), boxes.max(2)
        gap = np.maximum(
            0.0,
            np.maximum(self._target_bounds[0] - upper, lower - self._target_bounds[1]),
        )
        distance = np.linalg.norm(gap, axis=-1)
        if not np.isfinite(distance).all():
            raise ValueError("Native closing bound distance is unavailable")
        # Query the backend near a boundary; this reserve never relaxes its offsets.
        rounding = 1e-6 * np.maximum(1.0, np.abs(boxes).max(axis=(2, 3)))
        return distance > self._distance_thresholds[None] + rounding

    def inspect(self, tcp_in_object: np.ndarray) -> dict[str, object]:
        """Qualify one object-local TCP using the unchanged nominal sample order.

        Args:
            tcp_in_object: Proper TCP-to-object transform, shape ``(4, 4)``.

        Returns:
            Owned acceptance/reason record with the checked prefix and diagnostics.
            Cached repeats do not issue additional backend queries.
        """
        self._require_open()
        pose = _rigid_pose(tcp_in_object)
        key = pose.tobytes()
        if key in self.cache:
            return dict(self.cache[key])
        started = time.monotonic()
        separated = self._separated_parts(pose)
        outcome: dict[str, object] = {
            "valid": False,
            "reason": "bilateral_pad_rest_not_reached",
        }
        first_halo = None
        for step in range(self.steps):
            if self._policy.horizontal_pad_line:
                axis = pose[:3, :3] @ self._quality_axes[step]
                sine = float(abs(axis @ self._quality_gravity))
                if sine > math.sin(math.radians(2)) + 1e-9:
                    outcome = {
                        "valid": False,
                        "reason": "native_pad_line_tilt_exceeds_limit",
                        "pad_line_tilt_deg": math.degrees(math.asin(min(sine, 1.0))),
                        "pad_line_tilt_limit_deg": 2.0,
                    }
                    break
            pad_contact = 0
            rejected = None
            halo = None
            for part_index, (part, manager) in enumerate(
                zip(self.parts, self.managers, strict=True)
            ):
                if separated[step, part_index]:
                    continue
                for index in range(len(part.meshes)):
                    manager.set_transform(str(index), pose @ part.tcp_poses[step])
                rest = part.rest_offset + self.target_rest_offset
                band = (
                    rest
                    if part.contact_offset is None or self.target_contact_offset is None
                    else max(rest, part.contact_offset + self.target_contact_offset)
                )
                interval = manager.contact_query(self.target, band)
                distance = interval.lower
                if interval.below(rest):
                    if part.pad:
                        pad_contact += 1
                        if (
                            self._policy.reject_open_pad_contact
                            and step == 0
                            and rejected is None
                        ):
                            rejected = {
                                "valid": False,
                                "reason": "pad_rest_at_opening",
                                "link": part.name,
                                "clearance_lower_bound_m": distance,
                                "native_pair_rest_offset_m": rest,
                            }
                    elif rejected is None:
                        rejected = {
                            "valid": False,
                            "reason": "nonpad_rest_before_bilateral_pad",
                            "link": part.name,
                            "clearance_lower_bound_m": distance,
                            "native_pair_rest_offset_m": rest,
                        }
                if (
                    self._policy.contact_halo != "ignore"
                    and not part.pad
                    and halo is None
                ):
                    contact = part.contact_offset + self.target_contact_offset
                    if interval.below(contact):
                        halo = {
                            "valid": False,
                            "reason": "nonpad_contact_envelope_before_bilateral_pad_rest",
                            "link": part.name,
                            "clearance_lower_bound_m": distance,
                            "native_pair_contact_offset_m": contact,
                        }
            if (
                self._policy.contact_halo == "report"
                and halo is not None
                and first_halo is None
            ):
                first_halo = {
                    "first_nonpad_contact_halo_link": halo["link"],
                    "first_nonpad_contact_halo_clearance_lower_bound_m": halo[
                        "clearance_lower_bound_m"
                    ],
                    "first_nonpad_contact_halo_sample_index": step,
                    "first_nonpad_contact_halo_native_pair_contact_offset_m": halo[
                        "native_pair_contact_offset_m"
                    ],
                }
            if rejected is not None:
                if pad_contact == 2 and rejected["reason"] != "pad_rest_at_opening":
                    rejected["reason"] = "nonpad_and_pad_order_unresolved_at_sample"
                outcome = rejected
                break
            if pad_contact == 2:
                outcome = {"valid": True, "reason": "nominal_bilateral_pad_rest_first"}
                if self._policy.contact_halo != "ignore":
                    # Halo overlap alone does not establish actual body contact.
                    outcome["potential_nonpad_halo_at_pad_rest_endpoint"] = (
                        halo is not None
                    )
                break
            if halo is not None and self._policy.contact_halo != "report":
                outcome = halo
                break
        if first_halo is not None:
            outcome.update(first_halo)
        outcome["checked_samples"] = step + 1
        outcome["nominal_closing_fraction"] = step / (self.steps - 1)
        self.query_seconds += time.monotonic() - started
        self.cache[key] = dict(outcome)
        return dict(outcome)

    def mask(self, tcp_poses: torch.Tensor, object_poses: torch.Tensor) -> torch.Tensor:
        """Keep candidate axes intact while checking exact final TCP poses.

        Args:
            tcp_poses: World TCP transforms with shape ``(1, ..., 4, 4)``.
            object_poses: Matching object-to-world pose, shape ``(1, 4, 4)``.

        Returns:
            Boolean mask shaped ``tcp_poses.shape[:-2]`` on the input device,
            including an attached opening query when the policy enables it.
        """
        self._require_open()
        if (
            tcp_poses.ndim < 3
            or tcp_poses.shape[0] != 1
            or tcp_poses.shape[-2:] != (4, 4)
            or object_poses.shape != (1, 4, 4)
            or not torch.isfinite(tcp_poses).all()
            or not torch.isfinite(object_poses).all()
        ):
            raise ValueError("Closing screen requires finite B1 TCP/object poses")
        relative = np.linalg.inv(object_poses[0].detach().cpu().double().numpy())
        output = []
        for pose in tcp_poses.detach().cpu().double().reshape(-1, 4, 4).numpy():
            grasp = relative @ pose
            source = self.inspect(grasp)
            valid = source["valid"]
            if valid and self._policy.opening_check and self.opening_screen is not None:
                valid = self.opening_screen.inspect(grasp, source)["valid"]
            output.append(valid)
        return torch.tensor(output, dtype=torch.bool, device=tcp_poses.device).reshape(
            tcp_poses.shape[:-2]
        )

    def _depth_offsets(self) -> tuple[float, ...]:
        """Bound contact-depth alternatives by the actual pads, not object names."""
        if not self._policy.depth_alternatives:
            return (0.0,)
        spans = []
        rest = []
        for part in self.parts:
            if not part.pad:
                continue
            vertices = np.concatenate([np.asarray(m.vertices) for m in part.meshes])
            projected = vertices @ part.tcp_poses[:, 2, :3].T
            spans.append(float(np.ptp(projected, axis=0).min()))
            rest.append(part.rest_offset + self.target_rest_offset)
        usable = min(spans) - 2 * max(rest)
        if not math.isfinite(usable) or usable <= 0:
            return (0.0,)
        return tuple(float(value) for value in np.linspace(0, usable, 5))

    def report(self) -> dict[str, object]:
        """Return owned source-compatible counters and scoped geometry diagnostics.

        Returns:
            A neutral record retained after close. It never certifies physical
            acquisition, release, continuous collision freedom or scene execution.
        """
        if self.closed:
            return deepcopy(self._closed_report)
        result = {
            **self._collision_metadata,
            "query_scope": asdict(self.scope),
            **self._report_metadata,
            "identity": deepcopy(self.identity),
            "object_pose_at_capture": (
                None
                if self.object_pose_at_capture is None
                else self.object_pose_at_capture.detach().cpu().double().tolist()
            ),
            "build_seconds": self.build_seconds,
            "scope": (
                "discrete_nominal_hand_target_rest_order_not_physical_acquisition"
                if self._policy.contact_halo == "ignore"
                else "discrete_nominal_phase_scoped_contact_order_not_physical_acquisition"
            ),
            "native_pair_rest_offsets_used": True,
            "native_pair_contact_offsets_used": self._policy.contact_halo != "ignore",
            "closing_grid_samples": self.steps,
            "unique_tcp_queries": len(self.cache),
            "query_seconds": self.query_seconds,
            "accepted_queries": sum(item["valid"] for item in self.cache.values()),
            "reasons": [dict(item) for item in self.cache.values()],
        }
        if self._policy.contact_halo == "report":
            result["scope"] = (
                "discrete_nominal_rest_order_with_contact_halo_diagnostics"
                "_not_physical_acquisition"
            )
            result["nonpad_contact_halo_is_veto"] = False
        if self._policy.reject_open_pad_contact:
            result["native_opening_rest_screen"] = True
            result["whole_object_opening_envelope"] = False
        if self._policy.depth_alternatives:
            result["native_depth_offsets_m"] = list(self._depth_offsets())
        if self._policy.opening_check:
            result["opening"] = (
                None if self.opening_screen is None else self.opening_screen.report()
            )
        if self._policy.defer_source_proposals:
            result["proposal_source_proxy_deferred"] = (
                self.deferred_source_proposal_calls > 0
            )
            result["deferred_source_proposal_calls"] = (
                self.deferred_source_proposal_calls
            )
        if self._policy.horizontal_pad_line:
            result["pad_line_quality"] = {
                "tilt_limit_deg": 2.0,
                "gravity_in_object": self._quality_gravity.tolist(),
                "basis": "native_pad_collision_bounds_centers",
                "scope": "precontact_nominal_geometry_not_physical_grasp",
            }
        return result

    def depth_offsets(self) -> tuple[float, ...]:
        """Return the source-ordered depth alternatives for this open query.

        Returns:
            The original five pad-bounded offsets when enabled and usable;
            otherwise the unchanged zero-offset tuple.
        """
        self._require_open()
        return self._depth_offsets()

    def record_deferred_proposal(self) -> None:
        """Record one external proposal whose source qualification was deferred."""
        self._require_open()
        self.deferred_source_proposal_calls += 1

    def close(self) -> None:
        """Release owned collision resources while retaining the collected report."""
        if self.closed:
            return
        self._closed_report = self.report()
        self._closed = True
        for query in (self.opening_screen, self.rigid_scene_screen):
            if query is not None:
                query.close()
        self.opening_screen = self.rigid_scene_screen = None
        for manager in self.managers:
            manager.close()
        self.managers.clear()
        self.target.close()
        self.target = None
        self.parts = ()
        self._part_tcp_boxes = None
        self.cache.clear()
        self.object_pose_at_capture = None

    def __enter__(self) -> NativeClosingQuery:
        """Enter this query's owned lifetime.

        Returns:
            This open query.
        """
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release query resources without suppressing an exception.

        Args:
            exc_type: Exception type supplied by the context manager protocol.
            exc: Exception value, if any.
            traceback: Original traceback, if any.
        """
        self.close()


class NativeOpeningQuery(_QueryLifetime):
    """Check the qualified source-contact-to-open grid against one native support."""

    def __init__(
        self,
        closing: NativeClosingQuery,
        reference_meshes: tuple[trimesh.Trimesh, ...],
        reference_pose: np.ndarray,
        object_target_pose: np.ndarray,
        reference_rest_offset: float,
        reference_contact_offset: float,
        *,
        identity: dict[str, object],
        scope: NativeQueryScope,
    ) -> None:
        """Bind an already-grounded support snapshot to one exact closing scope.

        Args:
            closing: Open query owning the qualified closing stroke.
            reference_meshes: Captured support-local collision meshes.
            reference_pose: Support-to-world transform.
            object_target_pose: Final object-to-world target from the same grounding.
            reference_rest_offset: Original support rest offset.
            reference_contact_offset: Original support contact offset.
            identity: Owned neutral support/placement diagnostic metadata.
            scope: Must equal the closing query's preparation scope.
        """
        self._start_scope(scope)
        closing._require_open()
        if scope != closing.scope:
            raise ValueError(
                "Opening and closing require the same query/observation scope."
            )
        if not reference_meshes or not closing.parts:
            raise ValueError("Opening requires complete native hand/support meshes.")
        offsets = [(reference_rest_offset, reference_contact_offset)] + [
            (part.rest_offset, part.contact_offset) for part in closing.parts
        ]
        if any(
            contact is None
            or not math.isfinite(rest)
            or not math.isfinite(contact)
            or not 0 <= rest <= contact
            for rest, contact in offsets
        ):
            raise ValueError("Opening requires original native rest/contact offsets.")
        self.closing = closing
        self._collision_factory = closing._collision_factory
        self.object_target_pose = _rigid_pose(object_target_pose)
        self.reference_pose = _rigid_pose(reference_pose)
        self.identity = deepcopy(identity)
        self.reference = self._collision_factory()
        for index, mesh in enumerate(reference_meshes):
            self.reference.add_object(str(index), mesh, transform=self.reference_pose)
        self.reference_shape_count = len(reference_meshes)
        self.managers = []
        for part in closing.parts:
            manager = self._collision_factory()
            for index, mesh in enumerate(part.meshes):
                manager.add_object(str(index), mesh)
            self.managers.append(manager)
        self.rest_offset = reference_rest_offset
        self.contact_offset = reference_contact_offset
        self.cache: dict[bytes, dict[str, object]] = {}

    def inspect(
        self, grasp: np.ndarray, closing_record: dict[str, object]
    ) -> dict[str, object]:
        """Check the qualified contact-to-open prefix for one original candidate.

        Args:
            grasp: TCP-to-object transform, shape ``(4, 4)``.
            closing_record: Acceptance record from this bound closing query and pose.

        Returns:
            Owned opening clearance result and contact-halo diagnostics.
        """
        self._require_open()
        self.closing._require_open()
        if not closing_record.get("valid"):
            return {"valid": False, "reason": "source_closing_unqualified"}
        stop = closing_record.get("checked_samples", 0) - 1
        if (
            type(stop) is not int
            or not 0 <= stop < self.closing.steps
            or closing_record.get("reason") != "nominal_bilateral_pad_rest_first"
        ):
            raise ValueError("Opening requires this source's first bilateral sample.")
        grasp = _rigid_pose(grasp)
        if self.closing.cache.get(grasp.tobytes()) != closing_record:
            raise ValueError("Opening requires its bound query's closing result.")
        key = grasp.tobytes() + stop.to_bytes(8, "little")
        if key not in self.cache:
            terminal = self.object_target_pose @ grasp
            result: dict[str, object] = {
                "valid": True,
                "reason": "nominal_opening_rest_clear",
                "source_bilateral_grid_index": stop,
                "terminal_tcp": terminal.tolist(),
                "checked_queries": 0,
                "contact_halo_links": [],
                "clearance_lower_bound_m": None,
            }
            for step in range(stop, -1, -1):
                for part, manager in zip(
                    self.closing.parts, self.managers, strict=True
                ):
                    for index in range(len(part.meshes)):
                        manager.set_transform(
                            str(index), terminal @ part.tcp_poses[step]
                        )
                    rest = self.rest_offset + part.rest_offset
                    contact = self.contact_offset + part.contact_offset
                    interval = manager.contact_query(self.reference, max(rest, contact))
                    distance = interval.lower
                    result["checked_queries"] += 1
                    previous = result["clearance_lower_bound_m"]
                    result["clearance_lower_bound_m"] = (
                        distance if previous is None else min(previous, distance)
                    )
                    if (
                        interval.below(contact)
                        and part.name not in result["contact_halo_links"]
                    ):
                        result["contact_halo_links"].append(part.name)
                    if interval.below(rest):
                        result.update(
                            valid=False,
                            reason="hand_support_rest_envelope_overlap",
                            link=part.name,
                            grid_index=step,
                            native_pair_rest_offset_m=rest,
                            native_pair_contact_offset_m=contact,
                        )
                        self.cache[key] = result
                        return deepcopy(result)
            self.cache[key] = result
        return deepcopy(self.cache[key])

    def report(self) -> dict[str, object]:
        """Return owned per-candidate opening diagnostics, including after close.

        Returns:
            Source-compatible geometric records, not measured release evidence.
        """
        if self.closed:
            return deepcopy(self._closed_report)
        records = list(self.cache.values())
        return {
            **self.closing._collision_metadata,
            "query_scope": asdict(self.scope),
            "scope": "discrete_native_opening_prefix_not_physical_release",
            "physical_evidence": False,
            "valid": (
                None if not records else any(record["valid"] for record in records)
            ),
            "aggregation": "any_candidate_passed_not_physical_release",
            "identity": deepcopy(self.identity),
            "native_reference_shapes": self.reference_shape_count,
            "native_reference_rest_offset_m": self.rest_offset,
            "native_reference_contact_offset_m": self.contact_offset,
            "reference_pose": self.reference_pose.tolist(),
            "object_target_pose": self.object_target_pose.tolist(),
            "unique_queries": len(records),
            "accepted_queries": sum(record["valid"] for record in records),
            "rejected_queries": sum(not record["valid"] for record in records),
            "reasons": deepcopy(records),
        }

    def close(self) -> None:
        """Release only owned support/hand collision resources and retain diagnostics."""
        if self.closed:
            return
        self._closed_report = self.report()
        self._closed = True
        for manager in self.managers:
            manager.close()
        self.managers.clear()
        self.reference.close()
        self.reference = None
        self.closing = None
        self.cache.clear()


@dataclass(frozen=True, slots=True)
class NativeRigidObstacle:
    """Owned physical obstacle declaration, independent of collision backend APIs.

    Args:
        uid: Exact native rigid identity.
        rest_offset: Original nonnegative native rest offset.
        meshes: Ordered, already-scaled physical convex shape meshes.
        pose: Captured rigid world pose, never refreshed during a query.
    """

    uid: str
    rest_offset: float
    meshes: tuple[trimesh.Trimesh, ...]
    pose: np.ndarray

    def __post_init__(self) -> None:
        if type(self.uid) is not str or not self.uid or self.uid != self.uid.strip():
            raise ValueError("Native obstacle requires an exact identity.")
        if not math.isfinite(self.rest_offset) or self.rest_offset < 0:
            raise ValueError(
                "Native obstacle requires a finite nonnegative rest offset."
            )
        meshes = tuple(self.meshes)
        if not meshes or any(not isinstance(mesh, trimesh.Trimesh) for mesh in meshes):
            raise ValueError("Native obstacle requires physical convex shape meshes.")
        object.__setattr__(self, "meshes", tuple(mesh.copy() for mesh in meshes))
        object.__setattr__(self, "pose", _rigid_pose(self.pose))


class NativeRigidRestScene(_QueryLifetime):
    """Read the original cooked shapes and pair rest offsets, never contact halos."""

    def __init__(
        self,
        parts: Sequence[NativeContactPart],
        targets: Sequence[NativeRigidObstacle],
        *,
        scope: NativeQueryScope,
        collision_factory: Callable[[], ConvexMeshQuery],
    ) -> None:
        """Bind ordered captured hand parts and rigid query groups to one scope.

        Args:
            parts: Captured source hand stroke, frozen as a tuple on construction.
            targets: Ordered physical obstacle declarations. Backend query resources
                are created and owned here; callers never construct a backend manager.
                Closing releases query resources, never borrowed native entities.
            scope: Same preparation/observation scope as the owning closing query.
            collision_factory: Host-owned factory for isolated convex query groups.
        """
        self._start_scope(scope)
        self._collision_factory = collision_factory
        parts = tuple(parts)
        declarations = tuple(targets)
        if any(type(value) is not NativeRigidObstacle for value in declarations):
            raise TypeError("Native obstacles must use typed physical declarations.")
        targets = []
        for value in declarations:
            manager = self._collision_factory()
            for index, mesh in enumerate(value.meshes):
                manager.add_object(f"{value.uid}:{index}", mesh, transform=value.pose)
            targets.append((value.uid, value.rest_offset, manager))
        targets = tuple(targets)
        if not parts or any(
            not math.isfinite(value) or value < 0
            for value in [
                *(part.rest_offset for part in parts),
                *(rest for _, rest, _ in targets),
            ]
        ):
            raise ValueError(
                "Native scene requires complete hand geometry and finite rest offsets."
            )
        self.parts, self.targets = parts, targets
        self.managers = []
        for part in parts:
            manager = self._collision_factory()
            for index, mesh in enumerate(part.meshes):
                manager.add_object(str(index), mesh)
            self.managers.append(manager)
        self.cache: dict[tuple[bytes, int], dict[str, object]] = {}

    def inspect(self, tcp_world: np.ndarray, count: int) -> dict[str, object]:
        """Qualify exactly the source screen's nominal precontact grid prefix.

        Args:
            tcp_world: Proper TCP-to-world transform, shape ``(4, 4)``.
            count: Number of source prefix samples, bounded by the captured grid.

        Returns:
            Owned rest-clearance result using original pair offsets and source order.
        """
        self._require_open()
        pose = _rigid_pose(tcp_world)
        if (
            type(count) is not int
            or count < 1
            or any(count > len(part.tcp_poses) for part in self.parts)
        ):
            raise ValueError(
                "Native scene requires a finite TCP and its bounded precontact grid."
            )
        key = (pose.tobytes(), count)
        if key in self.cache:
            return dict(self.cache[key])
        minimum = math.inf
        result: dict[str, object] = {
            "valid": True,
            "sample_count": count,
            "scope": "native_rigid_pair_rest_precontact_not_physical_grip",
        }
        for step in range(count):
            for part, manager in zip(self.parts, self.managers, strict=True):
                for index in range(len(part.meshes)):
                    manager.set_transform(str(index), pose @ part.tcp_poses[step])
                for uid, rest, target in self.targets:
                    pair_rest = part.rest_offset + rest
                    interval = manager.contact_query(target, pair_rest)
                    distance = interval.lower
                    margin = distance - pair_rest
                    minimum = min(minimum, margin)
                    if interval.below(pair_rest):
                        result.update(
                            valid=False,
                            kind="native_rigid_pair_rest",
                            link=part.name,
                            scene_uid=uid,
                            sample_index=step,
                            mesh_clearance_lower_bound_m=distance,
                            native_pair_rest_offset_m=pair_rest,
                            minimum_rest_clearance_lower_bound_m=minimum,
                        )
                        self.cache[key] = result
                        return dict(result)
        result["minimum_rest_clearance_lower_bound_m"] = (
            minimum if math.isfinite(minimum) else None
        )
        self.cache[key] = result
        return dict(result)

    def close(self) -> None:
        """Release this snapshot's collision groups, never the native scene bodies."""
        if self.closed:
            return
        self._closed = True
        for manager in self.managers:
            manager.close()
        self.managers.clear()
        for _, _, target in self.targets:
            target.close()
        self.targets = self.parts = ()
