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

"""Independent Newton collision queries over declared physical convex shapes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import version
import math
from numbers import Integral

import numpy as np

from .._validation import _identifier, _rigid_pose
from ._values import count, digest, finite, names
from .contracts import FeasibilityCheck

__all__ = ["CollisionBody", "NewtonCollisionChecker", "UnsupportedFeasibilityError"]


class UnsupportedFeasibilityError(RuntimeError):
    """Required geometry, task adaptation or runtime capability is unavailable."""


def _query_float32(values: object, label: str) -> np.ndarray:
    """Reject collapsed or unsafe-scale values before entering native geometry."""
    original = np.asarray(values, dtype=np.float64)
    if not np.isfinite(original).all() or np.any(np.abs(original) > 1e6):
        raise UnsupportedFeasibilityError(
            f"{label} is outside the supported query precision range"
        )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        converted = original.astype(np.float32)
    if not np.isfinite(converted).all() or np.any((original != 0) & (converted == 0)):
        raise UnsupportedFeasibilityError(
            f"{label} cannot be represented by float32 query geometry"
        )
    return converted


@dataclass(frozen=True)
class CollisionBody:
    """One physical shape in an explicitly named link/entity frame.

    Args:
        collision_id: Stable physical ID for phase-specific contact permissions.
        frame_uid: Pose-provider frame identity; several shapes can share it.
        kind: box, sphere, capsule, cylinder, halfspace, convex, or mesh.
            Convex means the physical collider is the convex hull of its
            vertices. A generic triangle mesh is retained as mesh and cannot
            be implicitly converted into a convex collider.
        dimensions: Full box extents, radius, or radius/full cylinder height.
        local_pose: Shape-to-frame transform.
        vertices: Mesh-local positions; scale must already be applied.
        triangles: Mesh triangle indices.
        robot: Whether this shape belongs to a robot.
    """

    collision_id: str
    frame_uid: str
    kind: str
    dimensions: tuple[float, ...] = ()
    local_pose: tuple = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    vertices: tuple[tuple[float, float, float], ...] = ()
    triangles: tuple[tuple[int, int, int], ...] = ()
    robot: bool = False

    def __post_init__(self) -> None:
        _identifier(self.collision_id, "collision_id")
        _identifier(self.frame_uid, "frame_uid")
        object.__setattr__(self, "local_pose", _rigid_pose(self.local_pose))
        size = finite(self.dimensions, "shape dimensions")
        expected = {
            "box": 3,
            "sphere": 1,
            "capsule": 2,
            "cylinder": 2,
            "halfspace": 4,
            "convex": 0,
            "mesh": 0,
        }
        if self.kind not in expected:
            raise UnsupportedFeasibilityError(
                f"Unsupported physical shape {self.kind!r}"
            )
        if (
            len(size) != expected[self.kind]
            or (self.kind != "halfspace" and any(v <= 0 for v in size))
            or (
                self.kind == "halfspace"
                and not np.isclose(np.linalg.norm(size[:3]), 1.0)
            )
        ):
            raise ValueError("invalid physical shape dimensions")
        object.__setattr__(self, "dimensions", size)
        vertices = tuple(finite(v, "vertex") for v in self.vertices)
        raw_triangles = tuple(tuple(v) for v in self.triangles)
        if any(
            isinstance(i, bool) or not isinstance(i, Integral)
            for t in raw_triangles
            for i in t
        ):
            raise ValueError("triangle indices must be integers")
        triangles = tuple(tuple(int(i) for i in t) for t in raw_triangles)
        if self.kind in ("mesh", "convex") and (not vertices or not triangles):
            raise ValueError("mesh collision requires vertices and triangles")
        if any(len(v) != 3 for v in vertices) or any(
            len(t) != 3
            or any(type(i) is not int or not 0 <= i < len(vertices) for i in t)
            for t in triangles
        ):
            raise ValueError("invalid collision mesh")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "triangles", triangles)
        if type(self.robot) is not bool:
            raise TypeError("robot flag must be boolean")


class NewtonCollisionChecker:
    """Check an isolated B=1 convex collision state without advancing physics.

    Args:
        bodies: Physical shapes, including fixed scene geometry. Triangle mesh
            and SDF inputs require an explicit adapter and return unsupported.
        ignored_pairs: Permanent structural exemptions, as physical IDs or
            frame aliases. Adjacency is never inferred from link names.
        clearance_m: Required separation for non-exempt pairs, in metres.
        device: Owning Warp CPU/CUDA device; CPU is the default.
        detection_gap_m: Contact detection band, enlarged to cover clearance.
        numeric_tolerance_m: Conservative contact threshold and returned-distance
            allowance for float32 geometry.
        use_cuda_graph: Capture repeated CUDA queries after a warm call.
        max_shapes: Explicit shape budget, at most 512; bounds pair storage.
        max_contacts: Upper contact-buffer budget; saturation is unsupported.

    Returns from check contain a capped conservative clearance estimate, not
    a global minimum distance or a continuous collision certificate. Halfspaces
    use analytic physical-shape support. Only query-owned transforms and masks
    change; the live simulator's states, filters and clock are never touched.
    Calls on one instance must be serialized; geometry and settings are fixed
    at construction. Queries are translated to a local origin, and unsafe
    representation/precision budgets return unsupported.
    """

    def __init__(
        self,
        bodies: Sequence[CollisionBody],
        *,
        ignored_pairs: Sequence[tuple[str, str]] = (),
        clearance_m: float = 0.0,
        device: str = "cpu",
        detection_gap_m: float = 0.03,
        numeric_tolerance_m: float = 1e-6,
        use_cuda_graph: bool = True,
        max_shapes: int = 256,
        max_contacts: int = 1048576,
    ) -> None:
        self.bodies = tuple(bodies)
        count(max_shapes, "max_shapes")
        count(max_contacts, "max_contacts")
        if max_contacts > 1048576:
            raise ValueError("max_contacts must not exceed 1048576")
        if max_shapes > 512 or not 1 <= len(self.bodies) <= max_shapes:
            raise UnsupportedFeasibilityError(
                "Physical collision shape budget exceeded"
            )
        if any(not isinstance(b, CollisionBody) for b in self.bodies):
            raise TypeError("collision bodies must be CollisionBody values")
        if len({b.collision_id for b in self.bodies}) != len(self.bodies):
            raise ValueError("collision bodies must have unique IDs")
        if any(b.kind == "mesh" for b in self.bodies):
            raise UnsupportedFeasibilityError(
                "Triangle meshes require an explicit collision adapter; preserve native CONVEX shapes as convex"
            )
        if not isinstance(device, str) or not (
            device == "cpu" or device.startswith("cuda")
        ):
            raise ValueError("collision device must be cpu or CUDA")
        if type(use_cuda_graph) is not bool:
            raise TypeError("use_cuda_graph must be boolean")
        for label, value, positive in (
            ("clearance", clearance_m, False),
            ("detection gap", detection_gap_m, True),
            ("numeric tolerance", numeric_tolerance_m, True),
        ):
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or (value <= 0 if positive else value < 0)
            ):
                raise ValueError(
                    f"{label} must be finite and {'positive' if positive else 'nonnegative'}"
                )
        self.clearance_m = float(clearance_m)
        self.numeric_tolerance_m = float(numeric_tolerance_m)
        self.detection_gap_m = max(
            float(detection_gap_m),
            self.clearance_m + 4 * self.numeric_tolerance_m,
            self.clearance_m * 1.01,
        )
        _query_float32(
            (self.clearance_m, self.detection_gap_m, self.numeric_tolerance_m),
            "Collision precision settings",
        )
        self._geometry_radius = 0.0
        self._geometry_error = 0.0
        for body in self.bodies:
            if body.kind == "halfspace":
                continue
            if body.kind == "convex":
                original = np.asarray(body.vertices)
                converted = _query_float32(original, "Convex vertices")
                radius = float(np.linalg.norm(original, axis=1).max())
                error = float(np.linalg.norm(original - converted, axis=1).max())
            else:
                original = np.asarray(body.dimensions)
                converted = _query_float32(original, "Physical shape dimensions")
                if np.any(converted <= 0):
                    raise UnsupportedFeasibilityError(
                        "Physical shape dimensions collapsed in query precision"
                    )
                radius = float(np.linalg.norm(original))
                error = float(np.abs(original - converted).sum())
            self._geometry_radius = max(self._geometry_radius, radius)
            self._geometry_error = max(self._geometry_error, error)
        self._max_effective_tolerance = self.numeric_tolerance_m
        if (
            2 * self._geometry_error
            + 32 * np.finfo(np.float32).eps * self._geometry_radius
            >= self.detection_gap_m - self.clearance_m
        ):
            raise UnsupportedFeasibilityError(
                "Physical geometry exceeds the contact-band precision budget"
            )
        self.device = device
        self._use_cuda_graph = use_cuda_graph
        self._aliases = {}
        ids = {b.collision_id for b in self.bodies}
        for body in self.bodies:
            self._aliases.setdefault(body.frame_uid, set()).add(body.collision_id)
        if any(
            token in ids and targets != {token}
            for token, targets in self._aliases.items()
        ):
            raise ValueError(
                "Physical collision IDs and frame aliases must be unambiguous"
            )
        for collision_id in ids:
            self._aliases.setdefault(collision_id, {collision_id})
        self.ignored_pairs = frozenset(self._expand_pairs(ignored_pairs))
        self.last_pair = None
        self.last_clearance_interval: tuple[float, float] | None = None
        self._frame_uids = tuple(dict.fromkeys(b.frame_uid for b in self.bodies))
        self._native = tuple(
            i for i, body in enumerate(self.bodies) if body.kind != "halfspace"
        )
        self._native_lookup = {
            logical: index for index, logical in enumerate(self._native)
        }
        self._pairs = tuple(
            (i, j)
            for i in range(len(self.bodies))
            for j in range(i + 1, len(self.bodies))
            if self.bodies[i].frame_uid != self.bodies[j].frame_uid
            and frozenset((self.bodies[i].collision_id, self.bodies[j].collision_id))
            not in self.ignored_pairs
        )
        native_pairs = [
            (self._native_lookup[i], self._native_lookup[j])
            for i, j in self._pairs
            if i in self._native_lookup and j in self._native_lookup
        ]
        requested_capacity = max(64, 16 * len(native_pairs))
        if requested_capacity > 1048576:
            raise UnsupportedFeasibilityError("Physical collision pair budget exceeded")
        self._capacity = min(max_contacts, requested_capacity)
        self._mask_cache = OrderedDict()
        self._model = self._state = self._pipeline = self._contacts = None
        self._graph = self._active = self._output = self._empty_count = (
            self._explicit
        ) = None
        self._tolerance = None
        self._meshes = []
        self._closed = False
        self._version = version("newton")
        self.verification_id = digest(
            (
                "newton-convex-query/v1",
                "relative-origin/scale-allowance/v1",
                self._version,
                self.bodies,
                self.ignored_pairs,
                self.clearance_m,
                self.detection_gap_m,
                self.numeric_tolerance_m,
                max_shapes,
                self._capacity,
                device,
                use_cuda_graph,
            )
        )
        try:
            if native_pairs:
                self._build_query(native_pairs)
        except Exception:
            self.close()
            raise

    @property
    def metadata(self) -> dict[str, str | float | int | bool]:
        """Return owned backend, precision, distance-scope and query identity data."""
        return {
            "backend": "newton.CollisionPipeline",
            "version": self._version,
            "device": self.device,
            "verification_id": self.verification_id,
            "clearance_scope": "capped_contact_band",
            "detection_gap_m": self.detection_gap_m,
            "required_clearance_m": self.clearance_m,
            "numeric_tolerance_m": self.numeric_tolerance_m,
            "geometry_dtype": "float32",
            "halfspace_dtype": "float64",
            "halfspace_tolerance_m": 1e-8,
            "precision_policy": "relative-origin/scale-allowance/v1",
            "max_effective_tolerance_m": self._max_effective_tolerance,
            "max_abs_query_coordinate_m": 1e6,
            "shape_count": len(self.bodies),
            "contact_capacity": self._capacity,
            "cuda_graph": "captured" if self._graph is not None else "disabled",
            "halfspace_check": "analytic_physical_support",
            "continuous": False,
        }

    def _build_query(self, pairs: list[tuple[int, int]]) -> None:
        import newton
        import warp as wp
        from embodichain.compute.geometry._warp.contact_query import _contact_summary

        self._wp, self._kernel = wp, _contact_summary
        self.device = str(wp.get_device(self.device))
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(
            gap=self.detection_gap_m / 2,
            margin=0.0,
            density=0.0,
            collision_filter_parent=False,
        )
        for index in self._native:
            body = self.bodies[index]
            owner = builder.add_body(
                mass=1.0,
                inertia=wp.mat33(np.eye(3, dtype=np.float32)),
                is_kinematic=True,
            )
            options = {"body": owner, "cfg": cfg}
            if body.kind == "box":
                shape = builder.add_shape_box(
                    hx=body.dimensions[0] / 2,
                    hy=body.dimensions[1] / 2,
                    hz=body.dimensions[2] / 2,
                    **options,
                )
            elif body.kind == "sphere":
                shape = builder.add_shape_sphere(radius=body.dimensions[0], **options)
            elif body.kind in ("capsule", "cylinder"):
                shape = getattr(builder, "add_shape_" + body.kind)(
                    radius=body.dimensions[0],
                    half_height=body.dimensions[1] / 2,
                    **options,
                )
            else:
                mesh = newton.Mesh(
                    np.asarray(body.vertices, dtype=np.float32),
                    np.asarray(body.triangles, dtype=np.int32).reshape(-1),
                    compute_inertia=False,
                    maxhullvert=len(body.vertices),
                )
                self._meshes.append(mesh)
                shape = builder.add_shape_convex_hull(mesh=mesh, **options)
            if shape != owner:
                raise UnsupportedFeasibilityError(
                    "Collision query requires one physical shape per owned body"
                )
        self._model = builder.finalize(device=self.device)
        self._state = self._model.state()
        self._explicit = wp.array(
            np.asarray(pairs, dtype=np.int32), dtype=wp.vec2i, device=self.device
        )
        self._pipeline = newton.CollisionPipeline(
            self._model,
            broad_phase="explicit",
            shape_pairs_filtered=self._explicit,
            include_static_kinematic_pairs=True,
            rigid_contact_max=self._capacity,
            soft_contact_max=0,
        )
        if self._pipeline.narrow_phase.global_contact_reducer is not None:
            raise UnsupportedFeasibilityError(
                "Unsupported Newton narrow-phase route for declared convex shapes"
            )
        self._contacts = self._pipeline.contacts()
        self._active = wp.zeros(
            len(self._native) ** 2, dtype=wp.int32, device=self.device
        )
        self._output = wp.empty(4, dtype=float, device=self.device)
        self._empty_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._tolerance = wp.empty(1, dtype=float, device=self.device)

    def _expand_pairs(self, pairs: Sequence[tuple[str, str]]) -> set[frozenset[str]]:
        output = set()
        for pair in pairs:
            if (
                len(pair) != 2
                or pair[0] == pair[1]
                or any(p not in self._aliases for p in pair)
            ):
                raise ValueError(
                    "contact permissions require two distinct known physical IDs or frames"
                )
            output.update(
                frozenset((a, b))
                for a in self._aliases[pair[0]]
                for b in self._aliases[pair[1]]
                if a != b
            )
        return output

    def _mask(
        self,
        allowed_contacts: Sequence[tuple[str, str]],
        moving_objects: Sequence[str],
        all_pairs: bool,
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...], bool]:
        if type(all_pairs) is not bool:
            raise TypeError("all_pairs must be boolean")
        moving = names(moving_objects, "moving_objects")
        if any(uid not in self._frame_uids for uid in moving):
            raise UnsupportedFeasibilityError(
                "Moving object has no physical collision geometry"
            )
        allowed = frozenset(self._expand_pairs(allowed_contacts))
        key = (allowed, frozenset(moving), all_pairs)
        if key in self._mask_cache:
            self._mask_cache.move_to_end(key)
            return self._mask_cache[key]
        mask = np.zeros(len(self._native) ** 2, dtype=np.int32)
        planes = []
        for i, j in self._pairs:
            a, b = self.bodies[i], self.bodies[j]
            if frozenset((a.collision_id, b.collision_id)) in allowed or not (
                all_pairs
                or a.robot
                or b.robot
                or a.frame_uid in moving
                or b.frame_uid in moving
            ):
                continue
            if a.kind == "halfspace" or b.kind == "halfspace":
                planes.append((i, j))
            else:
                x, y = self._native_lookup[i], self._native_lookup[j]
                mask[x * len(self._native) + y] = mask[y * len(self._native) + x] = 1
        mask.setflags(write=False)
        value = mask, tuple(planes), bool(mask.any())
        self._mask_cache[key] = value
        if len(self._mask_cache) > 16:
            self._mask_cache.popitem(last=False)
        return value

    def _run_query(self) -> None:
        wp, phase, c = self._wp, self._pipeline.narrow_phase, self._contacts
        self._output.fill_(math.inf)
        self._pipeline.collide(self._state, self._contacts)
        counters = []
        for count_name, storage_name in (
            ("gjk_candidate_pairs_count", "gjk_candidate_pairs"),
            ("split_gjk_work_count", "split_gjk_work_items"),
            ("split_manifold_work_count", "split_manifold_work_items"),
        ):
            value, storage = getattr(phase, count_name), getattr(phase, storage_name)
            counters.extend(
                (
                    value if value is not None else self._empty_count,
                    storage.shape[0] if storage is not None else 0,
                )
            )
        wp.launch(
            self._kernel,
            self._capacity,
            inputs=[
                c.rigid_contact_count,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.rigid_contact_normal,
                c.rigid_contact_margin0,
                c.rigid_contact_margin1,
                self._model.shape_body,
                self._state.body_q,
                self._active,
                len(self._native),
                self.clearance_m,
                self._tolerance,
                self._capacity,
                self._pipeline.broad_phase_pair_count,
                len(self._explicit),
                *counters,
                self._output,
            ],
            device=self.device,
        )

    def _plane_distance(
        self, plane: CollisionBody, shape: CollisionBody, pp: np.ndarray, sp: np.ndarray
    ) -> float:
        if shape.kind == "halfspace":
            raise UnsupportedFeasibilityError(
                "Pairs of halfspaces require an explicit geometry adapter"
            )
        normal = pp[:3, :3] @ np.asarray(plane.dimensions[:3])
        local_normal = sp[:3, :3].T @ normal
        if shape.kind == "convex":
            support = float((np.asarray(shape.vertices) @ local_normal).min())
        elif shape.kind == "sphere":
            support = -shape.dimensions[0]
        elif shape.kind == "capsule":
            support = (
                -shape.dimensions[0] - abs(local_normal[2]) * shape.dimensions[1] / 2
            )
        elif shape.kind == "cylinder":
            support = (
                -shape.dimensions[0] * np.linalg.norm(local_normal[:2])
                - abs(local_normal[2]) * shape.dimensions[1] / 2
            )
        else:
            support = -float(np.abs(local_normal) @ (np.asarray(shape.dimensions) / 2))
        center_separation = (
            float(normal @ (sp[:3, 3] - pp[:3, 3])) - plane.dimensions[3]
        )
        return center_separation + float(support)

    def check(
        self,
        frame_poses: Mapping[str, object],
        *,
        allowed_contacts: Sequence[tuple[str, str]] = (),
        moving_objects: Sequence[str] = (),
        all_pairs: bool = False,
    ) -> tuple[FeasibilityCheck, float]:
        """Check synchronized frames and return a capped clearance estimate.

        Args:
            frame_poses: Arena transforms for every physical body frame.
            allowed_contacts: Current-stage explicit ID/frame pair exemptions.
            moving_objects: Payload/dynamic frames requiring environment checks.
            all_pairs: Also inspect static layout pairs during preparation.

        Returns:
            Discrete collision evidence and a nonnegative clearance estimate
            capped at detection_gap_m; unreported contacts never imply infinity.
            ``last_clearance_interval`` records the lower/upper estimates under
            the query's conversion-precision policy. Its upper endpoint remains
            infinite when no native contact was reported in the detection band.
            An interval spanning a consumer threshold cannot classify that
            threshold. Acquisition failures leave the interval unavailable.
        """
        if self._closed:
            raise RuntimeError("collision checker is closed")
        self.last_pair = None
        self.last_clearance_interval = None
        frames = {}
        for uid in self._frame_uids:
            if uid not in frame_poses:
                raise UnsupportedFeasibilityError(
                    f"Missing physical collision frame {uid}"
                )
            frames[uid] = np.asarray(_rigid_pose(frame_poses[uid]))
        origin = frames[self.bodies[0].frame_uid][:3, 3].copy()
        with np.errstate(over="ignore", invalid="ignore"):
            for frame in frames.values():
                frame[:3, 3] -= origin
            poses = [frames[b.frame_uid] @ b.local_pose for b in self.bodies]
        if not all(np.isfinite(pose).all() for pose in poses):
            raise UnsupportedFeasibilityError(
                "Nonfinite composed physical collision pose"
            )
        mask, planes, has_native = self._mask(
            allowed_contacts, moving_objects, all_pairs
        )
        clearance = math.inf
        upper_clearance = math.inf
        for i, j in planes:
            plane, shape = (i, j) if self.bodies[i].kind == "halfspace" else (j, i)
            with np.errstate(over="ignore", invalid="ignore"):
                distance = self._plane_distance(
                    self.bodies[plane], self.bodies[shape], poses[plane], poses[shape]
                )
            if not math.isfinite(distance):
                raise UnsupportedFeasibilityError(
                    "Nonfinite analytic halfspace separation"
                )
            clearance = min(clearance, max(0.0, distance))
            upper_clearance = min(upper_clearance, max(0.0, distance))
            if distance < self.clearance_m - 1e-8:
                self.last_clearance_interval = (0.0, upper_clearance)
                return self._failure(i, j), max(0.0, distance)
        if has_native:
            from scipy.spatial.transform import Rotation

            matrices = np.asarray([poses[i] for i in self._native])
            with np.errstate(over="ignore", invalid="ignore"):
                relative = matrices[:, :3, 3] - matrices[0, :3, 3]
            rotations = Rotation.from_matrix(matrices[:, :3, :3])
            packed = _query_float32(
                np.column_stack((relative, rotations.as_quat())),
                "Packed physical collision pose",
            )
            rotation_loss = (
                float(
                    np.linalg.norm(
                        rotations.as_matrix() - matrices[:, :3, :3], axis=(1, 2)
                    ).max()
                )
                * self._geometry_radius
            )
            scale = float(np.abs(relative).max()) + self._geometry_radius
            tolerance = (
                self.numeric_tolerance_m
                + 2 * (self._geometry_error + rotation_loss)
                + 32 * np.finfo(np.float32).eps * scale
            )
            if (
                not math.isfinite(tolerance)
                or tolerance >= self.detection_gap_m - self.clearance_m
            ):
                raise UnsupportedFeasibilityError(
                    "Candidate exceeds the contact-band precision budget"
                )
            self._max_effective_tolerance = max(
                self._max_effective_tolerance, tolerance
            )
            self._state.body_q.assign(packed)
            self._active.assign(mask)
            self._tolerance.assign(np.array([tolerance], dtype=np.float32))
            if self._graph is None:
                self._run_query()
                if self.device.startswith("cuda") and self._use_cuda_graph:
                    self._wp.synchronize_device(self.device)
                    with self._wp.ScopedCapture(device=self.device) as capture:
                        self._run_query()
                    self._graph = capture.graph
                    self._wp.capture_launch(self._graph)
            else:
                self._wp.capture_launch(self._graph)
            summary = self._output.numpy().copy()
            if math.isfinite(summary[3]):
                raise UnsupportedFeasibilityError(
                    "Newton collision query overflowed or produced invalid geometry evidence"
                )
            separation = float(summary[0])
            if math.isnan(separation) or separation == -math.inf:
                raise UnsupportedFeasibilityError(
                    "Newton returned a nonfinite separation"
                )
            clearance = min(
                clearance,
                max(
                    0.0,
                    min(self.detection_gap_m, separation) - tolerance,
                ),
            )
            upper_clearance = min(upper_clearance, max(0.0, separation + tolerance))
            if math.isfinite(summary[1]):
                self.last_clearance_interval = (clearance, upper_clearance)
                a, b = divmod(int(summary[1]), len(self._native))
                return self._failure(self._native[a], self._native[b]), clearance
        self.last_clearance_interval = (clearance, upper_clearance)
        return FeasibilityCheck(
            "collision",
            "passed",
            "Discrete Newton convex/physical-halfspace query; clearance is capped by the detection band",
        ), (0.0 if math.isinf(clearance) else min(self.detection_gap_m, clearance))

    def _failure(self, i: int, j: int) -> FeasibilityCheck:
        self.last_pair = self.bodies[i].collision_id, self.bodies[j].collision_id
        return FeasibilityCheck(
            "collision",
            "failed",
            f"{self.last_pair[0]} / {self.last_pair[1]}: forbidden collision or insufficient clearance",
        )

    def close(self) -> None:
        """Release owned query arrays/graphs; repeated cleanup is harmless."""
        self._graph = self._pipeline = self._contacts = self._state = self._model = None
        self._active = self._output = self._empty_count = None
        self._tolerance = None
        self._explicit = None
        self._meshes.clear()
        self._mask_cache.clear()
        self._closed = True
