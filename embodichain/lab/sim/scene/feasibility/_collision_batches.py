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


"""Batch the pinned Newton convex query without touching the live World."""

from __future__ import annotations

import math

import numpy as np

__all__: list[str] = []


class _CollisionBatches:
    """Keep native shapes, precision, pair exemptions and halfspace supports.

    The query owns one numerical model containing independent replicas. It is
    never stepped and creates no simulation World. Explicit pairs prohibit
    cross-environment contacts; only the small per-row summaries are read back.
    """

    def __init__(self, checker, count, allowed, moving, all_pairs=False):
        import newton
        import warp as wp

        self.checker, self.count, self.wp = checker, count, wp
        self.graph = None
        self.queries = 0
        self.mask, self.planes, self.has_native = checker._mask(
            allowed, moving, all_pairs
        )
        self.locals = np.asarray([body.local_pose for body in checker.bodies])
        self.indices = [
            checker._frame_uids.index(body.frame_uid) for body in checker.bodies
        ]
        self.model = self.state = self.pipeline = self.contacts = None
        self.meshes = []
        if not self.has_native:
            return
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(
            gap=checker.detection_gap_m / 2,
            margin=0.0,
            density=0.0,
            collision_filter_parent=False,
        )
        for index in checker._native:
            body = checker.bodies[index]
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
                self.meshes.append(mesh)
                shape = builder.add_shape_convex_hull(mesh=mesh, **options)
            if shape != owner:
                raise RuntimeError("Collision query requires one shape per owned body")
        batch = newton.ModelBuilder()
        batch.replicate(builder, count)
        self.model = batch.finalize(device=checker.device)
        self.state = self.model.state()
        width = len(checker._native)
        pairs = [
            (checker._native_lookup[i], checker._native_lookup[j])
            for i, j in checker._pairs
            if i in checker._native_lookup and j in checker._native_lookup
        ]
        explicit = np.asarray(
            [
                (a + row * width, b + row * width)
                for row in range(count)
                for a, b in pairs
            ],
            dtype=np.int32,
        )
        self.explicit = wp.array(explicit, dtype=wp.vec2i, device=checker.device)
        self.capacity = checker._capacity * count
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=self.explicit,
            include_static_kinematic_pairs=True,
            rigid_contact_max=self.capacity,
            soft_contact_max=0,
        )
        if self.pipeline.narrow_phase.global_contact_reducer is not None:
            raise RuntimeError("Unsupported batched Newton narrow-phase route")
        self.contacts = self.pipeline.contacts()
        self.active = wp.array(self.mask, dtype=wp.int32, device=checker.device)
        self.tolerance = wp.empty(count, dtype=float, device=checker.device)
        self.output = wp.empty((count, 3), dtype=float, device=checker.device)
        self.empty_count = wp.zeros(1, dtype=wp.int32, device=checker.device)
        self.stream = (
            wp.Stream(checker.device) if checker.device.startswith("cuda") else None
        )
        if self.stream is not None:
            wp.synchronize_device(checker.device)

    def _run(self):
        from embodichain.compute.geometry._warp.batch_contact_query import (
            contact_summary,
        )

        checker, wp = self.checker, self.wp
        self.output.fill_(math.inf)
        self.pipeline.collide(self.state, self.contacts)
        phase, contacts = self.pipeline.narrow_phase, self.contacts
        counters = []
        for count_name, storage_name in (
            ("gjk_candidate_pairs_count", "gjk_candidate_pairs"),
            ("split_gjk_work_count", "split_gjk_work_items"),
            ("split_manifold_work_count", "split_manifold_work_items"),
        ):
            value, storage = getattr(phase, count_name), getattr(phase, storage_name)
            counters.extend(
                (
                    value if value is not None else self.empty_count,
                    storage.shape[0] if storage is not None else 0,
                )
            )
        wp.launch(
            contact_summary,
            self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                self.model.shape_body,
                self.state.body_q,
                self.active,
                len(checker._native),
                self.count,
                checker.clearance_m,
                self.tolerance,
                self.capacity,
                self.pipeline.broad_phase_pair_count,
                len(self.explicit),
                *counters,
                self.output,
            ],
            device=checker.device,
        )

    def check(self, frame_poses, rows):
        from scipy.spatial.transform import Rotation
        from embodichain.lab.sim.scene.feasibility import UnsupportedFeasibilityError
        from embodichain.lab.sim.scene.feasibility.contracts import FeasibilityCheck
        from embodichain.lab.sim.scene.feasibility.collision import _query_float32

        checker = self.checker
        if checker._closed:
            raise RuntimeError("collision checker is closed")
        frames = np.stack([frame_poses[uid] for uid in checker._frame_uids], axis=1)
        if frames.shape != (self.count, len(checker._frame_uids), 4, 4):
            raise ValueError("Collision frames must preserve every physical row")
        if frames.dtype.kind not in "fiu" or not np.isfinite(frames).all():
            raise ValueError("arena_pose must contain finite real numbers")
        frames = frames.astype(np.float64)
        rotation = frames[..., :3, :3]
        if (
            np.any(np.abs(frames[..., 3, :] - (0, 0, 0, 1)) > 1e-6)
            or np.any(
                np.abs(np.swapaxes(rotation, -1, -2) @ rotation - np.eye(3)) > 1e-6
            )
            or np.any(np.abs(np.linalg.det(rotation) - 1) > 1e-6)
        ):
            raise ValueError("arena_pose must be a proper rigid transform")
        frames[..., 3, :] = (0, 0, 0, 1)
        frames[..., :3, 3] -= frames[:, :1, :3, 3].copy()
        poses = frames[:, self.indices] @ self.locals
        if not np.isfinite(poses).all():
            raise UnsupportedFeasibilityError(
                "Nonfinite composed physical collision pose"
            )
        failures = {}
        for i, j in self.planes:
            plane, shape = (i, j) if checker.bodies[i].kind == "halfspace" else (j, i)
            a, b = checker.bodies[plane], checker.bodies[shape]
            normal = poses[:, plane, :3, :3] @ np.asarray(a.dimensions[:3])
            local = np.einsum("bji,bj->bi", poses[:, shape, :3, :3], normal)
            if b.kind == "convex":
                support = (local @ np.asarray(b.vertices).T).min(axis=-1)
            elif b.kind == "sphere":
                support = -b.dimensions[0]
            elif b.kind == "capsule":
                support = -b.dimensions[0] - np.abs(local[:, 2]) * b.dimensions[1] / 2
            elif b.kind == "cylinder":
                support = (
                    -b.dimensions[0] * np.linalg.norm(local[:, :2], axis=-1)
                    - np.abs(local[:, 2]) * b.dimensions[1] / 2
                )
            elif b.kind == "box":
                support = -np.abs(local) @ (np.asarray(b.dimensions) / 2)
            else:
                raise UnsupportedFeasibilityError("Unsupported analytic halfspace pair")
            distance = (normal * (poses[:, shape, :3, 3] - poses[:, plane, :3, 3])).sum(
                -1
            )
            distance += support - a.dimensions[3]
            if not np.isfinite(distance).all():
                raise UnsupportedFeasibilityError(
                    "Nonfinite analytic halfspace separation"
                )
            for row in rows:
                if distance[row] < checker.clearance_m - 1e-8 and row not in failures:
                    failures[row] = (i, j)
        if self.has_native:
            matrices = poses[:, checker._native]
            relative = matrices[..., :3, 3] - matrices[:, :1, :3, 3]
            rotations = Rotation.from_matrix(matrices[..., :3, :3].reshape(-1, 3, 3))
            quaternions = rotations.as_quat().reshape(self.count, -1, 4)
            packed = _query_float32(
                np.concatenate((relative, quaternions), axis=-1),
                "Packed physical collision pose",
            )
            recovered = rotations.as_matrix().reshape(matrices[..., :3, :3].shape)
            loss = np.linalg.norm(recovered - matrices[..., :3, :3], axis=(-2, -1)).max(
                -1
            )
            tolerance = (
                checker.numeric_tolerance_m
                + 2 * (checker._geometry_error + loss * checker._geometry_radius)
                + 32
                * np.finfo(np.float32).eps
                * (np.abs(relative).max(axis=(-2, -1)) + checker._geometry_radius)
            )
            if not np.isfinite(tolerance).all() or np.any(
                tolerance >= checker.detection_gap_m - checker.clearance_m
            ):
                raise UnsupportedFeasibilityError(
                    "Candidate exceeds the contact-band precision budget"
                )
            checker._max_effective_tolerance = max(
                checker._max_effective_tolerance, float(tolerance.max())
            )
            # Solvers may temporarily borrow external CUDA streams. Own this
            # query's stream through assignment, capture, launch and readback.
            # Inputs have already crossed a measured CPU boundary above.
            with self.wp.ScopedStream(self.stream, sync_enter=False):
                self.state.body_q.assign(packed.reshape(-1, 7))
                self.tolerance.assign(tolerance.astype(np.float32))
                if self.graph is None:
                    self._run()
                    if self.stream is not None and checker._use_cuda_graph:
                        self.wp.synchronize_stream(self.stream)
                        with self.wp.ScopedCapture(stream=self.stream) as capture:
                            self._run()
                        self.graph = capture.graph
                        self.wp.capture_launch(self.graph, stream=self.stream)
                else:
                    self.wp.capture_launch(self.graph, stream=self.stream)
                summary = self.output.numpy()
            for row in rows:
                if (
                    math.isfinite(summary[row, 2])
                    or math.isnan(summary[row, 0])
                    or summary[row, 0] == -math.inf
                ):
                    raise UnsupportedFeasibilityError(
                        "Batched Newton query overflowed or returned invalid evidence"
                    )
                if row not in failures and math.isfinite(summary[row, 1]):
                    a, b = divmod(int(summary[row, 1]), len(checker._native))
                    failures[row] = (checker._native[a], checker._native[b])
        self.queries += 1
        result = []
        for row in rows:
            if row in failures:
                i, j = failures[row]
                pair = checker.bodies[i].collision_id, checker.bodies[j].collision_id
                report = FeasibilityCheck(
                    "collision",
                    "failed",
                    f"{pair[0]} / {pair[1]}: forbidden collision or insufficient clearance",
                )
            else:
                pair = None
                report = FeasibilityCheck(
                    "collision",
                    "passed",
                    "Discrete Newton convex/physical-halfspace query; batched physical rows",
                )
            result.append((row, report, pair))
        return result

    def close(self):
        with self.wp.ScopedStream(getattr(self, "stream", None), sync_enter=False):
            self.graph = self.contacts = self.pipeline = self.state = self.model = None
            for name in ("explicit", "active", "tolerance", "output", "empty_count"):
                setattr(self, name, None)
            self.meshes.clear()
        self.stream = None
