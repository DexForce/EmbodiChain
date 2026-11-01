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

"""Benchmark-only Newton candidate geometry, independent of a live simulator.

This measures replicated batches of the same explicitly declared physical
convex shapes as the production B=1 checker. No physical world is stepped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

import newton
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from embodichain.lab.sim.scene.feasibility import CollisionBody
from embodichain.lab.sim.scene._validation import _rigid_pose


@wp.kernel
def _minimum_separation(
    count: wp.array(dtype=wp.int32),
    first: wp.array(dtype=wp.int32),
    second: wp.array(dtype=wp.int32),
    point0: wp.array(dtype=wp.vec3),
    point1: wp.array(dtype=wp.vec3),
    normal: wp.array(dtype=wp.vec3),
    margin0: wp.array(dtype=float),
    margin1: wp.array(dtype=float),
    shape_body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    active: wp.array2d(dtype=wp.int32),
    shapes_per_candidate: int,
    batch: int,
    output: wp.array(dtype=float),
):
    index = wp.tid()
    if index == 0:
        output[batch] = float(count[0])
    if index < count[0]:
        a, b = first[index], second[index]
        row = a // shapes_per_candidate
        local0, local1 = a % shapes_per_candidate, b % shapes_per_candidate
        if row < batch and active[row, local0 * shapes_per_candidate + local1] != 0:
            p0 = wp.vec3(point0[index][0], point0[index][1], point0[index][2])
            p1 = wp.vec3(point1[index][0], point1[index][1], point1[index][2])
            owner0, owner1 = shape_body[a], shape_body[b]
            if owner0 >= 0:
                p0 = wp.transform_point(body_q[owner0], p0)
            if owner1 >= 0:
                p1 = wp.transform_point(body_q[owner1], p1)
            distance = wp.dot(normal[index], p1 - p0) - margin0[index] - margin1[index]
            wp.atomic_min(output, row, distance)


class NewtonCollisionProbe:
    """Own isolated, bounded query states for an identically shaped batch.

    All replicas share immutable mesh resources. Explicit shape pairs contain
    only pairs within one candidate. Phase permissions are masks in the result
    reduction; they never globally disable collision geometry. Returned values
    are signed separations *inside the detection band*, with infinity meaning
    no reported active pair, not an exact global distance.
    """

    def __init__(
        self,
        bodies: Sequence[CollisionBody],
        ignored_pairs: Sequence[Sequence[str]],
        *,
        batch: int,
        device: str,
        detection_gap: float = 0.03,
    ) -> None:
        if not 1 <= batch <= 256 or not bodies:
            raise ValueError("Probe requires 1..256 candidates and nonempty geometry")
        self.bodies = tuple(bodies)
        self.batch, self.device, self.gap = batch, device, detection_gap
        self.n = len(bodies)
        self.aliases = {}
        for i, body in enumerate(bodies):
            self.aliases.setdefault(body.frame_uid, set()).add(i)
            self.aliases.setdefault(body.collision_id, set()).add(i)
        ignored = self._expand(ignored_pairs)
        self.pairs = [
            (i, j)
            for i in range(self.n)
            for j in range(i + 1, self.n)
            if bodies[i].frame_uid != bodies[j].frame_uid and (i, j) not in ignored
        ]
        if not self.pairs:
            raise ValueError("Probe requires at least one non-exempt pair")
        self.meshes = {}
        self.mesh_convex = {}
        self.convex_vertex_counts = {}
        builder = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(
            gap=detection_gap / 2,
            margin=0.0,
            density=0.0,
            collision_filter_parent=False,
        )
        for row in range(batch):
            builder.begin_world(label=f"candidate_{row}")
            for i, body in enumerate(bodies):
                owner = builder.add_body(
                    mass=1.0,
                    inertia=wp.mat33(np.eye(3).astype(np.float32)),
                    is_kinematic=True,
                )
                kw = {"body": owner, "cfg": cfg}
                if body.kind == "box":
                    shape = builder.add_shape_box(
                        hx=body.dimensions[0] / 2,
                        hy=body.dimensions[1] / 2,
                        hz=body.dimensions[2] / 2,
                        **kw,
                    )
                elif body.kind == "sphere":
                    shape = builder.add_shape_sphere(radius=body.dimensions[0], **kw)
                elif body.kind in ("capsule", "cylinder"):
                    factory = getattr(builder, "add_shape_" + body.kind)
                    shape = factory(
                        radius=body.dimensions[0],
                        half_height=body.dimensions[1] / 2,
                        **kw,
                    )
                elif body.kind == "halfspace":
                    normal = body.dimensions[:3]
                    shape = builder.add_shape_plane(
                        plane=(*normal, -body.dimensions[3]),
                        width=0.0,
                        length=0.0,
                        **kw,
                    )
                elif body.kind == "convex":
                    if i not in self.meshes:
                        self.mesh_convex[i] = True
                        self.meshes[i] = newton.Mesh(
                            np.asarray(body.vertices, dtype=np.float32),
                            np.asarray(body.triangles, dtype=np.int32).reshape(-1),
                            compute_inertia=False,
                            maxhullvert=len(body.vertices),
                        )
                        self.convex_vertex_counts[i] = len(body.vertices)
                    shape = builder.add_shape_convex_hull(mesh=self.meshes[i], **kw)
                else:
                    raise ValueError(f"Unsupported benchmark geometry {body.kind}")
                if shape != row * self.n + i or owner != shape:
                    raise AssertionError(
                        "Probe's shape/body ownership must remain one-to-one"
                    )
            builder.end_world()
        self.model = builder.finalize(device=device)
        self.state = self.model.state()
        self.untouched_state = self.model.state()
        pairs = np.array(
            [
                (i + row * self.n, j + row * self.n)
                for row in range(batch)
                for i, j in self.pairs
            ],
            dtype=np.int32,
        )
        self.explicit_pairs = wp.array(pairs, dtype=wp.vec2i, device=device)
        self.capacity = max(64, 16 * len(pairs))
        if self.capacity > 2**24:
            raise ValueError("Contact capacity exceeds the exact-count reduction limit")
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            shape_pairs_filtered=self.explicit_pairs,
            include_static_kinematic_pairs=True,
            rigid_contact_max=self.capacity,
            soft_contact_max=0,
        )
        self.contacts = self.pipeline.contacts()
        self.active = wp.empty((batch, self.n * self.n), dtype=wp.int32, device=device)
        self.output = wp.empty(batch + 1, dtype=float, device=device)
        self._initial = self.untouched_state.body_q.numpy().copy()

    def _expand(self, pairs: Sequence[Sequence[str]]) -> set[tuple[int, int]]:
        """Resolve explicit physical IDs and frame aliases into unique pairs."""
        result = set()
        for a, b in pairs:
            for i in self.aliases[a]:
                for j in self.aliases[b]:
                    if i != j:
                        result.add((min(i, j), max(i, j)))
        return result

    def pack(self, samples: Sequence[Mapping]) -> tuple[np.ndarray, np.ndarray]:
        """Compose local geometry transforms and phase masks on the host."""
        if len(samples) != self.batch:
            raise ValueError("Sample count must match the allocated query batch")
        poses = np.empty((self.batch, self.n, 4, 4))
        active = np.zeros((self.batch, self.n * self.n), dtype=np.int32)
        for row, sample in enumerate(samples):
            frames = {
                uid: np.asarray(_rigid_pose(sample["frames"][uid]))
                for uid in {body.frame_uid for body in self.bodies}
            }
            options = sample.get("options", {})
            allowed = self._expand(options.get("allowed_contacts", ()))
            moving = set(options.get("moving_objects", ()))
            for i, body in enumerate(self.bodies):
                poses[row, i] = frames[body.frame_uid] @ body.local_pose
            for i, j in self.pairs:
                if (i, j) not in allowed and (
                    options.get("all_pairs", False)
                    or self.bodies[i].robot
                    or self.bodies[j].robot
                    or self.bodies[i].frame_uid in moving
                    or self.bodies[j].frame_uid in moving
                ):
                    active[row, i * self.n + j] = active[row, j * self.n + i] = 1
        if not np.isfinite(poses).all():
            raise ValueError("Nonfinite candidate geometry")
        q = np.empty((self.batch * self.n, 7), dtype=np.float32)
        q[:, :3] = poses[:, :, :3, 3].reshape(-1, 3)
        q[:, 3:] = Rotation.from_matrix(poses[:, :, :3, :3].reshape(-1, 3, 3)).as_quat()
        return q, active

    def upload(self, packed: tuple[np.ndarray, np.ndarray]) -> None:
        """Update scratch transforms and masks, leaving the original state intact."""
        self.state.body_q.assign(packed[0])
        self.active.assign(packed[1])

    def run(self) -> None:
        """Generate contacts and reduce on the owning Warp device without readback."""
        self.output.fill_(math.inf)
        self.pipeline.collide(self.state, self.contacts)
        c = self.contacts
        wp.launch(
            _minimum_separation,
            self.capacity,
            inputs=[
                c.rigid_contact_count,
                c.rigid_contact_shape0,
                c.rigid_contact_shape1,
                c.rigid_contact_point0,
                c.rigid_contact_point1,
                c.rigid_contact_normal,
                c.rigid_contact_margin0,
                c.rigid_contact_margin1,
                self.model.shape_body,
                self.state.body_q,
                self.active,
                self.n,
                self.batch,
                self.output,
            ],
            device=self.device,
        )

    def download(self) -> np.ndarray:
        """Copy owned results and reject contact-buffer saturation explicitly."""
        values = self.output.numpy().copy()
        if values[-1] >= self.capacity:
            raise RuntimeError("Newton contact buffer saturated; result is unverified")
        return values[:-1]

    def check(self, samples: Sequence[Mapping]) -> np.ndarray:
        """Measure the host-input to host-result path used by current consumers."""
        self.upload(self.pack(samples))
        self.run()
        return self.download()

    def verify_isolation(self) -> None:
        """Assert queries did not mutate the separately owned original state."""
        if not np.array_equal(self._initial, self.untouched_state.body_q.numpy()):
            raise AssertionError("Candidate query mutated original coordinates")

    def verify_buffers(self) -> None:
        """Validate public narrow-phase counters outside identical-input timing."""
        phase = self.pipeline.narrow_phase
        for count_name, storage_name in (
            ("gjk_candidate_pairs_count", "gjk_candidate_pairs"),
            ("split_gjk_work_count", "split_gjk_work_items"),
            ("split_manifold_work_count", "split_manifold_work_items"),
            ("shape_pairs_mesh_count", "shape_pairs_mesh"),
            ("triangle_pairs_count", "triangle_pairs"),
            ("shape_pairs_mesh_plane_count", "shape_pairs_mesh_plane"),
            ("shape_pairs_mesh_mesh_count", "shape_pairs_mesh_mesh"),
            ("shape_pairs_sdf_sdf_count", "shape_pairs_sdf_sdf"),
        ):
            count, storage = getattr(phase, count_name, None), getattr(
                phase, storage_name, None
            )
            if count is not None and storage is not None:
                if int(count.numpy()[0]) > storage.shape[0]:
                    raise RuntimeError(
                        f"Newton {count_name} exceeded capacity; query is unverified"
                    )
        if int(self.pipeline.broad_phase_pair_count.numpy()[0]) > len(
            self.explicit_pairs
        ):
            raise RuntimeError("Newton broad-phase capacity exceeded")
        reducer = phase.global_contact_reducer
        if reducer is not None and int(reducer.ht_insert_failures.numpy()[0]):
            raise RuntimeError("Newton contact reduction dropped entries")

    def persistent_bytes(self) -> int:
        """Count owned Warp array storage; excludes module caches and mesh BVHs."""
        seen = set()
        total = 0
        for owner in (
            self.model,
            self.state,
            self.untouched_state,
            self.contacts,
            self,
        ):
            for value in vars(owner).values():
                if isinstance(value, wp.array) and value.ptr not in seen:
                    seen.add(value.ptr)
                    total += value.capacity
        return total
