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

"""Validated Newton indexing and fixed-slot geometry for NeRD models."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from embodichain.compute.dynamics import plane_contact_features, quat_rotate

__all__: list[str] = []

_PHYSICS_SIGNATURE_SCHEMA = "nerd-reference-physics-v2"

_JOINTS = {
    "cartpole": ("slider_to_cart", "cart_to_pole"),
    "ant": tuple(name for i in range(1, 5) for name in (f"hip_{i}", f"ankle_{i}")),
}


def _numpy(value: Any) -> np.ndarray:
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else value.numpy()
    )


def _torch(value: Any) -> torch.Tensor:
    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)


def _local_name(label: str) -> str:
    return label.rsplit("/", 1)[-1]


def _fingerprint(payload: dict[str, Any]) -> str:
    # Infinities are legitimate unbounded joint limits. JSON strings retain
    # their semantics without emitting nonstandard JSON numeric values.
    def encode(value):
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        return value

    return hashlib.sha256(
        json.dumps(encode(payload), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


class NeRDBinding:
    """Resolve homogeneous reference robots without assuming native row order.

    Construction reads model topology once. All per-step indexing is resident
    on the model device and supports gradients through gather/scatter operations.
    """

    def __init__(self, model: Any, robot: str) -> None:
        import newton

        self.model = model
        self.robot = robot.lower()
        if self.robot not in _JOINTS:
            raise ValueError(f"Unsupported NeRD robot {robot!r}.")
        if model.particle_count:
            raise ValueError("NeRD does not integrate particle state.")
        self.device = torch.device(wp.device_to_torch(model.device))
        self.num_envs = max(int(model.world_count), 1)
        starts = _numpy(model.articulation_start)
        worlds = _numpy(model.articulation_world)
        if model.articulation_count != self.num_envs:
            raise ValueError(
                "NeRD requires exactly one reference articulation per world."
            )
        joint_type = _numpy(model.joint_type)
        parent = _numpy(model.joint_parent)
        child = _numpy(model.joint_child)
        q_starts = _numpy(model.joint_q_start)
        qd_starts = _numpy(model.joint_qd_start)
        shape_body = _numpy(model.shape_body)
        shape_type = _numpy(model.shape_type)
        flags = _numpy(model.shape_flags)
        collision_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        shape_transform = _numpy(model.shape_transform)
        shape_scale = _numpy(model.shape_scale)
        shape_margin = _numpy(model.shape_margin)
        body_com = _numpy(model.body_com)
        planes = np.flatnonzero(
            (shape_body == -1) & (shape_type == int(newton.GeoType.PLANE))
        )
        contact_pair_count = int(model.shape_contact_pair_count)
        contact_pairs = (
            _numpy(model.shape_contact_pairs)[:contact_pair_count]
            if contact_pair_count
            else np.empty((0, 2), dtype=np.int32)
        )
        if self.robot == "ant" and len(planes) != 1:
            raise ValueError("NeRD Ant requires exactly one shared static plane.")
        joint_xp, joint_xc = _numpy(model.joint_X_p), _numpy(model.joint_X_c)
        identity = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        q_rows, qd_rows, root_bodies, ordered_bodies, world_ids = [], [], [], [], []
        slot_bodies, slot_points, slot_thickness = [], [], []
        signatures = []
        for world_id in range(self.num_envs):
            match = np.flatnonzero(worlds == (world_id if model.world_count else -1))
            if len(match) != 1:
                raise ValueError(
                    "NeRD worlds must contain one locally owned articulation."
                )
            art = int(match[0])
            joints = list(range(int(starts[art]), int(starts[art + 1])))
            by_name = {_local_name(model.joint_label[j]): j for j in joints}
            missing = set(_JOINTS[self.robot]) - by_name.keys()
            if missing:
                raise ValueError(
                    f"NeRD reference joint names are missing: {sorted(missing)}."
                )
            scalar_joints = [by_name[n] for n in _JOINTS[self.robot]]
            if any(
                q_starts[j + 1] - q_starts[j] != 1
                or qd_starts[j + 1] - qd_starts[j] != 1
                for j in scalar_joints
            ):
                raise ValueError(
                    "NeRD reference joints must have one coordinate and velocity."
                )
            roots = [j for j in joints if parent[j] == -1]
            if len(roots) != 1:
                raise ValueError("NeRD requires one world-root joint per articulation.")
            root = roots[0]
            if self.robot == "ant":
                if int(joint_type[root]) != int(newton.JointType.FREE):
                    raise ValueError("Ant requires a free root joint.")
                if not np.allclose(joint_xp[root], identity) or not np.allclose(
                    joint_xc[root], identity
                ):
                    raise ValueError("NeRD Ant requires identity root joint frames.")
                selected = [root, *scalar_joints]
            else:
                if int(joint_type[root]) != int(newton.JointType.FIXED):
                    raise ValueError("Cartpole requires a fixed root joint.")
                selected = scalar_joints
            q_ids = [
                i
                for j in selected
                for i in range(int(q_starts[j]), int(q_starts[j + 1]))
            ]
            qd_ids = [
                i
                for j in selected
                for i in range(int(qd_starts[j]), int(qd_starts[j + 1]))
            ]
            if sum(q_starts[j + 1] - q_starts[j] for j in joints) != len(q_ids):
                raise ValueError("Scene contains unsupported articulation coordinates.")
            bodies = [int(child[root]), *[int(child[j]) for j in scalar_joints]]
            if len(set(bodies)) != len(bodies):
                raise ValueError(
                    "Reference topology must use distinct root and joint-child bodies."
                )
            shapes = [
                int(s)
                for b in bodies
                for s in np.flatnonzero(
                    (shape_body == b) & ((flags & collision_flag) != 0)
                )
            ]
            points, point_bodies, thickness = [], [], []
            if self.robot == "ant":
                for s in shapes:
                    transform = torch.as_tensor(shape_transform[s])
                    radius = float(shape_scale[s, 0])
                    if int(shape_type[s]) == int(newton.GeoType.SPHERE):
                        centers = torch.zeros(1, 3)
                    elif int(shape_type[s]) == int(newton.GeoType.CAPSULE):
                        half_height = float(shape_scale[s, 1])
                        centers = torch.tensor(
                            [[0.0, 0.0, half_height], [0.0, 0.0, -half_height]]
                        )
                    else:
                        raise ValueError(
                            "Reference Ant contacts require sphere/capsule primitives."
                        )
                    local = transform[:3] + quat_rotate(transform[3:], centers)
                    points.extend(local.tolist())
                    point_bodies.extend([int(shape_body[s])] * len(centers))
                    thickness.extend([radius + float(shape_margin[s])] * len(centers))
                if len(points) != 25:
                    raise ValueError(
                        f"Reference Ant requires 25 fixed contact slots, got {len(points)}."
                    )
            q_rows.append(q_ids)
            qd_rows.append(qd_ids)
            root_bodies.append(int(child[root]))
            ordered_bodies.append(bodies)
            world_ids.append(world_id if model.world_count else -1)
            slot_bodies.append(point_bodies)
            slot_points.append(points)
            slot_thickness.append(thickness)
            local_bodies = {body: index for index, body in enumerate(bodies)}
            contract_joints = [root, *scalar_joints]
            if any(
                int(parent[joint]) not in local_bodies and int(parent[joint]) != -1
                for joint in contract_joints
            ):
                raise ValueError(
                    "Reference articulation contains unsupported joint-parent topology."
                )
            local_shapes = {shape: index for index, shape in enumerate(shapes)}
            effective_pairs = set()
            for first, second in contact_pairs:
                first, second = int(first), int(second)
                if first not in local_shapes and second not in local_shapes:
                    continue
                endpoints = []
                for shape in (first, second):
                    if shape in local_shapes:
                        endpoints.append(local_shapes[shape])
                    elif self.robot == "ant" and shape == int(planes[0]):
                        endpoints.append(-1)
                    else:
                        raise ValueError(
                            "NeRD reference scenes cannot contain cross-world or unmodeled contacts."
                        )
                effective_pairs.add(tuple(sorted(endpoints)))
            expected_pairs = (
                {(-1, index) for index in range(len(shapes))}
                if self.robot == "ant"
                else set()
            )
            if effective_pairs != expected_pairs:
                raise ValueError(
                    "NeRD reference collision policy must be plane-only for Ant and contact-free for Cartpole."
                )
            signature = {
                "schema": _PHYSICS_SIGNATURE_SCHEMA,
                "robot": self.robot,
                "joint_names": list(_JOINTS[self.robot]),
                "body_names": [_local_name(model.body_label[b]) for b in bodies],
                "joint_parent": [
                    -1 if int(parent[j]) == -1 else local_bodies[int(parent[j])]
                    for j in contract_joints
                ],
                "joint_child": [local_bodies[int(child[j])] for j in contract_joints],
                "root_frame_orientation": [
                    joint_xp[root, 3:].tolist(),
                    joint_xc[root, 3:].tolist(),
                ],
                "allowed_contact_pairs": sorted(effective_pairs),
            }
            # Contact-free Cartpole may be translated with arena placement under
            # uniform gravity. Fixed-root orientations still change its dynamics.
            for name in ("body_mass", "body_inertia", "body_com", "body_flags"):
                signature[name] = _numpy(getattr(model, name))[bodies].tolist()
            for name in (
                "joint_type",
                "joint_axis_dim",
                "joint_dof_dim",
                "joint_enabled",
            ):
                value = getattr(model, name, None)
                if value is not None:
                    signature[name] = _numpy(value)[contract_joints].tolist()
            for name in ("joint_X_p", "joint_X_c"):
                signature[name] = _numpy(getattr(model, name))[scalar_joints].tolist()
            for name in (
                "joint_axis",
                "joint_limit_lower",
                "joint_limit_upper",
                "joint_limit_ke",
                "joint_limit_kd",
                "joint_friction",
                "joint_armature",
                "joint_damping",
                "joint_target_ke",
                "joint_target_kd",
                "joint_target_mode",
                "joint_effort_limit",
                "joint_velocity_limit",
            ):
                value = getattr(model, name, None)
                if value is not None:
                    signature[name] = _numpy(value)[qd_ids].tolist()
            for name in (
                "shape_type",
                "shape_scale",
                "shape_transform",
                "shape_margin",
                "shape_gap",
                "shape_is_solid",
                "shape_material_mu",
                "shape_material_ke",
                "shape_material_kd",
                "shape_material_kf",
                "shape_material_ka",
                "shape_material_kh",
                "shape_material_restitution",
                "shape_material_mu_torsional",
                "shape_material_mu_rolling",
            ):
                value = getattr(model, name, None)
                if value is not None:
                    signature[name] = _numpy(value)[shapes].tolist()
            signatures.append(_fingerprint(signature))
        if len(set(signatures)) != 1:
            raise ValueError("NeRD batch contains differing robot physics.")
        self.physics_signature = signatures[0]
        self.joint_names = list(_JOINTS[self.robot])
        self.q_indices = torch.tensor(q_rows, device=self.device, dtype=torch.long)
        self.qd_indices = torch.tensor(qd_rows, device=self.device, dtype=torch.long)
        self.root_bodies = torch.tensor(
            root_bodies, device=self.device, dtype=torch.long
        )
        self.world_ids = torch.tensor(world_ids, device=self.device, dtype=torch.long)
        self.base_com = body_com[root_bodies[0]].tolist()
        if not np.allclose(body_com[root_bodies], self.base_com):
            raise ValueError("NeRD batch has differing base COM offsets.")
        dynamic_bodies = set(np.flatnonzero(_numpy(model.body_inv_mass) > 0))
        if not dynamic_bodies <= {b for row in ordered_bodies for b in row}:
            raise ValueError("NeRD cannot integrate additional dynamic bodies.")
        self.contact_bodies = torch.tensor(
            slot_bodies, device=self.device, dtype=torch.long
        )
        self.contact_points = torch.tensor(
            slot_points, device=self.device, dtype=torch.float32
        ).reshape(self.num_envs, -1, 3)
        self.contact_thickness = torch.tensor(
            slot_thickness, device=self.device, dtype=torch.float32
        )
        gravity = _torch(model.gravity)[self.world_ids].clone()
        self.gravity = gravity
        self.gravity_dir = gravity / gravity.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.plane_normal = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self.plane_offset = torch.zeros((), device=self.device)
        self.plane_margin = 0.0
        planes = np.flatnonzero(
            (shape_body == -1) & (shape_type == int(newton.GeoType.PLANE))
        )
        if self.robot == "ant":
            if len(planes) != 1:
                raise ValueError("NeRD Ant requires exactly one shared static plane.")
            plane = int(planes[0])
            transform = torch.tensor(shape_transform[plane], device=self.device)
            self.plane_normal = quat_rotate(transform[3:], self.plane_normal)
            self.plane_offset = (transform[:3] * self.plane_normal).sum()
            self.plane_margin = float(shape_margin[plane])
        other_static = np.flatnonzero(
            (shape_body == -1)
            & (shape_type != int(newton.GeoType.PLANE))
            & ((flags & collision_flag) != 0)
        )
        if len(other_static):
            raise ValueError(
                "NeRD reference scenes cannot contain other static colliders."
            )
        owned_bodies = {b for row in ordered_bodies for b in row}
        unowned_colliders = [
            s
            for s, body in enumerate(shape_body)
            if int(body) >= 0
            and int(body) not in owned_bodies
            and int(flags[s]) & collision_flag
        ]
        if unowned_colliders:
            raise ValueError(
                "NeRD reference scenes cannot contain unmodeled fixed-body colliders."
            )
        if self.robot == "cartpole" and any(
            int(flags[s]) & collision_flag for s in planes
        ):
            raise ValueError(
                "Contact-free Cartpole requires the ground plane to be disabled."
            )
        environment = {
            "schema": _PHYSICS_SIGNATURE_SCHEMA,
            "robot_physics": self.physics_signature,
        }
        for name in (
            "shape_type",
            "shape_transform",
            "shape_margin",
            "shape_gap",
            "shape_material_mu",
            "shape_material_ka",
            "shape_material_kh",
            "shape_material_restitution",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_kf",
            "shape_material_mu_torsional",
            "shape_material_mu_rolling",
        ):
            values = getattr(model, name, None)
            if values is not None:
                environment[name] = _numpy(values)[planes].tolist()
        self.physics_signature = _fingerprint(environment)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], robot: str) -> "NeRDBinding":
        """Bind an owned public Scene metadata snapshot without native access."""
        values = dict(metadata)
        values["device"] = wp.get_device(values["device"])
        return cls(SimpleNamespace(**values), robot)

    def read(
        self, physical: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            physical["joint_q"][self.q_indices],
            physical["joint_qd"][self.qd_indices],
        )

    def contacts(self, body_q: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.robot == "cartpole":
            empty = body_q.new_empty((self.num_envs, 0))
            return {
                name: empty
                for name in (
                    "contact_points_0",
                    "contact_points_1",
                    "contact_normals",
                    "contact_depths",
                    "contact_thicknesses_0",
                    "contact_thicknesses_1",
                )
            }
        poses = body_q[self.contact_bodies]
        world_points = poses[..., :3] + quat_rotate(poses[..., 3:], self.contact_points)
        result = plane_contact_features(
            world_points,
            self.contact_thickness,
            self.plane_normal,
            self.plane_offset,
            self.plane_margin,
        )
        result["contact_points_0"] = self.contact_points.flatten(1)
        return result

    def metadata(self, dt: float) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "dt": dt,
            "joint_names": self.joint_names,
            "physics_signature": self.physics_signature,
            "physics_signature_schema": _PHYSICS_SIGNATURE_SCHEMA,
            "gravity": self.gravity[0].tolist(),
            "codec": {
                "robot": self.robot,
                "base_com": self.base_com,
                "continuous_q_indices": [1] if self.robot == "cartpole" else [],
            },
            "contact_schema": "nerd-static-plane-v1",
        }
