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

"""Candidate-only discrete nominal closing geometry; never physical evidence."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import torch
import trimesh
from embodichain.utils.math import pose_inv

__all__: list[str] = []


@dataclass(frozen=True, eq=False)
class ContactPart:
    """One native physical link and its nominal TCP-relative closing FK grid."""

    name: str
    meshes: tuple[Any, ...]
    tcp_poses: np.ndarray
    rest_offset: float
    pad: bool


class ClosingContactScreen:
    """Reject nonpad rest-envelope entry before nominal bilateral pad contact.

    Only the pre-contact nominal sweep is screened. Continuing the command FK
    after pads touch would simulate penetration that real contact can prevent.
    Native acquisition/retention evidence remains independently mandatory.
    """

    def __init__(
        self,
        parts: tuple[ContactPart, ...],
        target_meshes: tuple[Any, ...],
        target_rest_offset: float,
    ) -> None:
        if not parts or sum(part.pad for part in parts) != 2 or not target_meshes:
            raise ValueError("Closing screen requires a complete hand and target")
        counts = {len(part.tcp_poses) for part in parts}
        if len(counts) != 1 or min(counts) < 2:
            raise ValueError("Closing sample grids differ")
        offsets = [target_rest_offset, *(part.rest_offset for part in parts)]
        if any(not math.isfinite(value) or value < 0 for value in offsets):
            raise ValueError("Native rest offsets must be finite and nonnegative")
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
        self.steps = min(counts)
        self.target = trimesh.collision.CollisionManager()
        for index, mesh in enumerate(target_meshes):
            self.target.add_object(str(index), mesh)
        self.managers = []
        for part in parts:
            manager = trimesh.collision.CollisionManager()
            for index, mesh in enumerate(part.meshes):
                manager.add_object(str(index), mesh)
            self.managers.append(manager)
        self.cache: dict[bytes, dict[str, Any]] = {}
        self.query_seconds = 0.0
        self.identity: dict[str, Any] = {}
        self.build_seconds = 0.0

    def inspect(self, tcp_in_object: np.ndarray) -> dict[str, Any]:
        pose = np.asarray(tcp_in_object, dtype=np.float64)
        if (
            pose.shape != (4, 4)
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], (0, 0, 0, 1))
        ):
            raise ValueError("Closing TCP pose is unavailable")
        key = pose.tobytes()
        if key in self.cache:
            return dict(self.cache[key])
        started = time.monotonic()
        outcome: dict[str, Any] = {
            "valid": False,
            "reason": "bilateral_pad_rest_not_reached",
        }
        for step in range(self.steps):
            pad_contact = 0
            rejected = None
            for part, manager in zip(self.parts, self.managers, strict=True):
                for index in range(len(part.meshes)):
                    manager.set_transform(str(index), pose @ part.tcp_poses[step])
                distance = float(manager.min_distance_other(self.target))
                if not math.isfinite(distance):
                    raise ValueError("Native mesh distance is unavailable")
                rest = part.rest_offset + self.target_rest_offset
                if distance <= rest:
                    if part.pad:
                        pad_contact += 1
                    elif rejected is None:
                        rejected = {
                            "valid": False,
                            "reason": "nonpad_rest_before_bilateral_pad",
                            "link": part.name,
                            "minimum_distance_m": distance,
                            "native_pair_rest_offset_m": rest,
                        }
            if rejected is not None:
                if pad_contact == 2:
                    rejected["reason"] = "nonpad_and_pad_order_unresolved_at_sample"
                outcome = rejected
                break
            if pad_contact == 2:
                outcome = {"valid": True, "reason": "nominal_bilateral_pad_rest_first"}
                break
        outcome["checked_samples"] = step + 1
        outcome["nominal_closing_fraction"] = step / (self.steps - 1)
        self.query_seconds += time.monotonic() - started
        self.cache[key] = dict(outcome)
        return dict(outcome)

    def mask(self, tcp_poses: torch.Tensor, object_poses: torch.Tensor) -> torch.Tensor:
        """Keep candidate axes intact while checking exact final TCP poses."""
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
        output = [
            self.inspect(relative @ pose)["valid"]
            for pose in tcp_poses.detach().cpu().double().reshape(-1, 4, 4).numpy()
        ]
        return torch.tensor(output, dtype=torch.bool, device=tcp_poses.device).reshape(
            tcp_poses.shape[:-2]
        )

    def report(self) -> dict[str, Any]:
        return {
            "revision": 1,
            "identity": dict(self.identity),
            "build_seconds": self.build_seconds,
            "scope": "discrete_nominal_hand_target_rest_order_not_physical_acquisition",
            "native_pair_rest_offsets_used": True,
            "closing_grid_samples": self.steps,
            "unique_tcp_queries": len(self.cache),
            "query_seconds": self.query_seconds,
            "accepted_queries": sum(item["valid"] for item in self.cache.values()),
            "reasons": [dict(item) for item in self.cache.values()],
        }


def _native_meshes(body: Any) -> tuple[Any, ...]:
    """Copy native cooked shapes in body-local coordinates, never render meshes."""
    import dexsim.engine

    if body is None or not body.is_collision_enable():
        return ()
    meshes = []
    for index in range(int(body.get_shape_count())):
        shape = body.get_shape_geometry(index)
        if isinstance(shape, dexsim.engine.ConvexMeshGeometry):
            scale = np.asarray(shape.scale, dtype=np.float64)
            if (
                scale.shape != (3,)
                or not np.isfinite(scale).all()
                or (scale <= 0).any()
            ):
                raise ValueError("Pick closing native shape scale is unavailable")
            vertices = np.asarray(shape.vertices, dtype=np.float64) * scale
            faces = np.asarray(shape.triangles, dtype=np.int64)
            if (
                vertices.ndim != 2
                or vertices.shape[1] != 3
                or not len(vertices)
                or faces.ndim != 2
                or faces.shape[1] != 3
                or not len(faces)
                or (faces < 0).any()
                or (faces >= len(vertices)).any()
            ):
                raise ValueError("Pick closing native mesh topology is unavailable")
            mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), process=False)
        elif isinstance(shape, dexsim.engine.BoxGeometry):
            mesh = trimesh.creation.box(2 * np.asarray(shape.half_extents))
        else:
            raise ValueError(
                "Pick closing requires readable convex or box collision shapes"
            )
        pose = np.asarray(shape.local_pose, dtype=np.float64)
        if (
            not np.isfinite(mesh.vertices).all()
            or pose.shape != (4, 4)
            or not np.isfinite(pose).all()
        ):
            raise ValueError("Pick closing native geometry is nonfinite or unavailable")
        mesh.apply_transform(pose)
        meshes.append(mesh)
    return tuple(meshes)


def build_closing_contact_screen(
    robot: Any, simulation: Any, scene_binding: Any, request: Any
) -> ClosingContactScreen | None:
    """Capture invocation-owned rigid-target geometry using existing Default APIs.

    Args:
        robot: Current embodiment with complete named articulation FK.
        simulation: Current B1 Default simulation, read without stepping.
        scene_binding: Authoritative semantic-to-native rigid identities.
        request: Resolved Pick with explicit motion and complete hand commands.

    Returns:
        A fresh candidate screen, or None outside the bound rigid-target scope.
    """
    from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget

    object_id = request.goal.semantics.entity_id
    targets = [b for b in scene_binding.rigid_objects if b.entity_id == object_id]
    if not targets:
        return None
    if (
        len(targets) != 1
        or simulation.num_envs != 1
        or not simulation.is_default_backend
    ):
        raise ValueError(
            "Native Pick closing requires an exact B1 Default rigid binding"
        )
    started = time.monotonic()
    arm_endpoint = request.binding.endpoint("primary", "motion")
    hand_endpoint = request.binding.endpoint("primary", "grasp")
    arm = arm_endpoint.require_target(JointPositionTarget)
    hand = hand_endpoint.require_target(JointPositionTarget)
    if arm_endpoint.task_state_key != hand_endpoint.task_state_key:
        raise ValueError("Pick closing motion and hand resources differ")
    scene = simulation.spawn_result
    handles = [scene.get(path) for path in scene.get_paths()]
    robots = [
        h
        for h in handles
        if h.get_name() == robot.uid and hasattr(h, "render_articulation")
    ]
    if len(robots) != 1:
        raise ValueError("Native Pick robot identity is unavailable or ambiguous")
    native_robot = robots[0]
    objects = [
        h
        for h in handles
        if h.get_name() == targets[0].simulation_uid
        and h.arena_name == native_robot.arena_name
        and callable(getattr(h, "native", None))
    ]
    if len(objects) != 1:
        raise ValueError("Native Pick rigid identity is unavailable or ambiguous")
    object_handle = objects[0]
    target_body = object_handle.native().get_physical_body()
    target_meshes = _native_meshes(target_body)
    if not target_meshes:
        raise ValueError("Pick target has no enabled physical collision shapes")

    names = tuple(robot.joint_names)
    hand_names = [names[index] for index in hand.joint_ids]
    joints = {
        j.attrib["name"]: j
        for j in ET.parse(robot.cfg.fpath).getroot().findall("joint")
    }
    descendants = {joints[hand_names[0]].find("parent").attrib["link"]}
    while True:
        children = {
            j.find("child").attrib["link"]
            for j in joints.values()
            if j.find("parent").attrib["link"] in descendants
        }
        if children <= descendants:
            break
        descendants |= children
    moving = {
        name
        for name, joint in joints.items()
        if joint.attrib["type"] != "fixed"
        and joint.find("child").attrib["link"] in descendants
    }
    if not moving <= set(hand_names):
        raise ValueError(
            "Pick closing requires every moving hand joint, including mimics"
        )
    links = sorted(descendants)
    if not set(links) <= set(native_robot.get_link_names()):
        raise ValueError("Native Pick hand link topology is incomplete")
    solver = robot.cfg.solver_cfg[arm.control_part]
    observed = robot.get_qpos()[0:1].clone()
    measured_fk = robot.compute_fk(
        qpos=observed, link_names=links, qpos_joint_names=names
    )
    world_root = robot.get_local_pose(to_matrix=True)[0].detach().cpu().numpy()
    for index, name in enumerate(links):
        predicted = world_root @ measured_fk[0, index].detach().cpu().numpy()
        if not np.allclose(
            predicted, native_robot.get_link_pose(name), atol=2e-4, rtol=0
        ):
            raise ValueError("Named actual-mimic hand FK does not match native frames")
    states = observed.repeat(65, 1)
    opened, closed = (
        hand_endpoint.joint_positions(
            command, num_envs=1, device=states.device, dtype=states.dtype
        )
        for command in ("open", "grasp")
    )
    states[:, list(hand.joint_ids)] = torch.lerp(
        opened, closed, torch.linspace(0, 1, 65, device=states.device)[:, None]
    )
    fk = robot.compute_fk(
        qpos=states, link_names=[*links, solver.end_link_name], qpos_joint_names=names
    )
    tools = fk[:, -1] @ torch.as_tensor(solver.tcp, dtype=fk.dtype, device=fk.device)
    relative = pose_inv(tools)[:, None] @ fk[:, :-1]
    parts = []
    for index, name in enumerate(links):
        body = native_robot.render_articulation.get_physical_body(name)
        meshes = _native_meshes(body)
        if not meshes:
            continue
        parts.append(
            ContactPart(
                name,
                meshes,
                relative[:, index].detach().cpu().double().numpy(),
                float(body.get_rest_offset()),
                name.endswith("finger_pad"),
            )
        )
    screen = ClosingContactScreen(
        tuple(parts), target_meshes, float(target_body.get_rest_offset())
    )
    screen.identity = {
        "object_id": object_id,
        "simulation_uid": targets[0].simulation_uid,
        "native_object_path": object_handle.path,
        "native_robot_path": native_robot.path,
        "motion_part": arm.control_part,
        "hand_part": hand.control_part,
        "native_hand_links": [p.name for p in parts],
        "native_hand_shapes": sum(len(p.meshes) for p in parts),
        "native_target_shapes": len(target_meshes),
    }
    screen.build_seconds = time.monotonic() - started
    return screen
