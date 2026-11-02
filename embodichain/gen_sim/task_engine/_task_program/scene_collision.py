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

"""Discrete FCL qualification with context-owned poses and physical mesh inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from .articulation_collision import _descendants, check_robot_motion
from .staging_recovery import _pose

__all__: list[str] = []

_UNAVAILABLE = (
    ImportError,
    ValueError,
    RuntimeError,
    AttributeError,
    KeyError,
    TypeError,
    OSError,
    NotImplementedError,
)


@dataclass(frozen=True, slots=True)
class CollisionMesh:
    owner_id: str
    pose_entity_id: str
    name: str
    vertices: torch.Tensor
    faces: torch.Tensor


@dataclass(frozen=True, slots=True)
class SceneCollisionMeshes:
    components: tuple[CollisionMesh, ...] = ()
    unavailable: str | None = None


@dataclass(frozen=True, slots=True)
class PadContact:
    """An explicit route-owned target/pad exception, never a whole-hand exemption."""

    object_id: str
    pad_links: tuple[str, str]
    phase: Literal["close", "lift", "hold", "release"]
    first_frame: int
    stop_frame: int
    component_names: tuple[str, ...] = ()


def _component(owner: str, entity: str, name: str, mesh: Any) -> CollisionMesh:
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not len(vertices)
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or faces.min() < 0
        or faces.max() >= len(vertices)
    ):
        raise ValueError(f"scene_collision_topology_unavailable:{name}")
    return CollisionMesh(
        owner,
        entity,
        name,
        torch.tensor(vertices, dtype=torch.float64),
        torch.tensor(faces, dtype=torch.long),
    )


def capture_collision_meshes(simulation: Any, bindings: Any) -> SceneCollisionMeshes:
    """Capture complete collision inputs without querying any native body pose.

    Rigid inputs come from native physical shapes, not visual vertices. Articulated
    inputs are enabled source collision meshes in their owning link frames; these
    do not claim equivalence to the backend's convex cooking.
    """
    try:
        import trimesh

        if getattr(bindings, "rigidized_articulations", ()):
            raise ValueError("rigidized_articulation_collision_meshes_unavailable")
        components = []
        for binding in bindings.rigid_objects:
            native = simulation.get_rigid_object(binding.simulation_uid)
            shapes = native.get_collision_shapes()
            if not shapes:
                raise ValueError(
                    f"scene_collision_shapes_unavailable:{binding.entity_id}"
                )
            for index, shape in enumerate(shapes):
                if shape.vertices is not None and shape.triangles is not None:
                    mesh = trimesh.Trimesh(
                        vertices=shape.vertices.detach().cpu().numpy(),
                        faces=shape.triangles.detach().cpu().numpy(),
                        process=False,
                    )
                elif shape.half_extents is not None:
                    extents = shape.half_extents.detach().cpu().numpy()
                    if (
                        extents.shape != (3,)
                        or not np.isfinite(extents).all()
                        or (extents <= 0).any()
                    ):
                        raise ValueError("scene_collision_box_unavailable")
                    mesh = trimesh.creation.box(extents=2 * extents)
                else:
                    raise ValueError("scene_collision_shape_type_unavailable")
                mesh.apply_transform(_pose(shape.local_pose))
                components.append(
                    _component(
                        binding.entity_id,
                        binding.entity_id,
                        f"{binding.entity_id}:{index}",
                        mesh,
                    )
                )
        if getattr(bindings, "articulations", ()):
            from .drawer_geometry import collision_meshes
            from .articulation_binding import _stage
            from pxr import Tf, UsdGeom, UsdPhysics

            links: dict[tuple[str, str], list[str]] = {}
            for binding in bindings.links:
                links.setdefault(
                    (binding.articulation_id, binding.native_link_name), []
                ).append(binding.entity_id)
            for binding in bindings.articulations:
                native = simulation.get_articulation(binding.simulation_uid)
                try:
                    stage = _stage(native.cfg.fpath)
                    for prim in stage.Traverse():
                        if (
                            prim.HasAPI(UsdPhysics.CollisionAPI)
                            and UsdPhysics.CollisionAPI(prim)
                            .GetCollisionEnabledAttr()
                            .Get()
                            and not prim.IsA(UsdGeom.Mesh)
                        ):
                            raise ValueError(
                                f"scene_articulation_collision_type_unavailable:{prim.GetPath()}"
                            )
                    meshes = collision_meshes(
                        {"fpath": native.cfg.fpath, "body_scale": native.cfg.body_scale}
                    )
                except Tf.ErrorException as error:
                    raise ValueError(
                        f"scene_articulation_mesh_unavailable:{error}"
                    ) from error
                if not meshes:
                    raise ValueError(
                        f"scene_articulation_mesh_unavailable:{binding.entity_id}"
                    )
                for (link, name), mesh in meshes.items():
                    ids = links.get((binding.entity_id, link), ())
                    if len(ids) != 1:
                        raise ValueError(
                            f"scene_collision_link_binding_unavailable:{binding.entity_id}:{link}"
                        )
                    components.append(_component(binding.entity_id, ids[0], name, mesh))
        if not components:
            raise ValueError("scene_collision_inventory_unavailable")
        return SceneCollisionMeshes(tuple(components))
    except _UNAVAILABLE as error:
        return SceneCollisionMeshes(unavailable=str(error))


class _SnapshotRobot:
    """Adapt the existing robot proxy checker without a live pose/state read."""

    def __init__(self, robot: Any, context: Any, part: str) -> None:
        import yourdfpy

        if context.batch_size != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("scene_collision_single_environment_unavailable")
        self.cfg, self.joint_names = robot.cfg, tuple(robot.joint_names)
        self.qpos = context.robot.qpos.detach().clone()
        self.root = _pose(context.robot.root_pose).copy()
        if (
            self.qpos.shape != (1, len(self.joint_names))
            or not torch.isfinite(self.qpos).all()
        ):
            raise ValueError("scene_collision_robot_observation_unavailable")
        if not np.allclose(getattr(robot.cfg, "body_scale", (1, 1, 1)), (1, 1, 1)):
            raise ValueError("scene_collision_robot_scale_unavailable")
        self.solver = robot.get_solver(part)
        self.model = yourdfpy.URDF.load(
            self.cfg.fpath,
            load_meshes=False,
            build_scene_graph=True,
            build_collision_scene_graph=True,
            load_collision_meshes=True,
        )
        if not set(self.model.actuated_joint_names) <= set(self.joint_names):
            raise ValueError("scene_collision_joint_mapping_unavailable")
        self.configure(self.qpos[0])
        ids = tuple(robot.get_joint_ids(part))
        relative = self.solver.get_fk(self.qpos[:, list(ids)])
        root = self.model.get_transform(self.solver.root_link_name)
        predicted = self.model.get_transform(self.solver.end_link_name) @ _pose(
            self.solver.get_tcp()
        )
        if not np.allclose(root @ _pose(relative), predicted, atol=2e-4, rtol=0):
            raise ValueError("scene_collision_solver_fk_calibration_unavailable")
        children: dict[str, set[str]] = {}
        for joint in self.model.robot.joints:
            children.setdefault(joint.parent, set()).add(joint.child)
        self.hand_links = _descendants(children, self.solver.end_link_name)
        graph, links = self.model.collision_scene.graph, set(self.model.link_map)
        self.geometry = []
        for node in graph.nodes_geometry:
            owner, seen = node, set()
            while owner not in links:
                if owner in seen or owner not in graph.transforms.parents:
                    raise ValueError("scene_collision_robot_mesh_owner_unavailable")
                seen.add(owner)
                owner = graph.transforms.parents[owner]
            _, name = graph[node]
            self.geometry.append(
                (node, owner, self.model.collision_scene.geometry[name])
            )
        if not self.geometry:
            raise ValueError("scene_collision_robot_meshes_unavailable")

    def configure(self, qpos: torch.Tensor) -> None:
        self.model.update_cfg(
            {
                name: float(value)
                for name, value in zip(self.joint_names, qpos)
                if name in self.model.joint_map
            }
        )

    def get_qpos(self) -> torch.Tensor:
        return self.qpos.clone()

    def get_solver(self, _part: str) -> Any:
        return self.solver

    def get_local_pose(self, **_kwargs: Any) -> torch.Tensor:
        return torch.tensor(self.root)[None]

    def get_link_pose(self, link: str, **_kwargs: Any) -> torch.Tensor:
        self.configure(self.qpos[0])
        return torch.tensor(self.root @ self.model.get_transform(link))[None]


def _command_samples(context: Any, positions: torch.Tensor) -> None:
    if (
        not isinstance(positions, torch.Tensor)
        or positions.ndim != 3
        or positions.shape[0] != 1
        or not positions.shape[1]
        or positions.shape[-1] != context.robot.qpos.shape[-1]
        or not torch.isfinite(positions).all()
    ):
        raise ValueError("scene_collision_command_samples_unavailable")
    if context.scene.timestamp != context.robot.timestamp:
        raise ValueError("scene_collision_observation_timestamp_unavailable")


def snapshot_eef_poses(
    robot: Any, context: Any, positions: torch.Tensor, control_part: str
) -> torch.Tensor:
    """Compute source-URDF TCP poses from the context root, never native FK roots."""
    _command_samples(context, positions)
    snapshot = _SnapshotRobot(robot, context, control_part)
    tcp = _pose(snapshot.solver.get_tcp())
    poses = []
    for qpos in positions[0]:
        snapshot.configure(qpos)
        poses.append(
            snapshot.root
            @ snapshot.model.get_transform(snapshot.solver.end_link_name)
            @ tcp
        )
    return torch.as_tensor(
        np.stack(poses), device=positions.device, dtype=positions.dtype
    )[None]


def snapshot_pad_links(robot: Any, context: Any, control_part: str) -> tuple[str, str]:
    """Resolve exactly two physical pad owners within the selected hand subtree."""
    snapshot = _SnapshotRobot(robot, context, control_part)
    pads = sorted(
        {
            owner
            for _, owner, _ in snapshot.geometry
            if owner in snapshot.hand_links and owner.endswith("_finger_pad")
        }
    )
    if len(pads) != 2:
        raise ValueError("scene_collision_two_owned_pads_unavailable")
    return pads[0], pads[1]


def _inputs(
    robot: Any,
    context: Any,
    positions: torch.Tensor,
    part: str,
    meshes: SceneCollisionMeshes,
    contacts: tuple[PadContact, ...],
    phase_names: tuple[str, ...] | None,
) -> _SnapshotRobot:
    if meshes.unavailable or not meshes.components:
        raise ValueError(meshes.unavailable or "scene_collision_inventory_unavailable")
    _command_samples(context, positions)
    snapshot = _SnapshotRobot(robot, context, part)
    owners = {component.owner_id for component in meshes.components}
    if contacts and (phase_names is None or len(phase_names) != positions.shape[1]):
        raise ValueError("scene_collision_command_phases_unavailable")
    for contact in contacts:
        target_components = tuple(
            c for c in meshes.components if c.owner_id == contact.object_id
        )
        if (
            not isinstance(contact, PadContact)
            or contact.object_id not in owners
            or contact.phase not in {"close", "lift", "hold", "release"}
            or len(contact.pad_links) != 2
            or len(set(contact.pad_links)) != 2
            or not set(contact.pad_links) <= snapshot.hand_links
            or any(not name.endswith("_finger_pad") for name in contact.pad_links)
            or type(contact.first_frame) is not int
            or type(contact.stop_frame) is not int
            or not 0 <= contact.first_frame < contact.stop_frame <= positions.shape[1]
            or not set(contact.component_names) <= {c.name for c in target_components}
            or (
                any(c.pose_entity_id != c.owner_id for c in target_components)
                and not contact.component_names
            )
        ):
            raise ValueError("scene_collision_pad_contact_scope_unavailable")
    return snapshot


def _manager(snapshot: _SnapshotRobot) -> tuple[Any, dict[str, str]]:
    import trimesh

    manager = trimesh.collision.CollisionManager()
    owners = {}
    for node, owner, mesh in snapshot.geometry:
        owners[node] = owner
        pose, _ = snapshot.model.collision_scene.graph[node]
        manager.add_object(node, mesh, transform=snapshot.root @ pose)
    return manager, owners


def _update(snapshot: _SnapshotRobot, manager: Any, qpos: torch.Tensor) -> None:
    snapshot.configure(qpos)
    for node, _, _ in snapshot.geometry:
        pose, _ = snapshot.model.collision_scene.graph[node]
        manager.set_transform(node, snapshot.root @ pose)


def _allowed(
    contacts: tuple[PadContact, ...],
    owner: str,
    link: str,
    frame: int,
    phase_names: tuple[str, ...] | None,
    component_name: str,
) -> bool:
    return any(
        contact.object_id == owner
        and link in contact.pad_links
        and phase_names is not None
        and contact.phase == phase_names[frame]
        and (not contact.component_names or component_name in contact.component_names)
        and contact.first_frame <= frame < contact.stop_frame
        for contact in contacts
    )


def _validate_carried_object(context: Any, object_id: str) -> None:
    resources = [
        key
        for key, held in context.task.held_objects.items()
        if held is not None
        and held.semantics.entity_id == object_id
        and bool(context.task.held_object_mask(key).all())
    ]
    if (
        len(resources) != 1
        or not context.task.exclusive_held_object_mask(resources[0]).all()
    ):
        raise ValueError("scene_collision_exclusive_carried_state_unavailable")


def _mesh(component: CollisionMesh) -> Any:
    import trimesh

    return trimesh.Trimesh(
        vertices=component.vertices.detach().cpu().numpy(),
        faces=component.faces.detach().cpu().numpy(),
        process=False,
    )


def inspect_snapshot_motion(
    robot: Any,
    context: Any,
    positions: torch.Tensor,
    control_part: str,
    meshes: SceneCollisionMeshes,
    *,
    pad_contacts: tuple[PadContact, ...] = (),
    phase_names: tuple[str, ...] | None = None,
    carried_object_id: str | None = None,
    object_pose_samples: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Check full emitted robot proxies against a same-observation scene snapshot.

    Removing a carried object requires a separate payload/robot and payload/scene
    certificate. This function does not qualify contact force or velocity limits.
    """
    try:
        import trimesh

        snapshot = _inputs(
            robot, context, positions, control_part, meshes, pad_contacts, phase_names
        )
        schedules = dict(object_pose_samples or {})
        for owner, poses in schedules.items():
            components = [c for c in meshes.components if c.owner_id == owner]
            if (
                not components
                or any(c.pose_entity_id != owner for c in components)
                or owner == carried_object_id
                or not isinstance(poses, torch.Tensor)
                or poses.shape != (*positions.shape[:2], 4, 4)
                or not torch.isfinite(poses).all()
            ):
                raise ValueError("scene_collision_rigid_pose_schedule_unavailable")
            for pose in poses[0]:
                _pose(pose)
        if carried_object_id is not None and carried_object_id not in {
            c.owner_id for c in meshes.components
        }:
            raise ValueError("scene_collision_carried_object_unavailable")
        if carried_object_id is not None:
            _validate_carried_object(context, carried_object_id)
        commands = check_robot_motion(snapshot, positions[0], control_part)
        if not commands["valid"]:
            return {**commands, "scope": "discrete_snapshot_robot_scene_meshes"}
        active, robot_owners = _manager(snapshot)
        world = trimesh.collision.CollisionManager()
        scene_owners = {}
        for index, component in enumerate(meshes.components):
            if component.owner_id == carried_object_id:
                continue
            entity = context.scene.entities.get(component.pose_entity_id)
            if entity is None or getattr(entity, "confidence", 1) <= 0:
                raise ValueError(
                    f"scene_collision_pose_unavailable:{component.pose_entity_id}"
                )
            name = f"component_{index}"
            world.add_object(name, _mesh(component), transform=_pose(entity.pose))
            scene_owners[name] = component
        for frame, qpos in enumerate(positions[0]):
            _update(snapshot, active, qpos)
            for name, component in scene_owners.items():
                if component.owner_id in schedules:
                    world.set_transform(
                        name, _pose(schedules[component.owner_id][0, frame])
                    )
            _, pairs = active.in_collision_other(world, return_names=True)
            rejected = [
                [
                    robot_owners[link],
                    scene_owners[name].owner_id,
                    scene_owners[name].name,
                ]
                for link, name in pairs
                if not _allowed(
                    pad_contacts,
                    scene_owners[name].owner_id,
                    robot_owners[link],
                    frame,
                    phase_names,
                    scene_owners[name].name,
                )
            ]
            if rejected:
                return {
                    "valid": False,
                    "frame": frame,
                    "kind": "scene",
                    "pairs": sorted(rejected),
                }
        return {
            "valid": True,
            "frames": positions.shape[1],
            "scope": "discrete_snapshot_robot_scene_meshes",
            "carried_object_excluded": carried_object_id,
            "scheduled_object_ids": sorted(schedules),
            "conditional_prediction": bool(schedules),
            "not_certified": "continuous_collision_native_cooking_contact_force_or_velocity",
        }
    except _UNAVAILABLE as error:
        return {"valid": False, "unavailable": str(error)}


def inspect_payload_scene(
    context: Any,
    meshes: SceneCollisionMeshes,
    payload_id: str,
    object_poses: torch.Tensor,
    *,
    allowed_contacts: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Check a predicted rigid payload against every other physical scene component.

    Support permission belongs to the caller's measured floor/footprint contract.
    An allowance names one exact component and explicit frames, never its owner.
    """
    scope = "discrete_payload_snapshot_scene_meshes"
    try:
        import trimesh

        if meshes.unavailable or not meshes.components:
            raise ValueError(
                meshes.unavailable or "scene_collision_inventory_unavailable"
            )
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or context.scene.timestamp != context.robot.timestamp
            or not isinstance(object_poses, torch.Tensor)
            or object_poses.ndim != 4
            or object_poses.shape[0] != 1
            or not object_poses.shape[1]
            or object_poses.shape[-2:] != (4, 4)
            or not torch.isfinite(object_poses).all()
        ):
            raise ValueError("payload_scene_snapshot_or_pose_samples_unavailable")
        carried = [c for c in meshes.components if c.owner_id == payload_id]
        others = [c for c in meshes.components if c.owner_id != payload_id]
        if not carried or any(c.pose_entity_id != payload_id for c in carried):
            raise ValueError("payload_rigid_collision_meshes_unavailable")
        permissions = dict(allowed_contacts or {})
        for name, mask in permissions.items():
            if (
                sum(c.name == name for c in others) != 1
                or not isinstance(mask, torch.Tensor)
                or mask.dtype != torch.bool
                or mask.shape != object_poses.shape[:2]
            ):
                raise ValueError("payload_scene_component_contact_scope_unavailable")
        payload, world = (
            trimesh.collision.CollisionManager(),
            trimesh.collision.CollisionManager(),
        )
        for index, component in enumerate(carried):
            payload.add_object(
                str(index), _mesh(component), transform=_pose(object_poses[0, 0])
            )
        for index, component in enumerate(others):
            entity = context.scene.entities.get(component.pose_entity_id)
            if entity is None or getattr(entity, "confidence", 1) <= 0:
                raise ValueError(
                    f"scene_collision_pose_unavailable:{component.pose_entity_id}"
                )
            world.add_object(str(index), _mesh(component), transform=_pose(entity.pose))
        for frame, pose in enumerate(object_poses[0]):
            for index in range(len(carried)):
                payload.set_transform(str(index), _pose(pose))
            _, pairs = payload.in_collision_other(world, return_names=True)
            rejected = [
                [payload_id, others[int(other)].owner_id, others[int(other)].name]
                for _, other in pairs
                if others[int(other)].name not in permissions
                or not bool(permissions[others[int(other)].name][0, frame])
            ]
            if rejected:
                return {
                    "valid": False,
                    "scope": scope,
                    "frame": frame,
                    "kind": "payload_scene",
                    "pairs": sorted(rejected),
                }
        return {
            "valid": True,
            "scope": scope,
            "frames": object_poses.shape[1],
            "conditional_prediction": True,
            "allowed_components": sorted(permissions),
            "not_certified": "continuous_collision_native_articulation_cooking_support_or_contact_force",
        }
    except _UNAVAILABLE as error:
        return {"valid": False, "scope": scope, "unavailable": str(error)}


def inspect_payload_robot(
    robot: Any,
    context: Any,
    positions: torch.Tensor,
    control_part: str,
    meshes: SceneCollisionMeshes,
    payload_id: str,
    object_poses: torch.Tensor,
    *,
    pad_contacts: tuple[PadContact, ...] = (),
    phase_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Check carried physical meshes against every robot collider, including the other arm."""
    try:
        import trimesh

        snapshot = _inputs(
            robot, context, positions, control_part, meshes, pad_contacts, phase_names
        )
        if (
            object_poses.shape != (*positions.shape[:2], 4, 4)
            or not torch.isfinite(object_poses).all()
        ):
            raise ValueError("payload_collision_pose_samples_unavailable")
        components = [c for c in meshes.components if c.owner_id == payload_id]
        if not components or any(c.pose_entity_id != payload_id for c in components):
            raise ValueError("payload_rigid_collision_meshes_unavailable")
        manager, owners = _manager(snapshot)
        payload = trimesh.collision.CollisionManager()
        for index, component in enumerate(components):
            payload.add_object(
                str(index), _mesh(component), transform=_pose(object_poses[0, 0])
            )
        for frame, qpos in enumerate(positions[0]):
            _update(snapshot, manager, qpos)
            for index in range(len(components)):
                payload.set_transform(str(index), _pose(object_poses[0, frame]))
            _, pairs = manager.in_collision_other(payload, return_names=True)
            rejected = [
                [owners[link], payload_id, components[int(name)].name]
                for link, name in pairs
                if not _allowed(
                    pad_contacts,
                    payload_id,
                    owners[link],
                    frame,
                    phase_names,
                    components[int(name)].name,
                )
            ]
            if rejected:
                return {
                    "valid": False,
                    "frame": frame,
                    "kind": "payload_robot",
                    "pairs": sorted(rejected),
                }
        return {
            "valid": True,
            "frames": positions.shape[1],
            "scope": "discrete_payload_all_robot_meshes",
            "not_certified": "continuous_collision_native_cooking_or_contact_force",
        }
    except _UNAVAILABLE as error:
        return {"valid": False, "unavailable": str(error)}
