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

"""Exact Default native capture for invocation-owned hand geometry queries."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import trimesh

from embodichain.lab.sim.utility.grasp_geometry import (
    RobotHandGeometryCfg,
    _resolve_hand_frame,
)
from embodichain.utils.math import pose_inv
from ._collision_backends import _require_fcl
from .native_hand import (
    NativeContactPart,
    NativeClosingQuery,
    NativeOpeningQuery,
    NativeOpeningTarget,
    NativeQueryScope,
    NativeRigidRestScene,
    _neutral_identity,
    _rigid_pose,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__: list[str] = []


def _b1_pose(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (1, 4, 4)
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} requires a floating B1 pose.")
    _rigid_pose(value[0].detach().cpu().numpy())
    return value.detach().clone()


def _command(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[0] != 1
        or not value.shape[1]
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} requires a finite floating (1, D) command.")
    return value.detach()


def _native_meshes(body: Any) -> tuple[trimesh.Trimesh, ...]:
    """Copy original cooked convex/box shapes once, in body-local coordinates."""
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
                raise ValueError("Native shape scale is unavailable.")
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
                raise ValueError("Native mesh topology is unavailable.")
            mesh = trimesh.Trimesh(vertices.copy(), faces.copy(), process=False)
        elif isinstance(shape, dexsim.engine.BoxGeometry):
            half = np.asarray(shape.half_extents)
            if half.shape != (3,) or not np.isfinite(half).all() or (half <= 0).any():
                raise ValueError("Native box half extents must be finite and positive.")
            mesh = trimesh.creation.box(2 * half)
        else:
            raise ValueError(
                "Native hand queries require readable cooked convex or box shapes."
            )
        pose = _rigid_pose(shape.local_pose)
        if not np.isfinite(mesh.vertices).all():
            raise ValueError("Native collision geometry is nonfinite.")
        mesh.apply_transform(pose)
        meshes.append(mesh)
    return tuple(meshes)


def _require_simulation(simulation: SimulationManager, robot: Robot) -> None:
    if getattr(simulation, "num_envs", None) != 1 or not getattr(
        simulation, "is_default_backend", False
    ):
        raise ValueError("Native hand queries require B1 Default simulation.")
    getter = getattr(simulation, "get_robot", None)
    uid = getattr(robot, "uid", None)
    if not callable(getter) or type(uid) is not str or getter(uid) is not robot:
        raise ValueError(
            "Native hand queries require the exact simulation-owned Robot."
        )


def _source_handle(handles: list[Any], uid: str, arena: str) -> Any:
    matches = [
        h
        for h in handles
        if h.get_name() == uid
        and h.arena_name == arena
        and callable(getattr(h, "native", None))
    ]
    if len(matches) != 1:
        raise ValueError(f"Native rigid identity is unavailable or ambiguous: {uid!r}.")
    return matches[0]


def _physical_identity(
    extra: Mapping[str, object] | None, physical: dict[str, object]
) -> dict[str, object]:
    result = _neutral_identity({} if extra is None else extra)
    if set(result) & set(physical):
        raise ValueError("Caller metadata cannot override captured native identity.")
    return {**result, **physical}


def _build_native_closing_query(
    robot: Robot,
    simulation: SimulationManager,
    *,
    source_uid: str,
    hand_geometry_cfg: RobotHandGeometryCfg,
    open_qpos: torch.Tensor,
    grasp_qpos: torch.Tensor,
    observation_qpos: torch.Tensor,
    object_pose: torch.Tensor,
    scope: NativeQueryScope,
    revision: int,
    gravity_in_object: np.ndarray | None,
    opening: NativeOpeningTarget | None,
    identity: Mapping[str, object] | None,
) -> NativeClosingQuery:
    _require_fcl()
    if not isinstance(scope, NativeQueryScope):
        raise TypeError("scope must be NativeQueryScope.")
    if type(revision) is not int or revision not in range(1, 10):
        raise ValueError("Native closing revision must be between 1 and 9.")
    if opening is not None and (
        not isinstance(opening, NativeOpeningTarget)
        or opening.scope != scope
        or revision < 3
    ):
        raise ValueError(
            "Opening needs revision3-9 and the exact closing observation scope."
        )
    if (
        type(source_uid) is not str
        or not source_uid
        or source_uid != source_uid.strip()
    ):
        raise ValueError("Native closing requires an exact source UID.")
    _require_simulation(simulation, robot)
    if source_uid not in simulation.get_rigid_object_uid_list():
        raise ValueError(
            "Native closing requires an explicitly registered rigid source."
        )
    if not isinstance(hand_geometry_cfg, RobotHandGeometryCfg):
        raise TypeError("hand_geometry_cfg must be RobotHandGeometryCfg.")
    opened, closed = _command(open_qpos, "open_qpos"), _command(
        grasp_qpos, "grasp_qpos"
    )
    cfg = deepcopy(hand_geometry_cfg)
    cfg.open_qpos, cfg.closed_qpos = tuple(opened[0].cpu().tolist()), tuple(
        closed[0].cpu().tolist()
    )
    cfg.validate_values()
    _, selected_links, hand_names, solver = _resolve_hand_frame(robot, cfg)
    links = sorted(selected_links)  # Preserve the source native-query link order.
    names = tuple(robot.joint_names)
    if (
        not isinstance(observation_qpos, torch.Tensor)
        or observation_qpos.shape != (1, len(names))
        or not observation_qpos.is_floating_point()
        or not torch.isfinite(observation_qpos).all()
    ):
        raise ValueError(
            "Native closing requires the full finite B1 named robot state."
        )
    if len(set(names)) != len(names) or not set(hand_names) <= set(names):
        raise ValueError("Native closing requires one unambiguous full joint order.")
    object_pose = _b1_pose(object_pose, "object_pose")
    started = time.monotonic()
    scene = simulation.spawn_result
    handles = [scene.get(path) for path in scene.get_paths()]
    robots = [
        h
        for h in handles
        if h.get_name() == robot.uid and hasattr(h, "render_articulation")
    ]
    if len(robots) != 1:
        raise ValueError("Native Robot identity is unavailable or ambiguous.")
    native_robot = robots[0]
    source = _source_handle(handles, source_uid, native_robot.arena_name)
    target_body = source.native().get_physical_body()
    target_meshes = _native_meshes(target_body)
    if not target_meshes:
        raise ValueError("Native source has no enabled physical collision shapes.")
    if not set(links) <= set(native_robot.get_link_names()):
        raise ValueError("Native hand link topology is incomplete.")
    observed = observation_qpos.detach().clone()
    measured_fk = robot.compute_fk(
        qpos=observed, link_names=links, qpos_joint_names=names
    )
    if (
        not isinstance(measured_fk, torch.Tensor)
        or measured_fk.shape != (1, len(links), 4, 4)
        or not torch.isfinite(measured_fk).all()
    ):
        raise ValueError("Native hand verification requires finite named B1 FK.")
    world_root = (
        _b1_pose(robot.get_local_pose(to_matrix=True), "robot root")[0].cpu().numpy()
    )
    for index, name in enumerate(links):
        predicted = world_root @ measured_fk[0, index].detach().cpu().numpy()
        if not np.allclose(
            predicted, _rigid_pose(native_robot.get_link_pose(name)), atol=2e-4, rtol=0
        ):
            raise ValueError("Named actual-mimic hand FK does not match native frames.")
    states = observed.repeat(cfg.sample_count, 1)
    ids = [names.index(name) for name in hand_names]
    states[:, ids] = torch.lerp(
        opened.to(states),
        closed.to(states),
        torch.linspace(
            0, 1, cfg.sample_count, dtype=states.dtype, device=states.device
        )[:, None],
    )
    fk = robot.compute_fk(
        qpos=states, link_names=[*links, solver.end_link_name], qpos_joint_names=names
    )
    if (
        not isinstance(fk, torch.Tensor)
        or fk.shape != (cfg.sample_count, len(links) + 1, 4, 4)
        or not torch.isfinite(fk).all()
    ):
        raise ValueError("Native hand capture requires finite sampled named FK.")
    tcp = torch.as_tensor(solver.tcp, dtype=fk.dtype, device=fk.device)
    _rigid_pose(tcp.detach().cpu().numpy())
    tools = fk[:, -1] @ tcp
    if not torch.allclose(tools, tools[0:1].expand_as(tools), rtol=0, atol=1e-6):
        raise ValueError("The declared motion TCP moves during hand actuation.")
    relative = pose_inv(tools)[:, None] @ fk[:, :-1]
    parts = []
    for index, name in enumerate(links):
        body = native_robot.render_articulation.get_physical_body(name)
        meshes = _native_meshes(body)
        if meshes:
            parts.append(
                NativeContactPart(
                    name,
                    meshes,
                    relative[:, index].detach().cpu().double().numpy(),
                    float(body.get_rest_offset()),
                    name in cfg.pad_link_names,
                    float(body.get_contact_offset()) if revision >= 2 else None,
                )
            )
    query = NativeClosingQuery(
        tuple(parts),
        target_meshes,
        float(target_body.get_rest_offset()),
        scope=scope,
        revision=revision,
        target_contact_offset=(
            float(target_body.get_contact_offset()) if revision >= 2 else None
        ),
        gravity_in_object=gravity_in_object,
    )
    try:
        query.identity = _physical_identity(
            identity,
            {
                "simulation_uid": source_uid,
                "native_object_path": source.path,
                "native_robot_path": native_robot.path,
                "motion_part": cfg.motion_part,
                "hand_part": cfg.hand_part,
                "native_hand_links": [p.name for p in parts],
                "native_hand_shapes": sum(len(p.meshes) for p in parts),
                "native_target_shapes": len(target_meshes),
            },
        )
        query.identity.setdefault("object_id", source_uid)
        query.object_pose_at_capture = object_pose
        query._native_owners = (simulation, robot, native_robot, source)
        query.build_seconds = time.monotonic() - started
        if opening is not None:
            support = _source_handle(
                handles, opening.support_uid, native_robot.arena_name
            )
            body = support.native().get_physical_body()
            support_meshes = _native_meshes(body)
            if not support_meshes:
                raise ValueError(
                    "Native support has no enabled physical collision shapes."
                )
            query.opening_screen = NativeOpeningQuery(
                query,
                support_meshes,
                opening.reference_pose,
                opening.object_target_pose,
                float(body.get_rest_offset()),
                float(body.get_contact_offset()),
                scope=opening.scope,
                identity=_physical_identity(
                    opening.identity,
                    {
                        "reference_simulation_uid": opening.support_uid,
                        "native_reference_path": support.path,
                        "native_robot_path": native_robot.path,
                    },
                ),
            )
        return query
    except BaseException:
        query.close()
        raise


def _build_native_rigid_rest_scene(
    simulation: SimulationManager, closing: NativeClosingQuery
) -> NativeRigidRestScene:
    _require_fcl()
    if not isinstance(closing, NativeClosingQuery):
        raise TypeError("closing must be NativeClosingQuery.")
    closing._require_open()
    if closing.rigid_scene_screen is not None:
        raise RuntimeError("The closing query already owns its rigid scene capture.")
    owners = closing._native_owners
    if owners is None or owners[0] is not simulation:
        raise ValueError("Rigid rest capture requires the exact native closing owner.")
    _, robot, native_robot, source = owners
    _require_simulation(simulation, robot)
    scene = simulation.spawn_result
    try:
        unchanged = (
            scene.get(source.path) is source
            and scene.get(native_robot.path) is native_robot
        )
    except KeyError:
        unchanged = False
    if not unchanged:
        raise ValueError("Native source or Robot was replaced after closing capture.")
    source_uid = source.get_name()
    if source_uid not in simulation.get_rigid_object_uid_list():
        raise ValueError("The native source is no longer registered.")
    handles = [scene.get(path) for path in scene.get_paths()]
    targets = []
    for uid in simulation.get_rigid_object_uid_list():
        if uid == source_uid:
            continue
        handle = _source_handle(handles, uid, source.arena_name)
        body = handle.native().get_physical_body()
        meshes = _native_meshes(body)
        if not meshes:
            continue
        rest = float(body.get_rest_offset())
        if not math.isfinite(rest) or rest < 0:
            raise ValueError(
                "Native rigid rest offsets must be finite and nonnegative."
            )
        pose = (
            _b1_pose(
                simulation.get_rigid_object(uid).get_local_pose(to_matrix=True),
                "rigid pose",
            )[0]
            .cpu()
            .double()
            .numpy()
        )
        manager = trimesh.collision.CollisionManager()
        for index, mesh in enumerate(meshes):
            manager.add_object(f"{uid}:{index}", mesh, transform=pose)
        targets.append((uid, rest, manager))
    result = NativeRigidRestScene(closing.parts, targets, scope=closing.scope)
    closing.rigid_scene_screen = result
    return result
