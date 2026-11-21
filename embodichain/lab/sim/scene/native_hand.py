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

"""Read-only native hand/rigid geometry capture using the shared object decoder.

No collision query, grasp acceptance policy, or experiment revision is selected
here. Atomic Skills compose these observations with GraspKit qualification.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from embodichain.lab.sim.objects.backends.collision import (
    PhysicalCollisionGeometry,
    capture_physical_collision_geometry,
)
from embodichain.lab.sim.utility.grasp_geometry import (
    RobotHandGeometryCfg,
    _resolve_hand_frame,
    _sample_hand_stroke,
)
from embodichain.toolkits.graspkit.pg_grasp.native_contact import (
    NativeContactPart,
    NativeRigidObstacle,
    _rigid_pose,
)
from embodichain.utils.math import pose_inv

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = [
    "NativeHandGeometry",
    "capture_native_hand_geometry",
    "capture_native_rigid_geometry",
    "capture_native_rigid_obstacles",
]


@dataclass(frozen=True, slots=True)
class NativeHandGeometry:
    """Owned geometry for one observed hand stroke and exact native source.

    Args:
        parts: Ordered link-local cooked meshes and sampled TCP-relative FK.
        target: Captured target collision shapes and physical offsets.
        object_pose: Explicit same-observation object-to-world pose, shape (1, 4, 4).
        identity: Captured physical identity; caller labels cannot overwrite it.

    The private owner token is installed only by the capture function. Subsequent
    rigid capture validates it against the current simulation before reading.
    """

    parts: tuple[NativeContactPart, ...]
    target: PhysicalCollisionGeometry
    object_pose: torch.Tensor
    identity: Mapping[str, object]
    _owners: tuple[object, object, object, object] = field(repr=False)


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


def capture_native_hand_geometry(
    robot: Robot,
    simulation: SimulationManager,
    *,
    source_uid: str,
    hand_geometry_cfg: RobotHandGeometryCfg,
    open_qpos: torch.Tensor,
    grasp_qpos: torch.Tensor,
    observation_qpos: torch.Tensor,
    object_pose: torch.Tensor,
    include_contact_offsets: bool,
) -> NativeHandGeometry:
    """Capture cooked shapes and named closing FK without stepping simulation.

    Args:
        robot: Exact simulation-owned Robot.
        simulation: Supported B1 Default simulation.
        source_uid: Explicit registered native rigid target UID.
        hand_geometry_cfg: Declared topology, TCP and stroke sampling count.
        open_qpos: Resolved open command in full hand-part order, shape (1, D).
        grasp_qpos: Resolved closed command in the same order, shape (1, D).
        observation_qpos: Same-observation full named robot state, shape (1, J).
        object_pose: Same-observation object-to-world pose, shape (1, 4, 4).
        include_contact_offsets: Request original contact offsets as well as rest.

    Returns:
        Owned geometry and the exact runtime owners needed for subsequent reads.

    Raises:
        ValueError: Backend, batch, identity, geometry or declared frames are unsupported.
    """
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
    target = _capture_rigid(source, include_contact_offsets=include_contact_offsets)
    if not target.meshes:
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
    fk, tools = _sample_hand_stroke(
        robot,
        cfg,
        observed,
        link_names=links,
        joint_names=hand_names,
        solver=solver,
    )
    _rigid_pose(torch.as_tensor(solver.tcp))
    relative = pose_inv(tools)[:, None] @ fk
    parts = []
    for index, name in enumerate(links):
        body = native_robot.render_articulation.get_physical_body(name)
        geometry = capture_physical_collision_geometry(
            body,
            source_path=f"{native_robot.path}/{name}",
            include_contact_offsets=include_contact_offsets,
        )
        if geometry.meshes:
            parts.append(
                NativeContactPart(
                    name,
                    geometry.meshes,
                    relative[:, index].detach().cpu().double().numpy(),
                    geometry.rest_offset,
                    name in cfg.pad_link_names,
                    geometry.contact_offset,
                )
            )
    return NativeHandGeometry(
        tuple(parts),
        target,
        object_pose,
        {
            "simulation_uid": source_uid,
            "native_object_path": source.path,
            "native_robot_path": native_robot.path,
            "motion_part": cfg.motion_part,
            "hand_part": cfg.hand_part,
            "native_hand_links": [p.name for p in parts],
            "native_hand_shapes": sum(len(p.meshes) for p in parts),
            "native_target_shapes": len(target.meshes),
        },
        (simulation, robot, native_robot, source),
    )


def _require_owners(
    simulation: SimulationManager, geometry: NativeHandGeometry
) -> tuple[Any, Any]:
    if (
        not isinstance(geometry, NativeHandGeometry)
        or geometry._owners[0] is not simulation
    ):
        raise ValueError("Rigid capture requires the exact native hand owner.")
    _, robot, native_robot, source = geometry._owners
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
        raise ValueError("Native source or Robot was replaced after hand capture.")
    if source.get_name() not in simulation.get_rigid_object_uid_list():
        raise ValueError("The native source is no longer registered.")
    return native_robot, source


def capture_native_rigid_geometry(
    simulation: SimulationManager,
    hand: NativeHandGeometry,
    uid: str,
    *,
    include_contact_offsets: bool,
) -> PhysicalCollisionGeometry:
    """Read one same-arena rigid body's geometry under the captured hand ownership.

    Args:
        simulation: Exact simulation owning the hand observation.
        hand: Previously captured hand geometry, whose owners must still match.
        uid: Explicit registered rigid UID; no role or support selection is inferred.
        include_contact_offsets: Request contact offsets as well as rest offsets.

    Returns:
        Captured body-local shapes and offsets; disabled collision yields no meshes.
    """
    _, source = _require_owners(simulation, hand)
    if uid not in simulation.get_rigid_object_uid_list():
        raise ValueError("Native rigid geometry requires an explicitly registered UID.")
    scene = simulation.spawn_result
    handle = _source_handle(
        [scene.get(p) for p in scene.get_paths()], uid, source.arena_name
    )
    return _capture_rigid(handle, include_contact_offsets=include_contact_offsets)


def _capture_rigid(
    handle: Any, *, include_contact_offsets: bool
) -> PhysicalCollisionGeometry:
    return capture_physical_collision_geometry(
        handle.native().get_physical_body(),
        source_path=handle.path,
        include_contact_offsets=include_contact_offsets,
    )


def capture_native_rigid_obstacles(
    simulation: SimulationManager,
    hand: NativeHandGeometry,
) -> tuple[NativeRigidObstacle, ...]:
    """Read ordered same-arena rigid obstacles, excluding the exact captured source.

    Args:
        simulation: Exact runtime owning the captured hand and target.
        hand: Geometry observation whose original owners must remain registered.

    Returns:
        Ordered physical obstacle values; collision queries are constructed by the caller.
    """
    _, source = _require_owners(simulation, hand)
    scene = simulation.spawn_result
    handles = [scene.get(path) for path in scene.get_paths()]
    targets = []
    for uid in simulation.get_rigid_object_uid_list():
        if uid == hand.identity["simulation_uid"]:
            continue
        handle = _source_handle(handles, uid, source.arena_name)
        rigid = _capture_rigid(handle, include_contact_offsets=False)
        if rigid.meshes:
            pose = (
                _b1_pose(
                    simulation.get_rigid_object(uid).get_local_pose(to_matrix=True),
                    "rigid pose",
                )[0]
                .cpu()
                .double()
                .numpy()
            )
            targets.append(
                NativeRigidObstacle(uid, rigid.rest_offset, rigid.meshes, pose)
            )
    return tuple(targets)
