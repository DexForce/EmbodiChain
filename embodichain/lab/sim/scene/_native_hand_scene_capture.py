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

"""Native body/topology adaptation for explicit same-observation hand queries."""

from __future__ import annotations
from collections.abc import Callable
import math
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET
import numpy as np
import torch
from embodichain.lab.sim.objects.backends.collision import _native_meshes
from embodichain.toolkits.graspkit.pg_grasp.native_contact import NativeQueryScope
from .native_hand_scene import (
    NativeSceneBody,
    NativeScenePlane,
    NativeHandSceneQuery,
    _pose,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
__all__: list[str] = []


def _descendants(children: dict[str, set[str]], root: str) -> set[str]:
    result, pending = set(), [root]
    while pending:
        name = pending.pop()
        if name not in result:
            result.add(name)
            pending.extend(children.get(name, ()))
    return result


class _MeshShapeView:
    """Read only the finite native shapes through the existing mesh adapter."""

    def __init__(self, source: Any, shapes: tuple[Any, ...]) -> None:
        self.source, self.shapes = source, shapes

    def is_collision_enable(self) -> bool:
        return self.source.is_collision_enable()

    def get_shape_count(self) -> int:
        return len(self.shapes)

    def get_shape_name(self, index: int) -> str:
        return f"shape_{index}"

    def get_shape_geometry(self, index: int) -> Any:
        return self.shapes[index]


def _shapes(
    body: Any, name: str
) -> tuple[tuple[Any, ...], tuple[NativeScenePlane, ...]]:
    from dexsim.engine import BoxGeometry, ConvexMeshGeometry, PlaneGeometry

    finite, planes = [], []
    for index in range(int(body.get_shape_count())):
        try:
            shape = body.get_shape_geometry(index)
        except Exception as error:
            raise ValueError(
                f"handover_native_shape_readback_unavailable:{name}:shape={index}:{error}"
            ) from error
        if isinstance(shape, PlaneGeometry):
            normal = np.asarray(shape.normal, dtype=np.float64)
            distance = float(shape.distance)
            if (
                normal.shape != (3,)
                or not np.isfinite(normal).all()
                or not math.isfinite(distance)
                or np.linalg.norm(normal) == 0
            ):
                raise ValueError(
                    f"handover_native_plane_unavailable:{name}:shape={index}"
                )
            normal = normal / np.linalg.norm(normal)
            local = _pose(shape.local_pose)
            # SDK distance is measured from the origin along the plane normal.
            planes.append(
                NativeScenePlane(
                    local[:3, :3] @ normal,
                    local[:3, :3] @ (normal * distance) + local[:3, 3],
                )
            )
        elif isinstance(shape, (ConvexMeshGeometry, BoxGeometry)):
            finite.append(shape)
        else:
            kind = f"{type(shape).__module__}.{type(shape).__name__}"
            raise ValueError(
                f"handover_native_shape_unsupported:{name}:shape={index}:type={kind}"
            )
    try:
        meshes = _native_meshes(_MeshShapeView(body, tuple(finite))) if finite else ()
    except (ValueError, RuntimeError, TypeError, AttributeError) as error:
        raise ValueError(f"handover_native_mesh_unavailable:{name}:{error}") from error
    return meshes, tuple(planes)


def _body(
    name: str, owner: str, link: str | None, body: Any, pose: Any
) -> NativeSceneBody | None:
    if body is None:
        raise ValueError(f"handover_native_body_unavailable:{name}")
    if not body.is_collision_enable():
        return None
    meshes, planes = _shapes(body, name)
    if not meshes and not planes:
        raise ValueError(f"handover_enabled_native_shapes_unavailable:{name}")
    contact, rest = float(body.get_contact_offset()), float(body.get_rest_offset())
    if (
        not math.isfinite(contact)
        or not math.isfinite(rest)
        or not 0 <= contact
        or rest > contact
    ):
        raise ValueError(f"handover_native_offsets_unavailable:{name}")
    return NativeSceneBody(
        name, owner, link, meshes, _pose(pose).copy(), contact, rest, planes
    )


def _hand_links(robot, joint_ids, control_part):
    document = ET.parse(robot.cfg.fpath).getroot()
    joints = {joint.get("name"): joint for joint in document.findall("joint")}
    names = tuple(robot.joint_names)
    selected = {names[index] for index in joint_ids}
    if not selected or not selected <= joints.keys():
        raise ValueError("handover_hand_joint_identity_unavailable")
    parents = {
        joint.find("child").get("link"): joint.find("parent").get("link")
        for joint in joints.values()
    }

    root = robot.cfg.solver_cfg[control_part].end_link_name
    children = {}
    for child, parent in parents.items():
        children.setdefault(parent, set()).add(child)
    links = _descendants(children, root)
    moving = {
        name
        for name, joint in joints.items()
        if joint.get("type") != "fixed"
        and joint.find("child").get("link") in links
        and joint.find("parent").get("link") in links
    }
    if moving != selected:
        raise ValueError("handover_hand_requires_all_moving_joints_including_mimics")
    authored = {
        link.get("name")
        for link in document.findall("link")
        if link.findall("collision")
    }
    return tuple(sorted(links)), authored


def capture(
    robot: Robot,
    simulation: SimulationManager,
    *,
    scope: NativeQueryScope,
    expected_qpos: torch.Tensor,
    root_pose: torch.Tensor,
    receiver_control_part: str,
    receiver_joint_ids: tuple[int, ...],
    required_owner_uids: tuple[str, ...],
    excluded_owner_uid: str,
    native_revision: int,
    read_native_revision: Callable[[], int],
) -> NativeHandSceneQuery:
    """Capture native geometry only; callers own invocation/command semantics."""
    if (
        not isinstance(scope, NativeQueryScope)
        or simulation.num_envs != 1
        or not simulation.is_default_backend
    ):
        raise ValueError("handover_scene_requires_same_tick_Default_B1_context")
    if simulation.get_robot(robot.uid) is not robot:
        raise ValueError(
            "Native hand capture requires the exact registered robot instance."
        )
    if (
        not isinstance(expected_qpos, torch.Tensor)
        or expected_qpos.shape != (1, len(robot.joint_names))
        or not expected_qpos.is_floating_point()
        or not torch.isfinite(expected_qpos).all()
    ):
        raise ValueError(
            "Native hand capture requires finite complete B1 observed joints."
        )
    if (
        type(native_revision) is not int
        or native_revision < 0
        or not callable(read_native_revision)
    ):
        raise ValueError(
            "Native hand capture requires an actual revision and source reader."
        )
    expected_qpos = expected_qpos.detach().clone()
    observed_revision = read_native_revision()
    if type(observed_revision) is not int or observed_revision != native_revision:
        raise ValueError(
            "Native hand capture requires the same actual observation revision."
        )
    if (
        receiver_control_part not in robot.control_parts
        or not receiver_joint_ids
        or len(set(receiver_joint_ids)) != len(receiver_joint_ids)
        or any(
            type(i) is not int or not 0 <= i < len(robot.joint_names)
            for i in receiver_joint_ids
        )
    ):
        raise ValueError(
            "Native hand capture requires exact receiver group/joint identities."
        )
    if (
        not required_owner_uids
        or excluded_owner_uid not in required_owner_uids
        or any(
            type(uid) is not str or not uid or uid != uid.strip()
            for uid in required_owner_uids
        )
    ):
        raise ValueError(
            "Native hand capture requires its explicit physical roster and excluded owner."
        )
    if not torch.equal(robot.get_qpos(), expected_qpos):
        raise ValueError("handover_scene_observed_qpos_changed")
    root = _pose(root_pose).copy()
    scene = simulation.spawn_result
    if scene is None:
        raise ValueError("handover_materialized_scene_unavailable")
    handles = [scene.get(path) for path in scene.get_paths()]
    required_uids = set(required_owner_uids)
    if any(
        sum(handle.get_name() == uid for handle in handles) != 1
        for uid in required_uids
    ):
        raise ValueError("handover_materialized_physical_inventory_unavailable")
    candidates = [
        handle
        for handle in handles
        if handle.get_name() == robot.uid and hasattr(handle, "render_articulation")
    ]
    if len(candidates) != 1:
        raise ValueError("handover_native_robot_identity_unavailable")
    native_robot = candidates[0]
    arena_from_world = np.linalg.inv(
        _pose(scene.get_arena(native_robot.arena_name).get_root_node().get_world_pose())
    )
    if not np.allclose(
        root, arena_from_world @ _pose(native_robot.get_world_pose()), atol=2e-4, rtol=0
    ):
        raise ValueError("handover_native_robot_root_frame_mismatch")
    target_uid = excluded_owner_uid
    excluded = [
        handle
        for handle in handles
        if handle.get_name() == target_uid and callable(getattr(handle, "native", None))
    ]
    if len(excluded) != 1:
        raise ValueError("handover_exact_target_native_identity_unavailable")
    links, authored = _hand_links(robot, receiver_joint_ids, receiver_control_part)
    if not set(links) <= set(native_robot.get_link_names()):
        raise ValueError("handover_native_receiver_topology_unavailable")
    observed = robot.compute_fk(
        qpos=expected_qpos.detach().clone(),
        link_names=list(links),
        qpos_joint_names=tuple(robot.joint_names),
    )
    if observed.shape != (1, len(links), 4, 4) or not torch.isfinite(observed).all():
        raise ValueError("handover_observed_named_hand_fk_unavailable")
    parts, obstacles, inventory = [], [], []
    for index, link in enumerate(links):
        expected = root @ observed[0, index].detach().cpu().numpy()
        native_pose = arena_from_world @ _pose(native_robot.get_link_pose(link))
        if not np.allclose(expected, native_pose, atol=2e-4, rtol=0):
            raise ValueError("handover_actual_mimic_hand_fk_frame_mismatch")
        body = native_robot.render_articulation.get_physical_body(link)
        if link not in authored and (
            body is None
            or (body.is_collision_enable() and int(body.get_shape_count()) == 0)
        ):
            inventory.append(
                {
                    "owner": robot.uid,
                    "link": link,
                    "status": "authored_collisionless_hand_frame",
                }
            )
            continue
        part = _body(f"{robot.uid}/{link}", robot.uid, link, body, np.eye(4))
        inventory.append(
            {
                "owner": robot.uid,
                "link": link,
                "status": "enabled" if part else "native_collision_disabled",
                "shapes": len(part.meshes) if part else 0,
            }
        )
        if part is not None:
            if part.planes:
                raise ValueError(
                    f"handover_receiver_infinite_shape_unsupported:{part.name}"
                )
            parts.append(part)
    if not parts:
        raise ValueError("handover_receiver_enabled_native_geometry_unavailable")
    for handle in handles:
        if handle is native_robot or handle is excluded[0]:
            continue
        uid = handle.get_name()
        if hasattr(handle, "render_articulation"):
            for link in handle.get_link_names():
                body = handle.render_articulation.get_physical_body(link)
                if body is None or (
                    body.is_collision_enable() and int(body.get_shape_count()) == 0
                ):
                    if not handle.get_link_desc(link).collisions:
                        inventory.append(
                            {
                                "owner": uid,
                                "link": link,
                                "status": "retained_collisionless_link",
                            }
                        )
                        continue
                pose = arena_from_world @ _pose(handle.get_link_pose(link))
                obstacle = _body(f"{handle.path}/{link}", uid, link, body, pose)
                inventory.append(
                    {
                        "owner": uid,
                        "link": link,
                        "status": (
                            "enabled" if obstacle else "native_collision_disabled"
                        ),
                        "shapes": (
                            len(obstacle.meshes) + len(obstacle.planes)
                            if obstacle
                            else 0
                        ),
                        "plane_shapes": len(obstacle.planes) if obstacle else 0,
                        "retained_body_scale": np.asarray(
                            handle.desc.body_scale
                        ).tolist(),
                        "geometry_scale_source": "native_shape_geometry_only",
                    }
                )
                if obstacle is not None:
                    obstacles.append(obstacle)
        elif callable(getattr(handle, "native", None)):
            body = handle.native().get_physical_body()
            obstacle = _body(
                handle.path,
                uid,
                None,
                body,
                arena_from_world @ _pose(handle.get_world_pose()),
            )
            inventory.append(
                {
                    "owner": uid,
                    "link": None,
                    "status": "enabled" if obstacle else "native_collision_disabled",
                    "shapes": (
                        len(obstacle.meshes) + len(obstacle.planes) if obstacle else 0
                    ),
                    "plane_shapes": len(obstacle.planes) if obstacle else 0,
                    "geometry_scale_source": "native_shape_geometry_only",
                }
            )
            if obstacle is not None:
                obstacles.append(obstacle)
    if not obstacles:
        raise ValueError("handover_non_target_native_scene_geometry_unavailable")
    if not torch.equal(robot.get_qpos(), expected_qpos):
        raise ValueError("handover_scene_snapshot_robot_changed")
    # The query constructor rechecks the same native revision after all capture
    # reads, before allocating any owned native collision queries.
    return NativeHandSceneQuery(
        scope=scope,
        robot_uid=robot.uid,
        joint_names=tuple(robot.joint_names),
        root_pose=root,
        parts=tuple(parts),
        obstacles=tuple(obstacles),
        inventory=inventory,
        native_revision=native_revision,
        read_native_revision=read_native_revision,
    )
