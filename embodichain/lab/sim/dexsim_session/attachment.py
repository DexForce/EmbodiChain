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

"""Task-local bindings into a finalized, externally hosted DexSim World."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from functools import wraps
from weakref import ref
from threading import get_ident
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from embodichain.lab.sim.cfg import RigidObjectCfg, RobotCfg
from embodichain.lab.sim.cfg.source import _asset_file_path
from embodichain.lab.sim.objects.rigid_object import RigidObject
from embodichain.lab.sim.objects.robot import Robot
from embodichain.lab.sim.utility.gripper_binding import prepare_robot_grippers
from .robot_state import _select_dofs

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext
    from dexsim.scene import SpawnedArticulation

__all__ = ["AttachedSimulationManager"]


def _bind_robot_cfg(cfg: RobotCfg, handle: SpawnedArticulation) -> RobotCfg:
    """Validate a resolved URDF assembly against the selected native topology."""
    cfg = deepcopy(cfg)
    if cfg.gripper_bindings:
        prepare_robot_grippers(cfg)
    source = _asset_file_path(cfg)
    if source is None and cfg.urdf_cfg is not None:
        source = cfg.fpath = cfg.urdf_cfg.assemble_urdf()
    if not source or not str(source).lower().endswith(".urdf"):
        raise ValueError("An attached robot requires a resolved URDF assembly")
    root = ElementTree.parse(source).getroot()
    links = {item.attrib["name"] for item in root.findall("link")}
    missing = links.difference(handle.get_link_names())
    if missing:
        raise ValueError(f"{handle.path}: missing robot links {sorted(missing)}")
    joints = {
        item.attrib["name"]: item
        for item in root.findall("joint")
        if item.attrib["type"] != "fixed"
    }
    layout = handle.joint_dof_layout
    if set(joints) != {item.name for item in layout}:
        raise ValueError(f"{handle.path}: live DOF names do not match {cfg.uid}'s URDF")
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        desc = handle.get_joint_desc(name)
        parent_child = (
            joint.find("parent").attrib["link"],
            joint.find("child").attrib["link"],
        )
        if parent_child != (desc.parent_link_name, desc.child_link_name):
            raise ValueError(f"{handle.path}: incompatible links for {name}")
        origin = joint.find("origin")
        frame = np.eye(4)
        if origin is not None:
            frame[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            frame[:3, :3] = Rotation.from_euler(
                "xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]
            ).as_matrix()
        if not np.allclose(desc.origin_pose, frame, atol=2e-5, rtol=0):
            raise ValueError(f"{handle.path}: incompatible joint frame for {name}")
        if joint.attrib["type"] == "fixed":
            if desc.joint_type.name != "FIXED":
                raise ValueError(f"{handle.path}: expected fixed joint {name}")
        else:
            axis = joint.find("axis")
            xyz = np.array(
                [
                    float(v)
                    for v in (axis.get("xyz") if axis is not None else "1 0 0").split()
                ]
            )
            if not np.allclose(desc.axis, xyz / np.linalg.norm(xyz), atol=1e-6, rtol=0):
                raise ValueError(f"{handle.path}: incompatible motion axis for {name}")
    cursor = 0
    for item in layout:
        if item.dof_start != cursor or item.dof_count != 1:
            raise ValueError(
                f"{handle.path}: incompatible state layout for {item.name}"
            )
        cursor += item.dof_count
        joint = joints[item.name]
        desc = handle.get_joint_desc(item.name)
        kind = joint.attrib["type"]
        if kind not in (
            "continuous",
            "revolute",
            "prismatic",
        ) or item.position_encoding != (
            "translation" if kind == "prismatic" else "angle"
        ):
            raise ValueError(f"{handle.path}: incompatible joint type for {item.name}")
        mimic = joint.find("mimic")
        if (desc.mimic_parent or None) != (
            None if mimic is None else mimic.attrib["joint"]
        ):
            raise ValueError(
                f"{handle.path}: incompatible mimic parent for {item.name}"
            )
        if mimic is not None and (desc.mimic_multiplier, desc.mimic_offset) != (
            float(mimic.get("multiplier", "1")),
            float(mimic.get("offset", "0")),
        ):
            raise ValueError(
                f"{handle.path}: incompatible mimic relation for {item.name}"
            )
    for selectors in (cfg.control_parts or {}).values():
        _select_dofs(layout, selectors)
    return cfg


class _BorrowedView:
    """Guard retained data/batch operations without extending native lifetime."""

    def __init__(self, owner: AttachedSimulationManager, view: object):
        self._owner = owner
        self._ref = ref(view)

    def __getattr__(self, name: str) -> Any:
        self._owner.validate()
        obj = self._ref()
        if obj is None:
            raise RuntimeError("Released borrowed data view")
        value = getattr(obj, name)
        if name in ("articulation_view", "body_view"):
            return _BorrowedView(self._owner, value)
        if callable(value):

            def checked(*args: Any, **kwargs: Any) -> Any:
                self._owner.validate()
                return getattr(self._ref(), name)(*args, **kwargs)

            return checked
        return value


class _BorrowedFacade:
    """Check liveness even when a planner retains the facade between calls."""

    _attachment: AttachedSimulationManager | None = None

    def _release_local_views(self) -> None:
        """Retire cached batches and solvers even when a planner retains this facade."""
        owner = self._attachment
        self.__dict__.clear()
        self._attachment = owner

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)
        owner = object.__getattribute__(self, "_attachment")
        if owner is None:
            return super().__getattribute__(name)
        owner.validate()
        value = super().__getattribute__(name)
        if name == "body_data" and value is not None:
            return _BorrowedView(owner, value)
        if callable(value):

            @wraps(value)
            def checked(*args: Any, **kwargs: Any) -> Any:
                owner.validate()
                return value(*args, **kwargs)

            return checked
        return value


class _AttachedRobot(_BorrowedFacade, Robot):
    pass


class _AttachedRigidObject(_BorrowedFacade, RigidObject):
    pass


class AttachedSimulationManager:
    """A single-arena view borrowing native objects on their host thread."""

    def __init__(
        self,
        context: SessionContext,
        arena: str,
        robots: Sequence[RobotCfg],
        rigid_objects: Sequence[RigidObjectCfg],
        paths: Mapping[str, str],
        logical_entities: Mapping[str, str],
        device: torch.device,
    ) -> None:
        self.instance_id = -1
        self._thread = get_ident()
        self._context = context
        self._scene = context.scene
        self.session_id = context.session_id
        self.arena = arena
        scene = self._scene
        if scene is None or list(scene.arenas.names) != [arena]:
            raise ValueError("Attachment requires the named, finalized single arena")
        if (
            scene.backend not in ("dexsim", "newton")
            or not context.world.is_physics_manually_update()
        ):
            raise ValueError("Attachment requires a supported manually stepped World")
        if (
            scene.backend == "newton"
            and (scene.get_solver_diagnostics() or {}).get("solver_type")
            != "mujoco_warp"
        ):
            raise ValueError("Attachment requires the Newton MuJoCo-Warp backend")
        self.topology_revision = scene.topology_revision
        self.validate()
        configs = [*robots, *rigid_objects]
        uids = [cfg.uid for cfg in configs]
        if (
            not configs
            or not all(uids)
            or len(set(uids)) != len(configs)
            or set(uids) != set(paths)
            or len(set(paths.values())) != len(paths)
        ):
            raise ValueError(
                "Supply one distinct live path per configured simulation UID"
            )
        self.device = device
        self._logical_entities = dict(logical_entities)
        if set(logical_entities.values()).difference(paths):
            raise ValueError("Logical entity references an unbound simulation UID")
        self._paths = dict(paths)
        self.num_envs = 1
        self._robots: dict[str, Robot] = {}
        self._rigid_objects: dict[str, RigidObject] = {}
        try:
            for cfg in robots:
                handle = scene.get_articulation(paths[cfg.uid])
                if handle.arena_name != arena:
                    raise ValueError(f"{handle.path}: wrong arena")
                robot = _AttachedRobot(_bind_robot_cfg(cfg, handle), device)
                self._robots[cfg.uid] = robot
                robot._attachment = self
                robot._initialize_spawn_declaration(1)
                robot.attach_spawn_handles([handle])
                robot._initialize_spawn_bound(scene, borrowed=True)
            for cfg in rigid_objects:
                handle = scene.get_object(paths[cfg.uid])
                if (
                    handle.arena_name != arena
                    or cfg.body_type != handle.get_actor_type().name.lower()
                ):
                    raise ValueError(
                        f"{handle.path}: arena or rigid body type mismatch"
                    )
                body = _AttachedRigidObject(cfg, device)
                self._rigid_objects[cfg.uid] = body
                body._attachment = self
                body._initialize_spawn_declaration(1)
                body.attach_spawn_handles([handle])
                body._initialize_spawn_bound(scene, borrowed=True)
        except BaseException:
            self.detach()
            raise

    def validate(self) -> None:
        """Reject detached, replaced or rebuilt worlds before accessing a binding."""
        context = self._context
        if (
            get_ident() != self._thread
            or context is None
            or context.session_id != self.session_id
            or context.world is not self._scene.world
            or context.scene is not self._scene
            or self._scene.topology_revision != self.topology_revision
            or context.builder.has_pending_changes
            or self._scene.needs_rebuild
        ):
            raise RuntimeError("Stale or inactive DexSim World binding; attach again")

    @property
    def world_gravity(self) -> tuple[float, float, float]:
        """Read the host solver's gravity in World coordinates."""
        self.validate()
        if self._scene.backend == "newton":
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend

            values = (
                get_newton_backend(self._context.world)
                .model.gravity.numpy()
                .reshape(-1, 3)
            )
            if len(values) != 1:
                raise ValueError(
                    "Expected one native gravity vector for the selected arena"
                )
            x, y, z = values[0]
        else:
            desc = json.loads(
                self._context.world.get_physics_scene().dump_scene_config()
            )
            gravity = desc["scene_desc"]["PxSceneDesc"]["gravity"]
            x, y, z = (gravity[k] for k in ("x", "y", "z"))
        return float(x), float(y), float(z)

    def resolve_entity(self, logical_id: str) -> Robot | RigidObject:
        """Resolve a logical task entity via its configured simulation UID."""
        self.validate()
        uid = self._logical_entities[logical_id]
        return self._robots[uid] if uid in self._robots else self._rigid_objects[uid]

    def object_path(self, uid: str) -> str:
        """Resolve a task's configured simulation UID to the original live path."""
        self.validate()
        return self._paths[uid]

    def get_robot(self, uid: str) -> Robot:
        """Look up the selected live robot by configured simulation UID."""
        self.validate()
        return self._robots[uid]

    def get_rigid_object(self, uid: str) -> RigidObject:
        """Look up a selected rigid body by configured simulation UID."""
        self.validate()
        return self._rigid_objects[uid]

    def detach(self) -> None:
        """Release local references and planner registration without native cleanup."""
        from embodichain.lab.sim.sim_manager import SimulationManager

        if self._context is None:
            return
        if get_ident() != self._thread:
            raise RuntimeError("Detach the binding on the World owner thread")
        if SimulationManager._instances.get(self.instance_id) is self:
            SimulationManager.reset(self.instance_id)
        for facade in [*self._robots.values(), *self._rigid_objects.values()]:
            facade._release_local_views()
        self._robots.clear()
        self._rigid_objects.clear()
        self._context = self._scene = None

    def destroy(self, exit_process: bool | None = None) -> None:
        """Detach locally, including when invoked by manager lifecycle callers."""
        self.detach()
