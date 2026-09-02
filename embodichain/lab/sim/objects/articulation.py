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

from __future__ import annotations

import math

import torch
import dexsim
import numpy as np

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List, Sequence, Dict, Union, Tuple, Optional

from dexsim.scene import Scene
from dexsim.types import (
    ArticulationFlag,
    DriveType,
)
from dexsim.engine import MaterialInst

from embodichain.lab.sim import VisualMaterialInst, VisualMaterial, ReuseSegmentState
from embodichain.lab.sim.material import (
    _capture_render_materials,
    _restore_render_materials,
    _set_render_material,
    _wrap_first_render_material,
)
from embodichain.lab.sim.cfg import (
    _normalize_joint_target_mode,
    ArticulationCfg,
    JointDrivePropertiesCfg,
    RigidBodyPhysicsCfg,
)
from dexsim.types import PhysicalAttr
from embodichain.utils.string import (
    resolve_matching_names,
    resolve_matching_names_values,
)
from embodichain.lab.sim.common import BatchEntity
from embodichain.lab.sim.physics.newton import is_newton_gradient_mode
from embodichain.lab.sim.objects.backends import (
    SceneArticulationView,
)
from embodichain.lab.sim.objects.backends.base import ArticulationViewBase
from embodichain.lab.sim.objects.backends.newton import (
    _configure_newton_mimic_compliance,
)
from embodichain.utils.math import (
    convert_quat,
    matrix_from_quat,
    quat_from_matrix,
    matrix_from_euler,
)
from embodichain.lab.sim.utility.sim_utils import (
    _apply_default_articulation_root_properties,
    get_dexsim_drive_type,
)
from embodichain.lab.sim.utility.solver_utils import (
    create_pk_chain,
    create_pk_serial_chain,
)
from embodichain.utils import logger

if TYPE_CHECKING:
    from dexsim.scene import SpawnedArticulation


@dataclass(frozen=True, slots=True)
class _MimicInfo:
    """Mimic metadata expressed in the backing state-buffer index domain."""

    mimic_id: np.ndarray
    mimic_parent: np.ndarray
    mimic_multiplier: np.ndarray
    mimic_offset: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class ArticulationJointKinematics:
    """Backend-neutral kinematic description of one articulation joint.

    The value contains only stable names and copied numeric geometry. It does
    not expose the simulator's native joint-info object.

    Args:
        name: Stable joint name.
        joint_type: Normalized lowercase joint type, such as ``fixed``,
            ``revolute``, or ``prismatic``.
        parent_link_name: Name of the joint's parent link.
        child_link_name: Name of the joint's child link.
        origin_pose: Joint-frame pose in the parent-link frame with shape
            ``(4, 4)``.
        axis: Joint axis in the joint frame with shape ``(3,)``.
        joint_limits: Optional lower and upper position limits.
    """

    name: str
    joint_type: str
    parent_link_name: str
    child_link_name: str
    origin_pose: torch.Tensor
    axis: torch.Tensor
    joint_limits: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.name, "name"),
            (self.joint_type, "joint_type"),
            (self.parent_link_name, "parent_link_name"),
            (self.child_link_name, "child_link_name"),
        ):
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        origin_pose = torch.as_tensor(self.origin_pose, dtype=torch.float32)
        if origin_pose.shape != (4, 4) or not torch.isfinite(origin_pose).all():
            raise ValueError("origin_pose must be a finite tensor with shape (4, 4).")
        axis = torch.as_tensor(self.axis, dtype=torch.float32)
        if axis.shape != (3,) or not torch.isfinite(axis).all():
            raise ValueError("axis must be a finite tensor with shape (3,).")
        joint_limits = self.joint_limits
        if joint_limits is not None:
            if not isinstance(joint_limits, tuple) or len(joint_limits) != 2:
                raise TypeError("joint_limits must be a (lower, upper) tuple or None.")
            lower, upper = (float(value) for value in joint_limits)
            if math.isnan(lower) or math.isnan(upper) or lower > upper:
                raise ValueError("joint_limits must be ordered and cannot contain NaN.")
            joint_limits = lower, upper
        object.__setattr__(self, "joint_type", self.joint_type.lower())
        object.__setattr__(self, "origin_pose", origin_pose.clone())
        object.__setattr__(self, "axis", axis.clone())
        object.__setattr__(self, "joint_limits", joint_limits)


@dataclass
class ArticulationData:
    """Scene-batch data manager for articulations."""

    def __init__(
        self,
        entities: Sequence[SpawnedArticulation],
        scene: Scene,
        device: torch.device,
    ) -> None:
        """Initialize the ArticulationData.

        Args:
            entities: Articulation handles owned by ``scene``.
            scene: Finalized DexSim Scene.
            device: Device to use for the articulation data.
        """
        if not isinstance(scene, Scene):
            raise TypeError("ArticulationData requires a finalized DexSim Scene.")
        self.entities = entities
        self.scene = scene
        self.num_instances = len(entities)
        self.device = device
        self.articulation_view: ArticulationViewBase = (
            SceneArticulationView.from_entities(scene, entities, device)
        )

        # Backward-compatible alias for callers that use GPU/articulation ids.
        self.gpu_indices = self.articulation_view.articulation_ids_tensor

        self.dof = self.articulation_view.dof
        self.num_links = self.articulation_view.num_links
        self.link_names = self.articulation_view.link_names

        self._root_pose = torch.zeros(
            (self.num_instances, 7), dtype=torch.float32, device=self.device
        )
        self._root_lin_vel = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )
        self._root_ang_vel = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )

        self._body_link_pose = torch.zeros(
            (self.num_instances, self.num_links, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_link_vel = torch.zeros(
            (self.num_instances, self.num_links, 6),
            dtype=torch.float32,
            device=self.device,
        )

        self._body_link_lin_vel = torch.zeros(
            (self.num_instances, self.num_links, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_link_ang_vel = torch.zeros(
            (self.num_instances, self.num_links, 3),
            dtype=torch.float32,
            device=self.device,
        )

        # Current link mass-property buffers use the public articulation link
        # ordering. Initialization snapshots are captured after backend
        # materialization and remain unchanged by runtime writes.
        self._mass = torch.zeros(
            (self.num_instances, self.num_links),
            dtype=torch.float32,
            device=self.device,
        )
        self._inertia = torch.zeros(
            (self.num_instances, self.num_links, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._com_pose = torch.zeros(
            (self.num_instances, self.num_links, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self._default_mass: torch.Tensor | None = None
        self._default_inertia: torch.Tensor | None = None
        self._default_com_pose: torch.Tensor | None = None

        self._target_qpos = torch.zeros(
            (self.num_instances, self.dof), dtype=torch.float32, device=self.device
        )
        self._qpos = torch.zeros(
            (self.num_instances, self.dof), dtype=torch.float32, device=self.device
        )
        self._target_qvel = torch.zeros(
            (self.num_instances, self.dof), dtype=torch.float32, device=self.device
        )
        self._qvel = torch.zeros(
            (self.num_instances, self.dof), dtype=torch.float32, device=self.device
        )
        self._qacc = torch.zeros(
            (self.num_instances, self.dof), dtype=torch.float32, device=self.device
        )
        self._qf = torch.zeros(
            (self.num_instances, self.dof), dtype=torch.float32, device=self.device
        )
        self._qpos_limits = torch.as_tensor(
            np.array([entity.get_joint_position_limits() for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )
        self._qvel_limits = torch.as_tensor(
            np.array([entity.get_joint_velocity_limit() for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )
        self._qf_limits = torch.as_tensor(
            np.array([entity.get_joint_effort_limit() for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def is_newton_backend(self) -> bool:
        return self.articulation_view.is_newton_backend

    @property
    def is_ready(self) -> bool:
        return self.articulation_view.is_ready

    @property
    def root_pose(self) -> torch.Tensor:
        """Get the root pose of the articulation.

        Returns:
            torch.Tensor: Root poses with shape ``(num_instances, 7)`` in
            ``(x, y, z, qx, qy, qz, qw)`` order.
        """
        return self.articulation_view.fetch_root_pose(self._root_pose)

    @property
    def root_lin_vel(self) -> torch.Tensor:
        """Get the linear velocity of the root link of the articulation.

        Returns:
            torch.Tensor: The linear velocity of the root link with shape of (num_instances, 3).
        """
        return self.articulation_view.fetch_root_linear_velocity(self._root_lin_vel)

    @property
    def root_ang_vel(self) -> torch.Tensor:
        """Get the angular velocity of the root link of the articulation.

        Returns:
            torch.Tensor: The angular velocity of the root link with shape of (num_instances, 3).
        """
        return self.articulation_view.fetch_root_angular_velocity(self._root_ang_vel)

    @property
    def root_vel(self) -> torch.Tensor:
        """Get the velocity of the root link of the articulation.

        Returns:
            torch.Tensor: The velocity of the root link, concatenating linear and angular velocities.
        """
        return torch.cat((self.root_lin_vel, self.root_ang_vel), dim=-1)

    @property
    def qpos(self) -> torch.Tensor:
        """Get the current positions (qpos) of the articulation.

        Returns:
            torch.Tensor: The current positions of the articulation with shape of (num_instances, dof).
        """
        return self.articulation_view.fetch_qpos(self._qpos)

    @property
    def target_qpos(self) -> torch.Tensor:
        """Get the target positions (target_qpos) of the articulation.

        Returns:
            torch.Tensor: The target positions of the articulation with shape of (num_instances, dof).
        """
        return self.articulation_view.fetch_target_qpos(self._target_qpos)

    @property
    def qvel(self) -> torch.Tensor:
        """Get the current velocities (qvel) of the articulation.

        Returns:
            torch.Tensor: The current velocities of the articulation with shape of (num_instances, dof).
        """
        return self.articulation_view.fetch_qvel(self._qvel)

    @property
    def target_qvel(self) -> torch.Tensor:
        """Get the target velocities (target_qvel) of the articulation.
        Returns:
            torch.Tensor: The target velocities of the articulation with shape of (num_instances, dof).
        """
        return self.articulation_view.fetch_target_qvel(self._target_qvel)

    @property
    def qacc(self) -> torch.Tensor:
        """Get the current accelerations (qacc) of the articulation.

        Returns:
            torch.Tensor: The current accelerations of the articulation with shape of (num_instances, dof).
        """
        return self.articulation_view.fetch_qacc(self._qacc)

    @property
    def qf(self) -> torch.Tensor:
        """Get the current forces (qf) of the articulation.

        Returns:
            torch.Tensor: The current forces of the articulation with shape of (num_instances, dof).
        """
        return self.articulation_view.fetch_qf(self._qf)

    @property
    def body_link_pose(self) -> torch.Tensor:
        """Get the pose of all links in the articulation.

        Returns:
            torch.Tensor: Link poses with shape ``(N, num_links, 7)`` in
            ``(x, y, z, qx, qy, qz, qw)`` order.
        """
        return self.articulation_view.fetch_link_pose(self._body_link_pose)

    @property
    def body_link_vel(self) -> torch.Tensor:
        """Get the velocities of all links in the articulation.

        Returns:
            torch.Tensor: The poses of the links in the articulation with shape (N, num_links, 6).
        """
        return self.articulation_view.fetch_link_velocity(
            self._body_link_vel,
            self._body_link_lin_vel,
            self._body_link_ang_vel,
        )

    def _entity_drive_properties(self, entity: object) -> tuple[object, ...]:
        """Read drive values without conflating backend target semantics."""
        if (
            isinstance(self.articulation_view, SceneArticulationView)
            and self.is_newton_backend
        ):
            return tuple(entity.get_newton_drive())
        return tuple(entity.get_drive())

    def _entity_link_properties(self, entity: object, link_name: str) -> object:
        """Read native mass properties through the active backend contract."""
        if (
            isinstance(self.articulation_view, SceneArticulationView)
            and self.is_newton_backend
        ):
            return entity.get_newton_link_properties(link_name)
        return entity.get_physical_attr(link_name)

    def read_physical_properties(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refresh current mass, inertia diagonal, and local COM pose buffers.

        COM poses use the EmbodiChain convention ``xyz + xyzw`` and all
        tensors use the public link ordering. DexSim physical-property
        descriptors use ``wxyz`` and are converted at this boundary.
        """
        masses: list[list[float]] = []
        inertias: list[list[np.ndarray]] = []
        com_poses: list[list[np.ndarray]] = []
        for entity in self.entities:
            mass_row: list[float] = []
            inertia_row: list[np.ndarray] = []
            com_row: list[np.ndarray] = []
            for link_name in self.link_names:
                attr = self._entity_link_properties(entity, link_name)
                mass_row.append(float(attr.mass))
                inertia_row.append(np.asarray(attr.inertia, dtype=np.float32))
                com_row.append(
                    np.concatenate(
                        (
                            np.asarray(attr.com_position, dtype=np.float32),
                            convert_quat(
                                np.asarray(attr.com_quaternion, dtype=np.float32),
                                to="xyzw",
                            ),
                        )
                    )
                )
            masses.append(mass_row)
            inertias.append(inertia_row)
            com_poses.append(com_row)

        self._mass.copy_(
            torch.as_tensor(
                np.asarray(masses, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self._inertia.copy_(
            torch.as_tensor(
                np.asarray(inertias, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
        )
        self._com_pose.copy_(
            torch.as_tensor(
                np.asarray(com_poses, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )
        )
        return self._mass, self._inertia, self._com_pose

    @property
    def mass(self) -> torch.Tensor:
        """Current link masses with shape ``(N, num_links)``."""
        return self.read_physical_properties()[0]

    @property
    def inertia(self) -> torch.Tensor:
        """Current link inertia diagonals with shape ``(N, num_links, 3)``."""
        return self.read_physical_properties()[1]

    @property
    def com_pose(self) -> torch.Tensor:
        """Current local link COM poses as ``xyz + xyzw`` tensors."""
        return self.read_physical_properties()[2]

    @property
    def default_physical_properties_initialized(self) -> bool:
        """Whether initialization-time link mass properties are available."""
        return (
            self._default_mass is not None
            and self._default_inertia is not None
            and self._default_com_pose is not None
        )

    @property
    def default_mass(self) -> torch.Tensor:
        """Initialization-time link masses with shape ``(N, num_links)``."""
        if self._default_mass is None:
            raise RuntimeError("Default articulation link masses are unavailable.")
        return self._default_mass

    @property
    def default_inertia(self) -> torch.Tensor:
        """Initialization-time link inertia diagonals."""
        if self._default_inertia is None:
            raise RuntimeError("Default articulation link inertias are unavailable.")
        return self._default_inertia

    @property
    def default_com_pose(self) -> torch.Tensor:
        """Initialization-time local link COM poses in ``xyz + xyzw`` order."""
        if self._default_com_pose is None:
            raise RuntimeError("Default articulation link COM poses are unavailable.")
        return self._default_com_pose

    def capture_default_physical_properties(
        self,
        *,
        mass: torch.Tensor,
        inertia: torch.Tensor,
        com_pose: torch.Tensor,
    ) -> None:
        """Capture backend-resolved link mass properties exactly once."""
        expected_shapes = {
            "mass": (self.num_instances, self.num_links),
            "inertia": (self.num_instances, self.num_links, 3),
            "com_pose": (self.num_instances, self.num_links, 7),
        }
        values = {"mass": mass, "inertia": inertia, "com_pose": com_pose}
        for name, value in values.items():
            if tuple(value.shape) != expected_shapes[name]:
                raise ValueError(
                    f"Expected {name} shape {expected_shapes[name]}, "
                    f"got {tuple(value.shape)}."
                )
        if self.default_physical_properties_initialized:
            raise RuntimeError(
                "Default articulation link mass properties are already captured."
            )

        self._default_mass = mass.to(self.device, dtype=torch.float32).clone()
        self._default_inertia = inertia.to(self.device, dtype=torch.float32).clone()
        self._default_com_pose = com_pose.to(self.device, dtype=torch.float32).clone()

    @property
    def joint_stiffness(self) -> torch.Tensor:
        """Get the joint stiffness of the articulation.

        Returns:
            torch.Tensor: The joint stiffness of the articulation with shape (N, dof).
        """
        return torch.as_tensor(
            np.array(
                [self._entity_drive_properties(entity)[0] for entity in self.entities]
            ),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def joint_damping(self) -> torch.Tensor:
        """Get the joint damping of the articulation.

        Returns:
            torch.Tensor: The joint damping of the articulation with shape (N, dof).
        """
        return torch.as_tensor(
            np.array(
                [self._entity_drive_properties(entity)[1] for entity in self.entities]
            ),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def joint_friction(self) -> torch.Tensor:
        """Get the joint friction of the articulation.

        Returns:
            torch.Tensor: The joint friction of the articulation with shape (N, dof).
        """
        return torch.as_tensor(
            np.array(
                [self._entity_drive_properties(entity)[4] for entity in self.entities]
            ),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def joint_armature(self) -> torch.Tensor:
        """Get the joint armature of the articulation.

        Returns:
            torch.Tensor: The joint armature of the articulation with shape (N, dof).
        """
        return torch.as_tensor(
            np.array(
                [self._entity_drive_properties(entity)[5] for entity in self.entities]
            ),
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def qpos_limits(self) -> torch.Tensor:
        """Get the joint position limits of the articulation.

        Returns:
            torch.Tensor: The joint position limits of the articulation with shape (N, dof, 2).
        """
        return self._qpos_limits

    @property
    def qvel_limits(self) -> torch.Tensor:
        """Get the joint velocity limits of the articulation.

        Returns:
            torch.Tensor: The joint velocity limits of the articulation with shape (N, dof).
        """
        return self._qvel_limits

    @property
    def qf_limits(self) -> torch.Tensor:
        """Get the joint effort limits of the articulation.

        Returns:
            torch.Tensor: The joint effort limits of the articulation with shape (N, dof).
        """
        return self._qf_limits

    @cached_property
    def link_vert_face(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Get the vertices and faces of all links in the articulation.

        Returns:
            Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
                - key (str): The name of the link.
                - vertices (torch.Tensor): The vertices of the specified link with shape (V, 3).
                - faces (torch.Tensor): The faces of the specified link with shape (F, 3).
        """
        link_vert_face = dict()
        for link_name in self.link_names:
            verts, faces = self.entities[0].get_link_vert_face(link_name)
            vertices_tensor = torch.as_tensor(
                verts, dtype=torch.float32, device=self.device
            )
            faces_tensor = torch.as_tensor(faces, dtype=torch.int32, device=self.device)
            link_vert_face[link_name] = (vertices_tensor, faces_tensor)
        return link_vert_face


class Articulation(BatchEntity):
    """Articulation represents a batch of articulations in the simulation.

    An articulation is a collection of rigid bodies connected by joints. The joints can be either
    fixed or actuated. The joints can be of different types, such as revolute or prismatic.

    For fixed-base articulation, it can be a robot arm, door, etc.
    For floating-base articulation, it can be a humanoid, drawer, etc.

    Args:
        cfg: Configuration for the articulation.
        device: Device to use (CPU or CUDA).
    """

    def __init__(
        self,
        cfg: ArticulationCfg,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Create an unregistered articulation facade.

        ``SpawnScene`` owns the replicated instance count and the finalized
        ``Scene``. It injects them at declaration and binding time instead of
        exposing either lifecycle dependency through this constructor.
        """
        self._newton_mimic_compliance_configured = False
        self._prepared_default_root_topology_revision = -1
        self.cfg = deepcopy(cfg)
        self.uid = self.cfg.uid
        self.device = device
        self._entities: list[SpawnedArticulation] = []
        self._declared_num_instances: int | None = None
        self._spawn_result: Scene | None = None
        self._world = None
        self._ps = None
        self._data: ArticulationData | None = None
        self._all_indices = torch.empty(0, dtype=torch.int32)
        self._visual_material: List[Dict[str, VisualMaterialInst]] = []
        self.is_shared_visual_material = False
        self._has_collision_visible_node_dict: dict[str, bool] = {}

    def _initialize_spawn_declaration(self, num_instances: int) -> None:
        """Initialize instance-dependent declaration state from ``SpawnScene``."""
        if num_instances <= 0:
            raise ValueError("A declared Articulation requires num_instances > 0.")
        if self._declared_num_instances is not None:
            if self._declared_num_instances != num_instances:
                raise RuntimeError(
                    f"Articulation {self.uid!r} is already declared for "
                    f"{self._declared_num_instances} instances."
                )
            return

        self._declared_num_instances = num_instances
        self._all_indices = torch.arange(num_instances, dtype=torch.int32)
        self._visual_material = [{} for _ in range(num_instances)]

    def _require_declared_num_instances(self) -> int:
        """Return the Spawn-provided instance count or raise a lifecycle error."""
        if self._declared_num_instances is None:
            raise RuntimeError(
                f"Articulation {self.uid!r} must be registered through SpawnScene "
                "before it can be used."
            )
        return self._declared_num_instances

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade is bound to one finalized Scene."""
        return self._spawn_result is not None

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for its Scene binding."""
        return self._world is None

    @property
    def num_instances(self) -> int:
        if self._entities:
            return len(self._entities)
        return self._require_declared_num_instances()

    def attach_spawn_handles(
        self,
        entities: Sequence[SpawnedArticulation],
    ) -> None:
        """Store handles and expose metadata without initializing Batch data.

        This pre-finalize step supports eager Default loading and only reads
        articulation metadata. ``bind_spawn()`` performs result-dependent
        Batch/Data initialization after finalization.
        """
        handles = list(entities)
        expected = self._require_declared_num_instances()
        if len(handles) != expected:
            raise ValueError(
                f"Articulation {self.uid!r} expected "
                f"{expected} Spawn handles, got {len(handles)}."
            )
        self._entities = handles
        self._mimic_info = self._state_mimic_info()
        self.active_joint_ids = [
            index for index in range(self.dof) if index not in self.mimic_ids
        ]

    def _clear_spawn_cached_properties(self) -> None:
        """Drop metadata cached against declaration-time handles."""
        for name in (
            "dof",
            "active_dof",
            "num_links",
            "link_names",
            "user_ids",
            "root_link_name",
            "joint_names",
            "active_joint_names",
            "all_joint_names",
        ):
            self.__dict__.pop(name, None)

    def _initialize_spawn_bound(self, result: Scene) -> None:
        """Create result-dependent runtime state on this declared facade."""
        if not isinstance(result, Scene):
            raise TypeError(
                "Articulation binding requires a finalized DexSim Scene; use "
                "SimulationManager.prepare()."
            )

        entities = list(self._entities)
        expected = self._require_declared_num_instances()
        if len(entities) != expected:
            raise ValueError(
                f"Articulation {self.uid!r} expected {expected} Spawn handles, "
                f"got {len(entities)}."
            )

        cfg = deepcopy(self.cfg)
        self._clear_spawn_cached_properties()
        self._newton_mimic_compliance_configured = False
        self._spawn_result = result
        self._world = result.world
        self._ps = None
        self.cfg = cfg
        self._entities = entities
        self._all_indices = torch.arange(len(entities), dtype=torch.int32)
        self._data = ArticulationData(
            entities=entities,
            scene=result,
            device=self.device,
        )

        if self.cfg.init_qpos is None:
            self.cfg.init_qpos = torch.zeros(self.dof, dtype=torch.float32)

        self._capture_default_physical_properties()
        self.default_joint_stiffness = self._data.joint_stiffness.clone()
        self.default_joint_damping = self._data.joint_damping.clone()
        self.default_joint_friction = self._data.joint_friction.clone()
        self.default_joint_armature = self._data.joint_armature.clone()
        self.default_joint_max_effort = self._data.qf_limits.clone()
        self.default_joint_max_velocity = self._data.qvel_limits.clone()

        is_usd_source = str(self.cfg.fpath).lower().endswith((".usd", ".usda", ".usdc"))
        self.pk_chain = None
        if self.cfg.build_pk_chain and not is_usd_source:
            self.pk_chain = create_pk_chain(
                urdf_path=self.cfg.fpath, device=self.device
            )
        elif self.cfg.build_pk_chain:
            logger.log_warning(
                f"Articulation {self.uid!r} uses USD for simulation; skipping "
                "the URDF-only pk_chain. Configure a solver with its matching "
                "URDF when kinematics are required."
            )

        self._visual_material = [{} for _ in range(len(entities))]
        self.is_shared_visual_material = False
        self._mimic_info = self._state_mimic_info()
        self.active_joint_ids = [i for i in range(self.dof) if i not in self.mimic_ids]

        super().__init__(cfg, entities, self.device)
        self._initialize_existing_visual_material()
        self._has_collision_visible_node_dict = {
            link_name: False for link_name in self.link_names
        }
        self._initialize_spawn_bound_extension()

    def _initialize_spawn_bound_extension(self) -> None:
        """Initialize subclass state after the Scene batch becomes available."""

    def bind_spawn(
        self,
        result: Scene,
    ) -> None:
        """Initialize this declared facade from Spawn articulation handles."""
        if self.is_spawn_bound:
            raise RuntimeError(f"Articulation {self.uid!r} is already Spawn-bound.")
        if not self.is_declared:
            raise RuntimeError(
                f"Articulation {self.uid!r} was not created as a Spawn declaration."
            )

        declared_state = self.__dict__.copy()
        try:
            self._initialize_spawn_bound(result)
            self._apply_spawn_config()
            if is_newton_gradient_mode(result):
                initial_qpos = torch.as_tensor(self.cfg.init_qpos).reshape(-1)
                if initial_qpos.numel() != self.dof:
                    raise ValueError(
                        f"Articulation {self.uid!r} expected {self.dof} initial "
                        f"joint positions, got {initial_qpos.numel()}."
                    )
                if torch.any(initial_qpos != 0.0):
                    raise NotImplementedError(
                        "Newton gradient mode cannot apply non-zero init_qpos after "
                        "Spawn finalization. Author the initial coordinates in the "
                        "source asset or initialize them in a differentiable task "
                        "before opening a Warp tape."
                    )
                # Spawn already authored the root pose and zero joint/dynamics
                # state during model construction. Its Batch mutation APIs are
                # intentionally fenced once the model requires gradients.
            else:
                self.reset()
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(declared_state)
            raise

    def _apply_spawn_config(self) -> None:
        """Apply configuration that requires finalized backend resources.

        Link physics and joint-drive regex selection is resolved by
        EmbodiChain against the source descriptor before finalization. Default
        articulation-root properties are normally handled by the pre-runtime
        hook; calling it here keeps direct facade binding safe. Render
        operations also require materialized native resources.
        """
        spawn_result = getattr(self, "_spawn_result", None)
        self._prepare_spawn_runtime_config(spawn_result)

        self._newton_mimic_compliance_configured = _configure_newton_mimic_compliance(
            result=spawn_result,
            entities=self._entities,
            state_joint_names=self._state_joint_names(),
            mimic_ids=self.mimic_ids,
            mimic_parents=self.mimic_parents,
        )

        if not self.cfg.compute_uv:
            return

        for entity in self._entities:
            for link_name in self.link_names:
                render_body = entity.get_render_body(link_name)
                if render_body is not None:
                    render_body.set_projective_uv()

    def _prepare_spawn_runtime_config(self, result: Scene | None) -> None:
        """Apply Default root properties before Direct GPU initialization.

        PhysX snapshots articulation solver iteration counts when the Direct
        GPU runtime is initialized. Applying these values only during facade
        binding is too late because ``World.init_gpu_physics()`` has already
        performed its warm-up steps. CPU simulation accepts the late write,
        which otherwise makes identical hand mimic constraints substantially
        softer on CUDA.
        """
        if result is None or getattr(result, "backend", None) != "dexsim":
            return

        topology_revision = int(result.topology_revision)
        if self._prepared_default_root_topology_revision == topology_revision:
            return

        root_props = getattr(self.cfg, "root_props", None)
        default_root_values_configured = root_props is not None and (
            root_props.sleep_threshold is not None
            or root_props.min_position_iters is not None
            or root_props.min_velocity_iters is not None
        )
        if default_root_values_configured:
            for entity in self._entities:
                # SpawnedArticulation deliberately fences these setters, while
                # its Default-native binding exposes the articulation-root API.
                native_articulation = getattr(entity, "_physics_binding", None)
                if native_articulation is None:
                    raise RuntimeError(
                        "Default Spawn articulation has no native physics binding."
                    )
                _apply_default_articulation_root_properties(
                    native_articulation,
                    root_props,
                )
        self._prepared_default_root_topology_revision = topology_revision

    def __str__(self) -> str:
        if self.is_declared:
            parent_str = (
                f"{self.__class__}: declared {self.num_instances} Spawn "
                f"articulations | uid: {self.uid} | device: {self.device}"
            )
            return parent_str
        parent_str = super().__str__()
        return parent_str + f" | dof: {self.dof} | num_links: {self.num_links}"

    @cached_property
    def dof(self) -> int:
        """Get the degree of freedom of the articulation.

        Returns:
            int: The degree of freedom of the articulation.
        """
        if self._data is not None:
            return self._data.dof
        return self._entities[0].get_dof()

    @cached_property
    def active_dof(self) -> int:
        """Get the number of active degrees of freedom of the articulation.

        Returns:
            int: The number of active degrees of freedom of the articulation.
        """
        return len(self.active_joint_ids)

    @cached_property
    def num_links(self) -> int:
        """Get the number of links in the articulation.

        Returns:
            int: The number of links in the articulation.
        """
        if self._data is not None:
            return self._data.num_links
        return len(self._entities[0].get_link_names())

    @cached_property
    def link_names(self) -> List[str]:
        """Get the names of the links in the articulation.

        Returns:
            List[str]: The names of the links in the articulation.
        """
        if self._data is not None:
            return self._data.link_names
        return self._entities[0].get_link_names()

    @cached_property
    def user_ids(self) -> torch.Tensor:
        """Get the user-defined IDs of the articulation.

        Note:
            The return tensor has shape (num_instances, num_links), where each column corresponds to a link in the articulation.

        Returns:
            torch.Tensor: The user-defined IDs of the articulation with shape (num_instances, num_links).
        """
        user_ids = torch.zeros(
            (self.num_instances, self.num_links), dtype=torch.int32, device=self.device
        )
        for i, entity in enumerate(self._entities):
            for j, link_name in enumerate(self.link_names):
                user_ids[i, j] = entity.get_user_ids(link_name)[0]
        return user_ids

    @cached_property
    def root_link_name(self) -> str:
        """Get the name of the root link of the articulation.

        Returns:
            str: The name of the root link.
        """
        return self.entities[0].get_root_link_name()

    @cached_property
    def joint_names(self) -> List[str]:
        """Get active joint names in public qpos-buffer order.

        Returns:
            List[str]: Active joint names aligned with qpos, qvel, and qf.
        """
        if getattr(self, "_data", None) is not None:
            return list(self._data.articulation_view.joint_names)
        return self._state_joint_names()

    def _state_joint_names(self) -> List[str]:
        """Return active joint names in the backing qpos-buffer order.

        The Scene's Newton batch layout may differ from its source articulation
        order.  Joint IDs sent to the batch must therefore use the layout
        order. :attr:`joint_names` exposes this same order; query the Spawn
        handle directly only for source-topology resolution.
        """
        if not self._entities:
            return []
        entity = self._entities[0]
        try:
            layout = entity.joint_dof_layout
        except (AttributeError, RuntimeError):
            return entity.get_actived_joint_names()
        return [joint.name for joint in layout]

    def _source_qpos_to_state_order(self, qpos: torch.Tensor) -> torch.Tensor:
        """Map source-ordered initial qpos values to the runtime state order."""
        if not self.is_spawn_bound:
            return qpos

        source_joint_names = self._entities[0].get_actived_joint_names()
        state_joint_names = self._state_joint_names()
        if source_joint_names == state_joint_names:
            return qpos

        source_indices = {name: index for index, name in enumerate(source_joint_names)}
        try:
            state_order = [source_indices[name] for name in state_joint_names]
        except KeyError as error:
            raise RuntimeError(
                "Spawn articulation state layout contains a joint absent from "
                "the source articulation layout."
            ) from error
        return qpos[..., state_order]

    def _state_mimic_info(self) -> _MimicInfo:
        """Map source-articulation mimic indices to state-buffer indices."""
        entity = self._entities[0]
        source_info = entity.get_mimic_info()
        source_mimic_ids = np.asarray(source_info.mimic_id, dtype=np.int32).reshape(-1)
        source_parent_ids = np.asarray(
            source_info.mimic_parent, dtype=np.int32
        ).reshape(-1)
        multipliers = np.asarray(
            source_info.mimic_multiplier, dtype=np.float32
        ).reshape(-1)
        offsets = np.asarray(source_info.mimic_offset, dtype=np.float32).reshape(-1)
        relation_count = len(source_mimic_ids)
        if not all(
            len(values) == relation_count
            for values in (source_parent_ids, multipliers, offsets)
        ):
            raise RuntimeError("Articulation mimic metadata has inconsistent lengths.")
        if relation_count == 0:
            return _MimicInfo(
                mimic_id=source_mimic_ids,
                mimic_parent=source_parent_ids,
                mimic_multiplier=multipliers,
                mimic_offset=offsets,
            )

        source_joint_names = entity.get_actived_joint_names()
        try:
            state_joint_ids = {
                joint.name: int(joint.dof_start) for joint in entity.joint_dof_layout
            }
        except (AttributeError, RuntimeError):
            state_joint_ids = {
                name: index for index, name in enumerate(source_joint_names)
            }

        try:
            mimic_ids = np.asarray(
                [
                    state_joint_ids[source_joint_names[int(source_id)]]
                    for source_id in source_mimic_ids
                ],
                dtype=np.int32,
            )
            parent_ids = np.asarray(
                [
                    state_joint_ids[source_joint_names[int(source_id)]]
                    for source_id in source_parent_ids
                ],
                dtype=np.int32,
            )
        except (IndexError, KeyError) as error:
            raise RuntimeError(
                "Articulation mimic metadata references a joint absent from "
                "the backing state layout."
            ) from error

        return _MimicInfo(
            mimic_id=mimic_ids,
            mimic_parent=parent_ids,
            mimic_multiplier=multipliers,
            mimic_offset=offsets,
        )

    def _project_mimic_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        """Return qpos with every mimic child projected from its parent."""
        if not self.mimic_ids:
            return qpos

        projected = qpos.clone()
        mimic_ids = torch.as_tensor(
            self.mimic_ids, dtype=torch.long, device=qpos.device
        )
        parent_ids = torch.as_tensor(
            self.mimic_parents, dtype=torch.long, device=qpos.device
        )
        multipliers = torch.as_tensor(
            self.mimic_multipliers, dtype=qpos.dtype, device=qpos.device
        )
        offsets = torch.as_tensor(
            self.mimic_offsets, dtype=qpos.dtype, device=qpos.device
        )
        projected[..., mimic_ids] = projected[..., parent_ids] * multipliers + offsets
        return projected

    def _stabilize_newton_mimic_target_write(
        self,
        values: torch.Tensor,
        env_ids: torch.Tensor,
        joint_ids: torch.Tensor,
        *,
        velocity: bool,
    ) -> None:
        """Update weak follower-drive targets for written mimic leaders.

        The native Newton equality remains the physical coupling. This only
        keeps its low-gain follower stabilizer pointed at the same commanded
        relation; it never copies measured qpos or qvel into follower state.
        """
        if not self._newton_mimic_compliance_configured:
            return

        selected_columns = {
            int(joint_id): column
            for column, joint_id in enumerate(joint_ids.detach().cpu().tolist())
        }
        follower_ids: list[int] = []
        follower_targets: list[torch.Tensor] = []
        for child_id, parent_id, multiplier, offset in zip(
            self.mimic_ids,
            self.mimic_parents,
            self.mimic_multipliers,
            self.mimic_offsets,
            strict=True,
        ):
            parent_column = selected_columns.get(int(parent_id))
            if parent_column is None:
                continue
            target = values[:, parent_column] * float(multiplier)
            if not velocity:
                target = target + float(offset)
            follower_ids.append(int(child_id))
            follower_targets.append(target)

        if not follower_ids:
            return

        targets = torch.stack(follower_targets, dim=1)
        follower_ids_tensor = torch.as_tensor(
            follower_ids, dtype=torch.int32, device=self.device
        )
        if velocity:
            limits = self.body_data.qvel_limits[env_ids][:, follower_ids_tensor]
            targets = targets.clamp(-limits, limits)
            self._data.articulation_view.apply_qvel(
                targets,
                env_ids,
                follower_ids_tensor,
                target=True,
            )
            return

        limits = self.body_data.qpos_limits[env_ids][:, follower_ids_tensor, :]
        targets = targets.clamp(limits[..., 0], limits[..., 1])
        self._data.articulation_view.apply_qpos(
            targets,
            env_ids,
            follower_ids_tensor,
            target=True,
        )

    @cached_property
    def active_joint_names(self) -> List[str]:
        """Get the names of the active joints in the articulation.

        Returns:
            List[str]: The names of the active joints in the articulation.
        """
        state_joint_names = self._state_joint_names()
        return [state_joint_names[i] for i in self.active_joint_ids]

    @cached_property
    def all_joint_names(self) -> List[str]:
        """Get the names of the joints in the articulation.

        Returns:
            List[str]: The names of the joints in the articulation.
        """
        return self._entities[0].get_joint_names()

    def get_parent_joint_chain(
        self,
        link_name: str,
    ) -> tuple[ArticulationJointKinematics, ...]:
        """Return the joints from a link toward the articulation root.

        The immediate parent joint is first. Native simulator joint-info values
        are copied into :class:`ArticulationJointKinematics`, keeping callers
        independent of DexSim objects and the private entity collection.

        Args:
            link_name: Link whose parent chain should be queried.

        Returns:
            Parent-joint chain ordered from the requested link toward the root.

        Raises:
            TypeError: If ``link_name`` is not a string.
            ValueError: If the link is unknown, native topology is incomplete,
                multiple joints own one child link, or the chain contains a
                cycle.
        """
        if type(link_name) is not str:
            raise TypeError("link_name must be a string.")
        if not link_name or link_name != link_name.strip():
            raise ValueError("link_name must be a non-empty link name.")
        if link_name not in self.link_names:
            raise ValueError(
                f"Unknown articulation link {link_name!r}. Available links: "
                f"{list(self.link_names)}."
            )

        entity = self._entities[0]
        joints_by_child: dict[str, ArticulationJointKinematics] = {}
        for joint_name in entity.get_joint_names():
            native = entity.get_joint_info(joint_name)
            if native is None:
                raise ValueError(
                    f"Native articulation has no joint info for {joint_name!r}."
                )
            native_joint_type = getattr(
                native.joint_type,
                "name",
                native.joint_type,
            )
            lower_limit = getattr(native, "lower_limit", None)
            upper_limit = getattr(native, "upper_limit", None)
            joint_limits = (
                None
                if lower_limit is None or upper_limit is None
                else (float(lower_limit), float(upper_limit))
            )
            joint = ArticulationJointKinematics(
                name=native.name,
                joint_type=str(native_joint_type),
                parent_link_name=native.parent_link_name,
                child_link_name=native.child_link_name,
                origin_pose=torch.as_tensor(native.origin_pose),
                axis=torch.as_tensor(native.axis),
                joint_limits=joint_limits,
            )
            if joint.child_link_name in joints_by_child:
                raise ValueError(
                    "Articulation topology contains multiple parent joints for "
                    f"child link {joint.child_link_name!r}."
                )
            joints_by_child[joint.child_link_name] = joint

        chain: list[ArticulationJointKinematics] = []
        current_link = link_name
        visited_links: set[str] = set()
        while current_link in joints_by_child:
            if current_link in visited_links:
                raise ValueError(
                    f"Articulation parent chain for {link_name!r} contains a cycle."
                )
            visited_links.add(current_link)
            joint = joints_by_child[current_link]
            chain.append(joint)
            current_link = joint.parent_link_name
        return tuple(chain)

    @property
    def body_data(self) -> ArticulationData:
        """Get the rigid body data manager for this rigid object.

        Returns:
            RigidBodyData: The rigid body data manager.
        """
        return self._data

    @property
    def default_link_masses(self) -> torch.Tensor:
        """Initialization-time link masses retained for compatibility."""
        return self.body_data.default_mass

    def _capture_default_physical_properties(self) -> None:
        """Capture materialized link mass properties as reset defaults."""
        if self._data.default_physical_properties_initialized:
            return
        mass, inertia, com_pose = self._data.read_physical_properties()
        self._data.capture_default_physical_properties(
            mass=mass,
            inertia=inertia,
            com_pose=com_pose,
        )

    def _resolve_link_names(
        self, link_names: str | Sequence[str] | None
    ) -> tuple[list[str], torch.Tensor]:
        """Validate link names and return their public data-column indices."""
        names = (
            list(self.link_names)
            if link_names is None
            else [link_names] if isinstance(link_names, str) else list(link_names)
        )
        unknown = [name for name in names if name not in self.link_names]
        if unknown:
            raise ValueError(
                f"Unknown articulation links {unknown}; available links: "
                f"{self.link_names}."
            )
        indices = torch.as_tensor(
            [self.link_names.index(name) for name in names],
            dtype=torch.long,
            device=self.device,
        )
        return names, indices

    def _restore_default_physical_properties(
        self, env_ids: Sequence[int] | torch.Tensor
    ) -> None:
        """Restore initialization-time link mass properties for selected rows."""
        if not self._data.default_physical_properties_initialized or len(env_ids) == 0:
            return

        env_index = self._resolve_env_ids(env_ids)
        env_list = env_index.detach().cpu().tolist()
        default_mass = self._data.default_mass[env_index]
        default_inertia = self._data.default_inertia[env_index]
        default_com_pose = self._data.default_com_pose[env_index]
        current_mass, current_inertia, current_com_pose = (
            value[env_index] for value in self._data.read_physical_properties()
        )

        mass_changed = not torch.allclose(current_mass, default_mass)
        inertia_changed = not torch.allclose(current_inertia, default_inertia)
        if mass_changed:
            self.set_mass(default_mass, link_names=self.link_names, env_ids=env_list)
        if mass_changed or inertia_changed:
            self.set_inertia(
                default_inertia,
                link_names=self.link_names,
                env_ids=env_list,
            )
        if not torch.allclose(current_com_pose, default_com_pose):
            self.set_com_pose(
                default_com_pose,
                link_names=self.link_names,
                env_ids=env_list,
            )

    @property
    def root_state(self) -> torch.Tensor:
        """Get the root state of the articulation.

        Returns:
            torch.Tensor: The root state of the articulation with shape (N, 13).
        """
        root_pose = self.body_data.root_pose
        root_lin_vel = self.body_data.root_lin_vel
        root_ang_vel = self.body_data.root_ang_vel
        return torch.cat((root_pose, root_lin_vel, root_ang_vel), dim=-1)

    @property
    def body_state(self) -> torch.Tensor:
        """Get the body state of the articulation.

        Returns:
            torch.Tensor: The body state of the articulation with shape (N, num_links, 13).
        """
        body_pose = self.body_data.body_link_pose
        body_vel = self.body_data.body_link_vel
        return torch.cat((body_pose, body_vel), dim=-1)

    @property
    def mimic_ids(self) -> List[int | None]:
        """Get the mimic joint ids for the articulation.

        Returns:
            List[int | None]: The mimic joint ids.
        """
        return self._mimic_info.mimic_id.tolist()

    @property
    def mimic_parents(self) -> List[int | None]:
        """Get the mimic joint parent ids for the articulation.

        Returns:
            List[int | None]: The mimic joint parent ids.
        """
        return self._mimic_info.mimic_parent.tolist()

    @property
    def mimic_multipliers(self) -> List[float]:
        """Get the mimic joint multipliers for the articulation.

        Returns:
            List[float]: The mimic joint multipliers.
        """
        return self._mimic_info.mimic_multiplier.tolist()

    @property
    def mimic_offsets(self) -> List[float]:
        """Get the mimic joint offsets for the articulation.

        Returns:
            List[float]: The mimic joint offsets.
        """
        return self._mimic_info.mimic_offset.tolist()

    def _set_default_collision_filter(self) -> None:
        collision_filter_data = torch.zeros(
            size=(self.num_instances, 4), dtype=torch.int32
        )
        for i in range(self.num_instances):
            collision_filter_data[i, 0] = i
            collision_filter_data[i, 1] = 1
        self.set_collision_filter(collision_filter_data)

    def _resolve_env_ids(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ) -> torch.Tensor:
        """Resolve environment ids to a device tensor."""
        if env_ids is None:
            return torch.arange(
                self.num_instances, dtype=torch.long, device=self.device
            )
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _resolve_joint_ids(
        self, joint_ids: Sequence[int] | torch.Tensor | None
    ) -> torch.Tensor:
        """Resolve joint ids to a device tensor."""
        if joint_ids is None:
            return torch.arange(self.dof, dtype=torch.long, device=self.device)
        if isinstance(joint_ids, torch.Tensor):
            return joint_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)

    def set_collision_filter(
        self, filter_data: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """set collision filter data for the rigid object.

        Args:
            filter_data (torch.Tensor): [N, 4] of int.
                First element of each object is arena id.
                If 2nd element is 0, the object will collision with all other objects in world.
                3rd and 4th elements are not used currently.

            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used. Defaults to None.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(filter_data):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match pose length {len(filter_data)}."
            )

        filter_data_np = filter_data.cpu().numpy().astype(np.uint32)
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].set_collision_filter_data(filter_data_np[i])

    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set local pose of the articulation.

        Args:
            pose (torch.Tensor): The local pose of the articulation with shape (N, 7) or (N, 4, 4).
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if len(local_env_ids) != len(pose):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match pose length {len(pose)}."
            )

        if pose.dim() == 2 and pose.shape[1] == 7:
            target_pose = pose.to(device=self.device, dtype=torch.float32)
        elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
            xyz = pose[:, :3, 3]
            quat = quat_from_matrix(pose[:, :3, :3])
            target_pose = torch.cat((xyz, quat), dim=-1).to(
                device=self.device, dtype=torch.float32
            )
        else:
            logger.log_error(
                f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
            )
            return

        self._data.articulation_view.apply_root_pose(target_pose, local_env_ids)
        if self.device.type == "cpu" and not self._data.is_newton_backend:
            self._world.update(0.001)

    def get_local_pose(self, to_matrix=False) -> torch.Tensor:
        """Get local pose (root link pose) of the articulation.

        Args:
            to_matrix (bool, optional): If True, return the pose as a 4x4 matrix. If False, return as (x, y, z, qx, qy, qz, qw). Defaults to False.

        Returns:
            torch.Tensor: The local pose of the articulation with shape (N, 7) or (N, 4, 4) depending on `to_matrix`.
        """
        pose = self.body_data.root_pose
        if to_matrix:
            xyz = pose[:, :3]
            mat = matrix_from_quat(pose[:, 3:7])
            pose = (
                torch.eye(4, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(pose.shape[0], 1, 1)
            )
            pose[:, :3, 3] = xyz
            pose[:, :3, :3] = mat
        return pose

    def get_link_vert_face(self, link_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the vertices and faces of a specific link in the articulation.

        Args:
            link_name (str): The name of the link.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - vertices (torch.Tensor): The vertices of the specified link with shape (V, 3).
                - faces (torch.Tensor): The faces of the specified link with shape (F, 3).
        """
        if link_name not in self.link_names:
            logger.log_error(
                f"Link name {link_name} not found in {self.__class__.__name__}. Available links: {self.link_names}"
            )

        verts, faces = self.body_data.link_vert_face[link_name]
        return verts, faces

    def get_link_pose(
        self, link_name: str, env_ids: Sequence[int] | None = None, to_matrix=False
    ) -> torch.Tensor:
        """Get the pose of a specific link in the articulation.

        Args:
            link_name (str): The name of the link.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
            to_matrix (bool, optional): If True, return the pose as a 4x4 matrix. If False, return as (x, y, z, qx, qy, qz, qw). Defaults to False.

        Returns:
            torch.Tensor: The pose of the specified link with shape (N, 7) or (N, 4, 4) depending on `to_matrix`.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if link_name not in self.link_names:
            logger.log_error(
                f"Link name {link_name} not found in {self.__class__.__name__}. Available links: {self.link_names}"
            )

        link_idx = self.link_names.index(link_name)
        link_pose = self.body_data.body_link_pose[local_env_ids, link_idx, :]

        if to_matrix:
            xyz = link_pose[:, :3]
            mat = matrix_from_quat(link_pose[:, 3:7])
            link_pose = (
                torch.eye(4, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(link_pose.shape[0], 1, 1)
            )
            link_pose[:, :3, 3] = xyz
            link_pose[:, :3, :3] = mat
        return link_pose

    def get_qpos(self, target: bool = False) -> torch.Tensor:
        """Get the current positions (qpos) or target positions (target_qpos) of the articulation.

        Args:
            target (bool): If True, gets target positions for simulation. If False, gets current positions.

        Returns:
            torch.Tensor: Joint positions with shape (N, dof), where N is the number of environments.
        """
        return self.body_data.qpos if not target else self.body_data.target_qpos

    def get_qpos_limits(
        self,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get joint position limits for selected environments and joints.

        Args:
            joint_ids: Joint indices to query. If None, all joints are queried.
            env_ids: Environment indices to query. If None, all environments are
                queried.

        Returns:
            torch.Tensor: Joint position limits with shape (num_envs, num_joints, 2).
        """
        local_env_ids = self._resolve_env_ids(env_ids)
        local_joint_ids = self._resolve_joint_ids(joint_ids)
        return self.body_data.qpos_limits[local_env_ids][:, local_joint_ids, :]

    def _coerce_pair_limit_batch(
        self,
        values: torch.Tensor | np.ndarray,
        local_env_ids: torch.Tensor,
        local_joint_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize batched pair-valued limits to ``(num_envs, num_joints, 2)``."""
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if values.dim() == 2 and len(local_env_ids) == 1:
            values = values.unsqueeze(0)
        expected_shape = (len(local_env_ids), len(local_joint_ids), 2)
        if tuple(values.shape) != expected_shape:
            logger.log_error(
                f"Expected qpos limit shape {expected_shape}, got {tuple(values.shape)}."
            )
        return values

    def _coerce_scalar_limit_batch(
        self,
        values: torch.Tensor | np.ndarray,
        local_env_ids: torch.Tensor,
        local_joint_ids: torch.Tensor,
        limit_name: str,
    ) -> torch.Tensor:
        """Normalize batched scalar limits to ``(num_envs, num_joints)``."""
        values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if values.dim() == 1 and len(local_env_ids) == 1:
            values = values.unsqueeze(0)
        expected_shape = (len(local_env_ids), len(local_joint_ids))
        if tuple(values.shape) != expected_shape:
            logger.log_error(
                f"Expected {limit_name} shape {expected_shape}, got {tuple(values.shape)}."
            )
        return values

    def set_qpos_limits(
        self,
        qpos_limits: torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set joint position limits for selected environments and joints.

        Args:
            qpos_limits: Joint position limits with shape (num_envs, num_joints, 2).
                When a single environment is selected, a (num_joints, 2) tensor is also accepted.
            joint_ids: Joint indices to update. If None, all joints are updated.
            env_ids: Environment indices to update. If None, all environments are updated.
        """
        local_env_ids = self._resolve_env_ids(env_ids)
        local_joint_ids = self._resolve_joint_ids(joint_ids)
        qpos_limits = self._coerce_pair_limit_batch(
            qpos_limits, local_env_ids, local_joint_ids
        )
        joint_ids_np = (
            local_joint_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        )

        failed_envs = []
        for i, env_idx in enumerate(local_env_ids.detach().cpu().tolist()):
            result = self._entities[env_idx].set_joint_position_limits(
                qpos_limits[i].detach().cpu().numpy(),
                joint_ids_np,
            )
            if result == -1:
                failed_envs.append(env_idx)
                continue
            self.body_data.qpos_limits[env_idx, local_joint_ids, :] = qpos_limits[i]

        if failed_envs:
            logger.log_error(
                f"set_joint_position_limits failed for envs {failed_envs} and joint_ids {joint_ids_np.tolist()}."
            )

    def set_qvel_limits(
        self,
        qvel_limits: torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set joint velocity limits for selected environments and joints.

        Args:
            qvel_limits: Joint velocity limits with shape (num_envs, num_joints).
                When a single environment is selected, a (num_joints,) tensor is also accepted.
            joint_ids: Joint indices to update. If None, all joints are updated.
            env_ids: Environment indices to update. If None, all environments are updated.
        """
        local_env_ids = self._resolve_env_ids(env_ids)
        local_joint_ids = self._resolve_joint_ids(joint_ids)
        qvel_limits = self._coerce_scalar_limit_batch(
            qvel_limits, local_env_ids, local_joint_ids, "qvel limit"
        )
        joint_ids_np = (
            local_joint_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        )

        failed_envs = []
        for i, env_idx in enumerate(local_env_ids.detach().cpu().tolist()):
            result = self._entities[env_idx].set_joint_velocity_limit(
                qvel_limits[i].detach().cpu().numpy(),
                joint_ids_np,
            )
            if result == -1:
                failed_envs.append(env_idx)
                continue
            self.body_data.qvel_limits[env_idx, local_joint_ids] = qvel_limits[i]

        if failed_envs:
            logger.log_error(
                f"set_joint_velocity_limit failed for envs {failed_envs} and joint_ids {joint_ids_np.tolist()}."
            )

    def set_qf_limits(
        self,
        qf_limits: torch.Tensor,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set joint effort limits for selected environments and joints.

        Args:
            qf_limits: Joint effort limits with shape (num_envs, num_joints).
                When a single environment is selected, a (num_joints,) tensor is also accepted.
            joint_ids: Joint indices to update. If None, all joints are updated.
            env_ids: Environment indices to update. If None, all environments are updated.
        """
        local_env_ids = self._resolve_env_ids(env_ids)
        local_joint_ids = self._resolve_joint_ids(joint_ids)
        qf_limits = self._coerce_scalar_limit_batch(
            qf_limits, local_env_ids, local_joint_ids, "qf limit"
        )
        joint_ids_np = (
            local_joint_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        )

        failed_envs = []
        for i, env_idx in enumerate(local_env_ids.detach().cpu().tolist()):
            result = self._entities[env_idx].set_joint_effort_limit(
                qf_limits[i].detach().cpu().numpy(),
                joint_ids_np,
            )
            if result == -1:
                failed_envs.append(env_idx)
                continue
            self.body_data.qf_limits[env_idx, local_joint_ids] = qf_limits[i]

        if failed_envs:
            logger.log_error(
                f"set_joint_effort_limit failed for envs {failed_envs} and joint_ids {joint_ids_np.tolist()}."
            )

    def set_qpos(
        self,
        qpos: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        target: bool = True,
    ) -> None:
        """Set the joint positions (qpos) or target positions for the articulation.

        Args:
            qpos (torch.Tensor): Joint positions with shape (N, dof), where N is the number of environments.
            joint_ids (Sequence[int] | None, optional): Joint indices to apply the positions. If None, applies to all joints.
            env_ids (Sequence[int] | None): Environment indices to apply the positions. Defaults to all environments.
            target (bool): If True, sets target positions for simulation. If False, updates current positions directly.

        Raises:
            ValueError: If the length of `env_ids` does not match the length of `qpos`.
        """
        # TODO: Refactor this part to use a more generic and extensible approach,
        # such as a class decorator that can automatically convert ndarray to torch.Tensor
        # and handle dimension padding for specified member functions.
        # This will make the codebase cleaner and reduce repetitive type checks/conversions.
        # (e.g., support specifying which methods should be decorated for auto-conversion.)
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
        else:
            qpos = qpos.to(device=self.device, dtype=torch.float32)

        local_joint_ids = self._resolve_joint_ids(joint_ids)
        local_env_ids = self._resolve_env_ids(env_ids)

        # Make sure qpos is 2D tensor
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)

        if len(local_env_ids) != len(qpos):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match qpos length {len(qpos)}. "
                f"env_ids: {local_env_ids}, qpos.shape: {qpos.shape}"
            )

        selected_limits = self.body_data.qpos_limits[local_env_ids][
            :, local_joint_ids, :
        ]
        qpos = qpos.clamp(selected_limits[..., 0], selected_limits[..., 1])
        self._data.articulation_view.apply_qpos(
            qpos,
            local_env_ids,
            local_joint_ids,
            target=target,
        )
        if target:
            self._stabilize_newton_mimic_target_write(
                qpos,
                local_env_ids,
                local_joint_ids,
                velocity=False,
            )

    def get_qvel(self, target: bool = False) -> torch.Tensor:
        """Get the current velocities (qvel) or target velocities (target_qvel) of the articulation.

        Args:
            target (bool): If True, gets target velocities for simulation. If False, gets current velocities. The default is False.

        Returns:
            torch.Tensor: The current velocities of the articulation.
        """
        return self.body_data.qvel if not target else self.body_data.target_qvel

    def get_qvel_limits(
        self,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get joint velocity limits for selected environments and joints.

        Args:
            joint_ids: Joint indices to query. If None, all joints are queried.
            env_ids: Environment indices to query. If None, all environments are
                queried.

        Returns:
            torch.Tensor: Joint velocity limits with shape (num_envs, num_joints).
        """
        local_env_ids = self._resolve_env_ids(env_ids)
        local_joint_ids = self._resolve_joint_ids(joint_ids)
        return self.body_data.qvel_limits[local_env_ids][:, local_joint_ids]

    def set_qvel(
        self,
        qvel: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        target: bool = True,
    ) -> None:
        """Set the velocities (qvel) or target velocities of the articulation.

        Args:
            qvel (torch.Tensor): The velocities with shape (N, dof).
            joint_ids (Sequence[int] | None, optional): Joint indices to apply the velocities. If None, applies to all joints.
            env_ids (Sequence[int] | None, optional): Environment indices. Defaults to all indices.
            If True, sets target positions for simulation. If False, updates current positions directly.

        Raises:
            ValueError: If the length of `env_ids` does not match the length of `qvel`.
        """
        local_env_ids = self._resolve_env_ids(env_ids)

        if not isinstance(qvel, torch.Tensor):
            qvel = torch.as_tensor(qvel, dtype=torch.float32, device=self.device)
        else:
            qvel = qvel.to(device=self.device, dtype=torch.float32)

        if qvel.dim() == 1:
            qvel = qvel.unsqueeze(0)

        if len(local_env_ids) != len(qvel):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match qvel length {len(qvel)}."
            )

        local_joint_ids = self._resolve_joint_ids(joint_ids)

        self._data.articulation_view.apply_qvel(
            qvel,
            local_env_ids,
            local_joint_ids,
            target=target,
        )
        if target:
            self._stabilize_newton_mimic_target_write(
                qvel,
                local_env_ids,
                local_joint_ids,
                velocity=True,
            )

    def set_qf(
        self,
        qf: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set the generalized efforts (qf) of the articulation.

        Args:
            qf (torch.Tensor): The generalized efforts with shape (N, dof).
            joint_ids (Sequence[int] | None, optional): Joint indices to apply the efforts. If None, applies to all joints.
            env_ids (Sequence[int] | None, optional): Environment indices. Defaults to all indices.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if not isinstance(qf, torch.Tensor):
            qf = torch.as_tensor(qf, dtype=torch.float32, device=self.device)
        else:
            qf = qf.to(device=self.device, dtype=torch.float32)

        if qf.dim() == 1:
            qf = qf.unsqueeze(0)

        if len(local_env_ids) != len(qf):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match qf length {len(qf)}."
            )

        if joint_ids is None:
            local_joint_ids = torch.arange(
                self.dof, device=self.device, dtype=torch.int32
            )
        elif not isinstance(joint_ids, torch.Tensor):
            local_joint_ids = torch.as_tensor(
                joint_ids, dtype=torch.int32, device=self.device
            )
        else:
            local_joint_ids = joint_ids.to(device=self.device, dtype=torch.int32)

        self._data.articulation_view.apply_qf(qf, local_env_ids, local_joint_ids)

    def get_qf(self) -> torch.Tensor:
        """Get the current generalized efforts (qf) of the articulation.

        Returns:
            torch.Tensor: Joint efforts with shape (N, dof), where N is the
                number of environments.
        """
        return self.body_data.qf

    def get_qf_limits(
        self,
        joint_ids: Sequence[int] | torch.Tensor | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get joint effort limits for selected environments and joints.

        Args:
            joint_ids: Joint indices to query. If None, all joints are queried.
            env_ids: Environment indices to query. If None, all environments are
                queried.

        Returns:
            torch.Tensor: Joint effort limits with shape (num_envs, num_joints).
        """
        local_env_ids = self._resolve_env_ids(env_ids)
        local_joint_ids = self._resolve_joint_ids(joint_ids)
        return self.body_data.qf_limits[local_env_ids][:, local_joint_ids]

    def set_mass(
        self,
        mass: torch.Tensor,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set the mass of specific links in the articulation.

        Args:
            mass: Mass values with shape ``(num_envs, num_links)``.
            link_names: Link names to update. If None, all links are updated.
            env_ids: Environment indices. If None, all rows are updated.
        """
        env_index = self._resolve_env_ids(env_ids)
        env_list = env_index.detach().cpu().tolist()
        names, _ = self._resolve_link_names(link_names)
        mass = torch.as_tensor(mass, dtype=torch.float32, device=self.device)
        expected_shape = (len(env_list), len(names))
        if tuple(mass.shape) != expected_shape:
            raise ValueError(
                f"Expected mass shape {expected_shape}, got {tuple(mass.shape)}."
            )

        for i, env_idx in enumerate(env_list):
            entity = self._entities[env_idx]
            for j, name in enumerate(names):
                if self.is_spawn_bound or self._data.is_newton_backend:
                    entity.set_link_mass(name, mass[i, j].item())
                else:
                    entity.set_mass(name, mass[i, j].item())

    def get_mass(
        self,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get the mass of specific links in the articulation.

        Args:
            link_names: Link names to query. If None, all links are returned.
            env_ids: Environment indices. If None, all rows are returned.

        Returns:
            Selected link masses with shape ``(num_envs, num_links)``.
        """
        env_index = self._resolve_env_ids(env_ids)
        _, link_index = self._resolve_link_names(link_names)
        return self.body_data.mass[
            env_index[:, None],
            link_index[None, :],
        ]

    def set_inertia(
        self,
        inertia: torch.Tensor,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set principal moments of inertia for selected links."""
        env_index = self._resolve_env_ids(env_ids)
        env_list = env_index.detach().cpu().tolist()
        names, _ = self._resolve_link_names(link_names)
        inertia = torch.as_tensor(inertia, dtype=torch.float32, device=self.device)
        expected_shape = (len(env_list), len(names), 3)
        if tuple(inertia.shape) != expected_shape:
            raise ValueError(
                f"Expected inertia shape {expected_shape}, "
                f"got {tuple(inertia.shape)}."
            )

        values = inertia.detach().cpu().numpy()
        for i, env_idx in enumerate(env_list):
            entity = self._entities[env_idx]
            for j, name in enumerate(names):
                value = np.asarray(values[i, j], dtype=np.float32)
                if self.is_spawn_bound and self._data.is_newton_backend:
                    entity.set_newton_link_properties(
                        name,
                        rigid_body=dexsim.spawn.RigidBodyPhysicsDesc.dynamic(
                            inertia=value
                        ),
                    )
                elif not self._data.is_newton_backend:
                    entity.get_physical_body(name).set_mass_space_inertia_tensor(value)
                else:
                    attr = entity.get_physical_attr(name)
                    attr.inertia = value
                    entity.set_physical_attr(
                        attr,
                        name,
                        is_replace_inertial=False,
                    )

    def get_inertia(
        self,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get principal moments of inertia for selected links."""
        env_index = self._resolve_env_ids(env_ids)
        _, link_index = self._resolve_link_names(link_names)
        return self.body_data.inertia[
            env_index[:, None],
            link_index[None, :],
        ]

    def set_com_pose(
        self,
        com_pose: torch.Tensor,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        """Set local COM poses in EmbodiChain ``xyz + xyzw`` convention."""
        env_index = self._resolve_env_ids(env_ids)
        env_list = env_index.detach().cpu().tolist()
        names, _ = self._resolve_link_names(link_names)
        com_pose = torch.as_tensor(com_pose, dtype=torch.float32, device=self.device)
        expected_shape = (len(env_list), len(names), 7)
        if tuple(com_pose.shape) != expected_shape:
            raise ValueError(
                f"Expected COM pose shape {expected_shape}, "
                f"got {tuple(com_pose.shape)}."
            )

        values = com_pose.detach().cpu().numpy()
        for i, env_idx in enumerate(env_list):
            entity = self._entities[env_idx]
            for j, name in enumerate(names):
                position = np.asarray(values[i, j, :3], dtype=np.float32)
                quaternion = np.asarray(
                    convert_quat(values[i, j, 3:7], to="wxyz"),
                    dtype=np.float32,
                )
                if self.is_spawn_bound and self._data.is_newton_backend:
                    entity.set_newton_link_properties(
                        name,
                        rigid_body=dexsim.spawn.RigidBodyPhysicsDesc.dynamic(
                            com_position=position,
                            com_quaternion=quaternion,
                        ),
                    )
                elif not self._data.is_newton_backend:
                    entity.get_physical_body(name).set_cmass_local_pose(
                        position,
                        quaternion,
                    )
                else:
                    attr = entity.get_physical_attr(name)
                    attr.com_position = position
                    attr.com_quaternion = quaternion
                    entity.set_physical_attr(
                        attr,
                        name,
                        is_replace_inertial=False,
                    )

    def get_com_pose(
        self,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get local COM poses in EmbodiChain ``xyz + xyzw`` convention."""
        env_index = self._resolve_env_ids(env_ids)
        _, link_index = self._resolve_link_names(link_names)
        return self.body_data.com_pose[
            env_index[:, None],
            link_index[None, :],
        ]

    def get_link_physical_attr(
        self,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> list[PhysicalAttr]:
        """Get DexSim-native physical attributes for articulation links.

        Args:
            link_names: Link names or regex patterns. If None, all links are returned.
            env_ids: Environment indices. If None, only env 0 is queried.

        Returns:
            List of :class:`~dexsim.types.PhysicalAttr`, one per (env, link) pair in
            row-major order (env-major).
        """
        if self._data is not None and self._data.is_newton_backend:
            raise RuntimeError(
                "get_link_physical_attr() exposes DexSim PhysicalAttr semantics; "
                "use get_newton_link_properties() for Newton."
            )
        if link_names is None:
            matched_link_names = self.link_names
        elif isinstance(link_names, str):
            _, matched_link_names = resolve_matching_names(
                keys=link_names, list_of_strings=self.link_names
            )
        else:
            _, matched_link_names = resolve_matching_names(
                keys=link_names, list_of_strings=self.link_names
            )

        local_env_ids = [0] if env_ids is None else list(env_ids)
        attrs: list[PhysicalAttr] = []
        for env_idx in local_env_ids:
            entity = self._entities[env_idx]
            for name in matched_link_names:
                attrs.append(entity.get_physical_attr(name))
        return attrs

    def get_newton_link_properties(
        self,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> list[dexsim.spawn.RigidBodyPhysicsDesc]:
        """Get Newton model mass properties as typed Spawn descriptors.

        Args:
            link_names: Link names or regex patterns. If None, all links are
                returned.
            env_ids: Environment indices. If None, only environment 0 is
                queried.

        Returns:
            One typed descriptor per selected ``(environment, link)`` pair in
            environment-major order.
        """
        if not (
            self.is_spawn_bound
            and self._data is not None
            and self._data.is_newton_backend
        ):
            raise RuntimeError(
                "get_newton_link_properties() requires a Spawn-bound Newton "
                "articulation."
            )
        if link_names is None:
            matched_link_names = self.link_names
        else:
            _, matched_link_names = resolve_matching_names(
                keys=link_names,
                list_of_strings=self.link_names,
            )

        local_env_ids = [0] if env_ids is None else list(env_ids)
        properties = []
        for env_idx in local_env_ids:
            entity = self._entities[env_idx]
            for name in matched_link_names:
                properties.append(entity.get_newton_link_properties(name))
        return properties

    def set_link_physical_attr(
        self,
        attrs: RigidBodyPhysicsCfg | PhysicalAttr,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | None = None,
        *,
        base_attrs: RigidBodyPhysicsCfg | None = None,
        replace_inertial: bool = False,
    ) -> None:
        """Set physical attributes for selected articulation links.

        Args:
            attrs: Grouped or DexSim physical attributes to apply.
            link_names: Link names or regex patterns. If None, all links are updated.
            env_ids: Environment indices. If None, all environments are updated.
            base_attrs: Base config used when ``attrs`` is a partial override.
            replace_inertial: Recompute inertia when mass changes.

        .. attention::
            This compatibility API exposes DexSim ``PhysicalAttr`` semantics.
            Newton properties must use typed Spawn descriptors.
        """
        is_newton = self._data is not None and self._data.is_newton_backend
        if is_newton:
            raise TypeError(
                "set_link_physical_attr() is DexSim-only; use typed Newton "
                "link properties or set_mass()/set_inertia()/set_com_pose()."
            )

        if link_names is None:
            matched_link_names = self.link_names
        elif isinstance(link_names, str):
            _, matched_link_names = resolve_matching_names(
                keys=link_names, list_of_strings=self.link_names
            )
        else:
            _, matched_link_names = resolve_matching_names(
                keys=link_names, list_of_strings=self.link_names
            )

        if isinstance(attrs, RigidBodyPhysicsCfg):
            if base_attrs is None:
                base_attrs = self.cfg.attrs
            physical_attr = attrs.to_dexsim_physical_attr(
                base=base_attrs.to_dexsim_physical_attr()
            )
            mass_props = attrs.mass_props
            if mass_props is not None and mass_props.recompute_inertia is not None:
                replace_inertial = bool(mass_props.recompute_inertia)
        else:
            physical_attr = attrs

        local_env_ids = self._all_indices if env_ids is None else env_ids
        for env_idx in local_env_ids:
            entity = self._entities[env_idx]
            for name in matched_link_names:
                entity.set_physical_attr(
                    physical_attr,
                    name,
                    is_replace_inertial=replace_inertial,
                )

    def set_joint_drive(
        self,
        stiffness: torch.Tensor | None = None,
        damping: torch.Tensor | None = None,
        max_effort: torch.Tensor | None = None,
        max_velocity: torch.Tensor | None = None,
        friction: torch.Tensor | None = None,
        armature: torch.Tensor | None = None,
        drive_type: str | None = None,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        *,
        target_mode: str | int | None = None,
    ) -> None:
        """Set the drive properties for the articulation.

        Args:
            stiffness (torch.Tensor): The stiffness of the joint drive with shape (len(env_ids), len(joint_ids)).
            damping (torch.Tensor): The damping of the joint drive with shape (len(env_ids), len(joint_ids)).
            max_effort (torch.Tensor): The maximum effort of the joint drive with shape (len(env_ids), len(joint_ids)).
            max_velocity (torch.Tensor): The maximum velocity of the joint drive with shape (len(env_ids), len(joint_ids)).
            friction (torch.Tensor): The joint friction coefficient with shape (len(env_ids), len(joint_ids)).
            armature (torch.Tensor): The joint armature with shape (len(env_ids), len(joint_ids)).
            drive_type: ``force``, ``acceleration``, or ``none``. ``None``
                preserves the current mode unless a target mode activates a
                force drive.
            joint_ids (Sequence[int] | None, optional): The joint indices to apply the drive to. If None, applies to all joints. Defaults to None.
            env_ids (Sequence[int] | None, optional): The environment indices to apply the drive to. If None, applies to all environments. Defaults to None.
            target_mode: Portable target mode: ``none``, ``position``,
                ``velocity``, ``position_velocity``, ``effort``, or integer
                value 0 through 4.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        local_joint_ids = np.arange(self.dof) if joint_ids is None else joint_ids
        cache_env_ids = self._resolve_env_ids(env_ids)
        cache_joint_ids = self._resolve_joint_ids(joint_ids)

        mode_cfg = JointDrivePropertiesCfg(
            target_mode=target_mode,
            drive_type=drive_type,
        )
        resolved_target_mode, resolved_drive_type = mode_cfg._resolve_modes()
        if isinstance(resolved_target_mode, dict):
            raise TypeError(
                "set_joint_drive() accepts one scalar target_mode; configure "
                "per-joint mappings through JointDrivePropertiesCfg."
            )
        target_mode_value = (
            None
            if resolved_target_mode is None
            else _normalize_joint_target_mode(resolved_target_mode)
        )
        if target_mode_value in {1, 2, 3} and resolved_drive_type == "none":
            raise ValueError(
                "drive_type='none' conflicts with an active target_mode; use "
                "target_mode='none' or 'effort'."
            )

        def _drive_arg(value: torch.Tensor, index: int) -> float | np.ndarray:
            result = value[index].detach().cpu().numpy()
            return result.item() if result.size == 1 else result

        for i, env_idx in enumerate(local_env_ids):
            if self.is_spawn_bound and self.body_data.is_newton_backend:
                if resolved_drive_type == "acceleration" and target_mode_value in {
                    1,
                    2,
                    3,
                }:
                    raise NotImplementedError(
                        "Newton Spawn does not have an exact equivalent of "
                        "the Default acceleration drive. Use "
                        "drive_type='force' or disable the drive."
                    )
                drive_args = {"joint_ids": local_joint_ids}
                if target_mode_value is not None:
                    drive_args["target_mode"] = target_mode_value
                if stiffness is not None:
                    drive_args["target_ke"] = _drive_arg(stiffness, i)
                if damping is not None:
                    drive_args["target_kd"] = _drive_arg(damping, i)
                if max_effort is not None:
                    drive_args["effort_limit"] = _drive_arg(max_effort, i)
                if max_velocity is not None:
                    drive_args["velocity_limit"] = _drive_arg(max_velocity, i)
                if friction is not None:
                    drive_args["friction"] = _drive_arg(friction, i)
                if armature is not None:
                    drive_args["armature"] = _drive_arg(armature, i)
                if target_mode_value in {0, 4}:
                    drive_args["target_ke"] = 0.0
                    drive_args["target_kd"] = 0.0
                elif target_mode_value == 2:
                    drive_args["target_ke"] = 0.0
                self._entities[env_idx].set_newton_drive(**drive_args)
                continue

            drive_args = {"joint_ids": local_joint_ids}
            default_drive_type = resolved_drive_type
            if target_mode_value in {0, 4}:
                default_drive_type = "none"
            elif target_mode_value in {1, 2, 3} and default_drive_type is None:
                default_drive_type = "force"
            if default_drive_type is not None:
                drive_args["drive_type"] = get_dexsim_drive_type(default_drive_type)
            if stiffness is not None:
                drive_args["stiffness"] = _drive_arg(stiffness, i)
            if damping is not None:
                drive_args["damping"] = _drive_arg(damping, i)
            if max_effort is not None:
                drive_args["max_force"] = _drive_arg(max_effort, i)
            if max_velocity is not None:
                drive_args["max_velocity"] = _drive_arg(max_velocity, i)
            if friction is not None:
                drive_args["joint_friction"] = _drive_arg(friction, i)
            if armature is not None:
                drive_args["armature"] = _drive_arg(armature, i)
            if target_mode_value in {0, 4}:
                drive_args["stiffness"] = 0.0
                drive_args["damping"] = 0.0
            elif target_mode_value == 2:
                drive_args["stiffness"] = 0.0
            self._entities[env_idx].set_drive(**drive_args)

        if max_velocity is not None:
            max_velocity = torch.as_tensor(
                max_velocity, dtype=torch.float32, device=self.device
            )
            self._data._qvel_limits[cache_env_ids[:, None], cache_joint_ids] = (
                max_velocity
            )
        if max_effort is not None:
            max_effort = torch.as_tensor(
                max_effort, dtype=torch.float32, device=self.device
            )
            self._data._qf_limits[cache_env_ids[:, None], cache_joint_ids] = max_effort

    def get_joint_drive(
        self,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Get the drive properties for the articulation.

        Args:
            joint_ids (Sequence[int] | None, optional): The joint indices to get the drive properties for.
                If None, gets for all joints. Defaults to None.
            env_ids (Sequence[int] | None, optional): The environment indices to get the drive properties for.
                If None, gets for all environments. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing the stiffness, damping, max_effort,
                max_velocity, friction, and armature tensors with shape (N, len(joint_ids))
                for the specified environments.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        if joint_ids is None:
            local_joint_ids = np.arange(self.dof, dtype=np.int32)
        elif isinstance(joint_ids, torch.Tensor):
            local_joint_ids = (
                joint_ids.detach().cpu().numpy().astype(np.int32, copy=False)
            )
        else:
            local_joint_ids = np.asarray(joint_ids, dtype=np.int32)

        local_joint_ids_tensor = torch.as_tensor(
            local_joint_ids, dtype=torch.long, device=self.device
        )
        stiffness = torch.zeros(
            (len(local_env_ids), len(local_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        damping = torch.zeros(
            (len(local_env_ids), len(local_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        max_effort = torch.zeros(
            (len(local_env_ids), len(local_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        max_velocity = torch.zeros(
            (len(local_env_ids), len(local_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        friction = torch.zeros(
            (len(local_env_ids), len(local_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        armature = torch.zeros(
            (len(local_env_ids), len(local_joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        for i, env_idx in enumerate(local_env_ids):
            (
                stiffness_i,
                damping_i,
                max_effort_i,
                max_velocity_i,
                friction_i,
                armature_i,
                *_,
            ) = self._data._entity_drive_properties(self._entities[env_idx])
            stiffness[i] = torch.as_tensor(
                stiffness_i, dtype=torch.float32, device=self.device
            )[local_joint_ids_tensor]
            damping[i] = torch.as_tensor(
                damping_i, dtype=torch.float32, device=self.device
            )[local_joint_ids_tensor]
            max_effort[i] = torch.as_tensor(
                max_effort_i, dtype=torch.float32, device=self.device
            )[local_joint_ids_tensor]
            max_velocity[i] = torch.as_tensor(
                max_velocity_i, dtype=torch.float32, device=self.device
            )[local_joint_ids_tensor]
            friction[i] = torch.as_tensor(
                friction_i, dtype=torch.float32, device=self.device
            )[local_joint_ids_tensor]
            armature[i] = torch.as_tensor(
                armature_i, dtype=torch.float32, device=self.device
            )[local_joint_ids_tensor]
        return stiffness, damping, max_effort, max_velocity, friction, armature

    def get_joint_drive_type(
        self,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> list[list[DriveType]]:
        """Get the portable drive type for the selected joints.

        Args:
            joint_ids: Joint indices to query. If None, queries all joints.
            env_ids: Environment indices to query. If None, queries all environments.

        Returns:
            Drive types grouped by environment, with one
            :class:`~dexsim.types.DriveType` per selected joint.

            Newton has no acceleration-drive equivalent. Its passive and
            direct-effort target modes map to :attr:`DriveType.NONE` because
            neither installs a PD drive; position and velocity target modes
            map to :attr:`DriveType.FORCE`.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        if joint_ids is None:
            local_joint_ids = np.arange(self.dof, dtype=np.int32)
        elif isinstance(joint_ids, torch.Tensor):
            local_joint_ids = (
                joint_ids.detach().cpu().numpy().astype(np.int32, copy=False)
            )
        else:
            local_joint_ids = np.asarray(joint_ids, dtype=np.int32)

        drive_types: list[list[DriveType]] = []
        for env_idx in local_env_ids:
            entity = self._entities[int(env_idx)]
            if self._data is not None and self._data.is_newton_backend:
                target_modes = np.asarray(entity.get_newton_drive()[-1])[
                    local_joint_ids
                ]
                drive_types.append(
                    [
                        (DriveType.NONE if int(mode) in {0, 4} else DriveType.FORCE)
                        for mode in target_modes
                    ]
                )
            else:
                entity_drive_types = np.asarray(entity.get_drive()[-1])[local_joint_ids]
                drive_types.append(list(entity_drive_types))
        return drive_types

    def get_joint_target_mode(
        self,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> list[list[int]]:
        """Get Newton ``JointTargetMode`` integer values by environment.

        Args:
            joint_ids: Flattened DOF indices. If None, all DOFs are queried.
            env_ids: Environment indices. If None, all environments are
                queried.

        Returns:
            Integer target modes grouped by selected environment.
        """
        if not (
            self.is_spawn_bound
            and self._data is not None
            and self._data.is_newton_backend
        ):
            raise RuntimeError(
                "get_joint_target_mode() requires a Spawn-bound Newton " "articulation."
            )
        local_env_ids = self._all_indices if env_ids is None else env_ids
        if joint_ids is None:
            local_joint_ids = np.arange(self.dof, dtype=np.int32)
        elif isinstance(joint_ids, torch.Tensor):
            local_joint_ids = (
                joint_ids.detach().cpu().numpy().astype(np.int32, copy=False)
            )
        else:
            local_joint_ids = np.asarray(joint_ids, dtype=np.int32)

        target_modes = []
        for env_idx in local_env_ids:
            modes = self._entities[int(env_idx)].get_newton_drive()[-1]
            target_modes.append(
                [int(value) for value in np.asarray(modes)[local_joint_ids]]
            )
        return target_modes

    def get_user_ids(
        self, link_name: str | None = None, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Get the user ids of the articulation.

        Args:
            link_name: (str | None): The name of the link. If None, returns user ids for all links.
            env_ids: (Sequence[int] | None): Environment indices. If None, then all indices are used.

        Returns:
            torch.Tensor: The user ids of the articulation with shape (N, 1) for given link_name or (N, num_links) if link_name is None.
        """
        if link_name is not None and link_name not in self.link_names:
            logger.log_error(
                f"Link name {link_name} not found in {self.__class__.__name__}. Available links: {self.link_names}"
            )

        local_env_ids = self._all_indices if env_ids is None else env_ids

        if link_name is None:
            return self.user_ids[local_env_ids]
        else:
            link_idx = self.link_names.index(link_name)
            return self.user_ids[local_env_ids, link_idx]

    def clear_dynamics(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the dynamics of the articulation.

        Args:
            env_ids (Sequence[int] | None): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        self._data.articulation_view.clear_dynamics(local_env_ids)

    def reallocate_body_data(self) -> None:
        """Reallocate body data tensors to match the current articulation state in the GPU physics scene."""
        if self.device.type == "cpu":
            logger.log_warning(f"Reallocating body data on CPU is not supported.")
            return

        max_dof = self._ps.gpu_get_articulation_max_dof()
        max_num_links = self._ps.gpu_get_articulation_max_link_count()
        self._data._qpos = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._data._target_qpos = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._data._qvel = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._data._target_qvel = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._data._qacc = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._data._qf = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._data._body_link_pose = torch.zeros(
            (self.num_instances, max_num_links, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self._data._body_link_vel = torch.zeros(
            (self.num_instances, max_num_links, 6),
            dtype=torch.float32,
            device=self.device,
        )

        self._data._body_link_lin_vel = torch.zeros(
            (self.num_instances, max_num_links, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._data._body_link_ang_vel = torch.zeros(
            (self.num_instances, max_num_links, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self.reset()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        local_env_ids = self._all_indices if env_ids is None else env_ids
        num_instances = len(local_env_ids)
        self.cfg: ArticulationCfg

        self.restore_visual_material(env_ids=local_env_ids)
        self._restore_default_physical_properties(local_env_ids)

        if self.cfg.init_local_pose is not None:
            pose = (
                torch.as_tensor(
                    self.cfg.init_local_pose,
                    dtype=torch.float32,
                    device=self.device,
                )
                .reshape(1, 4, 4)
                .repeat(num_instances, 1, 1)
            )
        else:
            pos = torch.as_tensor(
                self.cfg.init_pos, dtype=torch.float32, device=self.device
            )
            rot = (
                torch.as_tensor(
                    self.cfg.init_rot, dtype=torch.float32, device=self.device
                )
                * torch.pi
                / 180.0
            )
            pos = pos.unsqueeze(0).repeat(num_instances, 1)
            rot = rot.unsqueeze(0).repeat(num_instances, 1)
            pose = (
                torch.eye(4, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .repeat(num_instances, 1, 1)
            )
            pose[:, :3, 3] = pos
            pose[:, :3, :3] = matrix_from_euler(rot, "XYZ")
        self.set_local_pose(pose, env_ids=local_env_ids)

        qpos = torch.as_tensor(
            self.cfg.init_qpos, dtype=torch.float32, device=self.device
        )
        qpos = qpos.unsqueeze(0).repeat(num_instances, 1)
        qpos = self._source_qpos_to_state_order(qpos)
        if (
            self.body_data.is_newton_backend
            and not self._newton_mimic_compliance_configured
        ):
            # Native Newton mimic constraints can generate a large corrective
            # impulse when initialized away from their equality manifold.
            qpos = self._project_mimic_qpos(qpos)
        self.set_qpos(qpos, target=False, env_ids=local_env_ids)
        # Set drive target to hold position.
        self.set_qpos(qpos, target=True, env_ids=local_env_ids)

        self.clear_dynamics(env_ids=local_env_ids)

        self._data.articulation_view.compute_kinematics(local_env_ids)
        if self.device.type == "cpu" and not self._data.is_newton_backend:
            self._world.update(0.001)

    def _set_default_joint_drive(
        self,
        joint_drive_props: JointDrivePropertiesCfg | dict | None = None,
    ) -> None:
        """Set default joint drive parameters based on the configuration."""
        import numbers
        from embodichain.utils.string import resolve_matching_names_values

        if joint_drive_props is None:
            joint_drive_props = self.cfg.joint_drive_props
        if joint_drive_props is None:
            return

        joint_property_targets = [
            ("damping", self.default_joint_damping),
            ("stiffness", self.default_joint_stiffness),
            ("max_effort", self.default_joint_max_effort),
            ("max_velocity", self.default_joint_max_velocity),
            ("friction", self.default_joint_friction),
            ("armature", self.default_joint_armature),
        ]

        for prop_name, default_array in joint_property_targets:
            value = (
                joint_drive_props.get(prop_name)
                if isinstance(joint_drive_props, dict)
                else getattr(joint_drive_props, prop_name, None)
            )
            if value is None:
                continue
            if isinstance(value, numbers.Number):
                default_array[:] = value
            else:
                try:
                    indices, _, values = resolve_matching_names_values(
                        value, self.joint_names
                    )
                    default_array[:, indices] = torch.as_tensor(
                        values, dtype=torch.float32, device=self.device
                    )
                except Exception as e:
                    logger.log_error(f"Failed to set {prop_name}: {e}")

        if isinstance(joint_drive_props, dict):
            drive_type = joint_drive_props.get("drive_type")
            target_mode = joint_drive_props.get("target_mode")
        else:
            drive_type = getattr(joint_drive_props, "drive_type", None)
            target_mode = getattr(joint_drive_props, "target_mode", None)
        if isinstance(target_mode, dict):
            logger.log_warning(
                "Per-joint target_mode mappings require a Spawn-bound "
                "articulation; the retained raw-articulation path preserves "
                "its current target modes."
            )
            target_mode = None

        # Apply drive parameters to all articulations in the batch
        self.set_joint_drive(
            stiffness=self.default_joint_stiffness,
            damping=self.default_joint_damping,
            max_effort=self.default_joint_max_effort,
            max_velocity=self.default_joint_max_velocity,
            friction=self.default_joint_friction,
            armature=self.default_joint_armature,
            drive_type=drive_type,
            target_mode=target_mode,
        )

    def compute_fk(
        self,
        qpos: torch.Tensor | np.ndarray | None,
        link_names: str | list[str] | tuple[str] | None = None,
        end_link_name: str | None = None,
        root_link_name: str | None = None,
        to_dict: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, dict[str, "pk.Transform3d"]]:
        """Compute the forward kinematics (FK) for the given joint positions.

        Args:
            qpos (torch.Tensor): Joint positions. Shape can be (dof,) for a single configuration or
                                (batch_size, dof) for batched configurations.
            link_names (Union[str, list[str], tuple[str]], optional): Names of the links for which FK is computed.
                                                                    If None, all links are considered.
            end_link_name (str, optional): Name of the end link for which FK is computed. If None, all links are considered.
            root_link_name (str, optional): Name of the root link for which FK is computed. Defaults to None.
            to_dict (bool, optional): If True, returns the FK result as a dictionary of Transform3d objects. Defaults to False.
            **kwargs: Additional keyword arguments for customization.

        Raises:
            RuntimeError: If the pk_chain is not initialized.
            TypeError: If an invalid type is provided for `link_names`.
            ValueError: If the shape of the resulting matrices is unexpected.

        Returns:
            torch.Tensor: The homogeneous transformation matrix/matrices for the specified links.
                        Shape is (batch_size, 4, 4) for batched input or (4, 4) for single input.
                        If `to_dict` is True, returns a dictionary of Transform3d objects instead.
        """
        frame_indices = None
        if self.pk_chain is None:
            logger.log_error("pk_chain is not initialized for this articulation.")

        # Adapt link_names to work with get_frame_indices
        if link_names is not None:
            if isinstance(link_names, str):
                # Single link name
                frame_indices = self.pk_chain.get_frame_indices(link_names)
            elif isinstance(link_names, (list, tuple)):
                # Multiple link names
                frame_indices = self.pk_chain.get_frame_indices(*link_names)
            else:
                raise TypeError(
                    f"Invalid type for link_names: {type(link_names)}. Expected str, list, or tuple."
                )

        if end_link_name is None and root_link_name is None:
            result = self.pk_chain.forward_kinematics(
                th=qpos, frame_indices=frame_indices
            )
        else:
            pk_serial_chain = create_pk_serial_chain(
                chain=self.pk_chain,
                root_link_name=root_link_name,
                end_link_name=end_link_name,
                device=self.device,
            )
            result = pk_serial_chain.forward_kinematics(th=qpos, end_only=True)

        if to_dict:
            return result

        # Extract transformation matrices
        if isinstance(result, dict):
            if link_names:
                matrices = torch.stack(
                    [result[name].get_matrix() for name in link_names], dim=0
                )
            else:
                link_name = end_link_name if end_link_name else list(result.keys())[-1]
                matrices = result[link_name].get_matrix()
        elif isinstance(result, list):
            matrices = torch.stack(
                [xpos.get_matrix().squeeze() for xpos in result], dim=0
            )
        else:
            matrices = result.get_matrix()

        # Ensure batch format
        if matrices.dim() == 2:
            matrices = matrices.unsqueeze(0)

        # Create result tensor with proper homogeneous coordinates
        if matrices.dim() == 4:  # Multiple links
            num_links, batch_size, _, _ = matrices.shape
            result = (
                torch.eye(4, device=self.device)
                .expand(num_links, batch_size, 4, 4)
                .clone()
            )
            result[:, :, :3, :] = matrices[:, :, :3, :]
            result = result.permute(1, 0, 2, 3)  # (batch_size, num_links, 4, 4)
        elif matrices.dim() == 3:  # Single link
            batch_size, _, _ = matrices.shape
            result = torch.eye(4, device=self.device).expand(batch_size, 4, 4).clone()
            result[:, :3, :] = matrices[:, :3, :]
        else:
            raise ValueError(f"Unexpected matrices shape: {matrices.shape}")

        return result

    def compute_jacobian(
        self,
        qpos: torch.Tensor | np.ndarray | None,
        end_link_name: str = None,
        root_link_name: str = None,
        locations: torch.Tensor | np.ndarray | None = None,
        jac_type: str = "full",
    ) -> torch.Tensor:
        """Compute the Jacobian matrix for the given joint positions using the pk_serial_chain.

        Args:
            qpos (torch.Tensor): The joint positions. Shape can be (dof,) for a single configuration
                                 or (batch_size, dof) for batched configurations.
            end_link_name (str, optional): The name of the end link for which the Jacobian is computed.
                                           Defaults to the last link in the chain.
            root_link_name (str, optional): The name of the root link for which the Jacobian is computed.
                                            Defaults to the first link in the chain.
            locations (torch.Tensor | np.ndarray, optional): Offset points relative to the end-effector
                                                                   frame for which the Jacobian is computed.
                                                                   Shape can be (batch_size, 3) or (3,) for a single offset.
                                                                   Defaults to None (origin of the end-effector frame).
            jac_type (str, optional): Specifies the part of the Jacobian to return:
                                      - 'full': Returns the full Jacobian (6, dof) or (batch_size, 6, dof).
                                      - 'trans': Returns only the translational part (3, dof) or (batch_size, 3, dof).
                                      - 'rot': Returns only the rotational part (3, dof) or (batch_size, 3, dof).
                                      Defaults to 'full'.

        Raises:
            RuntimeError: If the pk_chain is not initialized.
            ValueError: If an invalid `jac_type` is provided.

        Returns:
            torch.Tensor: The Jacobian matrix. Shape depends on the input:
                          - For a single link: (6, dof) or (batch_size, 6, dof).
                          - For multiple links: (num_links, 6, dof) or (num_links, batch_size, 6, dof).
                          The shape also depends on the `jac_type` parameter.
        """
        if self.pk_chain is None:
            logger.log_error("pk_chain is not initialized for this articulation.")

        if qpos is None:
            qpos = torch.zeros(self.dof, device=self.device)

        # Ensure qpos is a tensor on the correct device
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)

        # Default root and end link names if not provided
        frame_names = self.pk_chain.get_frame_names()
        if root_link_name is None:
            root_link_name = frame_names[0]  # Default to the first frame
        if end_link_name is None:
            end_link_name = frame_names[-1]  # Default to the last frame

        # Create pk_serial_chain
        pk_serial_chain = create_pk_serial_chain(
            urdf_path=self.cfg.fpath,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
            device=self.device,
        )

        # Compute the Jacobian using the kinematics chain
        J = pk_serial_chain.jacobian(th=qpos, locations=locations)

        # Handle jac_type to return the desired part of the Jacobian
        if jac_type == "trans":
            return J[:, :3, :] if J.dim() == 3 else J[:3, :]
        elif jac_type == "rot":
            return J[:, 3:, :] if J.dim() == 3 else J[3:, :]
        elif jac_type == "full":
            return J
        else:
            raise ValueError(
                f"Invalid jac_type '{jac_type}'. Must be 'full', 'trans', or 'rot'."
            )

    def set_visual_material(
        self,
        mat: VisualMaterial,
        env_ids: Sequence[int] | None = None,
        link_names: List[str] | None = None,
        shared: bool = False,
        update_default: bool = False,
    ) -> None:
        """Set visual material for the rigid object.

        Args:
            mat (VisualMaterial): The material to set.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
            link_names (List[str] | None, optional): List of link names to apply the material to. If None, applies to all links.
            shared (bool, optional): Whether to share the material instance across links and environments. Defaults to False.
            update_default: Whether the assigned material should become the baseline
                restored by :meth:`reset`. Defaults to False.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        link_names = self.link_names if link_names is None else link_names

        if shared:
            if len(local_env_ids) != self.num_instances:
                logger.log_error(f"Cannot share material instance for partial env_ids.")

            for link_name in link_names:
                mat_inst = mat.create_instance(f"{mat.uid}_{self.uid}_{link_name}")
                for i, env_idx in enumerate(local_env_ids):
                    if self.is_spawn_bound:
                        self._entities[env_idx].set_material_inst(
                            link_name, mat_inst.mat
                        )
                    else:
                        self._entities[env_idx].set_material(link_name, mat_inst.mat)
                    self._visual_material[env_idx][link_name] = mat_inst
                    if update_default:
                        self._original_visual_material[env_idx][link_name] = (
                            _capture_render_materials(
                                self._entities[env_idx].get_render_body(link_name)
                            )
                        )
                        self._original_visual_material_inst[env_idx][
                            link_name
                        ] = mat_inst
            self.is_shared_visual_material = True
        else:
            for i, env_idx in enumerate(local_env_ids):
                for link_name in link_names:
                    mat_inst = mat.create_instance(
                        f"{mat.uid}_{self.uid}_{link_name}_{env_idx}"
                    )
                    if self.is_spawn_bound:
                        self._entities[env_idx].set_material_inst(
                            link_name, mat_inst.mat
                        )
                    else:
                        self._entities[env_idx].set_material(link_name, mat_inst.mat)
                    self._visual_material[env_idx][link_name] = mat_inst
                    if update_default:
                        self._original_visual_material[env_idx][link_name] = (
                            _capture_render_materials(
                                self._entities[env_idx].get_render_body(link_name)
                            )
                        )
                        self._original_visual_material_inst[env_idx][
                            link_name
                        ] = mat_inst
            self.is_shared_visual_material = False

    def get_visual_material_inst(
        self,
        env_ids: Sequence[int] | None = None,
        link_names: List[str] | None = None,
    ) -> List[Dict[str, VisualMaterialInst]]:
        """Get visual material instances for the rigid object.

        Args:
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
            link_names (List[str] | None, optional): List of link names to filter materials. If None, returns materials for all links.
        Returns:
            List[Dict[str, VisualMaterialInst]]: A list where each element corresponds to an environment and contains a dictionary mapping link names to their VisualMaterialInst.
        """
        if env_ids is None and link_names is None:
            return self._visual_material

        local_env_ids = self._all_indices if env_ids is None else env_ids
        link_names = self.link_names if link_names is None else link_names

        result = []
        for i, env_idx in enumerate(local_env_ids):
            if link_names is None:
                result.append(self._visual_material[env_idx])
            else:
                mat_dict = {
                    link_name: self._visual_material[env_idx][link_name]
                    for link_name in link_names
                    if link_name in self._visual_material[env_idx]
                }
                result.append(mat_dict)
        return result

    def _initialize_existing_visual_material(self) -> None:
        """Wrap asset-parsed materials during articulation construction.

        The public material mapping stores one representative material per link.
        For links with multiple mesh segments, the first segment with a valid
        material is registered. Segment-specific materials remain available
        through :meth:`get_existing_visual_material`.
        """
        self._original_visual_material = [{} for _ in self._entities]
        self._original_visual_material_inst = [{} for _ in self._entities]
        for env_idx, entity in enumerate(self._entities):
            for link_name in self.link_names:
                render_body = entity.get_render_body(link_name)
                if render_body is None:
                    continue
                original_materials = _capture_render_materials(render_body)
                self._original_visual_material[env_idx][link_name] = original_materials
                wrapped = _wrap_first_render_material(original_materials)
                if wrapped is not None:
                    self._visual_material[env_idx][link_name] = wrapped
                    self._original_visual_material_inst[env_idx][link_name] = wrapped

    def restore_visual_material(
        self,
        env_ids: Sequence[int] | None = None,
        link_names: List[str] | None = None,
    ) -> None:
        """Restore visual materials captured when the articulation was created.

        Args:
            env_ids: Environment indices. If None, all instances are restored.
            link_names: Links to restore. If None, all links are restored.
        """
        if not hasattr(self, "_original_visual_material"):
            return
        local_env_ids = self._all_indices if env_ids is None else env_ids
        local_link_names = self.link_names if link_names is None else link_names
        for env_idx in local_env_ids:
            for link_name in local_link_names:
                original_materials = self._original_visual_material[env_idx].get(
                    link_name
                )
                if original_materials is None:
                    continue
                render_body = self._entities[env_idx].get_render_body(link_name)
                if render_body is None:
                    continue
                _restore_render_materials(render_body, original_materials)
                original_inst = self._original_visual_material_inst[env_idx].get(
                    link_name
                )
                if original_inst is None:
                    self._visual_material[env_idx].pop(link_name, None)
                else:
                    self._visual_material[env_idx][link_name] = original_inst
        self.is_shared_visual_material = False

    def get_existing_visual_material(
        self,
        env_ids: Sequence[int] | None = None,
        link_names: List[str] | None = None,
        shared: bool = False,
    ) -> List[Dict[str, List[ReuseSegmentState]]]:
        """Build reuse state from materials dexsim parsed onto each link's render body.

        Each segment keeps its original material for restoration. Segments on the
        same link share one working material so randomized property updates happen
        once per link instead of once per mesh segment.

        Args:
            env_ids: Environment indices. If None, all instances are used.
            link_names: Links to include. If None, all links are used.
            shared: If True, build state for the first env only.

        Returns:
            Per-env dict mapping link name to per-segment :obj:`ReuseSegmentState`.

        Raises:
            ValueError: If a link/segment has no material or no retrievable template.
        """
        if shared:
            local_env_ids = [self._all_indices[0]]
        else:
            local_env_ids = self._all_indices if env_ids is None else list(env_ids)
        link_names = self.link_names if link_names is None else list(link_names)

        if not hasattr(self, "_original_visual_material"):
            self._original_visual_material = [{} for _ in self._entities]
        for env_idx in local_env_ids:
            for link_name in link_names:
                if link_name in self._original_visual_material[env_idx]:
                    continue
                render_body = self._entities[env_idx].get_render_body(link_name)
                if render_body is not None:
                    self._original_visual_material[env_idx][link_name] = (
                        _capture_render_materials(render_body)
                    )

        per_env: List[Dict[str, List[ReuseSegmentState]]] = []
        for env_idx in local_env_ids:
            link_map: Dict[str, List[ReuseSegmentState]] = {}
            for link_name in link_names:
                if self._entities[env_idx].get_render_body(link_name) is None:
                    raise ValueError(
                        f"Articulation '{self.uid}' link '{link_name}' has no render body."
                    )
                segments: List[ReuseSegmentState] = []
                working_inst = None
                for mesh_id, original_inst in enumerate(
                    self._original_visual_material[env_idx][link_name]
                ):
                    if original_inst is None:
                        raise ValueError(
                            f"Articulation '{self.uid}' link '{link_name}' segment {mesh_id} has no material."
                        )
                    template = original_inst.get_template()
                    if template is None:
                        raise ValueError(
                            f"Articulation '{self.uid}' link '{link_name}' material has no template."
                        )
                    if working_inst is None:
                        working_name = f"{self.uid}_reuse_{env_idx}_{link_name}"
                        template.create_inst(working_name)
                        working_inst = VisualMaterialInst(working_name, template)
                    segments.append(
                        ReuseSegmentState(
                            mesh_id=mesh_id,
                            original_inst=original_inst,
                            working_inst=working_inst,
                        )
                    )
                link_map[link_name] = segments
            per_env.append(link_map)
        return per_env

    def apply_render_material_inst(
        self,
        env_idx: int,
        mat_inst: MaterialInst,
        link_name: str,
        mesh_id: int = 0,
    ) -> None:
        """Swap a dexsim MaterialInst onto a link's render-body segment for the given env.

        Args:
            env_idx: Environment index.
            mat_inst: dexsim ``MaterialInst`` to attach.
            link_name: Link whose render body receives the material.
            mesh_id: Render-body segment index.
        """
        _set_render_material(
            self._entities[env_idx].get_render_body(link_name), mesh_id, mat_inst
        )

    def set_physical_visible(
        self,
        visible: bool = True,
        link_names: List[str] | None = None,
        rgba: Sequence[float] | None = None,
    ):
        """set collision

        Args:
            visible (bool, optional): is collision body visible. Defaults to True.
            link_names (List[str] | None, optional): links to set visibility. Defaults to None.
            rgba (Sequence[float] | None, optional): collision body visible rgba. It will be defined at the first time the function is called. Defaults to None.
        """
        rgba = rgba if rgba is not None else (0.8, 0.2, 0.2, 0.7)
        if len(rgba) != 4:
            logger.log_error(f"Invalid rgba {rgba}, should be a sequence of 4 floats.")
        rgba = np.array(
            [
                rgba[0],
                rgba[1],
                rgba[2],
                rgba[3],
            ]
        )
        link_names = self.link_names if link_names is None else link_names

        if self.is_spawn_bound:
            for env_idx in self._all_indices:
                entity = self._entities[env_idx]
                for link_name in link_names:
                    self._spawn_result.set_physical_visible(
                        (entity, link_name), rgba, visible
                    )
            for link_name in link_names:
                self._has_collision_visible_node_dict[link_name] = True
            return

        # create collision visible node if not exist
        if visible:
            for i, env_idx in enumerate(self._all_indices):
                for link_name in link_names:
                    if self._has_collision_visible_node_dict[link_name] is False:
                        self._entities[env_idx].create_physical_visible_node(
                            rgba, link_name
                        )
                        self._has_collision_visible_node_dict[link_name] = True

        # set visibility
        for i, env_idx in enumerate(self._all_indices):
            for link_name in link_names:
                self._entities[env_idx].set_physical_visible(visible, link_name)

    def set_fix_base(
        self,
        fix: bool = True,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set whether the base of the articulation is fixed.

        Args:
            fix (bool, optional): Whether to fix the base. Defaults to True.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].set_articulation_flag(
                ArticulationFlag.FIX_BASE, fix
            )

    def set_self_collision(
        self,
        enable: bool = False,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Set whether self-collision is enabled for the articulation.

        Args:
            enable (bool, optional): Whether to enable self-collision. Defaults to True.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        for i, env_idx in enumerate(local_env_ids):
            self._entities[env_idx].set_articulation_flag(
                ArticulationFlag.DISABLE_SELF_COLLISION, not enable
            )

    def destroy(self) -> None:
        if self.is_declared or self.is_spawn_bound:
            # The finalized Scene is the sole owner of native lifetime.
            return
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        for i, entity in enumerate(self._entities):
            if self._data.is_newton_backend:
                arenas[i].remove_skeleton(entity)
            else:
                arenas[i].remove_articulation(entity)


__all__ = ["ArticulationData", "Articulation", "ArticulationJointKinematics"]
