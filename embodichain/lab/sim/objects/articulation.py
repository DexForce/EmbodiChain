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

import torch
import dexsim
import numpy as np

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List, Sequence, Dict, Union, Tuple, Optional

from dexsim.engine import Articulation as _Articulation
from dexsim.types import (
    ArticulationFlag,
    ArticulationGPUAPIWriteType,
    ArticulationGPUAPIReadType,
    DriveType,
)
from dexsim.engine import CudaArray, MaterialInst, PhysicsScene

from embodichain.lab.sim import VisualMaterialInst, VisualMaterial, ReuseSegmentState
from embodichain.lab.sim.material import (
    _capture_render_materials,
    _restore_render_materials,
    _set_render_material,
    _wrap_first_render_material,
)
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RigidBodyAttributesOverrideCfg,
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
    DefaultArticulationView,
    NewtonArticulationView,
    SpawnArticulationView,
    is_newton_scene,
)
from embodichain.lab.sim.objects.backends.base import ArticulationViewBase
from embodichain.utils.math import (
    matrix_from_quat,
    quat_from_matrix,
    matrix_from_euler,
)
from embodichain.lab.sim.utility.sim_utils import get_dexsim_drive_type
from embodichain.lab.sim.utility.solver_utils import (
    create_pk_chain,
    create_pk_serial_chain,
)
from embodichain.utils import logger

if TYPE_CHECKING:
    from dexsim.spawn import SpawnResult, SpawnedArticulation


@dataclass
class ArticulationData:
    """GPU data manager for articulation."""

    def __init__(
        self,
        entities: Sequence[_Articulation | SpawnedArticulation],
        ps: PhysicsScene | None,
        device: torch.device,
        articulation_view: ArticulationViewBase | None = None,
    ) -> None:
        """Initialize the ArticulationData.

        Args:
            entities (List[_Articulation]): List of DexSim Articulation objects.
            ps (PhysicsScene): The physics scene.
            device (torch.device): The device to use for the articulation data.
        """
        self.entities = entities
        self.ps = ps
        self.num_instances = len(entities)
        self.device = device
        if articulation_view is not None:
            self.articulation_view = articulation_view
        elif is_newton_scene(ps):
            self.articulation_view = NewtonArticulationView(
                entities=entities, scene=ps, device=device
            )
        else:
            self.articulation_view = DefaultArticulationView(
                entities=entities, ps=ps, device=device
            )

        # Backward-compatible alias for callers that use GPU/articulation ids.
        self.gpu_indices = self.articulation_view.articulation_ids_tensor

        if isinstance(self.articulation_view, SpawnArticulationView):
            self.dof = self.articulation_view.dof
            self.num_links = self.articulation_view.num_links
            self.link_names = self.articulation_view.link_names
        else:
            self.dof = self.entities[0].get_dof()
            self.num_links = self.entities[0].get_links_num()
            self.link_names = self.entities[0].get_link_names()

        self._root_pose = torch.zeros(
            (self.num_instances, 7), dtype=torch.float32, device=self.device
        )
        self._root_lin_vel = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )
        self._root_ang_vel = torch.zeros(
            (self.num_instances, 3), dtype=torch.float32, device=self.device
        )

        max_num_links = self.num_links
        if (
            articulation_view is None
            and self.device.type == "cuda"
            and not self.is_newton_backend
        ):
            max_num_links = self.ps.gpu_get_articulation_max_link_count()
        self._body_link_pose = torch.zeros(
            (self.num_instances, max_num_links, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_link_vel = torch.zeros(
            (self.num_instances, max_num_links, 6),
            dtype=torch.float32,
            device=self.device,
        )

        self._body_link_lin_vel = torch.zeros(
            (self.num_instances, max_num_links, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._body_link_ang_vel = torch.zeros(
            (self.num_instances, max_num_links, 3),
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

        max_dof = self.dof
        if (
            articulation_view is None
            and self.device.type == "cuda"
            and not self.is_newton_backend
        ):
            max_dof = self.ps.gpu_get_articulation_max_dof()
        self._target_qpos = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._qpos = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._target_qvel = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._qvel = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._qacc = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
        )
        self._qf = torch.zeros(
            (self.num_instances, max_dof), dtype=torch.float32, device=self.device
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
            torch.Tensor: The root pose of the articulation with shape of (num_instances, 7).
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
            torch.Tensor: The poses of the links in the articulation with shape (N, num_links, 7).
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

    def _entity_link_name(self, entity: object, link_name: str) -> str:
        """Resolve one public link name to an entity-local backend name."""
        resolver = getattr(self.articulation_view, "entity_link_name", None)
        if resolver is not None:
            return resolver(entity, link_name)
        return link_name

    def _entity_drive_properties(self, entity: object) -> tuple[object, ...]:
        """Read drive values without conflating backend target semantics."""
        if (
            isinstance(self.articulation_view, SpawnArticulationView)
            and self.is_newton_backend
        ):
            return tuple(entity.get_newton_drive())
        return tuple(entity.get_drive())

    def _entity_link_properties(self, entity: object, link_name: str) -> object:
        """Read native mass properties through the active backend contract."""
        if (
            isinstance(self.articulation_view, SpawnArticulationView)
            and self.is_newton_backend
        ):
            return entity.get_newton_link_properties(link_name)
        return entity.get_physical_attr(link_name)

    def read_physical_properties(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refresh current mass, inertia diagonal, and local COM pose buffers.

        COM poses use the articulation convention ``xyz + wxyz`` and all
        tensors use the public link ordering.
        """
        masses: list[list[float]] = []
        inertias: list[list[np.ndarray]] = []
        com_poses: list[list[np.ndarray]] = []
        for entity in self.entities:
            mass_row: list[float] = []
            inertia_row: list[np.ndarray] = []
            com_row: list[np.ndarray] = []
            for link_name in self.link_names:
                local_name = self._entity_link_name(entity, link_name)
                attr = self._entity_link_properties(entity, local_name)
                mass_row.append(float(attr.mass))
                inertia_row.append(np.asarray(attr.inertia, dtype=np.float32))
                com_row.append(
                    np.concatenate(
                        (
                            np.asarray(attr.com_position, dtype=np.float32),
                            np.asarray(attr.com_quaternion, dtype=np.float32),
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
        """Current local link COM poses with shape ``(N, num_links, 7)``."""
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
        """Initialization-time local link COM poses in ``xyz + wxyz`` order."""
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
        cfg (ArticulationCfg): Configuration for the articulation.
        entities (List[_Articulation], optional): List of articulation entities.
        device (torch.device, optional): Device to use (CPU or CUDA).
    """

    def __init__(
        self,
        cfg: ArticulationCfg,
        entities: Sequence[_Articulation | SpawnedArticulation] | None = None,
        device: torch.device = torch.device("cpu"),
        *,
        spawn_result: SpawnResult | None = None,
        declared_num_instances: int | None = None,
    ) -> None:
        if entities is None:
            if declared_num_instances is None or declared_num_instances <= 0:
                raise ValueError(
                    "A declared Articulation requires declared_num_instances > 0."
                )
            self.cfg = deepcopy(cfg)
            self.uid = self.cfg.uid
            self.device = device
            self._entities = []
            self._declared_num_instances = declared_num_instances
            self._spawn_result = None
            self._world = None
            self._ps = None
            self._data = None
            self._all_indices = torch.arange(declared_num_instances, dtype=torch.int32)
            self._visual_material = [{} for _ in range(declared_num_instances)]
            self.is_shared_visual_material = False
            self._has_collision_visible_node_dict = {}
            return

        self._declared_num_instances = len(entities)
        self._spawn_result = spawn_result
        if spawn_result is None:
            # Legacy initialization remains temporarily while SimulationManager
            # migration is in progress. Spawn-bound facades never reach for a
            # process-global World or raw PhysicsScene.
            self._world = dexsim.default_world()
            from embodichain.lab.sim.sim_manager import get_physics_scene

            self._ps = get_physics_scene()
        else:
            self._world = spawn_result.world
            self._ps = None

        self.cfg = cfg
        self._entities = entities
        self.device = device

        # Store all indices for batch operations
        self._all_indices = torch.arange(len(entities), dtype=torch.int32)

        if (
            spawn_result is None
            and device.type == "cuda"
            and not is_newton_scene(self._ps)
        ):
            self._world.update(0.001)

        articulation_view = None
        if spawn_result is not None:
            batch = spawn_result.create_articulation_batch(entities)
            articulation_view = SpawnArticulationView(spawn_result, batch, device)
        self._data = ArticulationData(
            entities=entities,
            ps=self._ps,
            device=device,
            articulation_view=articulation_view,
        )

        self.cfg: ArticulationCfg
        if self.cfg.init_qpos is None:
            self.cfg.init_qpos = torch.zeros(self.dof, dtype=torch.float32)

        self._capture_default_physical_properties()

        preserve_asset_physics = self.cfg.resolve_asset_physics_mode() == "preserve"
        self.default_joint_stiffness = self._data.joint_stiffness.clone()
        self.default_joint_damping = self._data.joint_damping.clone()
        self.default_joint_friction = self._data.joint_friction.clone()
        self.default_joint_armature = self._data.joint_armature.clone()
        self.default_joint_max_effort = self._data.qf_limits.clone()
        self.default_joint_max_velocity = self._data.qvel_limits.clone()

        # Spawn descriptors already contain build-time overlays. The retained
        # legacy path applies only explicitly requested drive fields here.
        if (
            spawn_result is None
            and not preserve_asset_physics
            and self.cfg.drive_pros is not None
        ):
            self._set_default_joint_drive()

        # Regex limits for Spawn-owned URDF and authored USD articulations are
        # already applied by EmbodiChain to the source-resolved descriptor.
        # Array limits still require the runtime path because they are not
        # declaration-time name rules. Preserve mode keeps all source limits.
        qpos_limits_are_source_resolved = (
            spawn_result is not None
            and isinstance(self.cfg.qpos_limits, dict)
            and not preserve_asset_physics
        )
        if (
            self.cfg.qpos_limits is not None
            and not preserve_asset_physics
            and not qpos_limits_are_source_resolved
        ):
            if isinstance(self.cfg.qpos_limits, dict):
                indices, _, values = resolve_matching_names_values(
                    self.cfg.qpos_limits, self.joint_names
                )
                local_joint_ids = torch.as_tensor(
                    indices, dtype=torch.long, device=self.device
                )
                values_tensor = torch.as_tensor(
                    values, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                values_tensor = values_tensor.expand(self.num_instances, -1, -1)
                self.set_qpos_limits(values_tensor, joint_ids=local_joint_ids)
            else:
                qpos_limits = torch.as_tensor(
                    self.cfg.qpos_limits, dtype=torch.float32, device=self.device
                )
                if qpos_limits.dim() == 2:
                    qpos_limits = qpos_limits.unsqueeze(0).expand(
                        self.num_instances, -1, -1
                    )
                self.set_qpos_limits(qpos_limits)

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

        # For rendering purposes, each articulation can have multiple material instances associated with its links.
        self._visual_material: List[Dict[str, VisualMaterialInst]] = [
            {} for _ in range(len(entities))
        ]
        self.is_shared_visual_material = False

        # Stores mimic information for joints.
        self._mimic_info = entities[0].get_mimic_info()

        self.active_joint_ids = [i for i in range(self.dof) if i not in self.mimic_ids]

        # TODO: very weird that we must call update here to make sure the GPU indices are valid.
        if (
            spawn_result is None
            and device.type == "cuda"
            and not is_newton_scene(self._ps)
        ):
            self._world.update(0.001)

        # Spawn-bound articulations receive post-load configuration before
        # their initial reset. Legacy construction keeps its historical reset.
        super().__init__(cfg, entities, device)
        if spawn_result is None:
            self.reset()

        self._initialize_existing_visual_material()

        # set default collision filter
        if spawn_result is None:
            self._set_default_collision_filter()

        # flag for collision visible node existence
        self._has_collision_visible_node_dict = dict()
        for link_name in self.link_names:
            self._has_collision_visible_node_dict[link_name] = False

    @property
    def is_spawn_bound(self) -> bool:
        """Whether this facade is bound to one finalized SpawnResult."""
        return self._spawn_result is not None

    @property
    def is_declared(self) -> bool:
        """Whether this facade is waiting for its SpawnResult binding."""
        return self._world is None

    @property
    def num_instances(self) -> int:
        if self._entities:
            return len(self._entities)
        return self._declared_num_instances

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
        if len(handles) != self._declared_num_instances:
            raise ValueError(
                f"Articulation {self.uid!r} expected "
                f"{self._declared_num_instances} Spawn handles, got {len(handles)}."
            )
        self._entities = handles
        self._mimic_info = self._entities[0].get_mimic_info()
        self.active_joint_ids = [
            index for index in range(self.dof) if index not in self.mimic_ids
        ]

    def bind_spawn(
        self,
        result: SpawnResult,
    ) -> None:
        """Initialize this declared facade from Spawn articulation handles."""
        if self.is_spawn_bound:
            raise RuntimeError(f"Articulation {self.uid!r} is already Spawn-bound.")
        if not self.is_declared:
            raise RuntimeError(
                f"Articulation {self.uid!r} was not created as a Spawn declaration."
            )

        cfg = self.cfg
        device = self.device
        entities = list(self._entities)
        if len(entities) != self._declared_num_instances:
            raise ValueError(
                f"Articulation {self.uid!r} expected "
                f"{self._declared_num_instances} Spawn handles, got {len(entities)}."
            )

        # Build and configure the bound state off to the side. If batch
        # creation or post-load configuration fails, the public facade remains
        # declared and can be retried by SimulationManager.prepare().
        bound = type(self)(
            cfg,
            entities,
            device,
            spawn_result=result,
        )
        bound._apply_spawn_config()
        if is_newton_gradient_mode(result):
            initial_qpos = torch.as_tensor(bound.cfg.init_qpos).reshape(-1)
            if initial_qpos.numel() != bound.dof:
                raise ValueError(
                    f"Articulation {bound.uid!r} expected {bound.dof} initial "
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
            bound.reset()
        self.__dict__.clear()
        self.__dict__.update(bound.__dict__)

    def _apply_spawn_config(self) -> None:
        """Apply render-only configuration requiring finalized source metadata.

        Link physics and joint-drive regex selection is resolved by
        EmbodiChain against the source descriptor before finalization. Only
        render operations that require materialized bodies remain here.
        """
        if not self.cfg.compute_uv:
            return

        for entity in self._entities:
            for link_name in self.link_names:
                render_body = entity.get_render_body(link_name)
                if render_body is not None:
                    render_body.set_projective_uv()

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
        """Get the names of the joints in the articulation.

        Returns:
            List[str]: The names of the actived joints in the articulation.
        """
        return self._entities[0].get_actived_joint_names()

    @cached_property
    def active_joint_names(self) -> List[str]:
        """Get the names of the active joints in the articulation.

        Returns:
            List[str]: The names of the active joints in the articulation.
        """
        return [self.joint_names[i] for i in self.active_joint_ids]

    @cached_property
    def all_joint_names(self) -> List[str]:
        """Get the names of the joints in the articulation.

        Returns:
            List[str]: The names of the joints in the articulation.
        """
        return self._entities[0].get_joint_names()

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

    def _entity_link_name(self, env_idx: int, link_name: str) -> str:
        """Resolve a canonical link name to the backend entity's local name."""
        if isinstance(env_idx, torch.Tensor):
            env_idx = int(env_idx.detach().cpu().item())
        entity = self._entities[int(env_idx)]
        view = self._data.articulation_view
        if hasattr(view, "entity_link_name"):
            return view.entity_link_name(entity, link_name)
        return link_name

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
            to_matrix (bool, optional): If True, return the pose as a 4x4 matrix. If False, return as (x, y, z, qw, qx, qy, qz). Defaults to False.

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
            to_matrix (bool, optional): If True, return the pose as a 4x4 matrix. If False, return as (x, y, z, qw, qx, qy, qz). Defaults to False.

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
        local_env_ids = self._all_indices if env_ids is None else env_ids

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

        self._data.articulation_view.apply_qvel(
            qvel,
            local_env_ids,
            local_joint_ids,
            target=target,
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
                if self.is_spawn_bound:
                    local_name = self._entity_link_name(env_idx, name)
                    entity.set_link_mass(local_name, mass[i, j].item())
                elif self._data.is_newton_backend:
                    local_name = self._entity_link_name(env_idx, name)
                    entity.set_link_mass(local_name, mass[i, j].item())
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
                local_name = self._entity_link_name(env_idx, name)
                value = np.asarray(values[i, j], dtype=np.float32)
                if self.is_spawn_bound and self._data.is_newton_backend:
                    entity.set_newton_link_properties(
                        local_name,
                        rigid_body=dexsim.spawn.RigidBodyPhysicsDesc.dynamic(
                            inertia=value
                        ),
                    )
                elif not self._data.is_newton_backend:
                    entity.get_physical_body(local_name).set_mass_space_inertia_tensor(
                        value
                    )
                else:
                    attr = entity.get_physical_attr(local_name)
                    attr.inertia = value
                    entity.set_physical_attr(
                        attr,
                        local_name,
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
        """Set local COM poses in articulation ``xyz + wxyz`` convention."""
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
                local_name = self._entity_link_name(env_idx, name)
                position = np.asarray(values[i, j, :3], dtype=np.float32)
                quaternion = np.asarray(values[i, j, 3:7], dtype=np.float32)
                if self.is_spawn_bound and self._data.is_newton_backend:
                    entity.set_newton_link_properties(
                        local_name,
                        rigid_body=dexsim.spawn.RigidBodyPhysicsDesc.dynamic(
                            com_position=position,
                            com_quaternion=quaternion,
                        ),
                    )
                elif not self._data.is_newton_backend:
                    entity.get_physical_body(local_name).set_cmass_local_pose(
                        position,
                        quaternion,
                    )
                else:
                    attr = entity.get_physical_attr(local_name)
                    attr.com_position = position
                    attr.com_quaternion = quaternion
                    entity.set_physical_attr(
                        attr,
                        local_name,
                        is_replace_inertial=False,
                    )

    def get_com_pose(
        self,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get local COM poses in articulation ``xyz + wxyz`` convention."""
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
                attrs.append(
                    entity.get_physical_attr(self._entity_link_name(env_idx, name))
                )
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
                properties.append(
                    entity.get_newton_link_properties(
                        self._entity_link_name(env_idx, name)
                    )
                )
        return properties

    def set_link_physical_attr(
        self,
        attrs: RigidBodyAttributesCfg | RigidBodyAttributesOverrideCfg | PhysicalAttr,
        link_names: str | Sequence[str] | None = None,
        env_ids: Sequence[int] | None = None,
        *,
        base_attrs: RigidBodyAttributesCfg | RigidBodyPhysicsCfg | None = None,
        replace_inertial: bool = False,
    ) -> None:
        """Set physical attributes for selected articulation links.

        Args:
            attrs: Full, partial, or DexSim physical attributes to apply.
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

        if isinstance(attrs, RigidBodyAttributesOverrideCfg):
            if base_attrs is None:
                base_attrs = self.cfg.attrs
            if isinstance(base_attrs, RigidBodyPhysicsCfg):
                base_attrs = RigidBodyAttributesCfg.from_grouped(base_attrs)
            physical_attr = attrs.merge_with(base_attrs)
            if attrs.mass is not None:
                replace_inertial = True
        elif isinstance(attrs, RigidBodyAttributesCfg):
            physical_attr = attrs.attr()
        else:
            physical_attr = attrs

        local_env_ids = self._all_indices if env_ids is None else env_ids
        for env_idx in local_env_ids:
            entity = self._entities[env_idx]
            for name in matched_link_names:
                local_name = self._entity_link_name(env_idx, name)
                entity.set_physical_attr(
                    physical_attr,
                    local_name,
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
    ) -> None:
        """Set the drive properties for the articulation.

        Args:
            stiffness (torch.Tensor): The stiffness of the joint drive with shape (len(env_ids), len(joint_ids)).
            damping (torch.Tensor): The damping of the joint drive with shape (len(env_ids), len(joint_ids)).
            max_effort (torch.Tensor): The maximum effort of the joint drive with shape (len(env_ids), len(joint_ids)).
            max_velocity (torch.Tensor): The maximum velocity of the joint drive with shape (len(env_ids), len(joint_ids)).
            friction (torch.Tensor): The joint friction coefficient with shape (len(env_ids), len(joint_ids)).
            armature (torch.Tensor): The joint armature with shape (len(env_ids), len(joint_ids)).
            drive_type: Optional drive type. ``None`` preserves the current mode.
            joint_ids (Sequence[int] | None, optional): The joint indices to apply the drive to. If None, applies to all joints. Defaults to None.
            env_ids (Sequence[int] | None, optional): The environment indices to apply the drive to. If None, applies to all environments. Defaults to None.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        local_joint_ids = np.arange(self.dof) if joint_ids is None else joint_ids
        cache_env_ids = self._resolve_env_ids(env_ids)
        cache_joint_ids = self._resolve_joint_ids(joint_ids)

        def _drive_arg(value: torch.Tensor, index: int) -> float | np.ndarray:
            result = value[index].detach().cpu().numpy()
            return result.item() if result.size == 1 else result

        for i, env_idx in enumerate(local_env_ids):
            if self.is_spawn_bound and self.body_data.is_newton_backend:
                if drive_type == "acceleration":
                    raise NotImplementedError(
                        "Newton Spawn does not have an exact equivalent of "
                        "DexSim's acceleration drive. Use drive_type='force' "
                        "or provide a Newton-native drive descriptor."
                    )
                if drive_type is not None and drive_type not in {"force", "none"}:
                    raise ValueError(f"Unsupported joint drive type {drive_type!r}.")
                drive_args = {"joint_ids": local_joint_ids}
                if drive_type is not None:
                    drive_args["target_mode"] = 3 if drive_type == "force" else 0
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
                self._entities[env_idx].set_newton_drive(**drive_args)
                continue

            drive_args = {"joint_ids": local_joint_ids}
            if drive_type is not None:
                drive_args["drive_type"] = get_dexsim_drive_type(drive_type)
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
            ) = self._entity_drive_properties(self._entities[env_idx])
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
        """Get the backend drive type for the selected joints.

        Args:
            joint_ids: Joint indices to query. If None, queries all joints.
            env_ids: Environment indices to query. If None, queries all environments.

        Returns:
            Backend drive types grouped by environment, with one
            :class:`~dexsim.types.DriveType` per selected joint.
        """
        if self._data is not None and self._data.is_newton_backend:
            raise RuntimeError(
                "get_joint_drive_type() exposes DexSim DriveType semantics; "
                "use get_joint_target_mode() for Newton."
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

        drive_types: list[list[DriveType]] = []
        for env_idx in local_env_ids:
            entity_drive_types = self._entities[int(env_idx)].get_drive()[-1]
            drive_types.append(list(np.asarray(entity_drive_types)[local_joint_ids]))
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
        self.set_qpos(qpos, target=False, env_ids=local_env_ids)
        # Set drive target to hold position.
        self.set_qpos(qpos, target=True, env_ids=local_env_ids)

        self.clear_dynamics(env_ids=local_env_ids)

        self._data.articulation_view.compute_kinematics(local_env_ids)
        if self.device.type == "cpu" and not self._data.is_newton_backend:
            self._world.update(0.001)

    def _set_default_joint_drive(
        self,
        drive_pros: JointDrivePropertiesCfg | dict | None = None,
    ) -> None:
        """Set default joint drive parameters based on the configuration."""
        import numbers
        from embodichain.utils.string import resolve_matching_names_values

        if drive_pros is None:
            drive_pros = self.cfg.drive_pros
        if drive_pros is None:
            return

        drive_props = [
            ("damping", self.default_joint_damping),
            ("stiffness", self.default_joint_stiffness),
            ("max_effort", self.default_joint_max_effort),
            ("max_velocity", self.default_joint_max_velocity),
            ("friction", self.default_joint_friction),
            ("armature", self.default_joint_armature),
        ]

        for prop_name, default_array in drive_props:
            value = (
                drive_pros.get(prop_name)
                if isinstance(drive_pros, dict)
                else getattr(drive_pros, prop_name, None)
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

        if isinstance(drive_pros, dict):
            drive_type = drive_pros.get("drive_type")
        else:
            drive_type = getattr(drive_pros, "drive_type", None)

        # Apply drive parameters to all articulations in the batch
        self.set_joint_drive(
            stiffness=self.default_joint_stiffness,
            damping=self.default_joint_damping,
            max_effort=self.default_joint_max_effort,
            max_velocity=self.default_joint_max_velocity,
            friction=self.default_joint_friction,
            armature=self.default_joint_armature,
            drive_type=drive_type,
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
            # SpawnResult is the sole owner of native lifetime.
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


__all__ = ["ArticulationData", "Articulation"]
