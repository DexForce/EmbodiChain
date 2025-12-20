# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
import dexsim
import numpy as np

from dataclasses import dataclass
from functools import cached_property
from typing import List, Sequence, Dict, Union, Tuple

from dexsim.engine import Articulation as _Articulation
from dexsim.types import (
    ArticulationGPUAPIWriteType,
    ArticulationGPUAPIReadType,
)
from dexsim.engine import CudaArray, PhysicsScene

from embodichain.lab.sim import VisualMaterialInst, VisualMaterial
from embodichain.lab.sim.cfg import ArticulationCfg, JointDrivePropertiesCfg
from embodichain.lab.sim.common import BatchEntity
from embodichain.utils.math import (
    matrix_from_quat,
    quat_from_matrix,
    convert_quat,
    matrix_from_euler,
)
from embodichain.lab.sim.utility.sim_utils import (
    get_dexsim_drive_type,
    set_dexsim_articulation_cfg,
    is_rt_enabled,
)
from embodichain.lab.sim.utility.solver_utils import (
    create_pk_chain,
    create_pk_serial_chain,
)
from embodichain.utils import logger


@dataclass
class ArticulationData:
    """GPU data manager for articulation."""

    def __init__(
        self, entities: List[_Articulation], ps: PhysicsScene, device: torch.device
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

        # get gpu indices for the entities.
        # only meaningful when using GPU physics.
        self.gpu_indices = (
            torch.as_tensor(
                [entity.get_gpu_index() for entity in self.entities],
                dtype=torch.int32,
                device=self.device,
            )
            if self.device.type == "cuda"
            else None
        )

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

        max_num_links = (
            self.ps.gpu_get_articulation_max_link_count()
            if self.device.type == "cuda"
            else self.num_links
        )
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

        max_dof = (
            self.ps.gpu_get_articulation_max_dof()
            if self.device.type == "cuda"
            else self.dof
        )
        self._qpos = torch.zeros(
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

    @property
    def root_pose(self) -> torch.Tensor:
        """Get the root pose of the articulation.

        Returns:
            torch.Tensor: The root pose of the articulation with shape of (num_instances, 7).
        """
        if self.device.type == "cpu":
            # Fetch pose from CPU entities
            root_pose = torch.as_tensor(
                np.array([entity.get_local_pose() for entity in self.entities]),
                dtype=torch.float32,
                device=self.device,
            )
            xyzs = root_pose[:, :3, 3]
            quats = quat_from_matrix(root_pose[:, :3, :3])
            return torch.cat((xyzs, quats), dim=-1)
        else:
            self.ps.gpu_fetch_root_data(
                data=self._root_pose,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.ROOT_GLOBAL_POSE,
            )
            self._root_pose[:, :4] = convert_quat(self._root_pose[:, :4], to="wxyz")
            return self._root_pose[:, [4, 5, 6, 0, 1, 2, 3]]

    @property
    def root_lin_vel(self) -> torch.Tensor:
        """Get the linear velocity of the root link of the articulation.

        Returns:
            torch.Tensor: The linear velocity of the root link with shape of (num_instances, 3).
        """
        if self.device.type == "cpu":
            # Fetch linear velocity from CPU entities
            return torch.as_tensor(
                np.array(
                    [entity.get_root_link_velocity()[:3] for entity in self.entities]
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.ps.gpu_fetch_root_data(
                data=self._root_lin_vel,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.ROOT_LINEAR_VELOCITY,
            )
            return self._root_lin_vel.clone()

    @property
    def root_ang_vel(self) -> torch.Tensor:
        """Get the angular velocity of the root link of the articulation.

        Returns:
            torch.Tensor: The angular velocity of the root link with shape of (num_instances, 3).
        """
        if self.device.type == "cpu":
            # Fetch angular velocity from CPU entities
            return torch.as_tensor(
                np.array(
                    [entity.get_root_link_velocity()[3:] for entity in self.entities]
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.ps.gpu_fetch_root_data(
                data=self._root_ang_vel,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.ROOT_ANGULAR_VELOCITY,
            )
        return self._root_ang_vel.clone()

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
        if self.device.type == "cpu":
            # Fetch qpos from CPU entities
            return torch.as_tensor(
                np.array(
                    [entity.get_current_qpos() for entity in self.entities],
                ),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.ps.gpu_fetch_joint_data(
                data=self._qpos,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.JOINT_POSITION,
            )
            return self._qpos[:, : self.dof].clone()

    @property
    def qvel(self) -> torch.Tensor:
        """Get the current velocities (qvel) of the articulation.

        Returns:
            torch.Tensor: The current velocities of the articulation with shape of (num_instances, dof).
        """
        if self.device.type == "cpu":
            # Fetch qvel from CPU entities
            return torch.as_tensor(
                np.array([entity.get_current_qvel() for entity in self.entities]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.ps.gpu_fetch_joint_data(
                data=self._qvel,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.JOINT_VELOCITY,
            )
            return self._qvel[:, : self.dof].clone()

    @property
    def qacc(self) -> torch.Tensor:
        """Get the current accelerations (qacc) of the articulation.

        Returns:
            torch.Tensor: The current accelerations of the articulation with shape of (num_instances, dof).
        """
        if self.device.type == "cpu":
            # Fetch qacc from CPU entities
            return torch.as_tensor(
                np.array([entity.get_current_qacc() for entity in self.entities]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.ps.gpu_fetch_joint_data(
                data=self._qacc,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.JOINT_ACCELERATION,
            )
            return self._qacc[:, : self.dof].clone()

    @property
    def qf(self) -> torch.Tensor:
        """Get the current forces (qf) of the articulation.

        Returns:
            torch.Tensor: The current forces of the articulation with shape of (num_instances, dof).
        """
        if self.device.type == "cpu":
            # Fetch qf from CPU entities
            return torch.as_tensor(
                np.array([entity.get_current_qf() for entity in self.entities]),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self.ps.gpu_fetch_joint_data(
                data=self._qf,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.JOINT_FORCE,
            )
            return self._qf[:, : self.dof].clone()

    @property
    def body_link_pose(self) -> torch.Tensor:
        """Get the pose of all links in the articulation.

        Returns:
            torch.Tensor: The poses of the links in the articulation with shape (N, num_links, 7).
        """
        if self.device.type == "cpu":
            from embodichain.lab.sim.utility import get_dexsim_arenas

            arenas = get_dexsim_arenas()
            for j, entity in enumerate(self.entities):

                link_pose = np.zeros((self.num_links, 4, 4), dtype=np.float32)
                for i, link_name in enumerate(self.link_names):
                    pose = entity.get_link_pose(link_name)
                    arena_pose = arenas[j].get_root_node().get_local_pose()
                    pose[:2, 3] -= arena_pose[:2, 3]
                    link_pose[i] = pose

                link_pose = torch.from_numpy(link_pose)
                xyz = link_pose[:, :3, 3]
                quat = quat_from_matrix(link_pose[:, :3, :3])
                self._body_link_pose[j][: self.num_links, :] = torch.cat(
                    (xyz, quat), dim=-1
                )
            return self._body_link_pose[:, : self.num_links, :]
        else:
            self.ps.gpu_fetch_link_data(
                data=self._body_link_pose,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.LINK_GLOBAL_POSE,
            )
            quat = convert_quat(self._body_link_pose[..., :4], to="wxyz")
            return torch.cat((self._body_link_pose[..., 4:], quat), dim=-1)

    @property
    def body_link_vel(self) -> torch.Tensor:
        """Get the velocities of all links in the articulation.

        Returns:
            torch.Tensor: The poses of the links in the articulation with shape (N, num_links, 6).
        """
        if self.device.type == "cpu":
            for i, entity in enumerate(self.entities):
                self._body_link_vel[i][: self.num_links] = torch.from_numpy(
                    entity.get_link_general_velocities()
                )
            return self._body_link_vel[:, : self.num_links, :]
        else:
            self.ps.gpu_fetch_link_data(
                data=self._body_link_lin_vel,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.LINK_LINEAR_VELOCITY,
            )
            self.ps.gpu_fetch_link_data(
                data=self._body_link_ang_vel,
                gpu_indices=self.gpu_indices,
                data_type=ArticulationGPUAPIReadType.LINK_ANGULAR_VELOCITY,
            )
            self._body_link_vel[..., :3] = self._body_link_lin_vel
            self._body_link_vel[..., 3:] = self._body_link_ang_vel
        return self._body_link_vel[:, : self.num_links, :]

    @property
    def joint_stiffness(self) -> torch.Tensor:
        """Get the joint stiffness of the articulation.

        Returns:
            torch.Tensor: The joint stiffness of the articulation with shape (N, dof).
        """
        return torch.as_tensor(
            np.array([entity.get_drive()[0] for entity in self.entities]),
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
            np.array([entity.get_drive()[1] for entity in self.entities]),
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
            np.array([entity.get_drive()[4] for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )

    @cached_property
    def qpos_limits(self) -> torch.Tensor:
        """Get the joint position limits of the articulation.

        Returns:
            torch.Tensor: The joint position limits of the articulation with shape (N, dof, 2).
        """
        return torch.as_tensor(
            np.array([entity.get_joint_limits() for entity in self.entities]),
            dtype=torch.float32,
            device=self.device,
        )

    @cached_property
    def qvel_limits(self) -> torch.Tensor:
        """Get the joint velocity limits of the articulation.

        Returns:
            torch.Tensor: The joint velocity limits of the articulation with shape (N, dof).
        """
        # TODO: get joint velocity limits always returns zero?
        return torch.as_tensor(
            np.array(
                [entity.get_drive()[3] for entity in self.entities],
            ),
            dtype=torch.float32,
            device=self.device,
        )

    @cached_property
    def qf_limits(self) -> torch.Tensor:
        """Get the joint effort limits of the articulation.

        Returns:
            torch.Tensor: The joint effort limits of the articulation with shape (N, dof).
        """
        return torch.as_tensor(
            np.array(
                [entity.get_drive()[2] for entity in self.entities],
            ),
            dtype=torch.float32,
            device=self.device,
        )

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
        entities: List[_Articulation] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        # Initialize world and physics scene
        self._world = dexsim.default_world()
        self._ps = self._world.get_physics_scene()

        self.cfg = cfg
        self._entities = entities
        self.device = device

        # Store all indices for batch operations
        self._all_indices = torch.arange(len(entities), dtype=torch.int32).tolist()

        if device.type == "cuda":
            self._world.update(0.001)

        self._data = ArticulationData(entities=entities, ps=self._ps, device=device)

        self.cfg: ArticulationCfg
        if self.cfg.init_qpos is None:
            self.cfg.init_qpos = torch.zeros(self.dof, dtype=torch.float32)

        # Set articulation configuration in DexSim
        set_dexsim_articulation_cfg(entities, self.cfg)

        # Init joint drive parameters.
        num_entities = len(entities)
        dof = self._data.dof
        default_cfg = JointDrivePropertiesCfg()
        self.default_joint_damping = torch.full(
            (num_entities, dof), default_cfg.damping, dtype=torch.float32, device=device
        )
        self.default_joint_stiffness = torch.full(
            (num_entities, dof),
            default_cfg.stiffness,
            dtype=torch.float32,
            device=device,
        )
        self.default_joint_max_effort = torch.full(
            (num_entities, dof),
            default_cfg.max_effort,
            dtype=torch.float32,
            device=device,
        )
        self.default_joint_max_velocity = torch.full(
            (num_entities, dof),
            default_cfg.max_velocity,
            dtype=torch.float32,
            device=device,
        )
        self.default_joint_friction = torch.full(
            (num_entities, dof),
            default_cfg.friction,
            dtype=torch.float32,
            device=device,
        )
        self._set_default_joint_drive()

        self.pk_chain = None
        if self.cfg.build_pk_chain:
            self.pk_chain = create_pk_chain(
                urdf_path=self.cfg.fpath, device=self.device
            )

        # For rendering purposes, each articulation can have multiple material instances associated with its links.
        self._visual_material: List[Dict[str, VisualMaterialInst]] = [
            {} for _ in range(len(entities))
        ]

        # Stores mimic information for joints.
        self._mimic_info = entities[0].get_mimic_info()

        # TODO: very weird that we must call update here to make sure the GPU indices are valid.
        if device.type == "cuda":
            self._world.update(0.001)

        super().__init__(cfg, entities, device)

        # set default collision filter
        self._set_default_collision_filter()

        # flag for collision visible node existence
        self._has_collision_visible_node_dict = dict()
        for link_name in self.link_names:
            self._has_collision_visible_node_dict[link_name] = False

    def __str__(self) -> str:
        parent_str = super().__str__()
        return parent_str + f" | dof: {self.dof} | num_links: {self.num_links}"

    @property
    def dof(self) -> int:
        """Get the degree of freedom of the articulation.

        Returns:
            int: The degree of freedom of the articulation.
        """
        return self._data.dof

    @property
    def num_links(self) -> int:
        """Get the number of links in the articulation.

        Returns:
            int: The number of links in the articulation.
        """
        return self._data.num_links

    @property
    def link_names(self) -> List[str]:
        """Get the names of the links in the articulation.

        Returns:
            List[str]: The names of the links in the articulation.
        """
        return self._data.link_names

    @property
    def root_link_name(self) -> str:
        """Get the name of the root link of the articulation.

        Returns:
            str: The name of the root link.
        """
        return self.entities[0].get_root_link_name()

    @property
    def joint_names(self) -> List[str]:
        """Get the names of the actived joints in the articulation.

        Returns:
            List[str]: The names of the actived joints in the articulation.
        """
        return self._entities[0].get_actived_joint_names()

    @property
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

        if self.device.type == "cpu":
            pose = pose.cpu()
            if pose.dim() == 2 and pose.shape[1] == 7:
                pose_matrix = torch.eye(4).unsqueeze(0).repeat(pose.shape[0], 1, 1)
                pose_matrix[:, :3, 3] = pose[:, :3]
                pose_matrix[:, :3, :3] = matrix_from_quat(pose[:, 3:7])
                for i, env_idx in enumerate(local_env_ids):
                    self._entities[env_idx].set_local_pose(pose_matrix[i])
            elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
                for i, env_idx in enumerate(local_env_ids):
                    self._entities[env_idx].set_local_pose(pose[i])
            else:
                logger.log_error(
                    f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
                )

            # TODO: in manual physics mode, the update should be explicitly called after
            # setting the pose to synchronize the state to renderer.
            self._world.update(0.001)

        else:
            if pose.dim() == 2 and pose.shape[1] == 7:
                xyz = pose[:, :3]
                quat = convert_quat(pose[:, 3:7], to="xyzw")
            elif pose.dim() == 3 and pose.shape[1:] == (4, 4):
                xyz = pose[:, :3, 3]
                quat = quat_from_matrix(pose[:, :3, :3])
                quat = convert_quat(quat, to="xyzw")
            else:
                logger.log_error(
                    f"Invalid pose shape {pose.shape}. Expected (N, 7) or (N, 4, 4)."
                )

            # we should keep `pose_` life cycle to the end of the function.
            pose_ = torch.cat((quat, xyz), dim=-1)
            indices = self.body_data.gpu_indices[local_env_ids]
            torch.cuda.synchronize(self.device)
            self._ps.gpu_apply_root_data(
                data=pose_,
                gpu_indices=indices,
                data_type=ArticulationGPUAPIWriteType.ROOT_GLOBAL_POSE,
            )
            self._ps.gpu_compute_articulation_kinematic(gpu_indices=indices)

            # TODO: To be removed when gpu articulation data sync is supported.
            if is_rt_enabled() is False:
                self.body_data.body_link_pose
                link_pose = self.body_data._body_link_pose[local_env_ids]
                self._world.sync_poses_gpu_to_cpu(
                    link_pose=CudaArray(link_pose),
                    articulation_gpu_indices=CudaArray(indices),
                )

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

    def get_qpos(self) -> torch.Tensor:
        """Get the current positions (qpos) of the articulation."""
        return self.body_data.qpos

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
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)

        if joint_ids is None:
            local_joint_ids = torch.arange(
                self.dof, device=self.device, dtype=torch.int32
            )
        elif not isinstance(joint_ids, torch.Tensor):
            local_joint_ids = torch.as_tensor(
                joint_ids, dtype=torch.int32, device=self.device
            )
        else:
            local_joint_ids = joint_ids

        local_env_ids = self._all_indices if env_ids is None else env_ids

        # TODO: Refactor this part to use a more generic and extensible approach,
        # such as a class decorator that can automatically convert ndarray to torch.Tensor
        # and handle dimension padding for specified member functions.
        # This will make the codebase cleaner and reduce repetitive type checks/conversions.
        # (e.g., support specifying which methods should be decorated for auto-conversion.)
        qpos = torch.as_tensor(qpos, dtype=torch.float32, device=self.device)

        if self.device.type == "cuda":
            limits = self.body_data.qpos_limits[0].T
            # clamp qpos to limits
            lower_limits = limits[0][local_joint_ids]
            upper_limits = limits[1][local_joint_ids]
            qpos = qpos.clamp(lower_limits, upper_limits)

        # Make sure qpos is 2D tensor
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        # If only one qpos is provided, repeat it for all envs
        if len(qpos) == 1 and len(local_env_ids) > 1:
            qpos = qpos.repeat(len(local_env_ids), 1)

        if len(local_env_ids) != len(qpos):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match qpos length {len(qpos)}. "
                f"env_ids: {local_env_ids}, qpos.shape: {qpos.shape}"
            )

        data_type = (
            ArticulationGPUAPIWriteType.JOINT_TARGET_POSITION
            if target
            else ArticulationGPUAPIWriteType.JOINT_POSITION
        )

        if self.device.type == "cpu":
            for i, env_idx in enumerate(local_env_ids):
                setter = (
                    self._entities[env_idx].set_current_qpos
                    if target
                    else self._entities[env_idx].set_qpos
                )
                setter(qpos[i].numpy(), local_joint_ids.numpy())
        else:
            # TODO: trigger qpos getter to sync data, otherwise crash
            if joint_ids is not None:
                self.body_data.qpos

            indices = self.body_data.gpu_indices[local_env_ids]
            qpos_set = self.body_data._qpos[local_env_ids]
            qpos_set[:, local_joint_ids] = qpos
            torch.cuda.synchronize(self.device)
            self._ps.gpu_apply_joint_data(
                data=qpos_set,
                gpu_indices=indices,
                data_type=data_type,
            )

    def get_qvel(self) -> torch.Tensor:
        """Get the current velocities (qvel) of the articulation.

        Returns:
            torch.Tensor: The current velocities of the articulation.
        """
        return self.body_data.qvel

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

        if len(local_env_ids) != len(qvel):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match qvel length {len(qvel)}."
            )

        data_type = (
            ArticulationGPUAPIWriteType.JOINT_TARGET_VELOCITY
            if target
            else ArticulationGPUAPIWriteType.JOINT_VELOCITY
        )

        if self.device.type == "cpu":
            local_joint_ids = np.arange(self.dof) if joint_ids is None else joint_ids
            for i, env_idx in enumerate(local_env_ids):
                setter = (
                    self._entities[env_idx].set_current_qvel
                    if target
                    else self._entities[env_idx].set_qvel
                )
                setter(qvel[i].numpy(), local_joint_ids)
        else:
            indices = self.body_data.gpu_indices[local_env_ids]
            if joint_ids is None:
                qvel_set = self.body_data._qvel[local_env_ids]
                qvel_set[:, : self.dof] = qvel
            else:
                self.body_data.qvel
                qvel_set = self.body_data._qvel[local_env_ids]
                qvel_set[:, joint_ids] = qvel
            torch.cuda.synchronize(self.device)
            self._ps.gpu_apply_joint_data(
                data=qvel_set,
                gpu_indices=indices,
                data_type=data_type,
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

        if len(local_env_ids) != len(qf):
            logger.log_error(
                f"Length of env_ids {len(local_env_ids)} does not match qf length {len(qf)}."
            )

        if self.device.type == "cpu":
            local_joint_ids = np.arange(self.dof) if joint_ids is None else joint_ids
            for i, env_idx in enumerate(local_env_ids):
                setter = self._entities[env_idx].set_current_qf
                setter(qf[i].numpy(), local_joint_ids)
        else:
            indices = self.body_data.gpu_indices[local_env_ids]
            if joint_ids is None:
                qf_set = self.body_data._qf[local_env_ids]
                qf_set[:, : self.dof] = qf
            else:
                self.body_data.qf
                qf_set = self.body_data._qf[local_env_ids]
                qf_set[:, joint_ids] = qf
            torch.cuda.synchronize(self.device)
            self._ps.gpu_apply_joint_data(
                data=qf_set,
                gpu_indices=indices,
                data_type=ArticulationGPUAPIWriteType.JOINT_FORCE,
            )

    def set_drive(
        self,
        stiffness: torch.Tensor | None = None,
        damping: torch.Tensor | None = None,
        max_effort: torch.Tensor | None = None,
        max_velocity: torch.Tensor | None = None,
        friction: torch.Tensor | None = None,
        drive_type: str = "force",
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
            drive_type (str, optional): The type of drive to apply. Defaults to "force".
            joint_ids (Sequence[int] | None, optional): The joint indices to apply the drive to. If None, applies to all joints. Defaults to None.
            env_ids (Sequence[int] | None, optional): The environment indices to apply the drive to. If None, applies to all environments. Defaults to None.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        local_joint_ids = np.arange(self.dof) if joint_ids is None else joint_ids

        for i, env_idx in enumerate(local_env_ids):
            drive_args = {
                "drive_type": get_dexsim_drive_type(drive_type),
                "joint_ids": local_joint_ids,
            }
            if stiffness is not None:
                drive_args["stiffness"] = stiffness[i].cpu().numpy()
            if damping is not None:
                drive_args["damping"] = damping[i].cpu().numpy()
            if max_effort is not None:
                drive_args["max_force"] = max_effort[i].cpu().numpy()
            if max_velocity is not None:
                drive_args["max_velocity"] = max_velocity[i].cpu().numpy()
            if friction is not None:
                drive_args["joint_friction"] = friction[i].cpu().numpy()
            self._entities[env_idx].set_drive(**drive_args)

    def get_user_ids(self) -> torch.Tensor:
        """Get the user ids of the articulation.

        Returns:
            torch.Tensor: The user ids of the articulation with shape (N, num_link).
        """
        return torch.as_tensor(
            np.array(
                [entity.get_user_ids() for entity in self._entities],
            ),
            dtype=torch.int32,
            device=self.device,
        )

    def clear_dynamics(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the dynamics of the articulation.

        Args:
            env_ids (Sequence[int] | None): Environment indices. If None, then all indices are used.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        if self.device.type == "cpu":
            zero_joint_data = np.zeros((len(local_env_ids), self.dof), dtype=np.float32)
            for i, env_idx in enumerate(local_env_ids):
                self._entities[env_idx].set_qvel(zero_joint_data[i])
                self._entities[env_idx].set_current_qf(zero_joint_data[i])
        else:
            zeros = torch.zeros(
                (len(local_env_ids), self.dof), dtype=torch.float32, device=self.device
            )
            indices = self.body_data.gpu_indices[local_env_ids]
            torch.cuda.synchronize(self.device)
            self._ps.gpu_apply_joint_data(
                data=zeros,
                gpu_indices=indices,
                data_type=ArticulationGPUAPIWriteType.JOINT_VELOCITY,
            )
            torch.cuda.synchronize(self.device)
            self._ps.gpu_apply_joint_data(
                data=zeros,
                gpu_indices=indices,
                data_type=ArticulationGPUAPIWriteType.JOINT_FORCE,
            )

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
        self._data._qvel = torch.zeros(
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

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        local_env_ids = self._all_indices if env_ids is None else env_ids
        num_instances = len(local_env_ids)
        self.cfg: ArticulationCfg
        pos = torch.as_tensor(
            self.cfg.init_pos, dtype=torch.float32, device=self.device
        )
        rot = (
            torch.as_tensor(self.cfg.init_rot, dtype=torch.float32, device=self.device)
            * torch.pi
            / 180.0
        )
        pos = pos.unsqueeze(0).repeat(num_instances, 1)
        rot = rot.unsqueeze(0).repeat(num_instances, 1)
        mat = matrix_from_euler(rot, "XYZ")
        pose = (
            torch.eye(4, dtype=torch.float32, device=self.device)
            .unsqueeze(0)
            .repeat(num_instances, 1, 1)
        )
        pose[:, :3, 3] = pos
        pose[:, :3, :3] = mat
        self.set_local_pose(pose, env_ids=local_env_ids)

        qpos = torch.as_tensor(
            self.cfg.init_qpos, dtype=torch.float32, device=self.device
        )
        qpos = qpos.unsqueeze(0).repeat(num_instances, 1)
        self.set_qpos(qpos, target=False, env_ids=local_env_ids)
        # Set drive target to hold position.
        self.set_qpos(qpos, target=True, env_ids=local_env_ids)

        self.clear_dynamics(env_ids=local_env_ids)

        if self.device.type == "cuda":
            self._ps.gpu_compute_articulation_kinematic(
                gpu_indices=self.body_data.gpu_indices[local_env_ids]
            )

            # TODO: To be removed when gpu articulation data sync is supported.
            if is_rt_enabled() is False:
                self.body_data.body_link_pose
                link_pose = self.body_data._body_link_pose[local_env_ids]
                indices = self.body_data.gpu_indices[local_env_ids]
                self._world.sync_poses_gpu_to_cpu(
                    link_pose=CudaArray(link_pose),
                    articulation_gpu_indices=CudaArray(indices),
                )
        else:
            self._world.update(0.001)

    def _set_default_joint_drive(self) -> None:
        """Set default joint drive parameters based on the configuration."""
        import numbers
        from embodichain.utils.string import resolve_matching_names_values

        drive_props = [
            ("damping", self.default_joint_damping),
            ("stiffness", self.default_joint_stiffness),
            ("max_effort", self.default_joint_max_effort),
            ("max_velocity", self.default_joint_max_velocity),
            ("friction", self.default_joint_friction),
        ]

        for prop_name, default_array in drive_props:
            value = getattr(self.cfg.drive_pros, prop_name, None)
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

        drive_pros = self.cfg.drive_pros
        if isinstance(drive_pros, dict):
            drive_type = drive_pros.get("drive_type", None)
        else:
            drive_type = getattr(drive_pros, "drive_type", None)

        # Apply drive parameters to all articulations in the batch
        self.set_drive(
            stiffness=self.default_joint_stiffness,
            damping=self.default_joint_damping,
            max_effort=self.default_joint_max_effort,
            max_velocity=self.default_joint_max_velocity,
            friction=self.default_joint_friction,
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
            locations (Union[torch.Tensor, np.ndarray], optional): Offset points relative to the end-effector
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
            chain=self.pk_chain,
            root_link_name=root_link_name,
            end_link_name=end_link_name,
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
    ) -> None:
        """Set visual material for the rigid object.

        Args:
            mat (VisualMaterial): The material to set.
            env_ids (Sequence[int] | None, optional): Environment indices. If None, then all indices are used.
            link_names (List[str] | None, optional): List of link names to apply the material to. If None, applies to all links.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        link_names = self.link_names if link_names is None else link_names

        for i, env_idx in enumerate(local_env_ids):
            for link_name in link_names:
                mat_inst = mat.create_instance(
                    f"{mat.uid}_{self.uid}_{link_name}_{env_idx}"
                )
                self._entities[env_idx].set_material(link_name, mat_inst.mat)
                self._visual_material[env_idx][link_name] = mat_inst

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

    def destroy(self) -> None:
        env = self._world.get_env()
        arenas = env.get_all_arenas()
        if len(arenas) == 0:
            arenas = [env]
        for i, entity in enumerate(self._entities):
            arenas[i].remove_articulation(entity)
