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
import numpy as np

from typing import List, Dict, Tuple, Union, Sequence
from dataclasses import dataclass, field

from dexsim.engine import Articulation as _Articulation
from embodichain.lab.sim.cfg import RobotCfg
from embodichain.lab.sim.solvers import SolverCfg, BaseSolver
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.sim.utility.tensor import to_tensor
from embodichain.utils.math import quat_from_matrix
from embodichain.utils.string import (
    is_regular_expression,
    resolve_matching_names_values,
)
from embodichain.utils import logger


@dataclass
class ControlGroup:
    r"""Represents a group of controllable joints in a robot.

    Attributes:
        joint_names (List[str]): Names of the joints in this control group.
        joint_ids (List[int]): IDs corresponding to the joints in this control group.
        link_names (List[str]): Names of child links associated with the joints.
    """

    joint_names: List[str] = field(default_factory=list)
    joint_ids: List[int] = field(default_factory=list)
    link_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        pass


class Robot(Articulation):
    """A class representing a batch of robots in the simulation environment.

    Robot is a specific type of articulation that can have additional properties or methods.
    - `control_parts`: Specify the parts that can be controlled in a different manner. Different part may have
        different joint ids, drive properties, pyhsical attributes, kinematic solvers or motion planners.
    - `solvers`: Specify the kinematic solvers for the robot.
    - `planners`: Specify the motion planner for the robot.
    """

    def __init__(
        self,
        cfg: RobotCfg,
        entities: List[_Articulation],
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__(cfg, entities, device)

        self._solvers = {}

        # Initialize joint ids for control parts.
        self._joint_ids: Dict[str, List[int]] = {}

        self._control_groups: Dict[str, ControlGroup] = {}

        if self.cfg.control_parts:
            self._init_control_parts(self.cfg.control_parts)

        if self.cfg.solver_cfg:
            self.init_solver(self.cfg.solver_cfg)

    def __str__(self) -> str:
        parent_str = super().__str__()
        return (
            parent_str
            + f" | control_parts: {self.control_parts}, solvers: {self._solvers}"
        )

    @property
    def control_parts(self) -> Dict[str, List[str]] | None:
        """Get the control parts of the robot."""
        return self.cfg.control_parts

    def get_joint_ids(
        self, name: str | None = None, remove_mimic: bool = False
    ) -> List[int]:
        """Get the joint ids of the robot for a specific control part.

        Args:
            name (str | None): The name of the control part to get the joint ids for. If None, the default part is used.
            remove_mimic (bool, optional): If True, mimic joints will be excluded from the returned joint ids. Defaults to False.

        Returns:
            List[int]: The joint ids of the robot for the specified control part.
        """
        if not self.control_parts or name is None:
            return (
                torch.arange(self.dof, dtype=torch.int32).tolist()
                if not remove_mimic
                else [i for i in range(self.dof) if i not in self.mimic_ids]
            )

        if name not in self.control_parts:
            logger.log_error(
                f"The control part '{name}' does not exist in the robot's control parts."
            )
        return (
            self._joint_ids[name]
            if not remove_mimic
            else [i for i in self._joint_ids[name] if i not in self.mimic_ids]
        )

    def get_proprioception(self) -> Dict[str, torch.Tensor]:
        """Gets robot proprioception information, primarily for agent state representation in robot learning scenarios.

        The default proprioception information includes:
            - qpos: Joint positions.
            - qvel: Joint velocities.
            - qf: Joint efforts.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the robot's proprioception information
        """

        return dict(
            qpos=self.body_data.qpos, qvel=self.body_data.qvel, qf=self.body_data.qf
        )

    def compute_fk(
        self,
        qpos: torch.Tensor | np.ndarray | None,
        name: str | None = None,
        link_names: List[str] | None = None,
        end_link_name: str | None = None,
        root_link_name: str | None = None,
        env_ids: Sequence[int] | None = None,
        to_matrix: bool = False,
    ) -> torch.Tensor:
        """Compute the forward kinematics of the robot given joint positions and optionally a specific part name.
        The output pose will be in the local arena frame.

        Args:
            qpos (torch.Tensor | np.ndarray | None): Joint positions of the robot, (n_envs, num_joints).
            name (str | None): The name of the control part to compute the FK for. If None, the default part is used.
            link_names (List[str] | None): The names of the links to compute the FK for. If None, all links are used.
            end_link_name (str | None): The name of the end link to compute the FK for. If None, the default end link is used.
            root_link_name (str | None): The name of the root link to compute the FK for. If None, the default root link is used.
            env_ids (Sequence[int] | None): The environment ids to compute the FK for. If None, all environments are used.
            to_matrix (bool): If True, returns the transformation in the form of a 4x4 matrix.

        Returns:
            torch.Tensor: The forward kinematics result with shape (n_envs, 7) or (n_envs, 4, 4) if `to_matrix` is True.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        if name is None and hasattr(super(), "compute_fk"):
            return super().compute_fk(
                qpos=qpos,
                link_names=link_names,
                end_link_name=end_link_name,
                root_link_name=root_link_name,
            )

        if not self._solvers:
            logger.log_error(
                "No solvers are defined for the robot. Please ensure that the robot has solvers configured."
            )

        solver = self._solvers.get(name if name is not None else "default", None)
        if solver is None:
            logger.log_error(
                f"The control part '{name}' does not have an associated solver. Please ensure that a valid control part with an available solver is provided."
            )
            return None

        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)

        if qpos.shape[0] != len(local_env_ids):
            logger.log_error(
                f"Joint positions batch size mismatch. Expected {len(local_env_ids)} but got {qpos.shape[0]}."
            )

        if qpos.shape[1] != solver.dof:
            logger.log_error(
                f"Joint positions shape mismatch. Expected {solver.dof} joints, got {qpos.shape[1]}."
            )

        result_matrix = solver.get_fk(qpos=qpos)

        base_pose = self.get_link_pose(
            link_name=solver.root_link_name, env_ids=local_env_ids, to_matrix=True
        )
        result_matrix = torch.bmm(base_pose, result_matrix)

        if to_matrix:
            return result_matrix
        else:
            pos = result_matrix[:, :3, 3]
            quat = quat_from_matrix(result_matrix[:, :3, :3])
            return torch.cat((pos, quat), dim=-1)

    def compute_ik(
        self,
        pose: torch.Tensor | np.ndarray,
        joint_seed: torch.Tensor | np.ndarray | None = None,
        name: str | None = None,
        env_ids: Sequence[int] | None = None,
        return_all_solutions: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | None:
        """Compute the inverse kinematics of the robot given joint positions and optionally a specific part name.
        The input pose should be in the local arena frame.

        Args:
            pose (torch.Tensor): The end effector pose of the robot, (n_envs, 7) or (n_envs, 4, 4).
            joint_seed (torch.Tensor | None): The joint positions to use as a seed for the IK computation, (n_envs, dof).
                If None, the zero joint positions will be used as the seed.
            name (str | None): The name of the control part to compute the IK for. If None, the default part is used.
            env_ids (Sequence[int] | None): Environment indices to apply the positions. Defaults to all environments.
            return_all_solutions (bool): Whether to return all IK solutions or just the best one. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor] | None: The success Tensor with shape (n_envs, ) and qpos Tensor with shape (n_envs, max_results, dof), or None if solver not found.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        solver = self._solvers.get(name if name is not None else "default", None)
        if solver is None:
            logger.log_error(
                f"The control part '{name}' does not have an associated solver. Please ensure that a valid control part with an available solver is provided."
            )
            return None

        pose = to_tensor(pose, device=self.device)
        if (pose.dim() == 1 and pose.shape[1] == 7) or (
            pose.dim() == 2 and pose.shape[1] == 4
        ):
            pose = pose.unsqueeze(0)

        if pose.shape[0] != len(local_env_ids):
            logger.log_error(
                f"Pose batch size mismatch. Expected {len(local_env_ids)} but got {pose.shape[0]}."
            )

        if joint_seed is not None:
            joint_seed = to_tensor(joint_seed, device=self.device)
            if joint_seed.dim() == 1:
                joint_seed = joint_seed.unsqueeze(0)

            if joint_seed.shape[0] != len(local_env_ids):
                logger.log_error(
                    f"Joint seed batch size mismatch. Expected {len(local_env_ids)} but got {joint_seed.shape[0]}."
                )

        if pose.shape[-1] == 7 and pose.dim() == 2:
            # Convert pose from (batch, 7) to (batch, 4, 4)
            pose = torch.cat(
                (
                    pose[:, :3].unsqueeze(-1),  # Position
                    quat_from_matrix(pose[:, 3:]).unsqueeze(-1),  # Quaternion
                ),
                dim=-1,
            )
            pose = torch.cat(
                (
                    pose,
                    torch.tensor([[0, 0, 0, 1]], device=pose.device).expand(
                        pose.shape[0], -1, -1
                    ),
                ),
                dim=1,
            )

        base_pose = self.get_link_pose(
            link_name=solver.root_link_name, env_ids=local_env_ids, to_matrix=True
        )
        pose = torch.bmm(torch.inverse(base_pose), pose)

        ret, qpos = solver.get_ik(
            target_xpos=pose,
            qpos_seed=joint_seed,
            return_all_solutions=return_all_solutions,
        )
        dof = qpos.shape[-1]
        if not return_all_solutions:
            qpos = qpos.reshape(-1, dof)

        return ret.to(self.device), qpos.to(self.device)

    def compute_batch_fk(
        self,
        qpos: torch.Tensor,
        name: str,
        env_ids: Sequence[int] | None = None,
        to_matrix: bool = False,
    ):
        """Compute the forward kinematics of the robot given joint positions and optionally a specific part name.
        The output pose will be in the local arena frame.

        Args:
            qpos (torch.Tensor | np.ndarray | None): Joint positions of the robot, (n_envs, n_batch, num_joints).
            name (str | None): The name of the control part to compute the FK for. If None, the default part is used.
            env_ids (Sequence[int] | None): The environment ids to compute the FK for. If None, all environments are used.
            to_matrix (bool): If True, returns the transformation in the form of a 4x4 matrix.

        Returns:
            torch.Tensor: The forward kinematics result with shape (n_envs, batch, 7) or (n_envs, batch, 4, 4) if `to_matrix` is True.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids
        if not self._solvers:
            logger.log_error(
                "No solvers are defined for the robot. Please ensure that the robot has solvers configured."
            )

        solver = self._solvers.get(name if name is not None else "default", None)
        if solver is None:
            logger.log_error(
                f"The control part '{name}' does not have an associated solver. Please ensure that a valid control part with an available solver is provided."
            )
            return None

        if qpos.shape[0] != len(local_env_ids):
            logger.log_error(
                f"Joint positions batch size mismatch. Expected {len(local_env_ids)} but got {qpos.shape[0]}."
            )

        if qpos.shape[2] != solver.dof:
            logger.log_error(
                f"Joint positions shape mismatch. Expected {solver.dof} joints, got {qpos.shape[1]}."
            )

        n_batch = qpos.shape[1]
        qpos_batch = qpos.reshape(-1, solver.dof)
        xpos_batch = solver.get_fk(qpos=qpos_batch)

        # get xpos from link root
        base_xpos_n_envs = self.get_link_pose(
            link_name=solver.root_link_name, env_ids=local_env_ids, to_matrix=True
        )
        base_xpos_batch = (
            base_xpos_n_envs[:, None, :, :].repeat(1, n_batch, 1, 1).reshape(-1, 4, 4)
        )
        result_matrix = torch.bmm(base_xpos_batch, xpos_batch)

        if to_matrix:
            result_matrix = result_matrix.reshape(len(local_env_ids), n_batch, 4, 4)
            return result_matrix
        else:
            pos = result_matrix[:, :3, 3]
            quat = quat_from_matrix(result_matrix[:, :3, :3])
            result = torch.cat((pos, quat), dim=-1)
            result = result.reshape(len(local_env_ids), n_batch, 7)
            return result

    def compute_batch_ik(
        self,
        pose: torch.Tensor | np.ndarray,
        joint_seed: torch.Tensor | np.ndarray | None,
        name: str,
        env_ids: Sequence[int] | None = None,
    ):
        """Compute the inverse kinematics of the robot given joint positions and optionally a specific part name.
        The input pose should be in the local arena frame.

        Args:
            pose (torch.Tensor): The end effector pose of the robot, (n_envs, n_batch, 7) or (n_envs, n_batch, 4, 4).
            joint_seed (torch.Tensor | None): The joint positions to use as a seed for the IK computation, (n_envs, n_batch, dof). If None, the zero joint positions will be used as the seed.
            name (str | None): The name of the control part to compute the IK for. If None, the default part is used.
            env_ids (Sequence[int] | None): Environment indices to apply the positions. Defaults to all environments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Success Tensor with shape (n_envs, n_batch)
                Qpos Tensor with shape (n_envs, n_batch, dof).
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        solver = self._solvers.get(name if name is not None else "default", None)
        if solver is None:
            logger.log_error(
                f"The control part '{name}' does not have an associated solver. Please ensure that a valid control part with an available solver is provided."
            )
            return None
        pose = to_tensor(pose, device=self.device)

        if pose.shape[0] != len(local_env_ids):
            logger.log_error(
                f"Pose batch size mismatch. Expected {len(local_env_ids)} but got {pose.shape[0]}."
            )

        n_batch = pose.shape[1]
        n_dof = solver.dof
        if joint_seed is None:
            joint_seed = torch.zeros(
                (len(local_env_ids), n_batch, n_dof),
                dtype=torch.float32,
                device=self.device,
            )

        if joint_seed.shape[0] != len(local_env_ids):
            logger.log_error(
                f"Joint seed env size mismatch. Expected {len(local_env_ids)} but got {joint_seed.shape[0]}."
            )

        if joint_seed.shape[1] != n_batch:
            logger.log_error(
                f"Joint seed batch size mismatch. Expected {n_batch} but got {joint_seed.shape[1]}."
            )

        if joint_seed.shape[-1] != n_dof:
            logger.log_error(
                f"Joint seed dof size mismatch. Expected {n_batch} but got {joint_seed.shape[-1]}."
            )

        if pose.shape[-1] == 7 and pose.dim() == 3:
            # Convert pose from (n_envs, n_batch, 7) to (n_envs * n_batch, 4, 4)
            pose_batch = torch.reshape(-1, 7)
            pose_batch = torch.cat(
                (
                    pose_batch[:, :3].unsqueeze(-1),  # Position
                    quat_from_matrix(pose_batch[:, 3:]).unsqueeze(-1),  # Quaternion
                ),
                dim=-1,
            )
            pose_batch = torch.cat(
                (
                    pose_batch,
                    torch.tensor([[0, 0, 0, 1]], device=pose_batch.device).expand(
                        pose_batch.shape[0], -1, -1
                    ),
                ),
                dim=1,
            )
        else:
            # Convert pose from (n_envs, n_batch, 4, 4) to (n_envs * n_batch, 4, 4)
            pose_batch = pose.reshape(-1, 4, 4)

        # get xpos from link root
        base_xpos_n_envs = self.get_link_pose(
            link_name=solver.root_link_name, env_ids=local_env_ids, to_matrix=True
        )
        base_inv_xpos_n_envs = torch.inverse(base_xpos_n_envs)
        base_inv_xpos_batch = (
            base_inv_xpos_n_envs[:, None, :, :]
            .repeat(1, n_batch, 1, 1)
            .reshape(-1, 4, 4)
        )
        pose_batch = torch.bmm(base_inv_xpos_batch, pose_batch)

        joint_seed_batch = joint_seed.reshape(-1, n_dof)
        ret, qpos_batch = solver.get_ik(
            target_xpos=pose_batch,
            qpos_seed=joint_seed_batch,
            return_all_solutions=False,
        )
        ret = ret.reshape(len(local_env_ids), n_batch)
        qpos = qpos_batch.reshape(len(local_env_ids), n_batch, n_dof)
        return ret, qpos

    def _init_control_parts(self, control_parts: Dict[str, List[str]]) -> None:
        """Initialize the control parts of the robot.

        Args:
            control_parts (Dict[str, List[str]]): A dictionary where keys are control part names and values are lists of
                joint names or regular expressions that match joint names.
        """
        joint_name_to_ids = {name: i for i, name in enumerate(self.joint_names)}
        for name, joint_names in control_parts.items():
            # convert joint_names which is a regular expression to a list of joint names
            joint_names_expanded = []
            for jn in joint_names:
                if is_regular_expression(jn):
                    _, names, _ = resolve_matching_names_values(
                        {jn: None}, self.joint_names
                    )
                    joint_names_expanded.extend(names)
                else:
                    joint_names_expanded.append(jn)

            self._joint_ids[name] = [
                joint_name_to_ids[joint_name]
                for joint_name in joint_names_expanded
                if joint_name in joint_name_to_ids
            ]
            if len(self._joint_ids[name]) != len(joint_names_expanded):
                logger.log_error(
                    f"joint names in control part '{name}' do not match the robot's joint names. The full joint names are: {self.joint_names}."
                )
            self.cfg.control_parts[name] = joint_names_expanded

        # Initialize control groups
        self._control_groups = self._extract_control_groups()

    def init_solver(self, cfg: Union[SolverCfg, Dict[str, SolverCfg]]) -> None:
        """Initialize the kinematic solver for the robot.

        Args:
            cfg (Union[SolverCfg, Dict[str, SolverCfg]]): The configuration for the kinematic solver.
        """
        self.cfg: RobotCfg

        if isinstance(cfg, SolverCfg):
            if self.control_parts:
                logger.log_error(
                    "Control parts are defined in the robot configuration, solver_cfg must be a dictionary."
                )

            if cfg.urdf_path is None:
                cfg.urdf_path = self.cfg.fpath
            self._solvers["default"] = cfg.init_solver(device=self.device)
        elif isinstance(cfg, Dict):
            if isinstance(self.cfg.control_parts, Dict) is False:
                logger.log_error(
                    "When `solver_cfg` is a dictionary, `control_parts` must also be a dictionary."
                )

            # If solver_cfg is a dictionary, iterate through it to create solvers
            for name, solver_cfg in cfg.items():
                if solver_cfg.urdf_path is None:
                    solver_cfg.urdf_path = self.cfg.fpath
                _, part_names, value = resolve_matching_names_values(
                    {name: solver_cfg}, self.cfg.control_parts.keys()
                )
                for part_name in part_names:
                    if (
                        not hasattr(solver_cfg, "joint_names")
                        or solver_cfg.joint_names is None
                    ):
                        solver_cfg.joint_names = self.cfg.control_parts[part_name]
                    self._solvers[name] = solver_cfg.init_solver(device=self.device)

    def get_solver(self, name: str | None = None) -> BaseSolver | None:
        """Get the kinematic solver for a specific control part.

        Args:
            name (str | None): The name of the control part to get the solver for. If None, the default part is used.

        Returns:
            BaseSolver | None: The kinematic solver for the specified control part, or None if not found.
        """

        if not self._solvers:
            logger.log_error(
                "No solvers are defined for the robot. Please ensure that the robot has solvers configured."
            )
            return None

        return self._solvers.get(name if name is not None else "default", None)

    def get_control_part_base_pose(
        self,
        name: str | None = None,
        env_ids: Sequence[int] | None = None,
        to_matrix: bool = False,
    ) -> torch.Tensor:
        """Retrieves the base pose of the control part for a specified robot.

        Args:
            name (str | None): The name of the control part the solver adhere to. If None, the default solver is used.
            env_ids (Sequence[int] | None): A sequence of environment IDs to specify the environments. If None, all indices are used.
            to_matrix (bool): If True, returns the pose in the form of a 4x4 matrix.

        Returns:
            The pose of the specified link in the form of a matrix.
        """
        local_env_ids = self._all_indices if env_ids is None else env_ids

        root_link_name = None
        if name in self._control_groups:
            root_link_name = self._control_groups[name].link_names[0]

        return self.get_link_pose(
            link_name=root_link_name, env_ids=local_env_ids, to_matrix=to_matrix
        )

    def get_control_part_link_names(self, name: str | None = None) -> List[str]:
        """Get the link names of the control part.

        Args:
            name (str | None): The name of the control part. If None, return all link names.
        Returns:
            List[str]: link names of the control part.
        """
        if name is None:
            return self.link_names
        if name in self._control_groups:
            return self._control_groups[name].link_names
        else:
            logger.log_warning(
                f"The control part '{name}' does not exist in the robot's control parts."
            )
            return []

    def _extract_control_groups(self) -> Dict[str, ControlGroup]:
        r"""Extract control groups from the active joint names.

        This method creates a dictionary of control groups where each control
        group is associated with its corresponding joint names. It utilizes
        the `_extract_control_group` method to populate the control groups.

        Returns:
            Dict[str, ControlGroup]: A dictionary mapping control group names
                                    to their corresponding ControlGroup instances.
        """
        if not self.control_parts:
            return {}

        control_groups = {
            control_group_name: self._extract_control_group(joint_names)
            for control_group_name, joint_names in self.control_parts.items()
        }

        return control_groups

    def _extract_control_group(self, joint_names: List[str]) -> ControlGroup:
        r"""Extract a control group from the given list of joint names.

        Args:
            joint_names (List[str]): A list of joint names
                                        to be included in the control group.

        Returns:
            ControlGroup: An instance of ControlGroup containing the specified joints
                            and their associated links.
        """
        control_group = ControlGroup()
        joint_id_list = []

        for joint_name in joint_names:
            if joint_name in self.joint_names:
                joint_index = self.joint_names.index(joint_name)
                joint_id_list.append(joint_index)
                control_group.joint_names.append(joint_name)

                # Set root link for first joint
                if len(control_group.link_names) == 0:
                    parent_names = self._entities[0].get_ancestral_link_names(
                        joint_index
                    )
                    control_group.link_names.extend(parent_names)

                child_name = self._entities[0].get_child_link_name(joint_index)
                control_group.link_names.append(child_name)

        control_group.joint_ids = joint_id_list
        return control_group

    def build_pk_serial_chain(self) -> None:
        """Build the kinematic serial chain for the robot.

        This method is mainly used for robot learning scenarios, for example:
            - Imitation learning dataset generation.
        """
        self.pk_serial_chain = self.cfg.build_pk_serial_chain(device=self.device)

    def set_physical_visible(
        self,
        visible: bool = True,
        control_part: str | None = None,
        rgba: Sequence[float] | None = None,
    ):
        """set collision of the robot or a specific control part.

        Args:
            visible (bool, optional): is collision body visible. Defaults to True.
            control_part (str | None, optional): control part to set visibility. Defaults to None. If None, all links are set.
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
        link_names = self.get_control_part_link_names(name=control_part)

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
        return super().destroy()
